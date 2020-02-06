import functools
import importlib
import logging
import logging.handlers
from pathlib import Path
from queue import Queue
import sys
import threading
import time
import yaml

import numpy as np
import tensorflow as tf
import grpc

from .proto import replay_pb2, replay_pb2_grpc
from .proto import learner_pb2, learner_pb2_grpc
from .proto.ndarray_pb2 import Empty
from .proto.numproto import ndarray_to_proto, proto_to_ndarray
from .proto.pingpong_pb2 import Ping, Pong

from .sac_ds_base import SAC_DS_Base
from .utils import rpc_error_inspector

from algorithm.agent import Agent
import algorithm.config_helper as config_helper
from algorithm.env_wrapper import EnvWrapper


WAITING_CONNECTION_TIME = 2
PING_INTERVAL = 5
RECONNECT_TIME = 2


class Actor(object):
    train_mode = True
    _websocket_connected = False
    _agent_class = Agent

    def __init__(self, config_path, args):
        self.config_path = config_path
        self.cmd_args = args
        net_config = self._init_constant_config(self.config_path, args)

        self._stub = StubController(net_config)
        self._run()

    def _init_constant_config(self, config_path, args):
        config_file_path = f'{config_path}/{args.config}' if args.config is not None else None
        config = config_helper.initialize_config_from_yaml(f'{Path(__file__).resolve().parent}/default_config.yaml',
                                                           config_file_path)

        self.config_file_path = config_file_path
        self.train_mode = not args.run
        self.run_in_editor = args.editor

        self.logger = config_helper.set_logger('ds.actor', args.logger_file)

        return config['net_config']

    def _init_env(self):
        # each time actor connects to the learner and replay, initialize env

        # initialize config
        config = config_helper.initialize_config_from_yaml(f'{Path(__file__).resolve().parent}/default_config.yaml',
                                                           self.config_file_path)

        if self.cmd_args.build_port is not None:
            config['base_config']['build_port'] = self.cmd_args.build_port
        if self.cmd_args.sac is not None:
            config['base_config']['sac'] = self.cmd_args.sac
        if self.cmd_args.agents is not None:
            config['reset_config']['copy'] = self.cmd_args.agents

        self.config = config['base_config']
        self.reset_config = config['reset_config']

        # initialize Unity environment
        if self.run_in_editor:
            self.env = EnvWrapper(train_mode=self.train_mode, base_port=5004)
        else:
            self.env = EnvWrapper(train_mode=self.train_mode,
                                  file_name=self.config['build_path'],
                                  no_graphics=self.train_mode,
                                  base_port=self.config['build_port'],
                                  args=['--scene', self.config['scene']])

        self.obs_dim, self.action_dim = self.env.init()

        self.logger.info(f'{self.config["build_path"]} initialized')

        custom_sac_model = importlib.import_module(f'{self.config_path.replace("/",".")}.{self.config["sac"]}')

        self.sac_actor = SAC_DS_Base(obs_dim=self.obs_dim,
                                     action_dim=self.action_dim,
                                     model_root_path=None,
                                     model=custom_sac_model,
                                     train_mode=False,

                                     burn_in_step=self.config['burn_in_step'],
                                     n_step=self.config['n_step'],
                                     use_rnn=self.config['use_rnn'])

        self.logger.info(f'sac_actor initialized')

    def _update_policy_variables(self):
        variables = self._stub.update_policy_variables()
        if variables is not None:
            self.sac_actor.update_policy_variables(variables)

    def _add_trans(self, n_obses, n_actions, n_rewards, obs_, n_dones,
                   n_rnn_states=None):

        self._stub.post_rewards(n_rewards)

        if n_obses.shape[1] < self.config['burn_in_step'] + self.config['n_step']:
            return

        if self.config['use_rnn']:
            n_mu_probs = self.sac_actor.get_rnn_n_step_probs(n_obses, n_actions,
                                                             n_rnn_states[:, 0, :]).numpy()
            self._stub.add_transitions(n_obses, n_actions, n_rewards, obs_, n_dones, n_mu_probs,
                                       n_rnn_states)
        else:
            n_mu_probs = self.sac_actor.get_n_step_probs(n_obses, n_actions).numpy()
            self._stub.add_transitions(n_obses, n_actions, n_rewards, obs_, n_dones, n_mu_probs)

    def _run(self):
        iteration = 0

        while True:
            # replay or learner is offline, waiting...
            if not self._stub.connected:
                if iteration != 0:
                    self.env.close()
                    self.logger.info(f'{self.config["build_path"]} closed')
                    iteration = 0

                self.logger.warning('waiting for connection')
                time.sleep(WAITING_CONNECTION_TIME)
                continue

            # learner is online, reset all settings
            if iteration == 0 and self._stub.connected:
                self._init_env()

                agent_ids, obs = self.env.reset(reset_config=self.reset_config)

                agents = [self._agent_class(i,
                                            tran_len=self.config['burn_in_step'] + self.config['n_step'],
                                            stagger=self.config['stagger'],
                                            use_rnn=self.config['use_rnn'])
                          for i in agent_ids]

                if self.config['use_rnn']:
                    initial_rnn_state = self.sac_actor.get_initial_rnn_state(len(agents))
                    rnn_state = initial_rnn_state

            if self.config['reset_on_iteration']:
                _, obs = self.env.reset(reset_config=self.reset_config)
                for agent in agents:
                    agent.clear()

                if self.config['use_rnn']:
                    rnn_state = initial_rnn_state
            else:
                for agent in agents:
                    agent.reset()

            # burn in padding
            if self.config['use_rnn']:
                for agent in agents:
                    if agent.is_empty():
                        for _ in range(self.config['burn_in_step']):
                            agent.add_transition(np.zeros(self.obs_dim),
                                                 np.zeros(self.action_dim),
                                                 0, False, False,
                                                 np.zeros(self.obs_dim),
                                                 initial_rnn_state[0])

            step = 0

            if self.config['update_policy_mode'] and self.config['update_policy_variables_per_step'] == -1:
                self._update_policy_variables()

            while False in [a.done for a in agents] and self._stub.connected:
                if self.config['update_policy_mode']:
                    # update policy variables each "update_policy_variables_per_step"
                    if self.config['update_policy_variables_per_step'] != -1 and step % self.config['update_policy_variables_per_step'] == 0:
                        self._update_policy_variables()

                    if self.config['use_rnn']:
                        action, next_rnn_state = self.sac_actor.choose_rnn_action(obs.astype(np.float32),
                                                                                  rnn_state)
                        next_rnn_state = next_rnn_state.numpy()
                    else:
                        action = self.sac_actor.choose_action(obs.astype(np.float32))

                    action = action.numpy()
                else:
                    # get action from learner each step
                    if self.config['use_rnn']:
                        action_rnn_state = self._stub.get_action(obs.astype(np.float32), rnn_state)
                        if action_rnn_state is None:
                            break
                        action, next_rnn_state = action_rnn_state
                    else:
                        action = self._stub.get_action(obs.astype(np.float32))
                        if action is None:
                            break

                obs_, reward, local_done, max_reached = self.env.step(action)

                if step == self.config['max_step']:
                    local_done = [True] * len(agents)
                    max_reached = [True] * len(agents)

                tmp_results = [agents[i].add_transition(obs[i],
                                                        action[i],
                                                        reward[i],
                                                        local_done[i],
                                                        max_reached[i],
                                                        obs_[i],
                                                        rnn_state[i] if self.config['use_rnn'] else None)
                               for i in range(len(agents))]

                if self.train_mode:
                    _, episode_trans_list = zip(*tmp_results)

                    episode_trans_list = [t for t in episode_trans_list if t is not None]
                    if len(episode_trans_list) != 0:
                        for episode_trans in episode_trans_list:
                            self._add_trans(*episode_trans)

                obs = obs_
                if self.config['use_rnn']:
                    rnn_state = next_rnn_state
                    rnn_state[local_done] = initial_rnn_state[local_done]

                step += 1

            self._log_episode_info(iteration, agents)
            iteration += 1

    def _log_episode_info(self, iteration, agents):
        rewards = [a.reward for a in agents]
        rewards_sorted = ", ".join([f"{i:.1f}" for i in sorted(rewards)])
        self.logger.info(f'{iteration}, rewards {rewards_sorted}')


class StubController:
    def __init__(self, net_config):
        self._replay_channel = grpc.insecure_channel(f'{net_config["replay_host"]}:{net_config["replay_port"]}',
                                                     [('grpc.max_reconnect_backoff_ms', 5000)])
        self._replay_stub = replay_pb2_grpc.ReplayServiceStub(self._replay_channel)

        self._learner_channel = grpc.insecure_channel(f'{net_config["learner_host"]}:{net_config["learner_port"]}',
                                                      [('grpc.max_reconnect_backoff_ms', 5000)])
        self._learner_stub = learner_pb2_grpc.LearnerServiceStub(self._learner_channel)

        self._replay_connected = False
        self._learner_connected = False
        self._logger = logging.getLogger('ds.actor.stub')

        t_replay = threading.Thread(target=self._start_replay_persistence)
        t_replay.start()
        t_learner = threading.Thread(target=self._start_learner_persistence)
        t_learner.start()

    @property
    def connected(self):
        return self._replay_connected and self._learner_connected

    @rpc_error_inspector
    def add_transitions(self, n_obses, n_actions, n_rewards, obs_, n_dones, n_mu_probs,
                        n_rnn_states=None):
        self._replay_stub.Add(replay_pb2.AddRequest(n_obses=ndarray_to_proto(n_obses),
                                                    n_actions=ndarray_to_proto(n_actions),
                                                    n_rewards=ndarray_to_proto(n_rewards),
                                                    obs_=ndarray_to_proto(obs_),
                                                    n_dones=ndarray_to_proto(n_dones),
                                                    n_mu_probs=ndarray_to_proto(n_mu_probs),
                                                    n_rnn_states=ndarray_to_proto(n_rnn_states)))

    @rpc_error_inspector
    def get_action(self, obs, rnn_state=None):
        request = learner_pb2.GetActionRequest(obs=ndarray_to_proto(obs),
                                               rnn_state=ndarray_to_proto(rnn_state))

        response = self._learner_stub.GetAction(request)
        action = proto_to_ndarray(response.action)
        rnn_state = proto_to_ndarray(response.rnn_state)

        if rnn_state is None:
            return action
        else:
            return action, rnn_state

    @rpc_error_inspector
    def update_policy_variables(self):
        response = self._learner_stub.GetPolicyVariables(Empty())
        return [proto_to_ndarray(v) for v in response.variables]

    @rpc_error_inspector
    def post_rewards(self, n_rewards):
        self._learner_stub.PostRewards(learner_pb2.PostRewardsRequest(n_rewards=ndarray_to_proto(n_rewards)))

    def _start_replay_persistence(self):
        def request_messages():
            while True:
                yield Ping(time=int(time.time() * 1000))
                time.sleep(PING_INTERVAL)
                if not self._replay_connected:
                    break

        while True:
            try:
                reponse_iterator = self._replay_stub.Persistence(request_messages())
                for response in reponse_iterator:
                    if not self._replay_connected:
                        self._replay_connected = True
                        self._logger.info('replay connected')
            except grpc.RpcError:
                if self._replay_connected:
                    self._replay_connected = False
                    self._logger.error('replay disconnected')
            finally:
                time.sleep(RECONNECT_TIME)

    def _start_learner_persistence(self):
        def request_messages():
            while True:
                yield Ping(time=int(time.time() * 1000))
                time.sleep(PING_INTERVAL)
                if not self._learner_connected:
                    break

        while True:
            try:
                reponse_iterator = self._learner_stub.Persistence(request_messages())
                for response in reponse_iterator:
                    if not self._learner_connected:
                        self._learner_connected = True
                        self._logger.info('learner connected')
            except grpc.RpcError:
                if self._learner_connected:
                    self._learner_connected = False
                    self._logger.error('learner disconnected')
            finally:
                time.sleep(RECONNECT_TIME)

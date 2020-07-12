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


WAITING_CONNECTION_TIME = 2
PING_INTERVAL = 5
RECONNECT_TIME = 2


class Actor(object):
    train_mode = True
    _agent_class = Agent
    _logged_waiting_for_connection = False

    def __init__(self, config_path, args):
        self.config_path = config_path
        self.cmd_args = args
        net_config = self._init_constant_config(self.config_path, args)

        self._stub = StubController(net_config)
        self._run()

    def _init_constant_config(self, config_path, args):
        config_file_path = f'{config_path}/config_ds.yaml'
        config = config_helper.initialize_config_from_yaml(f'{Path(__file__).resolve().parent}/default_config.yaml',
                                                           config_file_path,
                                                           args.config)
        self.config_file_path = config_file_path

        # Initialize config from command line arguments
        self.train_mode = not args.run
        self.run_in_editor = args.editor
        self.config_cat = args.config

        self.logger = config_helper.set_logger('ds.actor', args.logger_file)

        return config['net_config']

    def _init_env(self):
        # Each time actor connects to the learner and replay, initialize env

        # Initialize config
        config = config_helper.initialize_config_from_yaml(f'{Path(__file__).resolve().parent}/default_config.yaml',
                                                           self.config_file_path,
                                                           self.config_cat)

        if self.cmd_args.build_port is not None:
            config['base_config']['build_port'] = self.cmd_args.build_port
        if self.cmd_args.sac is not None:
            config['base_config']['sac'] = self.cmd_args.sac
        if self.cmd_args.agents is not None:
            config['base_config']['n_agents'] = self.cmd_args.agents
        if self.cmd_args.noise is not None:
            config['sac_config']['noise'] = self.cmd_args.noise

        self.config = config['base_config']
        sac_config = config['sac_config']
        self.reset_config = config['reset_config']

        # Initialize environment
        if self.config['env_type'] == 'UNITY':
            from algorithm.env_wrapper.unity_wrapper import UnityWrapper

            if self.run_in_editor:
                self.env = UnityWrapper(train_mode=self.train_mode, base_port=5004)
            else:
                self.env = UnityWrapper(train_mode=self.train_mode,
                                        file_name=self.config['build_path'][sys.platform],
                                        no_graphics=self.train_mode,
                                        base_port=self.config['build_port'],
                                        scene=self.config['scene'],
                                        n_agents=self.config['n_agents'])

        elif self.config['env_type'] == 'GYM':
            from algorithm.env_wrapper.gym_wrapper import GymWrapper

            self.env = GymWrapper(train_mode=self.train_mode,
                                  env_name=self.config['build_path'],
                                  n_agents=self.config['n_agents'])
        else:
            raise RuntimeError(f'Undefined Environment Type: {self.config["env_type"]}')

        self.obs_dims, self.action_dim, is_discrete = self.env.init()

        self.logger.info(f'{self.config["build_path"]} initialized')

        custom_sac_model = importlib.import_module(f'{self.config_path.replace("/",".")}.{self.config["sac"]}')

        self.sac_actor = SAC_DS_Base(obs_dims=self.obs_dims,
                                     action_dim=self.action_dim,
                                     is_discrete=is_discrete,
                                     model_root_path=None,
                                     model=custom_sac_model,
                                     train_mode=False,

                                     **sac_config)

        self.logger.info(f'sac_actor initialized')

    def _update_policy_variables(self):
        variables = self._stub.update_policy_variables()
        if variables is not None:
            self.sac_actor.update_policy_variables(variables)

    def _add_trans(self,
                   n_obses_list,
                   n_actions,
                   n_rewards,
                   next_obs_list,
                   n_dones,
                   n_rnn_states=None):

        self._stub.post_rewards(n_rewards)

        if n_obses_list[0].shape[1] < self.sac_actor.burn_in_step + self.sac_actor.n_step:
            return

        if self.sac_actor.use_rnn:
            n_mu_probs = self.sac_actor.get_n_probs(n_obses_list, n_actions,
                                                    n_rnn_states[:, 0, ...]).numpy()
            self._stub.add_transitions(n_obses_list,
                                       n_actions,
                                       n_rewards,
                                       next_obs_list,
                                       n_dones,
                                       n_mu_probs,
                                       n_rnn_states)
        else:
            n_mu_probs = self.sac_actor.get_n_probs(n_obses_list, n_actions,
                                                    None).numpy()
            self._stub.add_transitions(n_obses_list,
                                       n_actions,
                                       n_rewards,
                                       next_obs_list,
                                       n_dones,
                                       n_mu_probs)

    def _run(self):
        iteration = 0

        while True:
            # Replay or learner is offline, waiting...
            if not self._stub.connected:
                if iteration != 0:
                    self.env.close()
                    self.logger.info(f'{self.config["build_path"]} closed')
                    iteration = 0

                if not self._logged_waiting_for_connection:
                    self.logger.warning('waiting for connection')
                    self._logged_waiting_for_connection = True
                time.sleep(WAITING_CONNECTION_TIME)
                continue
            self._logged_waiting_for_connection = False

            # Learner is online, reset all settings
            if iteration == 0 and self._stub.connected:
                self._init_env()
                use_rnn = self.sac_actor.use_rnn

                obs_list = self.env.reset(reset_config=self.reset_config)

                agents = [self._agent_class(i, use_rnn=use_rnn)
                          for i in range(self.config['n_agents'])]

                if use_rnn:
                    initial_rnn_state = self.sac_actor.get_initial_rnn_state(len(agents))
                    rnn_state = initial_rnn_state

            if self.config['reset_on_iteration']:
                obs_list = self.env.reset(reset_config=self.reset_config)
                for agent in agents:
                    agent.clear()

                if use_rnn:
                    rnn_state = initial_rnn_state
            else:
                for agent in agents:
                    agent.reset()

            action = np.zeros([len(agents), self.action_dim], dtype=np.float32)
            step = 0

            if self.config['update_policy_mode'] and self.config['update_policy_variables_per_step'] == -1:
                self._update_policy_variables()

            while False in [a.done for a in agents] and self._stub.connected:
                # burn in padding
                for agent in agents:
                    if agent.is_empty():
                        for _ in range(self.sac_actor.burn_in_step):
                            agent.add_transition([np.zeros(t) for t in self.obs_dims],
                                                 np.zeros(self.action_dim),
                                                 0, False, False,
                                                 [np.zeros(t) for t in self.obs_dims],
                                                 initial_rnn_state[0])

                if self.config['update_policy_mode']:
                    # Update policy variables each "update_policy_variables_per_step"
                    if self.config['update_policy_variables_per_step'] != -1 and step % self.config['update_policy_variables_per_step'] == 0:
                        self._update_policy_variables()

                    if use_rnn:
                        action, next_rnn_state = self.sac_actor.choose_rnn_action([o.astype(np.float32) for o in obs_list],
                                                                                  action,
                                                                                  rnn_state)
                        next_rnn_state = next_rnn_state.numpy()
                    else:
                        action = self.sac_actor.choose_action([o.astype(np.float32) for o in obs_list])

                    action = action.numpy()
                else:
                    # Get action from learner each step
                    if use_rnn:
                        action_rnn_state = self._stub.get_action([o.astype(np.float32) for o in obs_list],
                                                                 rnn_state)
                        if action_rnn_state is None:
                            break
                        action, next_rnn_state = action_rnn_state
                    else:
                        action = self._stub.get_action([o.astype(np.float32) for o in obs_list])
                        if action is None:
                            break

                next_obs_list, reward, local_done, max_reached = self.env.step(action)

                if step == self.config['max_step_per_iter']:
                    local_done = [True] * len(agents)
                    max_reached = [True] * len(agents)

                episode_trans_list = [agents[i].add_transition([o[i] for o in obs_list],
                                                               action[i],
                                                               reward[i],
                                                               local_done[i],
                                                               max_reached[i],
                                                               [o[i] for o in next_obs_list],
                                                               rnn_state[i] if use_rnn else None)
                                      for i in range(len(agents))]

                if self.train_mode:
                    episode_trans_list = [t for t in episode_trans_list if t is not None]
                    if len(episode_trans_list) != 0:
                        for episode_trans in episode_trans_list:
                            self._add_trans(*episode_trans)

                obs_list = next_obs_list
                action[local_done] = np.zeros(self.action_dim)
                if use_rnn:
                    rnn_state = next_rnn_state
                    rnn_state[local_done] = initial_rnn_state[local_done]

                step += 1

            self._log_episode_info(iteration, agents)
            iteration += 1

    def _log_episode_info(self, iteration, agents):
        rewards = [a.reward for a in agents]
        rewards = ", ".join([f"{i:6.1f}" for i in rewards])
        self.logger.info(f'{iteration}, rewards {rewards}')


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
    def add_transitions(self,
                        n_obses_list,
                        n_actions,
                        n_rewards,
                        next_obs_list,
                        n_dones,
                        n_mu_probs,
                        n_rnn_states=None):
        self._replay_stub.Add(replay_pb2.AddRequest(n_obses_list=[ndarray_to_proto(n_obses)
                                                                  for n_obses in n_obses_list],
                                                    n_actions=ndarray_to_proto(n_actions),
                                                    n_rewards=ndarray_to_proto(n_rewards),
                                                    next_obs_list=[ndarray_to_proto(next_obs)
                                                                   for next_obs in next_obs_list],
                                                    n_dones=ndarray_to_proto(n_dones),
                                                    n_mu_probs=ndarray_to_proto(n_mu_probs),
                                                    n_rnn_states=ndarray_to_proto(n_rnn_states)))

    @rpc_error_inspector
    def get_action(self, obs_list, rnn_state=None):
        request = learner_pb2.GetActionRequest(obs_list=[ndarray_to_proto(obs)
                                                         for obs in obs_list],
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
                        self._logger.info('Replay connected')
            except grpc.RpcError:
                if self._replay_connected:
                    self._replay_connected = False
                    self._logger.error('Replay disconnected')
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
                        self._logger.info('Learner connected')
            except grpc.RpcError:
                if self._learner_connected:
                    self._learner_connected = False
                    self._logger.error('Learner disconnected')
            finally:
                time.sleep(RECONNECT_TIME)

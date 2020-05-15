from concurrent import futures
import importlib
import logging
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import yaml

import numpy as np
import grpc

from .proto import learner_pb2, learner_pb2_grpc
from .proto import replay_pb2, replay_pb2_grpc
from .proto.ndarray_pb2 import Empty
from .proto.numproto import ndarray_to_proto, proto_to_ndarray
from .proto.pingpong_pb2 import Ping, Pong
from .utils import PeerSet, rpc_error_inspector

from .sac_ds_base import SAC_DS_Base

from algorithm.agent import Agent
import algorithm.config_helper as config_helper


EVALUATION_INTERVAL = 10
EVALUATION_WAITING_TIME = 1
RESAMPLE_TIME = 2


class Learner(object):
    train_mode = True
    _agent_class = Agent

    _training_lock = threading.Lock()
    _is_training = False

    def __init__(self, config_path, args):
        self._now = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))

        (self.config,
         self.net_config,
         self.reset_config,
         replay_config,
         sac_config,
         model_root_path) = self._init_config(config_path, args)

        self._init_env(config_path, replay_config, sac_config, model_root_path)
        self._run()

    def _init_config(self, config_path, args):
        config_file_path = f'{config_path}/config_ds.yaml'
        config = config_helper.initialize_config_from_yaml(f'{Path(__file__).resolve().parent}/default_config.yaml',
                                                           config_file_path,
                                                           args.config)

        # Initialize config from command line arguments
        self.train_mode = not args.run
        self.last_ckpt = args.ckpt
        self.render = args.render
        self.run_in_editor = args.editor

        if args.name is not None:
            config['base_config']['name'] = args.name
        if args.build_port is not None:
            config['base_config']['build_port'] = args.build_port
        if args.sac is not None:
            config['base_config']['sac'] = args.sac
        if args.agents is not None:
            config['reset_config']['copy'] = args.agents

        config['base_config']['name'] = config['base_config']['name'].replace('{time}', self._now)
        model_root_path = f'models/ds/{config["base_config"]["scene"]}/{config["base_config"]["name"]}'

        logger_file = f'{model_root_path}/{args.logger_file}' if args.logger_file is not None else None
        self.logger = config_helper.set_logger('ds.learner', logger_file)

        if self.train_mode:
            config_helper.save_config(config, model_root_path, 'config.yaml')

        config_helper.display_config(config, self.logger)

        return (config['base_config'],
                config['net_config'],
                config['reset_config'],
                config['replay_config'],
                config['sac_config'],
                model_root_path)

    def _init_env(self, config_path, replay_config, sac_config, model_root_path):
        self._stub = StubController(self.net_config)

        if self.config['env_type'] == 'UNITY':
            from algorithm.env_wrapper.unity_wrapper import UnityWrapper

            if self.run_in_editor:
                self.env = UnityWrapper(train_mode=self.train_mode, base_port=5004)
            else:
                self.env = UnityWrapper(train_mode=self.train_mode,
                                        file_name=self.config['build_path'][sys.platform],
                                        base_port=self.config['build_port'],
                                        scene=self.config['scene'],
                                        n_agents=self.config['n_agents'])

        elif self.config['env_type'] == 'GYM':
            from algorithm.env_wrapper.gym_wrapper import GymWrapper

            self.env = GymWrapper(train_mode=self.train_mode,
                                  env_name=self.config['build_path'],
                                  render=self.render,
                                  n_agents=self.config['n_agents'])
        else:
            raise RuntimeError(f'Undefined Environment Type: {self.config["env_type"]}')

        obs_dims, action_dim, is_discrete = self.env.init()

        self.logger.info(f'{self.config["build_path"]} initialized')

        # If model exists, load saved model, or copy a new one
        if os.path.isfile(f'{model_root_path}/sac_model.py'):
            custom_sac_model = importlib.import_module(f'{model_root_path.replace("/",".")}.sac_model')
        else:
            custom_sac_model = importlib.import_module(f'{config_path.replace("/",".")}.{self.config["sac"]}')
            shutil.copyfile(f'{config_path}/{self.config["sac"]}.py', f'{model_root_path}/sac_model.py')

        self.sac = SAC_DS_Base(obs_dims=obs_dims,
                               action_dim=action_dim,
                               is_discrete=is_discrete,
                               model_root_path=model_root_path,
                               model=custom_sac_model,
                               train_mode=self.train_mode,
                               last_ckpt=self.last_ckpt,

                               **sac_config)

    def _get_policy_variables(self):
        with self._training_lock:
            variables = self.sac.get_policy_variables()

        return [v.numpy() for v in variables]

    def _get_action(self, obs_list, rnn_state=None):
        if self.sac.use_rnn:
            assert rnn_state is not None

        with self._training_lock:
            if self.sac.use_rnn:
                action, next_rnn_state = self.sac.choose_rnn_action(obs_list, rnn_state)
                next_rnn_state = next_rnn_state
                return action.numpy(), next_rnn_state.numpy()
            else:
                action = self.sac.choose_action(obs_list)
                return action.numpy()

    def _get_td_error(self,
                      n_obses_list,
                      n_actions,
                      n_rewards,
                      next_obs_list,
                      n_dones,
                      n_mu_probs,
                      n_rnn_states=None):
        """
        n_obses_list: list([1, episode_len, obs_dim_i], ...)
        n_actions: [1, episode_len, action_dim]
        n_rewards: [1, episode_len]
        next_obs_list: list([1, obs_dim_i], ...)
        n_dones: [1, episode_len]
        n_rnn_states: [1, episode_len, rnn_state_dim]
        """
        td_error = self.sac.get_episode_td_error(n_obses_list=n_obses_list,
                                                 n_actions=n_actions,
                                                 n_rewards=n_rewards,
                                                 next_obs_list=next_obs_list,
                                                 n_dones=n_dones,
                                                 n_mu_probs=n_mu_probs,
                                                 n_rnn_states=n_rnn_states if self.sac.use_rnn else None)

        return td_error

    def _post_rewards(self, peer, n_rewards):
        pass

    def _policy_evaluation(self):
        use_rnn = self.sac.use_rnn

        iteration = 0
        start_time = time.time()

        n_agents, obs_list = self.env.reset(reset_config=self.reset_config)

        agents = [self._agent_class(i, use_rnn=use_rnn)
                  for i in range(n_agents)]

        if use_rnn:
            initial_rnn_state = self.sac.get_initial_rnn_state(len(agents))
            rnn_state = initial_rnn_state

        while True:
            # not training, waiting...
            if self.train_mode and not self._is_training:
                time.sleep(EVALUATION_WAITING_TIME)
                continue

            if self.config['reset_on_iteration']:
                _, obs_list = self.env.reset(reset_config=self.reset_config)
                for agent in agents:
                    agent.clear()

                if use_rnn:
                    rnn_state = initial_rnn_state
            else:
                for agent in agents:
                    agent.reset()

            step = 0

            while False in [a.done for a in agents] and (not self.train_mode or self._is_training):
                with self._training_lock:
                    if use_rnn:
                        action, next_rnn_state = self.sac.choose_rnn_action([o.astype(np.float32) for o in obs_list],
                                                                            rnn_state)
                        next_rnn_state = next_rnn_state.numpy()
                    else:
                        action = self.sac.choose_action([o.astype(np.float32) for o in obs_list])

                action = action.numpy()

                next_obs_list, reward, local_done, max_reached = self.env.step(action)

                if step == self.config['max_step']:
                    local_done = [True] * len(agents)
                    max_reached = [True] * len(agents)

                for i, agent in enumerate(agents):
                    agent.add_transition([o[i] for o in obs_list],
                                         action[i],
                                         reward[i],
                                         local_done[i],
                                         max_reached[i],
                                         [o[i] for o in next_obs_list],
                                         rnn_state[i] if use_rnn else None)

                obs_list = next_obs_list
                if use_rnn:
                    rnn_state = next_rnn_state
                    rnn_state[local_done] = initial_rnn_state[local_done]

                step += 1

            if self.train_mode:
                with self._training_lock:
                    self._log_episode_summaries(iteration, agents)

                    if iteration % self.config['save_model_per_iter'] == 0:
                        self.sac.save_model(iteration)

            self._log_episode_info(iteration, start_time, agents)

            iteration += 1

            if self.train_mode:
                time.sleep(EVALUATION_INTERVAL)

    def _log_episode_summaries(self, iteration, agents):
        rewards = np.array([a.reward for a in agents])
        self.sac.write_constant_summaries([
            {'tag': 'reward/mean', 'simple_value': rewards.mean()},
            {'tag': 'reward/max', 'simple_value': rewards.max()},
            {'tag': 'reward/min', 'simple_value': rewards.min()}
        ], iteration)

    def _log_episode_info(self, iteration, start_time, agents):
        time_elapse = (time.time() - start_time) / 60
        rewards = [a.reward for a in agents]
        rewards = ", ".join([f"{i:6.1f}" for i in rewards])
        self.logger.info(f'{iteration}, {time_elapse:.2f}, rewards {rewards}')

    def _get_sampled_data(self):
        while True:
            sampled = self._stub.get_sampled_data()

            if sampled is None:
                self.logger.warning('no data sampled')
                self._is_training = False
                time.sleep(RESAMPLE_TIME)
                continue
            else:
                self._is_training = True
                return sampled

    def _run_training_client(self):
        self._stub.clear_replay_buffer()

        while True:
            (pointers,
             n_obses_list,
             n_actions,
             n_rewards,
             next_obs_list,
             n_dones,
             n_mu_probs,
             rnn_state,
             priority_is) = self._get_sampled_data()

            with self._training_lock:
                td_error, update_data = self.sac.train(pointers=pointers,
                                                       n_obses_list=n_obses_list,
                                                       n_actions=n_actions,
                                                       n_rewards=n_rewards,
                                                       next_obs_list=next_obs_list,
                                                       n_dones=n_dones,
                                                       n_mu_probs=n_mu_probs,
                                                       priority_is=priority_is,
                                                       rnn_state=rnn_state)

            self._stub.update_td_error(pointers, td_error)
            for pointers, key, data in update_data:
                self._stub.update_transitions(pointers, key, data)

    def _run(self):
        t_evaluation = threading.Thread(target=self._policy_evaluation)
        t_evaluation.start()

        if self.train_mode:
            servicer = LearnerService(self._get_action,
                                      self._get_policy_variables,
                                      self._get_td_error,
                                      self._post_rewards)
            server = grpc.server(futures.ThreadPoolExecutor(max_workers=20))
            learner_pb2_grpc.add_LearnerServiceServicer_to_server(servicer, server)
            server.add_insecure_port(f'[::]:{self.net_config["learner_port"]}')
            server.start()
            self.logger.info(f'Learner server is running on [{self.net_config["learner_port"]}]...')

            self._run_training_client()


class LearnerService(learner_pb2_grpc.LearnerServiceServicer):
    def __init__(self,
                 get_action, get_policy_variables, get_td_error, post_rewards):
        self._get_action = get_action
        self._get_policy_variables = get_policy_variables
        self._get_td_error = get_td_error
        self._post_rewards = post_rewards

        self._peer_set = PeerSet(logging.getLogger('ds.learner.service'))

    def _record_peer(self, context):
        def _unregister_peer():
            self._peer_set.disconnect(context.peer())
        context.add_callback(_unregister_peer)
        self._peer_set.connect(context.peer())

    def Persistence(self, request_iterator, context):
        self._record_peer(context)
        for request in request_iterator:
            yield Pong(time=int(time.time() * 1000))

    def GetAction(self, request: learner_pb2.GetActionRequest, context):
        obs_list = [proto_to_ndarray(obs) for obs in request.obs_list]
        rnn_state = proto_to_ndarray(request.rnn_state)

        if rnn_state is None:
            action = self._get_action(obs_list)
            next_rnn_state = None
        else:
            action, next_rnn_state = self._get_action(obs_list, rnn_state)

        return learner_pb2.Action(action=ndarray_to_proto(action),
                                  rnn_state=ndarray_to_proto(next_rnn_state))

    def GetPolicyVariables(self, request, context):
        variables = self._get_policy_variables()
        return learner_pb2.PolicyVariables(variables=[ndarray_to_proto(v) for v in variables])

    def GetTDError(self, request: learner_pb2.GetTDErrorRequest, context):
        n_obses_list = [proto_to_ndarray(n_obses) for n_obses in request.n_obses_list]
        n_actions = proto_to_ndarray(request.n_actions)
        n_rewards = proto_to_ndarray(request.n_rewards)
        next_obs_list = [proto_to_ndarray(next_obs) for next_obs in request.next_obs_list]
        n_dones = proto_to_ndarray(request.n_dones)
        n_mu_probs = proto_to_ndarray(request.n_mu_probs)
        n_rnn_states = proto_to_ndarray(request.n_rnn_states)

        td_error = self._get_td_error(n_obses_list,
                                      n_actions,
                                      n_rewards,
                                      next_obs_list,
                                      n_dones,
                                      n_mu_probs,
                                      n_rnn_states)
        return learner_pb2.TDError(td_error=ndarray_to_proto(td_error))

    def PostRewards(self, request: learner_pb2.PostRewardsRequest, context):
        n_rewards = proto_to_ndarray(request.n_rewards)
        self._post_rewards(context.peer(), n_rewards)
        return Empty()


class StubController:
    def __init__(self, net_config):
        self._replay_channel = grpc.insecure_channel(f'{net_config["replay_host"]}:{net_config["replay_port"]}')
        self._replay_stub = replay_pb2_grpc.ReplayServiceStub(self._replay_channel)
        self._logger = logging.getLogger('ds.learner.stub')

    @rpc_error_inspector
    def get_sampled_data(self):
        response = self._replay_stub.Sample(Empty())
        if response.has_data:
            return (proto_to_ndarray(response.pointers),
                    [proto_to_ndarray(n_obses) for n_obses in response.n_obses_list],
                    proto_to_ndarray(response.n_actions),
                    proto_to_ndarray(response.n_rewards),
                    [proto_to_ndarray(next_obs) for next_obs in response.next_obs_list],
                    proto_to_ndarray(response.n_dones),
                    proto_to_ndarray(response.n_mu_probs),
                    proto_to_ndarray(response.rnn_state),
                    proto_to_ndarray(response.priority_is))
        else:
            return None

    @rpc_error_inspector
    def update_td_error(self, pointers, td_error):
        self._replay_stub.UpdateTDError(
            replay_pb2.UpdateTDErrorRequest(pointers=ndarray_to_proto(pointers),
                                            td_error=ndarray_to_proto(td_error)))

    @rpc_error_inspector
    def update_transitions(self, pointers, key, data):
        self._replay_stub.UpdateTransitions(
            replay_pb2.UpdateTransitionsRequest(pointers=ndarray_to_proto(pointers),
                                                key=key,
                                                data=ndarray_to_proto(data)))

    @rpc_error_inspector
    def clear_replay_buffer(self):
        self._replay_stub.Clear(Empty())

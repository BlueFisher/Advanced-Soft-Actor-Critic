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
from .peer_set import PeerSet

from .sac_ds_base import SAC_DS_Base

from mlagents.envs.environment import UnityEnvironment
from algorithm.agent import Agent
import algorithm.config_helper as config_helper


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
        config_file_path = f'{config_path}/{args.config}' if args.config is not None else None
        config = config_helper.initialize_config_from_yaml(f'{Path(__file__).resolve().parent}/default_config.yaml',
                                                           config_file_path)

        # initialize config from command line arguments
        self.train_mode = not args.run
        self.run_in_editor = args.editor

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

        if self.run_in_editor:
            self.env = UnityEnvironment(base_port=5004)
        else:
            self.env = UnityEnvironment(file_name=self.config['build_path'],
                                        no_graphics=self.train_mode,
                                        base_port=self.config['build_port'],
                                        args=['--scene', self.config['scene']])

        self.env.reset()
        self.logger.info(f'{self.config["build_path"]} initialized')
        self.default_brain_name = self.env.external_brain_names[0]

        brain_params = self.env.brains[self.default_brain_name]
        state_dim = brain_params.vector_observation_space_size
        action_dim = brain_params.vector_action_space_size[0]

        # if model exists, load saved model, else, copy a new one
        if os.path.isfile(f'{model_root_path}/sac_model.py'):
            custom_sac_model = importlib.import_module(f'{model_root_path.replace("/",".")}.sac_model')
        else:
            custom_sac_model = importlib.import_module(f'{config_path.replace("/",".")}.{self.config["sac"]}')
            shutil.copyfile(f'{config_path}/{self.config["sac"]}.py', f'{model_root_path}/sac_model.py')

        self.sac = SAC_DS_Base(state_dim=state_dim,
                               action_dim=action_dim,
                               model_root_path=model_root_path,
                               model=custom_sac_model,
                               train_mode=self.train_mode,

                               burn_in_step=self.config['burn_in_step'],
                               n_step=self.config['n_step'],
                               use_rnn=self.config['use_rnn'],
                               use_prediction=self.config['use_prediction'],

                               replay_batch_size=replay_config['batch_size'],

                               **sac_config)

    def _get_policy_variables(self):
        with self._training_lock:
            variables = self.sac.get_policy_variables()

        return [v.numpy() for v in variables]

    def _get_td_error(self, *trans):
        with self._training_lock:
            td_error = self.sac.get_td_error(*trans)

        return td_error.numpy()

    def _get_next_rnn_state(self, n_states):
        with self._training_lock:
            rnn_state = self.sac.get_next_rnn_state(n_states)

        return rnn_state

    def _policy_evaluation(self):
        iteration = 0
        start_time = time.time()

        brain_info = self.env.reset(train_mode=False, config=self.reset_config)[self.default_brain_name]
        if self.config['use_rnn']:
            initial_rnn_state = self.sac.get_initial_rnn_state(len(brain_info.agents))
            rnn_state = initial_rnn_state

        while True:
            # not training, waiting...
            if not self._is_training:
                time.sleep(1)
                continue

            if self.config['reset_on_iteration']:
                brain_info = self.env.reset(train_mode=False)[self.default_brain_name]

            agents = [self._agent_class(i,
                                        tran_len=self.config['burn_in_step'] + self.config['n_step'],
                                        stagger=self.config['stagger'],
                                        use_rnn=self.config['use_rnn'])
                      for i in brain_info.agents]

            states = brain_info.vector_observations
            step = 0

            while False in [a.done for a in agents] and self._is_training:
                with self._training_lock:
                    if self.config['use_rnn']:
                        actions, next_rnn_state = self.sac.choose_rnn_action(states.astype(np.float32),
                                                                             rnn_state)
                        next_rnn_state = next_rnn_state.numpy()
                    else:
                        actions = self.sac.choose_action(states.astype(np.float32))

                actions = actions.numpy()

                brain_info = self.env.step({
                    self.default_brain_name: actions
                })[self.default_brain_name]

                states_ = brain_info.vector_observations
                if step == self.config['max_step']:
                    brain_info.local_done = [True] * len(brain_info.agents)
                    brain_info.max_reached = [True] * len(brain_info.agents)

                for i, agent in enumerate(agents):
                    agent.add_transition(states[i],
                                         actions[i],
                                         brain_info.rewards[i],
                                         brain_info.local_done[i],
                                         brain_info.max_reached[i],
                                         states_[i],
                                         rnn_state[i] if self.config['use_rnn'] else None)

                states = states_
                if self.config['use_rnn']:
                    rnn_state = next_rnn_state
                    rnn_state[brain_info.local_done] = initial_rnn_state[brain_info.local_done]

                step += 1

            with self._training_lock:
                self._log_episode_info(iteration, start_time, agents)

            iteration += 1
            time.sleep(10)

    def _log_episode_info(self, iteration, start_time, agents):
        rewards = np.array([a.reward for a in agents])

        self.sac.write_constant_summaries([
            {'tag': 'reward/mean', 'simple_value': rewards.mean()},
            {'tag': 'reward/max', 'simple_value': rewards.max()},
            {'tag': 'reward/min', 'simple_value': rewards.min()}
        ], iteration)

        time_elapse = (time.time() - start_time) / 60
        rewards_sorted = ", ".join([f"{i:.1f}" for i in sorted(rewards)])
        self.logger.info(f'{iteration}, {time_elapse:.2f}min, rewards {rewards_sorted}')

    def _get_sampled_data(self):
        while True:
            sampled = self._stub.get_sampled_data()

            if sampled is None:
                self.logger.warning('no data sampled')
                self._is_training = False
                time.sleep(2)
                continue
            else:
                self._is_training = True
                return sampled

    def _run_training_client(self):
        self._stub.clear_replay_buffer()

        while True:
            pointers, trans, priority_is, n_states_for_next_rnn_state_list, episode_trans = self._get_sampled_data()

            if self.config['use_rnn']:
                n_states, n_actions, n_rewards, state_, done, mu_n_probs, rnn_state = trans
                if self.config['use_prediction']:
                    assert n_states_for_next_rnn_state_list is not None and episode_trans is not None
            else:
                n_states, n_actions, n_rewards, state_, done, mu_n_probs = trans
                rnn_state = None

            with self._training_lock:
                td_error, pi_n_probs = self.sac.train(n_states,
                                                      n_actions,
                                                      n_rewards,
                                                      state_,
                                                      done,
                                                      mu_n_probs,
                                                      priority_is,
                                                      rnn_state=rnn_state,
                                                      n_states_for_next_rnn_state_list=n_states_for_next_rnn_state_list,
                                                      episode_trans=episode_trans)

            self._stub.update_td_error(pointers, td_error)
            self._stub.update_transitions(pointers, 5, pi_n_probs)

    def _run(self):
        servicer = LearnerService(self._get_policy_variables,
                                  self._get_td_error,
                                  self._get_next_rnn_state)
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=20))
        learner_pb2_grpc.add_LearnerServiceServicer_to_server(servicer, server)
        server.add_insecure_port(f'[::]:{self.net_config["learner_port"]}')
        server.start()
        self.logger.info(f'learner server is running on [{self.net_config["learner_port"]}]...')

        t_evaluation = threading.Thread(target=self._policy_evaluation)
        t_evaluation.start()

        self._run_training_client()


class LearnerService(learner_pb2_grpc.LearnerServiceServicer):
    def __init__(self,
                 get_policy_variables, get_td_error, get_next_rnn_state):
        self._get_policy_variables = get_policy_variables
        self._get_td_error = get_td_error
        self._get_next_rnn_state = get_next_rnn_state

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

    def GetPolicyVariables(self, request, context):
        variables = self._get_policy_variables()
        return learner_pb2.PolicyVariables(variables=[ndarray_to_proto(v) for v in variables])

    def GetTDError(self, request: learner_pb2.GetTDErrorRequest, context):
        td_error = self._get_td_error(*[proto_to_ndarray(t) for t in request.transitions])
        return learner_pb2.TDError(td_error=ndarray_to_proto(td_error))


class StubController:
    def __init__(self, net_config):
        self._replay_channel = grpc.insecure_channel(f'{net_config["replay_host"]}:{net_config["replay_port"]}')
        self._replay_stub = replay_pb2_grpc.ReplayServiceStub(self._replay_channel)
        self._logger = logging.getLogger('ds.learner.stub')

    def get_sampled_data(self):
        try:
            response = self._replay_stub.Sample(Empty())
            if response.has_data:
                pointers = proto_to_ndarray(response.pointers)
                transitions = [proto_to_ndarray(t) for t in response.transitions]
                priority_is = proto_to_ndarray(response.priority_is)

                if response.has_episode_data:
                    n_states_for_next_rnn_state_list = [proto_to_ndarray(t) for t in response.n_states_for_next_rnn_state_list]
                    episode_transitions = [proto_to_ndarray(t) for t in response.episode_transitions]
                    return pointers, transitions, priority_is, n_states_for_next_rnn_state_list, episode_transitions
                else:
                    return pointers, transitions, priority_is, None
            else:
                return None

        except grpc.RpcError:
            self._logger.error('connection lost in "get_sampled_data"')
            return None

    def update_td_error(self, pointers, td_error):
        try:
            self._replay_stub.UpdateTDError(
                replay_pb2.UpdateTDErrorRequest(pointers=ndarray_to_proto(pointers),
                                                td_error=ndarray_to_proto(td_error)))
        except grpc.RpcError:
            self._logger.error('connection lost in "update_td_error"')

    def update_transitions(self, pointers, index, data):
        try:
            self._replay_stub.UpdateTransitions(
                replay_pb2.UpdateTransitionsRequest(pointers=ndarray_to_proto(pointers),
                                                    index=index,
                                                    data=ndarray_to_proto(data)))
        except grpc.RpcError:
            self._logger.error('connection lost in "update_transitions"')

    def clear_replay_buffer(self):
        try:
            self._replay_stub.Clear(Empty())
        except grpc.RpcError:
            self._logger.error('connection lost in "clear_replay_buffer"')

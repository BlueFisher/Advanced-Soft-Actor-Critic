from concurrent import futures
from pathlib import Path
import logging
import sys
import threading
import time
import yaml

import numpy as np
import grpc

from .proto import replay_pb2, replay_pb2_grpc
from .proto import learner_pb2, learner_pb2_grpc
from .proto.ndarray_pb2 import Empty
from .proto.numproto import ndarray_to_proto, proto_to_ndarray
from .proto.pingpong_pb2 import Ping, Pong
from .utils import PeerSet, rpc_error_inspector


from algorithm.replay_buffer import PrioritizedReplayBuffer
import algorithm.config_helper as config_helper


class Replay(object):
    def __init__(self, config_path, args):
        self.config, net_config, replay_config, episode_buffer_config = self._init_config(config_path, args)

        self._replay_buffer = PrioritizedReplayBuffer(**replay_config)

        self._replay_buffer_lock = threading.Lock()

        self._stub = StubController(net_config)

        self._run_replay_server(net_config)

    def _init_config(self, config_path, args):
        config_file_path = f'{config_path}/{args.config}' if args.config is not None else None
        config = config_helper.initialize_config_from_yaml(f'{Path(__file__).resolve().parent}/default_config.yaml',
                                                           config_file_path)

        self.logger = config_helper.set_logger('ds.replay', args.logger_file)

        config_helper.display_config(config, self.logger)

        return config['base_config'], config['net_config'], config['replay_config'], config['episode_buffer_config']

    def _add(self, n_obses, n_actions, n_rewards, obs_, n_dones, n_mu_probs,
             n_rnn_states=None):

        obs = n_obses.reshape([-1, n_obses.shape[-1]])
        action = n_actions.reshape([-1, n_actions.shape[-1]])
        reward = n_rewards.reshape([-1])
        done = n_dones.reshape([-1])
        mu_prob = n_mu_probs.reshape([-1, n_mu_probs.shape[-1]])

        # padding obs_
        obs = np.concatenate([obs, obs_])
        action = np.concatenate([action,
                                 np.empty([1, action.shape[-1]], dtype=np.float32)])
        reward = np.concatenate([reward,
                                 np.zeros([1], dtype=np.float32)])
        done = np.concatenate([done,
                               np.zeros([1], dtype=np.float32)])
        mu_prob = np.concatenate([mu_prob,
                                  np.empty([1, mu_prob.shape[-1]], dtype=np.float32)])

        storage_data = {
            'obs': obs,
            'action': action,
            'reward': reward,
            'done': done,
            'mu_prob': mu_prob
        }

        if self.config['use_rnn']:
            rnn_state = n_rnn_states.reshape([-1, n_rnn_states.shape[-1]])
            rnn_state = np.concatenate([rnn_state,
                                        np.empty([1, rnn_state.shape[-1]], dtype=np.float32)])
            storage_data['rnn_state'] = rnn_state

        # get td_error
        ignore_size = self.config['burn_in_step'] + self.config['n_step']

        n_obses = np.concatenate([n_obses[:, i:i + ignore_size] for i in range(n_obses.shape[1] - ignore_size + 1)], axis=0)
        n_actions = np.concatenate([n_actions[:, i:i + ignore_size] for i in range(n_actions.shape[1] - ignore_size + 1)], axis=0)
        n_rewards = np.concatenate([n_rewards[:, i:i + ignore_size] for i in range(n_rewards.shape[1] - ignore_size + 1)], axis=0)
        obs_ = obs[ignore_size:]
        n_dones = np.concatenate([n_dones[:, i:i + ignore_size] for i in range(n_dones.shape[1] - ignore_size + 1)], axis=0)
        n_mu_probs = np.concatenate([n_mu_probs[:, i:i + ignore_size] for i in range(n_mu_probs.shape[1] - ignore_size + 1)], axis=0)
        if self.config['use_rnn']:
            rnn_state = rnn_state[:-ignore_size]

        td_error = self._stub.get_td_error(n_obses, n_actions, n_rewards, obs_, n_dones, n_mu_probs,
                                           rnn_state=rnn_state if self.config['use_rnn'] else None)
        # td_error = np.abs(np.random.randn(n_obses.shape[0], 1).astype(np.float32))
        if td_error is not None:
            td_error = td_error.flatten()
            td_error = np.concatenate([td_error,
                                        np.zeros(ignore_size, dtype=np.float32)])
            with self._replay_buffer_lock:
                self._replay_buffer.add_with_td_error(td_error, storage_data, ignore_size=ignore_size)

            percent = self._replay_buffer.size / self._replay_buffer.capacity * 100
            print(f'buffer size, {percent:.2f}%', end='\r')

    def _sample(self):
        with self._replay_buffer_lock:
            sampled = self._replay_buffer.sample()

        if sampled is None:
            return None

        pointers, trans, priority_is = sampled
        # get n_step transitions
        trans = {k: [v] for k, v in trans.items()}
        # k: [v, v, ...]
        for i in range(1, self.config['burn_in_step'] + self.config['n_step'] + 1):
            t_trans = self._replay_buffer.get_storage_data(pointers + i).items()
            for k, v in t_trans:
                trans[k].append(v)

        for k, v in trans.items():
            trans[k] = np.concatenate([np.expand_dims(t, 1) for t in v], axis=1)

        m_obses = trans['obs']
        m_actions = trans['action']
        m_rewards = trans['reward']
        m_dones = trans['done']
        m_mu_probs = trans['mu_prob']

        n_obses = m_obses[:, :-1, :]
        n_actions = m_actions[:, :-1, :]
        n_rewards = m_rewards[:, :-1]
        obs_ = m_obses[:, -1, :]
        n_dones = m_dones[:, :-1]
        n_mu_probs = m_mu_probs[:, :-1, :]

        if self.config['use_rnn']:
            m_rnn_states = trans['rnn_state']
            rnn_state = m_rnn_states[:, 0, :]
        else:
            rnn_state = None

        return pointers, (n_obses,
                          n_actions,
                          n_rewards,
                          obs_,
                          n_dones,
                          n_mu_probs,
                          rnn_state), priority_is

    def _update_td_error(self, pointers, td_error):
        with self._replay_buffer_lock:
            self._replay_buffer.update(pointers, td_error)

    def _update_transitions(self, pointers, key, data):
        with self._replay_buffer_lock:
            self._replay_buffer.update_transitions(pointers, key, data)

    def _clear(self):
        self._replay_buffer.clear()
        self.logger.info('replay buffer cleared')

    def _run_replay_server(self, net_config):
        servicer = ReplayService(self._add,
                                 self._sample,
                                 self._update_td_error,
                                 self._update_transitions,
                                 self._clear)
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        replay_pb2_grpc.add_ReplayServiceServicer_to_server(servicer, server)
        server.add_insecure_port(f'[::]:{net_config["replay_port"]}')
        server.start()
        self.logger.info(f'replay server is running on [{net_config["replay_port"]}]...')
        server.wait_for_termination()


class ReplayService(replay_pb2_grpc.ReplayServiceServicer):
    def __init__(self,
                 add, sample, update_td_error, update_transitions, clear):
        self._add = add
        self._sample = sample
        self._update_td_error = update_td_error
        self._update_transitions = update_transitions
        self._clear = clear

        self._peer_set = PeerSet(logging.getLogger('ds.replay.service'))

    def _record_peer(self, context):
        def _unregister_peer():
            self._peer_set.disconnect(context.peer())
        context.add_callback(_unregister_peer)
        self._peer_set.connect(context.peer())

    def Persistence(self, request_iterator, context):
        self._record_peer(context)
        for request in request_iterator:
            yield Pong(time=int(time.time() * 1000))

    def Add(self, request: replay_pb2.AddRequest, context):
        self._add(n_obses=proto_to_ndarray(request.n_obses),
                  n_actions=proto_to_ndarray(request.n_actions),
                  n_rewards=proto_to_ndarray(request.n_rewards),
                  obs_=proto_to_ndarray(request.obs_),
                  n_dones=proto_to_ndarray(request.n_dones),
                  n_mu_probs=proto_to_ndarray(request.n_mu_probs),
                  n_rnn_states=proto_to_ndarray(request.n_rnn_states))
        return Empty()

    def Sample(self, request, context):
        sampled = self._sample()
        if sampled is None:
            return replay_pb2.SampledData(has_data=False)
        else:
            pointers, trans, priority_is = sampled
            (n_obses,
             n_actions,
             n_rewards,
             obs_,
             n_dones,
             n_mu_probs,
             rnn_state) = [ndarray_to_proto(t) for t in trans]

            return replay_pb2.SampledData(pointers=ndarray_to_proto(pointers),
                                          n_obses=n_obses,
                                          n_actions=n_actions,
                                          n_rewards=n_rewards,
                                          obs_=obs_,
                                          n_dones=n_dones,
                                          n_mu_probs=n_mu_probs,
                                          rnn_state=rnn_state,
                                          priority_is=ndarray_to_proto(priority_is),
                                          has_data=True)

    def UpdateTDError(self, request: replay_pb2.UpdateTDErrorRequest, context):
        self._update_td_error(proto_to_ndarray(request.pointers),
                              proto_to_ndarray(request.td_error))
        return Empty()

    def UpdateTransitions(self, request: replay_pb2.UpdateTransitionsRequest, context):
        self._update_transitions(proto_to_ndarray(request.pointers),
                                 request.key,
                                 proto_to_ndarray(request.data))
        return Empty()

    def Clear(self, request, context):
        self._clear()
        return Empty()


class StubController:
    def __init__(self, net_config):
        self._learner_channel = grpc.insecure_channel(f'{net_config["learner_host"]}:{net_config["learner_port"]}')
        self._learner_stub = learner_pb2_grpc.LearnerServiceStub(self._learner_channel)

        self._logger = logging.getLogger('ds.replay.stub')

    @rpc_error_inspector
    def get_td_error(self, n_obses, n_actions, n_rewards, obs_, n_dones, n_mu_probs,
                     rnn_state=None):
        request = learner_pb2.GetTDErrorRequest(n_obses=ndarray_to_proto(n_obses),
                                                n_actions=ndarray_to_proto(n_actions),
                                                n_rewards=ndarray_to_proto(n_rewards),
                                                obs_=ndarray_to_proto(obs_),
                                                n_dones=ndarray_to_proto(n_dones),
                                                n_mu_probs=ndarray_to_proto(n_mu_probs),
                                                rnn_state=ndarray_to_proto(rnn_state))

        response = self._learner_stub.GetTDError(request)

        return proto_to_ndarray(response.td_error)

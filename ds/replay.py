import json
import logging
import socket
import threading
import time
from concurrent import futures
from pathlib import Path

import grpc
import numpy as np

import algorithm.config_helper as config_helper
from algorithm.replay_buffer import PrioritizedReplayBuffer

from . import constants as C
from .proto import learner_pb2, learner_pb2_grpc, replay_pb2, replay_pb2_grpc
from .proto.ndarray_pb2 import Empty
from .proto.numproto import ndarray_to_proto, proto_to_ndarray
from .proto.pingpong_pb2 import Ping, Pong
from .utils import PeerSet, rpc_error_inspector


class Replay(object):
    _replay_buffer_lock = threading.Lock()

    def __init__(self, root_dir, config_dir, args):
        self.root_dir = root_dir
        self.cmd_args = args

        self.logger = logging.getLogger('ds.replay')

        constant_config = self._init_constant_config(root_dir, config_dir, args)
        net_config = constant_config['net_config']

        self._stub = StubController(net_config['learner_host'],
                                    net_config['learner_port'],
                                    self.close)

        self._init(constant_config)

        try:
            self._run_replay_server(net_config['replay_port'])
        except KeyboardInterrupt:
            self.logger.warning('KeyboardInterrupt in _run_replay_server')
            self.close()

    def _init_constant_config(self, root_dir, config_dir, args):
        config_abs_dir = Path(root_dir).joinpath(config_dir)
        config_abs_path = config_abs_dir.joinpath('config_ds.yaml')
        default_config_abs_path = Path(__file__).resolve().parent.joinpath('default_config.yaml')
        config = config_helper.initialize_config_from_yaml(default_config_abs_path,
                                                           config_abs_path,
                                                           args.config)

        if args.learner_host is not None:
            config['net_config']['learner_host'] = args.learner_host
        if args.learner_port is not None:
            config['net_config']['learner_port'] = args.learner_port
        if args.replay_port is not None:
            config['net_config']['replay_port'] = args.replay_port

        if args.in_k8s:
            hostname = socket.gethostname()
            host_id = hostname.split('-')[-1]
            learner_host = config['net_config']['learner_host']
            replay_host = config['net_config']['replay_host']
            config['net_config']['learner_host'] = f'{learner_host}-{host_id}.{learner_host}'
            config['net_config']['replay_host'] = f'{replay_host}-{host_id}.{replay_host}'

        return config

    def _init(self, config):
        register_response = None
        self.logger.info('Registering...')
        while register_response is None:
            register_response = self._stub.register_to_learner()
            if register_response:
                self.logger.info('Registered')
                break

            time.sleep(C.RECONNECTION_TIME)

        (model_abs_dir,
         reset_config,
         replay_config,
         sac_config) = register_response

        config['reset_config'] = reset_config
        config['replay_config'] = replay_config
        config['sac_config'] = sac_config

        self.use_rnn = sac_config['use_rnn']
        self.burn_in_step = sac_config['burn_in_step']
        self.n_step = sac_config['n_step']

        self._replay_buffer = PrioritizedReplayBuffer(**replay_config)
        self._curr_percent = -1

        if self.cmd_args.logger_in_file:
            logger_file = Path(model_abs_dir).joinpath('replay.log')
            config_helper.set_logger(logger_file)
            self.logger.info(f'Set to logger {logger_file}')

        config_helper.display_config(config, self.logger)

    def _add(self,
             n_obses_list,
             n_actions,
             n_rewards,
             next_obs_list,
             n_dones,
             n_mu_probs,
             n_rnn_states=None):

        self.obs_list_len = len(n_obses_list)
        # Reshape [1, episode_len, ...] to [episode_len, ...]
        obs_list = [n_obses.reshape([-1, *n_obses.shape[2:]]) for n_obses in n_obses_list]
        action = n_actions.reshape([-1, n_actions.shape[-1]])
        reward = n_rewards.reshape([-1])
        done = n_dones.reshape([-1])
        mu_prob = n_mu_probs.reshape([-1])

        # Padding next_obs for episode experience replay
        obs_list = [np.concatenate([obs, next_obs]) for obs, next_obs in zip(obs_list, next_obs_list)]
        action = np.concatenate([action,
                                 np.empty([1, action.shape[-1]], dtype=np.float32)])
        reward = np.concatenate([reward,
                                 np.zeros([1], dtype=np.float32)])
        done = np.concatenate([done,
                               np.zeros([1], dtype=np.float32)])
        mu_prob = np.concatenate([mu_prob,
                                  np.empty([1], dtype=np.float32)])

        storage_data = {f'obs_{i}': obs for i, obs in enumerate(obs_list)}
        storage_data = {
            **storage_data,
            'action': action,
            'reward': reward,
            'done': done,
            'mu_prob': mu_prob
        }

        if self.use_rnn:
            rnn_state = n_rnn_states.reshape([-1, n_rnn_states.shape[-1]])
            rnn_state = np.concatenate([rnn_state,
                                        np.empty([1, rnn_state.shape[-1]], dtype=np.float32)])
            storage_data['rnn_state'] = rnn_state

        # Get td_error
        td_error = self._stub.get_td_error(n_obses_list,
                                           n_actions,
                                           n_rewards,
                                           next_obs_list,
                                           n_dones,
                                           n_mu_probs,
                                           n_rnn_states=n_rnn_states if self.use_rnn else None)

        if td_error is not None:
            td_error = td_error.flatten()
            with self._replay_buffer_lock:
                self._replay_buffer.add_with_td_error(td_error, storage_data,
                                                      ignore_size=self.burn_in_step + self.n_step)

            percent = int(self._replay_buffer.size / self._replay_buffer.capacity * 100)
            if percent > self._curr_percent:
                self.logger.info(f'Buffer size: {percent}%')
                self._curr_percent = percent

    def _sample(self):
        with self._replay_buffer_lock:
            sampled = self._replay_buffer.sample()

            if sampled is None:
                return None

            pointers, trans, priority_is = sampled

            # Get n_step transitions
            trans = {k: [v] for k, v in trans.items()}
            # k: [v, v, ...]
            for i in range(1, self.burn_in_step + self.n_step + 1):
                t_trans = self._replay_buffer.get_storage_data(pointers + i).items()
                for k, v in t_trans:
                    trans[k].append(v)

        for k, v in trans.items():
            trans[k] = np.concatenate([np.expand_dims(t, 1) for t in v], axis=1)

        m_obses_list = [trans[f'obs_{i}'] for i in range(self.obs_list_len)]
        m_actions = trans['action']
        m_rewards = trans['reward']
        m_dones = trans['done']
        m_mu_probs = trans['mu_prob']

        n_obses_list = [m_obses[:, :-1, ...] for m_obses in m_obses_list]
        n_actions = m_actions[:, :-1, ...]
        n_rewards = m_rewards[:, :-1]
        next_obs_list = [m_obses[:, -1, ...] for m_obses in m_obses_list]
        n_dones = m_dones[:, :-1]
        n_mu_probs = m_mu_probs[:, :-1]

        if self.use_rnn:
            m_rnn_states = trans['rnn_state']
            rnn_state = m_rnn_states[:, 0, :]
        else:
            rnn_state = None

        return pointers, (n_obses_list,
                          n_actions,
                          n_rewards,
                          next_obs_list,
                          n_dones,
                          n_mu_probs,
                          rnn_state), priority_is

    def _update_td_error(self, pointers, td_error):
        with self._replay_buffer_lock:
            mask = pointers == self._replay_buffer.get_storage_data_ids(pointers)
            self._replay_buffer.update(pointers[mask], td_error[mask])

    def _update_transitions(self, pointers, key, data):
        with self._replay_buffer_lock:
            mask = pointers == self._replay_buffer.get_storage_data_ids(pointers)
            self._replay_buffer.update_transitions(pointers[mask], key, data[mask])

    def _run_replay_server(self, replay_port):
        servicer = ReplayService(self._add,
                                 self._sample,
                                 self._update_td_error,
                                 self._update_transitions)
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=C.MAX_THREAD_WORKERS),
                                  options=[
            ('grpc.max_send_message_length', C.MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', C.MAX_MESSAGE_LENGTH)
        ])
        replay_pb2_grpc.add_ReplayServiceServicer_to_server(servicer, self.server)
        self.server.add_insecure_port(f'[::]:{replay_port}')
        self.server.start()
        self.logger.info(f'Replay server is running on [{replay_port}]...')
        self.server.wait_for_termination()

    def close(self):
        self.server.stop(None)


class ReplayService(replay_pb2_grpc.ReplayServiceServicer):
    def __init__(self,
                 add,
                 sample,
                 update_td_error,
                 update_transitions):
        self._add = add
        self._sample = sample
        self._update_td_error = update_td_error
        self._update_transitions = update_transitions

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
        self._add(n_obses_list=[proto_to_ndarray(n_obses) for n_obses in request.n_obses_list],
                  n_actions=proto_to_ndarray(request.n_actions),
                  n_rewards=proto_to_ndarray(request.n_rewards),
                  next_obs_list=[proto_to_ndarray(next_obs) for next_obs in request.next_obs_list],
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
            (n_obses_list,
             n_actions,
             n_rewards,
             next_obs_list,
             n_dones,
             n_mu_probs,
             rnn_state) = trans

            return replay_pb2.SampledData(pointers=ndarray_to_proto(pointers),
                                          n_obses_list=[ndarray_to_proto(n_obses) for n_obses in n_obses_list],
                                          n_actions=ndarray_to_proto(n_actions),
                                          n_rewards=ndarray_to_proto(n_rewards),
                                          next_obs_list=[ndarray_to_proto(next_obs) for next_obs in next_obs_list],
                                          n_dones=ndarray_to_proto(n_dones),
                                          n_mu_probs=ndarray_to_proto(n_mu_probs),
                                          rnn_state=ndarray_to_proto(rnn_state),
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


class StubController:
    _closed = False

    def __init__(self, learner_host, learner_port, close_replay):
        self._learner_channel = grpc.insecure_channel(f'{learner_host}:{learner_port}',
                                                      [('grpc.max_reconnect_backoff_ms', C.MAX_RECONNECT_BACKOFF_MS)])
        self._learner_stub = learner_pb2_grpc.LearnerServiceStub(self._learner_channel)

        self._learner_connected = False
        self._logger = logging.getLogger('ds.replay.stub')

        self._close_replay = close_replay

        t_learner = threading.Thread(target=self._start_learner_persistence, daemon=True)
        t_learner.start()

    @rpc_error_inspector
    def register_to_learner(self):
        response = self._learner_stub.RegisterReplay(Empty())
        if response.model_abs_dir:
            return (response.model_abs_dir,
                    json.loads(response.reset_config_json),
                    json.loads(response.replay_config_json),
                    json.loads(response.sac_config_json))

    @rpc_error_inspector
    def get_td_error(self,
                     n_obses_list,
                     n_actions,
                     n_rewards,
                     next_obs_list,
                     n_dones,
                     n_mu_probs,
                     n_rnn_states=None):
        request = learner_pb2.GetTDErrorRequest(n_obses_list=[ndarray_to_proto(n_obses) for n_obses in n_obses_list],
                                                n_actions=ndarray_to_proto(n_actions),
                                                n_rewards=ndarray_to_proto(n_rewards),
                                                next_obs_list=[ndarray_to_proto(next_obs) for next_obs in next_obs_list],
                                                n_dones=ndarray_to_proto(n_dones),
                                                n_mu_probs=ndarray_to_proto(n_mu_probs),
                                                n_rnn_states=ndarray_to_proto(n_rnn_states))

        response = self._learner_stub.GetTDError(request)

        return proto_to_ndarray(response.td_error)

    def _start_learner_persistence(self):
        def request_messages():
            while not self._closed:
                yield Ping(time=int(time.time() * 1000))
                time.sleep(C.PING_INTERVAL)
                if not self._learner_connected:
                    break

        while not self._closed:
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
                    self._close_replay()
                    self.close()
                    break
            finally:
                time.sleep(C.RECONNECTION_TIME)

    def close(self):
        self._closed = True

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
from .peer_set import PeerSet


from algorithm.replay_buffer import PrioritizedReplayBuffer, EpisodeBuffer
import algorithm.config_helper as config_helper


class Replay(object):
    def __init__(self, config_path, args):
        self.config, net_config, replay_config, episode_buffer_config = self._init_config(config_path, args)

        self._replay_buffer = PrioritizedReplayBuffer(**replay_config)
        if self.config['use_rnn'] and self.config['use_prediction']:
            self._episode_buffer = EpisodeBuffer(**episode_buffer_config)

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

    def _add(self, *transitions):
        # n_states, n_actions, n_rewards, state_, done, mu_n_probs, rnn_state
        td_error = self._stub.get_td_error(*transitions[:7])
        if td_error is not None:
            with self._replay_buffer_lock:
                self._replay_buffer.add_with_td_error(td_error, *transitions)

            percent = self._replay_buffer.size / self._replay_buffer.capacity * 100
            print(f'buffer size, {percent:.2f}%', end='\r')

    def _add_episode(self, *transitions):
        assert self.config['use_rnn'] and self.config['use_prediction']

        self._episode_buffer.add(*transitions)

    def _sample(self):
        sampled = self._replay_buffer.sample()
        if self.config['use_rnn'] and self.config['use_prediction']:
            episode_sampled = self._episode_buffer.sample_without_rnn_state()

            # transitions and episode_transitions should appear at the same time
            if episode_sampled is None:
                sampled = None
        else:
            episode_sampled = None

        return sampled, episode_sampled

    def _update_td_error(self, pointers, td_error):
        with self._replay_buffer_lock:
            self._replay_buffer.update(pointers, td_error)

    def _update_transitions(self, pointers, index, data):
        with self._replay_buffer_lock:
            self._replay_buffer.update_transitions(pointers, index, data)

    def _clear(self):
        self._replay_buffer.clear()
        self.logger.info('replay buffer cleared')

    def _run_replay_server(self, net_config):
        servicer = ReplayService(self._add,
                                 self._add_episode,
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
                 add, add_episode, sample, update_td_error, update_transitions, clear):
        self._add = add
        self._add_episode = add_episode
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
        self._add(*[proto_to_ndarray(t) for t in request.transitions])
        return Empty()

    def AddEpisode(self, request: replay_pb2.AddEpisodeRequest, context):
        self._add_episode(*[proto_to_ndarray(t) for t in request.transitions])
        return Empty()

    def Sample(self, request, context):
        sampled, episode_sampled = self._sample()
        if sampled is None:
            return replay_pb2.SampledData(has_data=False)
        else:
            pointers, transitions, priority_is = sampled
            if episode_sampled is None:
                return replay_pb2.SampledData(pointers=ndarray_to_proto(pointers),
                                              transitions=[ndarray_to_proto(t) for t in transitions],
                                              priority_is=ndarray_to_proto(priority_is),
                                              has_data=True)
            else:
                n_states_for_next_rnn_state_list, episode_transitions = episode_sampled
                return replay_pb2.SampledData(pointers=ndarray_to_proto(pointers),
                                              transitions=[ndarray_to_proto(t) for t in transitions],
                                              priority_is=ndarray_to_proto(priority_is),
                                              has_data=True,

                                              n_states_for_next_rnn_state_list=[ndarray_to_proto(t) for t in n_states_for_next_rnn_state_list],
                                              episode_transitions=[ndarray_to_proto(t) for t in episode_transitions],
                                              has_episode_data=True)

    def UpdateTDError(self, request: replay_pb2.UpdateTDErrorRequest, context):
        self._update_td_error(proto_to_ndarray(request.pointers),
                              proto_to_ndarray(request.td_error))
        return Empty()

    def UpdateTransitions(self, request: replay_pb2.UpdateTransitionsRequest, context):
        self._update_transitions(proto_to_ndarray(request.pointers),
                                 request.index,
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

    def get_td_error(self, *transitions):
        try:
            response = self._learner_stub.GetTDError(
                learner_pb2.GetTDErrorRequest(transitions=[ndarray_to_proto(t) for t in transitions]))
            return proto_to_ndarray(response.td_error)
        except grpc.RpcError:
            self._logger.error('connection lost in "get_td_error"')

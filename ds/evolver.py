import logging
import os
import random
import string
import threading
import time
from collections import deque
from concurrent import futures
from pathlib import Path

import grpc
import numpy as np

import algorithm.config_helper as config_helper

from .proto import evolver_pb2, evolver_pb2_grpc, learner_pb2, learner_pb2_grpc
from .proto.ndarray_pb2 import Empty
from .proto.numproto import ndarray_to_proto, proto_to_ndarray
from .proto.pingpong_pb2 import Ping, Pong
from .utils import PeerSet, rpc_error_inspector

MAX_THREAD_WORKERS = 64
MAX_MESSAGE_LENGTH = 256 * 1024 * 1024


class Evolver:
    def __init__(self, root_dir, config_dir, args):
        self._now = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))

        (self.config,
         self.net_config,
         model_abs_dir,
         config_abs_dir) = self._init_config(root_dir, config_dir, args)

        try:
            self._run()
        except KeyboardInterrupt:
            self.logger.warning('KeyboardInterrupt in _run')
            self.close()

    def _init_config(self, root_dir, config_dir, args):
        config_abs_dir = Path(root_dir).joinpath(config_dir)
        config_abs_path = config_abs_dir.joinpath('config_ds.yaml')
        default_config_file_path = Path(__file__).resolve().parent.joinpath('default_config.yaml')
        config = config_helper.initialize_config_from_yaml(default_config_file_path,
                                                           config_abs_path,
                                                           args.config)

        if args.evolver_host is not None:
            config['net_config']['evolver_host'] = args.evolver_host
        if args.evolver_port is not None:
            config['net_config']['evolver_port'] = args.evolver_port
        if args.name is not None:
            config['base_config']['name'] = args.name

        # Replace {time} from current time and random letters
        rand = ''.join(random.sample(string.ascii_letters, 4))
        config['base_config']['name'] = config['base_config']['name'].replace('{time}', self._now + rand)
        model_abs_dir = Path(root_dir).joinpath(f'models/{config["base_config"]["scene"]}/{config["base_config"]["name"]}')
        os.makedirs(model_abs_dir)

        logger_file = f'{model_abs_dir}/{args.logger_file}' if args.logger_file is not None else None
        self.logger = config_helper.set_logger('ds.learner', logger_file)

        config_helper.save_config(config, model_abs_dir, 'config_ds.yaml')

        config_helper.display_config(config, self.logger)

        return (config['base_config'],
                config['net_config'],
                model_abs_dir,
                config_abs_dir)

    _learner_rewards = dict()

    def _learner_connected(self, peer, connected):
        if connected:
            self._learner_rewards[peer] = deque(maxlen=self.config['evolver_cem_length'])
        else:
            del self._learner_rewards[peer]

    def _post_reward(self, reward, peer):
        self._learner_rewards[peer].append(reward)

        if len(self._learner_rewards) > 1 and \
                all([len(i) == self.config['evolver_cem_length'] for i in self._learner_rewards.values()]):
            # if all([len(i) == self.config['evolver_cem_length'] for i in self._learner_rewards.values()]):
            learner_reward = [(l, float(np.mean(r))) for l, r in self._learner_rewards.items()]
            learner_reward.sort(key=lambda i: i[1], reverse=True)

            best_size = int(len(learner_reward) * self.config['evolver_cem_best'])
            best_size = max(best_size, 1)

            best_learners = [i[0] for i in learner_reward[:best_size]]
            nn_variable_list = list()
            for learner in best_learners:
                stub = self.servicer.get_learner_info(learner)['stub']
                nn_variable_list.append(stub.get_nn_variables())

            mean = [np.mean(i, axis=0) for i in zip(*nn_variable_list)]
            std = [np.std(i, axis=0) for i in zip(*nn_variable_list)]

            for learner in self.servicer.learners:
                stub = self.servicer.get_learner_info(learner)['stub']
                nn_variables = [np.random.normal(mean[i], std[i]) for i in range(len(mean))]
                stub.update_nn_variables(nn_variables)

                self._learner_rewards[learner].clear()

            self.logger.info('Dispatched all nn variables')

    def _run(self):
        self.servicer = EvolverService(self.config['name'],
                                       self._learner_connected,
                                       self._post_reward)
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_THREAD_WORKERS))
        evolver_pb2_grpc.add_EvolverServiceServicer_to_server(self.servicer, self.server)
        self.server.add_insecure_port(f'[::]:{self.net_config["evolver_port"]}')
        self.server.start()
        self.logger.info(f'Evolver server is running on [{self.net_config["evolver_port"]}]...')

        self.server.wait_for_termination()

    def close(self):
        self.server.stop(0)

        self.logger.warning('Closed')


class LearnerStubController:
    def __init__(self, host, port):
        self._channel = grpc.insecure_channel(f'{host}:{port}', [
            ('grpc.max_reconnect_backoff_ms', 5000),
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)
        ])
        self._stub = learner_pb2_grpc.LearnerServiceStub(self._channel)

        self._logger = logging.getLogger('ds.evolver.learner_stub')

    @rpc_error_inspector
    def get_nn_variables(self):
        response = self._stub.GetNNVariables(Empty())
        return [proto_to_ndarray(v) for v in response.variables]

    @rpc_error_inspector
    def update_nn_variables(self, variables):
        self._stub.UpdateNNVariables(learner_pb2.NNVariables(
            variables=[ndarray_to_proto(v) for v in variables]))

    def close(self):
        self._channel.close()


class EvolverService(evolver_pb2_grpc.EvolverServiceServicer):
    def __init__(self, name, learner_connected, post_reward):
        self._logger = logging.getLogger('ds.evolver.service')
        self._peer_set = PeerSet(self._logger)

        self._learner_lock = threading.Lock()
        self._learner_id = 0
        self._learner_actors = dict()
        self._actor_learner = dict()

        self.name = name
        self._learner_connected = learner_connected
        self._post_reward = post_reward

    def _record_peer(self, context):
        peer = context.peer()

        def _unregister_peer():
            self._peer_set.disconnect(peer)
            with self._learner_lock:
                if peer in self._learner_actors:
                    del self._learner_actors[peer]
                    self._learner_connected(peer, connected=False)
                    self._logger.warning(f'Learner {peer} disconnected')
                elif peer in self._actor_learner:
                    learner_peer = self._actor_learner[peer]
                    self._learner_actors[learner_peer].remove(peer)
                    del self._actor_learner[peer]
                    self._logger.warning(f'Actor {peer} disconnected')

        context.add_callback(_unregister_peer)
        self._peer_set.connect(peer)

    @property
    def learners(self):
        with self._learner_lock:
            return self._learner_actors.keys()

    def get_learner_info(self, peer):
        return self._peer_set[peer]

    def display_learner_actors(self):
        with self._learner_lock:
            info = 'Learner-actors:'
            for learner in self._learner_actors:
                learner_info = self._peer_set[learner]
                learner_id = learner_info['id']
                learner_host = learner_info['learner_host']
                learner_port = learner_info['learner_port']
                info += f'\n{learner} ({learner_host}:{learner_port} id={learner_id}): {len(self._learner_actors[learner])}'

        self._logger.info(info)

    # From learner and actor
    def Persistence(self, request_iterator, context):
        self._record_peer(context)
        for request in request_iterator:
            yield Pong(time=int(time.time() * 1000))

    def RegisterLearner(self, request: evolver_pb2.RegisterLearnerRequest, context):
        peer = context.peer()
        learner_id = self._learner_id

        self._peer_set.add_info(peer, {
            'id': learner_id,
            'learner_host': request.learner_host,
            'learner_port': request.learner_port,
            'replay_host': request.replay_host,
            'replay_port': request.replay_port,
            'stub': LearnerStubController(request.learner_host, request.learner_port)
        })
        with self._learner_lock:
            self._learner_actors[peer] = set()
        self._learner_connected(peer, connected=True)

        self._logger.info(f'Learner {peer} (id={learner_id}) registered')
        self.display_learner_actors()

        self._learner_id += 1

        return evolver_pb2.RegisterLearnerResponse(name=self.name, id=learner_id)

    def RegisterActor(self, request, context):
        peer = context.peer()

        with self._learner_lock:
            if len(self._learner_actors) == 0:
                self._logger.info(f'Actor {peer} register failed')
                return evolver_pb2.RegisterActorResponse(succeeded=False)

            assigned_learner = sorted(self._learner_actors.items(),
                                      key=lambda t: len(t[1]))[0][0]

            self._learner_actors[assigned_learner].add(peer)
            self._actor_learner[peer] = assigned_learner

        assigned_learner = self._peer_set[assigned_learner]

        learner_host, learner_port = assigned_learner['learner_host'], assigned_learner['learner_port']
        replay_host, replay_port = assigned_learner['replay_host'], assigned_learner['replay_port']
        self._logger.info(f'Actor {peer} registered to learner {learner_host}:{learner_port}, replay {replay_host}:{replay_port}')
        self.display_learner_actors()

        return evolver_pb2.RegisterActorResponse(succeeded=True,
                                                 learner_host=learner_host,
                                                 learner_port=learner_port,
                                                 replay_host=replay_host,
                                                 replay_port=replay_port)

    # From learner
    def PostReward(self, request: evolver_pb2.PostRewardToEvolverRequest, context):
        self._post_reward(float(request.reward), context.peer())

        return Empty()

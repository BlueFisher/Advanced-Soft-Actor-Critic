import copy
import json
import logging
import threading
import time
from collections import deque
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import grpc
import numpy as np

import algorithm.config_helper as config_helper
from algorithm.utils import RLock

from .constants import *
from .proto import evolver_pb2, evolver_pb2_grpc, learner_pb2, learner_pb2_grpc
from .proto.ndarray_pb2 import Empty
from .proto.numproto import ndarray_to_proto, proto_to_ndarray
from .proto.pingpong_pb2 import Ping, Pong
from .proto.ma_variables_proto import ma_variables_to_proto, proto_to_ma_variables
from .utils import PeerSet, rpc_error_inspector


def update_nn_variables(stub, mean, std):
    nn_variables = [np.random.normal(mean[i], std[i]) for i in range(len(mean))]
    stub.update_nn_variables(nn_variables)


class ConfigGenerator:
    def __init__(self, config):
        self._ori_config = config

        for k, v in config.items():
            if v is not None and 'random_params' in v:
                random_params = v['random_params']
                for param, opt in random_params.items():
                    if 'in' in opt:
                        opt['dirichlet'] = [1] * len(opt['in'])

        self._learner_config = dict()

    def generate(self, learner):
        config = copy.deepcopy(self._ori_config)

        for k, v in config.items():
            if v is not None and 'random_params' in v:
                random_params = v['random_params']
                del v['random_params']

                for param, opt in random_params.items():
                    if 'in' in opt:
                        probs = np.random.dirichlet(opt['dirichlet'])
                        v[param] = np.random.choice(opt['in'], p=probs)
                        if isinstance(v[param], np.int32) or isinstance(v[param], np.int64):
                            v[param] = int(v[param])
                        elif isinstance(v[param], np.float64):
                            v[param] = float(v[param])
                        elif isinstance(v[param], np.bool_):
                            v[param] = bool(v[param])
                    elif 'std' in opt:
                        v[param] = np.random.normal(v[param], opt['std'])
                        if 'truncated' in opt:
                            v[param] = np.clip(v[param], opt['truncated'][0], opt['truncated'][1])
                    elif 'truncated' in opt:
                        v[param] = np.random.random() * (opt['truncated'][1] - opt['truncated'][0]) + opt['truncated'][0]

        self._learner_config[learner] = config

        return config

    def learner_selected(self, learner):
        for k, v in self._ori_config.items():
            if v is not None and 'random_params' in v:
                random_params = v['random_params']

                for param, opt in random_params.items():
                    if 'in' in opt:
                        i = opt['in'].index(self._learner_config[learner][k][param])
                        opt['dirichlet'][i] += 1


class Evolver:
    def __init__(self, root_dir, config_dir, args):
        self._logger = logging.getLogger('ds.evolver')

        self._init_config(root_dir, config_dir, args)

        self._config_generator = ConfigGenerator(self.config)

        self.base_config = self.config['base_config']
        self.net_config = self.config['net_config']

        self._learners = dict()
        self._lock = threading.Lock()

        self._last_update_nn_variable = time.time()
        self._ma_selected_times = None
        self._ma_saved_nn_variables_mean = None
        self._ma_saved_nn_variables_std = None

        self._update_nn_variables_executors = ThreadPoolExecutor(10)

        try:
            self._run()
        except KeyboardInterrupt:
            self._logger.warning('KeyboardInterrupt in _run')
            self.close()

    def _init_config(self, root_dir, config_dir, args):
        default_config_file_path = Path(__file__).resolve().parent.joinpath('default_config.yaml')
        config_abs_dir = Path(root_dir).joinpath(config_dir)
        config_abs_path = config_abs_dir.joinpath('config_ds.yaml')
        config = config_helper.initialize_config_from_yaml(default_config_file_path,
                                                           config_abs_path,
                                                           args.config,
                                                           is_evolver=True)

        if args.evolver_port is not None:
            config['net_config']['evolver_port'] = args.evolver_port
        if args.name is not None:
            config['base_config']['name'] = args.name

        config['base_config']['name'] = config_helper.generate_base_name(config['base_config']['name'], 'ds')
        model_abs_dir = Path(root_dir).joinpath('models',
                                                config['base_config']['env_name'],
                                                config['base_config']['name'])
        model_abs_dir.mkdir(parents=True, exist_ok=True)

        if args.logger_in_file:
            config_helper.set_logger(Path(model_abs_dir).joinpath('evolver.log'))

        config_helper.save_config(config, model_abs_dir, 'config_ds.yaml')

        config_helper.display_config(config, self._logger)

        self.config = config

    def _get_new_learner_config(self, peer):
        config = self._config_generator.generate(peer)
        return (config['reset_config'],
                config['model_config'],
                config['sac_config'])

    def _learner_connected(self, peer, connected):
        with self._lock:
            if connected:
                self._learners[peer] = {
                    'ma_rewards': {},
                    'ma_selected': {}
                }
            else:
                if peer in self._learners:
                    del self._learners[peer]

    def register_ma_names(self, peer, ma_names: List[str]):
        with self._lock:
            self._learners[peer]['ma_rewards'] = {
                n: deque(maxlen=self.base_config['evolver_cem_length']) for n in ma_names
            }
            self._learners[peer]['ma_selected'] = {
                n: 0 for n in ma_names
            }

        self._ma_selected_times = {n: 0 for n in ma_names}
        self._ma_saved_nn_variables_mean = {n: None for n in ma_names}
        self._ma_saved_nn_variables_std = {n: None for n in ma_names}

    def _post_reward(self, ma_name, reward, peer):
        if not self.base_config['evolver_enabled']:
            return

        with self._lock:
            self._learners[peer]['ma_rewards'][ma_name].append(reward)

            # If there is only one learner, return
            if len(self._learners) <= 1:
                return

            rewards_length_map = map(lambda x: len(x['ma_rewards'][ma_name]), self._learners.values())
            # All learners have evaluated more than evolver_cem_length times
            # or all learners have evaluated more than evolver_cem_min_length times and
            #   it has been more than evolver_cem_time mins since last evolution
            if all([l == self.base_config['evolver_cem_length'] for l in rewards_length_map]) or \
                    (all([l >= self.base_config['evolver_cem_min_length'] for l in rewards_length_map]) and
                     time.time() - self._last_update_nn_variable >= self.base_config['evolver_cem_time'] * 60):

                # Sort learners by the mean of evaluated rewards
                learner_reward = [(peer, float(np.mean(v['ma_rewards'][ma_name]))) for peer, v in self._learners.items()]
                learner_reward.sort(key=lambda i: i[1], reverse=True)

                # Select top evolver_cem_best learners and get their nn variables
                best_size = int(len(learner_reward) * self.base_config['evolver_cem_best'])
                best_size = max(best_size, 1)

                best_learners = [i[0] for i in learner_reward[:best_size]]
                nn_variables_list = list()
                for learner in best_learners:
                    self._learners[learner]['ma_selected'][ma_name] += 1
                    self._config_generator.learner_selected(learner)
                    stub = self.servicer.get_learner_stub(learner)
                    if stub:
                        nn_variables = stub.get_nn_variables(ma_name)
                        if nn_variables is not None:
                            for v in nn_variables:
                                if np.isnan(np.min(v)):
                                    self._logger.warning('NAN in learner nn_variables, closing')
                                    self.close()
                                    return
                            nn_variables_list.append(nn_variables)

                if len(nn_variables_list) == 0:
                    self._logger.warning('No nn_variables_list')
                    return

                # Calculate the mean and std of best_size variables of learners
                mean = [np.mean(i, axis=0) for i in zip(*nn_variables_list)]
                std = [np.minimum(np.std(i, axis=0), 1.) for i in zip(*nn_variables_list)]

                self._saved_nn_variables_mean, self._saved_nn_variables_std = mean, std

                self._selected_times += 1

                # Remove the least selected learner
                if self.base_config['evolver_remove_worst'] != -1:
                    learner_selecteds = [(l, v['selected']) for l, v in self._learners.items()]
                    mean_selected = np.mean([s for _, s in learner_selecteds])
                    learner_selecteds.sort(key=lambda x: x[1])
                    worst_learner, worst_learner_selected = learner_selecteds[0]

                    worst_degree = abs(mean_selected - worst_learner_selected)
                    self._logger.info(f'Worst degree: {worst_degree}')

                    if worst_degree >= self.base_config['evolver_remove_worst']:
                        stub = self.servicer.get_learner_stub(worst_learner)
                        stub.force_close()
                        del self._learners[worst_learner]
                        self._logger.info(f'Removed the least selected learner {self.servicer.get_learner_id(worst_learner)}')
                        for l, v in self._learners.items():
                            v['selected'] = 1 if l in best_learners else 0

                # Dispatch all nn variables
                for learner in self._learners.keys():
                    stub = self.servicer.get_learner_stub(learner)
                    if stub:
                        self._update_nn_variables_executors.submit(update_nn_variables,
                                                                   stub, mean, std)

                    self._learners[learner]['rewards'].clear()

                self._last_update_nn_variable = time.time()

                # Log
                _best_learner_ids = [str(self.servicer.get_learner_id(l)) for l in best_learners]
                self._logger.info(f'{self._selected_times}, Selected {",".join(_best_learner_ids)} learners')

                _learner_id_selecteds = [(str(self.servicer.get_learner_id(l)), v['selected']) for l, v in self._learners.items()]
                _learner_id_selecteds.sort(key=lambda x: x[1], reverse=True)
                _learner_id_selecteds = [f'{i[0]}({i[1]})' for i in _learner_id_selecteds]
                self._logger.info(f'Learner id (selected): {", ".join(_learner_id_selecteds)}')

                std = [(np.min(s), np.mean(s), np.max(s)) for s in std]
                _min, _mean, _max = [np.mean(s) for s in zip(*std)]
                self._logger.info(f'Variables std: {_min:.2f}, {_mean:.2f}, {_max:.2f}')

    def _get_ma_nn_variables(self):
        if self._saved_nn_variables_mean is None:
            return None

        mean, std = self._saved_nn_variables_mean, self._saved_nn_variables_std
        return [np.random.normal(mean[i], std[i]) for i in range(len(mean))]

    def _run(self):
        self.servicer = EvolverService(self.base_config['name'],
                                       self.base_config['max_actors_each_learner'],
                                       self)
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_THREAD_WORKERS))
        evolver_pb2_grpc.add_EvolverServiceServicer_to_server(self.servicer, self.server)
        self.server.add_insecure_port(f'[::]:{self.net_config["evolver_port"]}')
        self.server.start()
        self._logger.info(f'Evolver server is running on [{self.net_config["evolver_port"]}]...')

        self.server.wait_for_termination()

    def close(self):
        self.server.stop(0)

        self._logger.warning('Closed')


class LearnerStubController:
    def __init__(self, host, port):
        self._channel = grpc.insecure_channel(f'{host}:{port}', [
            ('grpc.max_reconnect_backoff_ms', MAX_RECONNECT_BACKOFF_MS),
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)
        ])
        self._stub = learner_pb2_grpc.LearnerServiceStub(self._channel)

        self._logger = logging.getLogger('ds.evolver.learner_stub')

    @rpc_error_inspector
    def get_nn_variables(self, ma_name: str):
        response = self._stub.GetNNVariables(learner_pb2.GetNNVariablesRequest(ma_name=ma_name))
        return [proto_to_ndarray(v) for v in response.variables]

    @rpc_error_inspector
    def update_ma_nn_variables(self, ma_variables):
        self._stub.UpdateMANNVariables(ma_variables_to_proto(ma_variables))

    @rpc_error_inspector
    def force_close(self):
        self._stub.ForceClose(Empty())

    def close(self):
        self._channel.close()


class EvolverService(evolver_pb2_grpc.EvolverServiceServicer):
    def __init__(self, name, max_actors_each_learner,
                 evolver: Evolver):
        self.name = name
        self.max_actors_each_learner = max_actors_each_learner

        self._learner_connected = evolver._learner_connected
        self._get_new_learner_config = evolver._get_new_learner_config
        self._post_reward = evolver._post_reward
        self._get_ma_nn_variables = evolver._get_ma_nn_variables

        self._logger = logging.getLogger('ds.evolver.service')
        self._peer_set = PeerSet(self._logger)

        self._lock = RLock(timeout=1, logger=self._logger)
        self._learner_id = 0
        self._learner_actors = dict()
        """
        {
            learner_peer: {actor_peer, ...}
        }
        """
        self._actor_learner = dict()
        """
        {
            actor_peer: learner_peer
        }
        """

    def _record_peer(self, context):
        peer = context.peer()

        def _unregister_peer():
            with self._lock:
                if peer in self._learner_actors:
                    info = self._peer_set.get_info(peer)
                    _id = info['id']
                    del self._learner_actors[peer]
                    self._learner_connected(peer, connected=False)
                    self._logger.warning(f'Learner {peer} (id={_id}) disconnected')
                elif peer in self._actor_learner:
                    learner_peer = self._actor_learner[peer]
                    if learner_peer in self._learner_actors:
                        self._learner_actors[learner_peer].remove(peer)
                    del self._actor_learner[peer]
                    self._logger.warning(f'Actor {peer} disconnected')
            self._peer_set.disconnect(peer)

        context.add_callback(_unregister_peer)
        self._peer_set.connect(peer)

    @property
    def learners(self):
        with self._lock:
            return list(self._learner_actors.keys())

    def get_learner_stub(self, peer):
        info = self._peer_set.get_info(peer)
        if info:
            return info['stub']

    def get_learner_id(self, peer):
        info = self._peer_set.get_info(peer)
        if info:
            return info['id']

    def display_learner_actors(self):
        with self._lock:
            info = 'Learner-actors:'
            for learner in self._learner_actors:
                learner_info = self._peer_set.get_info(learner)
                learner_id = learner_info['id']
                learner_host = learner_info['learner_host']
                learner_port = learner_info['learner_port']
                info += f'\n{learner} ({learner_host}:{learner_port} id={learner_id}): '
                info += f'[Actors count]: {len(self._learner_actors[learner])}'

        self._logger.info(info)

    # From learner and actor
    def Persistence(self, request_iterator, context):
        self._record_peer(context)
        for request in request_iterator:
            yield Pong(time=int(time.time() * 1000))

    def RegisterLearner(self, request: evolver_pb2.RegisterLearnerRequest, context):
        peer = context.peer()
        self._logger.info(f'{peer} is registering learner...')

        with self._lock:
            learner_id = self._learner_id

            self._peer_set.add_info(peer, {
                'id': learner_id,
                'learner_host': request.learner_host,
                'learner_port': request.learner_port,
                'stub': LearnerStubController(request.learner_host, request.learner_port)
            })

            self._learner_actors[peer] = set()

            self._learner_connected(peer, connected=True)

            self._logger.info(f'Learner {peer} (id={learner_id}, {request.learner_host}:{request.learner_port}) registered')

            self.display_learner_actors()

            self._learner_id += 1

        (reset_config,
         model_config,
         sac_config) = self._get_new_learner_config(peer)
        return evolver_pb2.RegisterLearnerResponse(name=self.name, id=learner_id,
                                                   reset_config_json=json.dumps(reset_config),
                                                   model_config_json=json.dumps(model_config),
                                                   sac_config_json=json.dumps(sac_config))

    def RegisterLearnerMANames(self, request, context):
        peer = context.peer()

    def RegisterActor(self, request, context):
        peer = context.peer()
        self._logger.info(f'{peer} starts registering actor')

        with self._lock:
            if len(self._learner_actors) == 0:
                self._logger.info(f'Actor {peer} register failed, no learner exists')
                return evolver_pb2.RegisterActorResponse(succeeded=False)

            assigned_learner = sorted(self._learner_actors.items(),
                                      key=lambda t: len(t[1]))[0][0]

            if len(self._learner_actors[assigned_learner]) == self.max_actors_each_learner:
                self._logger.info(f'Actor {peer} register failed, all learners have max actors')
                return evolver_pb2.RegisterActorResponse(succeeded=False)

            self._learner_actors[assigned_learner].add(peer)
            self._actor_learner[peer] = assigned_learner

        learner_info = self._peer_set.get_info(assigned_learner)
        learner_id = learner_info['id']
        learner_host, learner_port = learner_info['learner_host'], learner_info['learner_port']

        log = f'Actor {peer} registered to ' +\
            f'learner (id={learner_id} {learner_host}:{learner_port})'
        self._logger.info(log)
        self.display_learner_actors()

        return evolver_pb2.RegisterActorResponse(succeeded=True,
                                                 learner_host=learner_host,
                                                 learner_port=learner_port)

    # From learner
    def PostReward(self, request: evolver_pb2.PostRewardToEvolverRequest, context):
        self._post_reward(float(request.reward), context.peer())

        return Empty()

    # From learner
    def GetMANNVariables(self, request, context):
        ma_variables = self._get_ma_nn_variables()  # TODO
        if ma_variables is None:
            return ma_variables_to_proto(None)
        else:
            return ma_variables_to_proto(ma_variables)

import copy
import importlib
import json
import logging
import multiprocessing as mp
import shutil
import socket
import sys
import threading
import time
from concurrent import futures
from pathlib import Path
from typing import List

import grpc
import numpy as np

import algorithm.config_helper as config_helper
from algorithm.agent import Agent, MultiAgentsManager
from algorithm.utils import (ReadWriteLock, RLock, UselessEpisodeException,
                             gen_pre_n_actions)
from algorithm.utils.enums import *

from .constants import *
from .learner_trainer import Trainer
from .proto import evolver_pb2, evolver_pb2_grpc, learner_pb2, learner_pb2_grpc
from .proto.ndarray_pb2 import Empty
from .proto.numproto import ndarray_to_proto, proto_to_ndarray
from .proto.pingpong_pb2 import Ping, Pong
from .proto.ma_variables_proto import ma_variables_to_proto, proto_to_ma_variables
from .sac_ds_base import SAC_DS_Base
from .utils import (PeerSet, SharedMemoryManager, get_episode_shapes_dtypes,
                    rpc_error_inspector)


class Learner:
    _agent_class = Agent
    _ma_policy_variables_cache = {}

    def __init__(self, root_dir, config_dir, args):
        self._closed = False
        self._registered = False

        self._logger = logging.getLogger('ds.learner')

        constant_config, config_abs_dir = self._init_constant_config(root_dir, config_dir, args)
        learner_host = constant_config['net_config']['learner_host']
        learner_port = constant_config['net_config']['learner_port']

        self._evolver_stub = EvolverStubController(
            constant_config['net_config']['evolver_host'],
            constant_config['net_config']['evolver_port']
        )

        (_id, name,
         reset_config,
         model_config,
         sac_config) = self._evolver_stub.register_to_evolver(learner_host, learner_port)

        self._logger.info(f'Registered id: {_id}, name: {name}')

        constant_config['base_config']['name'] = name
        constant_config['reset_config'] = reset_config
        constant_config['model_config'] = model_config
        constant_config['sac_config'] = sac_config
        self._registered = True

        self._init_config(_id, root_dir, constant_config, args)
        self._init_env()
        self._init_sac(config_abs_dir)

        threading.Thread(target=self._policy_evaluation, daemon=True).start()

        self._run_learner_server(learner_port)

        self.close()

    def _init_constant_config(self, root_dir, config_dir, args):
        default_config_abs_path = Path(__file__).resolve().parent.joinpath('default_config.yaml')
        config_abs_dir = Path(root_dir).joinpath(config_dir)
        config_abs_path = config_abs_dir.joinpath('config_ds.yaml')
        config = config_helper.initialize_config_from_yaml(default_config_abs_path,
                                                           config_abs_path,
                                                           args.config)

        # Initialize config from command line arguments
        self.logger_in_file = args.logger_in_file
        self.render = args.render

        self.run_in_editor = args.editor

        self.device = args.device
        self.last_ckpt = args.ckpt

        if args.evolver_host is not None:
            config['net_config']['evolver_host'] = args.evolver_host
        if args.evolver_port is not None:
            config['net_config']['evolver_port'] = args.evolver_port
        if args.learner_host is not None:
            config['net_config']['learner_host'] = args.learner_host
        if args.learner_port is not None:
            config['net_config']['learner_port'] = args.learner_port

        if config['net_config']['evolver_host'] is None:
            self._logger.fatal('evolver_host is None')

        # If learner_host is not set, use ip as default
        if config['net_config']['learner_host'] is None:
            hostname = socket.gethostname()
            ip = socket.gethostbyname(hostname)
            config['net_config']['learner_host'] = ip

        return config, config_abs_dir

    def _init_config(self, _id, root_dir, config, args):
        if args.name is not None:
            config['base_config']['name'] = args.name
        if args.env_args is not None:
            config['base_config']['env_args'] = args.env_args
        if args.build_port is not None:
            config['base_config']['unity_args']['build_port'] = args.build_port

        if args.nn is not None:
            config['base_config']['nn'] = args.nn
        if args.agents is not None:
            config['base_config']['n_agents'] = args.agents

        model_abs_dir = Path(root_dir).joinpath('models',
                                                config['base_config']['env_name'],
                                                config['base_config']['name'],
                                                f'learner{_id}')
        model_abs_dir.mkdir(parents=True, exist_ok=True)
        self.model_abs_dir = model_abs_dir

        if args.logger_in_file:
            config_helper.set_logger(Path(model_abs_dir).joinpath(f'learner.log'))

        config_helper.display_config(config, self._logger)

        convert_config_to_enum(config['sac_config'])

        self.config = config
        self.base_config = config['base_config']
        self.reset_config = config['reset_config']
        self.model_config = config['model_config']
        self.sac_config = config['sac_config']

    def _init_env(self):
        if self.base_config['env_type'] == 'UNITY':
            from algorithm.env_wrapper.unity_wrapper import UnityWrapper

            if self.run_in_editor:
                self.env = UnityWrapper(train_mode=False,
                                        n_agents=self.base_config['n_agents'])
            else:
                self.env = UnityWrapper(train_mode=False,
                                        file_name=self.base_config['unity_args']['build_path'][sys.platform],
                                        base_port=self.base_config['unity_args']['build_port'],
                                        no_graphics=self.base_config['unity_args']['no_graphics'] and not self.render,
                                        scene=self.base_config['env_name'],
                                        additional_args=self.base_config['env_args'],
                                        n_agents=self.base_config['n_agents'])

        elif self.base_config['env_type'] == 'GYM':
            from algorithm.env_wrapper.gym_wrapper import GymWrapper

            self.env = GymWrapper(train_mode=False,
                                  env_name=self.base_config['env_name'],
                                  render=self.render,
                                  n_agents=self.base_config['n_agents'])

        elif self.base_config['env_type'] == 'DM_CONTROL':
            from algorithm.env_wrapper.dm_control_wrapper import \
                DMControlWrapper

            self.env = DMControlWrapper(train_mode=False,
                                        env_name=self.base_config['env_name'],
                                        render=self.render,
                                        n_agents=self.base_config['n_agents'])

        elif self.base_config['env_type'] == 'TEST':
            from algorithm.env_wrapper.test_wrapper import TestWrapper

            self.env = TestWrapper(env_args=self.base_config['env_args'],
                                   n_agents=self.base_config['n_agents'])

        else:
            raise RuntimeError(f'Undefined Environment Type: {self.base_config["env_type"]}')

        ma_obs_shapes, ma_d_action_size, ma_c_action_size = self.env.init()
        self.ma_manager = MultiAgentsManager(self._agent_class,
                                             ma_obs_shapes,
                                             ma_d_action_size,
                                             ma_c_action_size)

        self._logger.info(f'{self.base_config["env_name"]} initialized')

    def _init_sac(self, config_abs_dir: Path):
        # If nn model exists, load saved model, or copy a new one
        nn_model_abs_path = self.model_abs_dir.joinpath('nn_models.py')
        if nn_model_abs_path.exists():
            spec = importlib.util.spec_from_file_location('nn', str(nn_model_abs_path))
        else:
            nn_abs_path = config_abs_dir.joinpath(f'{self.base_config["nn"]}.py')
            spec = importlib.util.spec_from_file_location('nn', str(nn_abs_path))
            shutil.copyfile(nn_abs_path, nn_model_abs_path)

        custom_nn_model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_nn_model)

        for n, mgr in self.ma_manager:
            mgr.set_sac(SAC_DS_Base(obs_shapes=mgr.obs_shapes,
                                    d_action_size=mgr.d_action_size,
                                    c_action_size=mgr.c_action_size,
                                    model_abs_dir=None,
                                    model=custom_nn_model,
                                    model_config=self.model_config,
                                    device=self.device,
                                    ma_name=None if len(self.ma_manager) == 1 else n,
                                    train_mode=False,
                                    last_ckpt=self.last_ckpt,

                                    **self.sac_config))
            mgr['lock'] = ReadWriteLock(None, 2, 2, logger=self._logger)

            self._logger.info(f'{n} SAC_BAK started')

        # Initialize all queues for learner_trainer
        for n, mgr in self.ma_manager:
            mgr['all_variables_buffer'] = SharedMemoryManager(1)
            mgr['all_variables_buffer'].init_from_data_buffer(mgr.sac.get_all_variables())

            max_episode_length = self.base_config['max_episode_length']

            episode_shapes, episode_dtypes = get_episode_shapes_dtypes(
                max_episode_length,
                mgr.obs_shapes,
                mgr.action_size,
                mgr.sac.seq_hidden_state_shape if mgr.seq_encoder is not None else None)

            mgr['episode_buffer'] = SharedMemoryManager(self.base_config['episode_queue_size'],
                                                        logger=self._logger,
                                                        counter_get_shm_index_empty_log='Episode shm index is empty',
                                                        timer_get_shm_index_log='Get an episode shm index',
                                                        timer_get_data_log='Get an episode',
                                                        log_repeat=ELAPSED_REPEAT)
            mgr['episode_buffer'].init_from_shapes(episode_shapes, episode_dtypes)
            mgr['episode_length_array'] = mp.Array('i', range(self.base_config['episode_queue_size']))

            mgr['cmd_pipe_client'], cmd_pipe_server = mp.Pipe()
            mgr['learner_trainer_process'] = mp.Process(target=Trainer, kwargs={
                'all_variables_buffer': mgr['all_variables_buffer'],
                'episode_buffer': mgr['episode_buffer'],
                'episode_length_array': mgr['episode_length_array'],
                'cmd_pipe_server': cmd_pipe_server,

                'logger_in_file': self.logger_in_file,
                'obs_shapes': mgr.obs_shapes,
                'd_action_size': mgr.d_action_size,
                'c_action_size': mgr.c_action_size,
                'model_abs_dir': self.model_abs_dir,
                'model_spec': spec,
                'device': self.device,
                'ma_name': None if len(self.ma_manager) == 1 else n,
                'last_ckpt': self.last_ckpt,
                'config': self.config
            })
            mgr['learner_trainer_process'].start()

        ma_nn_variables = self._evolver_stub.get_nn_variables()
        # TODO
        # if ma_nn_variables:
        #     self._udpate_nn_variables(ma_nn_variables)
        #     self._logger.info(f'Initialized from evolver')

        self._logger.info('Waiting for updating sac_bak in first time')
        self._update_sac_bak()

        threading.Thread(target=self._forever_update_sac_bak, daemon=True).start()

    def _update_sac_bak(self):
        for n, mgr in self.ma_manager:
            all_variables, _ = mgr['all_variables_buffer'].get()  # Block, waiting for all variables available

            with mgr['lock'].write():
                res = mgr.sac.update_all_variables(all_variables)

            if not res:
                self._logger.warning('NAN in variables, closing...')
                self._force_close()
                return

            self._ma_policy_variables_cache[n] = mgr.sac.get_policy_variables()

        self._logger.info('Updated sac_bak')

    def _forever_update_sac_bak(self):
        while True:
            self._update_sac_bak()

    def _get_actor_register_result(self, actor_id):
        if self._registered:
            noise = self.base_config['noise_increasing_rate'] * actor_id
            noise = min(noise, self.base_config['noise_max'])

            actor_sac_config = copy.deepcopy(self.sac_config)
            actor_sac_config['action_noise'] = [noise, noise]
            convert_config_to_string(actor_sac_config)

            return (str(self.model_abs_dir),
                    self.reset_config,
                    self.model_config,
                    actor_sac_config)

    def _get_ma_policy_variables(self):
        return self._ma_policy_variables_cache

    def _get_ma_nn_variables(self):
        ma_nn_variables = {}

        for n, mgr in self.ma_manager:
            mgr['cmd_pipe_client'].send(('GET', None))
            ma_nn_variables[n] = mgr['cmd_pipe_client'].recv()

        return ma_nn_variables

    def _udpate_ma_nn_variables(self, ma_variables):
        for n, mgr in self.ma_manager:
            mgr['cmd_pipe_client'].send(('UPDATE', ma_variables[n]))

    def _save_model(self):
        for n, mgr in self.ma_manager:
            mgr['cmd_pipe_client'].send(('SAVE_MODEL', None))

    def _add_episode(self,
                     ma_name: str,
                     l_indexes: np.ndarray,
                     l_padding_masks: np.ndarray,
                     l_obses_list: List[np.ndarray],
                     l_actions: np.ndarray,
                     l_rewards: np.ndarray,
                     next_obs_list: List[np.ndarray],
                     l_dones: np.ndarray,
                     l_probs: List[np.ndarray],
                     l_seq_hidden_states: np.ndarray = None):
        """
        Args:
            ma_name: str
            l_indexes: [1, episode_len]
            l_padding_masks: [1, episode_len]
            l_obses_list: list([1, episode_len, *obs_shapes_i], ...)
            l_actions: [1, episode_len, action_size]
            l_rewards: [1, episode_len]
            next_obs_list: list([1, *obs_shapes_i], ...)
            l_dones: [1, episode_len]
            l_probs: [1, episode_len]
            l_seq_hidden_states: [1, episode_len, *seq_hidden_state_shape]
        """
        episode_idx = self.ma_manager[ma_name]['episode_buffer'].put([
            l_indexes,
            l_padding_masks,
            l_obses_list,
            l_actions,
            l_rewards,
            next_obs_list,
            l_dones,
            l_probs,
            l_seq_hidden_states
        ])
        self.ma_manager[ma_name]['episode_length_array'][episode_idx] = l_indexes.shape[1]

    def _policy_evaluation(self):
        self.ma_manager.init(self.base_config['n_agents'])

        ma_obs_list = self.env.reset(reset_config=self.reset_config)
        self.ma_manager.set_obs_list(ma_obs_list)

        force_reset = False
        iteration = 0
        start_time = time.time()

        try:
            while not self._closed:
                if self.base_config['reset_on_iteration'] \
                        or self.ma_manager.is_max_reached() \
                        or force_reset:
                    ma_obs_list = self.env.reset(reset_config=self.reset_config)
                    self.ma_manager.set_obs_list(ma_obs_list)
                    self.ma_manager.clear()

                    force_reset = False
                else:
                    self.ma_manager.reset()

                step = 0

                while not self.ma_manager.is_done() \
                        and not self._closed:
                    self.ma_manager.burn_in_padding()

                    for n, mgr in self.ma_manager:
                        with mgr['lock'].read('choose_action'):
                            mgr.get_action()

                    ma_d_action = {n: mgr['d_action'] for n, mgr in self.ma_manager}
                    ma_c_action = {n: mgr['c_action'] for n, mgr in self.ma_manager}

                    (ma_next_obs_list,
                     ma_reward,
                     ma_local_done,
                     ma_max_reached) = self.env.step(ma_d_action, ma_c_action)

                    if ma_next_obs_list is None:
                        force_reset = True

                        if self.base_config['evolver_enabled']:
                            self._evolver_stub.post_reward(float('-inf'))

                        self._logger.warning('Step encounters error, episode ignored')
                        continue

                    for n, mgr in self.ma_manager:
                        if step == self.base_config['max_step_each_iter']:
                            ma_local_done[n] = [True] * len(mgr.agents)
                            ma_max_reached[n] = [True] * len(mgr.agents)

                        for i, a in enumerate(mgr.agents):
                            a.add_transition(
                                obs_list=[o[i] for o in mgr['obs_list']],
                                action=mgr['action'][i],
                                reward=ma_reward[n][i],
                                local_done=ma_local_done[n][i],
                                max_reached=ma_max_reached[n][i],
                                next_obs_list=[o[i] for o in ma_next_obs_list[n]],
                                prob=mgr['prob'][i],
                                is_padding=False,
                                seq_hidden_state=mgr['seq_hidden_state'][i] if mgr.seq_encoder is not None else None,
                            )

                    self.ma_manager.post_step(ma_next_obs_list, ma_local_done)

                    step += 1

                self._log_episode_summaries()
                self._log_episode_info(iteration, start_time)

                p = self.model_abs_dir.joinpath('save_model')
                if p.exists():
                    self._save_model()
                    p.unlink()

                # TODO
                # if self.base_config['evolver_enabled']:
                #     self._evolver_stub.post_reward(np.mean([a.reward for a in agents]))

                iteration += 1

        finally:
            self._save_model()
            self.env.close()

            self._logger.warning('Evaluation terminated')

    def _log_episode_summaries(self):
        for n, mgr in self.ma_manager:
            rewards = np.array([a.reward for a in mgr.agents])

            mgr['cmd_pipe_client'].send(('LOG_EPISODE_SUMMARIES', [
                {'tag': 'reward/mean', 'simple_value': float(rewards.mean())},
                {'tag': 'reward/max', 'simple_value': float(rewards.max())},
                {'tag': 'reward/min', 'simple_value': float(rewards.min())}
            ]))

    def _log_episode_info(self, iteration, start_time):
        time_elapse = (time.time() - start_time) / 60
        for n, mgr in self.ma_manager:
            rewards = [a.reward for a in mgr.agents]
            rewards = ", ".join([f"{i:6.1f}" for i in rewards])
            max_step = max([a.steps for a in mgr.agents])

            self._logger.info(f'{n} {iteration}, {time_elapse:.2f}m, S {max_step}, R {rewards}')

    def _run_learner_server(self, learner_port):
        servicer = LearnerService(self)
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_THREAD_WORKERS),
                                  options=[
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)
        ])
        learner_pb2_grpc.add_LearnerServiceServicer_to_server(servicer, self.server)
        self.server.add_insecure_port(f'[::]:{learner_port}')
        self.server.start()
        self._logger.info(f'Learner server is running on [{learner_port}]...')

        self.server.wait_for_termination()

    def _force_close(self):
        self._logger.warning('Force closing')
        f = open(self.model_abs_dir.joinpath('force_closed'), 'w')
        f.close()
        self.close()

    def close(self):
        self._closed = True

        if hasattr(self, 'env'):
            self.env.close()
        if hasattr(self, 'server'):
            self.server.stop(None)

        self._evolver_stub.close()
        self.learner_trainer_process.close()

        self._logger.warning('Closed')


class LearnerService(learner_pb2_grpc.LearnerServiceServicer):
    _ma_policy_variables_cache = None

    def __init__(self, learner: Learner):
        self._get_actor_register_result = learner._get_actor_register_result

        self._add_episode = learner._add_episode
        self._get_ma_policy_variables = learner._get_ma_policy_variables
        self._get_ma_nn_variables = learner._get_ma_nn_variables
        self._udpate_ma_nn_variables = learner._udpate_ma_nn_variables

        self._force_close = learner._force_close

        self._logger = logging.getLogger('ds.learner.service')
        self._peer_set = PeerSet(self._logger)

        self._lock = RLock(timeout=1, logger=self._logger)
        self._actor_id = 0

    def _record_peer(self, context):
        peer = context.peer()

        def _unregister_peer():
            with self._lock:
                self._logger.warning(f'Actor {peer} disconnected')
                self._peer_set.disconnect(context.peer())

        context.add_callback(_unregister_peer)
        self._peer_set.connect(peer)

    def Persistence(self, request_iterator, context):
        self._record_peer(context)
        for request in request_iterator:
            yield Pong(time=int(time.time() * 1000))

    def RegisterActor(self, request, context):
        peer = context.peer()
        self._logger.info(f'{peer} is registering actor...')

        with self._lock:
            res = self._get_actor_register_result(self._actor_id)
            if res is not None:
                actor_id = self._actor_id

                self._peer_set.add_info(peer, {
                    'ma_policy_variables_id_cache': None
                })

                self._logger.info(f'Actor {peer} registered')

                self._actor_id += 1

                (model_abs_dir,
                 reset_config,
                 model_config,
                 sac_config) = res
                return learner_pb2.RegisterActorResponse(model_abs_dir=model_abs_dir,
                                                         unique_id=actor_id,
                                                         reset_config_json=json.dumps(reset_config),
                                                         model_config_json=json.dumps(model_config),
                                                         sac_config_json=json.dumps(sac_config))
            else:
                return learner_pb2.RegisterActorResponse(unique_id=-1)

    # From actor
    def GetPolicyVariables(self, request, context):
        peer = context.peer()

        ma_policy_variables = self._get_ma_policy_variables()
        if len(ma_policy_variables) == 0:
            return ma_variables_to_proto(None)

        self._ma_policy_variables_cache = ma_policy_variables

        actor_info = self._peer_set.get_info(peer)

        if id(list(self._ma_policy_variables_cache.values())[0]) == actor_info['ma_policy_variables_id_cache']:
            return ma_variables_to_proto(None)

        actor_info['ma_policy_variables_id_cache'] = id(list(self._ma_policy_variables_cache.values())[0])
        return ma_variables_to_proto(self._ma_policy_variables_cache)

    # From actor
    def Add(self, request, context):
        self._add_episode(request.ma_name,
                          proto_to_ndarray(request.l_indexes),
                          proto_to_ndarray(request.l_padding_masks),
                          [proto_to_ndarray(l_obses) for l_obses in request.l_obses_list],
                          proto_to_ndarray(request.l_actions),
                          proto_to_ndarray(request.l_rewards),
                          [proto_to_ndarray(obs) for obs in request.next_obs_list],
                          proto_to_ndarray(request.l_dones),
                          proto_to_ndarray(request.l_mu_probs),
                          proto_to_ndarray(request.l_seq_hidden_states))

        return Empty()

    # From evolver
    def GetNNVariables(self, request, context):
        ma_nn_variables = self._get_ma_nn_variables()
        return ma_variables_to_proto(ma_nn_variables)

    # From evolver
    def UpdateNNVariables(self, request, context):
        ma_variables = ma_variables_to_proto(request)
        self._udpate_ma_nn_variables(ma_variables)
        return Empty()

    def ForceClose(self, request, context):
        self._force_close()
        return Empty()


class EvolverStubController:
    _closed = False

    def __init__(self, evolver_host, evolver_port):
        self._logger = logging.getLogger('ds.learner.evolver_stub')

        self._evolver_channel = grpc.insecure_channel(f'{evolver_host}:{evolver_port}', [
            ('grpc.max_reconnect_backoff_ms', MAX_RECONNECT_BACKOFF_MS),
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)
        ])
        self._evolver_stub = evolver_pb2_grpc.EvolverServiceStub(self._evolver_channel)
        self._logger.info(f'Starting evolver stub [{evolver_host}:{evolver_port}]')

        self._evolver_connected = False

        t_evolver = threading.Thread(target=self._start_evolver_persistence, daemon=True)
        t_evolver.start()

    @property
    def connected(self):
        return not self._closed and self._evolver_connected

    @rpc_error_inspector
    def register_to_evolver(self, learner_host, learner_port):
        self._logger.warning('Waiting for evolver connection')
        while not self.connected:
            time.sleep(RECONNECTION_TIME)
            continue

        response = None
        self._logger.info('Registering to evolver...')
        while response is None:
            response = self._evolver_stub.RegisterLearner(
                evolver_pb2.RegisterLearnerRequest(learner_host=learner_host,
                                                   learner_port=learner_port))

            if response:
                self._logger.info('Registered to evolver')

                return (response.id, response.name,
                        json.loads(response.reset_config_json),
                        json.loads(response.model_config_json),
                        json.loads(response.sac_config_json))
            else:
                response = None
                time.sleep(RECONNECTION_TIME)

    @rpc_error_inspector
    def post_reward(self, reward):
        self._evolver_stub.PostReward(
            evolver_pb2.PostRewardToEvolverRequest(reward=float(reward)))

    @rpc_error_inspector
    def get_nn_variables(self):
        response = self._evolver_stub.GetNNVariables(Empty())
        return proto_to_ma_variables(response)

    def _start_evolver_persistence(self):
        def request_messages():
            while not self._closed:
                yield Ping(time=int(time.time() * 1000))
                time.sleep(PING_INTERVAL)
                if not self._evolver_connected:
                    break

        while not self._closed:
            try:
                reponse_iterator = self._evolver_stub.Persistence(request_messages())
                for response in reponse_iterator:
                    if not self._evolver_connected:
                        self._evolver_connected = True
                        self._logger.info('Evolver connected')
            except grpc.RpcError:
                if self._evolver_connected:
                    self._evolver_connected = False
                    self._logger.error('Evolver disconnected')
            finally:
                time.sleep(RECONNECTION_TIME)

    def close(self):
        self._evolver_channel.close()
        self._closed = True

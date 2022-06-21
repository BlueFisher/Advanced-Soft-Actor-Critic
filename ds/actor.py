import importlib
import json
import logging
import logging.handlers
import multiprocessing as mp
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import List

import grpc
import numpy as np

import algorithm.config_helper as config_helper
from algorithm.agent import Agent, MultiAgentsManager
from algorithm.utils import ReadWriteLock, elapsed_timer, gen_pre_n_actions
from algorithm.utils.enums import *

from .constants import *
from .proto import learner_pb2, learner_pb2_grpc
from .proto.ma_variables_proto import (ma_variables_to_proto,
                                       proto_to_ma_variables)
from .proto.ndarray_pb2 import Empty
from .proto.numproto import ndarray_to_proto, proto_to_ndarray
from .proto.pingpong_pb2 import Ping, Pong
from .sac_ds_base import SAC_DS_Base
from .utils import (SharedMemoryManager, get_episode_shapes_dtypes,
                    rpc_error_inspector, traverse_lists)


class EpisodeSender:
    def __init__(self,
                 logger_in_file: bool,
                 model_abs_dir: str,
                 learner_host: str,
                 learner_port: int,
                 ma_name: str,
                 episode_buffer: SharedMemoryManager,
                 episode_length_array: mp.Array):
        self.ma_name = ma_name
        self._episode_buffer = episode_buffer
        self._episode_length_array = episode_length_array

        config_helper.set_logger(Path(model_abs_dir).joinpath(f'actor_episode_sender_{os.getpid()}.log') if logger_in_file else None)
        self._logger = logging.getLogger(f'ds.actor.episode_sender_{os.getpid()}')

        self._stub = StubEpisodeSenderController(learner_host, learner_port)

        self._logger.info(f'EpisodeSender {os.getpid()} initialized')

        episode_buffer.init_logger(self._logger)

        self._run()

    def _run(self):
        timer_add_trans = elapsed_timer(self._logger, 'Add trans', repeat=ELAPSED_REPEAT)

        while True:
            episode, episode_idx = self._episode_buffer.get(timeout=EPISODE_QUEUE_TIMEOUT)
            if episode is None:
                continue

            episode_length = self._episode_length_array[episode_idx]

            episode = traverse_lists(episode, lambda e: e[:, :episode_length])

            with timer_add_trans:
                self._stub.add_transitions(self.ma_name, *episode)


class Actor(object):
    _agent_class = Agent

    def __init__(self, root_dir, config_dir, args):
        self._logger = logging.getLogger('ds.actor')

        config_abs_dir = self._init_config(root_dir, config_dir, args)

        self._sac_actor_lock = ReadWriteLock(5, 1, 1, logger=self._logger)
        self._init_env()
        self._init_sac(config_abs_dir)
        self._init_episode_sender()

        self._run()

        self.close()

    def _init_config(self, root_dir, config_dir, args):
        config_abs_dir = Path(root_dir).joinpath(config_dir)
        config_abs_path = config_abs_dir.joinpath('config_ds.yaml')
        default_config_abs_path = Path(__file__).resolve().parent.joinpath('default_config.yaml')
        # Merge default_config.yaml and custom config.yaml
        config, ma_configs = config_helper.initialize_config_from_yaml(default_config_abs_path,
                                                                       config_abs_path,
                                                                       args.config)

        # Initialize config from command line arguments
        self.logger_in_file = args.logger_in_file

        self.device = args.device

        if args.learner_host is not None:
            config['net_config']['learner_host'] = args.learner_host
        if args.learner_port is not None:
            config['net_config']['learner_port'] = args.learner_port
        if args.env_args is not None:
            config['base_config']['env_args'] = args.env_args
        if args.unity_port is not None:
            config['base_config']['unity_args']['port'] = args.unity_port
        if args.agents is not None:
            config['base_config']['n_agents'] = args.agents

        self._stub = StubController(config['net_config']['learner_host'],
                                    config['net_config']['learner_port'])

        register_response = self._stub.register_to_learner()
        (model_abs_dir, _id,
         reset_config,
         nn_config,
         ma_nn_configs,
         sac_config,
         ma_sac_configs) = register_response

        self.model_abs_dir = model_abs_dir
        config['reset_config'] = reset_config
        config['nn_config'] = nn_config
        config['sac_config'] = sac_config
        for n in ma_configs:
            ma_configs[n]['nn_config'] = ma_nn_configs[n]
            ma_configs[n]['sac_config'] = ma_sac_configs[n]

        if self.logger_in_file:
            config_helper.set_logger(model_abs_dir.joinpath(f'actor-{_id}.log'))

        self._logger.info(f'Assigned to id {_id}')

        config_helper.display_config(config, self._logger)
        convert_config_to_enum(config['sac_config'])

        for n, ma_config in ma_configs.items():
            config_helper.display_config(ma_config, self._logger, n)
            convert_config_to_enum(ma_config['sac_config'])

        self.base_config = config['base_config']
        self.net_config = config['net_config']
        self.reset_config = config['reset_config']
        self.config = config
        self.ma_configs = ma_configs

        return config_abs_dir

    def _init_env(self):
        if self.base_config['env_type'] == 'UNITY':
            from algorithm.env_wrapper.unity_wrapper import UnityWrapper

            if self.run_in_editor:
                self.env = UnityWrapper(n_agents=self.base_config['n_agents'])
            else:
                self.env = UnityWrapper(file_name=self.base_config['unity_args']['build_path'][sys.platform],
                                        base_port=self.base_config['unity_args']['port'],
                                        no_graphics=self.base_config['unity_args']['no_graphics'],
                                        scene=self.base_config['env_name'],
                                        additional_args=self.base_config['env_args'],
                                        n_agents=self.base_config['n_agents'])

        elif self.base_config['env_type'] == 'GYM':
            from algorithm.env_wrapper.gym_wrapper import GymWrapper

            self.env = GymWrapper(env_name=self.base_config['env_name'],
                                  n_agents=self.base_config['n_agents'])

        elif self.base_config['env_type'] == 'DM_CONTROL':
            from algorithm.env_wrapper.dm_control_wrapper import \
                DMControlWrapper

            self.env = DMControlWrapper(env_name=self.base_config['env_name'],
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
                                             ma_c_action_size,
                                             self.model_abs_dir)

        for n, mgr in self.ma_manager:
            if n not in self.ma_configs:
                self._logger.warning(f'{n} not in ma_configs')
                mgr.set_config(self.config)
            else:
                mgr.set_config(self.ma_configs[n])

        self._logger.info(f'{self.base_config["env_name"]} initialized')

    def _init_sac(self, config_abs_dir):
        for n, mgr in self.ma_manager:
            # If nn models exists, load saved model, or copy a new one
            saved_nn_abs_path = mgr.model_abs_dir / 'saved_nn.py'
            if saved_nn_abs_path.exists():
                spec = importlib.util.spec_from_file_location('nn', str(saved_nn_abs_path))
                self._logger.info(f'Loaded nn from existed {saved_nn_abs_path}')
            else:
                nn_abs_path = config_abs_dir / f'{mgr.config["sac_config"]["nn"]}.py'

                spec = importlib.util.spec_from_file_location('nn', str(nn_abs_path))
                self._logger.info(f'Loaded nn in env dir: {nn_abs_path}')

            nn = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(nn)
            mgr.config['sac_config']['nn'] = nn

            mgr.set_sac(SAC_DS_Base(obs_shapes=mgr.obs_shapes,
                                    d_action_size=mgr.d_action_size,
                                    c_action_size=mgr.c_action_size,
                                    model_abs_dir=None,
                                    device=self.device,
                                    ma_name=None if len(self.ma_manager) == 1 else n,
                                    train_mode=False,

                                    nn_config=mgr.config['nn_config'],
                                    **mgr.config['sac_config']))

        self._logger.info(f'SAC_ACTOR started')

    def _init_episode_sender(self):
        max_episode_length = self.base_config['max_episode_length']

        for n, mgr in self.ma_manager:
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

            for _ in range(self.base_config['episode_sender_process_num']):
                mp.Process(target=EpisodeSender, kwargs={
                    'logger_in_file': self.logger_in_file,
                    'model_abs_dir': self.model_abs_dir,
                    'learner_host': self.net_config['learner_host'],
                    'learner_port': self.net_config['learner_port'],
                    'ma_name': n,
                    'episode_buffer': mgr['episode_buffer'],
                    'episode_length_array': mgr['episode_length_array']
                }).start()

    def _update_policy_variables(self):
        ma_variables = self._stub.get_ma_policy_variables()
        if ma_variables is None:
            return

        with self._sac_actor_lock.write():
            for n, variables in ma_variables.items():
                if not any([np.isnan(np.min(v)) for v in variables]):
                    self.ma_manager[n].sac.update_policy_variables(variables)
                    self._logger.info('Policy variables updated')
                else:
                    self._logger.warning('NAN in variables, skip updating')

    def _add_trans(self,
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

        mgr = self.ma_manager[ma_name]
        if l_indexes.shape[1] < mgr.sac.burn_in_step + mgr.sac.n_step:
            return

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
        episode_idx = mgr['episode_buffer'].put([
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
        mgr['episode_length_array'][episode_idx] = l_indexes.shape[1]

    def _run(self):
        self.ma_manager.init(self.base_config['n_agents'])

        ma_obs_list = self.env.reset(reset_config=self.reset_config)
        self.ma_manager.set_obs_list(ma_obs_list)

        force_reset = False
        iteration = 0

        try:
            while self._stub.connected:
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

                self._update_policy_variables()

                while not self.ma_manager.is_done() and self._stub.connected:
                    self.ma_manager.burn_in_padding()

                    with self._sac_actor_lock.read():
                        ma_d_action, ma_c_action = self.ma_manager.get_ma_action(force_rnd_if_available=True)

                    (ma_next_obs_list,
                     ma_reward,
                     ma_local_done,
                     ma_max_reached) = self.env.step(ma_d_action, ma_c_action)

                    if ma_next_obs_list is None:
                        force_reset = True

                        self._logger.warning('Step encounters error, episode ignored')
                        continue

                    for n, mgr in self.ma_manager:
                        if step == self.base_config['max_step_each_iter']:
                            ma_local_done[n] = [True] * len(mgr.agents)
                            ma_max_reached[n] = [True] * len(mgr.agents)

                        episode_trans_list = [
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
                            ) for i, a in enumerate(mgr.agents)
                        ]

                        episode_trans_list = [t for t in episode_trans_list if t is not None]
                        if len(episode_trans_list) != 0:
                            # ep_indexes, ep_padding_masks,
                            # ep_obses_list, ep_actions, ep_rewards, next_obs_list, ep_dones, ep_probs,
                            # ep_seq_hidden_states
                            for episode_trans in episode_trans_list:
                                self._add_trans(n, *episode_trans)  # TODO

                    self.ma_manager.post_step(ma_next_obs_list, ma_local_done)

                    step += 1

                self._log_episode_info(iteration)
                iteration += 1

        finally:
            self.close()

            self._logger.error('Actor terminated')

    def _log_episode_info(self, iteration):
        for n, mgr in self.ma_manager:
            rewards = [a.reward for a in mgr.agents]
            rewards = ", ".join([f"{i:6.1f}" for i in rewards])
            max_step = max([a.steps for a in mgr.agents])

            self._logger.info(f'{n} {iteration}, S {max_step}, R {rewards}')

    def close(self):
        if hasattr(self, 'env'):
            self.env.close()
        if hasattr(self, '_stub'):
            self._stub.close()

        self._logger.warning('Closed')


class StubController:
    _closed = False

    def __init__(self, learner_host, learner_port):
        self._logger = logging.getLogger('ds.actor.stub')

        self._learner_channel = grpc.insecure_channel(f'{learner_host}:{learner_port}', [
            ('grpc.max_reconnect_backoff_ms', MAX_RECONNECT_BACKOFF_MS)
        ])
        self._learner_stub = learner_pb2_grpc.LearnerServiceStub(self._learner_channel)
        self._logger.info(f'Starting learner stub [{learner_host}:{learner_port}]')

        self._learner_connected = False

        t_learner = threading.Thread(target=self._start_learner_persistence)
        t_learner.start()

    @property
    def connected(self):
        return self._learner_connected

    @rpc_error_inspector
    def register_to_learner(self):
        self._logger.info('Waiting for learner connection')
        while not self.connected:
            time.sleep(RECONNECTION_TIME)
            continue

        response = None
        self._logger.info('Registering to learner...')
        while response is None:
            response = self._learner_stub.RegisterActor(Empty())

            if response.model_abs_dir and response.unique_id != -1:
                self._logger.info('Registered to learner')
                return (Path(response.model_abs_dir),
                        response.unique_id,
                        json.loads(response.reset_config_json),
                        json.loads(response.nn_config_json),
                        {n: json.loads(j) for n, j in response.ma_nn_configs_json.items()},
                        json.loads(response.sac_config_json),
                        {n: json.loads(j) for n, j in response.ma_sac_configs_json.items()})
            else:
                response = None
                time.sleep(RECONNECTION_TIME)

    @rpc_error_inspector
    def get_ma_policy_variables(self):
        response = self._learner_stub.GetPolicyVariables(Empty())
        if response.succeeded:
            return proto_to_ma_variables(response)

    def _start_learner_persistence(self):
        def request_messages():
            while not self._closed:
                yield Ping(time=int(time.time() * 1000))
                time.sleep(PING_INTERVAL)
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
            finally:
                time.sleep(RECONNECTION_TIME)

    def close(self):
        self._closed = True


class StubEpisodeSenderController:
    def __init__(self, learner_host, learner_port):
        self._learner_channel = grpc.insecure_channel(f'{learner_host}:{learner_port}', [
            ('grpc.max_reconnect_backoff_ms', MAX_RECONNECT_BACKOFF_MS)
        ])
        self._learner_stub = learner_pb2_grpc.LearnerServiceStub(self._learner_channel)

    @rpc_error_inspector
    def add_transitions(self,
                        ma_name,
                        l_indexes,
                        l_padding_masks,
                        l_obses_list,
                        l_actions,
                        l_rewards,
                        next_obs_list,
                        l_dones,
                        l_mu_probs,
                        l_seq_hidden_states=None):
        self._learner_stub.Add(learner_pb2.AddRequest(ma_name=ma_name,
                                                      l_indexes=ndarray_to_proto(l_indexes),
                                                      l_padding_masks=ndarray_to_proto(l_padding_masks),
                                                      l_obses_list=[ndarray_to_proto(l_obses)
                                                                    for l_obses in l_obses_list],
                                                      l_actions=ndarray_to_proto(l_actions),
                                                      l_rewards=ndarray_to_proto(l_rewards),
                                                      next_obs_list=[ndarray_to_proto(next_obs)
                                                                     for next_obs in next_obs_list],
                                                      l_dones=ndarray_to_proto(l_dones),
                                                      l_mu_probs=ndarray_to_proto(l_mu_probs),
                                                      l_seq_hidden_states=ndarray_to_proto(l_seq_hidden_states)))

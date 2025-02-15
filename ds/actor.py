import importlib
import json
import logging
import logging.handlers
import multiprocessing as mp
import os
import threading
import time
from multiprocessing.sharedctypes import SynchronizedArray
from pathlib import Path

import grpc
import numpy as np

import algorithm.config_helper as config_helper
from algorithm.agent import Agent, AgentManager
from algorithm.sac_main import Main
from algorithm.utils import ElapsedTimer, ReadWriteLock, gen_pre_n_actions
from algorithm.utils.elapse_timer import UnifiedElapsedTimer
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


class AgentManagerBuffer:
    def __init__(self,
                 episode_buffer: SharedMemoryManager,
                 episode_length_array: SynchronizedArray,
                 episode_sender_processes: list[mp.Process]):
        self.episode_buffer = episode_buffer
        self.episode_length_array = episode_length_array
        self.episode_sender_processes = episode_sender_processes


class EpisodeSender:
    def __init__(self,
                 actor_id: int,
                 _id: int,
                 episode_buffer: SharedMemoryManager,
                 episode_length_array: SynchronizedArray,

                 logger_in_file: bool,
                 debug: bool,
                 ma_name: str,

                 model_abs_dir: Path,
                 learner_host: str,
                 learner_port: int):
        self._episode_buffer = episode_buffer
        self._episode_length_array = episode_length_array
        self.ma_name = ma_name

        # Since no set_logger() in main.py
        config_helper.set_logger(debug)

        if logger_in_file:
            config_helper.add_file_logger(model_abs_dir.joinpath(f'actor_{actor_id}.log'))

        if ma_name is None:
            self._logger = logging.getLogger(f'ds.actor-{actor_id}.episode_sender-{_id}')
        else:
            self._logger = logging.getLogger(f'ds.actor-{actor_id}.episode_sender-{_id}.{ma_name}')

        episode_buffer.init_logger(self._logger)

        self._stub = StubEpisodeSenderController(actor_id, learner_host, learner_port)

        self._logger.info(f'EpisodeSender {_id} ({os.getpid()}) initialized')

        self._run()

    def _run(self):
        timer_add_trans = ElapsedTimer('Add trans', self._logger, logging.INFO,
                                       repeat=ELAPSED_REPEAT,
                                       force_report=False)

        while True:
            episode, episode_idx = self._episode_buffer.get(timeout=EPISODE_QUEUE_TIMEOUT)
            if episode is None:
                continue

            episode_length = self._episode_length_array[episode_idx]

            episode = traverse_lists(episode, lambda e: e[:, :episode_length])

            with timer_add_trans:
                self._stub.add_episode(self.ma_name, *episode)


class Actor(Main):
    train_mode = True

    _id = -1
    _stub = None

    def __init__(self, root_dir, config_dir, args):
        self.root_dir = root_dir

        self._logger = logging.getLogger('ds.actor')

        self._profiler = UnifiedElapsedTimer(self._logger)

        self._config_abs_dir = self._init_config(root_dir, config_dir, args)

        self._sac_actor_lock = ReadWriteLock(5, 1, 1, logger=self._logger)
        self._init_env()
        self._init_sac()
        self._init_episode_sender()

        self._run()

    def _init_config(self, root_dir, config_dir, args):
        config_abs_dir = Path(root_dir).joinpath(config_dir)
        config_abs_path = config_abs_dir.joinpath('config_ds.yaml')
        default_config_abs_path = Path(__file__).resolve().parent.joinpath('default_config.yaml')
        # Merge default_config.yaml and custom config.yaml
        config, ma_configs = config_helper.initialize_config_from_yaml(default_config_abs_path,
                                                                       config_abs_path,
                                                                       args.config)

        # Initialize config from command line arguments
        self.debug = args.debug
        self.logger_in_file = args.logger_in_file
        self.inference_ma_names = set()

        self.render = args.render
        self.unity_run_in_editor = args.u_editor
        self.unity_time_scale = args.u_timescale

        self.device = args.device

        if args.learner_host is not None:
            config['net_config']['learner_host'] = args.learner_host
        if args.learner_port is not None:
            config['net_config']['learner_port'] = args.learner_port

        for env_arg in args.env_args:
            k, v = env_arg.split('=')
            if k in config['base_config']['env_args']:
                config['base_config']['env_args'][k] = config_helper.convert_config_value_by_src(v, config['base_config']['env_args'][k])
            else:
                config['base_config']['env_args'][k] = config_helper.convert_config_value(v)
        if args.u_port is not None:
            config['base_config']['unity_args']['port'] = args.u_port
        if args.envs is not None:
            config['base_config']['n_envs'] = args.envs

        self._stub = StubController(config['net_config']['learner_host'],
                                    config['net_config']['learner_port'])

        register_response = self._stub.register_to_learner()
        (model_abs_dir, _id,
         reset_config,
         nn_config,
         ma_nn_configs,
         sac_config,
         ma_sac_configs) = register_response

        self._id = _id

        self.model_abs_dir = model_abs_dir
        config['reset_config'] = reset_config
        config['nn_config'] = nn_config
        config['sac_config'] = sac_config
        for n in ma_configs:
            ma_configs[n]['nn_config'] = ma_nn_configs[n]
            ma_configs[n]['sac_config'] = ma_sac_configs[n]

        self._logger.name = self._logger.name + f'-{_id}'
        self._logger.info(f'Assigned to id {_id}')
        self._stub.update_actor_id(_id)

        if self.logger_in_file:
            config_helper.add_file_logger(model_abs_dir.joinpath(f'actor_{_id}.log'))

        config_helper.display_config(config, self._logger)
        convert_config_to_enum(config['sac_config'])
        convert_config_to_enum(config['oc_config'])

        for n, ma_config in ma_configs.items():
            config_helper.display_config(ma_config, self._logger, n)
            convert_config_to_enum(ma_config['sac_config'])
            convert_config_to_enum(ma_config['oc_config'])

        self.base_config = config['base_config']
        self.net_config = config['net_config']
        self.reset_config = config['reset_config']
        self.config = config
        self.ma_configs = ma_configs

        return config_abs_dir

    def _init_sac(self):
        self._ma_agent_manager_buffer: dict[str, AgentManagerBuffer] = {}

        for n, mgr in self.ma_manager:
            # If nn models exists, load saved model, or copy a new one
            saved_nn_abs_dir = mgr.model_abs_dir / 'nn'
            if saved_nn_abs_dir.exists():
                nn_abs_path = saved_nn_abs_dir / f'{mgr.config["sac_config"]["nn"]}.py'
                spec = importlib.util.spec_from_file_location(f'{self._get_relative_package(nn_abs_path)}.{mgr.config["sac_config"]["nn"]}',
                                                              nn_abs_path)
                self._logger.info(f'Loaded nn from existed {nn_abs_path}')
            else:
                nn_abs_path = self._config_abs_dir / f'{mgr.config["sac_config"]["nn"]}.py'

                spec = importlib.util.spec_from_file_location(f'{self._get_relative_package(nn_abs_path)}.{mgr.config["sac_config"]["nn"]}',
                                                              nn_abs_path)
                self._logger.info(f'Loaded nn in env dir: {nn_abs_path}')

            nn = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(nn)
            mgr.config['sac_config']['nn'] = nn

            mgr.set_rl(SAC_DS_Base(obs_names=mgr.obs_names,
                                   obs_shapes=mgr.obs_shapes,
                                   d_action_sizes=mgr.d_action_sizes,
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
                mgr.rl.seq_hidden_state_shape)

            ep_shmm = SharedMemoryManager(self.base_config['episode_queue_size'],
                                          logger=logging.getLogger(f'ds.actor.ep_shmm.{n}'),
                                          logger_level=logging.INFO,
                                          counter_get_shm_index_empty_log='Get an episode but is empty',
                                          timer_get_shm_index_log='Get an episode waited',
                                          log_repeat=ELAPSED_REPEAT,
                                          force_report=ELAPSED_FORCE_REPORT)
            ep_shmm.init_from_shapes(episode_shapes, episode_dtypes)
            ep_length_array = mp.Array('i', range(self.base_config['episode_queue_size']))

            episode_sender_processes = []
            for i in range(self.base_config['episode_sender_process_num']):
                p = mp.Process(target=EpisodeSender, kwargs={
                    'actor_id': self._id,
                    '_id': i,
                    'episode_buffer': ep_shmm,
                    'episode_length_array': ep_length_array,

                    'logger_in_file': self.logger_in_file,
                    'debug': self.debug,
                    'ma_name': n,

                    'model_abs_dir': self.model_abs_dir,
                    'learner_host': self.net_config['learner_host'],
                    'learner_port': self.net_config['learner_port'],
                })
                p.start()
                episode_sender_processes.append(p)

            self._ma_agent_manager_buffer[n] = AgentManagerBuffer(ep_shmm,
                                                                  ep_length_array,
                                                                  episode_sender_processes)

    def _update_policy_variables(self):
        ma_variables = self._stub.get_ma_policy_variables()

        with self._sac_actor_lock.write():
            for n, variables in ma_variables.items():
                if variables is not None:
                    if any([np.isnan(np.min(v)) for v in variables]):
                        self._logger.warning(f'NAN in {n} variables, skip updating')
                    else:
                        self.ma_manager[n].rl.update_policy_variables(variables)
                        self._logger.info(f'{n} policy variables updated')

    def _add_trans(self,
                   ma_name: str,

                   ep_indexes: np.ndarray,
                   ep_obses_list: list[np.ndarray],
                   ep_actions: np.ndarray,
                   ep_rewards: np.ndarray,
                   ep_dones: np.ndarray,
                   ep_probs: list[np.ndarray],
                   ep_pre_seq_hidden_states: np.ndarray = None):

        mgr = self.ma_manager[ma_name]
        mgr_buffer = self._ma_agent_manager_buffer[ma_name]
        if ep_indexes.shape[1] < mgr.rl.n_step:
            return

        """
        Args:
            ma_name: str

            ep_indexes: [1, episode_len]
            ep_obses_list: list([1, episode_len, *obs_shapes_i], ...)
            ep_actions: [1, episode_len, action_size]
            ep_rewards: [1, episode_len]
            ep_dones: [1, episode_len]
            ep_probs: [1, episode_len]
            ep_pre_seq_hidden_states: [1, episode_len, *seq_hidden_state_shape]
        """
        episode_idx = mgr_buffer.episode_buffer.put([
            ep_indexes,
            ep_obses_list,
            ep_actions,
            ep_rewards,
            ep_dones,
            ep_probs,
            ep_pre_seq_hidden_states
        ])
        mgr_buffer.episode_length_array[episode_idx] = ep_indexes.shape[1]

    def _run(self):
        force_reset = False
        iteration = 0

        try:
            while self._stub.connected:
                step = 0
                iter_time = time.time()

                self._update_policy_variables()

                if iteration == 0 \
                        or self.base_config['reset_on_iteration'] \
                        or self.ma_manager.max_reached \
                        or force_reset:
                    self.ma_manager.reset()
                    ma_agent_ids, ma_obs_list = self.env.reset(reset_config=self.reset_config)
                    ma_d_action, ma_c_action = self.ma_manager.get_ma_action(
                        ma_agent_ids=ma_agent_ids,
                        ma_obs_list=ma_obs_list,
                        ma_last_reward={n: np.zeros(len(agent_ids), dtype=bool)
                                        for n, agent_ids in ma_agent_ids.items()},
                        force_rnd_if_available=True
                    )

                    force_reset = False
                else:
                    self.ma_manager.reset_and_continue()

                while not self.ma_manager.done and self._stub.connected:
                    with self._profiler('env.step', repeat=10):
                        (decision_step,
                         terminal_step,
                         all_envs_done) = self.env.step(ma_d_action, ma_c_action)

                    if decision_step is None:
                        force_reset = True

                        self._logger.error('Step encounters error, episode ignored')
                        break

                    with self._sac_actor_lock.read():
                        with self._profiler('get_ma_action', repeat=10):
                            ma_d_action, ma_c_action = self.ma_manager.get_ma_action(
                                ma_agent_ids=decision_step.ma_agent_ids,
                                ma_obs_list=decision_step.ma_obs_list,
                                ma_last_reward=decision_step.ma_last_reward,
                                force_rnd_if_available=True
                            )

                    self.ma_manager.end_episode(
                        ma_agent_ids=terminal_step.ma_agent_ids,
                        ma_obs_list=terminal_step.ma_obs_list,
                        ma_last_reward=terminal_step.ma_last_reward,
                        ma_max_reached=terminal_step.ma_max_reached
                    )
                    if all_envs_done or step == self.base_config['max_step_each_iter']:
                        self.ma_manager.end_episode(
                            ma_agent_ids=decision_step.ma_agent_ids,
                            ma_obs_list=decision_step.ma_obs_list,
                            ma_last_reward=decision_step.ma_last_reward,
                            ma_max_reached={n: np.ones_like(agent_ids, dtype=bool)
                                            for n, agent_ids in decision_step.ma_agent_ids.items()}
                        )
                        self.ma_manager.force_end_all_episode()

                    for n, mgr in self.ma_manager:
                        episode_trans_list = mgr.get_tmp_episode_trans_list()

                        # ep_indexes,
                        # ep_obses_list, ep_actions, ep_rewards, ep_dones, ep_probs,
                        # ep_pre_seq_hidden_states
                        for episode_trans in episode_trans_list:
                            self._add_trans(n, **episode_trans)

                        mgr.clear_tmp_episode_trans_list()

                    step += 1

                self._log_episode_info(iteration, time.time() - iter_time)

                self.ma_manager.reset_dead_agents()
                self.ma_manager.clear_tmp_episode_trans_list()

                iteration += 1

        finally:
            self.close()

            self._logger.warning('Actor terminated')

    def _log_episode_info(self, iteration, iter_time):
        for n, mgr in self.ma_manager:
            if len(mgr.non_empty_agents) == 0:
                continue

            rewards = [a.reward for a in mgr.non_empty_agents]
            rewards = ", ".join([f"{i:6.1f}" for i in rewards])
            max_step = max([a.steps for a in mgr.non_empty_agents])
            self._logger.info(f'{n} {iteration}, T {iter_time:.2f}s, S {max_step}, R {rewards}')

    def close(self):
        if hasattr(self, 'env'):
            self.env.close()
        if self._stub is not None:
            self._stub.close()

        for n, mgr_buffer in self._ma_agent_manager_buffer.items():
            for p in mgr_buffer.episode_sender_processes:
                p.terminate()

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

        t_learner = threading.Thread(target=self._start_learner_persistence, daemon=True)
        t_learner.start()

    @property
    def connected(self):
        return self._learner_connected

    @property
    def closed(self):
        return self._closed

    def update_actor_id(self, actor_id: int):
        self._logger.name = f'ds.actor-{actor_id}.stub'

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
                self._logger.info('Connecting to learner...')
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
    def __init__(self,
                 actor_id: int,
                 learner_host,
                 learner_port):
        self._learner_channel = grpc.insecure_channel(f'{learner_host}:{learner_port}', [
            ('grpc.max_reconnect_backoff_ms', MAX_RECONNECT_BACKOFF_MS)
        ])
        self._learner_stub = learner_pb2_grpc.LearnerServiceStub(self._learner_channel)

        self._logger = logging.getLogger(f'ds.actor-{actor_id}.stub_episode_sender')

    @rpc_error_inspector
    def add_episode(self,
                    ma_name,
                    ep_indexes,
                    ep_obses_list,
                    ep_actions,
                    ep_rewards,
                    ep_dones,
                    ep_mu_probs,
                    ep_pre_seq_hidden_states):
        self._learner_stub.Add(learner_pb2.AddRequest(ma_name=ma_name,
                                                      ep_indexes=ndarray_to_proto(ep_indexes),
                                                      ep_obses_list=[ndarray_to_proto(ep_obses)
                                                                     for ep_obses in ep_obses_list],
                                                      ep_actions=ndarray_to_proto(ep_actions),
                                                      ep_rewards=ndarray_to_proto(ep_rewards),
                                                      ep_dones=ndarray_to_proto(ep_dones),
                                                      ep_mu_probs=ndarray_to_proto(ep_mu_probs),
                                                      ep_pre_seq_hidden_states=ndarray_to_proto(ep_pre_seq_hidden_states)))

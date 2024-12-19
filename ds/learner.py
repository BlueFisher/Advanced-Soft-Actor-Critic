from collections import defaultdict
import importlib
import json
import logging
import multiprocessing as mp
import shutil
import socket
import threading
import time
from concurrent import futures
from copy import deepcopy
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Dict, List, final

import grpc
import numpy as np

import algorithm.config_helper as config_helper
from algorithm.sac_main import Main
from algorithm.utils import ReadWriteLock, RLock
from algorithm.utils.enums import *

from .constants import *
from .learner_trainer import Trainer
from .proto import learner_pb2, learner_pb2_grpc
from .proto.ma_variables_proto import ma_variables_to_proto
from .proto.ndarray_pb2 import Empty
from .proto.numproto import ndarray_to_proto, proto_to_ndarray
from .proto.pingpong_pb2 import Ping, Pong
from .sac_ds_base import SAC_DS_Base
from .utils import PeerSet, SharedMemoryManager, get_episode_shapes_dtypes


class AgentManagerBuffer:
    def __init__(self,
                 all_variables_buffer: SharedMemoryManager,
                 episode_buffer: SharedMemoryManager,
                 episode_length_array: mp.Array,
                 cmd_pipe_client: Connection,
                 learner_trainer_process: mp.Process):
        self.all_variables_buffer = all_variables_buffer
        self.episode_buffer = episode_buffer
        self.episode_length_array = episode_length_array
        self.cmd_pipe_client = cmd_pipe_client
        self.learner_trainer_process = learner_trainer_process


class Learner(Main):
    train_mode = False
    _initialized = False
    _closed = False
    _ma_policy_variables_cache = {}

    _servicer = None
    _server = None

    def __init__(self, root_dir, config_dir, args):
        self._logger = logging.getLogger('ds.learner')

        config_abs_dir = self._init_config(root_dir, config_dir, args)
        learner_host = self.config['net_config']['learner_host']
        learner_port = self.config['net_config']['learner_port']
        self._initialized = True

        self._sac_learner_eval_lock = ReadWriteLock(None, 2, 2, logger=self._logger)

        self._init_env()
        self._init_sac(config_abs_dir)

        threading.Thread(target=self._policy_evaluation, daemon=True).start()

        try:
            self._run_learner_server(learner_port)
        except KeyboardInterrupt:
            self._logger.warning('KeyboardInterrupt')
        finally:
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
        self.debug = args.debug
        self.logger_in_file = args.logger_in_file
        self.inference_ma_names = set()

        self.render = args.render
        self.unity_run_in_editor = args.u_editor
        self.unity_time_scale = args.u_timescale

        self.force_env_nn = args.use_env_nn
        self.device = args.device
        self.last_ckpt = args.ckpt

        if args.learner_host is not None:
            config['net_config']['learner_host'] = args.learner_host
        if args.learner_port is not None:
            config['net_config']['learner_port'] = args.learner_port

        if len(args.env_args) > 0:
            config['base_config']['env_args'] = args.env_args
        if args.u_port is not None:
            config['base_config']['unity_args']['port'] = args.u_port
        if args.envs is not None:
            config['base_config']['n_envs'] = args.envs
        if args.name is not None:
            config['base_config']['name'] = args.name
        if args.nn is not None:
            config['sac_config']['nn'] = args.nn
            for ma_config in ma_configs.values():
                ma_config['sac_config']['nn'] = args.nn

        config['base_config']['name'] = config_helper.generate_base_name(config['base_config']['name'])

        model_abs_dir = Path(root_dir).joinpath('models',
                                                config['base_config']['env_name'],
                                                config['base_config']['name'])
        model_abs_dir.mkdir(parents=True, exist_ok=True)
        self.model_abs_dir = model_abs_dir

        if self.logger_in_file:
            config_helper.add_file_logger(model_abs_dir.joinpath(f'learner.log'))

        config_helper.save_config(config, model_abs_dir, 'config.yaml')
        config_helper.display_config(config, self._logger)
        convert_config_to_enum(config['sac_config'])
        convert_config_to_enum(config['oc_config'])

        for n, ma_config in ma_configs.items():
            config_helper.save_config(ma_config, model_abs_dir, f'config_{n.replace("?", "-")}.yaml')
            config_helper.display_config(ma_config, self._logger, n)
            convert_config_to_enum(ma_config['sac_config'])
            convert_config_to_enum(ma_config['oc_config'])

        self.base_config = config['base_config']
        self.reset_config = config['reset_config']
        self.config = config
        self.ma_configs = ma_configs

        return config_abs_dir

    def _init_sac(self, config_abs_dir: Path):
        self._ma_agent_manager_buffer: Dict[str, AgentManagerBuffer] = {}

        for n, mgr in self.ma_manager:
            # If nn models exists, load saved model, or copy a new one
            saved_nn_abs_path = mgr.model_abs_dir / 'saved_nn.py'
            if not self.force_env_nn and saved_nn_abs_path.exists():
                spec = importlib.util.spec_from_file_location('nn', str(saved_nn_abs_path))
                self._logger.info(f'Loaded nn from existed {saved_nn_abs_path}')
            else:
                nn_abs_path = config_abs_dir / f'{mgr.config["sac_config"]["nn"]}.py'

                spec = importlib.util.spec_from_file_location('nn', str(nn_abs_path))
                self._logger.info(f'Loaded nn in env dir: {nn_abs_path}')
                if not self.force_env_nn:
                    shutil.copyfile(nn_abs_path, saved_nn_abs_path)

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
                                   last_ckpt=self.last_ckpt,

                                   nn_config=mgr.config['nn_config'],
                                   **mgr.config['sac_config']))

            self._logger.info(f'{n} SAC_BAK started')

            # Initialize all queues for learner_trainer
            all_variables_shmm = SharedMemoryManager(1)
            all_variables_shmm.init_from_data_buffer(mgr.rl.get_all_variables())

            max_episode_length = self.base_config['max_episode_length']

            episode_shapes, episode_dtypes = get_episode_shapes_dtypes(
                max_episode_length,
                mgr.obs_shapes,
                mgr.action_size,
                mgr.rl.seq_hidden_state_shape if mgr.seq_encoder is not None else None)

            ep_shmm = SharedMemoryManager(self.base_config['episode_queue_size'],
                                          logger=logging.getLogger(f'ds.learner.ep_shmm.{n}'),
                                          logger_level=logging.INFO,
                                          counter_get_shm_index_empty_log='Get an episode but is empty',
                                          timer_get_shm_index_log='Get an episode waited',
                                          log_repeat=ELAPSED_REPEAT,
                                          force_report=ELAPSED_FORCE_REPORT)
            ep_shmm.init_from_shapes(episode_shapes, episode_dtypes)
            ep_length_array = mp.Array('i', range(self.base_config['episode_queue_size']))

            cmd_pipe_client, cmd_pipe_server = mp.Pipe()

            tmp_nn = mgr.config['sac_config']['nn']
            mgr.config['sac_config']['nn'] = spec  # Cannot pickle module object
            learner_trainer_process = mp.Process(target=Trainer, kwargs={
                'all_variables_buffer': all_variables_shmm,
                'episode_buffer': ep_shmm,
                'episode_length_array': ep_length_array,
                'cmd_pipe_server': cmd_pipe_server,

                'logger_in_file': self.logger_in_file,
                'debug': self.debug,
                'ma_name': None if len(self.ma_manager) == 1 else n,

                'obs_names': mgr.obs_names,
                'obs_shapes': mgr.obs_shapes,
                'd_action_sizes': mgr.d_action_sizes,
                'c_action_size': mgr.c_action_size,
                'model_abs_dir': mgr.model_abs_dir,
                'device': self.device,
                'last_ckpt': self.last_ckpt,

                'config': mgr.config
            })
            learner_trainer_process.start()
            mgr.config['sac_config']['nn'] = tmp_nn

            self._ma_agent_manager_buffer[n] = AgentManagerBuffer(all_variables_shmm,
                                                                  ep_shmm,
                                                                  ep_length_array,
                                                                  cmd_pipe_client,
                                                                  learner_trainer_process)

            self._logger.info(f'Waiting for updating {n} SAC_BAK for the first time')
            self._update_sac_bak(n)
            self._logger.info(f'Updated {n} SAC_BAK, start forever updating')

            threading.Thread(target=self._forever_update_sac_bak,
                             args=(n,),
                             daemon=True).start()

    def _update_sac_bak(self, ma_name: str):
        mgr = self.ma_manager[ma_name]
        all_variables, _ = self._ma_agent_manager_buffer[ma_name].all_variables_buffer.get()  # Block, waiting for all variables available

        with self._sac_learner_eval_lock.write():
            res = mgr.rl.update_all_variables(all_variables)

        if not res:
            self._logger.error(f'NAN in {ma_name} SAC_BAK variables, closing...')
            with open(self.model_abs_dir.joinpath('force_closed'), 'w') as f:
                f.write(f'NAN in {ma_name} SAC_BAK variables')
            self.close()
            return

        self._ma_policy_variables_cache[ma_name] = mgr.rl.get_policy_variables()

    def _forever_update_sac_bak(self, ma_name: str):
        i = 1
        while not self._closed:
            self._logger.info(f'Updating {ma_name} SAC_BAK ({i})...')
            self._update_sac_bak(ma_name)
            self._logger.info(f'Updated {ma_name} SAC_BAK ({i})')
            i += 1

    def _get_actor_register_result(self, actor_id):
        if self._initialized:
            actor_nn_config = deepcopy(self.config['nn_config'])
            ma_actor_nn_configs = {n: deepcopy(c['nn_config']) for n, c in self.ma_configs.items()}
            actor_sac_config = deepcopy(self.config['sac_config'])
            ma_actor_sac_configs = {n: deepcopy(c['sac_config']) for n, c in self.ma_configs.items()}

            noise = self.base_config['noise_increasing_rate'] * actor_id
            noise = min(noise, self.base_config['noise_max'])
            actor_sac_config['action_noise'] = [noise, noise]
            convert_config_to_string(actor_sac_config)
            for n, c in ma_actor_sac_configs.items():
                c['action_noise'] = [noise, noise]
                convert_config_to_string(c)

            return (self.model_abs_dir,
                    self.reset_config,
                    actor_nn_config,
                    ma_actor_nn_configs,
                    actor_sac_config,
                    ma_actor_sac_configs)

    def _get_ma_policy_variables(self):
        return self._ma_policy_variables_cache

    def _save_model(self):
        for n, mgr_buffer in self._ma_agent_manager_buffer.items():
            mgr_buffer.cmd_pipe_client.send(('SAVE_MODEL', None))

    def _add_episode(self,
                     ma_name: str,
                     ep_indexes: np.ndarray,
                     ep_obses_list: List[np.ndarray],
                     ep_actions: np.ndarray,
                     ep_rewards: np.ndarray,
                     ep_dones: np.ndarray,
                     ep_probs: List[np.ndarray],
                     ep_seq_hidden_states: np.ndarray = None):
        """
        Args:
            ma_name: str
            ep_indexes: [1, episode_len]
            ep_obses_list: list([1, episode_len, *obs_shapes_i], ...)
            ep_actions: [1, episode_len, action_size]
            ep_rewards: [1, episode_len]
            ep_dones: [1, episode_len]
            ep_probs: [1, episode_len]
            ep_seq_hidden_states: [1, episode_len, *seq_hidden_state_shape]
        """
        if ep_indexes.shape[1] > self.base_config['max_episode_length']:
            self._logger.error(f'Episode length {ep_indexes.shape[1]} > max episode length {self.base_config["max_episode_length"]}')
            return

        episode_idx = self._ma_agent_manager_buffer[ma_name].episode_buffer.put([
            ep_indexes,
            ep_obses_list,
            ep_actions,
            ep_rewards,
            ep_dones,
            ep_probs,
            ep_seq_hidden_states
        ])
        self._ma_agent_manager_buffer[ma_name].episode_length_array[episode_idx] = ep_indexes.shape[1]

    def _policy_evaluation(self):
        force_reset = False
        iteration = 0

        try:
            while not self._closed:
                step = 0
                iter_time = time.time()

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

                while not self.ma_manager.done and not self._closed:
                    (decision_step,
                     terminal_step,
                     all_envs_done) = self.env.step(ma_d_action, ma_c_action)

                    if decision_step is None:
                        force_reset = True

                        self._logger.error('Step encounters error, episode ignored')
                        break

                    with self._sac_learner_eval_lock.read():
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

                    step += 1

                self._log_episode_summaries()
                self._log_episode_info(iteration, time.time() - iter_time)

                self.ma_manager.reset_dead_agents()
                self.ma_manager.clear_tmp_episode_trans_list()

                p_model = self.model_abs_dir.joinpath('save_model')
                if p_model.exists():
                    self._save_model()
                    p_model.unlink()

                iteration += 1

                while self.connected_actor_count == 0:
                    time.sleep(1)

        finally:
            try:
                self._save_model()
            except:
                pass
            self.env.close()

            self._logger.warning('Evaluation terminated')

    def _log_episode_summaries(self):
        for n, mgr in self.ma_manager:
            if len(mgr.non_empty_agents) == 0:
                continue
            rewards = np.array([a.reward for a in mgr.non_empty_agents])
            steps = np.array([a.steps for a in mgr.non_empty_agents])

            try:
                self._ma_agent_manager_buffer[n].cmd_pipe_client.send(('LOG_EPISODE_SUMMARIES', [
                    {'tag': 'reward/mean', 'simple_value': float(rewards.mean())},
                    {'tag': 'reward/max', 'simple_value': float(rewards.max())},
                    {'tag': 'reward/min', 'simple_value': float(rewards.min())},
                    {'tag': 'metric/steps', 'simple_value': steps.mean()}
                ]))
            except Exception as e:
                self._logger.error(e)

    def _log_episode_info(self, iteration, iter_time):
        for n, mgr in self.ma_manager:
            if len(mgr.non_empty_agents) == 0:
                continue
            rewards = [a.reward for a in mgr.non_empty_agents]
            rewards = ", ".join([f"{i:6.1f}" for i in rewards])
            max_step = max([a.steps for a in mgr.non_empty_agents])
            self._logger.info(f'{n} {iteration}, {iter_time:.2f}s, S {max_step}, R {rewards}')

    def _run_learner_server(self, learner_port):
        self._servicer = LearnerService(self)
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_THREAD_WORKERS),
                                   options=[
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)
        ])
        learner_pb2_grpc.add_LearnerServiceServicer_to_server(self._servicer, self._server)
        self._server.add_insecure_port(f'[::]:{learner_port}')
        self._server.start()
        self._logger.info(f'Learner server is running on [{learner_port}]...')

        self._server.wait_for_termination()

    @property
    def connected_actor_count(self):
        return self._servicer.get_connected_length()

    def close(self):
        if self._closed:
            return

        self._closed = True

        if hasattr(self, 'env'):
            self.env.close()
        if self._server is not None:
            self._server.stop(None)

        for n, mgr_buffer in self._ma_agent_manager_buffer.items():
            mgr_buffer.cmd_pipe_client.close()
            mgr_buffer.learner_trainer_process.terminate()

        self._logger.warning('Closed')


class LearnerService(learner_pb2_grpc.LearnerServiceServicer):
    def __init__(self, learner: Learner):
        self._get_actor_register_result = learner._get_actor_register_result

        self._add_episode = learner._add_episode
        self._get_ma_policy_variables = learner._get_ma_policy_variables

        self._logger = logging.getLogger('ds.learner.service')
        self._peer_set = PeerSet(self._logger)

        self._lock = RLock(timeout=1, logger=self._logger)
        self._actor_id = 0

    def get_connected_length(self):
        return len(self._peer_set)

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

                self._logger.info(f'Actor {peer} registered')

                self._actor_id += 1

                (model_abs_dir,
                 reset_config,
                 nn_config,
                 ma_nn_configs,
                 sac_config,
                 ma_sac_configs) = res

                self._peer_set.add_info(peer, {
                    'ma_policy_variables_id_cache': defaultdict(lambda: None)  # ma_name: id
                })

                return learner_pb2.RegisterActorResponse(model_abs_dir=str(model_abs_dir),
                                                         unique_id=actor_id,
                                                         reset_config_json=json.dumps(reset_config),
                                                         nn_config_json=json.dumps(nn_config),
                                                         ma_nn_configs_json={n: json.dumps(c) for n, c in ma_nn_configs.items()},
                                                         sac_config_json=json.dumps(sac_config),
                                                         ma_sac_configs_json={n: json.dumps(c) for n, c in ma_sac_configs.items()})
            else:
                return learner_pb2.RegisterActorResponse(unique_id=-1)

    # From actor
    def GetPolicyVariables(self, request, context):
        peer = context.peer()

        ma_policy_variables = self._get_ma_policy_variables()
        if len(ma_policy_variables) == 0:
            return ma_variables_to_proto(None)

        actor_info = self._peer_set.get_info(peer)

        _ma_policy_variables = {}
        for n, vs in ma_policy_variables.items():
            if id(vs) == actor_info['ma_policy_variables_id_cache'][n]:
                _ma_policy_variables[n] = None
            else:
                actor_info['ma_policy_variables_id_cache'][n] = id(vs)
                _ma_policy_variables[n] = vs

        return ma_variables_to_proto(_ma_policy_variables)

    # From actor
    def Add(self, request, context):
        self._add_episode(request.ma_name,
                          proto_to_ndarray(request.ep_indexes),
                          [proto_to_ndarray(ep_obses) for ep_obses in request.ep_obses_list],
                          proto_to_ndarray(request.ep_actions),
                          proto_to_ndarray(request.ep_rewards),
                          proto_to_ndarray(request.ep_dones),
                          proto_to_ndarray(request.ep_mu_probs),
                          proto_to_ndarray(request.ep_seq_hidden_states))

        return Empty()

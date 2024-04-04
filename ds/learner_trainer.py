import importlib
import logging
import math
import multiprocessing as mp
import os
import threading
import traceback
from multiprocessing.connection import Connection
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

import algorithm.config_helper as config_helper
from algorithm.utils import RLock, episode_to_batch, traverse_lists
from algorithm.utils.elapse_timer import ElapsedTimer

from .constants import *
from .sac_ds_base import SAC_DS_Base
from .utils import SharedMemoryManager, get_batch_shapes_dtype, traverse_lists


class BatchGenerator:
    def __init__(self,
                 _id: int,

                 episode_buffer: SharedMemoryManager,
                 episode_length_array: mp.Array,
                 batch_buffer: SharedMemoryManager,

                 logger_in_file: bool,
                 debug: bool,
                 ma_name: Optional[str],

                 model_abs_dir: Path,
                 burn_in_step: int,
                 n_step: int,
                 batch_size: int,

                 padding_action: np.ndarray):
        self._episode_buffer = episode_buffer
        self._episode_length_array = episode_length_array
        self._batch_buffer = batch_buffer

        self.burn_in_step = burn_in_step
        self.n_step = n_step
        self.batch_size = batch_size

        self.padding_action = padding_action

        # Since no set_logger() in main.py
        config_helper.set_logger(debug)

        if logger_in_file:
            config_helper.add_file_logger(model_abs_dir.joinpath(f'learner.log'))

        if ma_name is None:
            self._logger = logging.getLogger(f'ds.learner.trainer.batch_generator_{_id}')
        else:
            self._logger = logging.getLogger(f'ds.learner.trainer.{ma_name}.batch_generator_{_id}')

        self._logger.info(f'BatchGenerator({ma_name}) {_id} ({os.getpid()}) initialized')

        episode_buffer.init_logger(self._logger)
        batch_buffer.init_logger(self._logger)

        try:
            self.run()
        except KeyboardInterrupt:
            self._logger.warning('KeyboardInterrupt')

    def run(self):
        _rest_batch = None

        while True:
            episode, episode_idx = self._episode_buffer.get(timeout=EPISODE_QUEUE_TIMEOUT)
            if episode is None:
                continue

            episode_length = self._episode_length_array[episode_idx]

            # Fix episode length
            episode = traverse_lists(list(episode), lambda e: e[:, :episode_length])

            (ep_indexes,
             ep_obses_list,
             ep_actions,
             ep_rewards,
             ep_dones,
             ep_mu_probs,
             ep_seq_hidden_states) = episode

            ep_padding_masks = np.zeros_like(ep_indexes, dtype=bool)
            ep_padding_masks[:, -1] = True  # The last step is next_step
            ep_padding_masks[ep_indexes == -1] = True

            """
            bn_indexes: [episode_len - bn + 1, bn]
            bn_padding_masks: [episode_len - bn + 1, bn]
            bn_obses_list: list([episode_len - bn + 1, bn, *obs_shapes_i], ...)
            bn_actions: [episode_len - bn + 1, bn, action_size]
            bn_rewards: [episode_len - bn + 1, bn]
            next_obs_list: list([episode_len - bn + 1, *obs_shapes_i], ...)
            bn_dones: [episode_len - bn + 1, bn]
            bn_mu_probs: [episode_len - bn + 1, bn]
            bn_seq_hidden_states: [episode_len - bn + 1, bn, *seq_hidden_state_shape]
            """
            ori_batch = episode_to_batch(self.burn_in_step,
                                         self.n_step,
                                         self.padding_action,
                                         l_indexes=ep_indexes,
                                         l_padding_masks=ep_padding_masks,
                                         l_obses_list=ep_obses_list,
                                         l_actions=ep_actions,
                                         l_rewards=ep_rewards,
                                         l_dones=ep_dones,
                                         l_probs=ep_mu_probs,
                                         l_seq_hidden_states=ep_seq_hidden_states)

            if _rest_batch is not None:
                ori_batch = traverse_lists((_rest_batch, ori_batch), lambda rb, b: np.concatenate([rb, b]))
                _rest_batch = None

            ori_batch_size = ori_batch[0].shape[0]
            idx = np.random.permutation(ori_batch_size)
            ori_batch = traverse_lists(ori_batch, lambda b: b[idx])

            for i in range(math.ceil(ori_batch_size / self.batch_size)):
                b_i, b_j = i * self.batch_size, (i + 1) * self.batch_size

                batch = traverse_lists(ori_batch, lambda b: b[b_i:b_j, :])

                if b_j > ori_batch_size:
                    _rest_batch = batch
                else:
                    self._batch_buffer.put(batch)


class Trainer:
    _closed = False

    def __init__(self,
                 all_variables_buffer: SharedMemoryManager,
                 episode_buffer: SharedMemoryManager,
                 episode_length_array: mp.Array,
                 cmd_pipe_server: Connection,

                 logger_in_file: bool,
                 debug: bool,
                 ma_name: Optional[str],

                 obs_names: List[str],
                 obs_shapes: List[Tuple[int]],
                 d_action_sizes: List[int],
                 c_action_size: int,
                 model_abs_dir: Path,
                 device: str,
                 last_ckpt: Optional[str],

                 config):

        self._all_variables_buffer = all_variables_buffer

        # Since no set_logger() in main.py
        config_helper.set_logger(debug)

        if logger_in_file:
            config_helper.add_file_logger(model_abs_dir.joinpath('learner.log'))

        if ma_name is None:
            self._logger = logging.getLogger('ds.learner.trainer')
        else:
            self._logger = logging.getLogger(f'ds.learner.trainer.{ma_name}')

        all_variables_buffer.init_logger(self._logger)

        self.base_config = config['base_config']

        nn = importlib.util.module_from_spec(config['sac_config']['nn'])
        config['sac_config']['nn'].loader.exec_module(nn)
        config['sac_config']['nn'] = nn

        self.sac_lock = RLock(1)

        self.sac = SAC_DS_Base(
            obs_names=obs_names,
            obs_shapes=obs_shapes,
            d_action_sizes=d_action_sizes,
            c_action_size=c_action_size,
            model_abs_dir=model_abs_dir,
            device=device,
            ma_name=ma_name,
            last_ckpt=last_ckpt,

            nn_config=config['nn_config'],

            **config['sac_config'])

        self._logger.info('SAC started')

        batch_size = config['sac_config']['batch_size']
        batch_shapes, batch_dtypes = get_batch_shapes_dtype(
            batch_size,
            self.sac.burn_in_step + self.sac.n_step,
            obs_shapes,
            sum(d_action_sizes) + c_action_size,
            self.sac.seq_hidden_state_shape if self.sac.seq_encoder is not None else None)

        self._batch_buffer = SharedMemoryManager(self.base_config['batch_queue_size'],
                                                 logger=logging.getLogger(f'ds.learner.trainer.batch_shmm.{self._logger.name}'),
                                                 logger_level=logging.INFO,
                                                 counter_get_shm_index_empty_log='Batch shm index is empty',
                                                 timer_get_shm_index_log='Get a batch shm index',
                                                 timer_get_data_log='Get a batch',
                                                 timer_put_data_log='Put a batch',
                                                 log_repeat=ELAPSED_REPEAT,
                                                 force_report=ELAPSED_FORCE_REPORT)
        self._batch_buffer.init_from_shapes(batch_shapes, batch_dtypes)

        for i in range(self.base_config['batch_generator_process_num']):
            mp.Process(target=BatchGenerator, kwargs={
                '_id': i,

                'episode_buffer': episode_buffer,
                'episode_length_array': episode_length_array,
                'batch_buffer': self._batch_buffer,

                'logger_in_file': logger_in_file,
                'debug': debug,
                'ma_name': ma_name,

                'model_abs_dir': model_abs_dir,
                'burn_in_step': self.sac.burn_in_step,
                'n_step': self.sac.n_step,
                'batch_size': batch_size,

                'padding_action': self.sac._padding_action
            }).start()

        threading.Thread(target=self._forever_run_cmd_pipe,
                         args=[cmd_pipe_server],
                         daemon=True).start()

        self._update_sac_bak()

        try:
            self.run_train()
        except KeyboardInterrupt:
            self._logger.warning('KeyboardInterrupt')
        finally:
            self._closed = True

    def _update_sac_bak(self):
        step = self.sac.get_global_step()
        self._logger.info(f'Putting all variables for sac_bak updating...')
        with self.sac_lock:
            all_variables = self.sac.get_all_variables()

        self._all_variables_buffer.put(all_variables, pop_last=False)
        self._logger.info(f'All variables are put in buffer (S {step})')

    def _forever_run_cmd_pipe(self, cmd_pipe_server):
        while not self._closed:
            cmd, args = cmd_pipe_server.recv()
            if cmd == 'LOG_EPISODE_SUMMARIES':
                with self.sac_lock:
                    self.sac.write_constant_summaries(args)
            elif cmd == 'SAVE_MODEL':
                with self.sac_lock:
                    self.sac.save_model()

    def run_train(self):
        timer_train = ElapsedTimer('Train a step', self._logger,
                                   repeat=ELAPSED_REPEAT,
                                   force_report=False)

        self._logger.info('Start training...')

        while not self._closed:
            batch, _ = self._batch_buffer.get(timeout=BATCH_QUEUE_TIMEOUT)

            if batch is None:
                continue

            (bn_indexes,
             bn_padding_masks,
             bn_obses_list,
             bn_actions,
             bn_rewards,
             next_obs_list,
             bn_dones,
             bn_mu_probs,
             bn_seq_hidden_states) = batch

            with timer_train:
                with self.sac_lock:
                    try:
                        step = self.sac.train(bn_indexes=bn_indexes,
                                              bn_padding_masks=bn_padding_masks,
                                              bn_obses_list=bn_obses_list,
                                              bn_actions=bn_actions,
                                              bn_rewards=bn_rewards,
                                              next_obs_list=next_obs_list,
                                              bn_dones=bn_dones,
                                              bn_mu_probs=bn_mu_probs,
                                              f_seq_hidden_states=bn_seq_hidden_states[:, :1]
                                              if bn_seq_hidden_states is not None else None)
                    except Exception as e:
                        self._logger.error(e)
                        self._logger.error(traceback.format_exc())

            if step % self.base_config['update_sac_bak_per_step'] == 0:
                self._update_sac_bak()

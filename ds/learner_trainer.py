import importlib
import logging
import math
import multiprocessing as mp
import os
import threading
import traceback
from multiprocessing.connection import Connection
from pathlib import Path
from typing import List

import numpy as np

import algorithm.config_helper as config_helper
from algorithm.utils import (RLock, elapsed_counter, elapsed_timer,
                             episode_to_batch)

from .constants import *
from .sac_ds_base import SAC_DS_Base
from .utils import SharedMemoryManager, get_batch_shapes_dtype, traverse_lists


class BatchGenerator:
    def __init__(self,
                 logger_in_file: bool,
                 model_abs_dir: str,
                 burn_in_step: int,
                 n_step: int,
                 batch_size: int,
                 episode_buffer: SharedMemoryManager,
                 episode_length_array: mp.Array,
                 batch_buffer: SharedMemoryManager):
        self.burn_in_step = burn_in_step
        self.n_step = n_step
        self.batch_size = batch_size
        self._episode_buffer = episode_buffer
        self._episode_length_array = episode_length_array
        self._batch_buffer = batch_buffer

        # Since no set_logger() in main.py
        config_helper.set_logger(Path(model_abs_dir).joinpath(f'learner_trainer_batch_generator_{os.getpid()}.log') if logger_in_file else None)

        self._logger = logging.getLogger(f'ds.learner.trainer.batch_generator_{os.getpid()}')
        self._logger.info(f'BatchGenerator {os.getpid()} initialized')

        episode_buffer.init_logger(self._logger)
        batch_buffer.init_logger(self._logger)

        self.run()

    def run(self):
        rest_batch = None

        while True:
            episode, episode_idx = self._episode_buffer.get(timeout=EPISODE_QUEUE_TIMEOUT)
            if episode is None:
                continue

            (l_indexes,
             l_padding_masks,
             l_obses_list,
             l_actions,
             l_rewards,
             next_obs_list,
             l_dones,
             l_mu_probs,
             l_seq_hidden_states) = episode
            episode_length = self._episode_length_array[episode_idx]

            """
            bn_indexes: [episode_len - bn + 1, bn]
            bn_padding_masks: [episode_len - bn + 1, bn]
            bn_obses_list: list([episode_len - bn + 1, bn, *obs_shapes_i], ...)
            bn_actions: [episode_len - bn + 1, bn, action_size]
            bn_rewards: [episode_len - bn + 1, bn]
            next_obs_list: list([episode_len - bn + 1, *obs_shapes_i], ...)
            bn_dones: [episode_len - bn + 1, bn]
            bn_mu_probs: [episode_len - bn + 1, bn]
            f_seq_hidden_states: [episode_len - bn + 1, 1, *seq_hidden_state_shape]
            """
            ori_batch = episode_to_batch(self.burn_in_step + self.n_step,
                                         episode_length,
                                         l_indexes,
                                         l_padding_masks,
                                         l_obses_list,
                                         l_actions,
                                         l_rewards,
                                         next_obs_list,
                                         l_dones,
                                         l_probs=l_mu_probs,
                                         l_seq_hidden_states=l_seq_hidden_states)

            if rest_batch is not None:
                ori_batch = traverse_lists((rest_batch, ori_batch), lambda rb, b: np.concatenate([rb, b]))
                rest_batch = None

            ori_batch_size = ori_batch[0].shape[0]
            idx = np.random.permutation(ori_batch_size)
            ori_batch = traverse_lists(ori_batch, lambda b: b[idx])

            for i in range(math.ceil(ori_batch_size / self.batch_size)):
                b_i, b_j = i * self.batch_size, (i + 1) * self.batch_size

                batch = traverse_lists(ori_batch, lambda b: b[b_i:b_j, :])

                if b_j > ori_batch_size:
                    rest_batch = batch
                else:
                    self._batch_buffer.put(batch)


class Trainer:
    def __init__(self,
                 all_variables_buffer: SharedMemoryManager,
                 episode_buffer: SharedMemoryManager,
                 episode_length_array: mp.Array,
                 cmd_pipe_server: Connection,

                 logger_in_file,
                 obs_shapes,
                 d_action_size,
                 c_action_size,
                 model_abs_dir,
                 model_spec,
                 device,
                 last_ckpt,
                 config):

        self._all_variables_buffer = all_variables_buffer

        # Since no set_logger() in main.py
        config_helper.set_logger(Path(model_abs_dir).joinpath('learner_trainer.log') if logger_in_file else None)
        self._logger = logging.getLogger('ds.learner.trainer')

        all_variables_buffer.init_logger(self._logger)

        self.base_config = config['base_config']

        custom_nn_model = importlib.util.module_from_spec(model_spec)
        model_spec.loader.exec_module(custom_nn_model)

        self.sac_lock = RLock(1)

        self.sac = SAC_DS_Base(obs_shapes=obs_shapes,
                               d_action_size=d_action_size,
                               c_action_size=c_action_size,
                               model_abs_dir=model_abs_dir,
                               model=custom_nn_model,
                               model_config=config['model_config'],
                               device=device,
                               last_ckpt=last_ckpt,

                               **config['sac_config'])

        self._logger.info('SAC started')

        batch_size = config['sac_config']['batch_size']
        batch_shapes, batch_dtypes = get_batch_shapes_dtype(
            batch_size,
            self.sac.burn_in_step + self.sac.n_step,
            obs_shapes,
            d_action_size + c_action_size,
            self.sac.seq_hidden_state_shape if self.sac.seq_encoder is not None else None)

        self._batch_buffer = SharedMemoryManager(self.base_config['batch_queue_size'],
                                                 logger=self._logger,
                                                 counter_get_shm_index_empty_log='Batch shm index is empty',
                                                 timer_get_shm_index_log='Get a batch shm index',
                                                 timer_get_data_log='Get a batch',
                                                 log_repeat=ELAPSED_REPEAT)
        self._batch_buffer.init_from_shapes(batch_shapes, batch_dtypes)

        for _ in range(self.base_config['batch_generator_process_num']):
            mp.Process(target=BatchGenerator, kwargs={
                'logger_in_file': logger_in_file,
                'model_abs_dir': model_abs_dir,
                'burn_in_step': self.sac.burn_in_step,
                'n_step': self.sac.n_step,
                'batch_size': batch_size,
                'episode_buffer': episode_buffer,
                'episode_length_array': episode_length_array,
                'batch_buffer': self._batch_buffer
            }).start()

        threading.Thread(target=self._forever_run_cmd_pipe,
                         args=[cmd_pipe_server],
                         daemon=True).start()

        self._update_sac_bak()
        self.run_train()

    def _update_sac_bak(self):
        self._logger.info('Updating sac_bak...')
        with self.sac_lock:
            all_variables = self.sac.get_all_variables()

        self._all_variables_buffer.put(all_variables, pop_last=False)

    def _forever_run_cmd_pipe(self, cmd_pipe_server):
        while True:
            cmd, args = cmd_pipe_server.recv()
            if cmd == 'GET':
                cmd_pipe_server.send(self.sac.get_nn_variables())
                self._logger.info('Sent all nn variables')
            elif cmd == 'UPDATE':
                with self.sac_lock:
                    self.sac.update_nn_variables(args)
                self._logger.info('Updated all nn variables')
            elif cmd == 'LOG_EPISODE_SUMMARIES':
                with self.sac_lock:
                    self.sac.write_constant_summaries(args)
            elif cmd == 'SAVE_MODEL':
                with self.sac_lock:
                    self.sac.save_model()

    def run_train(self):
        timer_train = elapsed_timer(self._logger, 'Train a step', ELAPSED_REPEAT)

        self._logger.info('Start training...')

        while True:
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
             f_seq_hidden_states) = batch

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
                                              f_seq_hidden_states=f_seq_hidden_states)
                    except Exception as e:
                        self._logger.error(e)
                        self._logger.error(traceback.format_exc())

            if step % self.base_config['update_sac_bak_per_step'] == 0:
                self._update_sac_bak()

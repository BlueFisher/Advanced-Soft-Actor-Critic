import importlib
import logging
import math
import multiprocessing as mp
import os
import threading
from multiprocessing.connection import Connection
from pathlib import Path
from typing import List

import numpy as np

import algorithm.config_helper as config_helper
from algorithm.utils import RLock, elapsed_counter, elapsed_timer

from .constants import *
from .sac_ds_base import SAC_DS_Base
from .utils import SharedMemoryManager, traverse_lists


class BatchGenerator:
    def __init__(self,
                 logger_in_file: bool,
                 model_abs_dir: str,
                 use_rnn: bool,
                 burn_in_step: int,
                 n_step: int,
                 batch_size: int,
                 episode_buffer: SharedMemoryManager,
                 episode_size_array: mp.Array,
                 batch_buffer: SharedMemoryManager):
        self.use_rnn = use_rnn
        self.burn_in_step = burn_in_step
        self.n_step = n_step
        self.batch_size = batch_size
        self._episode_buffer = episode_buffer
        self._episode_size_array = episode_size_array
        self._batch_buffer = batch_buffer

        # Since no set_logger() in main.py
        config_helper.set_logger(Path(model_abs_dir).joinpath(f'learner_trainer_batch_generator_{os.getpid()}.log') if logger_in_file else None)

        self._logger = logging.getLogger(f'ds.learner.trainer.batch_generator_{os.getpid()}')
        self._logger.info(f'BatchGenerator {os.getpid()} initialized')

        episode_buffer.init_logger(self._logger)
        batch_buffer.init_logger(self._logger)

        self.run()

    def _episode_to_batch(self, episode_size: int,
                          n_obses_list: List[np.ndarray],
                          n_actions: np.ndarray,
                          n_rewards: np.ndarray,
                          next_obs_list: List[np.ndarray],
                          n_dones: np.ndarray,
                          n_mu_probs: np.ndarray = None,
                          n_rnn_states: np.ndarray = None):
        """
        Args:
            episode_size: int, Indicates true episode_len, not MAX_EPISODE_SIZE
            n_obses_list: list([1, episode_len, *obs_shapes_i], ...)
            n_actions: [1, episode_len, action_size]
            n_rewards: [1, episode_len]
            next_obs_list: list([1, *obs_shapes_i], ...)
            n_dones: [1, episode_len]
            n_mu_probs: [1, episode_len]
            n_rnn_states: [1, episode_len, *rnn_state_shape]

        Returns:
            n_obses_list: list([episode_len - N + 1, N, *obs_shapes_i], ...)
            n_actions: [episode_len - N + 1, N, action_size]
            n_rewards: [episode_len - N + 1, N]
            next_obs_list: list([episode_len - N + 1, *obs_shapes_i], ...)
            n_dones: [episode_len - N + 1, N]
            n_mu_probs: [episode_len - N + 1, N]
            rnn_state: [episode_len - N + 1, *rnn_state_shape]
        """
        N = self.burn_in_step + self.n_step

        tmp_n_obses_list = [None] * len(n_obses_list)
        for j, n_obses in enumerate(n_obses_list):
            tmp_n_obses_list[j] = np.concatenate([n_obses[:, i:i + N]
                                                  for i in range(episode_size - N + 1)], axis=0)
        n_actions = np.concatenate([n_actions[:, i:i + N]
                                    for i in range(episode_size - N + 1)], axis=0)
        n_rewards = np.concatenate([n_rewards[:, i:i + N]
                                    for i in range(episode_size - N + 1)], axis=0)
        tmp_next_obs_list = [None] * len(next_obs_list)
        for j, n_obses in enumerate(n_obses_list):
            tmp_next_obs_list[j] = np.concatenate([n_obses[:, i + N]
                                                   for i in range(episode_size - N)]
                                                  + [next_obs_list[j]],
                                                  axis=0)
        n_dones = np.concatenate([n_dones[:, i:i + N]
                                  for i in range(episode_size - N + 1)], axis=0)

        n_mu_probs = np.concatenate([n_mu_probs[:, i:i + N]
                                     for i in range(episode_size - N + 1)], axis=0)

        if self.use_rnn:
            rnn_state = np.concatenate([n_rnn_states[:, i]
                                        for i in range(episode_size - N + 1)], axis=0)

        return [tmp_n_obses_list,
                n_actions,
                n_rewards,
                tmp_next_obs_list,
                n_dones,
                n_mu_probs,
                rnn_state if self.use_rnn else None]

    def run(self):
        rest_batch = None

        while True:
            episode, episode_idx = self._episode_buffer.get(timeout=EPISODE_QUEUE_TIMEOUT)
            if episode is None:
                continue

            (n_obses_list,
             n_actions,
             n_rewards,
             next_obs_list,
             n_dones,
             n_mu_probs,
             n_rnn_states) = episode
            episode_size = self._episode_size_array[episode_idx]

            """
            n_obses_list: list([episode_len - N + 1, N, *obs_shapes_i], ...)
            n_actions: [episode_len - N + 1, N, action_size]
            n_rewards: [episode_len - N + 1, N]
            next_obs_list: list([episode_len - N + 1, *obs_shapes_i], ...)
            n_dones: [episode_len - N + 1, N]
            n_mu_probs: [episode_len - N + 1, N]
            rnn_state: [episode_len - N + 1, *rnn_state_shape]
            """
            ori_batch = self._episode_to_batch(episode_size,
                                               n_obses_list,
                                               n_actions,
                                               n_rewards,
                                               next_obs_list,
                                               n_dones,
                                               n_mu_probs,
                                               n_rnn_states)

            if rest_batch is not None:
                ori_batch = traverse_lists((rest_batch, ori_batch), lambda rb, b: np.concatenate([rb, b]))
                rest_batch = None

            ori_batch_size = ori_batch[0][0].shape[0]
            idx = np.arange(ori_batch_size)
            np.random.shuffle(idx)
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
                 episode_size_array: mp.Array,
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
                               device=device,
                               last_ckpt=last_ckpt,

                               **config['sac_config'])

        self._logger.info('SAC started')

        batch_size = config['replay_config']['batch_size']
        N = self.sac.burn_in_step + self.sac.n_step
        batch_shapes = [
            [(batch_size, N, *o) for o in obs_shapes],
            (batch_size, N, d_action_size + c_action_size),
            (batch_size, N),
            [(batch_size, *o) for o in obs_shapes],
            (batch_size, N),
            (batch_size, N),
            (batch_size, *self.sac.rnn_state_shape) if self.sac.use_rnn else None
        ]
        self._batch_buffer = SharedMemoryManager(BATCH_QUEUE_SIZE,
                                                 logger=self._logger,
                                                 counter_get_shm_index_empty_log='Batch shm index is empty',
                                                 timer_get_shm_index_log='Get a batch shm index',
                                                 timer_get_data_log='Get a batch',
                                                 log_repeat=ELAPSED_REPEAT)
        self._batch_buffer.init_from_shapes(batch_shapes, np.float32)

        for _ in range(BATCH_GENERATOR_PROCESS_NUM):
            mp.Process(target=BatchGenerator, kwargs={
                'logger_in_file': logger_in_file,
                'model_abs_dir': model_abs_dir,
                'use_rnn': self.sac.use_rnn,
                'burn_in_step': self.sac.burn_in_step,
                'n_step': self.sac.n_step,
                'batch_size': batch_size,
                'episode_buffer': episode_buffer,
                'episode_size_array': episode_size_array,
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

            (n_obses_list,
             n_actions,
             n_rewards,
             next_obs_list,
             n_dones,
             n_mu_probs,
             rnn_state) = batch

            with timer_train:
                with self.sac_lock:
                    step = self.sac.train(n_obses_list=n_obses_list,
                                          n_actions=n_actions,
                                          n_rewards=n_rewards,
                                          next_obs_list=next_obs_list,
                                          n_dones=n_dones,
                                          n_mu_probs=n_mu_probs,
                                          rnn_state=rnn_state)

            if step % self.base_config['update_sac_bak_per_step'] == 0:
                self._update_sac_bak()

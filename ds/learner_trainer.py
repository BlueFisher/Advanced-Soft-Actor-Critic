import importlib
import logging
import math
import multiprocessing as mp
import threading
import time
from multiprocessing.connection import Connection
from pathlib import Path
from queue import Queue
from typing import List

import numpy as np

import algorithm.config_helper as config_helper
import algorithm.constants as C
from algorithm.utils import RLock, elapsed_timer

from .sac_ds_base import SAC_DS_Base


class BatchDataBuffer:
    def __init__(self, sac: SAC_DS_Base, batch_size: int):
        self._sac = sac
        self.batch_size = batch_size

        self._buffer = Queue(maxsize=C.LEARNER_BATCH_DATA_BUFFER_SIZE)
        self._logger = logging.getLogger('ds.learner.trainer.batch_data_buffer')

    def add_episode(self,
                    n_obses_list: List[np.ndarray],
                    n_actions: np.ndarray,
                    n_rewards: np.ndarray,
                    next_obs_list: List[np.ndarray],
                    n_dones: np.ndarray,
                    n_mu_probs: np.ndarray,
                    n_rnn_states: np.ndarray = None):
        """
        Args:
            n_obses_list: list([1, episode_len, *obs_shapes_i], ...)
            n_actions: [1, episode_len, action_size]
            n_rewards: [1, episode_len]
            next_obs_list: list([1, *obs_shapes_i], ...)
            n_dones: [1, episode_len]
            n_mu_probs: [1, episode_len]
            n_rnn_states: [1, episode_len, *rnn_state_shape]
        """
        if self._buffer.full():
            self._logger.warning('Buffer is full, episode ignored')
            return

        (n_obses_list,
         n_actions,
         n_rewards,
         next_obs_list,
         n_dones,
         n_mu_probs,
         rnn_state) = self._sac.episode_to_batch(n_obses_list,
                                                 n_actions,
                                                 n_rewards,
                                                 next_obs_list,
                                                 n_dones,
                                                 n_mu_probs,
                                                 n_rnn_states)

        all_batch = n_obses_list[0].shape[0]
        for i in range(math.ceil(all_batch / self.batch_size)):
            b_i, b_j = i * self.batch_size, (i + 1) * self.batch_size

            _n_obses_list = [o[b_i:b_j, :] for o in n_obses_list]
            _n_actions = n_actions[b_i:b_j, :]
            _n_rewards = n_rewards[b_i:b_j, :]
            _next_obs_list = [o[b_i:b_j, :] for o in next_obs_list]
            _n_dones = n_dones[b_i:b_j, :]
            _n_mu_probs = n_mu_probs[b_i:b_j, :]
            _rnn_state = rnn_state[b_i:b_j, :] if self._sac.use_rnn else None

            self._buffer.put((_n_obses_list,
                              _n_actions,
                              _n_rewards,
                              _next_obs_list,
                              _n_dones,
                              _n_mu_probs,
                              _rnn_state))

    def get(self):
        return self._buffer.get()


class Trainer:
    def __init__(self,
                 all_variables_queue: mp.Queue,
                 episode_queue: mp.Queue,
                 cmd_pipe_server: Connection,

                 logger_in_file,
                 obs_shapes,
                 d_action_size,
                 c_action_size,
                 model_abs_dir,
                 model_spec,
                 last_ckpt,
                 config):

        self._logger = logging.getLogger('ds.learner.trainer')
        self._all_variables_queue = all_variables_queue

        self.base_config = config['base_config']

        # Since no set_logger() in main.py
        config_helper.set_logger(Path(model_abs_dir).joinpath(f'learner_trainer.log') if logger_in_file else None)

        custom_nn_model = importlib.util.module_from_spec(model_spec)
        model_spec.loader.exec_module(custom_nn_model)

        self.sac_lock = RLock(1)

        self.sac = SAC_DS_Base(obs_shapes=obs_shapes,
                               d_action_size=d_action_size,
                               c_action_size=c_action_size,
                               model_abs_dir=model_abs_dir,
                               model=custom_nn_model,
                               last_ckpt=last_ckpt,

                               **config['sac_config'])

        self._logger.info('SAC started')

        self._batch_data_buffer = BatchDataBuffer(self.sac,
                                                  batch_size=config['replay_config']['batch_size'])

        for _ in range(C.LEARNER_PROCESS_EPISODE_THREAD_NUM):
            threading.Thread(target=self._forever_process_add_episode,
                             args=[episode_queue],
                             daemon=True).start()

        threading.Thread(target=self._forever_run_cmd_pipe,
                         args=[cmd_pipe_server],
                         daemon=True).start()

        self._update_sac_bak()
        self.run_train()

    def _update_sac_bak(self):
        with self.sac_lock:
            self._logger.info('Updated sac_bak')
            all_variables = self.sac.get_all_variables()
            self._all_variables_queue.put(all_variables)

    def _forever_process_add_episode(self, episode_queue: mp.Queue):
        while True:
            self._batch_data_buffer.add_episode(*episode_queue.get())

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
        while True:
            (n_obses_list,
             n_actions,
             n_rewards,
             next_obs_list,
             n_dones,
             n_mu_probs,
             rnn_state) = self._batch_data_buffer.get()

            with elapsed_timer(self._logger, 'train'):
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

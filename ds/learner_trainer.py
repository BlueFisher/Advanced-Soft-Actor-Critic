import importlib
import logging
import math
import multiprocessing as mp
import os
import threading
from multiprocessing.connection import Connection
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from queue import Empty
from typing import List, Tuple, Union

import numpy as np

import algorithm.config_helper as config_helper
import algorithm.constants as C
from algorithm.utils import RLock, elapsed_counter, elapsed_timer

from .sac_ds_base import SAC_DS_Base


def traverse_lists(data: Tuple, process):
    buffer = []
    for d in zip(*data):
        if isinstance(d[0], list):
            buffer.append(traverse_lists(d, process))
        elif d[0] is None:
            buffer.append(None)
        else:
            buffer.append(process(*d))

    return buffer


class BatchGenerator:
    def __init__(self,
                 logger_in_file: bool,
                 model_abs_dir: str,
                 use_rnn: bool,
                 burn_in_step: int,
                 n_step: int,
                 batch_size: int,
                 episode_queue: mp.Queue,
                 batch_shapes: List[Union[Tuple, List[Tuple]]],
                 batch_shms: List[Union[SharedMemory, List[SharedMemory]]],
                 batch_shm_index_queue: mp.Queue,
                 batch_free_shm_index_queue: mp.Queue):
        self.use_rnn = use_rnn
        self.burn_in_step = burn_in_step
        self.n_step = n_step
        self.batch_size = batch_size
        self.episode_queue = episode_queue
        self.batch_shapes = batch_shapes
        self.batch_shms = batch_shms
        self.batch_shm_index_queue = batch_shm_index_queue
        self.batch_free_shm_index_queue = batch_free_shm_index_queue

        self._tmp = None

        # Since no set_logger() in main.py
        config_helper.set_logger(Path(model_abs_dir).joinpath(f'learner_trainer_batch_generator_{os.getpid()}.log') if logger_in_file else None)

        self._logger = logging.getLogger(f'ds.learner.trainer.batch_generator_{os.getpid()}')
        self._logger.info(f'BatchGenerator {os.getpid()} initialized')

        self.run()

    def _episode_to_batch(self,
                          n_obses_list: List[np.ndarray],
                          n_actions: np.ndarray,
                          n_rewards: np.ndarray,
                          next_obs_list: List[np.ndarray],
                          n_dones: np.ndarray,
                          n_mu_probs: np.ndarray = None,
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

        Returns:
            n_obses_list: list([episode_len - ignore + 1, N, *obs_shapes_i], ...)
            n_actions: [episode_len - ignore + 1, N, action_size]
            n_rewards: [episode_len - ignore + 1, N]
            next_obs_list: list([episode_len - ignore + 1, *obs_shapes_i], ...)
            n_dones: [episode_len - ignore + 1, N]
            n_mu_probs: [episode_len - ignore + 1, N]
            rnn_state: [episode_len - ignore + 1, *rnn_state_shape]
        """
        ignore_size = self.burn_in_step + self.n_step

        tmp_n_obses_list = [None] * len(n_obses_list)
        for j, n_obses in enumerate(n_obses_list):
            tmp_n_obses_list[j] = np.concatenate([n_obses[:, i:i + ignore_size]
                                                  for i in range(n_obses.shape[1] - ignore_size + 1)], axis=0)
        n_actions = np.concatenate([n_actions[:, i:i + ignore_size]
                                    for i in range(n_actions.shape[1] - ignore_size + 1)], axis=0)
        n_rewards = np.concatenate([n_rewards[:, i:i + ignore_size]
                                    for i in range(n_rewards.shape[1] - ignore_size + 1)], axis=0)
        tmp_next_obs_list = [None] * len(next_obs_list)
        for j, n_obses in enumerate(n_obses_list):
            tmp_next_obs_list[j] = np.concatenate([n_obses[:, i + ignore_size]
                                                   for i in range(n_obses.shape[1] - ignore_size)]
                                                  + [next_obs_list[j]],
                                                  axis=0)
        n_dones = np.concatenate([n_dones[:, i:i + ignore_size]
                                  for i in range(n_dones.shape[1] - ignore_size + 1)], axis=0)

        n_mu_probs = np.concatenate([n_mu_probs[:, i:i + ignore_size]
                                     for i in range(n_mu_probs.shape[1] - ignore_size + 1)], axis=0)

        if self.use_rnn:
            rnn_state = np.concatenate([n_rnn_states[:, i]
                                        for i in range(n_rnn_states.shape[1] - ignore_size + 1)], axis=0)

        return (
            tmp_n_obses_list,
            n_actions,
            n_rewards,
            tmp_next_obs_list,
            n_dones,
            n_mu_probs,
            rnn_state if self.use_rnn else None
        )

    def run(self):
        counter_episode_queue_empty = elapsed_counter(self._logger, 'Episode queue is empty', C.ELAPSED_COUNTER_REPEAT)
        timer_get = elapsed_timer(self._logger, 'Get an episode', C.ELAPSED_TIMER_REPEAT)
        timer_process = elapsed_timer(self._logger, 'Process an episode', C.ELAPSED_TIMER_REPEAT)

        while True:
            with timer_get, counter_episode_queue_empty:
                try:
                    (n_obses_list,
                     n_actions,
                     n_rewards,
                     next_obs_list,
                     n_dones,
                     n_mu_probs,
                     n_rnn_states) = self.episode_queue.get(timeout=C.EPISODE_QUEUE_TIMEOUT)
                except Empty:
                    counter_episode_queue_empty.add()
                    timer_get.ignore()
                    continue

            with timer_process:
                (n_obses_list,
                 n_actions,
                 n_rewards,
                 next_obs_list,
                 n_dones,
                 n_mu_probs,
                 rnn_state) = self._episode_to_batch(n_obses_list,
                                                     n_actions,
                                                     n_rewards,
                                                     next_obs_list,
                                                     n_dones,
                                                     n_mu_probs,
                                                     n_rnn_states)

                if self._tmp is not None:
                    (tmp_n_obses_list,
                     tmp_n_actions,
                     tmp_n_rewards,
                     tmp_next_obs_list,
                     tmp_n_dones,
                     tmp_n_mu_probs,
                     tmp_rnn_state) = self._tmp

                    n_obses_list = [np.concatenate([tmp_o, o]) for tmp_o, o in zip(tmp_n_obses_list, n_obses_list)]
                    n_actions = np.concatenate([tmp_n_actions, n_actions])
                    n_rewards = np.concatenate([tmp_n_rewards, n_rewards])
                    next_obs_list = [np.concatenate([tmp_o, o]) for tmp_o, o in zip(tmp_next_obs_list, next_obs_list)]
                    n_dones = np.concatenate([tmp_n_dones, n_dones])
                    n_mu_probs = np.concatenate([tmp_n_mu_probs, n_mu_probs])
                    rnn_state = np.concatenate([tmp_rnn_state, rnn_state]) if self.use_rnn else None

                    self._tmp = None

                all_batch_size = n_obses_list[0].shape[0]
                idx = np.arange(all_batch_size)
                np.random.shuffle(idx)

                n_obses_list = [o[idx] for o in n_obses_list]
                n_actions = n_actions[idx]
                n_rewards = n_rewards[idx]
                next_obs_list = [o[idx] for o in next_obs_list]
                n_dones = n_dones[idx]
                n_mu_probs = n_mu_probs[idx]
                rnn_state = rnn_state[idx] if self.use_rnn else None

                for i in range(math.ceil(all_batch_size / self.batch_size)):
                    b_i, b_j = i * self.batch_size, (i + 1) * self.batch_size

                    batch = [
                        [o[b_i:b_j, :] for o in n_obses_list],
                        n_actions[b_i:b_j, :],
                        n_rewards[b_i:b_j, :],
                        [o[b_i:b_j, :] for o in next_obs_list],
                        n_dones[b_i:b_j, :],
                        n_mu_probs[b_i:b_j, :],
                        rnn_state[b_i:b_j, :] if self.use_rnn else None
                    ]

                    if b_j > all_batch_size:
                        self._tmp = batch
                    else:
                        if self.batch_shm_index_queue.full():
                            shm_idx = self.batch_shm_index_queue.get()
                        else:
                            shm_idx = self.batch_free_shm_index_queue.get()

                        # Copy batch data to shm
                        def _tra(b, shape, shms):
                            shm_np = np.ndarray(shape, dtype=np.float32, buffer=shms[shm_idx].buf)
                            shm_np[:] = b[:]
                        traverse_lists((batch, self.batch_shapes, self.batch_shms), _tra)

                        self.batch_shm_index_queue.put(shm_idx)


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
                 device,
                 last_ckpt,
                 config):

        self._logger = logging.getLogger('ds.learner.trainer')
        self._all_variables_queue = all_variables_queue

        self.base_config = config['base_config']

        # Since no set_logger() in main.py
        config_helper.set_logger(Path(model_abs_dir).joinpath('learner_trainer.log') if logger_in_file else None)

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

        N = self.sac.burn_in_step + self.sac.n_step
        batch_size = config['replay_config']['batch_size']

        self.batch_shapes = [
            [(batch_size, N, *o) for o in obs_shapes],
            (batch_size, N, d_action_size + c_action_size),
            (batch_size, N),
            [(batch_size, *o) for o in obs_shapes],
            (batch_size, N),
            (batch_size, N),
            (batch_size, *self.sac.rnn_state_shape) if self.sac.use_rnn else None
        ]

        self.batch_buffer = traverse_lists((self.batch_shapes,),
                                           lambda d: np.empty(d, dtype=np.float32))  # Store numpy copys from shm
        self.batch_shms = traverse_lists((self.batch_buffer,),
                                         lambda d: [SharedMemory(create=True, size=d.nbytes) for _ in range(C.BATCH_QUEUE_SIZE)])

        self.batch_shm_index_queue = mp.Queue(C.BATCH_QUEUE_SIZE)  # shm index that have batch data
        self.batch_free_shm_index_queue = mp.Queue(C.BATCH_QUEUE_SIZE)  # shm index that have not batch data
        for i in range(C.BATCH_QUEUE_SIZE):
            self.batch_free_shm_index_queue.put(i)

        for _ in range(C.BATCH_GENERATOR_PROCESS_NUM):
            mp.Process(target=BatchGenerator, kwargs={
                'logger_in_file': logger_in_file,
                'model_abs_dir': model_abs_dir,
                'use_rnn': self.sac.use_rnn,
                'burn_in_step': self.sac.burn_in_step,
                'n_step': self.sac.n_step,
                'batch_size': batch_size,
                'episode_queue': episode_queue,
                'batch_shapes': self.batch_shapes,
                'batch_shms': self.batch_shms,
                'batch_shm_index_queue': self.batch_shm_index_queue,
                'batch_free_shm_index_queue': self.batch_free_shm_index_queue
            }).start()

        threading.Thread(target=self._forever_run_cmd_pipe,
                         args=[cmd_pipe_server],
                         daemon=True).start()

        self._update_sac_bak()
        self.run_train()

    def _update_sac_bak(self):
        with self.sac_lock:
            self._logger.info('Updating sac_bak...')
            all_variables = self.sac.get_all_variables()
            self._all_variables_queue.put(all_variables)

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
        counter_batch_shm_index_queue_empty = elapsed_counter(self._logger, 'Batch shm index queue is empty', C.ELAPSED_COUNTER_REPEAT)
        timer_get_shm = elapsed_timer(self._logger, 'Get a shm index', C.ELAPSED_TIMER_REPEAT)
        timer_get = elapsed_timer(self._logger, 'Get a batch from buffer', C.ELAPSED_TIMER_REPEAT)
        timer_train = elapsed_timer(self._logger, 'Train a step', C.ELAPSED_TIMER_REPEAT)

        self._logger.info('Start training...')

        while True:
            with timer_get_shm, counter_batch_shm_index_queue_empty:
                try:
                    shm_idx = self.batch_shm_index_queue.get(timeout=C.BATCH_QUEUE_TIMEOUT)
                except Empty:
                    counter_batch_shm_index_queue_empty.add()
                    timer_get_shm.ignore()
                    continue

            with timer_get:
                # Copy shm to batch_buffer
                for i, s in enumerate(self.batch_shapes):
                    if isinstance(s, list):
                        for j, c_s in enumerate(s):
                            shm_np = np.ndarray(c_s, dtype=np.float32, buffer=self.batch_shms[i][j][shm_idx].buf)
                            self.batch_buffer[i][j][:] = shm_np[:]
                    elif s is None:
                        continue
                    else:
                        shm_np = np.ndarray(s, dtype=np.float32, buffer=self.batch_shms[i][shm_idx].buf)
                        self.batch_buffer[i][:] = shm_np[:]

                self.batch_free_shm_index_queue.put(shm_idx)

            (n_obses_list,
             n_actions,
             n_rewards,
             next_obs_list,
             n_dones,
             n_mu_probs,
             rnn_state) = self.batch_buffer

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

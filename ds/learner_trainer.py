import importlib
import logging
import multiprocessing as mp
import threading
import time
from multiprocessing.connection import Connection
from pathlib import Path
from queue import Full, Queue

import numpy as np

import algorithm.config_helper as config_helper
import algorithm.constants as C
from algorithm.utils import RLock

from .sac_ds_base import SAC_DS_Base


class UpdateDataBuffer:
    def __init__(self, update_td_error, update_transitions):
        self._update_td_error = update_td_error
        self._update_transitions = update_transitions

        self._closed = False
        self._buffer = Queue(maxsize=C.UPDATE_DATA_BUFFER_MAXSIZE)
        self.logger = logging.getLogger('ds.learner.trainer.update_data_buffer')

        ts = [threading.Thread(target=self._run) for _ in range(C.UPDATE_DATA_BUFFER_THREADS)]
        for t in ts:
            t.start()

    def _run(self):
        # TODO .numpy() threading
        while not self._closed:
            is_td_error, *data = self._buffer.get()
            if is_td_error:
                pointers, td_error = data
                self._update_td_error(pointers, td_error)
            else:
                pointers, key, data = data
                self._update_transitions(pointers, key, data)

    def add_data(self, is_td_error, *data):
        try:
            self._buffer.put_nowait((is_td_error, *data))
        except Full:
            self.logger.warning('Buffer is full, the data to be update is ignored')

    def close(self):
        self._closed = True


class Trainer:
    def __init__(self,
                 get_sampled_data_queue: mp.Queue,
                 update_td_error_queue: mp.Queue,
                 update_transition_queue: mp.Queue,
                 update_sac_bak_queue: mp.Queue,

                 process_nn_variables_conn: Connection,

                 logger_in_file,
                 obs_shapes,
                 d_action_size,
                 c_action_size,
                 model_abs_dir,
                 model_spec,
                 last_ckpt,
                 config):

        self.logger = logging.getLogger('ds.learner.trainer')
        self._get_sampled_data_queue = get_sampled_data_queue
        self._update_td_error_queue = update_td_error_queue
        self._update_transition_queue = update_transition_queue
        self._update_sac_bak_queue = update_sac_bak_queue

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

        self.logger.info('SAC started')

        threading.Thread(target=self._forever_process_nn_variables,
                         args=[process_nn_variables_conn],
                         daemon=True).start()

        self._update_sac_bak()
        self.run_train()

    def _update_sac_bak(self):
        with self.sac_lock:
            self.logger.info('Updated sac_bak')
            all_variables = self.sac.get_all_variables()
            self._update_sac_bak_queue.put(all_variables)

    def _forever_process_nn_variables(self, process_nn_variables_conn):
        while True:
            cmd, args = process_nn_variables_conn.recv()
            if cmd == 'GET':
                process_nn_variables_conn.send(self.sac.get_nn_variables())
                self.logger.info('Sent all nn variables')
            elif cmd == 'UPDATE':
                with self.sac_lock:
                    self.sac.update_nn_variables(args)
                self.logger.info('Updated all nn variables')
            elif cmd == 'SAVE_MODEL':
                with self.sac_lock:
                    self.sac.save_model()

    def run_train(self):
        update_data_buffer = UpdateDataBuffer(lambda pointers, td_error:
                                              self._update_td_error_queue.put((pointers, td_error)),
                                              lambda pointers, key, data:
                                              self._update_transition_queue.put((pointers, key, data)))

        while True:
            (pointers,
             (n_obses_list,
              n_actions,
              n_rewards,
              next_obs_list,
              n_dones,
              n_mu_probs,
              priority_is,
              rnn_state)) = self._get_sampled_data_queue.get()
            self._is_training = True

            with self.sac_lock:
                step, td_error, update_data = self.sac.train(pointers=pointers,
                                                             n_obses_list=n_obses_list,
                                                             n_actions=n_actions,
                                                             n_rewards=n_rewards,
                                                             next_obs_list=next_obs_list,
                                                             n_dones=n_dones,
                                                             n_mu_probs=n_mu_probs,
                                                             priority_is=priority_is,
                                                             rnn_state=rnn_state)

            if step % self.base_config['update_sac_bak_per_step'] == 0:
                self._update_sac_bak()

            if np.isnan(np.min(td_error)):
                self.logger.error('NAN in td_error')
                break

            update_data_buffer.add_data(True, pointers, td_error)
            for pointers, key, data in update_data:
                update_data_buffer.add_data(False, pointers, key, data)

        update_data_buffer.close()
        self.logger.warning('Training exits')

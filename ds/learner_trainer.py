import importlib
import logging
import threading
import time
import traceback
from multiprocessing.connection import Connection
from multiprocessing.sharedctypes import SynchronizedArray
from pathlib import Path

import algorithm.config_helper as config_helper
from algorithm.utils import RLock, traverse_lists
from algorithm.utils.elapse_timer import ElapsedTimer

from .constants import *
from .sac_ds_base import SAC_DS_Base
from .utils import SharedMemoryManager, traverse_lists


class Trainer:
    _closed = False

    def __init__(self,
                 all_variables_buffer: SharedMemoryManager,
                 episode_buffer: SharedMemoryManager,
                 episode_length_array: SynchronizedArray,
                 cmd_pipe_server: Connection,

                 logger_in_file: bool,
                 debug: bool,
                 ma_name: str | None,

                 obs_names: list[str],
                 obs_shapes: list[tuple[int]],
                 d_action_sizes: list[int],
                 c_action_size: int,
                 model_abs_dir: Path,
                 device: str,
                 last_ckpt: str | None,

                 config):

        self._all_variables_buffer = all_variables_buffer
        self._episode_buffer = episode_buffer
        self._episode_length_array = episode_length_array

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
            train_mode=True,
            last_ckpt=last_ckpt,

            nn_config=config['nn_config'],
            **config['sac_config'],

            replay_config=config['replay_config'])

        self._logger.info('SAC started')

        threading.Thread(target=self._forever_run_cmd_pipe,
                         args=[cmd_pipe_server],
                         daemon=True).start()

        threading.Thread(target=self._forever_run_put_episode,
                         daemon=True).start()

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

    def _forever_run_put_episode(self):
        while not self._closed:
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
             ep_pre_seq_hidden_states) = episode

            self.sac.put_episode(ep_indexes=ep_indexes,
                                 ep_obses_list=ep_obses_list,
                                 ep_actions=ep_actions,
                                 ep_rewards=ep_rewards,
                                 ep_dones=ep_dones,
                                 ep_probs=ep_mu_probs,
                                 ep_pre_seq_hidden_states=ep_pre_seq_hidden_states)

    def run_train(self):
        timer_train = ElapsedTimer('train', self._logger,
                                   repeat=ELAPSED_REPEAT,
                                   force_report=True)

        self._logger.info('Start training...')

        self._update_sac_bak()

        pre_step = None
        while not self._closed:
            with timer_train:
                with self.sac_lock:
                    try:
                        step = self.sac.train()
                        if step == pre_step:
                            timer_train.ignore()
                            continue
                    except Exception as e:
                        self._logger.error(e)
                        self._logger.error(traceback.format_exc())

            if step > 0 and step % self.base_config['update_sac_bak_per_step'] == 0:
                self._update_sac_bak()

            pre_step = step

import multiprocessing as mp
import threading

from . import constants as C
from .replay import Replay


class AttachedReplay:
    def __init__(self,
                 get_sampled_data_queue: mp.Queue,
                 update_td_error_queue: mp.Queue,
                 update_transition_queue: mp.Queue,
                 init_config):

        self.get_sampled_data_queue = get_sampled_data_queue
        self.update_td_error_queue = update_td_error_queue
        self.update_transition_queue = update_transition_queue
        self.replay = Replay(*init_config, attached=True)

        for _ in range(C.GET_SAMPLED_DATA_THREAD_SIZE):
            t = threading.Thread(target=self._forever_sample_data,
                                 daemon=True)

        for _ in range(C.UPDATE_TD_ERROR_THREAD_SIZE):
            threading.Thread(target=self._forever_update_td_error,
                             daemon=True).start()

        for _ in range(C.UPDATE_TRANSITION_THREAD_SIZE):
            threading.Thread(target=self._forever_update_transition,
                             daemon=True).start()

        t.start()
        t.join()

    def _forever_sample_data(self):
        while True:
            sampled = self.replay.sample()
            if sampled is not None:
                pointers, (n_obses_list,
                           n_actions,
                           n_rewards,
                           next_obs_list,
                           n_dones,
                           n_mu_probs,
                           rnn_state), priority_is = sampled

                self.get_sampled_data_queue.put((pointers, (n_obses_list,
                                                            n_actions,
                                                            n_rewards,
                                                            next_obs_list,
                                                            n_dones,
                                                            n_mu_probs,
                                                            priority_is,
                                                            rnn_state)))

    def _forever_update_td_error(self):
        while True:
            pointers, td_error = self.update_td_error_queue.get()
            self.replay.update_td_error(pointers, td_error)

    def _forever_update_transition(self):
        while True:
            pointers, key, data = self.update_transition_queue.get()
            self.replay.update_transitions(pointers, key, data)

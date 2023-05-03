import math
from typing import List, Union

import numpy as np

from algorithm.utils import episode_to_batch, traverse_lists


class BatchBuffer:
    _rest_batch = None
    _batch_list = []

    def __init__(self,
                 burn_in_step: int,
                 n_step: int,
                 batch_size: int):
        self.burn_in_step = burn_in_step
        self.n_step = n_step
        self.batch_size = batch_size

    def put_episode(self,
                    l_indexes: np.ndarray,
                    l_obses_list: List[np.ndarray],
                    l_actions: np.ndarray,
                    l_rewards: np.ndarray,
                    next_obs_list: List[np.ndarray],
                    l_dones: np.ndarray,
                    l_probs: List[np.ndarray],
                    l_seq_hidden_states: np.ndarray = None) -> None:
        """
        Args:
            l_indexes (np.int32): [1, episode_len]
            l_obses_list: list([1, episode_len, *obs_shapes_i], ...)
            l_actions: [1, episode_len, action_size]
            l_rewards: [1, episode_len]
            next_obs_list: list([1, *obs_shapes_i], ...)
            l_dones (bool): [1, episode_len]
            l_probs: [1, episode_len, action_size]
            l_seq_hidden_states: [1, episode_len, *seq_hidden_state_shape]
        """
        self._batch_list.clear()

        ori_batch = episode_to_batch(burn_in_step=self.burn_in_step,
                                     n_step=self.n_step,
                                     l_indexes=l_indexes,
                                     l_obses_list=l_obses_list,
                                     l_actions=l_actions,
                                     l_rewards=l_rewards,
                                     next_obs_list=next_obs_list,
                                     l_dones=l_dones,
                                     l_probs=l_probs,
                                     l_seq_hidden_states=l_seq_hidden_states)

        if self._rest_batch is not None:
            ori_batch = traverse_lists((self._rest_batch, ori_batch), lambda rb, b: np.concatenate([rb, b]))
            self._rest_batch = None

        ori_batch_size = ori_batch[0].shape[0]
        idx = np.random.permutation(ori_batch_size)
        ori_batch = traverse_lists(ori_batch, lambda b: b[idx])

        for i in range(math.ceil(ori_batch_size / self.batch_size)):
            b_i, b_j = i * self.batch_size, (i + 1) * self.batch_size

            batch = traverse_lists(ori_batch, lambda b: b[b_i:b_j, :])

            if b_j > ori_batch_size:
                self._rest_batch = batch
            else:
                self._batch_list.append(batch)

    def get_batch(self) -> List[Union[np.ndarray, List[np.ndarray]]]:
        return self._batch_list

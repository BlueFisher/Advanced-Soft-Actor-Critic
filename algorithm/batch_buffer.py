import math
from typing import List, Optional, Union

import numpy as np

from algorithm.utils import episode_to_batch, traverse_lists


class BatchBuffer:
    _rest_batch = None

    def __init__(self,
                 burn_in_step: int,
                 n_step: int,
                 batch_size: int):
        self.burn_in_step = burn_in_step
        self.n_step = n_step
        self.batch_size = batch_size
        
        self._batch_list = []

    def put_episode(self,
                    ep_indexes: np.ndarray,
                    ep_padding_masks: np.ndarray,
                    ep_obses_list: List[np.ndarray],
                    ep_actions: np.ndarray,
                    ep_rewards: np.ndarray,
                    ep_dones: np.ndarray,
                    ep_probs: np.ndarray,
                    ep_seq_hidden_states: Optional[np.ndarray] = None) -> None:
        """
        Args:
            ep_indexes (np.int32): [1, ep_len]
            ep_padding_masks: (bool): [1, ep_len]
            ep_obses_list (np): list([1, ep_len, *obs_shapes_i], ...)
            ep_actions (np): [1, ep_len, action_size]
            ep_rewards (np): [1, ep_len]
            ep_dones (bool): [1, ep_len]
            ep_probs (np): [1, ep_len, action_size]
            ep_seq_hidden_states (np): [1, ep_len, *seq_hidden_state_shape]
        """
        self._batch_list.clear()

        ori_batch = episode_to_batch(burn_in_step=self.burn_in_step,
                                     n_step=self.n_step,
                                     l_indexes=ep_indexes,
                                     l_padding_masks=ep_padding_masks,
                                     l_obses_list=ep_obses_list,
                                     l_actions=ep_actions,
                                     l_rewards=ep_rewards,
                                     l_dones=ep_dones,
                                     l_probs=ep_probs,
                                     l_seq_hidden_states=ep_seq_hidden_states)

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

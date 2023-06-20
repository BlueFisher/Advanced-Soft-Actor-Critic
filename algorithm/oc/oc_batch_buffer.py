import math
from typing import List, Optional, Union

import numpy as np

from ..batch_buffer import BatchBuffer
from ..utils import episode_to_batch as vanilla_episode_to_batch
from ..utils import traverse_lists


def episode_to_batch(burn_in_step: int,
                     n_step: int,
                     padding_action: np.ndarray,
                     l_indexes: np.ndarray,
                     l_padding_masks: np.ndarray,
                     l_obses_list: List[np.ndarray],
                     l_option_indexes: np.ndarray,
                     l_option_changed_indexes: np.ndarray,
                     l_actions: np.ndarray,
                     l_rewards: np.ndarray,
                     l_dones: np.ndarray,
                     l_probs: Optional[np.ndarray] = None,
                     l_seq_hidden_states: Optional[np.ndarray] = None,
                     l_low_seq_hidden_states: np.ndarray = None) -> List[Union[np.ndarray, List[np.ndarray]]]:
    """
    Args:
        burn_in_step: int
        n_step: int
        padding_action (np): [action_size, ]
        l_indexes (np.int32): [1, ep_len]
        l_padding_masks (bool): [1, episode_len]
        l_obses_list: list([1, ep_len, *obs_shapes_i], ...)
        l_option_indexes (np.int8): [1, ep_len]
        l_option_changed_indexes (np.int32): [1, ep_len]
        l_actions: [1, ep_len, action_size]
        l_rewards: [1, ep_len]
        next_obs_list: list([1, *obs_shapes_i], ...)
        l_dones (bool): [1, ep_len]
        l_probs: [1, ep_len, action_size]
        l_seq_hidden_states: [1, ep_len, *seq_hidden_state_shape]
        l_low_seq_hidden_states: [1, ep_len, *low_seq_hidden_state_shape]

    Returns:
        bn_indexes (np.int32): [ep_len - bn + 1, bn]
        bn_padding_masks (bool): [ep_len - bn + 1, bn]
        bn_obses_list: list([ep_len - bn + 1, bn, *obs_shapes_i], ...)
        bn_option_indexes (np.int8): [ep_len - bn + 1, bn]
        bn_actions: [ep_len - bn + 1, bn, action_size]
        bn_rewards: [ep_len - bn + 1, bn]
        next_obs_list: list([ep_len - bn + 1, *obs_shapes_i], ...)
        bn_dones (bool): [ep_len - bn + 1, bn]
        bn_probs: [ep_len - bn + 1, bn, action_size]
        f_seq_hidden_states: [ep_len - bn + 1, 1, *seq_hidden_state_shape]
        f_low_seq_hidden_states: [ep_len - bn + 1, 1, *low_seq_hidden_state_shape]
    """
    (bn_indexes,
     bn_padding_masks,
     bn_obses_list,
     bn_actions,
     bn_rewards,
     next_obs_list,
     bn_dones,
     bn_probs,
     bn_seq_hidden_states) = vanilla_episode_to_batch(burn_in_step=burn_in_step,
                                                      n_step=n_step,
                                                      padding_action=padding_action,
                                                      l_indexes=l_indexes,
                                                      l_padding_masks=l_padding_masks,
                                                      l_obses_list=l_obses_list,
                                                      l_actions=l_actions,
                                                      l_rewards=l_rewards,
                                                      l_dones=l_dones,
                                                      l_probs=l_probs,
                                                      l_seq_hidden_states=l_seq_hidden_states)

    bn = burn_in_step + n_step
    ep_len = l_indexes.shape[1]

    # Padding burn_in_step and n_step
    l_option_indexes = np.concatenate([np.full((1, burn_in_step), 0, dtype=l_option_indexes.dtype),
                                       l_option_indexes,
                                       np.full((1, n_step - 1), 0, dtype=l_option_indexes.dtype)], axis=1)
    if l_low_seq_hidden_states is not None:
        l_low_seq_hidden_states = np.concatenate([np.zeros((1, burn_in_step, *l_low_seq_hidden_states.shape[2:]), dtype=l_low_seq_hidden_states.dtype),
                                                  l_low_seq_hidden_states,
                                                  np.zeros((1, n_step - 1, *l_low_seq_hidden_states.shape[2:]), dtype=l_low_seq_hidden_states.dtype)], axis=1)

    # Generate batch
    bn_option_indexes = np.concatenate([l_option_indexes[:, i:i + bn]
                                        for i in range(ep_len - 1)], axis=0)

    if l_low_seq_hidden_states is not None:
        bn_low_seq_hidden_states = np.concatenate([l_low_seq_hidden_states[:, i:i + bn]
                                                  for i in range(ep_len - 1)], axis=0)

    return [bn_indexes,
            bn_padding_masks,
            bn_obses_list,
            bn_option_indexes,
            bn_actions,
            bn_rewards,
            next_obs_list,
            bn_dones,
            bn_probs,
            bn_seq_hidden_states if l_seq_hidden_states is not None else None,
            bn_low_seq_hidden_states if l_seq_hidden_states is not None else None]


class BatchBuffer(BatchBuffer):
    def put_episode(self,
                    ep_indexes: np.ndarray,
                    ep_padding_masks: np.ndarray,
                    ep_obses_list: List[np.ndarray],
                    ep_option_indexes: np.ndarray,
                    ep_option_changed_indexes: np.ndarray,
                    ep_actions: np.ndarray,
                    ep_rewards: np.ndarray,
                    ep_dones: np.ndarray,
                    ep_probs: List[np.ndarray],
                    ep_seq_hidden_states: np.ndarray = None,
                    ep_low_seq_hidden_states: np.ndarray = None) -> None:
        """
        Args:
            ep_indexes (np.int32): [1, ep_len]
            ep_padding_masks: (bool): [1, ep_len]
            ep_obses_list: list([1, ep_len, *obs_shapes_i], ...)
            ep_option_indexes (np.int8): [1, ep_len]
            ep_option_changed_indexes (np.int32): [1, ep_len]
            ep_actions: [1, ep_len, action_size]
            ep_rewards: [1, ep_len]
            ep_dones (bool): [1, ep_len]
            ep_probs: [1, ep_len, action_size]
            ep_seq_hidden_states: [1, ep_len, *seq_hidden_state_shape]
            ep_low_seq_hidden_states: [1, ep_len, *low_seq_hidden_state_shape]
        """
        self._batch_list.clear()

        ori_batch = episode_to_batch(burn_in_step=self.burn_in_step,
                                     n_step=self.n_step,
                                     padding_action=self.padding_action,
                                     l_indexes=ep_indexes,
                                     l_padding_masks=ep_padding_masks,
                                     l_obses_list=ep_obses_list,
                                     l_option_indexes=ep_option_indexes,
                                     l_option_changed_indexes=ep_option_changed_indexes,
                                     l_actions=ep_actions,
                                     l_rewards=ep_rewards,
                                     l_dones=ep_dones,
                                     l_probs=ep_probs,
                                     l_seq_hidden_states=ep_seq_hidden_states,
                                     l_low_seq_hidden_states=ep_low_seq_hidden_states)

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

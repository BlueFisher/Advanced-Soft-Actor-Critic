from typing import List, Optional

import numpy as np

from ..batch_buffer import BatchBuffer
from ..nn_models import *
from ..utils import *


def episode_to_batch(bn: int,
                     episode_length: int,
                     l_indexes: np.ndarray,
                     l_padding_masks: np.ndarray,
                     l_obses_list: List[np.ndarray],
                     l_option_indexes: np.ndarray,
                     l_actions: np.ndarray,
                     l_rewards: np.ndarray,
                     next_obs_list: List[np.ndarray],
                     l_dones: np.ndarray,
                     l_probs: Optional[np.ndarray] = None,
                     l_seq_hidden_states: Optional[np.ndarray] = None,
                     l_low_seq_hidden_states: np.ndarray = None):
    """
    Args:
        bn: int, burn_in_step + n_step
        episode_length: int, Indicates true episode_len, not MAX_EPISODE_LENGTH
        l_indexes: [1, episode_len]
        l_padding_masks: [1, episode_len]
        l_obses_list: list([1, episode_len, *obs_shapes_i], ...)
        l_option_indexes: [1, episode_len]
        l_actions: [1, episode_len, action_size]
        l_rewards: [1, episode_len]
        next_obs_list: list([1, *obs_shapes_i], ...)
        l_dones: [1, episode_len]
        l_probs: [1, episode_len]
        l_seq_hidden_states: [1, episode_len, *seq_hidden_state_shape]
        l_low_seq_hidden_states: [1, episode_len, *low_seq_hidden_state_shape]

    Returns:
        bn_indexes: [episode_len - bn + 1, bn]
        bn_padding_masks: [episode_len - bn + 1, bn]
        bn_obses_list: list([episode_len - bn + 1, bn, *obs_shapes_i], ...)
        bn_options: [episode_len - bn + 1, bn]
        bn_actions: [episode_len - bn + 1, bn, action_size]
        bn_rewards: [episode_len - bn + 1, bn]
        next_obs_list: list([episode_len - bn + 1, *obs_shapes_i], ...)
        bn_dones: [episode_len - bn + 1, bn]
        bn_probs: [episode_len - bn + 1, bn]
        f_seq_hidden_states: [episode_len - bn + 1, 1, *seq_hidden_state_shape]
        f_low_seq_hidden_states: [episode_len - bn + 1, 1, *low_seq_hidden_state_shape]
    """
    bn_indexes = np.concatenate([l_indexes[:, i:i + bn]
                                for i in range(episode_length - bn + 1)], axis=0)
    bn_padding_masks = np.concatenate([l_padding_masks[:, i:i + bn]
                                       for i in range(episode_length - bn + 1)], axis=0)
    tmp_bn_obses_list = [None] * len(l_obses_list)
    for j, l_obses in enumerate(l_obses_list):
        tmp_bn_obses_list[j] = np.concatenate([l_obses[:, i:i + bn]
                                              for i in range(episode_length - bn + 1)], axis=0)
    bn_options = np.concatenate([l_option_indexes[:, i:i + bn]
                                 for i in range(episode_length - bn + 1)], axis=0)
    bn_actions = np.concatenate([l_actions[:, i:i + bn]
                                for i in range(episode_length - bn + 1)], axis=0)
    bn_rewards = np.concatenate([l_rewards[:, i:i + bn]
                                for i in range(episode_length - bn + 1)], axis=0)
    tmp_next_obs_list = [None] * len(next_obs_list)
    for j, l_obses in enumerate(l_obses_list):
        tmp_next_obs_list[j] = np.concatenate([l_obses[:, i + bn]
                                               for i in range(episode_length - bn)]
                                              + [next_obs_list[j]],
                                              axis=0)
    bn_dones = np.concatenate([l_dones[:, i:i + bn]
                              for i in range(episode_length - bn + 1)], axis=0)

    if l_probs is not None:
        bn_probs = np.concatenate([l_probs[:, i:i + bn]
                                   for i in range(episode_length - bn + 1)], axis=0)

    if l_seq_hidden_states is not None:
        f_seq_hidden_states = np.concatenate([l_seq_hidden_states[:, i:i + bn]
                                             for i in range(episode_length - bn + 1)], axis=0)

    if l_low_seq_hidden_states is not None:
        f_low_seq_hidden_states = np.concatenate([l_low_seq_hidden_states[:, i:i + bn]
                                                  for i in range(episode_length - bn + 1)], axis=0)

    return [bn_indexes,
            bn_padding_masks,
            tmp_bn_obses_list,
            bn_options,
            bn_actions,
            bn_rewards,
            tmp_next_obs_list,
            bn_dones,
            bn_probs if l_probs is not None else None,
            f_seq_hidden_states if l_seq_hidden_states is not None else None,
            f_low_seq_hidden_states if l_seq_hidden_states is not None else None]


class BatchBuffer(BatchBuffer):
    def put_episode(self,
                    l_indexes: np.ndarray,
                    l_padding_masks: np.ndarray,
                    l_obses_list: List[np.ndarray],
                    l_option_indexes: np.ndarray,
                    l_actions: np.ndarray,
                    l_rewards: np.ndarray,
                    next_obs_list: List[np.ndarray],
                    l_dones: np.ndarray,
                    l_probs: List[np.ndarray],
                    l_seq_hidden_states: np.ndarray = None,
                    l_low_seq_hidden_states: np.ndarray = None):
        """
        Args:
            l_indexes: [1, episode_len]
            l_padding_masks: [1, episode_len]
            l_obses_list: list([1, episode_len, *obs_shapes_i], ...)
            l_option_indexes: [1, episode_len]
            l_actions: [1, episode_len, action_size]
            l_rewards: [1, episode_len]
            next_obs_list: list([1, *obs_shapes_i], ...)
            l_dones: [1, episode_len]
            l_probs: [1, episode_len]
            l_seq_hidden_states: [1, episode_len, *seq_hidden_state_shape]
            l_low_seq_hidden_states: [1, episode_len, *low_seq_hidden_state_shape]
        """
        self._batch_list.clear()

        ori_batch = episode_to_batch(bn=self.burn_in_step + self.n_step,
                                     episode_length=l_indexes.shape[1],
                                     l_indexes=l_indexes,
                                     l_padding_masks=l_padding_masks,
                                     l_obses_list=l_obses_list,
                                     l_option_indexes=l_option_indexes,
                                     l_actions=l_actions,
                                     l_rewards=l_rewards,
                                     next_obs_list=next_obs_list,
                                     l_dones=l_dones,
                                     l_probs=l_probs,
                                     l_seq_hidden_states=l_seq_hidden_states,
                                     l_low_seq_hidden_states=l_low_seq_hidden_states)

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

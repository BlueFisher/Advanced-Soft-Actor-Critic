import math

import numpy as np

from ..batch_buffer import BatchBuffer
from ..utils import episode_to_batch as vanilla_episode_to_batch
from ..utils import traverse_lists


def episode_to_batch(burn_in_step: int,
                     n_step: int,
                     padding_action: np.ndarray,
                     l_indexes: np.ndarray,
                     l_padding_masks: np.ndarray,
                     l_obses_list: list[np.ndarray],
                     l_option_indexes: np.ndarray,
                     l_option_changed_indexes: np.ndarray,
                     l_actions: np.ndarray,
                     l_rewards: np.ndarray,
                     l_dones: np.ndarray,
                     l_probs: np.ndarray | None = None,
                     l_pre_seq_hidden_states: np.ndarray | None = None,
                     l_pre_low_seq_hidden_states: np.ndarray = None) -> tuple[list[np.ndarray | list[np.ndarray]],
                                                                              list[np.ndarray | list[np.ndarray]]]:
    """
    Args:
        burn_in_step: int
        n_step: int
        padding_action (np): [action_size, ]  The discrete padding actions cannot be all zeros
        l_indexes (np.int32): [1, ep_len]
        l_padding_masks (bool): [1, episode_len]
        l_obses_list: list([1, ep_len, *obs_shapes_i], ...)
        l_option_indexes (np.int8): [1, ep_len]
        l_option_changed_indexes (np.int32): [1, ep_len]
        l_actions: [1, ep_len, action_size]
        l_rewards: [1, ep_len]
        l_dones (bool): [1, ep_len]
        l_probs: [1, ep_len, action_size]
        l_pre_seq_hidden_states: [1, ep_len, *seq_hidden_state_shape]
        l_pre_low_seq_hidden_states: [1, ep_len, *low_seq_hidden_state_shape]

    Returns:
        bn_indexes (np.int32): [ep_len - bn + 1, bn]
        bn_padding_masks (bool): [ep_len - bn + 1, bn]
        m_obses_list: list([ep_len - bn + 1 + 1, bn, *obs_shapes_i], ...)
        bn_option_indexes (np.int8): [ep_len - bn + 1, bn]
        bn_actions: [ep_len - bn + 1, bn, action_size]
        bn_rewards: [ep_len - bn + 1, bn]
        bn_dones (bool): [ep_len - bn + 1, bn]
        bn_probs: [ep_len - bn + 1, bn, action_size]
        m_pre_seq_hidden_states: [ep_len - bn + 1 + 1, 1, *seq_hidden_state_shape]
        m_pre_low_seq_hidden_states: [ep_len - bn + 1 + 1, 1, *low_seq_hidden_state_shape]

        key_indexes (np.int32): [1, key_len]
        key_padding_masks (bool): [1, key_len]
        key_obses_list: list([1, key_len, *obs_shapes_i], ...)
        key_option_index (np.int8): [1, key_len]
        key_seq_hidden_state: [1, key_len, *seq_hidden_state_shape]
    """
    (bn_indexes,
     bn_padding_masks,
     m_obses_list,
     bn_actions,
     bn_rewards,
     bn_dones,
     bn_probs,
     m_pre_seq_hidden_states) = vanilla_episode_to_batch(burn_in_step=burn_in_step,
                                                         n_step=n_step,
                                                         padding_action=padding_action,
                                                         l_indexes=l_indexes,
                                                         l_padding_masks=l_padding_masks,
                                                         l_obses_list=l_obses_list,
                                                         l_actions=l_actions,
                                                         l_rewards=l_rewards,
                                                         l_dones=l_dones,
                                                         l_probs=l_probs,
                                                         l_pre_seq_hidden_states=l_pre_seq_hidden_states)

    bn = burn_in_step + n_step
    ep_len = l_indexes.shape[1]

    # Padding burn_in_step and n_step
    l_option_indexes = np.concatenate([np.full((1, burn_in_step), 0, dtype=l_option_indexes.dtype),
                                       l_option_indexes,
                                       np.full((1, n_step - 1), 0, dtype=l_option_indexes.dtype)], axis=1)
    l_pre_low_seq_hidden_states = np.concatenate([np.zeros((1, burn_in_step, *l_pre_low_seq_hidden_states.shape[2:]), dtype=l_pre_low_seq_hidden_states.dtype),
                                                  l_pre_low_seq_hidden_states,
                                                  np.zeros((1, n_step - 1, *l_pre_low_seq_hidden_states.shape[2:]), dtype=l_pre_low_seq_hidden_states.dtype)], axis=1)

    # Generate batch
    bn_option_indexes = np.concatenate([l_option_indexes[:, i:i + bn]
                                        for i in range(ep_len - 1)], axis=0)

    m_pre_low_seq_hidden_states = np.concatenate([l_pre_low_seq_hidden_states[:, i:i + bn + 1]
                                                  for i in range(ep_len - 1)], axis=0)

    # Generate key
    key_indexes = np.expand_dims(np.unique(l_option_changed_indexes), 0)  # [1, key_len]
    key_padding_masks = l_padding_masks.squeeze(0)[key_indexes]  # [1, key_len]
    key_obses_list = [o.squeeze(0)[key_indexes] for o in l_obses_list]  # list([1, key_len, *obs_shapes_i], ...)
    key_option_index = l_option_indexes.squeeze(0)[key_indexes]  # [1, key_len]
    key_seq_hidden_state = None
    key_seq_hidden_state = l_pre_seq_hidden_states.squeeze(0)[key_indexes]  # [1, key_len, *seq_hidden_state_shape]

    return (bn_indexes,
            bn_padding_masks,
            m_obses_list,
            bn_option_indexes,
            bn_actions,
            bn_rewards,
            bn_dones,
            bn_probs,
            m_pre_seq_hidden_states,
            m_pre_low_seq_hidden_states), \
        (key_indexes,
         key_padding_masks,
         key_obses_list,
         key_option_index,
         key_seq_hidden_state)


def _padding_key_batch(padding_len: int,
                       key_indexes: np.ndarray,
                       key_padding_masks: np.ndarray,
                       key_obses_list: list[np.ndarray],
                       key_option_index: np.ndarray,
                       key_seq_hidden_state: np.ndarray | None):
    bsz = key_indexes.shape[0]
    return [
        np.concatenate([-np.ones((bsz, padding_len), dtype=key_indexes.dtype), key_indexes], axis=1),
        np.concatenate([np.ones((bsz, padding_len), dtype=key_padding_masks.dtype), key_padding_masks], axis=1),
        [np.concatenate([np.zeros((bsz, padding_len, *o.shape[2:]), dtype=o.dtype), o], axis=1) for o in key_obses_list],
        np.concatenate([-np.ones((bsz, padding_len), dtype=key_option_index.dtype), key_option_index], axis=1),
        np.concatenate([np.zeros((bsz, padding_len, *key_seq_hidden_state.shape[2:]), dtype=key_seq_hidden_state.dtype), key_seq_hidden_state], axis=1)
        if key_seq_hidden_state is not None else None,
    ]


class BatchBuffer(BatchBuffer):
    _rest_key_batch = None

    def __init__(self,
                 burn_in_step: int,
                 n_step: int,
                 padding_action: np.ndarray,
                 batch_size: int):
        super().__init__(burn_in_step, n_step, padding_action, batch_size)

        self._key_batch_list = []

    def put_episode(self,
                    ep_indexes: np.ndarray,
                    ep_padding_masks: np.ndarray,
                    ep_obses_list: list[np.ndarray],
                    ep_option_indexes: np.ndarray,
                    ep_option_changed_indexes: np.ndarray,
                    ep_actions: np.ndarray,
                    ep_rewards: np.ndarray,
                    ep_dones: np.ndarray,
                    ep_probs: list[np.ndarray],
                    ep_pre_seq_hidden_states: np.ndarray,
                    ep_pre_low_seq_hidden_states: np.ndarray) -> None:
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
            ep_pre_seq_hidden_states: [1, ep_len, *seq_hidden_state_shape]
            ep_pre_low_seq_hidden_states: [1, ep_len, *low_seq_hidden_state_shape]
        """
        self._batch_list.clear()
        self._key_batch_list.clear()

        ori_batch, ori_key_trans = episode_to_batch(burn_in_step=self.burn_in_step,
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
                                                    l_pre_seq_hidden_states=ep_pre_seq_hidden_states,
                                                    l_pre_low_seq_hidden_states=ep_pre_low_seq_hidden_states)

        ori_batch = list(ori_batch)
        ori_key_trans = list(ori_key_trans)

        ori_key_batch = traverse_lists(ori_key_trans, lambda k: k.repeat(ori_batch[0].shape[0], axis=0))

        if self._rest_batch is not None:
            ori_batch = traverse_lists((self._rest_batch, ori_batch), lambda rb, b: np.concatenate([rb, b]))
            self._rest_batch = None

            max_key_len = max(self._rest_key_batch[0].shape[1], ori_key_batch[0].shape[1])
            self._rest_key_batch = _padding_key_batch(max_key_len - self._rest_key_batch[0].shape[1],
                                                      *self._rest_key_batch)
            ori_key_batch = _padding_key_batch(max_key_len - ori_key_batch[0].shape[1],
                                               *ori_key_batch)

            ori_key_batch = traverse_lists((self._rest_key_batch, ori_key_batch), lambda rb, b: np.concatenate([rb, b]))
            self._rest_key_batch = None

        ori_batch_size = ori_batch[0].shape[0]
        idx = np.random.permutation(ori_batch_size)
        ori_batch = traverse_lists(ori_batch, lambda b: b[idx])

        for i in range(math.ceil(ori_batch_size / self.batch_size)):
            b_i, b_j = i * self.batch_size, (i + 1) * self.batch_size

            batch = traverse_lists(ori_batch, lambda b: b[b_i:b_j, :])
            key_batch = traverse_lists(ori_key_batch, lambda b: b[b_i:b_j, :])

            if b_j > ori_batch_size:
                self._rest_batch = batch
                self._rest_key_batch = key_batch
            else:
                self._batch_list.append(batch)
                self._key_batch_list.append(key_batch)

    def get_batch(self) -> tuple[list[np.ndarray | list[np.ndarray]],
                                 list[np.ndarray | list[np.ndarray]]]:
        if len(self._batch_list) == 0 and self._rest_batch is not None:
            r = [self._rest_batch], [self._rest_key_batch]
            self._rest_batch, self._rest_key_batch = None, None
            return r

        return self._batch_list, self._key_batch_list

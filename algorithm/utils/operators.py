from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch


def squash_correction_log_prob(dist, x):
    return dist.log_prob(x) - torch.log(torch.maximum(1 - torch.square(torch.tanh(x)), torch.tensor(1e-2)))


def squash_correction_prob(dist, x):
    return torch.exp(dist.log_prob(x)) / (torch.maximum(1 - torch.square(torch.tanh(x)), torch.tensor(1e-2)))


def gen_pre_n_actions(n_actions, keep_last_action=False):
    if isinstance(n_actions, torch.Tensor):
        return torch.cat([
            torch.zeros_like(n_actions[:, 0:1, ...]),
            n_actions if keep_last_action else n_actions[:, :-1, ...]
        ], dim=1)
    else:
        return np.concatenate([
            np.zeros_like(n_actions[:, 0:1, ...]),
            n_actions if keep_last_action else n_actions[:, :-1, ...]
        ], axis=1)


def scale_h(x, epsilon=0.001):
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + epsilon * x


def scale_inverse_h(x, epsilon=0.001):
    t = 1 + 4 * epsilon * (torch.abs(x) + 1 + epsilon)
    return torch.sign(x) * ((torch.sqrt(t) - 1) / (2 * epsilon) - 1)


def format_global_step(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0

    if magnitude > 0:
        num = f'{num:.1f}'
    else:
        num = str(num)

    return '%s%s' % (num, ['', 'k', 'm', 'g', 't', 'p'][magnitude])


def traverse_lists(data: Union[Any, Tuple], process):
    if not isinstance(data, tuple):
        data = (data, )

    buffer = []
    for d in zip(*data):
        if isinstance(d[0], list):
            buffer.append(traverse_lists(d, process))
        elif d[0] is None:
            buffer.append(None)
        else:
            buffer.append(process(*d))

    return buffer


def episode_to_batch(bn: int,
                     episode_length: int,
                     l_indexes: np.ndarray,
                     l_padding_masks: np.ndarray,
                     l_obses_list: List[np.ndarray],
                     l_options: List[np.ndarray],
                     l_actions: np.ndarray,
                     l_rewards: np.ndarray,
                     next_obs_list: List[np.ndarray],
                     l_dones: np.ndarray,
                     l_probs: Optional[np.ndarray] = None,
                     l_seq_hidden_states: Optional[np.ndarray] = None):
    """
    Args:
        bn: int, burn_in_step + n_step
        episode_length: int, Indicates true episode_len, not MAX_EPISODE_LENGTH
        l_indexes: [1, episode_len]
        l_padding_masks: [1, episode_len]
        l_obses_list: list([1, episode_len, *obs_shapes_i], ...)
        l_actions: [1, episode_len, action_size]
        l_rewards: [1, episode_len]
        next_obs_list: list([1, *obs_shapes_i], ...)
        l_dones: [1, episode_len]
        l_probs: [1, episode_len]
        l_seq_hidden_states: [1, episode_len, *seq_hidden_state_shape]

    Returns:
        bn_indexes: [episode_len - bn + 1, bn]
        bn_padding_masks: [episode_len - bn + 1, bn]
        bn_obses_list: list([episode_len - bn + 1, bn, *obs_shapes_i], ...)
        bn_actions: [episode_len - bn + 1, bn, action_size]
        bn_rewards: [episode_len - bn + 1, bn]
        next_obs_list: list([episode_len - bn + 1, *obs_shapes_i], ...)
        bn_dones: [episode_len - bn + 1, bn]
        bn_probs: [episode_len - bn + 1, bn]
        f_seq_hidden_states: [episode_len - bn + 1, 1, *seq_hidden_state_shape]
    """
    bn_indexes = np.concatenate([l_indexes[:, i:i + bn]
                                for i in range(episode_length - bn + 1)], axis=0)
    bn_padding_masks = np.concatenate([l_padding_masks[:, i:i + bn]
                                       for i in range(episode_length - bn + 1)], axis=0)
    tmp_bn_obses_list = [None] * len(l_obses_list)
    for j, l_obses in enumerate(l_obses_list):
        tmp_bn_obses_list[j] = np.concatenate([l_obses[:, i:i + bn]
                                              for i in range(episode_length - bn + 1)], axis=0)
    bn_options = np.concatenate([l_options[:, i:i + bn]
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
        f_seq_hidden_states = np.concatenate([l_seq_hidden_states[:, i:i + 1]
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
            f_seq_hidden_states if l_seq_hidden_states is not None else None]

from typing import Any, List, Optional, Tuple

import numpy as np
import torch


def get_last_false_indexes(x: torch.Tensor, dim: int, keepdim: bool = False):
    flipped = torch.flip(x * torch.ones(1, dtype=torch.uint8, device=x.device), dims=[dim])
    return x.shape[dim] - flipped.argmin(1, keepdim=keepdim) - 1


def squash_correction_log_prob(dist: torch.distributions.Distribution,
                               x: torch.Tensor) -> torch.Tensor:
    return dist.log_prob(x) - torch.log(torch.maximum(1 - torch.square(torch.tanh(x)), torch.tensor(1e-2)))


def squash_correction_prob(dist: torch.distributions.Distribution,
                           x: torch.Tensor) -> torch.Tensor:
    return torch.exp(dist.log_prob(x)) / (torch.maximum(1 - torch.square(torch.tanh(x)), torch.tensor(1e-2)))


def sum_log_prob(log_prob: torch.Tensor, keepdim=False):
    log_prob[log_prob == torch.inf] = 0.
    return log_prob.sum(-1, keepdim=keepdim)


def prod_prob(prob: torch.Tensor, keepdim=False):
    prob[prob == torch.inf] = 1.
    return prob.prod(-1, keepdim=keepdim)


def sum_entropy(entropy: torch.Tensor):
    entropy[entropy == torch.inf] = 0.
    return entropy.sum(-1)


def gen_pre_n_actions(n_actions: torch.Tensor | np.ndarray,
                      keep_last_action=False) -> torch.Tensor | np.ndarray:
    if isinstance(n_actions, torch.Tensor):
        if n_actions.shape[1] == 0 and keep_last_action:
            return torch.zeros((n_actions.shape[0], 1, *n_actions.shape[2:]),
                               dtype=n_actions.dtype,
                               device=n_actions.device)

        return torch.cat([
            torch.zeros_like(n_actions[:, 0:1, ...]),
            n_actions if keep_last_action else n_actions[:, :-1, ...]
        ], dim=1)
    else:
        if n_actions.shape[1] == 0 and keep_last_action:
            return np.zeros((n_actions.shape[0], 1, *n_actions.shape[2:]),
                            dtype=n_actions.dtype)

        return np.concatenate([
            np.zeros_like(n_actions[:, 0:1, ...]),
            n_actions if keep_last_action else n_actions[:, :-1, ...]
        ], axis=1)


def scale_h(x: torch.Tensor, epsilon=0.001) -> torch.Tensor:
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + epsilon * x


def scale_inverse_h(x: torch.Tensor, epsilon=0.001) -> torch.Tensor:
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


def traverse_lists(data: Any | Tuple, process) -> List:
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


def episode_to_batch(burn_in_step: int,
                     n_step: int,
                     padding_action: np.ndarray,
                     l_indexes: np.ndarray,
                     l_padding_masks: np.ndarray,
                     l_obses_list: List[np.ndarray],
                     l_actions: np.ndarray,
                     l_rewards: np.ndarray,
                     l_dones: np.ndarray,
                     l_probs: np.ndarray,
                     l_pre_seq_hidden_states: np.ndarray) -> List[np.ndarray | List[np.ndarray]]:
    """
    Args:
        burn_in_step: int
        n_step: int
        padding_action (np): [action_size, ]  The discrete padding actions cannot be all zeros
        l_indexes (np.int32): [1, ep_len]
        l_padding_masks (bool): [1, ep_len]
        l_obses_list: list([1, ep_len, *obs_shapes_i], ...)
        l_actions: [1, ep_len, action_size]
        l_rewards: [1, ep_len]
        l_dones (bool): [1, ep_len]
        l_probs: [1, ep_len, action_size]
        l_pre_seq_hidden_states: [1, ep_len, *seq_hidden_state_shape]

    Returns:
        bn_indexes (np.int32): [ep_len - bn + 1, bn]
        bn_padding_masks (bool): [ep_len - bn + 1, bn]
        m_obses_list: list([ep_len - bn + 1 + 1, bn, *obs_shapes_i], ...)
        bn_actions: [ep_len - bn + 1, bn, action_size]
        bn_rewards: [ep_len - bn + 1, bn]
        bn_dones (bool): [ep_len - bn + 1, bn]
        bn_probs: [ep_len - bn + 1, bn, action_size]
        m_pre_seq_hidden_states: [ep_len - bn + 1 + 1, bn, *seq_hidden_state_shape]
    """

    bn = burn_in_step + n_step
    padding_action = padding_action.reshape(1, 1, -1)
    ep_len = l_indexes.shape[1]

    # Padding burn_in_step and n_step
    l_indexes = np.concatenate([np.full((1, burn_in_step), -1, dtype=l_indexes.dtype),
                                l_indexes,
                                np.full((1, n_step - 1), -1, dtype=l_indexes.dtype)], axis=1)
    l_padding_masks = np.concatenate([np.ones((1, burn_in_step), dtype=bool),
                                     l_padding_masks,
                                     np.ones((1, n_step - 1), dtype=bool)], axis=1)
    for j, l_obses in enumerate(l_obses_list):
        l_obses_list[j] = np.concatenate([np.zeros((1, burn_in_step, *l_obses.shape[2:]), dtype=l_obses.dtype),
                                          l_obses,
                                          np.zeros((1, n_step - 1, *l_obses.shape[2:]), dtype=l_obses.dtype)], axis=1)
    l_actions = np.concatenate([padding_action.repeat(burn_in_step, 1),
                                l_actions,
                                padding_action.repeat(n_step - 1, 1)], axis=1)
    l_rewards = np.concatenate([np.zeros((1, burn_in_step), dtype=l_rewards.dtype),
                                l_rewards,
                                np.zeros((1, n_step - 1), dtype=l_rewards.dtype)], axis=1)
    l_dones = np.concatenate([np.ones((1, burn_in_step), dtype=l_dones.dtype),
                              l_dones,
                              np.ones((1, n_step - 1), dtype=l_dones.dtype)], axis=1)
    l_probs = np.concatenate([np.ones((1, burn_in_step, *l_probs.shape[2:]), dtype=l_probs.dtype),
                              l_probs,
                              np.ones((1, n_step - 1, *l_probs.shape[2:]), dtype=l_probs.dtype)], axis=1)
    l_pre_seq_hidden_states = np.concatenate([np.zeros((1, burn_in_step, *l_pre_seq_hidden_states.shape[2:]), dtype=l_pre_seq_hidden_states.dtype),
                                              l_pre_seq_hidden_states,
                                              np.zeros((1, n_step - 1, *l_pre_seq_hidden_states.shape[2:]), dtype=l_pre_seq_hidden_states.dtype)], axis=1)

    # Generate batch
    bn_indexes = np.concatenate([l_indexes[:, i:i + bn]
                                for i in range(ep_len - 1)], axis=0)
    bn_padding_masks = np.concatenate([l_padding_masks[:, i:i + bn]
                                       for i in range(ep_len - 1)], axis=0)
    tmp_m_obses_list = [None] * len(l_obses_list)
    for j, l_obses in enumerate(l_obses_list):
        tmp_m_obses_list[j] = np.concatenate([l_obses[:, i:i + bn + 1]
                                              for i in range(ep_len - 1)], axis=0)
    bn_actions = np.concatenate([l_actions[:, i:i + bn]
                                for i in range(ep_len - 1)], axis=0)
    bn_rewards = np.concatenate([l_rewards[:, i:i + bn]
                                for i in range(ep_len - 1)], axis=0)
    bn_dones = np.concatenate([l_dones[:, i:i + bn]
                              for i in range(ep_len - 1)], axis=0)
    bn_probs = np.concatenate([l_probs[:, i:i + bn]
                               for i in range(ep_len - 1)], axis=0)
    m_pre_seq_hidden_states = np.concatenate([l_pre_seq_hidden_states[:, i:i + bn + 1]
                                              for i in range(ep_len - 1)], axis=0)

    return [bn_indexes,
            bn_padding_masks,
            tmp_m_obses_list,
            bn_actions,
            bn_rewards,
            bn_dones,
            bn_probs,
            m_pre_seq_hidden_states]

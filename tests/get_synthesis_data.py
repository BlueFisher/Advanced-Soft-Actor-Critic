import itertools
import random

import numpy as np


def get_product(possible_dict):
    for values in itertools.product(*possible_dict.values()):
        yield {
            k: values[i]
            for i, k in enumerate(possible_dict.keys())
        }


def get_action(batch, seq_len, d_action_size, c_action_size):
    if seq_len is None:
        batch = (batch, )
    else:
        batch = (batch, seq_len)

    if d_action_size:
        d_action = np.eye(d_action_size)[np.random.rand(*batch, d_action_size).argmax(axis=-1)]
    else:
        d_action = np.empty((*batch, 0))

    if c_action_size:
        c_action = np.random.rand(*batch, c_action_size)
        c_action = np.clip(c_action, -.99, 99.)
    else:
        c_action = np.empty((*batch, 0))

    return np.concatenate([d_action, c_action], axis=-1).astype(np.float32)


def gen_batch_obs(obs_shapes, batch=10):
    return (
        [np.random.randn(batch, *obs_shape).astype(np.float32) for obs_shape in obs_shapes],
    )


def gen_batch_obs_for_rnn(obs_shapes, d_action_size, c_action_size, seq_hidden_state_shape, batch=10):
    return (
        [np.random.randn(batch, *obs_shape).astype(np.float32) for obs_shape in obs_shapes],
        get_action(batch, None, d_action_size, c_action_size),
        np.random.randn(batch, *seq_hidden_state_shape).astype(np.float32)
    )


def gen_batch_obs_for_attn(obs_shapes, d_action_size, c_action_size, seq_hidden_state_shape, batch=10):
    episode_len = random.randint(1, 100)

    # ep_indexes
    # ep_padding_masks
    # ep_obses_list
    # ep_actions
    # ep_attn_hidden_states
    return (
        np.expand_dims(np.arange(episode_len), 0).repeat(batch, 0),
        np.random.randint(0, 2, size=(batch, episode_len), dtype=bool),
        [np.random.randn(batch, episode_len, *obs_shape).astype(np.float32) for obs_shape in obs_shapes],
        get_action(batch, episode_len, d_action_size, c_action_size),
        np.random.randn(batch, episode_len, *seq_hidden_state_shape).astype(np.float32)
    )


def gen_batch_trans(obs_shapes, d_action_size, c_action_size, n, seq_hidden_state_shape=None, batch=10):
    # n_obses_list, n_actions, n_rewards, next_obs_list, n_dones, n_mu_probs,
    vanilla_batch_trans = [
        [np.random.randn(batch, n, *obs_shape).astype(np.float32) for obs_shape in obs_shapes],
        get_action(batch, n, d_action_size, c_action_size),
        np.random.randn(batch, n).astype(np.float32),
        [np.random.randn(batch, *obs_shape).astype(np.float32) for obs_shape in obs_shapes],
        np.random.randint(0, 2, size=(batch, n), dtype=bool),
        np.random.randn(batch, n).astype(np.float32),
    ]

    if seq_hidden_state_shape:
        return vanilla_batch_trans + [
            np.random.randn(batch, *seq_hidden_state_shape).astype(np.float32)
        ]
    else:
        return vanilla_batch_trans


def gen_episode_trans(obs_shapes, d_action_size, c_action_size, seq_hidden_state_shape=None, episode_len=None):
    if episode_len is None:
        episode_len = random.randint(1, 100)

    # ep_indexes,
    # ep_padding_masks,
    # ep_obses_list,
    # ep_actions,
    # ep_rewards,
    # next_obs_list,
    # ep_dones,
    # ep_probs
    vanilla_episode_trans = [
        np.expand_dims(np.arange(episode_len), 0),
        np.random.randint(0, 2, size=(1, episode_len), dtype=bool),
        [np.random.randn(1, episode_len, *obs_shape).astype(np.float32) for obs_shape in obs_shapes],
        get_action(1, episode_len, d_action_size, c_action_size),
        np.random.randn(1, episode_len).astype(np.float32),
        [np.random.randn(1, *obs_shape).astype(np.float32) for obs_shape in obs_shapes],
        np.random.randint(0, 2, size=(1, episode_len), dtype=bool),
        np.random.randn(1, episode_len).astype(np.float32),
    ]
    if seq_hidden_state_shape:
        return vanilla_episode_trans + [
            np.random.randn(1, episode_len, *seq_hidden_state_shape).astype(np.float32)
        ]
    else:
        return vanilla_episode_trans

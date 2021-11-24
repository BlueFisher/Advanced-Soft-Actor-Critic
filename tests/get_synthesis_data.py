import itertools
import random

import numpy as np


def get_product(possible_dict):
    for values in itertools.product(*possible_dict.values()):
        yield {
            k: values[i]
            for i, k in enumerate(possible_dict.keys())
        }


def gen_batch_obs(obs_shapes, rnn_shape=None, d_action_size=None, c_action_size=None, batch=10):
    obs_list = [np.random.randn(batch, *obs_shape).astype(np.float32) for obs_shape in obs_shapes]

    if rnn_shape:
        if d_action_size:
            d_actoin = np.eye(d_action_size)[np.random.rand(batch, d_action_size).argmax(axis=-1)]
        else:
            d_actoin = np.empty((batch, 0))

        if c_action_size:
            c_action = np.random.rand(batch, c_action_size)
            c_action = np.clip(c_action, -.99, 99.)
        else:
            c_action = np.empty((batch, 0))

        action = np.concatenate([d_actoin, c_action], axis=-1).astype(np.float32)

        rnn_state = np.random.randn(batch, *rnn_shape).astype(np.float32)

        return obs_list, action, rnn_state
    else:
        return obs_list


def gen_batch_trans(obs_shapes, d_action_size, c_action_size, n, rnn_shape=None, batch=10):
    if d_action_size:
        d_actoin = np.eye(d_action_size)[np.random.rand(n * batch, d_action_size).argmax(axis=-1)]
        d_actoin = d_actoin.reshape(batch, n, d_action_size)
    else:
        d_actoin = np.empty((batch, n, 0))

    if c_action_size:
        c_action = np.random.rand(batch, n, c_action_size)
        c_action = np.clip(c_action, -.99, 99.)
    else:
        c_action = np.empty((batch, n, 0))
    n_actions = np.concatenate([d_actoin, c_action], axis=-1).astype(np.float32)

    # n_obses_list, n_actions, n_rewards, next_obs_list, n_dones, n_mu_probs,
    vanilla_batch_trans = [
        [np.random.randn(batch, n, *obs_shape).astype(np.float32) for obs_shape in obs_shapes],
        n_actions,
        np.random.randn(batch, n).astype(np.float32),
        [np.random.randn(batch, *obs_shape).astype(np.float32) for obs_shape in obs_shapes],
        np.random.randn(batch, n).astype(np.float32),
        np.random.randn(batch, n).astype(np.float32),
    ]

    if rnn_shape:
        return vanilla_batch_trans + [
            np.random.randn(batch, *rnn_shape).astype(np.float32)
        ]
    else:
        return vanilla_batch_trans


def gen_episode_trans(obs_shapes, d_action_size, c_action_size, rnn_shape=None, episode_len=None):
    if episode_len is None:
        episode_len = random.randint(1, 100)

    if d_action_size:
        d_actoin = np.eye(d_action_size)[np.random.rand(episode_len, d_action_size).argmax(axis=-1)]
        d_actoin = d_actoin.reshape(1, -1, d_action_size)
    else:
        d_actoin = np.empty((1, episode_len, 0))

    if c_action_size:
        c_action = np.random.rand(1, episode_len, c_action_size)
        c_action = np.clip(c_action, -.99, 99.)
    else:
        c_action = np.empty((1, episode_len, 0))
    n_actions = np.concatenate([d_actoin, c_action], axis=-1).astype(np.float32)

    # n_obses_list, n_actions, n_rewards, next_obs_list, n_dones
    vanilla_episode_trans = [
        [np.random.randn(1, episode_len, *obs_shape).astype(np.float32) for obs_shape in obs_shapes],
        n_actions,
        np.random.randn(1, episode_len).astype(np.float32),
        [np.random.randn(1, *obs_shape).astype(np.float32) for obs_shape in obs_shapes],
        np.random.randn(1, episode_len).astype(np.float32)
    ]
    if rnn_shape:
        return vanilla_episode_trans + [
            np.random.randn(1, episode_len, *rnn_shape).astype(np.float32)
        ]
    else:
        return vanilla_episode_trans

import itertools
import random

import numpy as np


def get_product(possible_dict):
    for values in itertools.product(*possible_dict.values()):
        yield {
            k: values[i]
            for i, k in enumerate(possible_dict.keys())
        }


def get_action(batch, seq_len, d_action_sizes, c_action_size):
    if seq_len is None:
        batch = (batch, )
    else:
        batch = (batch, seq_len)

    if d_action_sizes:
        d_action_list = [np.random.randint(0, d_action_size, size=batch)
                         for d_action_size in d_action_sizes]
        d_action_list = [np.eye(d_action_size, dtype=np.int32)[d_action]
                         for d_action, d_action_size in zip(d_action_list, d_action_sizes)]
        d_action = np.concatenate(d_action_list, axis=-1)
    else:
        d_action = np.zeros((*batch, 0))

    if c_action_size:
        c_action = np.random.rand(*batch, c_action_size)
        c_action = np.clip(c_action, -.99, 99.)
    else:
        c_action = np.zeros((*batch, 0))

    return np.concatenate([d_action, c_action], axis=-1).astype(np.float32)


def gen_batch_obs(obs_shapes, batch=10):
    return {
        'obs_list': [np.random.randn(batch, *obs_shape).astype(np.float32) for obs_shape in obs_shapes],
    }


def gen_batch_obs_for_rnn(obs_shapes, d_action_sizes, c_action_size, seq_hidden_state_shape, batch=10):
    return {
        'obs_list': [np.random.randn(batch, *obs_shape).astype(np.float32) for obs_shape in obs_shapes],
        'pre_action': get_action(batch, None, d_action_sizes, c_action_size),
        'rnn_state': np.random.randn(batch, *seq_hidden_state_shape).astype(np.float32)
    }


def gen_batch_obs_for_attn(obs_shapes, d_action_sizes, c_action_size, seq_hidden_state_shape, batch=10):
    episode_len = random.randint(1, 100)

    return {
        'ep_indexes': np.expand_dims(np.arange(episode_len), 0).repeat(batch, 0),
        'ep_padding_masks': np.random.randint(0, 2, size=(batch, episode_len), dtype=bool),
        'ep_obses_list': [np.random.randn(batch, episode_len, *obs_shape).astype(np.float32) for obs_shape in obs_shapes],
        'ep_pre_actions': get_action(batch, episode_len, d_action_sizes, c_action_size),
        'ep_attn_states': np.random.randn(batch, episode_len, *seq_hidden_state_shape).astype(np.float32)
    }


def gen_batch_oc_obs(obs_shapes, num_options=2, batch=10):
    return {
        **gen_batch_obs(obs_shapes, batch),
        'pre_option_index': np.random.randint(0, num_options, size=(batch, ), dtype=np.int8)
    }


def gen_batch_oc_obs_for_rnn(obs_shapes, d_action_sizes, c_action_size,
                             seq_hidden_state_shape, low_seq_hidden_state_shape,
                             num_options=2,
                             batch=10):
    return {
        'obs_list': [np.random.randn(batch, *obs_shape).astype(np.float32) for obs_shape in obs_shapes],
        'pre_option_index': np.random.randint(0, num_options, size=(batch, ), dtype=np.int8),
        'pre_action': get_action(batch, None, d_action_sizes, c_action_size),
        'rnn_state': np.random.randn(batch, *seq_hidden_state_shape).astype(np.float32),
        'low_rnn_state': np.random.randn(batch, *low_seq_hidden_state_shape).astype(np.float32)
    }


def gen_batch_oc_obs_for_attn(obs_shapes, d_action_sizes, c_action_size,
                              seq_hidden_state_shape, low_seq_hidden_state_shape,
                              num_options=2,
                              batch=10):
    episode_len = random.randint(1, 100)

    return {
        'ep_indexes': np.expand_dims(np.arange(episode_len), 0).repeat(batch, 0),
        'ep_padding_masks': np.random.randint(0, 2, size=(batch, episode_len), dtype=bool),
        'ep_obses_list': [np.random.randn(batch, episode_len, *obs_shape).astype(np.float32) for obs_shape in obs_shapes],
        'ep_pre_actions': get_action(batch, episode_len, d_action_sizes, c_action_size),
        'ep_attn_states': np.random.randn(batch, episode_len, *seq_hidden_state_shape).astype(np.float32),

        'pre_option_index': np.random.randint(0, num_options, size=(batch, ), dtype=np.int8),
        'low_rnn_state': np.random.randn(batch, *low_seq_hidden_state_shape).astype(np.float32)
    }


def gen_episode_trans(obs_shapes, d_action_sizes, c_action_size, seq_hidden_state_shape=None, episode_len=None):
    if episode_len is None:
        episode_len = random.randint(1, 100)

    vanilla_episode_trans = {
        'l_indexes': np.expand_dims(np.arange(episode_len), 0),
        'l_padding_masks': np.random.randint(0, 2, size=(1, episode_len), dtype=bool),
        'l_obses_list': [np.random.randn(1, episode_len, *obs_shape).astype(np.float32) for obs_shape in obs_shapes],
        'l_actions': get_action(1, episode_len, d_action_sizes, c_action_size),
        'l_rewards': np.random.randn(1, episode_len).astype(np.float32),
        'next_obs_list': [np.random.randn(1, *obs_shape).astype(np.float32) for obs_shape in obs_shapes],
        'l_dones': np.random.randint(0, 2, size=(1, episode_len), dtype=bool),
        'l_probs': np.random.randn(1, episode_len).astype(np.float32),
    }
    if seq_hidden_state_shape:
        return {
            **vanilla_episode_trans,
            'l_seq_hidden_states': np.random.randn(1, episode_len, *seq_hidden_state_shape).astype(np.float32)
        }
    else:
        return vanilla_episode_trans


def gen_episode_oc_trans(obs_shapes, d_action_sizes, c_action_size,
                         num_options=2,
                         seq_hidden_state_shape=None,
                         low_seq_hidden_state_shape=None,
                         episode_len=None):
    if episode_len is None:
        episode_len = random.randint(1, 100)

    vanilla_episode_trans = gen_episode_trans(obs_shapes=obs_shapes,
                                              d_action_sizes=d_action_sizes,
                                              c_action_size=c_action_size,
                                              seq_hidden_state_shape=seq_hidden_state_shape,
                                              episode_len=episode_len)

    vanilla_episode_trans = {
        **vanilla_episode_trans,
        'l_option_indexes': np.random.randint(0, num_options, size=(1, episode_len), dtype=np.int64)
    }

    if low_seq_hidden_state_shape:
        return {
            **vanilla_episode_trans,
            'l_low_seq_hidden_states': np.random.randn(1, episode_len, *low_seq_hidden_state_shape).astype(np.float32)
        }
    else:
        return vanilla_episode_trans

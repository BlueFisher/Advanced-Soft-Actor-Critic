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


def gen_batch_obs(obs_shapes, d_action_sizes, c_action_size, seq_hidden_state_shape, batch=10):
    return {
        'obs_list': [np.random.randn(batch, *obs_shape).astype(np.float32) for obs_shape in obs_shapes],
        'pre_action': get_action(batch, None, d_action_sizes, c_action_size),
        'pre_seq_hidden_state': np.random.randn(batch, *seq_hidden_state_shape).astype(np.float32)
    }


def gen_batch_obs_for_attn(obs_shapes, d_action_sizes, c_action_size, seq_hidden_state_shape, batch=10):
    episode_len = random.randint(1, 100)

    return {
        'ep_indexes': np.expand_dims(np.arange(episode_len, dtype=np.int32), 0).repeat(batch, 0),
        'ep_padding_masks': np.zeros((batch, episode_len), dtype=bool),
        'ep_obses_list': [np.random.randn(batch, episode_len, *obs_shape).astype(np.float32) for obs_shape in obs_shapes],
        'ep_pre_actions': get_action(batch, episode_len, d_action_sizes, c_action_size),
        'ep_pre_attn_states': np.random.randn(batch, episode_len, *seq_hidden_state_shape).astype(np.float32)
    }


def gen_batch_oc_obs(obs_shapes, d_action_sizes, c_action_size,
                     seq_hidden_state_shape, low_seq_hidden_state_shape,
                     num_options=2,
                     batch=10):
    return {
        **gen_batch_obs(obs_shapes, d_action_sizes, c_action_size, seq_hidden_state_shape, batch),
        'pre_option_index': np.random.randint(0, num_options, size=(batch, ), dtype=np.int8),
        'pre_low_seq_hidden_state': np.random.randn(batch, *low_seq_hidden_state_shape).astype(np.float32),
        'pre_termination': np.random.rand(batch, ).astype(np.float32)
    }


def gen_batch_oc_obs_for_attn(obs_shapes, d_action_sizes, c_action_size,
                              seq_hidden_state_shape, low_seq_hidden_state_shape,
                              num_options=2,
                              batch=10):
    episode_len = random.randint(1, 100)

    gen_batch = {
        'ep_indexes': np.expand_dims(np.arange(episode_len, dtype=np.int32), 0).repeat(batch, 0),
        'ep_padding_masks': np.zeros((batch, episode_len), dtype=bool),
        'ep_obses_list': [np.random.randn(batch, episode_len, *obs_shape).astype(np.float32) for obs_shape in obs_shapes],
        'ep_pre_actions': get_action(batch, episode_len, d_action_sizes, c_action_size),
        'ep_pre_attn_states': np.random.randn(batch, episode_len, *seq_hidden_state_shape).astype(np.float32),

        'pre_option_index': np.random.randint(0, num_options, size=(batch, ), dtype=np.int8),
        'pre_low_seq_hidden_state': np.random.randn(batch, *low_seq_hidden_state_shape).astype(np.float32),
        'pre_termination': np.random.rand(batch, ).astype(np.float32)
    }

    return gen_batch


def gen_batch_oc_obs_for_dilated_attn(obs_shapes, d_action_sizes, c_action_size,
                                      seq_hidden_state_shape, low_seq_hidden_state_shape,
                                      num_options=2,
                                      batch=10):
    key_len = random.randint(1, 20)

    gen_batch = {
        'key_indexes': np.expand_dims(np.arange(key_len, dtype=np.int32), 0).repeat(batch, 0),
        'key_padding_masks': np.zeros((batch, key_len), dtype=bool),
        'key_obses_list': [np.random.randn(batch, key_len, *obs_shape).astype(np.float32) for obs_shape in obs_shapes],
        'key_option_indexes': np.random.randint(0, num_options, size=(batch, key_len), dtype=np.int8),
        'key_attn_states': np.random.randn(batch, key_len, *seq_hidden_state_shape).astype(np.float32),

        'pre_option_index': np.random.randint(0, num_options, size=(batch, ), dtype=np.int32),
        'pre_action': get_action(batch, None, d_action_sizes, c_action_size),
    }

    if low_seq_hidden_state_shape is not None:
        gen_batch['low_rnn_state'] = np.random.randn(batch, *low_seq_hidden_state_shape).astype(np.float32)

    return gen_batch


def gen_episode_trans(obs_shapes, d_action_sizes, c_action_size, seq_hidden_state_shape, episode_len=None):
    if episode_len is None:
        episode_len = random.randint(1, 100)

    return {
        'ep_indexes': np.expand_dims(np.arange(episode_len, dtype=np.int32), 0),
        'ep_obses_list': [np.random.randn(1, episode_len, *obs_shape).astype(np.float32) for obs_shape in obs_shapes],
        'ep_actions': get_action(1, episode_len, d_action_sizes, c_action_size),
        'ep_rewards': np.random.randn(1, episode_len).astype(np.float32),
        'ep_dones': np.random.randint(0, 2, size=(1, episode_len), dtype=bool),
        'ep_probs': np.random.rand(1, episode_len, sum(d_action_sizes) + c_action_size).astype(np.float32),
        'ep_pre_seq_hidden_states': np.random.randn(1, episode_len, *seq_hidden_state_shape).astype(np.float32)
    }


def gen_episode_oc_trans(obs_shapes, d_action_sizes, c_action_size,
                         seq_hidden_state_shape,
                         low_seq_hidden_state_shape,
                         num_options=2,
                         episode_len=None):
    if episode_len is None:
        episode_len = random.randint(1, 100)

    vanilla_episode_trans = gen_episode_trans(obs_shapes=obs_shapes,
                                              d_action_sizes=d_action_sizes,
                                              c_action_size=c_action_size,
                                              seq_hidden_state_shape=seq_hidden_state_shape,
                                              episode_len=episode_len)

    l_option_indexes = np.random.randint(0, num_options, size=(1, episode_len), dtype=np.int8)
    l_option_changed_indexes = np.zeros((1, episode_len), dtype=np.int32)
    for i in range(1, episode_len):
        mask = l_option_indexes[:, i] == l_option_indexes[:, i - 1]
        l_option_changed_indexes[mask, i] = l_option_changed_indexes[mask, i - 1]
        l_option_changed_indexes[~mask, i] = i

    return {
        **vanilla_episode_trans,
        'ep_option_indexes': l_option_indexes,
        'ep_option_changed_indexes': l_option_changed_indexes,
        'ep_pre_low_seq_hidden_states': np.random.randn(1, episode_len, *low_seq_hidden_state_shape).astype(np.float32)
    }

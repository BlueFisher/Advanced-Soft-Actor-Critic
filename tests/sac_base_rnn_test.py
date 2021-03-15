import threading
import unittest
import sys

import numpy as np
from numpy import random

sys.path.append('..')

from algorithm.sac_base import SAC_Base


BATCH = 10
EPISODE_LEN = 200


def gen_batch_obs():
    return [np.random.randn(BATCH, 10).astype(np.float32),
            np.random.randn(BATCH, 6).astype(np.float32),
            np.random.randn(BATCH, 30, 30, 3).astype(np.float32)]


def gen_episode_trans(action_dim):
    ep_len = random.randint(20, EPISODE_LEN)
    return ([np.random.randn(1, ep_len, 30, 30, 3).astype(np.float32),
             np.random.randn(1, ep_len, 6).astype(np.float32)],

            np.random.randn(1, ep_len, action_dim).astype(np.float32),

            np.random.randn(1, ep_len).astype(np.float32),

            [np.random.randn(1, 30, 30, 3).astype(np.float32),
             np.random.randn(1, 6).astype(np.float32)],

            np.random.randn(1, ep_len).astype(np.float32),

            np.random.randn(1, ep_len).astype(np.float32),

            np.random.randn(1, ep_len, 64).astype(np.float32))


class TestRNNMethods(unittest.TestCase):
    def test_c_action(self):
        from . import nn_rnn

        sac = SAC_Base(
            obs_dims=[(30, 30, 3), (6,)],
            d_action_dim=0,
            c_action_dim=10,
            model_abs_dir='tests/model/models_test_c_rnn_action',
            model=nn_rnn,
            burn_in_step=10,
            n_step=3,
            use_rnn=True,
            discrete_dqn_like=False,
            use_priority=True,
            use_n_step_is=True,
            use_prediction=False,
            use_normalization=False
        )

        # for _ in range(1024):
        #     sac.fill_replay_buffer(*gen_episode_trans(10))

        # sac.train()

        def t(i):
            while True:
                print('start', i)
                t=sac.get_episode_td_error(*gen_episode_trans(10))
                print('end', i)
        t(0)
        # for i in range(5):
        #     tt = threading.Thread(target=t, args=[i])
        #     tt.start()

        # tt.join()

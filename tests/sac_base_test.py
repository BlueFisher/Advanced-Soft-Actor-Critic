import unittest
import sys

import numpy as np

sys.path.append('..')

from algorithm.sac_base import SAC_Base


BATCH = 10
EPISODE_LEN = 50


def gen_batch_obs():
    return [np.random.randn(BATCH, 10).astype(np.float32),
            np.random.randn(BATCH, 8).astype(np.float32),
            np.random.randn(BATCH, 30, 30, 3).astype(np.float32)]


def gen_episode_trans(action_dim):
    return ([np.random.randn(1, EPISODE_LEN, 10).astype(np.float32),
             np.random.randn(1, EPISODE_LEN, 8).astype(np.float32),
             np.random.randn(1, EPISODE_LEN, 30, 30, 3).astype(np.float32)],

            np.random.randn(1, EPISODE_LEN, action_dim).astype(np.float32),

            np.random.randn(1, EPISODE_LEN).astype(np.float32),

            [np.random.randn(1, 10).astype(np.float32),
             np.random.randn(1, 8).astype(np.float32),
             np.random.randn(1, 30, 30, 3).astype(np.float32)],

            np.random.randn(1, EPISODE_LEN).astype(np.float32))


class TestStringMethods(unittest.TestCase):
    def test_d_c_action(self):
        from . import nn_vanilla

        sac = SAC_Base(
            obs_dims=[(10,), (8,), (30, 30, 3)],
            d_action_dim=10,
            c_action_dim=4,
            model_abs_dir='tests/model/models_test_d_c_action',
            model=nn_vanilla,
            burn_in_step=0,
            n_step=1,
            use_rnn=False,
            discrete_dqn_like=False,
            use_priority=True,
            use_n_step_is=True,
            use_prediction=False,
            use_normalization=False
        )

        for _ in range(1024):
            sac.choose_action(gen_batch_obs())
            sac.fill_replay_buffer(*gen_episode_trans(14))
            sac.train()

    def test_d_action(self):
        from . import nn_vanilla

        sac = SAC_Base(
            obs_dims=[(10,), (8,), (30, 30, 3)],
            d_action_dim=10,
            c_action_dim=0,
            model_abs_dir='tests/model/models_test_d_action',
            model=nn_vanilla,
            burn_in_step=0,
            n_step=1,
            use_rnn=False,
            discrete_dqn_like=False,
            use_priority=True,
            use_n_step_is=True,
            use_prediction=False,
            use_normalization=False
        )

        for _ in range(1024):
            sac.choose_action(gen_batch_obs())
            sac.fill_replay_buffer(*gen_episode_trans(10))
            sac.train()

    def test_d_dqn_like_action(self):
        from . import nn_vanilla

        sac = SAC_Base(
            obs_dims=[(10,), (8,), (30, 30, 3)],
            d_action_dim=10,
            c_action_dim=0,
            model_abs_dir='tests/model/models_test_d_action',
            model=nn_vanilla,
            burn_in_step=0,
            n_step=1,
            use_rnn=False,
            discrete_dqn_like=True,
            use_priority=True,
            use_n_step_is=True,
            use_prediction=False,
            use_normalization=False
        )

        for _ in range(1024):
            sac.choose_action(gen_batch_obs())
            sac.fill_replay_buffer(*gen_episode_trans(10))
            sac.train()

    def test_c_action(self):
        from . import nn_vanilla

        sac = SAC_Base(
            obs_dims=[(10,), (8,), (30, 30, 3)],
            d_action_dim=0,
            c_action_dim=10,
            model_abs_dir='tests/model/models_test_c_action',
            model=nn_vanilla,
            burn_in_step=0,
            n_step=1,
            use_rnn=False,
            discrete_dqn_like=False,
            use_priority=True,
            use_n_step_is=True,
            use_prediction=False,
            use_normalization=False
        )

        for _ in range(1024):
            sac.choose_action(gen_batch_obs())
            sac.fill_replay_buffer(*gen_episode_trans(10))
            sac.train()

    def test_d_c_rnn_action(self):
        from . import nn_vanilla

        sac = SAC_Base(
            obs_dims=[(10,), (8,), (30, 30, 3)],
            d_action_dim=10,
            c_action_dim=4,
            model_abs_dir='tests/model/models_test_d_c_action',
            model=nn_vanilla,
            burn_in_step=0,
            n_step=1,
            use_rnn=False,
            discrete_dqn_like=False,
            use_priority=True,
            use_n_step_is=True,
            use_prediction=False,
            use_normalization=False
        )

        for _ in range(1024):
            sac.choose_action(gen_batch_obs())
            sac.fill_replay_buffer(*gen_episode_trans(14))
            sac.train()
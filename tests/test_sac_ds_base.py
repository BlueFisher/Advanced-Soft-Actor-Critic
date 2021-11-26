import unittest

from ds.sac_ds_base import SAC_DS_Base
from tests.get_synthesis_data import *

OBS_SHAPES = [(10,), (30, 30, 3)]
D_ACTION_SIZE = 2
C_ACTION_SIZE = 2
N_STEP = 3


class TestSAC_DS_Base(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        import tests.nn_conv_rnn as nn_conv_rnn
        import tests.nn_conv_vanilla as nn_vanilla

        self.sac = SAC_DS_Base(
            obs_shapes=OBS_SHAPES,
            d_action_size=D_ACTION_SIZE,
            c_action_size=C_ACTION_SIZE,
            model_abs_dir=None,
            model=nn_vanilla,
            n_step=N_STEP
        )

        self.sac_rnn = SAC_DS_Base(
            obs_shapes=OBS_SHAPES,
            d_action_size=D_ACTION_SIZE,
            c_action_size=C_ACTION_SIZE,
            model_abs_dir=None,
            model=nn_conv_rnn,
            use_rnn=True,
            n_step=N_STEP
        )

        self.rnn_shape = self.sac_rnn.get_initial_rnn_state(1).shape[1:]

    def test_vanilla_train(self):
        for _ in range(10):
            self.sac.choose_action(gen_batch_obs(OBS_SHAPES))
            self.sac.train(*gen_batch_trans(OBS_SHAPES, D_ACTION_SIZE, C_ACTION_SIZE, n=N_STEP))

    def test_rnn_train(self):
        for _ in range(10):
            self.sac_rnn.choose_rnn_action(*gen_batch_obs(OBS_SHAPES,
                                                          rnn_shape=self.rnn_shape,
                                                          d_action_size=D_ACTION_SIZE,
                                                          c_action_size=C_ACTION_SIZE))
            self.sac_rnn.train(*gen_batch_trans(OBS_SHAPES, D_ACTION_SIZE, C_ACTION_SIZE, n=N_STEP,
                                                rnn_shape=self.rnn_shape))

    def test_update_policy_variables(self):
        variables = self.sac.get_policy_variables()
        self.sac.update_policy_variables(variables)

    def test_update_nn_variables(self):
        variables = self.sac.get_nn_variables()
        self.sac.update_nn_variables(variables)

    def test_update_all_variables(self):
        variables = self.sac.get_all_variables()
        self.sac.update_all_variables(variables)

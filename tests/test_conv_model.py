import unittest

from algorithm.sac_base import SAC_Base
from tests.get_synthesis_data import *


class TestConvModel(unittest.TestCase):
    def _test_conv(self, param_dict):
        import algorithm.nn_models as m
        import tests.nn_conv as nn_conv

        conv_name = param_dict['conv']
        del param_dict['conv']

        class ModelRep(nn_conv.ModelRep):
            def _build_model(self):
                super()._build_model()

                self.conv = m.ConvLayers(84, 84, 3, conv_name, out_dense_depth=2, output_size=32)
        nn_conv.ModelRep = ModelRep

        obs_shapes = [(10,), (84, 84, 3)]

        sac = SAC_Base(
            obs_shapes=obs_shapes,
            d_action_size=4,
            c_action_size=4,
            model_abs_dir=None,
            model=nn_conv,
            use_rnn=True,
            **param_dict
        )

        rnn_shape = sac.get_initial_rnn_state(1).shape[1:]

        step = 0
        while step < 10:
            sac.choose_rnn_action(*gen_batch_obs(obs_shapes,
                                                 rnn_shape=rnn_shape,
                                                 d_action_size=4,
                                                 c_action_size=4))
            sac.fill_replay_buffer(*gen_episode_trans(obs_shapes,
                                                      4,
                                                      4,
                                                      episode_len=40,
                                                      rnn_shape=rnn_shape))
            step = sac.train()

    @staticmethod
    def gen_test_conv(param_dict):
        def func(self):
            self._test_conv(param_dict)
        return func


def __gen():
    possible_param_dicts = get_product({
        # 'conv': ['simple', 'nature', 'small'],
        'conv': ['simple'],
        'burn_in_step': [0, 5],
        'n_step': [1, 3],
        'use_prediction': [True, False],
        'use_extra_data': [True, False],
        'siamese': [None, 'SIMCLR', 'BYOL', 'SIMSIAM'],
        'siamese_use_adaptive': [False, True]
    })

    for i, param_dict in enumerate(possible_param_dicts):
        func_name = f'test_{i:03d}'
        for k, v in param_dict.items():
            func_name += f', {k}={v}'
        setattr(TestConvModel, func_name,
                TestConvModel.gen_test_conv(param_dict))


__gen()

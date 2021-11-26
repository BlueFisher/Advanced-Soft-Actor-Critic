import unittest

from algorithm.sac_base import SAC_Base
from tests.get_synthesis_data import *

OBS_SHAPES = [(10,), (30, 30, 3)]


class TestConvVanillaModel(unittest.TestCase):
    def _test(self, param_dict):
        import algorithm.nn_models as m
        import tests.nn_conv_vanilla as nn_conv

        conv_name = param_dict['conv']
        del param_dict['conv']

        class ModelRep(nn_conv.ModelRep):
            def _build_model(self):
                super()._build_model()

                self.conv = m.ConvLayers(30, 30, 3, conv_name, out_dense_depth=2, output_size=8)
        nn_conv.ModelRep = ModelRep

        sac = SAC_Base(
            obs_shapes=OBS_SHAPES,
            model_abs_dir=None,
            model=nn_conv,
            **param_dict
        )

        step = 0
        while step < 10:
            sac.choose_action(gen_batch_obs(OBS_SHAPES,
                                            d_action_size=param_dict['d_action_size'],
                                            c_action_size=param_dict['c_action_size']))
            sac.fill_replay_buffer(*gen_episode_trans(OBS_SHAPES,
                                                      d_action_size=param_dict['d_action_size'],
                                                      c_action_size=param_dict['c_action_size'],
                                                      episode_len=10))
            step = sac.train()

    @staticmethod
    def gen(param_dict):
        def func(self):
            self._test(param_dict)
        return func


class TestConvRNNModel(unittest.TestCase):
    def _test(self, param_dict):
        import algorithm.nn_models as m
        import tests.nn_conv_rnn as nn_conv

        conv_name = param_dict['conv']
        del param_dict['conv']

        class ModelRep(nn_conv.ModelRep):
            def _build_model(self):
                super()._build_model()

                self.conv = m.ConvLayers(30, 30, 3, conv_name, out_dense_depth=2, output_size=8)
        nn_conv.ModelRep = ModelRep

        sac = SAC_Base(
            obs_shapes=OBS_SHAPES,
            model_abs_dir=None,
            model=nn_conv,
            use_rnn=True,
            **param_dict
        )

        rnn_shape = sac.get_initial_rnn_state(1).shape[1:]

        step = 0
        while step < 10:
            sac.choose_rnn_action(*gen_batch_obs(OBS_SHAPES,
                                                 rnn_shape=rnn_shape,
                                                 d_action_size=param_dict['d_action_size'],
                                                 c_action_size=param_dict['c_action_size']))
            sac.fill_replay_buffer(*gen_episode_trans(OBS_SHAPES,
                                                      d_action_size=param_dict['d_action_size'],
                                                      c_action_size=param_dict['c_action_size'],
                                                      rnn_shape=rnn_shape,
                                                      episode_len=40))
            step = sac.train()

    @staticmethod
    def gen(param_dict):
        def func(self):
            self._test(param_dict)
        return func


def __gen_vanilla():
    param_dict_candidates = {
        'd_action_size': [0, 10],
        'c_action_size': [0, 4],
        # 'conv': ['simple', 'nature', 'small'],
        'conv': ['simple'],
        'n_step': [3],
        'siamese': [None, 'ATC', 'BYOL'],
        'siamese_use_q': [False, True],
        'siamese_use_adaptive': [False, True]
    }
    possible_param_dicts = get_product(param_dict_candidates)

    i = 0
    for param_dict in possible_param_dicts:
        if param_dict['d_action_size'] == 0 and param_dict['c_action_size'] == 0:
            continue

        func_name = f'test_{i:03d}'
        for k, v in param_dict.items():
            if len(param_dict_candidates[k]) > 1:
                func_name += f', {k}={v}'

        setattr(TestConvVanillaModel, func_name,
                TestConvVanillaModel.gen(param_dict))

        i += 1


def __gen_rnn():
    param_dict_candidates = {
        'd_action_size': [0, 10],
        'c_action_size': [0, 4],
        # 'conv': ['simple', 'nature', 'small'],
        'conv': ['simple'],
        # 'burn_in_step': [0, 5],
        'burn_in_step': [5],
        'n_step': [3],
        # 'use_prediction': [True, False],
        # 'use_extra_data': [True, False],
        'siamese': [None, 'ATC', 'BYOL'],
        'siamese_use_q': [False, True],
        'siamese_use_adaptive': [False, True]
    }

    possible_param_dicts = get_product(param_dict_candidates)

    i = 0
    for param_dict in possible_param_dicts:
        if param_dict['d_action_size'] == 0 and param_dict['c_action_size'] == 0:
            continue

        func_name = f'test_{i:03d}'
        for k, v in param_dict.items():
            if len(param_dict_candidates[k]) > 1:
                func_name += f', {k}={v}'

        setattr(TestConvRNNModel, func_name,
                TestConvRNNModel.gen(param_dict))

        i += 1


__gen_vanilla()
__gen_rnn()

import importlib
import unittest

from algorithm.sac_base import SAC_Base
from tests.get_synthesis_data import *

OBS_NAMES = ['vector', 'image']
OBS_SHAPES = [(10,), (84, 84, 3)]


class TestConvModel(unittest.TestCase):
    def _test(self, param_dict):
        import algorithm.nn_models as m
        import tests.nn_conv_vanilla as nn_conv

        importlib.reload(m)
        importlib.reload(nn_conv)

        conv_name = param_dict['conv']
        del param_dict['conv']

        class ModelRep(nn_conv.ModelRep):
            def _build_model(self):
                super()._build_model()
                self.conv = m.ConvLayers(84, 84, 3, conv_name, out_dense_depth=2, output_size=8)
        nn_conv.ModelRep = ModelRep

        sac = SAC_Base(
            obs_names=OBS_NAMES,
            obs_shapes=OBS_SHAPES,
            d_action_size=4,
            c_action_size=4,
            model_abs_dir=None,
            nn=nn_conv
        )

        step = 0
        while step < 10:
            sac.choose_action(**gen_batch_obs(OBS_SHAPES))
            sac.put_episode(**gen_episode_trans(OBS_SHAPES,
                                                d_action_size=4,
                                                c_action_size=4,
                                                episode_len=10))
            step = sac.train()

    @staticmethod
    def gen(param_dict):
        def func(self):
            self._test(param_dict)
        return func


def __gen():
    param_dict_candidates = {
        'conv': ['simple', 'nature', 'small'],
    }
    possible_param_dicts = get_product(param_dict_candidates)

    i = 0
    for param_dict in possible_param_dicts:
        func_name = f'test_{i:03d}'
        for k, v in param_dict.items():
            if len(param_dict_candidates[k]) > 1:
                func_name += f', {k}={v}'

        setattr(TestConvModel, func_name,
                TestConvModel.gen(param_dict))

        i += 1


__gen()

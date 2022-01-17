import unittest

from algorithm.sac_base import SAC_Base
from tests.get_synthesis_data import *


OBS_SHAPES = [(10,)]


class TestNNModel(unittest.TestCase):
    def _test(self, param_dict, is_q_model):
        import algorithm.nn_models as m
        import tests.nn_vanilla as nn_vanilla

        if is_q_model:
            class ModelQ(m.ModelQ):
                def _build_model(self):
                    super()._build_model(**param_dict)
            nn_vanilla.ModelQ = ModelQ
        else:
            class ModelPolicy(m.ModelPolicy):
                def _build_model(self):
                    super()._build_model(**param_dict)
            nn_vanilla.ModelPolicy = ModelPolicy

        sac = SAC_Base(
            obs_shapes=OBS_SHAPES,
            d_action_size=4,
            c_action_size=4,
            model_abs_dir=None,
            model=nn_vanilla,
        )

        step = 0
        while step < 5:
            sac.choose_action(*gen_batch_obs(OBS_SHAPES))
            sac.fill_replay_buffer(*gen_episode_trans(OBS_SHAPES,
                                                      d_action_size=4,
                                                      c_action_size=4,
                                                      episode_len=10))
            step = sac.train()

    @staticmethod
    def gen(param_dict, is_q_model):
        def func(self):
            self._test(param_dict, is_q_model)
        return func


def __gen():
    q_params = ['dense', 'd_dense', 'c_state', 'c_action', 'c_dense']
    possible_param_dicts = get_product({
        f'{n}_depth': [0, 2] for n in q_params
    })

    for i, param_dict in enumerate(possible_param_dicts):
        func_name = f'test_Q_model_{i:03d}'
        for k, v in param_dict.items():
            func_name += f', {k}={v}'

        setattr(TestNNModel, func_name,
                TestNNModel.gen(param_dict, True))

    policy_params = ['dense', 'd_dense', 'c_dense', 'mean', 'logstd']
    possible_param_dicts = get_product({
        f'{n}_depth': [0, 2] for n in policy_params
    })

    for i, param_dict in enumerate(possible_param_dicts):
        func_name = f'test_policy_model_{i:03d}'
        for k, v in param_dict.items():
            func_name += f', {k}={v}'

        setattr(TestNNModel, func_name,
                TestNNModel.gen(param_dict, False))


__gen()

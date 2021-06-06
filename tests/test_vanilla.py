import sys
import unittest

sys.path.append('..')

from algorithm.sac_base import SAC_Base

from .get_synthesis_data import *


class TestVanilla(unittest.TestCase):
    def _test_vanilla(self, param_dict):
        from . import nn_vanilla

        obs_shapes = [(10,)]

        sac = SAC_Base(
            obs_shapes=obs_shapes,
            model_abs_dir=None,
            model=nn_vanilla,
            device='cpu',
            use_add_with_td=True,
            **param_dict
        )

        step = 0
        while step < 3:
            sac.choose_action(gen_batch_obs(obs_shapes))
            sac.fill_replay_buffer(*gen_episode_trans(obs_shapes,
                                                      param_dict['d_action_size'],
                                                      param_dict['c_action_size']))
            step = sac.train()

    @staticmethod
    def gen_test_vanilla(param_dict):
        def func(self):
            self._test_vanilla(param_dict)
        return func


def __gen():
    possible_param_dicts = get_product({
        'd_action_size': [0, 10],
        'c_action_size': [0, 4],
        'n_step': [1, 3],
        'discrete_dqn_like': [True, False],
        'use_priority': [True, False],
        'use_n_step_is': [True, False],
        'use_curiosity': [True, False],
        'use_rnd': [True, False],
        'use_normalization': [True, False]
    })

    for i, param_dict in enumerate(possible_param_dicts):
        if param_dict['d_action_size'] == 0 and param_dict['c_action_size'] == 0:
            continue

        func_name = f'test_{i:03d}'
        for k, v in param_dict.items():
            func_name += f', {k}={v}'
        setattr(TestVanilla, func_name,
                TestVanilla.gen_test_vanilla(param_dict))


__gen()

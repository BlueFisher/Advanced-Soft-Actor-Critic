import unittest

from algorithm.sac_base import SAC_Base
from tests.get_synthesis_data import *


class TestRNN(unittest.TestCase):
    def _test_rnn(self, param_dict):
        import tests.nn_rnn as nn_rnn

        obs_shapes = [(10,)]

        sac = SAC_Base(
            obs_shapes=obs_shapes,
            model_abs_dir=None,
            model=nn_rnn,
            use_rnn=True,
            **param_dict
        )

        rnn_shape = sac.get_initial_rnn_state(1).shape[1:]

        step = 0
        while step < 10:
            sac.choose_rnn_action(*gen_batch_obs(obs_shapes,
                                                 rnn_shape=rnn_shape,
                                                 d_action_size=param_dict['d_action_size'],
                                                 c_action_size=param_dict['c_action_size']))
            sac.fill_replay_buffer(*gen_episode_trans(obs_shapes,
                                                      param_dict['d_action_size'],
                                                      param_dict['c_action_size'],
                                                      rnn_shape=rnn_shape))
            step = sac.train()

    @staticmethod
    def gen_test_rnn(param_dict):
        def func(self):
            self._test_rnn(param_dict)
        return func


def __gen():
    possible_param_dicts = get_product({
        'd_action_size': [0, 10],
        'c_action_size': [0, 4],
        'burn_in_step': [0, 5],
        'n_step': [1, 3],
        'use_prediction': [True, False],
        'use_extra_data': [True, False]
    })

    for i, param_dict in enumerate(possible_param_dicts):
        if param_dict['d_action_size'] == 0 and param_dict['c_action_size'] == 0:
            continue

        func_name = f'test_{i:03d}'
        for k, v in param_dict.items():
            func_name += f', {k}={v}'
        setattr(TestRNN, func_name,
                TestRNN.gen_test_rnn(param_dict))


__gen()

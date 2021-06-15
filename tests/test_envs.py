import importlib
import sys
import unittest
from pathlib import Path

sys.path.append('..')

from algorithm.sac_base import SAC_Base

from .get_synthesis_data import *


class TestEnvs(unittest.TestCase):
    def _test_env(self, obs_shapes, d_action_size, c_action_size, nn_abs_path, param_dict):
        spec = importlib.util.spec_from_file_location('nn', str(nn_abs_path))
        custom_nn_model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_nn_model)

        sac = SAC_Base(
            obs_shapes=obs_shapes,
            d_action_size=d_action_size,
            c_action_size=c_action_size,
            model_abs_dir=None,
            model=custom_nn_model,
            **param_dict
        )

        step = 0
        while step < 3:
            sac.choose_action(gen_batch_obs(obs_shapes))
            sac.fill_replay_buffer(*gen_episode_trans(obs_shapes,
                                                      d_action_size,
                                                      c_action_size))
            step = sac.train()

    @staticmethod
    def gen_test_env(obs_shapes, d_action_size, c_action_size, nn_abs_path, param_dict):
        def func(self):
            self._test_env(obs_shapes, d_action_size, c_action_size, nn_abs_path, param_dict)
        return func

    def _test_rnn_env(self, obs_shapes, d_action_size, c_action_size, nn_abs_path, param_dict):
        spec = importlib.util.spec_from_file_location('nn', str(nn_abs_path))
        custom_nn_model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_nn_model)

        sac = SAC_Base(
            obs_shapes=obs_shapes,
            d_action_size=d_action_size,
            c_action_size=c_action_size,
            model_abs_dir=None,
            model=custom_nn_model,
            use_rnn=True,
            **param_dict
        )

        rnn_shape = sac.get_initial_rnn_state(1).shape[1:]

        step = 0
        while step < 10:
            sac.choose_rnn_action(*gen_batch_obs(obs_shapes,
                                                 rnn_shape=rnn_shape,
                                                 d_action_size=d_action_size,
                                                 c_action_size=c_action_size))
            sac.fill_replay_buffer(*gen_episode_trans(obs_shapes,
                                                      d_action_size,
                                                      c_action_size,
                                                      rnn_shape=rnn_shape))
            step = sac.train()

    @staticmethod
    def gen_test_rnn_env(obs_shapes, d_action_size, c_action_size, nn_abs_path, param_dict):
        def func(self):
            self._test_rnn_env(obs_shapes, d_action_size, c_action_size, nn_abs_path, param_dict)
        return func


def _gen_test(env, obs_shapes, d_action_size, c_action_size, is_rnn, possible_param_dicts):
    p = Path(__file__).resolve().parent.parent.joinpath('envs').joinpath(f'{env}.py')

    if len(possible_param_dicts.keys()) == 0:
        if is_rnn:
            setattr(TestEnvs, f'test_env_{env}',
                    TestEnvs.gen_test_rnn_env(obs_shapes, d_action_size, c_action_size, p, {}))
        else:
            setattr(TestEnvs, f'test_env_{env}',
                    TestEnvs.gen_test_env(obs_shapes, d_action_size, c_action_size, p, {}))
    else:
        for param_dict in possible_param_dicts:
            func_name = f'test_env_{env}'
            for k, v in param_dict.items():
                func_name += f', {k}={v}'

            if is_rnn:
                setattr(TestEnvs, func_name,
                        TestEnvs.gen_test_rnn_env(obs_shapes, d_action_size, c_action_size, p, param_dict))
            else:
                setattr(TestEnvs, func_name,
                        TestEnvs.gen_test_env(obs_shapes, d_action_size, c_action_size, p, param_dict))


def __gen():
    _gen_test('roller/nn', [(6, )], 0, 6, False, {})
    _gen_test('roller/nn_hard', [(6, )], 0, 6, True, {})


__gen()

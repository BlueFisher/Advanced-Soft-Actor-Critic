import argparse
import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from algorithm.sac_main import Main


default_args = {
    'run': True,
    'run_a': [],
    'logger_in_file': False,

    'render': False,

    'envs': None,
    'max_iter': None,

    'port': None,
    'editor': None,
    'timescale': None,

    'name': None,
    'disable_sample': False,
    'use_env_nn': False,
    'device': 'cpu',
    'ckpt': None,
    'nn': None,

    'debug': True
}


class TestSACMain(unittest.TestCase):
    def _test_vanilla(self, env_args_dict):
        args = argparse.Namespace(
            config=None,
            env_args=env_args_dict,
            **default_args
        )

        root_dir = Path(__file__).resolve().parent.parent
        Main(root_dir, f'envs/test', args)

    def _test_rnn(self, env_args_dict):
        args = argparse.Namespace(
            config='rnn',
            env_args=env_args_dict,
            **default_args
        )

        root_dir = Path(__file__).resolve().parent.parent
        Main(root_dir, f'envs/test', args)

    def _test_attn(self, env_args_dict):
        args = argparse.Namespace(
            config='attn',
            env_args=env_args_dict,
            **default_args
        )

        root_dir = Path(__file__).resolve().parent.parent
        Main(root_dir, f'envs/test', args)

    @staticmethod
    def gen_vanilla(param_dict):
        def func(self):
            self._test_vanilla(param_dict)
        return func

    @staticmethod
    def gen_rnn(param_dict):
        def func(self):
            self._test_rnn(param_dict)
        return func

    @staticmethod
    def gen_attn(param_dict):
        def func(self):
            self._test_attn(param_dict)
        return func


def __gen():
    env_args_dicts = [
        {
            'ma_obs_shapes': {
                'test0': [(6,)]
            },
            'ma_d_action_sizes': {
                'test0': []
            },
            'ma_c_action_size': {
                'test0': 2
            }
        },
        {
            'ma_obs_shapes': {
                'test0': [(6,)]
            },
            'ma_d_action_sizes': {
                'test0': [2, 4]
            },
            'ma_c_action_size': {
                'test0': 0
            }
        },
        {
            'ma_obs_shapes': {
                'test0': [(6,)]
            },
            'ma_d_action_sizes': {
                'test0': [2, 4]
            },
            'ma_c_action_size': {
                'test0': 2
            }
        },
        {
            'ma_obs_shapes': {
                'test0': [(6,)],
                'test1': [(8,)]
            },
            'ma_d_action_sizes': {
                'test0': [],
                'test1': []
            },
            'ma_c_action_size': {
                'test0': 2,
                'test1': 4
            }
        },
        {
            'ma_obs_shapes': {
                'test0': [(6,)],
                'test1': [(8,)]
            },
            'ma_d_action_sizes': {
                'test0': [],
                'test1': []
            },
            'ma_c_action_size': {
                'test0': 2,
                'test1': 4
            }
        },
        {
            'ma_obs_shapes': {
                'test0': [(6,)],
                'test1': [(8,)]
            },
            'ma_d_action_sizes': {
                'test0': [2, 4],
                'test1': [4, 2]
            },
            'ma_c_action_size': {
                'test0': 2,
                'test1': 4
            }
        },
    ]

    i = 0
    for env_args_dict in env_args_dicts:
        func_name = f'test_vanilla_{i:03d}'

        setattr(TestSACMain, func_name,
                TestSACMain.gen_vanilla(env_args_dict))

        i += 1

    i = 0
    for env_args_dict in env_args_dicts:
        func_name = f'test_rnn_{i:03d}'

        setattr(TestSACMain, func_name,
                TestSACMain.gen_rnn(env_args_dict))

        i += 1

    i = 0
    for env_args_dict in env_args_dicts:
        func_name = f'test_attn_{i:03d}'

        setattr(TestSACMain, func_name,
                TestSACMain.gen_attn(env_args_dict))

        i += 1


__gen()

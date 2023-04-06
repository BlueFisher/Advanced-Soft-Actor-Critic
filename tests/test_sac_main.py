import argparse
import unittest
from pathlib import Path

from algorithm.sac_base import SAC_Base
from algorithm.sac_main import Main
from tests.get_synthesis_data import *


class TestSACMain(unittest.TestCase):
    def _test_vanilla(self, env_args_dict):
        args = argparse.Namespace(
            config=None,
            run=False,
            logger_in_file=False,

            render=False,
            env_args=env_args_dict,
            envs=None,
            max_iter=None,

            port=None,
            editor=None,

            name=None,
            nn=None,
            disable_sample=False,
            use_env_nn=False,
            device='cpu',
            ckpt=None
        )

        root_dir = Path(__file__).resolve().parent.parent
        Main(root_dir, f'envs/test', args)

    def _test_rnn(self, env_args_dict):
        args = argparse.Namespace(
            config='rnn',
            run=False,
            logger_in_file=False,

            render=False,
            env_args=env_args_dict,
            envs=None,
            max_iter=None,

            port=None,
            editor=None,

            name=None,
            nn=None,
            disable_sample=False,
            use_env_nn=False,
            device='cpu',
            ckpt=None
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


def __gen_vanilla():
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
                TestSACMain.gen_vanilla(env_args_dict))

        i += 1


__gen_vanilla()

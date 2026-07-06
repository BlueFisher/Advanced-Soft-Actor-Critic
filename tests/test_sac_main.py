import itertools
import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from algorithm.sac_main import Main


class TestSACMain(unittest.TestCase):
    def _test(self, config_cat, train_mode, env_args_dict):
        root_dir = Path(__file__).resolve().parent.parent
        Main(root_dir,
             f'envs/test',
             config_cat=config_cat,
             train_mode=train_mode,
             env_args=env_args_dict)

    @staticmethod
    def gen(config, train_mode, param_dict):
        def func(self):
            self._test(config, train_mode, param_dict)
        return func


def __gen():
    configs = [None, 'rnn', 'attn']

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
    for config, train_mode in itertools.product(configs, [True, False]):
        for j, env_args_dict in enumerate(env_args_dicts):
            func_name = f'test_{i:03d}, config={config}, train={train_mode}, env={j}'

            setattr(TestSACMain, func_name,
                    TestSACMain.gen(config, train_mode, env_args_dict))

            i += 1


__gen()

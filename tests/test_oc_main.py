import argparse
import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from algorithm.oc.oc_main import OC_Main


class TestOCMain(unittest.TestCase):
    def _test(self, config_cat, env_args_dict):
        root_dir = Path(__file__).resolve().parent.parent
        OC_Main(root_dir,
                f'envs/test',
                config_cat=config_cat,
                train_mode=False,
                env_args=env_args_dict)

    @staticmethod
    def gen_oc(config, param_dict):
        def func(self):
            self._test(config, param_dict)
        return func


def __gen():
    configs = ['oc',
               'oc_rnn',
               'oc_rnn_o_rnn',
               'oc_attn',
               'oc_attn_o_rnn',
               'oc_dilated_rnn',
               #    'oc_dilated_rnn_o_rnn',
               'oc_dilated_attn',
               #    'oc_dilated_attn_o_rnn'
               ]

    env_args_dict = {
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
    }

    i = 0
    for config in configs:
        func_name = f'test_{i:03d}_{config}'

        setattr(TestOCMain, func_name,
                TestOCMain.gen_oc(config, env_args_dict))

        i += 1


__gen()

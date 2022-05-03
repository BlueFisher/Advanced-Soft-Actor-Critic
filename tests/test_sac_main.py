import argparse
import unittest
from pathlib import Path

from algorithm.sac_base import SAC_Base
from algorithm.sac_main import Main
from tests.get_synthesis_data import *


class TestSACMain(unittest.TestCase):
    def test_vanilla(self):
        args = argparse.Namespace(
            config=None,
            run=False,
            logger_in_file=False,

            render=False,
            env_args=None,
            agents=None,
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

    def test_rnn(self):
        args = argparse.Namespace(
            config='rnn',
            run=False,
            logger_in_file=False,

            render=False,
            env_args=None,
            agents=None,
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
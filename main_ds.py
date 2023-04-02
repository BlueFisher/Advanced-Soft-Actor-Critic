"""
! The distributed training module is suspended for maintenance
"""

print('The distributed training module is suspended for maintenance')
exit()

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

from algorithm.config_helper import set_logger

# for correctly import protoc
sys.path.append(str(Path(__file__).resolve().parent.joinpath('ds/proto')))

HITTED_ENVS = {'roller', 'square', 'pyramid', 'usv', 'realcar'}

if __name__ == '__main__':
    set_logger()
    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument('env')
    parser.add_argument('process_type', choices=['learner', 'l',
                                                 'actor', 'a'])
    parser.add_argument('--config', '-c', help='config file')
    parser.add_argument('--run', action='store_true', help='inference mode')
    parser.add_argument('--logger_in_file', action='store_true', help='logging into a file')

    parser.add_argument('--learner_host', help='learner host')
    parser.add_argument('--learner_port', type=int, help='learner port')

    parser.add_argument('--render', action='store_true', help='render')
    parser.add_argument('--env_args', help='additional args for environments')
    parser.add_argument('--agents', type=int, help='number of agents')

    parser.add_argument('--unity_port', '-p', type=int, default=5005, help='UNITY: communication port')
    parser.add_argument('--editor', action='store_true', help='UNITY: running in Unity Editor')

    parser.add_argument('--name', '-n', help='training name')
    parser.add_argument('--device', help='cpu or gpu')
    parser.add_argument('--ckpt', help='ckeckpoint to restore')
    parser.add_argument('--nn', help='neural network model')
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent
    config_dir = f'envs/{args.env}'

    if args.process_type in ['learner', 'l']:
        if args.env in HITTED_ENVS:
            from ds.main_hitted import LearnerHitted as Learner
        else:
            from ds.learner import Learner
        Learner(root_dir, config_dir, args)
    elif args.process_type in ['actor', 'a']:
        if args.env in HITTED_ENVS:
            from ds.main_hitted import ActorHitted as Actor
        else:
            from ds.actor import Actor
        Actor(root_dir, config_dir, args)

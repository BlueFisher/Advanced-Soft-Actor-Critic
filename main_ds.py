import argparse
import os
from pathlib import Path
import sys

# for correctly import protoc
sys.path.append(str(Path(__file__).resolve().parent.joinpath('ds/proto')))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env')
    parser.add_argument('process_type', choices=['replay', 'r', 'learner', 'l', 'actor', 'a'])
    parser.add_argument('--config', '-c', help='config file')
    parser.add_argument('--run', action='store_true', help='inference mode')
    parser.add_argument('--render', action='store_true', help='render')
    parser.add_argument('--editor', action='store_true', help='running in Unity Editor')
    parser.add_argument('--logger_file', help='logging into a file')
    parser.add_argument('--name', '-n', help='training name')
    parser.add_argument('--build_port', '-p', type=int, default=5005, help='communication port')
    parser.add_argument('--nn', help='neural network model')
    parser.add_argument('--ckpt', help='ckeckpoint to restore')
    parser.add_argument('--agents', type=int, default=1, help='number of agents')
    parser.add_argument('--noise', type=float, default=0, help='additional noise for actor')
    args = parser.parse_args()

    root_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = f'envs/{args.env}'

    if args.process_type in ['replay', 'r']:
        from ds.replay import Replay
        Replay(root_dir, config_dir, args)
    elif args.process_type in ['learner', 'l']:
        if args.env in ['simple_roller', 'ray_roller']:
            from ds.main_hitted import LearnerHitted as Learner
        else:
            from ds.learner import Learner
        Learner(root_dir, config_dir, args)
    elif args.process_type in ['actor', 'a']:
        if args.env in ['simple_roller', 'ray_roller']:
            from ds.main_hitted import ActorHitted as Actor
        else:
            from ds.actor import Actor
        Actor(root_dir, config_dir, args)

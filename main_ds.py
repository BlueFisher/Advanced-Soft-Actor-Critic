import argparse
from pathlib import Path
import sys

# for correctly import protoc
sys.path.append(str(Path(__file__).parent.joinpath('ds/proto')))

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
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--sac', help='neural network model')
    parser.add_argument('--agents', type=int, default=1, help='number of agents')
    args = parser.parse_args()

    config_path = f'envs/{args.env}'

    if args.process_type in ['replay', 'r']:
        from ds.replay import Replay
        Replay(config_path, args)
    elif args.process_type in ['learner', 'l']:
        if args.env in ['simple_roller', 'ray_roller']:
            from ds.main_hitted import LearnerHitted as Learner
        else:
            from ds.learner import Learner
        Learner(config_path, args)
    elif args.process_type in ['actor', 'a']:
        if args.env in ['simple_roller', 'ray_roller']:
            from ds.main_hitted import ActorHitted as Actor
        else:
            from ds.actor import Actor
        Actor(config_path, args)

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

from algorithm.config_helper import set_logger

# for correctly import protoc
sys.path.append(str(Path(__file__).resolve().parent.joinpath('ds/proto')))

if __name__ == '__main__':
    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument('env')
    parser.add_argument('process_type', choices=['learner', 'l',
                                                 'actor', 'a'])
    parser.add_argument('--hit', action='store_true', default=False)
    parser.add_argument('--oc', action='store_true', default=False)

    parser.add_argument('--config', '-c', help='config file')
    parser.add_argument('--override', '-o', default=[], nargs='+', help='override config')
    parser.add_argument('--run', action='store_true', help='inference mode for all agents, ignore run_a')
    parser.add_argument('--run_a', action='append', default=[], help='inference mode for specific agents')
    parser.add_argument('--copy_model', default=None, help='copy existed model directory to current model directory')
    parser.add_argument('--logger_in_file', action='store_true', help='logging into a file')

    parser.add_argument('--learner_host', help='learner host')
    parser.add_argument('--learner_port', type=int, help='learner port')

    parser.add_argument('--render', action='store_true', help='render')
    parser.add_argument('--env_args', default=[], nargs='+', help='additional arguments for environments')
    parser.add_argument('--envs', type=int, help='number of env copies')
    parser.add_argument('--max_iter', type=int, help='max iteration')

    parser.add_argument('--u_port', '-p', type=int, default=5005, help='UNITY: communication port')
    parser.add_argument('--u_editor', action='store_true', help='UNITY: running in Unity Editor')
    parser.add_argument('--u_quality_level', type=int, default=2, help='UNITY: Quality level. \
                                                                        0: URP-Performant-Renderer, \
                                                                        1: URP-Balanced-Renderer, \
                                                                        2: URP-HighFidelity-Renderer')
    parser.add_argument('--u_timescale', type=float, default=None, help='UNITY: timescale')

    parser.add_argument('--name', '-n', help='training name')
    parser.add_argument('--disable_sample', action='store_true', help='disable sampling when choosing actions')
    parser.add_argument('--use_env_nn', action='store_true', help='always use nn.py in env, or use saved nn_models.py if existed')
    parser.add_argument('--device', help='cpu or gpu')
    parser.add_argument('--ckpt', help='ckeckpoint to restore')
    parser.add_argument('--nn', help='neural network model')

    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    set_logger(debug=args.debug)

    root_dir = Path(__file__).resolve().parent
    config_dir = f'envs/{args.env}'

    if args.process_type in ['learner', 'l']:
        if args.hit:
            from ds.main_hit import LearnerHit as Learner
        else:
            from ds.learner import Learner
        Learner(root_dir, config_dir, args)
    elif args.process_type in ['actor', 'a']:
        if args.hit:
            from ds.main_hit import ActorHit as Actor
        else:
            from ds.actor import Actor
        Actor(root_dir, config_dir, args)

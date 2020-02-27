import sys

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env')
    parser.add_argument('--config', '-c', help='config file')
    parser.add_argument('--run', action='store_true', help='inference mode')
    parser.add_argument('--render', action='store_true', help='render')
    parser.add_argument('--editor', action='store_true', help='running in Unity Editor')
    parser.add_argument('--logger_file', help='logging into a file')
    parser.add_argument('--name', '-n', help='training name')
    parser.add_argument('--port', '-p', type=int, default=5005, help='communication port')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--sac', help='neural network model')
    parser.add_argument('--agents', type=int, default=1, help='number of agents')
    parser.add_argument('--repeat', type=int, default=1, help='number of repeated experiments')
    args = parser.parse_args()

    if args.env in ['simple_roller', 'ray_roller']:
        from algorithm.sac_main_hitted import MainHitted as Main
    else:
        from algorithm.sac_main import Main

    if sys.platform == 'win32':
        for _ in range(args.repeat):
            Main(f'envs/{args.env}', args)
    elif sys.platform == 'linux':
        for i in range(args.repeat):
            Main(f'envs/{args.env}', args)
            args.port += 1

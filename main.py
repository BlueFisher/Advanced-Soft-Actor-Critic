import argparse
import sys
from pathlib import Path

from algorithm.config_helper import set_logger

try:
    import toy_memory
except:
    pass


HITTED_ENVS = {'roller',
               'square', 'square/obstacle', 'pyramid',
               'uav',
               'ugv', 'ugv/ugv_street_search', 'ugv/ugv_parking',
               'usv',
               'gym/toy_stack', 'gym/toy_queue', 'gym/toy_hallway'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env')
    parser.add_argument('--oc', action='store_true', default=False)

    parser.add_argument('--config', '-c', help='config file')
    parser.add_argument('--override', '-o', default=[], nargs='+', help='override config')
    parser.add_argument('--run', action='store_true', help='inference mode for all agents, ignore run_a')
    parser.add_argument('--run_a', action='append', default=[], help='inference mode for specific agents')
    parser.add_argument('--logger_in_file', action='store_true', help='logging into a file')

    parser.add_argument('--render', action='store_true', help='render')
    parser.add_argument('--env_args', default=[], nargs='+', help='additional arguments for environments')
    parser.add_argument('--envs', type=int, help='number of env copies')
    parser.add_argument('--max_iter', type=int, help='max iteration')

    parser.add_argument('--u_port', '-p', type=int, default=5005, help='UNITY: communication port')
    parser.add_argument('--u_editor', action='store_true', help='UNITY: running in Unity Editor')
    parser.add_argument('--u_timescale', type=float, default=None, help='UNITY: timescale')

    parser.add_argument('--name', '-n', help='training name')
    parser.add_argument('--disable_sample', action='store_true', help='disable sampling when choosing actions')
    parser.add_argument('--use_env_nn', action='store_true', help='always use nn.py in env, or use saved nn_models.py if existed')
    parser.add_argument('--device', help='cpu or gpu')
    parser.add_argument('--ckpt', help='ckeckpoint to restore')
    parser.add_argument('--nn', help='neural network model')
    parser.add_argument('--repeat', type=int, default=1, help='number of repeated experiments')

    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    set_logger(debug=args.debug)

    if args.env in HITTED_ENVS:
        if args.oc:
            from algorithm.oc.oc_main_hitted import OC_MainHitted as Main
        else:
            from algorithm.sac_main_hitted import MainHitted as Main
    else:
        if args.oc:
            from algorithm.oc.oc_main import OC_Main as Main
        else:
            from algorithm.sac_main import Main

    root_dir = Path(__file__).resolve().parent
    if sys.platform == 'win32':
        for _ in range(args.repeat):
            Main(root_dir, f'envs/{args.env}', args)
    elif sys.platform == 'linux':
        for i in range(args.repeat):
            Main(root_dir, f'envs/{args.env}', args)
            args.u_port += 1

import argparse
import sys
from pathlib import Path

from algorithm.config_helper import set_logger

try:
    import toy_memory
except:
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env')
    parser.add_argument('--oc', action='store_true', default=False)

    parser.add_argument('--config', '-c', help='Config file')
    parser.add_argument('--override', '-o', default=[], nargs='+', help='Override config')
    parser.add_argument('--run', action='store_true', help='Inference mode for all agents, ignore run_a')
    parser.add_argument('--run_a', action='append', default=[], help='Inference mode for specific agents')
    parser.add_argument('--copy_model', default=None, help='Copy existed model directory to current model directory')
    parser.add_argument('--logger_in_file', action='store_true', help='Logging into a file')

    parser.add_argument('--render', action='store_true', help='Render')
    parser.add_argument('--env_args', default=[], nargs='+', help='Additional arguments for environments. \
                                                                    Eqvilent to --override base_config.env_args')
    parser.add_argument('--envs', type=int, help='Number of env copies. Eqvilent to --override base_config.n_envs')
    parser.add_argument('--max_iter', type=int, help='Maximum iteration. Eqvilent to --override base_config.max_iter')

    parser.add_argument('--u_port', '-p', type=int, default=5005, help='UNITY: communication port')
    parser.add_argument('--u_editor', action='store_true', help='UNITY: running in Unity Editor')
    parser.add_argument('--u_quality_level', type=int, default=2, help='UNITY: Quality level. \
                                                                        0: URP-Performant-Renderer, \
                                                                        1: URP-Balanced-Renderer, \
                                                                        2: URP-HighFidelity-Renderer')
    parser.add_argument('--u_timescale', type=float, default=None, help='UNITY: Timescale')

    parser.add_argument('--name', '-n', help='Training name. Eqvilent to --override base_config.name')
    parser.add_argument('--disable_sample', action='store_true', help='Disable sampling when choosing actions')
    parser.add_argument('--use_env_nn', action='store_true', help='Always use nn.py in env, or use saved nn_models.py if existed')
    parser.add_argument('--device', help='CPU or GPU')
    parser.add_argument('--ckpt', help='Ckeckpoint to restore')
    parser.add_argument('--nn', help='Neural network model. Eqvilent to --override sac_config.nn')

    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    set_logger(debug=args.debug)

    override = []
    if args.override is not None:
        for kv in args.override:
            k, v = kv.split('=')
            k_list = k.split('.')
            override.append((k_list, v))

    env_args = {}
    for env_arg in args.env_args:
        k, v = env_arg.split('=')
        env_args[k] = v

    if args.oc:
        from algorithm.oc.oc_main import OC_Main as Main
    else:
        from algorithm.sac_main import Main

    root_dir = Path(__file__).resolve().parent
    Main(root_dir, f'envs/{args.env}',
         config_cat=args.config,
         override=override,
         train_mode=not args.run,
         inference_ma_names=set(args.run_a),
         copy_model=args.copy_model,
         logger_in_file=args.logger_in_file,
         render=args.render,
         env_args=env_args,
         n_envs=args.envs,
         max_iter=args.max_iter,
         unity_port=args.u_port,
         unity_run_in_editor=args.u_editor,
         unity_quality_level=args.u_quality_level,
         unity_time_scale=args.u_timescale,
         name=args.name,
         disable_sample=args.disable_sample,
         use_env_nn=args.use_env_nn,
         device=args.device,
         last_ckpt=args.ckpt,
         nn=args.nn)

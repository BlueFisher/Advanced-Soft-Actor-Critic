import logging
import logging.handlers
import time
import random
import string
import os

import numpy as np
import yaml


def initialize_config_from_yaml(default_config_path, config_file_path, config_cat=None):
    """
    config_cat: Specific experiment name. 
                The `config_cat` will override `defult` if it is not None
    """
    config = dict()

    with open(default_config_path) as f:
        default_config_file = yaml.load(f, Loader=yaml.FullLoader)
        config = default_config_file

    # Initialize config from config_file_path
    with open(config_file_path) as f:
        config_file = yaml.load(f, Loader=yaml.FullLoader)

    for cat in ['default', config_cat]:
        if cat is None:
            continue
        for k, v in config_file[cat].items():
            # `k` is the key like base_config, net_config, sac_config
            assert k in config.keys(), f'{k} in {cat} is invalid'
            if v is not None:
                if k == 'reset_config':
                    config[k] = v
                else:
                    for kk, vv in v.items():
                        # `kk` is the key in `k`
                        if kk != 'random_params':
                            assert kk in config[k].keys(), f'{kk} is invalid in {k}'
                        config[k][kk] = vv

    # Deal with random_params
    for k, v in config.items():
        if v is not None and 'random_params' in v:
            random_params = v['random_params']
            del v['random_params']

            for param, opt in random_params.items():
                assert param in v, f'{param} is invalid in random_params'
                assert not ('in' in opt and ('truncated' in opt or 'std' in opt)), f'option "in" cannot be used with "truncated" or "std"'
                if 'in' in opt:
                    v[param] = random.choice(opt['in'])
                elif 'std' in opt:
                    v[param] = np.random.normal(v[param], opt['std'])
                    if 'truncated' in opt:
                        v[param] = np.clip(v[param], opt['truncated'][0], opt['truncated'][1])
                elif 'truncated' in opt:
                    v[param] = np.random.random() * (opt['truncated'][1] - opt['truncated'][0]) + opt['truncated'][0]

    return config


def set_logger(logger_file=None):
    # logger config
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Remove default root logger handler
    logger.handlers = []

    # Create stream handler
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)

    # Add handler and formatter to logger
    sh.setFormatter(logging.Formatter('[%(levelname)s] - [%(name)s] - %(message)s'))
    logger.addHandler(sh)

    if logger_file is not None:
        # Create file handler
        fh = logging.handlers.RotatingFileHandler(logger_file, maxBytes=10 * 1024 * 1024, backupCount=20)
        fh.setLevel(logging.INFO)

        # Add handler and formatter to logger
        fh.setFormatter(logging.Formatter('%(asctime)-15s [%(levelname)s] - [%(name)s] - %(message)s'))
        logger.addHandler(fh)


def save_config(config, model_root_path, config_name):
    if not os.path.exists(model_root_path):
        os.makedirs(model_root_path)
    with open(f'{model_root_path}/{config_name}', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def display_config(config, logger):
    config_str = ''
    for k, v in config.items():
        if v is not None:
            config_str += f'\n{k}'
            for kk, vv in v.items():
                config_str += f'\n{kk:>30}: {vv}'
    logger.info(config_str)


def generate_base_name(name, prefix=None):
    """
    Replace {time} from current time and random letters
    """
    now = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    rand = ''.join(random.sample(string.ascii_letters, 4))

    replaced = now + rand
    if prefix:
        replaced = prefix + '_' + replaced

    return name.replace('{time}', replaced)

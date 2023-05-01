import logging
import logging.handlers
import random
import string
import time
from copy import copy, deepcopy
from pathlib import Path

import numpy as np
import yaml


def initialize_config_from_yaml(default_config_path, config_file_path,
                                config_cat=None,
                                is_evolver=False):
    """
    config_cat: Specific experiment name. 
                The `config_cat` will override `default` if it is not None
    """
    config = dict()

    with open(default_config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Initialize config from config_file_path
    with open(config_file_path) as f:
        config_file = yaml.load(f, Loader=yaml.FullLoader)

    def _tra_dict(dict_ori: dict, dict_new: dict):
        if 'inherited' in dict_new:
            if isinstance(dict_new['inherited'], str):
                _tra_dict(dict_ori, config_file[dict_new['inherited']])
            elif isinstance(dict_new['inherited'], list):
                for inherited_config_cat in dict_new['inherited']:
                    _tra_dict(dict_ori, config_file[inherited_config_cat])

        for k, v in dict_new.items():
            if k not in dict_ori:
                dict_ori[k] = v
            else:
                if isinstance(dict_ori[k], dict) and isinstance(v, dict):
                    _tra_dict(dict_ori[k], v)
                else:
                    dict_ori[k] = v

    _tra_dict(config, config_file['default'])
    if config_cat is not None:
        _tra_dict(config, config_file[config_cat])

    ma_configs = {}
    # Deal with multi-agents config
    if config['ma_config'] is not None:
        for ma_name, ma_config in config['ma_config'].items():
            ma_configs[ma_name] = deepcopy(config)
            del ma_configs[ma_name]['ma_config']
            _tra_dict(ma_configs[ma_name], ma_config)
    del config['ma_config']

    # Deal with random_params
    for k, v in config.items():
        if v is not None and 'random_params' in v:
            random_params = v['random_params']

            if is_evolver:
                for param, opt in random_params.items():
                    assert param in v, f'{param} is invalid in random_params'
                    assert not ('in' in opt and ('truncated' in opt or 'std' in opt)), f'option "in" cannot be used with "truncated" or "std"'
                    v[param] = '[placeholder]'
            else:
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

    return config, ma_configs


def save_config(config, model_root_dir: Path, config_name):
    model_root_dir.mkdir(parents=True, exist_ok=True)

    with open(model_root_dir.joinpath(config_name), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def display_config(config, logger, name=''):
    logger.info(f'Config {name}:\n' + yaml.dump(config, default_flow_style=False))


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


############# CONFIG LOGGING #############


MAPPING = {
    'DEBUG': 37,  # white
    'INFO': 36,  # cyan
    'WARNING': 33,  # yellow
    'ERROR': 31,  # red
    'CRITICAL': 41,  # white on red bg
}

PREFIX = '\033['
SUFFIX = '\033[0m'


class ColoredFormatter(logging.Formatter):
    def format(self, record):
        colored_record = copy(record)
        levelname = colored_record.levelname
        seq = MAPPING.get(levelname, 37)  # default white
        colored_levelname = ('{0}{1}m{2}{3}') \
            .format(PREFIX, seq, levelname, SUFFIX)
        colored_record.levelname = colored_levelname
        return logging.Formatter.format(self, colored_record)


def set_logger(logger_file=None, debug=False):
    # logger config
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    # Remove default root logger handler
    logger.handlers = []

    # Create stream handler
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG if debug else logging.INFO)

    # Add handler and formatter to logger
    sh.setFormatter(ColoredFormatter('[%(levelname)s] - [%(name)s] - %(message)s'))
    logger.addHandler(sh)

    if logger_file is not None:
        # Create file handler
        fh = logging.handlers.RotatingFileHandler(logger_file, maxBytes=10 * 1024 * 1024, backupCount=20)
        fh.setLevel(logging.DEBUG if debug else logging.INFOO)

        # Add handler and formatter to logger
        fh.setFormatter(logging.Formatter('%(asctime)-15s [%(levelname)s] - [%(name)s] - %(message)s'))

        logger.addHandler(fh)

    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

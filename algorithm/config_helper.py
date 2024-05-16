import logging
import logging.handlers
import random
import string
import sys
import time
from copy import copy, deepcopy
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml


def initialize_config_from_yaml(default_config_path: Path,
                                config_file_path: Path,
                                config_cat: Optional[str] = None,
                                override: Optional[List[str]] = None,
                                is_evolver=False):
    """
    config_cat: Specific experiment name. 
                The `config_cat` will override `default` if it is not None
    override: Override config
    """
    config = dict()

    with open(default_config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Initialize config from config_file_path
    with open(config_file_path) as f:
        config_file = yaml.load(f, Loader=yaml.FullLoader)

    def _modify_platform(d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                if 'win32' in v or 'linux' in v:
                    d[k] = v[sys.platform]
                else:
                    _modify_platform(v)

    def _update_dict(dict_ori: dict, dict_new: dict):
        if 'inherited' in dict_new:
            if isinstance(dict_new['inherited'], str):
                _update_dict(dict_ori, config_file[dict_new['inherited']])
            elif isinstance(dict_new['inherited'], list):
                for inherited_config_cat in dict_new['inherited']:
                    _update_dict(dict_ori, config_file[inherited_config_cat])

        for k, v in dict_new.items():
            if k not in dict_ori:
                dict_ori[k] = v
            else:
                if isinstance(dict_ori[k], dict) and isinstance(v, dict):
                    _update_dict(dict_ori[k], v)
                else:
                    dict_ori[k] = v

    _modify_platform(config)
    _modify_platform(config_file)
    _update_dict(config, config_file['default'])
    if config_cat is not None:
        _update_dict(config, config_file[config_cat])

    if override is not None:
        for kv in override:  # a.b.c=x
            k, v = kv.split('=')
            k_list = k.split('.')
            last_k = k_list[-1]

            tmp_config = config
            for k in k_list[:-1]:
                tmp_config = tmp_config[k]
            assert isinstance(tmp_config[last_k], (bool, int, float, str)), f'{kv} not in type {type(tmp_config[last_k])}'
            if isinstance(tmp_config[last_k], bool):
                tmp_config[last_k] = v.lower() in ('true', '1')
            elif isinstance(tmp_config[last_k], int):
                tmp_config[last_k] = int(v)
            elif isinstance(tmp_config[last_k], float):
                tmp_config[last_k] = float(v)
            else:
                assert isinstance(tmp_config[last_k], str)
                tmp_config[last_k] = v if v != 'null' else None

    ma_configs = {}
    # Deal with multi-agents config
    if config['ma_config'] is not None:
        for ma_name, ma_config in config['ma_config'].items():
            ma_configs[ma_name] = deepcopy(config)
            del ma_configs[ma_name]['ma_config']
            _update_dict(ma_configs[ma_name], ma_config)
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


def set_logger(debug=False):
    # logger config
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    # Remove default root logger handler
    logger.handlers.clear()

    # Create stream handler
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG if debug else logging.INFO)

    # Add handler and formatter to logger
    sh.setFormatter(ColoredFormatter('[%(levelname)s] [%(name)s] %(message)s'))
    logger.addHandler(sh)

    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


def add_file_logger(logger_file: Path):
    logger = logging.getLogger()

    # Create file handler
    fh = logging.FileHandler(logger_file)
    fh.setLevel(logger.level)

    # Add handler and formatter to logger
    fh.setFormatter(logging.Formatter('%(asctime)-15s [%(levelname)s] [%(name)s] %(message)s'))

    logger.addHandler(fh)

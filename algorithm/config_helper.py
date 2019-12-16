import logging
import os
import sys
import yaml


def initialize_config_from_yaml(default_config_path, config_file_path):
    config = dict()

    with open(default_config_path) as f:
        default_config_file = yaml.load(f, Loader=yaml.FullLoader)
        config = default_config_file

    # initialize config from config_file_path
    if config_file_path is not None:
        with open(config_file_path) as f:
            config_file = yaml.load(f, Loader=yaml.FullLoader)
            for k, v in config_file.items():
                assert k in config.keys(), f'{k} is invalid'
                if v is not None:
                    for kk, vv in v.items():
                        assert kk in config[k].keys(), f'{kk} is invalid in {k}'
                        config[k][kk] = vv

    config['base_config']['build_path'] = config['base_config']['build_path'][sys.platform]

    return config


def set_logger(logger_name, logger_file=None):
    # logger config
    _log = logging.getLogger()
    _log.setLevel(logging.INFO)
    # remove default root logger handler
    _log.handlers = []

    # create stream handler
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)

    # add handler and formatter to logger
    sh.setFormatter(logging.Formatter('[%(levelname)s] - [%(name)s] - %(message)s'))
    _log.addHandler(sh)

    logger = logging.getLogger(logger_name)
    logger.setLevel(level=logging.INFO)

    if logger_file is not None:
        # create file handler
        fh = logging.handlers.RotatingFileHandler(logger_file, maxBytes=1024 * 100, backupCount=5)
        fh.setLevel(logging.INFO)

        # add handler and formatter to logger
        fh.setFormatter(logging.Formatter('%(asctime)-15s [%(levelname)s] - [%(name)s] - %(message)s'))
        logger.addHandler(fh)

    return logger


def save_config(config, model_root_path, config_name):
    if not os.path.exists(model_root_path):
        os.makedirs(model_root_path)
    with open(f'{model_root_path}/{config_name}', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def display_config(config, logger):
    config_str = ''
    for k, v in config.items():
        config_str += f'\n{k}'
        for kk, vv in v.items():
            config_str += f'\n{kk:>25}: {vv}'
    logger.info(config_str)
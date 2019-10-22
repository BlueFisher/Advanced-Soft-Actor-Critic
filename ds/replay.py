from pathlib import Path
import getopt
import logging
import sys
import threading
import yaml

from flask import Flask, jsonify, request
import requests
import numpy as np

from trans_cache import TransCache

sys.path.append(str(Path(__file__).resolve().parent.parent))
from algorithm.replay_buffer import PrioritizedReplayBuffer


class Replay(object):
    def __init__(self, argv):
        self.config, self.net_config, self.replay_config = self._init_config(argv)
        self.learner_base_url = f'http://{self.net_config["learner_host"]}:{self.net_config["learner_port"]}'

        self._run()

    def _init_config(self, argv):
        config = dict()

        with open(f'{Path(__file__).resolve().parent}/default_config.yaml') as f:
            default_config_file = yaml.load(f, Loader=yaml.FullLoader)
            config = default_config_file

        # define command line arguments
        try:
            opts, args = getopt.getopt(argv, 'c:p:', ['config=',
                                                      'replay_port=',
                                                      'logger_file='])
        except getopt.GetoptError:
            raise Exception('ARGS ERROR')

        for opt, arg in opts:
            if opt in ('-c', '--config'):
                with open(arg) as f:
                    config_file = yaml.load(f, Loader=yaml.FullLoader)
                    for k, v in config_file.items():
                        if v is not None:
                            for kk, vv in v.items():
                                config[k][kk] = vv
                break

        logger_file = None
        for opt, arg in opts:
            if opt == '--replay_port':
                config['net_config']['replay_port'] = int(arg)
            elif opt == '--logger_file':
                logger_file = arg

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

        _log = logging.getLogger('werkzeug')
        _log.setLevel(level=logging.ERROR)

        self.logger = logging.getLogger('sac.ds.replay')
        self.logger.setLevel(level=logging.INFO)

        if logger_file is not None:
            # create file handler
            fh = logging.handlers.RotatingFileHandler(logger_file, maxBytes=1024 * 100, backupCount=5)
            fh.setLevel(logging.INFO)

            # add handler and formatter to logger
            fh.setFormatter(logging.Formatter('%(asctime)-15s [%(levelname)s] - [%(name)s] - %(message)s'))
            self.logger.addHandler(fh)

        # display config
        config_str = ''
        for k, v in config.items():
            config_str += f'\n{k}'
            for kk, vv in v.items():
                config_str += f'\n{kk:>25}: {vv}'
        self.logger.info(config_str)

        return config['base_config'], config['net_config'], config['replay_config']

    def _run(self):
        app = Flask('replay')

        replay_buffer = PrioritizedReplayBuffer(**self.replay_config)

        trans_cache_lock = threading.Lock()
        trans_cache = TransCache()
        cache_max_size = 256

        def _add_trans(*trans):
            # n_states, n_actions, n_rewards, state_, done, mu_n_probs, rnn_state
            trans = list(trans)
            if self.config['use_rnn']:
                tmp_trans = [t.tolist() for t in trans[:5] + [trans[6]]]
            else:
                tmp_trans = [t.tolist() for t in trans[:5]]

            # get td_errors
            try:
                r = requests.post(f'{self.learner_base_url}/get_td_errors',
                                  json=tmp_trans, timeout=2)
            except Exception as e:
                self.logger.error(f'_add_trans: {str(e)}')
                return

            td_errors = r.json()

            replay_buffer.add_with_td_errors(td_errors, *trans)

            self.logger.info(f'buffer_size: {replay_buffer.size}/{replay_buffer.capacity}, {replay_buffer.size/replay_buffer.capacity*100:.2f}%')

        @app.route('/update', methods=['POST'])
        def update():
            data = request.get_json()

            pointers = data['pointers']
            td_errors = data['td_errors']

            replay_buffer.update(pointers, td_errors)

            return jsonify({
                'succeeded': True
            })

        @app.route('/update_transitions', methods=['POST'])
        def update_transitions():
            data = request.get_json()

            pointers = data['pointers']
            index = data['index']
            transition_data = data['data']

            replay_buffer.update_transitions(pointers, index, transition_data)

            return jsonify({
                'succeeded': True
            })

        @app.route('/sample')
        def sample():
            sampled = replay_buffer.sample()
            if sampled is None:
                return jsonify({})

            points, trans, priority_is = sampled
            trans = [t.tolist() for t in trans]

            return jsonify({
                'pointers': points.tolist(),
                'trans': trans,
                'priority_is': priority_is.tolist()
            })

        @app.route('/add', methods=['POST'])
        def add():
            trans = request.get_json()
            trans = [np.array(t, dtype=np.float32) for t in trans]

            with trans_cache_lock:
                trans_cache.add(*trans)
                if trans_cache.size >= cache_max_size:
                    trans = trans_cache.get_trans_and_clear()
                    _add_trans(*trans)

            return jsonify({
                'succeeded': True
            })

        @app.route('/clear')
        def clear():
            replay_buffer.clear()
            self.logger.info('replay buffer cleared')
            return jsonify({
                'succeeded': True
            })

        app.run(host='0.0.0.0', port=self.net_config['replay_port'])

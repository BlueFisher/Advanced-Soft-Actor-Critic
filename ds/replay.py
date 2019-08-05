from pathlib import Path
import getopt
import logging
import sys
import threading
import yaml

from flask import Flask, jsonify, request
import requests
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from algorithm.replay_buffer import PrioritizedReplayBuffer


class Replay(object):
    _replay_port = 61000
    _learner_host = '127.0.0.1'
    _learner_port = 61001
    _replay_config = {}

    def __init__(self, argv):
        self._init_config(argv)
        self._run()

    def _init_config(self, argv):
        try:
            opts, args = getopt.getopt(argv, 'c:p:', ['config=',
                                                      'replay_port='])
        except getopt.GetoptError:
            raise Exception('ARGS ERROR')

        for opt, arg in opts:
            if opt in ('-c', '--config'):
                with open(arg) as f:
                    config_file = yaml.load(f, Loader=yaml.FullLoader)
                    for k, v in config_file.items():
                        if k == 'replay_port':
                            self._replay_port = v
                        elif k == 'learner_host':
                            self._learner_host = v
                        elif k == 'learner_port':
                            self._learner_port = v

                        elif k == 'replay_config':
                            self._replay_config = {} if v is None else v
                            
            elif opt in ('-p', '--replay_port'):
                self._replay_port = int(arg)

            # logger config
            _log = logging.getLogger()
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

    def _run(self):
        app = Flask('replay')

        _tmp_trans_arr_lock = threading.Lock()

        replay_buffer = PrioritizedReplayBuffer(self._replay_config['batch_size'],
                                                self._replay_config['capacity'],
                                                self._replay_config['alpha'])
        _tmp_trans_arr = []

        def _add_trans(*trans):
            # s, a, r, s_, done, gamma, n_states, n_actions, mu_n_probs
            while True:
                try:
                    r = requests.post(f'http://{self._learner_host}:{self._learner_port}/get_td_errors',
                                      json=trans[:6])
                except Exception as e:
                    self.logger.error(f'_clear_replay_buffer: {str(e)}')
                else:
                    break

            td_errors = r.json()

            replay_buffer.add_with_td_errors(td_errors, *trans)

            self.logger.info(f'buffer_size: {replay_buffer.size}/{replay_buffer.capacity}, {replay_buffer.size/replay_buffer.capacity*100:.2f}%')

        @app.route('/update', methods=['POST'])
        def update():
            data = request.get_json()
            points = data['points']
            td_errors = data['td_errors']

            replay_buffer.update(points, td_errors)

            _tmp_trans_arr_lock.acquire()
            for trans in _tmp_trans_arr:
                _add_trans(*trans)
            _tmp_trans_arr.clear()
            _tmp_trans_arr_lock.release()

            return jsonify({
                'succeeded': True
            })

        @app.route('/update_transitions', methods=['POST'])
        def update_transitions():
            data = request.get_json()
            transition_idx = data['transition_idx']
            points = data['points']
            transition_data = data['data']

            replay_buffer.update_transitions(transition_idx, points, transition_data)

            return jsonify({
                'succeeded': True
            })

        learning_starts = 2048

        @app.route('/sample')
        def sample():
            if replay_buffer.size < learning_starts:
                return jsonify({})

            points, trans, is_weights = replay_buffer.sample()

            return jsonify({
                'points': points.tolist(),
                'trans': trans,
                'is_weights': is_weights.tolist()
            })

        @app.route('/add', methods=['POST'])
        def add():
            trans = request.get_json()

            if replay_buffer.size < learning_starts:
                _add_trans(*trans)
            else:
                _tmp_trans_arr_lock.acquire()
                _tmp_trans_arr.append(trans)
                _tmp_trans_arr_lock.release()

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

        app.run(host='0.0.0.0', port=self._replay_port)

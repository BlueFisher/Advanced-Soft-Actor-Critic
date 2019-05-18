import sys
import logging
import getopt
import yaml
from pathlib import Path
import threading

from flask import Flask, jsonify, request
import requests
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from algorithm.replay_buffer import PrioritizedReplayBuffer

logger = logging.getLogger('sac.ds')


class Replay(object):
    _replay_port = 61000
    _learner_host = '127.0.0.1'
    _learner_port = 61001
    _batch_size = 256
    _capacity = 1e6

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
                        elif k == 'batch_size':
                            self._batch_size = v
                        elif k == 'capacity':
                            self._capacity = v
            elif opt in ('-p', '--replay_port'):
                self._replay_port = int(arg)

    def _run(self):
        app = Flask('replay')

        _tmp_trans_arr_lock = threading.Lock()

        replay_buffer = PrioritizedReplayBuffer(self._batch_size, self._capacity)
        _tmp_trans_arr = []

        def _add_trans(*trans):
            while True:
                try:
                    r = requests.post(f'http://{self._learner_host}:{self._learner_port}/get_td_errors',
                                      json=[t.tolist() for t in trans])
                except Exception as e:
                    logger.error(f'_clear_replay_buffer: {str(e)}')
                    time.sleep(1)
                else:
                    break

            td_errors = r.json()

            replay_buffer.add_with_td_errors(td_errors, trans)

            logger.info(f'buffer_size: {replay_buffer.size}/{replay_buffer.capacity}, {replay_buffer.size/replay_buffer.capacity*100:.2f}%')

        @app.route('/update', methods=['POST'])
        def update():
            data = request.get_json()
            points = np.array(data['points'])
            td_errors = np.array(data['td_errors'])

            replay_buffer.update(points, td_errors)

            print(np.sum(replay_buffer.get_leaves() == 1))

            _tmp_trans_arr_lock.acquire()
            for trans in _tmp_trans_arr:
                _add_trans(*trans)
            _tmp_trans_arr.clear()
            _tmp_trans_arr_lock.release()

            return jsonify({
                'succeeded': True
            })

        learning_starts = 2048

        @app.route('/sample')
        def sample():
            if replay_buffer.size < learning_starts:
                return jsonify({})

            points, trans, is_weights = replay_buffer.sample()

            trans = [t.tolist() for t in trans]

            return jsonify({
                'points': points.tolist(),
                'trans': trans,
                'is_weights': is_weights.tolist()
            })

        @app.route('/add', methods=['POST'])
        def add():
            trans = request.get_json()
            trans = [np.array(t) for t in trans]

            
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
            logger.info('replay buffer cleared')
            return jsonify({
                'succeeded': True
            })

        app.run(host='0.0.0.0', port=self._replay_port)

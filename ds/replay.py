import sys
import getopt
import yaml
from pathlib import Path

from flask import Flask, jsonify, request
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from algorithm.replay_buffer import PrioritizedReplayBuffer


class Replay(object):
    _replay_port = 8888
    _batch_size = 256
    _capacity = 1e6

    def __init__(self, argv):
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
                        elif k == 'batch_size':
                            self._batch_size = v
                        elif k == 'capacity':
                            self._capacity = v
            elif opt in ('-p', '--replay_port'):
                self._replay_port = int(arg)
        
        self._run()

    def _run(self):
        app = Flask('replay')

        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

        replay_buffer = PrioritizedReplayBuffer(self._batch_size, self._capacity)
        _tmp_trans_arr = []

        @app.route('/update', methods=['POST'])
        def update():
            data = request.get_json()
            points = np.array(data['points'])
            td_errors = np.array(data['td_errors'])

            replay_buffer.update(points, td_errors)

            for trans in _tmp_trans_arr:
                replay_buffer.add(*trans)

            _tmp_trans_arr.clear()

            return jsonify({
                'succeeded': True
            })

        @app.route('/sample')
        def sample():
            if not replay_buffer.is_lg_batch_size:
                return jsonify({})

            points, trans, is_weights = replay_buffer.sample()

            for i, p in enumerate(trans):
                trans[i] = p.tolist()

            return jsonify({
                'points': points.tolist(),
                'trans': trans,
                'is_weights': is_weights.tolist()
            })

        @app.route('/add', methods=['POST'])
        def add():
            trans = request.get_json()

            for i, p in enumerate(trans):
                trans[i] = np.array(p)

            if not replay_buffer.is_lg_batch_size:
                replay_buffer.add(*trans)
            else:
                _tmp_trans_arr.append(trans)

            return jsonify({
                'succeeded': True
            })

        app.run(host='0.0.0.0', port=self._replay_port)

from pathlib import Path
import asyncio
import importlib
import logging
import shutil
import sys
import threading
import time

import websockets
from flask import Flask, jsonify, request

import numpy as np
import tensorflow as tf

from learner import Learner, WebsocketServer
from sac_ds_with_replay_base import SAC_DS_with_Replay_Base
from trans_cache import TransCache

sys.path.append(str(Path(__file__).resolve().parent.parent))
from mlagents.envs.environment import UnityEnvironment
from algorithm.agent import Agent


class ReplayLearner(Learner):
    _trans_cache_lock = threading.Lock()

    def __init__(self, argv, agent_class=Agent):
        self._agent_class = agent_class
        self._now = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))

        (self.config,
         self.net_config,
         self.reset_config,
         replay_config,
         sac_config,
         model_root_path) = self._init_config(argv)

        self._init_env(replay_config, sac_config, model_root_path)
        self._trans_cache = TransCache()
        # n_states, n_actions, n_rewards, state_, done, rnn_state, mu_n_probs
        self._run()

    def _init_env(self, replay_config, sac_config, model_root_path):
        if self.config['build_path'] is None or self.config['build_path'] == '':
            self.env = UnityEnvironment()
        else:
            self.env = UnityEnvironment(file_name=self.config['build_path'],
                                        no_graphics=True,
                                        base_port=self.config['build_port'],
                                        args=['--scene', self.config['scene']])

        self.logger.info(f'{self.config["build_path"]} initialized')

        self.default_brain_name = self.env.brain_names[0]

        brain_params = self.env.brains[self.default_brain_name]
        state_dim = brain_params.vector_observation_space_size * brain_params.num_stacked_vector_observations
        action_dim = brain_params.vector_action_space_size[0]

        custom_sac_model = importlib.import_module(self.config['sac'])
        shutil.copyfile(f'{self.config["sac"]}.py', f'{model_root_path}/{self.config["sac"]}.py')

        self.sac = SAC_DS_with_Replay_Base(state_dim=state_dim,
                                           action_dim=action_dim,
                                           model_root_path=model_root_path,
                                           model=custom_sac_model,
                                           use_rnn=self.config['use_rnn'],

                                           replay_config=replay_config,

                                           burn_in_step=self.config['burn_in_step'],
                                           n_step=self.config['n_step'],
                                           **sac_config)

    def _run_learner_server(self):
        app = Flask('learner')

        @app.route('/get_policy_variables')
        def get_policy_variables():
            with self._training_lock:
                variables = self.sac.get_policy_variables()

            return jsonify(variables)

        @app.route('/add', methods=['POST'])
        def add():
            trans = request.get_json()
            trans = [np.array(t, dtype=np.float32) for t in trans]

            with self._trans_cache_lock:
                self._trans_cache.add(*trans)

            return jsonify({
                'succeeded': True
            })

        app.run(host='0.0.0.0', port=self.net_config['learner_port'])

    def _run_training_client(self):
        # asyncio.run(self._websocket_server.send_to_all('aaa'))

        while True:
            if self._trans_cache.size > 0:
                with self._trans_cache_lock:
                    trans = self._trans_cache.get_trans_and_clear()

                    with self._training_lock:
                        # n_states, n_actions, n_rewards, state_, done, mu_n_probs, rnn_state
                        if self.config['use_rnn']:
                            td_errors = self.sac.get_td_error(*trans[:5], trans[6]).numpy()
                        else:
                            td_errors = self.sac.get_td_error(*trans[:5]).numpy()

                    self.sac.add_with_td_errors(td_errors.flatten(), *trans)

            _t = time.time()

            with self._training_lock:
                result = self.sac.train()

            if result is not None:
                self.logger.debug(f'train {time.time() - _t}')
            else:
                time.sleep(0.1)

import time
import sys
import logging
import os
import getopt
import importlib
import yaml
from pathlib import Path

import threading
import asyncio

import websockets
from flask import Flask, jsonify, request

import numpy as np
import tensorflow as tf

from learner import Learner, WebsocketServer
from sac_ds_with_replay_base import SAC_DS_with_Replay_Base

sys.path.append(str(Path(__file__).resolve().parent.parent))
from mlagents.envs import UnityEnvironment
from algorithm.agent import Agent

logger = logging.getLogger('sac.ds')
NOW = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))


class ReplayLearner(Learner):
    def __init__(self, argv, agent_class=Agent):
        self._agent_class = agent_class

        self._config, self._reset_config, replay_config, agent_config = self._init_config(argv)
        self._init_env(self._config['sac'], self._config['name'], replay_config, agent_config)
        self._run()

    def _init_env(self, sac, name, replay_config, agent_config):
        self.env = UnityEnvironment(file_name=self._build_path,
                                    no_graphics=True,
                                    base_port=self._build_port)

        self.default_brain_name = self.env.brain_names[0]

        brain_params = self.env.brains[self.default_brain_name]
        state_dim = brain_params.vector_observation_space_size
        action_dim = brain_params.vector_action_space_size[0]

        class SAC(importlib.import_module(sac).SAC_Custom, SAC_DS_with_Replay_Base):
            pass

        self.sac = SAC(state_dim=state_dim,
                       action_dim=action_dim,
                       model_root_path=self.model_root_path,
                       replay_config=replay_config,
                       **agent_config)

    def _run_learner_server(self):
        app = Flask('learner')

        @app.route('/get_policy_variables')
        def get_policy_variables():
            variables = self.sac.get_policy_variables()
            return jsonify(variables)

        @app.route('/add', methods=['POST'])
        def add():
            trans = request.get_json()
            trans = [np.array(t) for t in trans]

            td_errors = self.sac.get_td_error(*trans)
            self.sac.add_with_td_errors(td_errors.flatten(), *trans)
            # self.sac.add(*trans)
            logger.debug('add')

            return jsonify({
                'succeeded': True
            })

        app.run(host='0.0.0.0', port=self._learner_port)

    def _run_training_client(self):
        # asyncio.run(self._websocket_server.send_to_all('aaa'))
        t_evaluation = threading.Thread(target=self._start_policy_evaluation)

        while True:
            if self.sac.train() is not None:
                if not t_evaluation.is_alive():
                    t_evaluation.start()
            else:
                time.sleep(0.1)

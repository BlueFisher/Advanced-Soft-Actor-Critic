from pathlib import Path
import asyncio
import getopt
import importlib
import json
import logging
import os
import sys
import threading
import time
import yaml

import websockets
from flask import Flask, jsonify, request
import requests

import numpy as np
import tensorflow as tf

from sac_ds_base import SAC_DS_Base

sys.path.append(str(Path(__file__).resolve().parent.parent))
from mlagents.envs import UnityEnvironment
from algorithm.agent import Agent

logger = logging.getLogger('sac.ds')
NOW = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))


class Learner(object):
    _replay_host = '127.0.0.1'
    _replay_port = 61000
    _learner_port = 61001
    _websocket_port = 61002
    _build_path = None
    _build_port = 5006

    def __init__(self, argv, agent_class=Agent):
        self._agent_class = agent_class

        self._config, self._reset_config, _, agent_config = self._init_config(argv)
        self._init_env(self._config['sac'], self._config['name'], agent_config)
        self._run()

    def _init_config(self, argv):
        config = {
            'name': NOW,
            'sac': 'sac',
            'save_model_per_step': 10000,
            'update_policy_variables_per_step': 100,
            'add_trans_threshold': 100,
            'reset_on_iteration': True,
            'gamma': 0.99,
            'n_step': 1
        }
        reset_config = {
            'copy': 1
        }
        replay_config = dict()
        agent_config = dict()

        try:
            opts, args = getopt.getopt(argv, 'c:en:n:', ['config=',
                                                         'replay_host',
                                                         'replay_port',
                                                         'learner_port=',
                                                         'build_port=',
                                                         'name=',
                                                         'sac='])
        except getopt.GetoptError:
            raise Exception('ARGS ERROR')

        # config from file
        for opt, arg in opts:
            if opt in ('-c', '--config'):
                with open(arg) as f:
                    config_file = yaml.load(f, Loader=yaml.FullLoader)
                    for k, v in config_file.items():
                        if k == 'replay_host':
                            self._replay_host = v
                        elif k == 'replay_port':
                            self._replay_port = v
                        elif k == 'learner_port':
                            self._learner_port = v
                        elif k == 'websocket_port':
                            self._websocket_port = v
                        elif k == 'build_path':
                            self._build_path = v[sys.platform]

                        elif k == 'reset_config':
                            reset_config = dict(reset_config, **({} if v is None else v))

                        elif k == 'replay_config':
                            replay_config = {} if v is None else v

                        elif k == 'sac_config':
                            agent_config = {} if v is None else v

                        elif k in config.keys():
                            config[k] = v
                break

        # config from console
        for opt, arg in opts:
            if opt == 'learner_port':
                config['learner_port'] == int(arg)
            elif opt == '--build_port':
                self._build_port = int(arg)
            elif opt in ('-n', '--name'):
                config['name'] = arg.replace('{time}', NOW)
            elif opt == '--sac':
                config['sac'] = arg

        self.model_root_path = f'models/{config["name"]}'

        if not os.path.exists(self.model_root_path):
            os.makedirs(self.model_root_path)
        with open(f'{self.model_root_path}/config.yaml', 'w') as f:
            yaml.dump({**config,
                       'reset_config': {**reset_config},
                       'replay_config': {**replay_config},
                       'agents_config': {**agent_config}
                       }, f, default_flow_style=False)

        config_str = '\ncommon_config'
        for k, v in config.items():
            config_str += f'\n{k:>25}: {v}'

        config_str += '\nreset_config:'
        for k, v in reset_config.items():
            config_str += f'\n{k:>25}: {v}'

        config_str += '\nreplay_config:'
        for k, v in replay_config.items():
            config_str += f'\n{k:>25}: {v}'

        config_str += '\nagent_config:'
        for k, v in agent_config.items():
            config_str += f'\n{k:>25}: {v}'
        logger.info(config_str)

        return config, reset_config, replay_config, agent_config

    def _init_env(self, sac, name, agent_config):
        self.env = UnityEnvironment(file_name=self._build_path,
                                    no_graphics=True,
                                    base_port=self._build_port)

        self.default_brain_name = self.env.brain_names[0]

        brain_params = self.env.brains[self.default_brain_name]
        state_dim = brain_params.vector_observation_space_size
        action_dim = brain_params.vector_action_space_size[0]

        class SAC(importlib.import_module(sac).SAC_Custom, SAC_DS_Base):
            pass

        self.sac = SAC(state_dim=state_dim,
                       action_dim=action_dim,
                       model_root_path=self.model_root_path,
                       **agent_config)

    def _start_policy_evaluation(self):
        eval_step = 0
        start_time = time.time()
        brain_info = self.env.reset(train_mode=True, config=self._reset_config)[self.default_brain_name]

        while True:
            if self.env.global_done or self._config['reset_on_iteration']:
                brain_info = self.env.reset(train_mode=True)[self.default_brain_name]

            agents = [self._agent_class(i,
                                        self._config['gamma'],
                                        self._config['n_step'])
                      for i in brain_info.agents]

            states = brain_info.vector_observations

            while False in [a.done for a in agents] and not self.env.global_done:
                actions = self.sac.choose_action(states)
                brain_info = self.env.step({
                    self.default_brain_name: actions
                })[self.default_brain_name]

                states_ = brain_info.vector_observations

                for i, agent in enumerate(agents):
                    agent.add_transition(states[i],
                                         actions[i],
                                         brain_info.rewards[i],
                                         brain_info.local_done[i],
                                         brain_info.max_reached[i],
                                         states_[i])

                states = states_

            self._log_episode_info(eval_step, start_time, agents)
            eval_step += 1

    def _log_episode_info(self, eval_step, start_time, agents):
        rewards = np.array([a.reward for a in agents])

        self.sac.write_constant_summaries([
            {'tag': 'reward/mean', 'simple_value': rewards.mean()},
            {'tag': 'reward/max', 'simple_value': rewards.max()},
            {'tag': 'reward/min', 'simple_value': rewards.min()}
        ], eval_step)

        time_elapse = (time.time() - start_time) / 60
        rewards_sorted = ", ".join([f"{i:.1f}" for i in sorted(rewards)])
        logger.info(f'{eval_step}, {time_elapse:.2f}min, rewards {rewards_sorted}')

    def _run_learner_server(self):
        app = Flask('learner')

        @app.route('/get_policy_variables')
        def get_policy_variables():
            variables = self.sac.get_policy_variables()
            return jsonify(variables)

        @app.route('/get_td_errors', methods=['POST'])
        def get_td_errors():
            trans = request.get_json()
            td_errors = self.sac.get_td_error(*trans)
            return jsonify(td_errors.flatten().tolist())

        app.run(host='0.0.0.0', port=self._learner_port)

    def _get_sampled_data(self):
        while True:
            try:
                r = requests.get(f'http://{self._replay_host}:{self._replay_port}/sample')
            except requests.ConnectionError:
                logger.error(f'get_sampled_data connecting error')
                time.sleep(1)
            except Exception as e:
                logger.error(f'get_sampled_data error {type(e)}, {str(e)}')
                time.sleep(1)
            else:
                break
        return r.json()

    def _update_td_errors(self, points, td_errors):
        while True:
            try:
                requests.post(f'http://{self._replay_host}:{self._replay_port}/update',
                              json={
                                  'points': points,
                                  'td_errors': td_errors
                              })
            except requests.ConnectionError:
                logger.error(f'update_td_errors connecting error')
                time.sleep(1)
            except Exception as e:
                logger.error(f'update_td_errors error {type(e)}, {str(e)}')
                time.sleep(1)
            else:
                break

    def _clear_replay_buffer(self):
        while True:
            try:
                requests.get(f'http://{self._replay_host}:{self._replay_port}/clear')
            except requests.ConnectionError:
                logger.error(f'clear_replay_buffer connecting error')
                time.sleep(1)
            except Exception as e:
                logger.error(f'clear_replay_buffer error {type(e)}, {str(e)}')
                time.sleep(1)
            else:
                break

    def _run_training_client(self):
        # asyncio.run(self._websocket_server.send_to_all('aaa'))
        t_evaluation = threading.Thread(target=self._start_policy_evaluation)
        self._clear_replay_buffer()

        while True:
            data = self._get_sampled_data()

            if data:
                if not t_evaluation.is_alive():
                    t_evaluation.start()

                trans = data['trans']
                is_weights = data['is_weights']

                is_weights = np.array(is_weights)
                for i, p in enumerate(trans):
                    trans[i] = np.array(p)

                s, a, r, s_, done = trans
                curr_step, td_errors = self.sac.train(s, a, r, s_, done, is_weights)
                self._update_td_errors(data['points'], td_errors.tolist())

    def _run(self):
        sac_reset_config = {
            'update_policy_variables_per_step': self._config['update_policy_variables_per_step'],
            'add_trans_threshold': self._config['add_trans_threshold'],
            'gamma': self._config['gamma'],
            'n_step': self._config['n_step'],
        }
        self._websocket_server = WebsocketServer(sac_reset_config, self._websocket_port)

        t_learner = threading.Thread(target=self._run_learner_server)
        t_training = threading.Thread(target=self._run_training_client)
        t_learner.start()
        t_training.start()

        asyncio.get_event_loop().run_forever()


class WebsocketServer:
    _websocket_clients = set()

    def __init__(self, sac_reset_config, port=61002):
        self._sac_reset_config = sac_reset_config

        start_server = websockets.serve(self._websocket_open, '0.0.0.0', port)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(start_server)
        logger.info('websocket server started')

    async def _websocket_open(self, websocket, path):
        try:
            async for raw_message in websocket:
                message = json.loads(raw_message)
                if message['cmd'] == 'actor':
                    self._websocket_clients.add(websocket)
                    self.print_websocket_clients()
                    message = {
                        'cmd': 'reset',
                        'config': self._sac_reset_config
                    }
                    await websocket.send(json.dumps(message))
        except websockets.ConnectionClosed:
            try:
                self._websocket_clients.remove(websocket)
            except:
                pass
            else:
                self.print_websocket_clients()

    def print_websocket_clients(self):
        log_str = f'{len(self._websocket_clients)} active actors'
        for i, client in enumerate(self._websocket_clients):
            log_str += (f'\n[{i+1}]. {client.remote_address[0]} : {client.remote_address[1]}')

        logger.info(log_str)

    async def send_to_all(self, message):
        tasks = []
        try:
            for client in self._websocket_clients:
                tasks.append(client.send(message))
            await asyncio.gather(*tasks)
        except websockets.ConnectionClosed:
            pass

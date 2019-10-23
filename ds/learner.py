from pathlib import Path
import asyncio
import getopt
import importlib
import json
import logging
import os
import shutil
import sys
import threading
import time
import yaml

import websockets
from flask import Flask, jsonify, request
import requests

import numpy as np

from sac_ds_base import SAC_DS_Base

sys.path.append(str(Path(__file__).resolve().parent.parent))
from mlagents.envs.environment import UnityEnvironment
from algorithm.agent import Agent


class Learner(object):
    _training_lock = threading.Lock()

    def __init__(self, argv, agent_class=Agent):
        self._agent_class = agent_class
        self._now = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))

        (self.config,
         self.net_config,
         self.reset_config,
         _,
         sac_config,
         model_root_path) = self._init_config(argv)

        self.replay_base_url = f'http://{self.net_config["replay_host"]}:{self.net_config["replay_port"]}'

        self._init_env(sac_config, model_root_path)
        self._run()

    def _init_config(self, argv):
        config = dict()

        with open(f'{Path(__file__).resolve().parent}/default_config.yaml') as f:
            default_config_file = yaml.load(f, Loader=yaml.FullLoader)
            config = default_config_file

        # define command line arguments
        try:
            opts, args = getopt.getopt(argv, 'c:', ['run',
                                                    'config=',
                                                    'build_path=',
                                                    'build_port=',
                                                    'logger_file=',
                                                    'sac=',
                                                    'agents='])
        except getopt.GetoptError:
            raise Exception('ARGS ERROR')

        # initialize config from config.yaml
        for opt, arg in opts:
            if opt in ('-c', '--config'):
                with open(arg) as f:
                    config_file = yaml.load(f, Loader=yaml.FullLoader)
                    for k, v in config_file.items():
                        assert k in config.keys(), f'{k} is invalid'
                        if v is not None:
                            for kk, vv in v.items():
                                assert kk in config[k].keys(), f'{kk} is invalid in {k}'
                                config[k][kk] = vv
                break

        config['base_config']['build_path'] = config['base_config']['build_path'][sys.platform]

        logger_file = None
        for opt, arg in opts:
            if opt == '--run':
                self._train_mode = False
            elif opt == '--build_path':
                config['base_config']['build_path'] = arg
            elif opt == '--build_port':
                config['base_config']['build_port'] = int(arg)
            elif opt == '--logger_file':
                logger_file = arg
            elif opt == '--sac':
                config['base_config']['sac'] = arg
            elif opt == '--agents':
                config['reset_config']['copy'] = int(arg)

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

        self.logger = logging.getLogger('sac.ds.learner')
        self.logger.setLevel(level=logging.INFO)

        if logger_file is not None:
            # create file handler
            fh = logging.handlers.RotatingFileHandler(logger_file, maxBytes=1024 * 100, backupCount=5)
            fh.setLevel(logging.INFO)

            # add handler and formatter to logger
            fh.setFormatter(logging.Formatter('%(asctime)-15s [%(levelname)s] - [%(name)s] - %(message)s'))
            self.logger.addHandler(fh)

        config['base_config']['name'] = config['base_config']['name'].replace('{time}', self._now)
        model_root_path = f'models/{config["base_config"]["name"]}'

        # save config
        if not os.path.exists(model_root_path):
            os.makedirs(model_root_path)
        with open(f'{model_root_path}/config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        # display config
        config_str = ''
        for k, v in config.items():
            config_str += f'\n{k}'
            for kk, vv in v.items():
                config_str += f'\n{kk:>25}: {vv}'
        self.logger.info(config_str)

        return (config['base_config'],
                config['net_config'],
                config['reset_config'],
                config['replay_config'],
                config['sac_config'],
                model_root_path)

    def _init_env(self, sac_config, model_root_path):
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

        self.sac = SAC_DS_Base(state_dim=state_dim,
                               action_dim=action_dim,
                               model_root_path=model_root_path,
                               model=custom_sac_model,
                               use_rnn=self.config['use_rnn'],

                               burn_in_step=self.config['burn_in_step'],
                               n_step=self.config['n_step'],
                               **sac_config)

    def _start_policy_evaluation(self):
        iteration = 0
        start_time = time.time()

        brain_info = self.env.reset(train_mode=False, config=self.reset_config)[self.default_brain_name]
        if self.config['use_rnn']:
            initial_rnn_state = self.sac.get_initial_rnn_state(len(brain_info.agents))
            rnn_state = initial_rnn_state

        while True:
            if self.config['reset_on_iteration']:
                brain_info = self.env.reset(train_mode=False)[self.default_brain_name]

            agents = [self._agent_class(i,
                                        tran_len=self.config['burn_in_step'] + self.config['n_step'],
                                        stagger=self.config['stagger'],
                                        use_rnn=self.config['use_rnn'])
                      for i in brain_info.agents]

            states = brain_info.vector_observations
            step = 0

            while False in [a.done for a in agents]:
                with self._training_lock:
                    if self.config['use_rnn']:
                        actions, next_rnn_state = self.sac.choose_rnn_action(states.astype(np.float32),
                                                                             rnn_state)
                        next_rnn_state = next_rnn_state.numpy()
                    else:
                        actions = self.sac.choose_action(states.astype(np.float32))

                actions = actions.numpy()

                brain_info = self.env.step({
                    self.default_brain_name: actions
                })[self.default_brain_name]

                states_ = brain_info.vector_observations
                if step == self.config['max_step']:
                    brain_info.local_done = [True] * len(brain_info.agents)
                    brain_info.max_reached = [True] * len(brain_info.agents)

                for i, agent in enumerate(agents):
                    agent.add_transition(states[i],
                                         actions[i],
                                         brain_info.rewards[i],
                                         brain_info.local_done[i],
                                         brain_info.max_reached[i],
                                         states_[i],
                                         rnn_state[i] if self.config['use_rnn'] else None)

                states = states_
                if self.config['use_rnn']:
                    rnn_state = next_rnn_state
                    rnn_state[brain_info.local_done] = initial_rnn_state[brain_info.local_done]

                step += 1

            with self._training_lock:
                self._log_episode_info(iteration, start_time, agents)

            iteration += 1

    def _log_episode_info(self, iteration, start_time, agents):
        rewards = np.array([a.reward for a in agents])

        self.sac.write_constant_summaries([
            {'tag': 'reward/mean', 'simple_value': rewards.mean()},
            {'tag': 'reward/max', 'simple_value': rewards.max()},
            {'tag': 'reward/min', 'simple_value': rewards.min()}
        ], iteration)

        time_elapse = (time.time() - start_time) / 60
        rewards_sorted = ", ".join([f"{i:.1f}" for i in sorted(rewards)])
        self.logger.info(f'{iteration}, {time_elapse:.2f}min, rewards {rewards_sorted}')

    def _run_learner_server(self):
        app = Flask('learner')

        @app.route('/get_policy_variables')
        def get_policy_variables():
            with self._training_lock:
                variables = self.sac.get_policy_variables()

            return jsonify(variables)

        @app.route('/get_td_errors', methods=['POST'])
        def get_td_errors():
            trans = request.get_json()
            trans = [np.array(t, dtype=np.float32) for t in trans]

            with self._training_lock:
                td_errors = self.sac.get_td_error(*trans)

            return jsonify(td_errors.numpy().flatten().tolist())

        app.run(host='0.0.0.0', port=self.net_config['learner_port'])

    def _get_sampled_data(self):
        while True:
            try:
                r = requests.get(f'{self.replay_base_url}/sample')
            except requests.ConnectionError:
                self.logger.error(f'get_sampled_data connecting error')
                time.sleep(1)
            except Exception as e:
                self.logger.error(f'get_sampled_data error {type(e)}, {str(e)}')
                time.sleep(1)
            else:
                break
        return r.json()

    def _update_td_errors(self, pointers, td_errors):
        while True:
            try:
                requests.post(f'{self.replay_base_url}/update',
                              json={
                                  'pointers': pointers,
                                  'td_errors': td_errors
                              })
            except requests.ConnectionError:
                self.logger.error(f'update_td_errors connecting error')
                time.sleep(1)
            except Exception as e:
                self.logger.error(f'update_td_errors error {type(e)}, {str(e)}')
                time.sleep(1)
            else:
                break

    def _update_transitions(self, pointers, index, data):
        while True:
            try:
                requests.post(f'{self.replay_base_url}/update_transitions',
                              json={
                                  'pointers': pointers,
                                  'index': index,
                                  'data': data
                              })
            except requests.ConnectionError:
                self.logger.error(f'_update_transitions connecting error')
                time.sleep(1)
            except Exception as e:
                self.logger.error(f'update_transitions error {type(e)}, {str(e)}')
                time.sleep(1)
            else:
                break

    def _clear_replay_buffer(self):
        while True:
            try:
                requests.get(f'{self.replay_base_url}/clear')
            except requests.ConnectionError:
                self.logger.error(f'clear_replay_buffer connecting error')
                time.sleep(1)
            except Exception as e:
                self.logger.error(f'clear_replay_buffer error {type(e)}, {str(e)}')
                time.sleep(1)
            else:
                break

    def _run_training_client(self):
        # asyncio.run(self._websocket_server.send_to_all('aaa'))
        self._clear_replay_buffer()

        while True:
            data = self._get_sampled_data()

            if data:
                pointers = data['pointers']
                trans = [np.array(t, dtype=np.float32) for t in data['trans']]
                priority_is = np.array(data['priority_is'], dtype=np.float32)

                if self.config['use_rnn']:
                    n_states, n_actions, n_rewards, state_, done, mu_n_probs, rnn_state = trans
                else:
                    n_states, n_actions, n_rewards, state_, done, mu_n_probs = trans

                with self._training_lock:
                    td_errors, pi_n_probs = self.sac.train(n_states,
                                                           n_actions,
                                                           n_rewards,
                                                           state_,
                                                           done,
                                                           mu_n_probs,
                                                           priority_is,
                                                           rnn_state if self.config['use_rnn'] else None)

                self._update_td_errors(pointers, td_errors.tolist())
                if self.config['use_rnn']:
                    self._update_transitions(pointers, 5, pi_n_probs.tolist())
            else:
                self.logger.warn('no data sampled')
                time.sleep(1)

    def _run(self):
        # TODO
        self._websocket_server = WebsocketServer({}, self.net_config['websocket_port'])

        t_learner = threading.Thread(target=self._run_learner_server)
        t_training = threading.Thread(target=self._run_training_client)
        t_evaluation = threading.Thread(target=self._start_policy_evaluation)

        t_learner.start()
        t_training.start()
        t_evaluation.start()

        asyncio.get_event_loop().run_forever()


class WebsocketServer:
    _websocket_clients = set()

    def __init__(self, sac_reset_config, port):
        self._sac_reset_config = sac_reset_config

        start_server = websockets.serve(self._websocket_open, '0.0.0.0', port)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(start_server)

        self.logger = logging.getLogger('websocket')
        self.logger.setLevel(level=logging.INFO)

        self.logger.info('websocket server started')

    async def _websocket_open(self, websocket, path):
        try:
            async for raw_message in websocket:
                message = json.loads(raw_message)
                if message['cmd'] == 'actor':
                    self._websocket_clients.add(websocket)
                    self.print_websocket_clients()
                    message = {
                        'cmd': 'reset',
                        'config': {}
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
            log_str += (f'\n\t[{i+1}]. {client.remote_address[0]} : {client.remote_address[1]}')

        self.logger.info(log_str)

    async def send_to_all(self, message):
        tasks = []
        try:
            for client in self._websocket_clients:
                tasks.append(client.send(message))
            await asyncio.gather(*tasks)
        except websockets.ConnectionClosed:
            pass

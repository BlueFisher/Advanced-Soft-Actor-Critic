from pathlib import Path
import asyncio
import functools
import getopt
import importlib
import json
import logging
import logging.handlers
import sys
import time
import yaml

import requests
import websockets

import numpy as np
import tensorflow as tf

from sac_ds_base import SAC_DS_Base
from trans_cache import TransCache

sys.path.append(str(Path(__file__).resolve().parent.parent))
from mlagents.envs.environment import UnityEnvironment
from algorithm.agent import Agent


class Actor(object):
    _train_mode = True
    _websocket_connected = False

    def __init__(self, argv, agent_class=Agent):
        self._agent_class = agent_class

        self.config, net_config, self.reset_config = self._init_config(argv)
        self.replay_base_url = f'http://{net_config["replay_host"]}:{net_config["replay_port"]}'
        self.learner_base_url = f'http://{net_config["learner_host"]}:{net_config["learner_port"]}'
        self.websocket_base_url = f'ws://{net_config["websocket_host"]}:{net_config["websocket_port"]}'

        self._init_websocket_client()
        self._init_env()
        self._trans_cache = TransCache()
        self._run()

    def _init_config(self, argv):
        config = dict()

        with open(f'{Path(__file__).resolve().parent}/default_config.yaml') as f:
            default_config_file = yaml.load(f, Loader=yaml.FullLoader)
            config = default_config_file

        # define command line arguments
        try:
            opts, args = getopt.getopt(argv, 'c:', ['config=',
                                                    'run',
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

        _log = logging.getLogger('tensorflow')
        _log.setLevel(level=logging.ERROR)

        self.logger = logging.getLogger('sac.ds.actor')
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

        return config['base_config'], config['net_config'], config['reset_config']

    def _init_websocket_client(self):
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, lambda: asyncio.run(self._connect_websocket()))

    async def _connect_websocket(self):
        while True:
            try:
                async with websockets.connect(self.websocket_base_url) as websocket:
                    await websocket.send(json.dumps({
                        'cmd': 'actor'
                    }))
                    self.logger.info('websocket connected')
                    while True:
                        try:
                            raw_message = await websocket.recv()
                            message = json.loads(raw_message)
                            if message['cmd'] == 'reset':
                                self._websocket_connected = True
                                self.config = dict(self.config, **message['config'])
                                self.logger.info(f'reinitialize config: {message["config"]}')
                        except websockets.ConnectionClosed:
                            self.logger.error('websocket connection closed')
                            break
                        except json.JSONDecodeError:
                            self.logger.error(f'websocket json decode error, {raw_message}')
            except (ConnectionRefusedError, websockets.InvalidMessage):
                self.logger.error(f'websocket connecting failed')
                time.sleep(1)
            except Exception as e:
                self.logger.error(f'websocket connecting error {type(e)}, {str(e)}')
                time.sleep(1)
            finally:
                self._websocket_connected = False

    def _init_env(self):
        if self.config['build_path'] is None or self.config['build_path'] == '':
            self.env = UnityEnvironment()
        else:
            self.env = UnityEnvironment(file_name=self.config['build_path'],
                                        no_graphics=self._train_mode,
                                        base_port=self.config['build_port'],
                                        args=['--scene', self.config['scene']])

        self.logger.info(f'{self.config["build_path"]} initialized')

        self.default_brain_name = self.env.brain_names[0]

    def _init_sac(self):
        brain_params = self.env.brains[self.default_brain_name]
        state_dim = brain_params.vector_observation_space_size * brain_params.num_stacked_vector_observations
        action_dim = brain_params.vector_action_space_size[0]

        custom_sac_model = importlib.import_module(self.config['sac'])

        self.sac_actor = SAC_DS_Base(state_dim=state_dim,
                                     action_dim=action_dim,
                                     model_root_path=None,
                                     model=custom_sac_model,
                                     use_rnn=self.config['use_rnn'])

        self.logger.info(f'actor initialized')

    def _update_policy_variables(self):
        while True and self._websocket_connected:
            try:
                r = requests.get(f'{self.learner_base_url}/get_policy_variables')

                new_variables = r.json()
                self.sac_actor.update_policy_variables(new_variables)
            except requests.ConnectionError:
                self.logger.error('update_policy_variables connecting error')
                time.sleep(1)
            except Exception as e:
                self.logger.error(f'update_policy_variables error {type(e)}, {str(e)}')
                break
            else:
                break

    def _add_trans(self, *trans):
        # n_states, n_actions, n_rewards, state_, done, mu_n_probs, rnn_state
        self._trans_cache.add(*trans)

        if self._trans_cache.size > self.config['add_trans_threshold']:
            trans = self._trans_cache.get_trans_list_and_clear()
            while True and self._websocket_connected:
                try:
                    requests.post(f'{self.replay_base_url}/add', json=trans)
                except requests.ConnectionError:
                    self.logger.error(f'add_trans connecting error')
                    time.sleep(1)
                except Exception as e:
                    self.logger.error(f'add_trans error {type(e)}, {str(e)}')
                    break
                else:
                    break

    def _run(self):
        iteration = 0

        while True:
            # learner is offline, waiting...
            if not self._websocket_connected:
                iteration = 0
                time.sleep(1)
                continue

            # learner is online, reset all settings
            if iteration == 0 and self._websocket_connected:
                self._trans_cache.clear()
                self._init_sac()

                brain_info = self.env.reset(train_mode=self._train_mode, config=self.reset_config)[self.default_brain_name]
                if self.config['use_rnn']:
                    initial_rnn_state = self.sac_actor.get_initial_rnn_state(len(brain_info.agents))
                    rnn_state = initial_rnn_state

            if self.config['reset_on_iteration']:
                brain_info = self.env.reset(train_mode=self._train_mode)[self.default_brain_name]
                if self.config['use_rnn']:
                    rnn_state = initial_rnn_state

            agents = [self._agent_class(i,
                                        tran_len=self.config['burn_in_step'] + self.config['n_step'],
                                        stagger=self.config['stagger'],
                                        use_rnn=self.config['use_rnn'])
                      for i in brain_info.agents]

            states = brain_info.vector_observations
            step = 0

            if self.config['update_policy_variables_per_step'] == -1:
                self._update_policy_variables()

            while False in [a.done for a in agents] and self._websocket_connected:
                if self.config['update_policy_variables_per_step'] != -1 and step % self.config['update_policy_variables_per_step'] == 0:
                    self._update_policy_variables()

                if self.config['use_rnn']:
                    actions, next_rnn_state = self.sac_actor.choose_rnn_action(states,
                                                                               rnn_state)
                    next_rnn_state = next_rnn_state.numpy()
                else:
                    actions = self.sac_actor.choose_action(states)

                actions = actions.numpy()

                brain_info = self.env.step({
                    self.default_brain_name: actions
                })[self.default_brain_name]

                states_ = brain_info.vector_observations
                if step == self.config['max_step']:
                    brain_info.local_done = [True] * len(brain_info.agents)
                    brain_info.max_reached = [True] * len(brain_info.agents)

                trans_list = [agents[i].add_transition(states[i],
                                                       actions[i],
                                                       brain_info.rewards[i],
                                                       brain_info.local_done[i],
                                                       brain_info.max_reached[i],
                                                       states_[i],
                                                       rnn_state[i] if self.config['use_rnn'] else None)
                              for i in range(len(agents))]

                trans_list = [t for t in trans_list if t is not None]
                if len(trans_list) != 0:
                    # n_states, n_actions, n_rewards, state_, done, rnn_state
                    trans = [np.concatenate(t, axis=0) for t in zip(*trans_list)]

                    if self.config['use_rnn']:
                        n_states, n_actions, n_rewards, state_, done, rnn_state = trans
                        # TODO: only need [:, burn_in_step:, :]
                        mu_n_probs = self.sac_actor.get_n_step_probs(n_states, n_actions, rnn_state).numpy()
                        self._add_trans(n_states, n_actions, n_rewards, state_, done, mu_n_probs, rnn_state)
                    else:
                        n_states, n_actions, n_rewards, state_, done = trans
                        mu_n_probs = self.sac_actor.get_n_step_probs(n_states, n_actions).numpy()
                        self._add_trans(n_states, n_actions, n_rewards, state_, done, mu_n_probs)

                states = states_
                if self.config['use_rnn']:
                    rnn_state = next_rnn_state
                    rnn_state[brain_info.local_done] = initial_rnn_state[brain_info.local_done]

                step += 1

            self._log_episode_info(iteration, agents)
            iteration += 1

    def _log_episode_info(self, iteration, agents):
        rewards = [a.reward for a in agents]
        rewards_sorted = ", ".join([f"{i:.1f}" for i in sorted(rewards)])
        self.logger.info(f'{iteration}, rewards {rewards_sorted}')

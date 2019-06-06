from pathlib import Path
import asyncio
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

sys.path.append(str(Path(__file__).resolve().parent.parent))
from mlagents.envs import UnityEnvironment
from algorithm.agent import Agent

logger = logging.getLogger('sac.ds')


class TransBuffer(object):
    _buffer = None

    def add(self, *args):
        for arg in args:
            assert len(arg.shape) == 2
            assert len(arg) == len(args[0])

        if self._buffer is None:
            self._buffer = list(args)
        else:
            for i in range(len(args)):
                self._buffer[i] = np.concatenate((self._buffer[i], args[i]))

    def get_trans_and_clear(self):
        trans = self._buffer
        self._buffer = None
        return trans

    def clear(self):
        self._buffer = None

    @property
    def size(self):
        if self._buffer is None:
            return 0
        else:
            return len(self._buffer[0])


class Actor(object):
    _replay_host = '127.0.0.1'
    _replay_port = 61000
    _learner_host = '127.0.0.1'
    _learner_port = 61001
    _websocket_host = '127.0.0.1'
    _websocket_port = 61002
    _build_path = None
    _build_port = 5005

    _reset_config = {
        'copy': 1
    }

    _config = {
        'update_policy_variables_per_step': 100,
        'add_trans_threshold': 100,
        'reset_on_iteration': True,
        'gamma': 0.99,
        'n_step': 1,
        'sac': 'sac'
    }

    _train_mode = True
    _websocket_connected = False

    def __init__(self, argv, agent_class=Agent):
        self._agent_class = agent_class

        self._init_config(argv)
        self._init_websocket_client()
        self._init_env()
        self._tmp_trans_buffer = TransBuffer()
        self._run()

    def _init_config(self, argv):
        try:
            opts, args = getopt.getopt(argv, 'c:', ['config=',
                                                    'replay_host=',
                                                    'replay_port=',
                                                    'learner_host=',
                                                    'learner_port=',
                                                    'build_path=',
                                                    'build_port=',
                                                    'logger_file=',
                                                    'sac=',
                                                    'agents=',
                                                    'run'])
        except getopt.GetoptError:
            raise Exception('ARGS ERROR')

        for opt, arg in opts:
            if opt in ('-c', '--config'):
                with open(arg) as f:
                    config_file = yaml.load(f, Loader=yaml.FullLoader)
                    for k, v in config_file.items():
                        if k == 'replay_host':
                            self._replay_host = v
                        elif k == 'replay_port':
                            self._replay_port = v
                        elif k == 'learner_host':
                            self._learner_host = v
                        elif k == 'learner_port':
                            self._learner_port = v
                        elif k == 'websocket_host':
                            self._websocket_host = v
                        elif k == 'websocket_port':
                            self._websocket_port = v
                        elif k == 'build_path':
                            self._build_path = v[sys.platform]

                        elif k == 'reset_config':
                            self._reset_config = dict(self._reset_config, **({} if v is None else v))

                        elif k in self._config.keys():
                            self._config[k] = v
                break

        logger_file = None
        for opt, arg in opts:
            if opt == '--replay_host':
                self._replay_host = arg
            elif opt == '--replay_port':
                self._replay_port = int(arg)
            elif opt == '--learner_host':
                self._learner_host = arg
            elif opt == '--learner_port':
                self._learner_port = int(arg)
            elif opt == '--build_path':
                self._build_path = arg
            elif opt == '--build_port':
                self._build_port = int(arg)
            elif opt == '--logger_file':
                logger_file = arg
            elif opt == '--sac':
                self._config['sac'] = arg
            elif opt == '--agents':
                self._reset_config['copy'] = int(arg)
            elif opt == '--run':
                self._train_mode = False

        # logger config
        if logger_file is not None:
            # create file handler
            fh = logging.handlers.RotatingFileHandler(logger_file, maxBytes=1024 * 100, backupCount=5)
            fh.setLevel(logging.INFO)

            # create formatter
            fmt = "%(asctime)-15s [%(levelname)s] - [%(name)s] - %(message)s"
            formatter = logging.Formatter(fmt)

            # add handler and formatter to logger
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    def _init_websocket_client(self):
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, lambda: asyncio.run(self._connect_websocket()))

    async def _connect_websocket(self):
        while True:
            try:
                async with websockets.connect(f'ws://{self._websocket_host}:{self._websocket_port}') as websocket:
                    await websocket.send(json.dumps({
                        'cmd': 'actor'
                    }))
                    logger.info('websocket connected')
                    while True:
                        try:
                            raw_message = await websocket.recv()
                            message = json.loads(raw_message)
                            if message['cmd'] == 'reset':
                                self._websocket_connected = True
                                self._config = dict(self._config, **message['config'])
                                logger.info(f'reset config: {message["config"]}')
                        except websockets.ConnectionClosed:
                            logger.error('websocket connection closed')
                            break
                        except json.JSONDecodeError:
                            logger.error(f'websocket json decode error, {raw_message}')
            except (ConnectionRefusedError, websockets.InvalidMessage):
                logger.error(f'websocket connecting failed')
                time.sleep(1)
            except Exception as e:
                logger.error(f'websocket connecting error {type(e)}, {str(e)}')
                time.sleep(1)
            finally:
                self._websocket_connected = False

    def _init_env(self):
        if self._build_path is None or self._build_path == '':
            self.env = UnityEnvironment()
        else:
            self.env = UnityEnvironment(file_name=self._build_path,
                                        no_graphics=True,
                                        base_port=self._build_port)

        logger.info(f'{self._build_path} initialized')

        self.default_brain_name = self.env.brain_names[0]

    def _init_sac(self):
        brain_params = self.env.brains[self.default_brain_name]
        state_dim = brain_params.vector_observation_space_size
        action_dim = brain_params.vector_action_space_size[0]

        class SAC(importlib.import_module(self._config['sac']).SAC_Custom, SAC_DS_Base):
            pass

        self.sac_actor = SAC(state_dim,
                             action_dim,
                             model_root_path=None)

        logger.info(f'actor initialized')

    def _update_policy_variables(self):
        while True and self._websocket_connected:
            try:
                r = requests.get(f'http://{self._learner_host}:{self._learner_port}/get_policy_variables')

                new_variables = r.json()
                self.sac_actor.update_policy_variables(new_variables)
            except requests.ConnectionError:
                logger.error('update_policy_variables connecting error')
                time.sleep(1)
            except Exception as e:
                logger.error(f'update_policy_variables error {type(e)}, {str(e)}')
                break
            else:
                break

    def _add_trans(self, *trans):
        self._tmp_trans_buffer.add(*trans)

        if self._tmp_trans_buffer.size > self._config['add_trans_threshold']:
            trans = self._tmp_trans_buffer.get_trans_and_clear()
            while True and self._websocket_connected:
                try:
                    requests.post(f'http://{self._replay_host}:{self._replay_port}/add',
                                  json=[t.tolist() for t in trans])
                except requests.ConnectionError:
                    logger.error(f'add_trans connecting error')
                    time.sleep(1)
                except Exception as e:
                    logger.error(f'add_trans error {type(e)}, {str(e)}')
                    break
                else:
                    break

    def _run(self):
        global_step = 0

        brain_info = self.env.reset(train_mode=self._train_mode, config=self._reset_config)[self.default_brain_name]

        while True:
            if not self._websocket_connected:
                global_step = 0
                time.sleep(1)
                continue

            if global_step == 0 and self._websocket_connected:
                self._tmp_trans_buffer.clear()
                self._init_sac()

            if self.env.global_done or self._config['reset_on_iteration']:
                brain_info = self.env.reset(train_mode=self._train_mode)[self.default_brain_name]

            agents = [self._agent_class(i, self._config['gamma'], self._config['n_step']) for i in brain_info.agents]

            states = brain_info.vector_observations

            while False in [a.done for a in agents] and not self.env.global_done and self._websocket_connected:
                if global_step % self._config['update_policy_variables_per_step'] == 0:
                    self._update_policy_variables()

                actions = self.sac_actor.choose_action(states)
                brain_info = self.env.step({
                    self.default_brain_name: actions
                })[self.default_brain_name]

                states_ = brain_info.vector_observations

                trans_list = [agents[i].add_transition(states[i],
                                                       actions[i],
                                                       brain_info.rewards[i],
                                                       brain_info.local_done[i],
                                                       brain_info.max_reached[i],
                                                       states_[i])
                              for i in range(len(agents))]

                trans = [np.concatenate(t) for t in zip(*trans_list)]

                self._add_trans(*trans)

                states = states_
                global_step += 1

            self._log_episode_info(global_step, agents)

    def _log_episode_info(self, global_step, agents):
        rewards = [a.reward for a in agents]
        rewards_sorted = ", ".join([f"{i:.1f}" for i in sorted(rewards)])
        logger.info(f'{global_step}, rewards {rewards_sorted}')

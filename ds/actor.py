import time
import sys
import getopt
import importlib
import yaml
import asyncio
from pathlib import Path

import requests
import websockets

import numpy as np
import tensorflow as tf


sys.path.append(str(Path(__file__).resolve().parent.parent))
from mlagents.envs import UnityEnvironment



class TransBuffer(object):
    _buffer = list()

    def add(self, *args):
        for arg in args:
            assert len(arg.shape) == 2
            assert len(arg) == len(args[0])

        for i in range(len(args[0])):
            self._buffer.append(tuple(arg[i] for arg in args))

    def get_trans_and_clear(self):
        trans = [np.array(e) for e in zip(*self._buffer)]
        self._buffer.clear()
        return trans

    @property
    def size(self):
        return len(self._buffer)

    @property
    def is_full(self):
        return self._size == self.capacity


class Actor(object):
    _replay_host = '127.0.0.1'
    _replay_port = 8888
    _learner_host = '127.0.0.1'
    _learner_port = 8889
    _websocket_host = '127.0.0.1'
    _websocket_port = 8890
    _build_path = None
    _build_port = 5005
    _sac = 'sac'
    _agents_num = 1
    _update_policy_variables_per_step = 100

    def __init__(self, argv):
        try:
            opts, args = getopt.getopt(argv, 'c:', ['config=',
                                                    'replay_host=',
                                                    'replay_port=',
                                                    'learner_host=',
                                                    'learner_port=',
                                                    'build_port=',
                                                    'sac=',
                                                    'agents='])
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
                        elif k == 'update_policy_variables_per_step':
                            self._update_policy_variables_per_step = v
                        elif k == 'sac':
                            self._sac = v
                break

        for opt, arg in opts:
            if opt == '--replay_host':
                self._replay_host = arg
            elif opt == '--replay_port':
                self._replay_port = int(arg)
            elif opt == '--learner_host':
                self._learner_host = arg
            elif opt == '--learner_port':
                self._learner_port = int(arg)
            elif opt == '--build_port':
                self._build_port = int(arg)
            elif opt == '--sac':
                self._sac = arg
            elif opt == '--agents':
                self._agents_num = int(arg)

        self._init_websocket_client()
        self._init_env()
        self._run()

    def _init_websocket_client(self):
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, lambda: asyncio.run(self._websocket_connected()))

    async def _websocket_connected(self):
        async with websockets.connect(f'ws://{self._websocket_host}:{self._websocket_port}') as websocket:
            while True:
                greeting = await websocket.recv()
                print(greeting)

    def _update_policy_variables(self):
        while True:
            try:
                r = requests.get(f'http://{self._learner_host}:{self._learner_port}/get_policy_variables')
            except Exception as e:
                print('Exception _update_policy_variables:', e)
            else:
                break

        new_variables = r.json()
        self.sac_actor.update_policy_variables(new_variables)

    def _add_trans(self, s, a, r, s_, done):
        self._tmp_trans_buffer.add(s, a, r, s_, done)

        if self._tmp_trans_buffer.size > 100:
            s, a, r, s_, done = self._tmp_trans_buffer.get_trans_and_clear()
            while True:
                try:
                    requests.post(f'http://{self._replay_host}:{self._replay_port}/add',
                                  json=[s.tolist(),
                                        a.tolist(),
                                        r.tolist(),
                                        s_.tolist(),
                                        done.tolist()])
                except Exception as e:
                    print('Exception _add_trans:', e)
                else:
                    break

    def _init_env(self):
        self.env = UnityEnvironment(file_name=self._build_path,
                                    no_graphics=True,
                                    base_port=self._build_port)

        self.default_brain_name = self.env.brain_names[0]

        brain_params = self.env.brains[self.default_brain_name]
        state_dim = brain_params.vector_observation_space_size
        action_dim = brain_params.vector_action_space_size[0]

        SAC = importlib.import_module(self._sac).SAC
        self.sac_actor = SAC(state_dim, action_dim, only_actor=True)

    def _run(self):
        self._tmp_trans_buffer = TransBuffer()

        reset_config = {
            'copy': self._agents_num,
        }
        global_step = 0

        brain_info = self.env.reset(train_mode=True, config=reset_config)[self.default_brain_name]

        while True:
            if self.env.global_done:
                brain_info = self.env.reset(train_mode=True, config=reset_config)[self.default_brain_name]

            len_agents = len(brain_info.agents)

            all_done = [False] * len_agents
            all_cumulative_rewards = [0] * len_agents

            hitted = 0
            states = brain_info.vector_observations

            while False in all_done and not self.env.global_done:
                if global_step % self._update_policy_variables_per_step == 0:
                    self._update_policy_variables()

                actions = self.sac_actor.choose_action(states)
                brain_info = self.env.step({
                    self.default_brain_name: actions
                })[self.default_brain_name]

                rewards = np.array(brain_info.rewards)
                local_dones = np.array(brain_info.local_done, dtype=bool)
                max_reached = np.array(brain_info.max_reached, dtype=bool)

                for i in range(len_agents):
                    if not all_done[i]:
                        all_cumulative_rewards[i] += rewards[i]
                        if rewards[i] > 0:
                            hitted += 1

                    all_done[i] = all_done[i] or local_dones[i]

                states_ = brain_info.vector_observations

                dones = np.logical_and(local_dones, np.logical_not(max_reached))

                s, a, r, s_, done = states, actions, rewards[:, np.newaxis], states_, dones[:, np.newaxis]
                self._add_trans(s, a, r, s_, done)

                states = states_
                global_step += 1

            print(f'{global_step}, rewards {", ".join([f"{i:.1f}" for i in sorted(all_cumulative_rewards)])}, hitted {hitted}')

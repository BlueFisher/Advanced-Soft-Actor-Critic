from collections import deque
from pathlib import Path
import functools
import getopt
import importlib
import logging
import os
import sys
import time
import yaml

import numpy as np
import tensorflow as tf

from .sac_base import SAC_Base
from .agent import Agent

sys.path.append(str(Path(__file__).resolve().parent.parent))
from mlagents.envs import UnityEnvironment

logger = logging.getLogger('sac')


class Main(object):
    train_mode = True

    def __init__(self, argv, agent_class=Agent):
        self._now = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
        self._agent_class = agent_class

        self.config, self.reset_config, replay_config, agent_config, model_root_path = self._init_config(argv)
        self._init_env(replay_config, agent_config, model_root_path)
        self._run()

    def _init_config(self, argv):
        config = {
            'name': self._now,
            'build_path': None,
            'port': 7000,
            'sac': 'sac',
            'max_iter': 1000,
            'agents_num': 1,
            'save_model_per_iter': 500,
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
            opts, args = getopt.getopt(argv, 'rc:n:b:p:', ['run',
                                                           'config=',
                                                           'name=',
                                                           'build=',
                                                           'port=',
                                                           'seed=',
                                                           'sac=',
                                                           'agents='])
        except getopt.GetoptError:
            raise Exception('ARGS ERROR')

        for opt, arg in opts:
            if opt in ('-c', '--config'):
                with open(arg) as f:
                    config_file = yaml.load(f, Loader=yaml.FullLoader)
                    for k, v in config_file.items():
                        if k == 'build_path':
                            config['build_path'] = v[sys.platform]

                        elif k == 'reset_config':
                            reset_config = dict(reset_config, **({} if v is None else v))

                        elif k == 'replay_config':
                            replay_config = {} if v is None else v

                        elif k == 'sac_config':
                            agent_config = {} if v is None else v

                        else:
                            config[k] = v
                break

        for opt, arg in opts:
            if opt in ('-r', '--run'):
                self.train_mode = False
            elif opt in ('-n', '--name'):
                config['name'] = arg.replace('{time}', self._now)
            elif opt in ('-b', '--build'):
                config['build_path'] = arg
            elif opt in ('-p', '--port'):
                config['port'] = int(arg)
            elif opt == '--seed':
                agent_config['seed'] = int(arg)
            elif opt == '--sac':
                config['sac'] = arg
            elif opt == '--agents':
                reset_config['copy'] = int(arg)

        model_root_path = f'models/{config["name"]}'

        if self.train_mode:
            if not os.path.exists(model_root_path):
                os.makedirs(model_root_path)
            with open(f'{model_root_path}/config.yaml', 'w') as f:
                yaml.dump({**config, **agent_config}, f, default_flow_style=False)

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

        return config, reset_config, replay_config, agent_config, model_root_path

    def _init_env(self, replay_config, agent_config, model_root_path):
        if self.config['build_path'] is None or self.config['build_path'] == '':
            self.env = UnityEnvironment()
        else:
            self.env = UnityEnvironment(file_name=self.config['build_path'],
                                        no_graphics=self.train_mode,
                                        base_port=self.config['port'])

        self.default_brain_name = self.env.brain_names[0]

        brain_params = self.env.brains[self.default_brain_name]
        state_dim = brain_params.vector_observation_space_size
        action_dim = brain_params.vector_action_space_size[0]

        class SAC(importlib.import_module(self.config['sac']).SAC_Custom, SAC_Base):
            pass

        self.sac = SAC(state_dim=state_dim,
                       action_dim=action_dim,
                       model_root_path=model_root_path,
                       replay_config=replay_config,
                       **agent_config)

    def _run(self):
        brain_info = self.env.reset(train_mode=self.train_mode, config=self.reset_config)[self.default_brain_name]

        for iteration in range(self.config['max_iter'] + 1):
            if self.env.global_done or self.config['reset_on_iteration']:
                brain_info = self.env.reset(train_mode=self.train_mode)[self.default_brain_name]

            agents = [self._agent_class(i,
                                        self.config['gamma'],
                                        self.config['n_step'])
                      for i in brain_info.agents]

            states = brain_info.vector_observations

            while False in [a.done for a in agents] and not self.env.global_done:
                actions = self.sac.choose_action(states)
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

                # s, a, r, s_, done, gamma, n_states, n_actions
                trans = [functools.reduce(lambda x, y: x + y, t) for t in zip(*trans_list)]

                if self.train_mode:
                    self.sac.train(*trans)

                states = states_

            if self.train_mode:
                self._log_episode_summaries(iteration, agents)

                if iteration % self.config['save_model_per_iter'] == 0:
                    self.sac.save_model(iteration)

            self._log_episode_info(iteration, agents)

        self.env.close()

    def _log_episode_summaries(self, iteration, agents):
        rewards = np.array([a.reward for a in agents])
        self.sac.write_constant_summaries([
            {'tag': 'reward/mean', 'simple_value': rewards.mean()},
            {'tag': 'reward/max', 'simple_value': rewards.max()},
            {'tag': 'reward/min', 'simple_value': rewards.min()}
        ], iteration)

    def _log_episode_info(self, iteration, agents):
        rewards = [a.reward for a in agents]
        rewards_sorted = ", ".join([f"{i:.1f}" for i in sorted(rewards)])
        logger.info(f'iter {iteration}, rewards {rewards_sorted}')

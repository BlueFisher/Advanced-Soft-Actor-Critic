from collections import deque
from pathlib import Path
import functools
import getopt
import importlib
import logging
import logging.handlers
import os
import shutil
import sys
import time
import yaml

import numpy as np
import tensorflow as tf

from .sac_base import SAC_Base
from .agent import Agent

sys.path.append(str(Path(__file__).resolve().parent.parent))
from mlagents.envs.environment import UnityEnvironment


class Main(object):
    train_mode = True
    _agent_class = Agent

    def __init__(self, argv):
        self._now = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))

        self.config, self.reset_config, replay_config, sac_config, model_root_path = self._init_config(argv)
        self._init_env(replay_config, sac_config, model_root_path)
        self._run()

    def _init_config(self, argv):
        config = dict()

        with open(f'{Path(__file__).resolve().parent}/default_config.yaml') as f:
            default_config_file = yaml.load(f, Loader=yaml.FullLoader)
            config = default_config_file

        # define command line arguments
        try:
            opts, args = getopt.getopt(argv, 'rc:n:b:p:', ['run',
                                                           'config=',
                                                           'name=',
                                                           'build=',
                                                           'port=',
                                                           'logger_file=',
                                                           'seed=',
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

        # initialize config from command line arguments
        logger_file = None
        for opt, arg in opts:
            if opt in ('-r', '--run'):
                self.train_mode = False
            elif opt in ('-n', '--name'):
                config['base_config']['name'] = arg
            elif opt in ('-b', '--build'):
                config['base_config']['build_path'] = arg
            elif opt in ('-p', '--port'):
                config['base_config']['port'] = int(arg)
            elif opt == '--logger_file':
                logger_file = arg
            elif opt == '--seed':
                config['sac_config']['seed'] = int(arg)
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

        self.logger = logging.getLogger('sac')
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
        if self.train_mode:
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
                config['reset_config'],
                config['replay_config'],
                config['sac_config'],
                model_root_path)

    def _init_env(self, replay_config, sac_config, model_root_path):
        if self.config['build_path'] is None or self.config['build_path'] == '':
            self.env = UnityEnvironment()
        else:
            self.env = UnityEnvironment(file_name=self.config['build_path'],
                                        no_graphics=self.train_mode,
                                        base_port=self.config['port'],
                                        args=['--scene', self.config['scene']])

        self.default_brain_name = self.env.brain_names[0]

        brain_params = self.env.brains[self.default_brain_name]
        state_dim = brain_params.vector_observation_space_size * brain_params.num_stacked_vector_observations
        action_dim = brain_params.vector_action_space_size[0]

        # if model exists, load saved model, else, copy a new one
        if os.path.isfile(f'{model_root_path}/sac_model.py'):
            custom_sac_model = importlib.import_module(f'{model_root_path.replace("/",".")}.sac_model')
        else:
            custom_sac_model = importlib.import_module(self.config['sac'])
            shutil.copyfile(f'{self.config["sac"]}.py', f'{model_root_path}/sac_model.py')

        self.sac = SAC_Base(state_dim=state_dim,
                            action_dim=action_dim,
                            model_root_path=model_root_path,
                            model=custom_sac_model,
                            train_mode=self.train_mode,

                            burn_in_step=self.config['burn_in_step'],
                            n_step=self.config['n_step'],
                            use_rnn=self.config['use_rnn'],
                            use_prediction=self.config['use_prediction'],

                            replay_config=replay_config,

                            **sac_config)

    def _run(self):
        brain_info = self.env.reset(train_mode=self.train_mode, config=self.reset_config)[self.default_brain_name]
        if self.config['use_rnn']:
            initial_rnn_state = self.sac.get_initial_rnn_state(len(brain_info.agents))
            rnn_state = initial_rnn_state

        for iteration in range(self.config['max_iter'] + 1):
            if self.config['reset_on_iteration']:
                brain_info = self.env.reset(train_mode=self.train_mode)[self.default_brain_name]
                if self.config['use_rnn']:
                    rnn_state = initial_rnn_state

            agents = [self._agent_class(i,
                                        tran_len=self.config['burn_in_step'] + self.config['n_step'],
                                        stagger=self.config['stagger'],
                                        use_rnn=self.config['use_rnn'])
                      for i in brain_info.agents]

            """
            s0    s1    s2    s3    s4    s5    s6
             └──burn_in_step───┘     └───n_step──┘
             └───────────deque_maxlen────────────┘
                         s2    s3    s4    s5    s6    s7    s8
             └─────┘
             stagger
            """

            states = brain_info.vector_observations
            step = 0

            while False in [a.done for a in agents]:
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

                tmp_results = [agents[i].add_transition(states[i],
                                                        actions[i],
                                                        brain_info.rewards[i],
                                                        brain_info.local_done[i],
                                                        brain_info.max_reached[i],
                                                        states_[i],
                                                        rnn_state[i] if self.config['use_rnn'] else None)
                               for i in range(len(agents))]

                if self.train_mode:
                    trans_list, episode_trans_list = zip(*tmp_results)

                    trans_list = [t for t in trans_list if t is not None]
                    if len(trans_list) != 0:
                        # n_states, n_actions, n_rewards, done, rnn_state
                        trans = [np.concatenate(t, axis=0) for t in zip(*trans_list)]
                        self.sac.fill_replay_buffer(*trans)

                    if self.config['use_rnn'] and self.config['use_prediction']:
                        episode_trans_list = [t for t in episode_trans_list if t is not None]
                        if len(episode_trans_list) != 0:
                            # n_states, n_actions, n_rewards, done, rnn_state
                            for episode_trans in episode_trans_list:
                                self.sac.fill_episode_replay_buffer(*episode_trans)

                    self.sac.train()

                states = states_
                if self.config['use_rnn']:
                    rnn_state = next_rnn_state
                    rnn_state[brain_info.local_done] = initial_rnn_state[brain_info.local_done]

                step += 1

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
        self.logger.info(f'iter {iteration}, rewards {rewards_sorted}')

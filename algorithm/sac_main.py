from collections import deque
from pathlib import Path
import functools
import getopt
import importlib
import logging
import logging.handlers
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


class Main(object):
    train_mode = True
    _agent_class = Agent

    def __init__(self, argv):
        self._now = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))

        self.config, self.reset_config, replay_config, sac_config, model_root_path = self._init_config(argv)
        self._init_env(replay_config, sac_config, model_root_path)
        self._run()

    def _init_config(self, argv):
        config = {
            'name': self._now,
            'build_path': None,
            'scene': None,
            'port': 7000,
            'sac': 'sac',
            'max_iter': 1000,
            'agents_num': 1,
            'save_model_per_iter': 500,
            'reset_on_iteration': True,
            'gamma': 0.99,
            'burn_in_step': 0,
            'n_step': 1,
            'stagger': 1,
            'use_rnn': False
        }
        reset_config = {
            'copy': 1
        }
        replay_config = dict()
        sac_config = dict()

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
                        if k == 'build_path':
                            config['build_path'] = v[sys.platform]

                        elif k == 'reset_config':
                            reset_config = dict(reset_config, **({} if v is None else v))

                        elif k == 'replay_config':
                            replay_config = {} if v is None else v

                        elif k == 'sac_config':
                            sac_config = {} if v is None else v

                        else:
                            config[k] = v
                break

        # initialize config from command line arguments
        logger_file = None
        for opt, arg in opts:
            if opt in ('-r', '--run'):
                self.train_mode = False
            elif opt in ('-n', '--name'):
                config['name'] = arg.replace('{time}', self._now)
            elif opt in ('-b', '--build'):
                config['build_path'] = arg
            elif opt in ('-p', '--port'):
                config['port'] = int(arg)
            elif opt == '--logger_file':
                logger_file = arg
            elif opt == '--seed':
                sac_config['seed'] = int(arg)
            elif opt == '--sac':
                config['sac'] = arg
            elif opt == '--agents':
                reset_config['copy'] = int(arg)

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

        self.logger = logging.getLogger('sac')
        self.logger.setLevel(level=logging.INFO)

        if logger_file is not None:
            # create file handler
            fh = logging.handlers.RotatingFileHandler(logger_file, maxBytes=1024 * 100, backupCount=5)
            fh.setLevel(logging.INFO)

            # add handler and formatter to logger
            fh.setFormatter(logging.Formatter('%(asctime)-15s [%(levelname)s] - [%(name)s] - %(message)s'))
            self.logger.addHandler(fh)

        model_root_path = f'models/{config["name"]}'

        # save config
        if self.train_mode:
            if not os.path.exists(model_root_path):
                os.makedirs(model_root_path)
            with open(f'{model_root_path}/config.yaml', 'w') as f:
                yaml.dump({**config,
                           'sac_config': {**sac_config}
                           }, f, default_flow_style=False)

        # display config
        config_str = '\ncommon_config'
        for k, v in config.items():
            config_str += f'\n{k:>25}: {v}'

        config_str += '\nreset_config:'
        for k, v in reset_config.items():
            config_str += f'\n{k:>25}: {v}'

        config_str += '\nreplay_config:'
        for k, v in replay_config.items():
            config_str += f'\n{k:>25}: {v}'

        config_str += '\nsac_config:'
        for k, v in sac_config.items():
            config_str += f'\n{k:>25}: {v}'
        self.logger.info(config_str)

        return config, reset_config, replay_config, sac_config, model_root_path

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

        custom_sac = importlib.import_module(self.config['sac'])

        self.sac = SAC_Base(state_dim=state_dim,
                            action_dim=action_dim,
                            ModelQ=custom_sac.ModelQ,
                            ModelPolicy=custom_sac.ModelPolicy,
                            model_root_path=model_root_path,
                            use_rnn=self.config['use_rnn'],
                            replay_config=replay_config,

                            gamma=self.config['gamma'],
                            burn_in_step=self.config['burn_in_step'],
                            n_step=self.config['n_step'],
                            **sac_config)

    def _run(self):
        brain_info = self.env.reset(train_mode=self.train_mode, config=self.reset_config)[self.default_brain_name]
        if self.config['use_rnn']:
            initial_lstm_state = lstm_state = self.sac.get_initial_lstm_state(len(brain_info.agents))

        for iteration in range(self.config['max_iter'] + 1):
            if self.env.global_done or self.config['reset_on_iteration']:
                brain_info = self.env.reset(train_mode=self.train_mode)[self.default_brain_name]
                if self.config['use_rnn']:
                    lstm_state = self.sac.get_initial_lstm_state(len(brain_info.agents))

            agents = [self._agent_class(i,
                                        gamma=self.config['gamma'],
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

            while False in [a.done for a in agents] and not self.env.global_done:
                if self.config['use_rnn']:
                    actions, lstm_state_ = self.sac.choose_lstm_action(lstm_state, states)
                else:
                    actions = self.sac.choose_action(states)

                brain_info = self.env.step({
                    self.default_brain_name: actions.numpy()
                })[self.default_brain_name]

                states_ = brain_info.vector_observations

                trans_list = [agents[i].add_transition(states[i],
                                                       actions[i],
                                                       brain_info.rewards[i],
                                                       brain_info.local_done[i],
                                                       brain_info.max_reached[i],
                                                       states_[i],
                                                       lstm_state.c[i] if self.config['use_rnn'] else None,
                                                       lstm_state.h[i] if self.config['use_rnn'] else None)
                              for i in range(len(agents))]

                # n_states, n_actions, n_rewards, done, lstm_state_c, lstm_state_h
                trans = [functools.reduce(lambda x, y: x + y, t) for t in zip(*trans_list)]

                if self.train_mode:
                    self.sac.fill_replay_buffer(*trans)
                    self.sac.train()

                states = states_
                if self.config['use_rnn']:
                    lstm_state_c = lstm_state_.c
                    lstm_state_h = lstm_state_.h
                    lstm_state_c[brain_info.local_done] = initial_lstm_state.c[brain_info.local_done]
                    lstm_state_h[brain_info.local_done] = initial_lstm_state.h[brain_info.local_done]
                    lstm_state = tf.nn.rnn_cell.LSTMStateTuple(c=lstm_state_c, h=lstm_state_h)

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

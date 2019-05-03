import sys
import getopt
import time
import random
import os
import importlib
import yaml

import numpy as np
import tensorflow as tf

sys.path.append('../..')
from mlagents.envs import UnityEnvironment

NOW = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
TRAIN_MODE = True

config = {
    'name': NOW,
    'build_path': None,
    'port': 7000,
    'sac': 'sac',
    'max_iter': 1000,
    'agents_num': 1,
    'save_model_per_iter': 500
}
agent_config = dict()

try:
    opts, args = getopt.getopt(sys.argv[1:], 'rc:n:b:p:', ['run',
                                                           'config=',
                                                           'name=',
                                                           'build=',
                                                           'port=',
                                                           'sac=',
                                                           'agents='])
except getopt.GetoptError:
    raise Exception('ARGS ERROR')

for opt, arg in opts:
    if opt in ('-c', '--config'):
        with open(arg) as f:
            config_file = yaml.load(f)
            for k, v in config_file.items():
                if k in config.keys():
                    if k == 'build_path':
                        config['build_path'] = v[sys.platform]
                    else:
                        config[k] = v
                else:
                    agent_config[k] = v
        break

for opt, arg in opts:
    if opt in ('-r', '--run'):
        TRAIN_MODE = False
    elif opt in ('-n', '--name'):
        config['name'] = arg.replace('{time}', NOW)
    elif opt in ('-b', '--build'):
        config['build_path'] = arg
    elif opt in ('-p', '--port'):
        config['port'] = int(arg)
    elif opt == '--sac':
        config['sac'] = arg
    elif opt == '--agents':
        config['agents_num'] = int(arg)

if TRAIN_MODE:
    if not os.path.exists('config'):
        os.makedirs('config')
    with open(f'config/{config["name"]}.yaml', 'w') as f:
        yaml.dump({**config, **agent_config}, f, default_flow_style=False)

for k, v in config.items():
    print(f'{k:>25}: {v}')
for k, v in agent_config.items():
    print(f'{k:>25}: {v}')
print('=' * 20)

if config['build_path'] is None or config['build_path'] == '':
    env = UnityEnvironment()
else:
    env = UnityEnvironment(file_name=config['build_path'],
                           no_graphics=TRAIN_MODE,
                           base_port=config['port'])

default_brain_name = env.brain_names[0]

brain_params = env.brains[default_brain_name]
state_dim = brain_params.vector_observation_space_size
action_dim = brain_params.vector_action_space_size[0]

SAC = importlib.import_module(config['sac']).SAC
sac = SAC(state_dim=state_dim,
          action_dim=action_dim,
          saver_model_path=f'model/{config["name"]}',
          summary_path='log' if TRAIN_MODE else None,
          summary_name=config["name"],
          **agent_config)

reset_config = {
    'copy': config['agents_num'],
    'reward': 0
}

brain_info = env.reset(train_mode=TRAIN_MODE, config=reset_config)[default_brain_name]
for iteration in range(config['max_iter'] + 1):
    if env.global_done:
        brain_info = env.reset(train_mode=TRAIN_MODE, config=reset_config)[default_brain_name]

    len_agents = len(brain_info.agents)
    
    all_done = [False] * len_agents
    all_cumulative_rewards = [0] * len_agents

    interaction_time_arr = []
    training_time_arr = []

    hitted = 0
    states = brain_info.vector_observations

    while False in all_done and not env.global_done:
        tmp_start = time.time()
        actions = sac.choose_action(states)
        brain_info = env.step({
            default_brain_name: actions
        })[default_brain_name]

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

        interaction_time_arr.append(time.time() - tmp_start)

        if TRAIN_MODE:
            tmp_start = time.time()
        
            dones = np.logical_and(local_dones, np.logical_not(max_reached))

            sac.train(states,
                      actions,
                      rewards[:, np.newaxis],
                      states_,
                      dones[:, np.newaxis])

            training_time_arr.append(time.time() - tmp_start)

        states = states_

    rewards = np.array(all_cumulative_rewards)
    sac.write_constant_summaries([
        {'tag': 'reward/mean', 'simple_value': rewards.mean()},
        {'tag': 'reward/max', 'simple_value': rewards.max()},
        {'tag': 'reward/min', 'simple_value': rewards.min()},
        {'tag': 'reward/hitted', 'simple_value': hitted}
    ], iteration)

    if iteration % config['save_model_per_iter'] == 0:
        sac.save_model(iteration)

    print(f'iter {iteration}, rewards {", ".join([f"{i:.1f}" for i in sorted(all_cumulative_rewards)])}, hitted {hitted}')
    print(f'interaction {sum(interaction_time_arr):.2f}, training {sum(training_time_arr):.2f}, buffer {sac.replay_buffer.size}, steps {len(training_time_arr)}')
    print('=' * 10)

env.close()

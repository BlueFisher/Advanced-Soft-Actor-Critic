# Advanced-Soft-Actor-Critic

This project is the algorithm [Soft Actor-Critic](https://arxiv.org/pdf/1812.05905) with a series of advanced features implemented by TensorFlow 2.x. It can be used to train Gym and Unity environments with ML-Agents.

## Features

- N-step
- V-trace (IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures)
- Prioritized Experience Replay
- Episode Experience Replay
- R2D2 (Recurrent Experience Replay In Distributed Reinforcement Learning)
- Representation model, Q function and policy strucutres
- Auxiliary prediction model
- Distributed training (Distributed Prioritized Experience Replay)
- Discrete action (Soft Actor-Critic for Discrete Action Settings)

## Supported Environments

Gym and Unity environments with ML-Agents. 

Observation can be any combination of vectors and images, which mean an agent can have multiple sensors and the resolution of each image can be different.

Action space can be continuous or discrete, but not supporting both continuous and discrete actions output.

Not supporting multi-agent environments.

## How to Use

### Training Settings

All neural network models should be in a .py file (default sac.py). All training configurations should be specified in a .yaml file (default config.yaml)

Both neural network models and training configurations should be placed in the same folder under `envs`.

All default training configurations are listed below. It can also be found in `algorithm/default_config.yaml`

```yaml
base_config:
  env_type: UNITY # UNITY or GYM
  scene: scene # The scene name. 
               # If in Unity envs, it indicates the specific scene. 
               # If in Gym envs, it is just a readable name displayed in TensorBoard

  # Only for Unity Environments
  build_path: # Unity executable path
    win32: path_win32
    linux: path_linux
  port: 5005
  
  # Only for Gym Enviroments
  build_path: GymEnv # Like CartPole-v1

  name: "{time}" # Training name. Placeholder "{time}" will be replaced to the time that trianing begins
  sac: sac # Neural network models file
  n_agents: 1 # N agents running in parallel
  max_iter: 1000 # Max iteration
  max_step: -1 # Max step in each iteration
  save_model_per_iter: 500 # Save model parameters every N iterations
  reset_on_iteration: true # Whether to force reset agent if an episode terminated

reset_config: null # Reset parameters sent to Unity

replay_config:
  batch_size: 256
  capacity: 1000000
  alpha: 0.9
  beta: 0.4
  beta_increment_per_sampling: 0.001
  td_error_min: 0.01
  td_error_max: 1.
  use_mongodb: false

sac_config:
  seed: null
  write_summary_per_step: 20

  burn_in_step: 0
  n_step: 1
  use_rnn: false

  tau: 0.005
  update_target_per_step: 1
  init_log_alpha: -2.3
  use_auto_alpha: true
  rep_lr: 0.0003
  q_lr: 0.0003
  policy_lr: 0.0003
  alpha_lr: 0.0003
  gamma: 0.99
  _lambda: 1.0
  use_priority: true
  use_n_step_is: true
  use_prediction: false
  use_reward_normalization: false
  use_curiosity: false
  curiosity_strength: 1
```

## Start Training

```
usage: main.py env_folder [--config CONFIG] [--run] [--render] [--editor]
               [--logger_file LOGGER_FILE] [--name NAME] [--port PORT]
               [--seed SEED] [--sac SAC] [--agents AGENTS] [--repeat REPEAT]

positional arguments:
  env_folder

optional arguments:
  --config CONFIG, -c CONFIG
                        config file
  --run                 inference mode
  --render              render mode
  --editor              running in Unity Editor
  --logger_file LOGGER_FILE
                        logging into a file
  --name NAME, -n NAME  training name
  --port PORT, -p PORT  communication port in Unity
  --seed SEED           random seed
  --sac SAC             neural network model
  --agents AGENTS       number of agents
  --repeat REPEAT       number of repeated experiments

examples:
python main.py pendulum -c config.yaml -n "test_{time}" --agents=10 --repeat=2
python main.py simple_roller -c config_hard.yaml -p 5006
python main.py simple_roller -c config.yaml -n nowall_202003251644192jWy --run
```
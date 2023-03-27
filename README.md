# Advanced-Soft-Actor-Critic

This project is the algorithm [Soft Actor-Critic](https://arxiv.org/pdf/1812.05905) with a series of advanced features implemented by PyTorch. It can be used to train Gym, PyBullet and Unity environments with ML-Agents.

## Features

- N-step
- V-trace ([IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](http://arxiv.org/abs/1802.01561))
- [Prioritized Experience Replay](http://arxiv.org/abs/1511.05952) (100% Numpy sumtree)
- *Episode Experience Replay
- R2D2 ([Recurrent Experience Replay In Distributed Reinforcement Learning](https://openreview.net/pdf?id=r1lyTjAqYX))
- *Representation model, Q function and policy strucutres
- *Recurrent Prediction Model
- Noisy Networks for Exploration ([Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295))
- Distributed training ([Distributed Prioritized Experience Replay](http://arxiv.org/abs/1803.00933))
- Discrete action ([Soft Actor-Critic for Discrete Action Settings](http://arxiv.org/abs/1910.07207))
- Curiosity mechanism ([Curiosity-driven Exploration by Self-supervised Prediction](http://arxiv.org/abs/1705.05363))
- *Large-scale Distributed Evolutionary Reinforcement Learning
- [ATC](http://proceedings.mlr.press/v139/stooke21a.html), [BYOL](https://proceedings.neurips.cc/paper/2020/hash/f3ada80d5c4ee70142b17b8192b2958e-Abstract.html)

\* denotes the features that we implemented.

## Supported Environments

Gym, PyBullet and Unity environments with ML-Agents. 

Observation can be any combination of vectors and images, which means an agent can have multiple sensors and the resolution of each image can be different.

Action space can be both continuous and discrete.

Not supporting multi-agent environments.

## How to Use

### Training Settings

All neural network models should be in a .py file (default `nn.py`). All training configurations should be specified in `config.yaml`.

Both neural network models and training configurations should be placed in the same folder under `envs`.

All default training configurations are listed below. It can also be found in `algorithm/default_config.yaml`

```yaml
base_config:
  env_type: UNITY # UNITY | GYM | DM_CONTROL
  env_name: env_name # The environment name.
  env_args: null
  unity_args: # Only for Unity Environments
    no_graphics: true # If an env does not need pixel input, set true
    build_path: # Unity executable path
      win32: path_win32
      linux: path_linux
    port: 5005

  name: "{time}" # Training name. Placeholder "{time}" will be replaced to the time that trianing begins

  n_envs: 1 # N environments running in parallel

  max_iter: -1 # Max iteration
  max_step: -1 # Max step. Training will be terminated if max_iter or max_step encounters
  max_step_each_iter: -1 # Max step in each iteration
  reset_on_iteration: true # Whether forcing reset agent if an episode terminated

reset_config: null # Reset parameters sent to Unity

nn_config:
  rep: null
  policy: null

replay_config:
  capacity: 524288
  alpha: 0.9 # [0~1] convert the importance of TD error to priority. If 0, PER will reduce to vanilla replay buffer
  beta: 0.4 # Importance-sampling, from initial value increasing to 1
  beta_increment_per_sampling: 0.001 # Increment step
  td_error_min: 0.01 # Small amount to avoid zero priority
  td_error_max: 1. # Clipped abs error

sac_config:
  nn: nn # Neural network models file
  seed: null # Random seed
  write_summary_per_step: 1000 # Write summaries in TensorBoard every N steps
  save_model_per_step: 5000 # Save model every N steps

  use_replay_buffer: true # Whether using prioritized replay buffer
  use_priority: true # Whether using PER importance ratio

  ensemble_q_num: 2 # Number of Qs
  ensemble_q_sample: 2 # Number of min Qs

  burn_in_step: 0 # Burn-in steps in R2D2
  n_step: 1 # Update Q function by N-steps
  seq_encoder: null # RNN | ATTN

  batch_size: 256 # Batch size for training

  tau: 0.005 # Coefficient of updating target network
  update_target_per_step: 1 # Update target network every N steps

  init_log_alpha: -2.3 # The initial log_alpha
  use_auto_alpha: true # Whether using automating entropy adjustment

  learning_rate: 0.0003 # Learning rate of all optimizers

  gamma: 0.99 # Discount factor
  v_lambda: 1.0 # Discount factor for V-trace
  v_rho: 1.0 # Rho for V-trace
  v_c: 1.0 # C for V-trace
  clip_epsilon: 0.2 # Epsilon for q clip

  discrete_dqn_like: false # Whether using policy or only Q network if discrete is in action spaces
  use_n_step_is: true # Whether using importance sampling
  siamese: null # ATC | BYOL
  siamese_use_q: false # Whether using contrastive q
  siamese_use_adaptive: false # Whether using adaptive weights
  use_prediction: false # Whether training a transition model
  transition_kl: 0.8 # The coefficient of KL of transition and standard normal
  use_extra_data: true # Whether using extra data to train prediction model
  curiosity: null # FORWARD | INVERSE
  curiosity_strength: 1 # Curiosity strength if using curiosity
  use_rnd: false # Whether using RND
  rnd_n_sample: 10 # RND sample times
  use_normalization: false # Whether using observation normalization
  use_add_with_td: false # Whether add transitions in replay buffer with td-error
  action_noise: null # [noise_min, noise_max]

ma_config: null
```

All default distributed training configurations are listed below. It can also be found in `ds/default_config.yaml`

```yaml
base_config:
  env_type: UNITY # UNITY or GYM
  env_name: env_name # The environment name.
  env_args: null
  unity_args: # Only for Unity Environments
    no_graphics: true # If an env does not need pixel input, set true
    build_path: # Unity executable path
      win32: path_win32
      linux: path_linux
    port: 5005

  name: "{time}" # Training name. Placeholder "{time}" will be replaced to the time that trianing begins
  update_sac_bak_per_step: 200 # Every N step update sac_bak
  n_envs: 1 # N environments running in parallel
  max_step_each_iter: -1 # Max step in each iteration
  reset_on_iteration: true # Whether forcing reset agent if an episode terminated

  evolver_enabled: true
  evolver_cem_length: 50 # Start CEM if all learners have eavluated evolver_cem_length times
  evolver_cem_best: 0.3 # The ratio of best learners
  evolver_cem_min_length:
    2 # Start CEM if all learners have eavluated `evolver_cem_min_length` times,
    # and it has been more than `evolver_cem_time` minutes since the last update
  evolver_cem_time: 3
  evolver_remove_worst: 4

  max_actors_each_learner: -1 # The max number of actors of each learner, -1 indicates no limit
  noise_increasing_rate: 0 # Noise = N * number of actors
  noise_max: 0.1 # Max noise for actors
  max_episode_length: 500
  episode_queue_size: 5
  episode_sender_process_num: 5
  batch_queue_size: 5
  batch_generator_process_num: 5

net_config:
  learner_host: null
  learner_port: 61001

reset_config: null # Reset parameters sent to Unity

nn_config:
  rep: null
  policy: null

sac_config:
  nn: nn # Neural network models file
  seed: null # Random seed
  write_summary_per_step: 1000 # Write summaries in TensorBoard every N steps
  save_model_per_step: 100000 # Save model every N steps

  ensemble_q_num: 2 # Number of Qs
  ensemble_q_sample: 2 # Number of min Qs

  burn_in_step: 0 # Burn-in steps in R2D2
  n_step: 1 # Update Q function by N steps
  seq_encoder: null # RNN | ATTN

  batch_size: 256

  tau: 0.005 # Coefficient of updating target network
  update_target_per_step: 1 # Update target network every N steps

  init_log_alpha: -2.3 # The initial log_alpha
  use_auto_alpha: true # If using automating entropy adjustment

  learning_rate: 0.0003 # Learning rate of all optimizers

  gamma: 0.99 # Discount factor
  v_lambda: 1.0 # Discount factor for V-trace
  v_rho: 1.0 # Rho for V-trace
  v_c: 1.0 # C for V-trace
  clip_epsilon: 0.2 # Epsilon for q clip

  discrete_dqn_like: false # If using policy or only Q network if discrete is in action spaces
  siamese: null # ATC | BYOL
  siamese_use_q: false # If using contrastive q
  siamese_use_adaptive: false # If using adaptive weights
  use_prediction: false # If train a transition model
  transition_kl: 0.8 # The coefficient of KL of transition and standard normal
  use_extra_data: true # If using extra data to train prediction model
  curiosity: null # FORWARD | INVERSE
  curiosity_strength: 1 # Curiosity strength if using curiosity
  use_rnd: false # If using RND
  rnd_n_sample: 10 # RND sample times
  use_normalization: false # If using observation normalization
  action_noise: null # [noise_min, noise_max]

  # random_params:
  #   param_name:
  #     in: [n1, n2, n3]
  #     truncated: [n1 ,n2]
  #     std: n

ma_config: null
```

## Start Training

```
usage: main.py [-h] [--config CONFIG] [--run] [--logger_in_file] [--render] [--env_args ENV_ARGS] [--agents AGENTS]
               [--max_iter MAX_ITER] [--port PORT] [--editor] [--name NAME] [--disable_sample] [--use_env_nn]
               [--device DEVICE] [--ckpt CKPT] [--nn NN] [--repeat REPEAT]
               env

positional arguments:
  env

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG, -c CONFIG
                        config file
  --run                 inference mode
  --logger_in_file      logging into a file
  --render              render
  --env_args ENV_ARGS   additional args for environments
  --agents AGENTS       number of agents
  --max_iter MAX_ITER   max iteration
  --port PORT, -p PORT  UNITY: communication port
  --editor              UNITY: running in Unity Editor
  --name NAME, -n NAME  training name
  --disable_sample      disable sampling when choosing actions
  --use_env_nn          always use nn.py in env, or use saved nn_models.py if existed
  --device DEVICE       cpu or gpu
  --ckpt CKPT           ckeckpoint to restore
  --nn NN               neural network model
  --repeat REPEAT       number of repeated experiments

examples:
# Train gym environment mountain_car with name "test_{time}", 10 agents and repeating training two times
python main.py gym/mountain_car -n "test_{time}" --agents=10 --repeat=2
# Train unity environment roller with vanilla config and port 5006
python main.py roller -c vanilla -p 5006
# Inference unity environment roller with model "nowall_202003251644192jWy"
python main.py roller -c vanilla -n nowall_202003251644192jWy --run --agents=1
```

## Start Distributed Training

```
usage: main_ds.py [-h] [--config CONFIG] [--run] [--logger_in_file] [--learner_host LEARNER_HOST]
                  [--learner_port LEARNER_PORT] [--render] [--env_args ENV_ARGS] [--agents AGENTS]
                  [--unity_port UNITY_PORT] [--editor] [--name NAME] [--device DEVICE] [--ckpt CKPT] [--nn NN]
                  env {learner,l,actor,a}

positional arguments:
  env
  {learner,l,actor,a}

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG, -c CONFIG
                        config file
  --run                 inference mode
  --logger_in_file      logging into a file
  --learner_host LEARNER_HOST
                        learner host
  --learner_port LEARNER_PORT
                        learner port
  --render              render
  --env_args ENV_ARGS   additional args for environments
  --agents AGENTS       number of agents
  --unity_port UNITY_PORT, -p UNITY_PORT
                        UNITY: communication port
  --editor              UNITY: running in Unity Editor
  --name NAME, -n NAME  training name
  --device DEVICE       cpu or gpu
  --ckpt CKPT           ckeckpoint to restore
  --nn NN               neural network model

examples:
python main_ds.py test learner --logger_in_file
python main_ds.py test actor --logger_in_file
```
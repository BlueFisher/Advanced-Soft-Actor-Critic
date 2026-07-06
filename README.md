# Advanced-Soft-Actor-Critic

This project is the algorithm [Soft Actor-Critic](https://arxiv.org/pdf/1812.05905) with a series of advanced features implemented by PyTorch. It can be used to train Gym, PyBullet and Unity environments with ML-Agents.

## Features

- N-step
- V-trace ([IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](http://arxiv.org/abs/1802.01561))
- [Prioritized Experience Replay](http://arxiv.org/abs/1511.05952) (100% Numpy sumtree)
- *Episode Experience Replay
- R2D2 ([Recurrent Experience Replay In Distributed Reinforcement Learning](https://openreview.net/pdf?id=r1lyTjAqYX))
- *Representation model, Q function and policy strucutres
- *[Recurrent Prediction Model](https://www.sciencedirect.com/science/article/pii/S0020025522013548)
- Noisy Networks for Exploration ([Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295))
- Discrete action ([Soft Actor-Critic for Discrete Action Settings](http://arxiv.org/abs/1910.07207))
- Curiosity mechanism ([Curiosity-driven Exploration by Self-supervised Prediction](http://arxiv.org/abs/1705.05363))
- [ATC](http://proceedings.mlr.press/v139/stooke21a.html), [BYOL](https://proceedings.neurips.cc/paper/2020/hash/f3ada80d5c4ee70142b17b8192b2958e-Abstract.html)
- *Siamese Q ([Zero-shot sim-to-real transfer using Siamese-Q-Based reinforcement learning](https://linkinghub.elsevier.com/retrieve/pii/S1566253524004421))
- *DAM ([Dilated memory in hierarchical reinforcement learning for long-horizontal task](https://linkinghub.elsevier.com/retrieve/pii/S0893608025013231))

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
  env_type: UNITY # UNITY | GYM | TEST
  env_name: env_name # The environment name.
  env_args: null # The environment build arguments
  hit_reward: null
  unity_args: # Specific for Unity Environments
    max_n_envs_per_process: 10 # The max env copies count in each process
    no_graphics: true # If an env does not need pixel input, set true
    force_vulkan: false # -force-vulkan
    build_path: # Unity executable path
      win32: path_win32
      linux: path_linux

  offline_env_config:
    enabled: false
    env_name: env_name
    env_args: null
    n_envs: 50

  name: "{time}-{hostname}" # Training name. 
                            # Placeholder "{time}" will be replaced to the time that trianing begins
                            # Placeholder "{hostname}" will be replaced to the hostname of the machine

  n_envs: 1 # N agents running in parallel

  obs_preprocessor: null

  max_iter: -1 # Max iteration
  max_step: -1 # Max step. Training will be terminated if max_iter or max_step encounters
  max_step_each_iter: -1 # Max step in each iteration
  reset_on_iteration: true # Whether forcing reset agent if an episode terminated

reset_config: null # Reset parameters sent to Unity

nn_config:
  rep: null
  policy: null

replay_config:
  capacity: 524288 # 262144 131072 65536
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
  target_d_alpha: 0.98 # Target discrete alpha ratio
  target_c_alpha: 1.0 # Target continuous alpha ratio
  d_policy_entropy_penalty: 0.5 # Discrete policy entropy penalty ratio

  learning_rate: 0.0003 # Learning rate of all optimizers

  gamma: 0.99 # Discount factor
  v_lambda: 1.0 # Discount factor for V-trace
  v_rho: 1.0 # Rho for V-trace
  v_c: 1.0 # C for V-trace
  clip_epsilon: 0.2 # Epsilon for q clip

  discrete_dqn_like: false # Whether using policy or only Q network if discrete is in action spaces
  discrete_dqn_epsilon: 0.2 # Probability of using random action
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

  offline_loss: false # Whether using offline loss

  action_noise: null # [noise_min, noise_max]

oc_config:
  use_dilation: false
  option_burn_in_step: -1
  option_seq_encoder: null # RNN
  option_epsilon: 0.2 # Probability of switching options
  terminal_entropy: 0.01 # Tending not to terminate >0, tending to terminate <0

  option_nn_config:
    rep: null

  option_configs:
    - name: option_0
      fix_policy: false # Fix Policy and Representation
      random_q: false # Ignore saved Q network and randomly initialize Q network
    - name: option_1
      fix_policy: false # Fix Policy and Representation
      random_q: false # Ignore saved Q network and randomly initialize Q network

ma_config: null

```


## Start Training

```
usage: main.py [-h] [--oc] [--config CONFIG] [--override OVERRIDE [OVERRIDE ...]] [--run] [--run_a RUN_A] [--copy_model COPY_MODEL]
               [--logger_in_file] [--render] [--env_args ENV_ARGS [ENV_ARGS ...]] [--envs ENVS] [--max_iter MAX_ITER] [--u_port U_PORT]
               [--u_editor] [--u_quality_level U_QUALITY_LEVEL] [--u_timescale U_TIMESCALE] [--name NAME] [--disable_sample]
               [--use_env_nn] [--device DEVICE] [--ckpt CKPT] [--nn NN] [--debug]
               env

positional arguments:
  env

options:
  -h, --help            show this help message and exit
  --oc
  --config CONFIG, -c CONFIG
                        Config file
  --override OVERRIDE [OVERRIDE ...], -o OVERRIDE [OVERRIDE ...]
                        Override config
  --run                 Inference mode for all agents, ignore run_a
  --run_a RUN_A         Inference mode for specific agents
  --copy_model COPY_MODEL
                        Copy existed model directory to current model directory
  --logger_in_file      Logging into a file
  --render              Render
  --env_args ENV_ARGS [ENV_ARGS ...]
                        Additional arguments for environments. Eqvilent to --override base_config.env_args
  --envs ENVS           Number of env copies. Eqvilent to --override base_config.n_envs
  --max_iter MAX_ITER   Maximum iteration. Eqvilent to --override base_config.max_iter
  --u_port U_PORT, -p U_PORT
                        UNITY: communication port
  --u_editor            UNITY: running in Unity Editor
  --u_quality_level U_QUALITY_LEVEL
                        UNITY: Quality level. 0: URP-Performant-Renderer, 1: URP-Balanced-Renderer, 2: URP-HighFidelity-Renderer
  --u_timescale U_TIMESCALE
                        UNITY: Timescale
  --name NAME, -n NAME  Training name. Eqvilent to --override base_config.name
  --disable_sample      Disable sampling when choosing actions
  --use_env_nn          Always use nn.py in env, or use saved nn_models.py if existed
  --device DEVICE       CPU or GPU
  --ckpt CKPT           Ckeckpoint to restore
  --nn NN               Neural network model. Eqvilent to --override sac_config.nn
  --debug
```

Examples:

```
# Train gym environment mountain_car with name "test_{time}", 10 agents
python main.py gym/mountain_car -n "test_{time}" --envs=10

# Train unity environment roller with vanilla config and port 5006
python main.py roller -c vanilla -u_port 5006

# Inference unity environment roller with model "nowall_202003251644192jWy"
python main.py roller -c vanilla -n nowall_202003251644192jWy --run --envs=1
```

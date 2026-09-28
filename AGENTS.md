# AGENTS.md

## Project Overview

Advanced-SAC is a general-purpose reinforcement learning project. It is built
on Soft Actor-Critic (SAC), implemented with PyTorch, and extends the base SAC
algorithm with a set of advanced features, including N-step returns, V-trace,
Prioritized Experience Replay, Episode Replay, R2D2-style recurrent replay,
pluggable representation / Q / policy network structures, prediction models,
Noisy Networks, discrete-action SAC, Curiosity, RND, ATC/BYOL, Siamese-Q, DAM,
and Option-Critic related extensions.

The project supports Gym and Unity ML-Agents environments.
Observations may contain multiple vector and image sensors, and action spaces
may be continuous, discrete, or mixed. The current code is primarily designed
for single-agent training workflows, while preserving a manager structure that
can separate agents by Unity behavior name.

## Top-Level Directory Roles

- `algorithm/`: Core reinforcement learning code, including SAC orchestration,
  training bases, replay buffers, batch construction, environment wrappers,
  neural-network base modules, Option-Critic extensions, and common utilities.
- `envs/`: Environment-specific configurations and network definitions. Each
  environment directory usually contains `config.yaml` and `nn.py`, and may also
  contain an observation preprocessor or multiple network variants.
- `mlagents_envs/`: Unity ML-Agents Python communication layer and generated
  protocol files used to connect to Unity executables or the Unity Editor.
- `models/`: Training output directory. Runs save configs, logs, checkpoints,
  TensorBoard summaries, copied `nn` directories, and related artifacts here,
  grouped by environment name and experiment name.
- `models_bak/`: Model backup directory. Usually not a code-edit target.
- `summary_plot/`: Plotting scripts for training results or summary data.
- `tests/`: Tests and experimental network scripts covering SAC/OC main-flow
  parameters, network structure examples, and synthetic data generation.
- `.vscode/`: Local editor configuration.
- `__pycache__/`: Python cache directory. Do not maintain or add cache files
  manually.

## Main Entrypoints And Scripts

- `main.py`: Command-line entrypoint. It parses the environment name, config
  category, override values, Unity options, training/inference mode,
  checkpoints, device settings, and selects either SAC or Option-Critic based on
  `--oc`.
- `README.md`: Project description, supported features, configuration fields,
  and training command examples.
- `setup.py`: Python package installation configuration.
- `run.sh` / `train.sh`: Convenience scripts for running the project.
- `clear_models.py`: Helper script for cleaning model outputs.
- `clear.py`: Simple cleanup script.
- `test.py`, `test_env.py`, `test_oc.py`: Top-level test/debug entrypoints.

## `algorithm/` Directory Structure

- `algorithm/default_config.yaml`: Global default training configuration.
  Environment-level `config.yaml` files are merged with this file.
- `algorithm/config_helper.py`: YAML config loading, inheritance, overrides,
  platform-specific field selection, random parameter handling, experiment-name
  generation, and logger setup.
- `algorithm/sac_main.py`: SAC training orchestrator. It merges configs, creates
  model directories, initializes environments, loads network definitions from
  environment directories, builds `SAC_Base`, runs the training/inference loop,
  writes logs, and saves models.
- `algorithm/sac_base.py`: Core SAC implementation. It builds representation,
  Q, policy, target networks, optimizers, entropy alpha, replay buffer, and
  checkpoints, and implements action selection, episode ingestion, sampling,
  Q/policy/representation/auxiliary-loss training, and summaries.
- `algorithm/agent.py`: Defines `Agent`, `AgentManager`, and
  `MultiAgentsManager`. These classes maintain temporary episode trajectories
  by agent id, merge observations/actions, handle recurrent/attention hidden
  states, write episodes into the RL module, trigger training, and save models.
- `algorithm/replay_buffer.py`: Prioritized replay buffer and related sampling
  / priority update logic.
- `algorithm/batch_buffer.py`: Batch storage and sampling logic used when the
  replay buffer is disabled.
- `algorithm/imitation_base.py`: Offline/imitation-learning loss wrapper.

### `algorithm/env_wrapper/`

Environment adapters that expose different backends through the common
interface consumed by the training loop.

- `env_wrapper.py`: Wrapper base class and common data structures.
- `gym_wrapper.py`: Gym environment wrapper.
- `unity_wrapper.py`: Unity ML-Agents environment wrapper.
- `offline_wrapper.py`: Offline environment wrapper.
- `test_wrapper.py`: Test environment wrapper.
- `test_offline_wrapper.py`: Offline test-environment wrapper.
- `obs_preprocessor_wrapper.py`: Observation-preprocessor wrapper base or helper
  implementation.

### `algorithm/nn_models/`

Base neural-network modules. Environment-level `nn.py` files usually inherit
from or compose these base classes.

- `representation.py`: Observation encoder / representation base classes.
- `q.py`: Q-network base classes.
- `policy.py`: Policy base classes for continuous and discrete action
  distributions.
- `predictions.py`: Transition, curiosity, and auxiliary prediction models.
- `exploration.py`: Exploration-related models such as RND.
- `layers/linear_layers.py`: Linear-layer components such as MLPs and
  NoisyLinear.
- `layers/image_layers.py`: Image-encoder related layers.
- `layers/seq_layers.py`: RNN, attention, and sequence-encoder related layers.

### `algorithm/oc/`

Option-Critic / hierarchical RL extensions.

- `oc_main.py`: OC main-flow entrypoint. It reuses the environment and training
  loop from `sac_main.Main`, but builds `OptionSelectorBase`.
- `option_selector_base.py`: Core option-selector training and action-selection
  logic.
- `option_base.py`: Training wrapper for an individual option's policy/Q logic.
- `oc_agent.py`: OC-specific agent manager with additional option state.
- `oc_batch_buffer.py`: Batch buffer used by OC training.

### `algorithm/utils/`

Common utilities, including enums, timers, locks, tensor/array transforms,
episode plotting, mathematical operators, and option/attention/ray/image
visualization tools.

## `envs/` Directory Conventions

Each environment directory usually follows this structure:

- `config.yaml`: Environment and training configuration. It must contain a
  `default` config section. Other config sections can be selected from the
  command line with `-c <config_cat>`. Configs inherit from and override
  `algorithm/default_config.yaml`.
- `nn.py`: Default neural-network definition. Training selects it through
  `sac_config.nn`; the default module name is `nn`.
- `obs_preprocessor.py`: Optional. If `base_config.obs_preprocessor` is set, it
  wraps the online/offline environment after environment initialization.
- Other `nn_*.py` files: Network structure variants that can be selected with
  `--nn` or `sac_config.nn`.

Existing environments are organized by task type, for example:

- `envs/gym/`: Gym tasks such as `cart_pole`, `pendulum`, `mountain_car`,
  `humanoid`, and toy memory tasks.
- `envs/roller/`, `envs/pyramid/`, `envs/3dball/`, `envs/square/`: Unity or
  test-style tasks.
- `envs/uav/`, `envs/ugv/`, `envs/usv/`: UAV, UGV, and USV tasks.
- `envs/test/`: Test environments and network variants.

## Training And Running

Common commands:

```bash
python main.py gym/mountain_car -n "test_{time}" --envs=10
python main.py roller -c vanilla -u_port 5006
python main.py roller -c vanilla -n nowall_202003251644192jWy --run --envs=1
python main.py <env_path> --oc
```

Use dotted paths for command-line config overrides:

```bash
python main.py gym/pendulum -o sac_config.batch_size=128 base_config.max_iter=100
```

Training artifacts are written to:

```text
models/<env_name>/<experiment_name>/
```

`<experiment_name>` comes from `base_config.name`, which supports the `{time}`
and `{hostname}` placeholders.

## Notes For Code Changes

- Keep `algorithm/default_config.yaml`, environment `config.yaml` files, and the
  configuration documentation in `README.md` consistent.
- When adding a new environment, put task-specific config and networks under
  `envs/<category>/<env_name>/`. Avoid putting task-specific logic into the
  algorithm core unless it is truly reusable.
- When adding a new network structure, prefer reusing base classes and layers
  from `algorithm/nn_models/`.
- Before changing the training core, check whether the change affects
  continuous actions, discrete actions, mixed actions, RNN, attention, PER,
  N-step, offline loss, and OC workflows.
- `models/`, `models_bak/`, and `__pycache__/` are usually generated artifacts
  or backups, not regular code-edit targets.
- Environment configs may contain active user experiment parameters. Check git
  status before modifying `envs/**/config.yaml` to avoid overwriting uncommitted
  experiment settings.

## Testing And Verification

Choose verification commands based on the change scope:

```bash
python -m pytest tests
python test_env.py
python test.py
python test_oc.py
```

For documentation-only changes, at minimum check that the Markdown content is
complete, paths are accurate, and generated artifacts are not described as
source entrypoints. For algorithm or network changes, prefer adding/running
tests related to SAC, OC, environment wrappers, and the affected network
structure.

## Collaboration Guidance For Future Agents

- Start by reading `README.md`, this file, and the target environment's
  `config.yaml` / `nn.py`.
- Use `rg` to search config fields or class names instead of guessing the call
  chain from filenames alone.
- For training behavior changes, trace the full data flow through `main.py` ->
  `algorithm/sac_main.py` -> `algorithm/agent.py` -> `algorithm/sac_base.py`.
- For OC behavior changes, also inspect selector, option, and agent-management
  logic under `algorithm/oc/`.
- Do not clean, move, or rebuild existing model directories unless the user
  explicitly asks for it.

import importlib.util
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np

import algorithm.config_helper as config_helper
from algorithm.env_wrapper.test_offline_wrapper import TestOfflineWrapper

from .agent import MultiAgentsManager
from .sac_base import SAC_Base
from .imitation_base import ImitationBase
from .utils import UnifiedElapsedTimer, format_global_step
from .utils.enums import *


class Main:
    train_mode = True
    render = False
    unity_run_in_editor = False

    ma_manager: MultiAgentsManager

    def __init__(self,
                 root_dir: Path | str,
                 config_dir: Path | str,

                 config_cat: str | None = None,
                 override: list[tuple[list[str], str]] | None = None,
                 train_mode: bool = True,
                 inference_ma_names: set[str] | None = None,
                 copy_model: str | None = None,
                 logger_in_file: bool = False,

                 render: bool = False,
                 env_args: dict | None = None,
                 n_envs: int | None = None,
                 max_iter: int | None = None,

                 unity_port: int | None = None,
                 unity_run_in_editor: bool = False,
                 unity_quality_level: int = 2,
                 unity_time_scale: float | None = None,

                 name: str | None = None,
                 disable_sample: bool = False,
                 use_env_nn: bool = False,
                 device: str | None = None,
                 last_ckpt: str | None = None,
                 nn: str | None = None):
        """
        Args:
            root_dir: The root directory for the project.
            config_dir: The directory containing configuration files.
            config_cat: The category of the configuration. Defaults to None, which is default.
            override: A list of overrides for configuration settings. Defaults to None.
            train_mode: Whether the algorithm is in training mode. Defaults to True.
            inference_ma_names: Names of multi-agent names for inference. Defaults to None.
            copy_model: Path to a model to copy. Defaults to None.
            logger_in_file: Whether to log output to a file. Defaults to False.

            render: Whether to render the environment. Defaults to False.
            env_args: Additional arguments for the environment. Defaults to None.
            n_envs: Number of environments to use. Defaults to None.
            max_iter: Maximum number of iterations. Defaults to None.
            unity_port: Port for Unity environment communication. Defaults to None.
            unity_run_in_editor: Whether Unity is running in editor mode. Defaults to False.
            unity_quality_level: Quality level for Unity environment. Defaults to 2.
            unity_time_scale: Time scale for Unity environment. Defaults to None.

            name: Name of the experiment. Defaults to None.
            disable_sample: Whether to disable sampling. Defaults to False.
            use_env_nn: Whether to force the use of environment neural networks. Defaults to False.
            device: Device to use for computation (e.g., "cpu" or "cuda"). Defaults to None.
            last_ckpt: Path to the last checkpoint file. Defaults to None.
            nn: Neural network model file name. Defaults to None.
        """
        self.root_dir = Path(root_dir)

        self.train_mode = train_mode
        self.inference_ma_names = inference_ma_names if inference_ma_names is not None else set()
        self.render = render

        self.unity_run_in_editor = unity_run_in_editor
        self.unity_quality_level = unity_quality_level
        self.unity_time_scale = unity_time_scale

        self.disable_sample = disable_sample
        self.force_env_nn = use_env_nn
        self.device = device
        self.last_ckpt = last_ckpt

        self._logger = logging.getLogger('sac')

        self._profiler = UnifiedElapsedTimer(self._logger)

        self.model_abs_dir, self._config_abs_dir = self._init_config(config_dir,
                                                                     config_cat,
                                                                     override,

                                                                     env_args,
                                                                     n_envs,
                                                                     max_iter,

                                                                     unity_port,

                                                                     name,
                                                                     nn)

        if logger_in_file:
            config_helper.add_file_logger(self.model_abs_dir.joinpath(f'log.log'))

        self._handle_copy_model(copy_model)

        self._init_env()
        try:
            self._init_sac()
        except:
            self.env.close()
            self._logger.warning('Training terminated by exception')
            raise

        self._run()

    def _init_config(self,
                     config_dir: Path | str,
                     config_cat: str | None,
                     override: list[tuple[list[str], str]] | None,

                     env_args: dict | None,
                     n_envs: int | None,
                     max_iter: int | None,

                     unity_port: int | None,

                     name: str | None,
                     nn: str | None):
        config_abs_dir = self.root_dir.joinpath(config_dir)
        config_abs_path = config_abs_dir.joinpath('config.yaml')
        default_config_abs_path = Path(__file__).resolve().parent.joinpath('default_config.yaml')
        # Merge default_config.yaml and custom config.yaml
        config, ma_configs = config_helper.initialize_config_from_yaml(default_config_abs_path,
                                                                       config_abs_path,
                                                                       config_cat=config_cat,
                                                                       override=override)

        if env_args is None:
            env_args = {}
        for k, v in env_args.items():
            if config['base_config']['env_args'] is None:
                config['base_config']['env_args'] = {}

            if k in config['base_config']['env_args']:
                config['base_config']['env_args'][k] = config_helper.convert_config_value_by_src(v, config['base_config']['env_args'][k])
            else:
                config['base_config']['env_args'][k] = config_helper.convert_config_value(v)

        if n_envs is not None:
            config['base_config']['n_envs'] = n_envs
        if max_iter is not None:
            config['base_config']['max_iter'] = max_iter
        if unity_port is not None:
            config['base_config']['unity_args']['port'] = unity_port
        if name is not None:
            config['base_config']['name'] = name
        if nn is not None:
            config['sac_config']['nn'] = nn
            for ma_config in ma_configs.values():
                ma_config['sac_config']['nn'] = nn

        config['base_config']['name'] = config_helper.generate_base_name(config['base_config']['name'])

        # The absolute directory of a specific training
        model_abs_dir = self.root_dir.joinpath('models',
                                               config['base_config']['env_name'],
                                               config['base_config']['name'])
        model_abs_dir.mkdir(parents=True, exist_ok=True)

        if self.train_mode:
            config_helper.save_config(config, model_abs_dir, 'config.yaml')
        config_helper.display_config(config, self._logger)
        convert_config_to_enum(config['sac_config'])
        convert_config_to_enum(config['oc_config'])

        for n, ma_config in ma_configs.items():
            if self.train_mode and n not in self.inference_ma_names:
                config_helper.save_config(ma_config, model_abs_dir, f'config_{n.replace("?", "-")}.yaml')
            config_helper.display_config(ma_config, self._logger, n)
            convert_config_to_enum(ma_config['sac_config'])
            convert_config_to_enum(ma_config['oc_config'])

        self.base_config = config['base_config']
        self.reset_config = config['reset_config']
        self.config = config
        self.ma_configs = ma_configs

        return model_abs_dir, config_abs_dir

    def _handle_copy_model(self,
                           copy_model: str | None):
        if copy_model is None:
            return

        src_model_abs_dir = self.root_dir.joinpath('models',
                                                   self.config['base_config']['env_name'],
                                                   copy_model)

        for item in src_model_abs_dir.iterdir():
            dest_item = self.model_abs_dir / item.name

            if item.is_dir():  # handle directory
                if not dest_item.exists():
                    shutil.copytree(item, dest_item)
                    self._logger.info(f"Copying directory: {item} -> {dest_item}")
                else:
                    self._logger.warning(f"{dest_item} exists, stop copying")
            else:  # handle file
                if not dest_item.exists():
                    shutil.copy2(item, dest_item)
                    self._logger.info(f"Copying file: {item} -> {dest_item}")
                else:
                    self._logger.warning(f"{dest_item} exists, stop copying")

    def _get_relative_package(self, abs_path: Path):
        r = abs_path.parent.relative_to(self.root_dir)
        return r.as_posix().replace("/", ".")

    def _init_env(self):
        if self.base_config['env_type'] == 'UNITY':
            from algorithm.env_wrapper.unity_wrapper import UnityWrapper

            if self.unity_run_in_editor:
                self.env = UnityWrapper(train_mode=self.train_mode,
                                        n_envs=self.base_config['n_envs'],
                                        model_abs_dir=self.model_abs_dir,
                                        max_n_envs_per_process=self.base_config['unity_args']['max_n_envs_per_process'],
                                        time_scale=self.unity_time_scale,
                                        env_args=self.base_config['env_args'])
            else:
                self.env = UnityWrapper(train_mode=self.train_mode,
                                        env_name=self.base_config['unity_args']['build_path'],
                                        n_envs=self.base_config['n_envs'],
                                        model_abs_dir=self.model_abs_dir,
                                        base_port=self.base_config['unity_args']['port'],
                                        max_n_envs_per_process=self.base_config['unity_args']['max_n_envs_per_process'],
                                        no_graphics=self.base_config['unity_args']['no_graphics'] and not self.render,
                                        force_vulkan=self.base_config['unity_args']['force_vulkan'],
                                        time_scale=self.unity_time_scale,
                                        scene=self.base_config['env_name'],
                                        env_args=self.base_config['env_args'])

        elif self.base_config['env_type'] == 'GYM':
            from algorithm.env_wrapper.gym_wrapper import GymWrapper

            self.env = GymWrapper(train_mode=self.train_mode,
                                  env_name=self.base_config['env_name'],
                                  env_args=self.base_config['env_args'],
                                  n_envs=self.base_config['n_envs'],
                                  model_abs_dir=self.model_abs_dir,
                                  render=self.render)

        elif self.base_config['env_type'] == 'TEST':
            from algorithm.env_wrapper.test_wrapper import TestWrapper

            self.env = TestWrapper(env_args=self.base_config['env_args'],
                                   n_envs=self.base_config['n_envs'],
                                   model_abs_dir=self.model_abs_dir)

        else:
            raise RuntimeError(f'Undefined Environment Type: {self.base_config["env_type"]}')

        if self.train_mode and self.base_config['offline_env_config']['enabled']:
            from algorithm.env_wrapper.offline_wrapper import OfflineWrapper

            if self.base_config['offline_env_config']['env_name'] == 'TEST':
                self.offline_env = TestOfflineWrapper(env_name=self.base_config['offline_env_config']['env_name'],
                                                      env_args=self.base_config['offline_env_config']['env_args'],
                                                      n_envs=self.base_config['n_envs'],
                                                      model_abs_dir=self.model_abs_dir)
            else:
                self.offline_env = OfflineWrapper(env_name=self.base_config['offline_env_config']['env_name'],
                                                  env_args=self.base_config['offline_env_config']['env_args'],
                                                  n_envs=self.base_config['offline_env_config']['n_envs'],
                                                  model_abs_dir=self.model_abs_dir)
            offline_ma_obs_names, offline_ma_obs_shapes, offline_ma_d_action_sizes, offline_ma_c_action_size = self.offline_env.init()
        else:
            self.offline_env = None

        if self.base_config['obs_preprocessor']:
            obs_preprocessor_abs_path = self._config_abs_dir / f'{self.base_config["obs_preprocessor"]}.py'

            spec = importlib.util.spec_from_file_location(f'{self._get_relative_package(obs_preprocessor_abs_path)}.{self.base_config["obs_preprocessor"]}',
                                                          obs_preprocessor_abs_path)
            self._logger.info(f'Loaded obs preprocessor in env dir: {obs_preprocessor_abs_path}')
            obs_preprocessor = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(obs_preprocessor)
            self.env = obs_preprocessor.ObsPreprocessor(self.env)

            if self.offline_env is not None:
                self.offline_env = obs_preprocessor.ObsPreprocessor(self.offline_env)

        ma_obs_names, ma_obs_shapes, ma_obs_dtypes, ma_d_action_sizes, ma_c_action_size = self.env.init()

        if self.offline_env is not None:
            for n in offline_ma_obs_names:
                if n not in ma_obs_names:
                    self._logger.warning(f'Offline env ma_name {n} not in online env')
                    continue
                for online_name, offline_name in zip(ma_obs_names[n], offline_ma_obs_names[n]):
                    if online_name != offline_name:
                        self._logger.warning(f'{n} obs_name offline {offline_name} not match online {online_name}')
                for online_os, offline_os in zip(ma_obs_shapes[n], offline_ma_obs_shapes[n]):
                    if online_os != offline_os:
                        self._logger.warning(f'{n} obs_shape offline {offline_os} not match online {online_os}')

        self.ma_manager = MultiAgentsManager(ma_obs_names,
                                             ma_obs_shapes,
                                             ma_obs_dtypes,
                                             ma_d_action_sizes,
                                             ma_c_action_size,
                                             self.inference_ma_names,
                                             self.model_abs_dir,
                                             max_episode_length=self.base_config['max_step_each_iter'],
                                             hit_reward=self.base_config['hit_reward'])
        for n, mgr in self.ma_manager:
            if n not in self.ma_configs:
                self._logger.warning(f'{n} not in ma_configs')
                mgr.set_config(self.config)
            else:
                mgr.set_config(self.ma_configs[n])

            self._logger.info(f'{n} observation names: {mgr.obs_names}')
            self._logger.info(f'{n} observation shapes: {mgr.obs_shapes}')
            self._logger.info(f'{n} observation dtyps: {mgr.obs_dtypes}')
            self._logger.info(f'{n} discrete action sizes: {mgr.d_action_sizes}')
            self._logger.info(f'{n} continuous action size: {mgr.c_action_size}')

        self._logger.info(f'{self.base_config["env_name"]} initialized')

    def _init_sac(self):
        for n, mgr in self.ma_manager:
            # If nn models exists, load saved model, or copy a new one
            saved_nn_abs_dir = mgr.model_abs_dir / 'nn'
            if not self.force_env_nn and saved_nn_abs_dir.exists():
                nn_abs_path = saved_nn_abs_dir / f'{mgr.config["sac_config"]["nn"]}.py'
                spec = importlib.util.spec_from_file_location(f'{self._get_relative_package(nn_abs_path)}.{mgr.config["sac_config"]["nn"]}',
                                                              nn_abs_path)
                self._logger.info(f'Loaded nn from existed {nn_abs_path}')
            else:
                nn_abs_path = self._config_abs_dir / f'{mgr.config["sac_config"]["nn"]}.py'

                spec = importlib.util.spec_from_file_location(f'{self._get_relative_package(nn_abs_path)}.{mgr.config["sac_config"]["nn"]}',
                                                              nn_abs_path)
                self._logger.info(f'Loaded nn in env dir: {nn_abs_path}')
                if not self.force_env_nn:
                    shutil.copytree(self._config_abs_dir, saved_nn_abs_dir,
                                    ignore=lambda _, names: [name for name in names
                                                             if name == '__pycache__'])

            nn = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(nn)
            mgr.config['sac_config']['nn'] = nn

            sac = SAC_Base(obs_names=mgr.obs_names,
                           obs_shapes=mgr.obs_shapes,
                           d_action_sizes=mgr.d_action_sizes,
                           c_action_size=mgr.c_action_size,
                           model_abs_dir=mgr.model_abs_dir,
                           device=self.device,
                           ma_name=None if len(self.ma_manager) == 1 else n,
                           train_mode=self.train_mode and n not in self.inference_ma_names,
                           last_ckpt=self.last_ckpt,

                           nn_config=mgr.config['nn_config'],
                           **mgr.config['sac_config'],

                           offline_enabled=self.base_config['offline_env_config']['enabled'],

                           replay_config=mgr.config['replay_config'])
            mgr.set_rl(sac)

            if self.train_mode and self.base_config['offline_env_config']['enabled']:
                mgr.set_il(ImitationBase(sac))

    def _run(self):
        force_reset = False
        is_training_iteration = False  # Is current iteration training
        inference_iterations = 0  # The inference iteration count
        trained_steps = 0  # The steps that RL trained

        self.ma_manager.set_train_mode(is_training_iteration)  # The first iteration is inference

        try:
            while inference_iterations != self.base_config['max_iter']:
                if self.base_config['max_step'] != -1 and trained_steps >= self.base_config['max_step']:
                    break

                step = 0
                iter_time = time.time()

                if inference_iterations == 0 \
                        or self.base_config['reset_on_iteration'] \
                        or self.ma_manager.max_reached \
                        or self.offline_env is not None \
                        or force_reset:
                    self.ma_manager.reset()

                    if is_training_iteration and self.offline_env is not None:
                        ma_agent_ids, ma_obs_list, ma_offline_action = self.offline_env.reset(reset_config=self.reset_config)
                    else:
                        ma_agent_ids, ma_obs_list = self.env.reset(reset_config=self.reset_config)
                        ma_offline_action = None

                    ma_d_action, ma_c_action = self.ma_manager.get_ma_action(
                        ma_agent_ids=ma_agent_ids,
                        ma_obs_list=ma_obs_list,
                        ma_last_reward={n: np.zeros(len(agent_ids))
                                        for n, agent_ids in ma_agent_ids.items()},
                        ma_offline_action=ma_offline_action,
                        disable_sample=self.disable_sample
                    )

                    force_reset = False
                else:
                    self.ma_manager.reset_and_continue()

                while not self.ma_manager.done:
                    try:
                        if is_training_iteration and self.offline_env is not None:
                            with self._profiler('offline_env.step', repeat=10):
                                (decision_step,
                                 terminal_step,
                                 all_envs_done) = self.offline_env.step()
                        else:
                            with self._profiler('env.step', repeat=10):
                                (decision_step,
                                 terminal_step,
                                 all_envs_done) = self.env.step(ma_d_action, ma_c_action)
                            self._extra_step(ma_d_action, ma_c_action)
                    except RuntimeError:
                        force_reset = True

                        self._logger.error('Step encounters error, episode ignored')
                        break

                    with self._profiler('get_ma_action', repeat=10):
                        ma_d_action, ma_c_action = self.ma_manager.get_ma_action(
                            ma_agent_ids=decision_step.ma_agent_ids,
                            ma_obs_list=decision_step.ma_obs_list,
                            ma_last_reward=decision_step.ma_last_reward,
                            ma_offline_action=decision_step.ma_offline_action,
                            disable_sample=self.disable_sample
                        )

                    self.ma_manager.end_episode(
                        ma_agent_ids=terminal_step.ma_agent_ids,
                        ma_obs_list=terminal_step.ma_obs_list,
                        ma_last_reward=terminal_step.ma_last_reward,
                        ma_max_reached=terminal_step.ma_max_reached
                    )
                    if all_envs_done or step == self.base_config['max_step_each_iter']:
                        self.ma_manager.end_episode(
                            ma_agent_ids=decision_step.ma_agent_ids,
                            ma_obs_list=decision_step.ma_obs_list,
                            ma_last_reward=decision_step.ma_last_reward,
                            ma_max_reached={n: np.ones_like(agent_ids, dtype=bool)
                                            for n, agent_ids in decision_step.ma_agent_ids.items()},
                            force_terminated=True
                        )
                        self.ma_manager.force_end_all_episode()

                    if self.train_mode:
                        if not is_training_iteration:
                            self.ma_manager.log_episode()

                        # If the offline RL is disabled
                        # OR
                        # offline RL is enabled and is training iteration
                        if self.offline_env is None or is_training_iteration:
                            self.ma_manager.put_episode()

                            with self._profiler('train', repeat=10) as profiler:
                                next_trained_steps = self.ma_manager.train(trained_steps)
                                if next_trained_steps == trained_steps:
                                    profiler.ignore()
                                trained_steps = next_trained_steps
                    else:
                        self.ma_manager.log_episode(force=True)
                        self.ma_manager.clear_tmp_episode_trans_list()

                    step += 1

                if self.train_mode and not is_training_iteration:
                    self._log_episode_summaries()

                if not is_training_iteration:
                    self._log_episode_info(inference_iterations, time.time() - iter_time)

                if self.train_mode:
                    is_training_iteration = not is_training_iteration
                    self.ma_manager.set_train_mode(is_training_iteration)

                self.ma_manager.reset_dead_agents()
                self.ma_manager.clear_tmp_episode_trans_list()

                p_model = self.model_abs_dir.joinpath('save_model')
                if self.train_mode and p_model.exists():
                    p_replay_buffer = self.model_abs_dir.joinpath('save_replay_buffer')
                    if p_replay_buffer.exists():
                        self.ma_manager.save_model(p_replay_buffer.exists())
                        p_replay_buffer.unlink()
                    else:
                        self.ma_manager.save_model(False)

                    p_model.unlink()

                if not is_training_iteration:
                    inference_iterations += 1

        except KeyboardInterrupt:
            self._logger.warning('KeyboardInterrupt')

        finally:
            if self.train_mode:
                self.ma_manager.save_model()
            self.env.close()
            self.ma_manager.close()

            self._logger.info('Training terminated')

    def _extra_step(self,
                    ma_d_action: dict[str, np.ndarray],
                    ma_c_action: dict[str, np.ndarray]):
        pass

    def _log_episode_summaries(self):
        for n, mgr in self.ma_manager:
            if n in self.inference_ma_names or len(mgr.non_empty_agents) == 0:
                continue

            rewards = np.array([a.reward for a in mgr.non_empty_agents])
            steps = np.array([a.steps for a in mgr.non_empty_agents])

            summaries = [
                {'tag': 'reward/mean', 'simple_value': rewards.mean()},
                {'tag': 'reward/max', 'simple_value': rewards.max()},
                {'tag': 'reward/min', 'simple_value': rewards.min()},
                {'tag': 'metric/steps', 'simple_value': steps.mean()},
            ]
            if mgr.hit_reward is not None:
                hit = sum([a.hit for a in mgr.non_empty_agents])
                summaries.append({'tag': 'reward/hit', 'simple_value': hit / len(mgr.non_empty_agents)})

            mgr.rl.write_constant_summaries(summaries)

    def _log_episode_info(self, iteration, iter_time):
        for n, mgr in self.ma_manager:
            if len(mgr.non_empty_agents) == 0:
                continue
            global_step = format_global_step(mgr.rl.get_global_step())
            rewards = [a.reward for a in mgr.non_empty_agents]
            rewards = ", ".join([f"{i:6.1f}" for i in rewards])
            max_step = max([a.steps for a in mgr.non_empty_agents])

            if mgr.hit_reward is None:
                self._logger.info(f'{n} {iteration}({global_step}), T {iter_time:.2f}s, S {max_step}, R {rewards} [{len(mgr.non_empty_agents)}]')
            else:
                hit = sum([a.hit for a in mgr.non_empty_agents])
                self._logger.info(f'{n} {iteration}({global_step}), T {iter_time:.2f}s, S {max_step}, R {rewards}, hit {hit}/[{len(mgr.non_empty_agents)}]')

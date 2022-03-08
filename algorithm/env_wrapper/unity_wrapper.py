import itertools
import logging
import math
import multiprocessing
import multiprocessing.connection
import os
import traceback
from typing import List

import numpy as np
from mlagents_envs.environment import ActionTuple, UnityEnvironment
from mlagents_envs.exception import UnityTimeOutException
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfig, EngineConfigurationChannel)
from mlagents_envs.side_channel.environment_parameters_channel import \
    EnvironmentParametersChannel

INIT = 0
RESET = 1
STEP = 2
CLOSE = 3
MAX_N_AGENTS_PER_PROCESS = 10


class UnityWrapperProcess:
    def __init__(self,
                 conn: multiprocessing.connection.Connection = None,
                 train_mode=True,
                 file_name=None,
                 worker_id=0,
                 base_port=5005,
                 no_graphics=True,
                 seed=None,
                 scene=None,
                 additional_args=None,
                 n_agents=1):
        """
        Args:
            train_mode: If in train mode, Unity will speed up
            file_name: The executable path. The UnityEnvironment will run in editor if None
            worker_id: Offset from base_port
            base_port: The port that communicate to Unity. It will be set to 5004 automatically if in editor.
            no_graphics: If Unity runs in no graphic mode. It must be set to False if Unity has camera sensor.
            seed: Random seed
            scene: The scene name
            n_agents: The agents count
        """
        self.scene = scene
        self.n_agents = n_agents

        seed = seed if seed is not None else np.random.randint(0, 65536)
        additional_args = [] if additional_args is None else additional_args.split(' ')

        self.engine_configuration_channel = EngineConfigurationChannel()
        self.environment_parameters_channel = EnvironmentParametersChannel()

        self.environment_parameters_channel.set_float_parameter('env_copys', float(n_agents))

        if conn:
            try:
                from algorithm import config_helper
                config_helper.set_logger()
            except:
                pass

            self._logger = logging.getLogger(f'UnityWrapper.Process_{os.getpid()}')
        else:
            self._logger = logging.getLogger('UnityWrapper.Process')

        self._env = UnityEnvironment(file_name=file_name,
                                     worker_id=worker_id,
                                     base_port=base_port if file_name else None,
                                     no_graphics=no_graphics and train_mode,
                                     seed=seed,
                                     additional_args=['--scene', scene] + additional_args,
                                     side_channels=[self.engine_configuration_channel,
                                                    self.environment_parameters_channel])

        self.engine_configuration_channel.set_configuration_parameters(
            width=200 if train_mode else 1280,
            height=200 if train_mode else 720,
            quality_level=5,
            time_scale=20 if train_mode else 1)

        self._env.reset()
        self.bahavior_name = list(self._env.behavior_specs)[0]

        if conn:
            try:
                while True:
                    cmd, data = conn.recv()
                    if cmd == INIT:
                        conn.send(self.init())
                    elif cmd == RESET:
                        conn.send(self.reset(data))
                    elif cmd == STEP:
                        conn.send(self.step(*data))
                    elif cmd == CLOSE:
                        self.close()
            except:
                self._logger.error(traceback.format_exc())

    def init(self):
        """
        Returns:
            observation shapes: tuple[(o1, ), (o2, ), (o3_1, o3_2, o3_3), ...]
            discrete action size: int, sum of all action branches
            continuous action size: int
        """
        behavior_spec = self._env.behavior_specs[self.bahavior_name]
        obs_names = [o.name for o in behavior_spec.observation_specs]
        self._logger.info(f'Observation names: {obs_names}')
        obs_shapes = [o.shape for o in behavior_spec.observation_specs]
        self._logger.info(f'Observation shapes: {obs_shapes}')

        self._empty_action = behavior_spec.action_spec.empty_action

        discrete_action_size = 0
        if behavior_spec.action_spec.discrete_size > 0:
            discrete_action_size = 1
            action_product_list = []
            for action, branch_size in enumerate(behavior_spec.action_spec.discrete_branches):
                discrete_action_size *= branch_size
                action_product_list.append(range(branch_size))
                self._logger.info(f"Discrete action branch {action} has {branch_size} different actions")

            self.action_product = np.array(list(itertools.product(*action_product_list)))

        continuous_action_size = behavior_spec.action_spec.continuous_size

        self._logger.info(f'Continuous action size: {continuous_action_size}')

        self.d_action_size = discrete_action_size
        self.c_action_size = continuous_action_size

        for o in behavior_spec.observation_specs:
            if len(o.shape) >= 3:
                self.engine_configuration_channel.set_configuration_parameters(quality_level=5)
                break

        return obs_shapes, discrete_action_size, continuous_action_size

    def reset(self, reset_config=None):
        """
        return:
            observations: list[(NAgents, o1), (NAgents, o2), (NAgents, o3_1, o3_2, o3_3)]
        """
        reset_config = {} if reset_config is None else reset_config
        for k, v in reset_config.items():
            self.environment_parameters_channel.set_float_parameter(k, float(v))

        self._env.reset()
        decision_steps, terminal_steps = self._env.get_steps(self.bahavior_name)

        return [obs.astype(np.float32) for obs in decision_steps.obs]

    def step(self, d_action, c_action):
        """
        Args:
            d_action: (NAgents, discrete_action_size), one hot like action
            c_action: (NAgents, continuous_action_size)

        Returns:
            observations: list[(NAgents, o1), (NAgents, o2), (NAgents, o3_1, o3_2, o3_3)]
            rewards: (NAgents, )
            done: (NAgents, ), np.bool
            max_step: (NAgents, ), np.bool
        """

        if self.d_action_size:
            d_action = np.argmax(d_action, axis=1)
            d_action = self.action_product[d_action]

        self._env.set_actions(self.bahavior_name,
                              ActionTuple(continuous=c_action, discrete=d_action))
        self._env.step()

        decision_steps, terminal_steps = self._env.get_steps(self.bahavior_name)

        tmp_terminal_steps = terminal_steps

        while len(decision_steps) == 0:
            self._env.set_actions(self.bahavior_name, self._empty_action(0))
            self._env.step()
            decision_steps, terminal_steps = self._env.get_steps(self.bahavior_name)
            tmp_terminal_steps.agent_id = np.concatenate([tmp_terminal_steps.agent_id,
                                                          terminal_steps.agent_id])
            tmp_terminal_steps.reward = np.concatenate([tmp_terminal_steps.reward,
                                                        terminal_steps.reward])
            tmp_terminal_steps.interrupted = np.concatenate([tmp_terminal_steps.interrupted,
                                                            terminal_steps.interrupted])

        reward = decision_steps.reward
        reward[tmp_terminal_steps.agent_id] = tmp_terminal_steps.reward

        done = np.full([len(decision_steps), ], False, dtype=np.bool)
        done[tmp_terminal_steps.agent_id] = True

        max_step = np.full([len(decision_steps), ], False, dtype=np.bool)
        max_step[tmp_terminal_steps.agent_id] = tmp_terminal_steps.interrupted

        return ([obs.astype(np.float32) for obs in decision_steps.obs],
                decision_steps.reward.astype(np.float32),
                done,
                max_step)

    def close(self):
        self._env.close()
        self._logger.warning(f'Process {os.getpid()} exits')


class UnityWrapper:
    def __init__(self,
                 train_mode=True,
                 file_name=None,
                 base_port=5005,
                 no_graphics=True,
                 seed=None,
                 scene=None,
                 additional_args=None,
                 n_agents=1,
                 force_seq=None):
        """
        Args:
            train_mode: If in train mode, Unity will run in the highest quality
            file_name: The executable path. The UnityEnvironment will run in editor if None
            base_port: The port that communicate to Unity. It will be set to 5004 automatically if in editor.
            no_graphics: If Unity runs in no graphic mode. It must be set to False if Unity has camera sensor.
            seed: Random seed
            scene: The scene name
            n_agents: The agents count
        """
        # If use multiple processes
        if force_seq is None:
            self._seq_envs: bool = n_agents <= MAX_N_AGENTS_PER_PROCESS
        else:
            self._seq_envs: bool = force_seq

        self.env_length = math.ceil(n_agents / MAX_N_AGENTS_PER_PROCESS)

        if self._seq_envs:
            # All environments are executed sequentially
            self._envs: List[UnityWrapperProcess] = []

            for i in range(self.env_length):
                self._envs.append(UnityWrapperProcess(None,
                                                      train_mode,
                                                      file_name,
                                                      i,
                                                      base_port,
                                                      no_graphics,
                                                      seed,
                                                      scene,
                                                      additional_args,
                                                      min(MAX_N_AGENTS_PER_PROCESS, n_agents - i * MAX_N_AGENTS_PER_PROCESS)))
        else:
            # All environments are executed in parallel
            self._conns: List[multiprocessing.connection.Connection] = []

            for i in range(self.env_length):
                parent_conn, child_conn = multiprocessing.Pipe()
                self._conns.append(parent_conn)
                p = multiprocessing.Process(target=UnityWrapperProcess,
                                            args=(child_conn,
                                                  train_mode,
                                                  file_name,
                                                  i,
                                                  base_port,
                                                  no_graphics,
                                                  seed,
                                                  scene,
                                                  additional_args,
                                                  min(MAX_N_AGENTS_PER_PROCESS, n_agents - i * MAX_N_AGENTS_PER_PROCESS)),
                                            daemon=True)
                p.start()

    def init(self):
        """
        Returns:
            observation shapes: tuple[(o1, ), (o2, ), (o3_1, o3_2, o3_3), ...]
            discrete action size: int, sum of all action branches
            continuous action size: int
        """
        if self._seq_envs:
            for env in self._envs:
                results = env.init()
        else:
            for conn in self._conns:
                conn.send((INIT, None))
                results = conn.recv()

        return results

    def reset(self, reset_config=None):
        """
        return:
            observations: list[(NAgents, o1), (NAgents, o2), (NAgents, o3_1, o3_2, o3_3)]
        """
        if self._seq_envs:
            all_obses_list = [env.reset(reset_config) for env in self._envs]
        else:
            for conn in self._conns:
                conn.send((RESET, reset_config))

            all_obses_list = [conn.recv() for conn in self._conns]

        obses_list = [np.concatenate(obses) for obses in zip(*all_obses_list)]
        return obses_list

    def step(self, d_action, c_action):
        """
        Args:
            d_action: (NAgents, discrete_action_size), one hot like action
            c_action: (NAgents, continuous_action_size)

        Returns:
            observations: list[(NAgents, o1), (NAgents, o2), (NAgents, o3_1, o3_2, o3_3)]
            rewards: (NAgents, )
            dones: (NAgents, ), np.bool
            max_steps: (NAgents, ), np.bool
        """
        all_obses_list = []
        all_rewards = []
        all_dones = []
        all_max_steps = []

        for i in range(self.env_length):
            actions = (
                d_action[i * MAX_N_AGENTS_PER_PROCESS:(i + 1) * MAX_N_AGENTS_PER_PROCESS] if d_action is not None else None,
                c_action[i * MAX_N_AGENTS_PER_PROCESS:(i + 1) * MAX_N_AGENTS_PER_PROCESS] if c_action is not None else None
            )

            if self._seq_envs:
                (obses_list,
                 rewards,
                 dones,
                 max_steps) = self._envs[i].step(*actions)

                all_obses_list.append(obses_list)
                all_rewards.append(rewards)
                all_dones.append(dones)
                all_max_steps.append(max_steps)
            else:
                self._conns[i].send((STEP, actions))

        if not self._seq_envs:
            for conn in self._conns:
                (obses_list,
                 rewards,
                 dones,
                 max_steps) = conn.recv()

                all_obses_list.append(obses_list)
                all_rewards.append(rewards)
                all_dones.append(dones)
                all_max_steps.append(max_steps)

        obses_list = [np.concatenate(obses) for obses in zip(*all_obses_list)]
        rewards = np.concatenate(all_rewards)
        dones = np.concatenate(all_dones)
        max_steps = np.concatenate(all_max_steps)

        return obses_list, rewards, dones, max_steps

    def close(self):
        if self._seq_envs:
            for env in self._envs:
                env.close()
        else:
            for conn in self._conns:
                try:
                    conn.send((CLOSE, None))
                except:
                    pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    env = UnityWrapper(train_mode=True,
                       file_name=r'/data/linux-RL-Envs/RLEnvironments.x86_64',
                       scene='3DBall',
                       n_agents=10)
    obs_shape_list, d_action_size, c_action_size = env.init()

    for i in range(100):
        obs_list = env.reset()
        n_agents = obs_list[0].shape[0]
        for j in range(100):
            d_action, c_action = None, None
            if d_action_size:
                d_action = np.random.randint(0, d_action_size, size=n_agents)
                d_action = np.eye(d_action_size, dtype=np.int32)[d_action]
            if c_action_size:
                c_action = np.random.randn(n_agents, c_action_size)

            obs_list, reward, done, max_step = env.step(d_action, c_action)

    env.close()

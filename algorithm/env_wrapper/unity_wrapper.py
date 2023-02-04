import itertools
import logging
import math
import multiprocessing
import multiprocessing.connection
import os
import uuid
from typing import List

import numpy as np
from mlagents_envs.environment import (ActionTuple, DecisionSteps,
                                       TerminalSteps, UnityEnvironment)
from mlagents_envs.exception import UnityTimeOutException
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfig, EngineConfigurationChannel)
from mlagents_envs.side_channel.environment_parameters_channel import \
    EnvironmentParametersChannel
from mlagents_envs.side_channel.side_channel import (IncomingMessage,
                                                     OutgoingMessage,
                                                     SideChannel)

INIT = 0
RESET = 1
STEP = 2
CLOSE = 3
MAX_N_AGENTS_PER_PROCESS = 10


class OptionChannel(SideChannel):
    def __init__(self) -> None:
        super().__init__(uuid.UUID("621f0a70-4f87-11ea-a6bf-784f4387d1f7"))

    def send_option(self, data: int) -> None:
        # Add the string to an OutgoingMessage
        msg = OutgoingMessage()
        msg.write_int32(data)
        # We call this method to queue the data we want to send
        super().queue_message_to_send(msg)

    def on_message_received(self, msg: IncomingMessage) -> None:
        pass


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

        self.option_channel = OptionChannel()

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
                                                    self.environment_parameters_channel,
                                                    self.option_channel])

        self.engine_configuration_channel.set_configuration_parameters(
            width=200 if train_mode else 1280,
            height=200 if train_mode else 720,
            quality_level=5,
            time_scale=20 if train_mode else 1)

        self._env.reset()
        self.behavior_names = list(self._env.behavior_specs)

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
            finally:
                self._logger.warning(f'Process {os.getpid()} exits with error')

    def init(self):
        """
        Returns:
            observation shapes: tuple[(o1, ), (o2, ), (o3_1, o3_2, o3_3), ...]
            discrete action size: int, sum of all action branches
            continuous action size: int
        """
        self.ma_obs_names = {}
        self.ma_obs_shapes = {}
        self.ma_d_action_size = {}
        self.ma_c_action_size = {}

        for n in self.behavior_names:
            behavior_spec = self._env.behavior_specs[n]
            obs_names = [o.name for o in behavior_spec.observation_specs]
            self._logger.info(f'{n} Observation names: {obs_names}')
            self.ma_obs_names[n] = obs_names
            obs_shapes = [o.shape for o in behavior_spec.observation_specs]
            self._logger.info(f'{n} Observation shapes: {obs_shapes}')
            self.ma_obs_shapes[n] = obs_shapes

            self._empty_action = behavior_spec.action_spec.empty_action

            discrete_action_size = 0
            if behavior_spec.action_spec.discrete_size > 0:
                discrete_action_size = 1
                action_product_list = []
                for action, branch_size in enumerate(behavior_spec.action_spec.discrete_branches):
                    discrete_action_size *= branch_size
                    action_product_list.append(range(branch_size))
                    self._logger.info(f"{n} Discrete action branch {action} has {branch_size} different actions")

                self.action_product = np.array(list(itertools.product(*action_product_list)))

            continuous_action_size = behavior_spec.action_spec.continuous_size

            self._logger.info(f'{n} Continuous action size: {continuous_action_size}')

            self.ma_d_action_size[n] = discrete_action_size
            self.ma_c_action_size[n] = continuous_action_size

            for o in behavior_spec.observation_specs:
                if len(o.shape) >= 3:
                    self.engine_configuration_channel.set_configuration_parameters(quality_level=5)
                    break

        self._logger.info('Initialized')

        return self.ma_obs_shapes, self.ma_d_action_size, self.ma_c_action_size

    def reset(self, reset_config=None):
        """
        return:
            observations: list[(NAgents, o1), (NAgents, o2), (NAgents, o3_1, o3_2, o3_3)]
        """
        reset_config = {} if reset_config is None else reset_config
        for k, v in reset_config.items():
            self.environment_parameters_channel.set_float_parameter(k, float(v))

        self._env.reset()

        ma_obs_list = {}
        for n in self.behavior_names:
            decision_steps, terminal_steps = self._env.get_steps(n)
            ma_obs_list[n] = [obs.astype(np.float32) for obs in decision_steps.obs]

        return ma_obs_list

    def step(self, ma_d_action, ma_c_action):
        """
        Args:
            d_action: (NAgents, discrete_action_size), one hot like action
            c_action: (NAgents, continuous_action_size)

        Returns:
            observations: list[(NAgents, o1), (NAgents, o2), (NAgents, o3_1, o3_2, o3_3)]
            reward: (NAgents, )
            done: (NAgents, ), bool
            max_step: (NAgents, ), bool
        """
        ma_obs_list = {}
        ma_reward = {}
        ma_done = {}
        ma_max_step = {}

        for n in self.behavior_names:
            d_action = c_action = None

            if self.ma_d_action_size[n]:
                d_action = ma_d_action[n]
                d_action = np.argmax(d_action, axis=1)
                d_action = self.action_product[d_action]
            c_action = ma_c_action[n]

            self._env.set_actions(n,
                                  ActionTuple(continuous=c_action, discrete=d_action))

        self._env.step()

        tmp_ma_decision_steps = {}
        tmp_ma_terminal_steps = {}

        for n in self.behavior_names:
            decision_steps, terminal_steps = self._env.get_steps(n)
            tmp_ma_decision_steps[n] = decision_steps
            tmp_ma_terminal_steps[n] = terminal_steps

        while any([len(decision_steps) == 0 for decision_steps in tmp_ma_decision_steps.values()]):
            for n in self.behavior_names:
                if len(tmp_ma_decision_steps[n]) > 0:
                    continue
                self._env.set_actions(n, self._empty_action(0))

            self._env.step()

            for n in self.behavior_names:
                if len(tmp_ma_decision_steps[n]) > 0:
                    continue
                decision_steps, terminal_steps = self._env.get_steps(n)
                tmp_ma_decision_steps[n] = decision_steps
                tmp_ma_terminal_steps[n].agent_id = np.concatenate([tmp_ma_terminal_steps[n].agent_id,
                                                                    terminal_steps.agent_id])

                tmp_ma_terminal_steps[n].reward = np.concatenate([tmp_ma_terminal_steps[n].reward,
                                                                  terminal_steps.reward])
                tmp_ma_terminal_steps[n].interrupted = np.concatenate([tmp_ma_terminal_steps[n].interrupted,
                                                                       terminal_steps.interrupted])

        for n in self.behavior_names:
            decision_steps: DecisionSteps = tmp_ma_decision_steps[n]
            terminal_steps: TerminalSteps = tmp_ma_terminal_steps[n]

            reward = decision_steps.reward
            for i, agent_id in enumerate(terminal_steps.agent_id):
                reward[decision_steps.agent_id_to_index[agent_id]] = terminal_steps.reward[i]

            done = np.full([len(decision_steps), ], False, dtype=bool)
            for i, agent_id in enumerate(terminal_steps.agent_id):
                done[decision_steps.agent_id_to_index[agent_id]] = True

            max_step = np.full([len(decision_steps), ], False, dtype=bool)
            for i, agent_id in enumerate(terminal_steps.agent_id):
                max_step[decision_steps.agent_id_to_index[agent_id]] = terminal_steps.interrupted[i]

            ma_obs_list[n] = [obs.astype(np.float32) for obs in decision_steps.obs]
            ma_reward[n] = reward.astype(np.float32)
            ma_done[n] = done
            ma_max_step[n] = max_step

        return ma_obs_list, ma_reward, ma_done, ma_max_step

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
        self.train_mode = train_mode
        self.file_name = file_name
        self.base_port = base_port
        self.no_graphics = no_graphics
        self.seed = seed
        self.scene = scene
        self.additional_args = additional_args
        self.n_agents = n_agents

        # If use multiple processes
        if force_seq is None:
            self._seq_envs: bool = n_agents <= MAX_N_AGENTS_PER_PROCESS
        else:
            self._seq_envs: bool = force_seq

        self._process_id = 0

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
            self._conns: List[multiprocessing.connection.Connection] = [None] * self.env_length

            self._generate_processes()

    def _generate_processes(self, force_init=False):
        if self._seq_envs:
            return

        for i, conn in enumerate(self._conns):
            if conn is None:
                parent_conn, child_conn = multiprocessing.Pipe()
                self._conns[i] = parent_conn
                p = multiprocessing.Process(target=UnityWrapperProcess,
                                            args=(child_conn,
                                                  self.train_mode,
                                                  self.file_name,
                                                  self._process_id,
                                                  self.base_port,
                                                  self.no_graphics,
                                                  self.seed,
                                                  self.scene,
                                                  self.additional_args,
                                                  min(MAX_N_AGENTS_PER_PROCESS, self.n_agents - i * MAX_N_AGENTS_PER_PROCESS)),
                                            daemon=True)
                p.start()

                if force_init:
                    parent_conn.send((INIT, None))
                    parent_conn.recv()

                self._process_id += 1

    def send_option(self, option: int):
        self._envs[0].option_channel.send_option(option)

    def init(self):
        """
        Returns:
            observation shapes: dict[str, tuple[(o1, ), (o2, ), (o3_1, o3_2, o3_3), ...]]
            discrete action size: dict[str, int], sum of all action branches
            continuous action size: dict[str, int]
        """
        if self._seq_envs:
            for env in self._envs:
                results = env.init()
        else:
            for conn in self._conns:
                conn.send((INIT, None))
                results = conn.recv()

        self.behavior_names = list(results[0].keys())

        return results

    def reset(self, reset_config=None):
        """
        return:
            observation: dict[str, list[(NAgents, o1), (NAgents, o2), (NAgents, o3_1, o3_2, o3_3)]]
        """
        if self._seq_envs:
            envs_ma_obs_list = [env.reset(reset_config) for env in self._envs]
        else:
            for conn in self._conns:
                conn.send((RESET, reset_config))

            envs_ma_obs_list = [conn.recv() for conn in self._conns]

        ma_obs_list = {n: [np.concatenate(env_obs_list) for env_obs_list in zip(*[ma_obs_list[n] for ma_obs_list in envs_ma_obs_list])]
                       for n in self.behavior_names}

        return ma_obs_list

    def step(self, ma_d_action, ma_c_action):
        """
        Args:
            d_action: dict[str, (NAgents, discrete_action_size)], one hot like action
            c_action: dict[str, (NAgents, continuous_action_size)]

        Returns:
            observation: dict[str, list[(NAgents, o1), (NAgents, o2), (NAgents, o3_1, o3_2, o3_3)]]
            reward: dict[str, (NAgents, )]
            done: dict[str, (NAgents, )], bool
            max_step: dict[str, (NAgents, )], bool
        """
        ma_envs_obs_list = {n: [] for n in self.behavior_names}
        ma_envs_reward = {n: [] for n in self.behavior_names}
        ma_envs_done = {n: [] for n in self.behavior_names}
        ma_envs_max_step = {n: [] for n in self.behavior_names}

        for i in range(self.env_length):
            tmp_ma_d_actions = {
                n:
                ma_d_action[n][i * MAX_N_AGENTS_PER_PROCESS:(i + 1) * MAX_N_AGENTS_PER_PROCESS] if ma_d_action[n] is not None else None
                for n in self.behavior_names}

            tmp_ma_c_actions = {
                n:
                ma_c_action[n][i * MAX_N_AGENTS_PER_PROCESS:(i + 1) * MAX_N_AGENTS_PER_PROCESS] if ma_c_action[n] is not None else None
                for n in self.behavior_names}

            if self._seq_envs:
                (ma_obs_list,
                 ma_reward,
                 ma_done,
                 ma_max_step) = self._envs[i].step(tmp_ma_d_actions, tmp_ma_c_actions)

                for n in self.behavior_names:
                    ma_envs_obs_list[n].append(ma_obs_list[n])
                    ma_envs_reward[n].append(ma_reward[n])
                    ma_envs_done[n].append(ma_done[n])
                    ma_envs_max_step[n].append(ma_max_step[n])
            else:
                self._conns[i].send((STEP, (tmp_ma_d_actions, tmp_ma_c_actions)))

        if not self._seq_envs:
            succeeded = True

            for i, conn in enumerate(self._conns):
                try:
                    (ma_obs_list,
                     ma_reward,
                     ma_done,
                     ma_max_step) = conn.recv()

                    for n in self.behavior_names:
                        ma_envs_obs_list[n].append(ma_obs_list[n])
                        ma_envs_reward[n].append(ma_reward[n])
                        ma_envs_done[n].append(ma_done[n])
                        ma_envs_max_step[n].append(ma_max_step[n])
                except:
                    self._conns[i] = None
                    succeeded = False

            if not succeeded:
                self._generate_processes(force_init=True)

                return None, None, None, None

        ma_obs_list = {n: [np.concatenate(obs) for obs in zip(*ma_envs_obs_list[n])] for n in self.behavior_names}
        ma_reward = {n: np.concatenate(ma_envs_reward[n]) for n in self.behavior_names}
        ma_done = {n: np.concatenate(ma_envs_done[n]) for n in self.behavior_names}
        ma_max_step = {n: np.concatenate(ma_envs_max_step[n]) for n in self.behavior_names}

        return ma_obs_list, ma_reward, ma_done, ma_max_step

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

    N_AGENTS = 2

    env = UnityWrapper(train_mode=True,
                       file_name=r'D:\Unity\win-RL-Envs\RLEnvironments.exe',
                       scene='Roller',
                       n_agents=N_AGENTS)
    ma_obs_shapes, ma_d_action_size, ma_c_action_size = env.init()
    ma_names = list(ma_obs_shapes.keys())

    for i in range(100):
        ma_obs_list = env.reset()

        for j in range(100):
            ma_d_action = {}
            ma_c_action = {}
            for n in ma_names:
                d_action, c_action = None, None
                if ma_d_action_size[n]:
                    d_action = np.random.randint(0, ma_d_action_size[n], size=N_AGENTS)
                    d_action = np.eye(ma_d_action_size[n], dtype=np.int32)[d_action]
                if ma_c_action_size[n]:
                    c_action = np.random.randn(N_AGENTS, ma_c_action_size[n])

            ma_d_action[n] = d_action
            ma_c_action[n] = c_action

            ma_obs_list, ma_reward, ma_done, ma_max_step = env.step(ma_d_action, ma_c_action)

    env.close()

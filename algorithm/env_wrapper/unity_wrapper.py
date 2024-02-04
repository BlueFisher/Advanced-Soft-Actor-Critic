import json
import logging
import math
import multiprocessing
import multiprocessing.connection
import os
import random
import time
import uuid
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from mlagents_envs.environment import (ActionTuple, DecisionSteps,
                                       TerminalSteps, UnityEnvironment)
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfig, EngineConfigurationChannel)
from mlagents_envs.side_channel.environment_parameters_channel import \
    EnvironmentParametersChannel
from mlagents_envs.side_channel.side_channel import (IncomingMessage,
                                                     OutgoingMessage,
                                                     SideChannel)

from .env_wrapper import EnvWrapper

INIT = 0
RESET = 1
STEP = 2
CLOSE = 3
MAX_N_ENVS_PER_PROCESS = 10


class OptionChannel(SideChannel):
    def __init__(self) -> None:
        super().__init__(uuid.UUID("621f0a70-4f87-11ea-a6bf-784f4387d1f7"))

    def send_option(self, data: Dict[str, int]) -> None:
        # Add the string to an OutgoingMessage
        msg = OutgoingMessage()
        msg.write_string(json.dumps(data))
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
                 time_scale=None,
                 seed=None,
                 scene=None,
                 additional_args=None,
                 n_envs=1,
                 group_aggregation=False):
        """
        Args:
            conn: Connection if run in multiprocessing mode
            train_mode: If in train mode, Unity will speed up
            file_name: The executable path. The UnityEnvironment will run in editor if None
            worker_id: Offset from base_port
            base_port: The port that communicate to Unity. It will be set to 5004 automatically if in editor.
            no_graphics: If Unity runs in no graphic mode. It must be set to False if Unity has camera sensor.
            time_scale: Time scale of Unity. If None: time_scale = 20 if train_mode else 1
            seed: Random seed
            scene: The scene name
            n_envs: The env copies count
            group_aggregation: If aggregate group agents
        """
        self.scene = scene
        self.n_envs = n_envs
        self.group_aggregation = group_aggregation

        seed = seed if seed is not None else random.randint(0, 65536)
        if additional_args is None:
            additional_args = []
        elif isinstance(additional_args, str):
            additional_args = additional_args.split(' ')

        self.engine_configuration_channel = EngineConfigurationChannel()
        self.environment_parameters_channel = EnvironmentParametersChannel()

        self.environment_parameters_channel.set_float_parameter('env_copys', float(n_envs))

        self.option_channel = OptionChannel()

        if conn:  # If run in multiprocessing mode
            try:
                from algorithm import config_helper
                config_helper.set_logger()
            except:
                pass

            self._logger = logging.getLogger(f'UnityWrapper.Process_{worker_id}_p{os.getpid()}')
        else:
            self._logger = logging.getLogger(f'UnityWrapper.Process_{worker_id}')

        self._env = UnityEnvironment(file_name=file_name,
                                     worker_id=worker_id,
                                     base_port=base_port if file_name else None,
                                     no_graphics=no_graphics and train_mode,
                                     seed=seed,
                                     additional_args=['--scene', scene] + additional_args,
                                     side_channels=[self.engine_configuration_channel,
                                                    self.environment_parameters_channel,
                                                    self.option_channel])

        if time_scale is None:
            time_scale = 20 if train_mode else 1
        self.engine_configuration_channel.set_configuration_parameters(
            width=200 if train_mode else 1280,
            height=200 if train_mode else 720,
            quality_level=2,
            # 0: URP-Performant-Renderer
            # 1: URP-Balanced-Renderer
            # 2: URP-HighFidelity-Renderer
            time_scale=time_scale)

        self._env.reset()
        self.behavior_names: List[str] = list(self._env.behavior_specs)

        self.ma_n_agents = {n: 0 for n in self.behavior_names}
        for n in self.behavior_names:
            decision_steps, terminal_steps = self._env.get_steps(n)

            n_agents = decision_steps.obs[0].shape[0]
            self.ma_n_agents[n] = n_agents
            self._logger.info(f'{n} Number of Agents: {n_agents}')

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
                        self._logger.warning('Receive CLOSE')
                        break
            except Exception as e:
                self._logger.error(e)
            finally:
                self.close()
                conn.close()

    def init(self):
        """
        Returns:
            observation shapes: tuple[(o1, ), (o2, ), (o3_1, o3_2, o3_3), ...]
            discrete action sizes: list[int], list of all action branches
            continuous action size: int
        """
        self.ma_obs_names: Dict[str, List[str]] = {}
        self.ma_padding_index: Dict[str, int] = {}  # The index of obs with name `_Padding`
        self.ma_obs_shapes: Dict[str, Tuple[int, ...]] = {}
        self.ma_d_action_sizes: Dict[str, List[int]] = {}
        self.ma_c_action_size: Dict[str, int] = {}

        self.ma_unity_obs_shapes: Dict[str, Tuple[int, ...]] = {}
        self.ma_unity_d_action_sizes: Dict[str, List[int]] = {}
        self.ma_unity_c_action_size: Dict[str, int] = {}

        for n in self.behavior_names:
            behavior_spec = self._env.behavior_specs[n]
            obs_names = [o.name for o in behavior_spec.observation_specs]
            if '_Padding' in obs_names:
                self.ma_padding_index[n] = obs_names.index('_Padding')
                self._logger.info(f'{n} Padding index: {self.ma_padding_index[n]}')
                del obs_names[self.ma_padding_index[n]]
            else:
                self.ma_padding_index[n] = -1
            self._logger.info(f'{n} Observation names: {obs_names}')
            self.ma_obs_names[n] = obs_names

            obs_shapes = [o.shape for o in behavior_spec.observation_specs]
            self.ma_unity_obs_shapes[n] = obs_shapes.copy()

            if self.ma_padding_index[n] != -1:
                del obs_shapes[self.ma_padding_index[n]]
            self._logger.info(f'{n} Observation shapes: {obs_shapes}')
            self.ma_obs_shapes[n] = obs_shapes

            self._empty_action = behavior_spec.action_spec.empty_action

            discrete_action_sizes = []
            if behavior_spec.action_spec.discrete_size > 0:
                for branch, branch_size in enumerate(behavior_spec.action_spec.discrete_branches):
                    discrete_action_sizes.append(branch_size)
                    self._logger.info(f"{n} Discrete action branch {branch} has {branch_size} different actions")

            continuous_action_size = behavior_spec.action_spec.continuous_size

            self._logger.info(f'{n} Continuous action size: {continuous_action_size}')

            self.ma_d_action_sizes[n] = discrete_action_sizes  # list[int]
            self.ma_c_action_size[n] = continuous_action_size  # int

            self.ma_unity_d_action_sizes[n] = discrete_action_sizes
            self.ma_unity_c_action_size[n] = continuous_action_size

            if self.group_aggregation:
                decision_steps, terminal_steps = self._env.get_steps(n)
                u_group_ids, u_group_id_counts = np.unique(decision_steps.group_id, return_counts=True)
                # u_group_id_counts: agent count of each group id
                max_u_group_id_count = u_group_id_counts.max()  # the max agent count in each group id

                self.ma_obs_shapes[n] = [(max_u_group_id_count, *obs_shape) for obs_shape in self.ma_obs_shapes[n]]
                self.ma_d_action_sizes[n] = self.ma_d_action_sizes[n] * max_u_group_id_count  # list[int]
                self.ma_c_action_size[n] = self.ma_c_action_size[n] * max_u_group_id_count  # int

                for branch, branch_size in enumerate(self.ma_d_action_sizes[n]):
                    self._logger.info(f"{n} Aggregated discrete action branch {branch} has {branch_size} different actions")
                self._logger.info(f'{n} Aggregated continuous action size: {self.ma_c_action_size[n]}')

            for o_name, o_shape in zip(obs_names, obs_shapes):
                if ('camera' in o_name.lower() or 'visual' in o_name.lower() or 'image' in o_name.lower()) \
                        and len(o_shape) >= 3:
                    self.engine_configuration_channel.set_configuration_parameters(quality_level=5)
                    break

        self._logger.info('Initialized')

        return (self.ma_obs_names,
                self.ma_obs_shapes,
                self.ma_d_action_sizes,
                self.ma_c_action_size,
                self.ma_n_agents)

    def reset(self, reset_config=None):
        """
        return:
            observations: list[(NAgents, o1), (NAgents, o2), (NAgents, o3_1, o3_2, o3_3)]
        """
        reset_config = {} if reset_config is None else reset_config
        for k, v in reset_config.items():
            self.environment_parameters_channel.set_float_parameter(k, float(v))

        self._env.reset()

        self._ma_agents_ids: Dict[str, np.ndarray] = {}
        self._ma_agent_id_to_index: Dict[int, int] = {}

        self._ma_group_ids: Dict[str, np.ndarray] = {}
        self._ma_u_group_ids: Dict[str, np.ndarray] = {}
        self._ma_u_group_id_counts: Dict[str, np.ndarray] = {}

        ma_obs_list = {}
        for n in self.behavior_names:
            decision_steps, terminal_steps = self._env.get_steps(n)
            self._ma_agents_ids[n] = decision_steps.agent_id
            self._ma_agent_id_to_index[n] = {
                a_id: a_idx
                for a_idx, a_id in enumerate(decision_steps.agent_id)
            }
            ma_obs_list[n] = [obs.astype(np.float32) for obs in decision_steps.obs]

            if self.ma_padding_index[n] != -1:
                del ma_obs_list[n][self.ma_padding_index[n]]

            if self.group_aggregation:
                self._ma_group_ids[n] = decision_steps.group_id
                u_group_ids, u_group_id_counts = np.unique(decision_steps.group_id, return_counts=True)
                self._ma_u_group_ids[n] = u_group_ids
                self._ma_u_group_id_counts[n] = u_group_id_counts

                for i, obs in enumerate(ma_obs_list[n]):
                    aggr_obs = np.zeros((len(u_group_ids), u_group_id_counts.max(), *obs.shape[1:]), dtype=obs.dtype)
                    for j, (group_id, group_id_count) in enumerate(zip(u_group_ids, u_group_id_counts)):
                        aggr_obs[j, :group_id_count] = obs[decision_steps.group_id == group_id]

                    ma_obs_list[n][i] = aggr_obs

        return ma_obs_list

    def step(self, ma_d_action, ma_c_action):
        """
        Args:
            d_action: (NAgents, discrete_action_summed_size), one hot like action
            c_action: (NAgents, continuous_action_size)

        Returns:
            observations: list[(NAgents, o1), (NAgents, o2), (NAgents, o3_1, o3_2, o3_3)]
            reward: (NAgents, )
            done: (NAgents, ), bool
            max_step: (NAgents, ), bool
            padding_mask: (NAgents, ), bool
        """

        if self.group_aggregation:
            for n in self.behavior_names:
                if self.ma_d_action_sizes[n]:
                    d_action = ma_d_action[n]
                    unity_d_action_summed_size = sum(self.ma_unity_d_action_sizes[n])
                    unity_d_action = np.zeros((len(self._ma_agents_ids[n]), unity_d_action_summed_size), dtype=ma_d_action[n].dtype)

                    for action, group_id, group_id_count in zip(d_action, self._ma_u_group_ids[n], self._ma_u_group_id_counts[n]):
                        unity_d_action[self._ma_group_ids[n] == group_id] = action.reshape(-1, unity_d_action_summed_size)[:group_id_count]

                    ma_d_action[n] = unity_d_action

                if self.ma_c_action_size[n]:
                    c_action = ma_c_action[n]
                    unity_c_action_size = self.ma_unity_c_action_size[n]
                    unity_c_action = np.zeros((len(self._ma_agents_ids[n]), unity_c_action_size), dtype=c_action.dtype)

                    for action, group_id, group_id_count in zip(c_action, self._ma_u_group_ids[n], self._ma_u_group_id_counts[n]):
                        unity_c_action[self._ma_group_ids[n] == group_id] = action.reshape(-1, unity_c_action_size)[:group_id_count]

                    ma_c_action[n] = unity_c_action

        ma_obs_list: Dict[str, List[np.ndarray]] = {}
        ma_reward: Dict[str, List[np.ndarray]] = {}
        ma_done: Dict[str, List[np.ndarray]] = {}
        ma_max_step: Dict[str, List[np.ndarray]] = {}
        ma_padding_mask: Dict[str, List[np.ndarray]] = {}

        for n in self.behavior_names:  # padding all data
            agents_len = len(self._ma_agents_ids[n])
            ma_obs_list[n] = [np.zeros((agents_len, *obs_shape), dtype=np.float32) for obs_shape in self.ma_unity_obs_shapes[n]]
            ma_reward[n] = np.zeros(agents_len, dtype=np.float32)
            ma_done[n] = np.zeros(agents_len, dtype=bool)
            ma_max_step[n] = np.zeros(agents_len, dtype=bool)
            ma_padding_mask[n] = np.zeros(agents_len, dtype=bool)

        visited_ma_agent_ids = {n: set() for n in self.behavior_names}
        ma_decision_steps_len = {n: 0 for n in self.behavior_names}

        for n in self.behavior_names:  # sending actions to the environment
            d_action = c_action = None

            if self.ma_unity_d_action_sizes[n]:
                d_action_list = np.split(ma_d_action[n], np.cumsum(self.ma_unity_d_action_sizes[n]), axis=-1)[:-1]
                d_action_list = [np.argmax(d_action, axis=-1) for d_action in d_action_list]
                d_action = np.stack(d_action_list, axis=-1)

            if self.ma_unity_c_action_size[n]:
                c_action = ma_c_action[n]

            self._env.set_actions(n,
                                  ActionTuple(continuous=c_action, discrete=d_action))

        self._env.step()

        for n in self.behavior_names:  # receving data from the environment
            decision_steps, terminal_steps = self._env.get_steps(n)

            agent_id_to_index = self._ma_agent_id_to_index[n]
            ma_decision_steps_len[n] = len(decision_steps)

            for agent_id in decision_steps:
                agent_idx = agent_id_to_index[agent_id]
                info = decision_steps[agent_id]
                for obs, _obs in zip(ma_obs_list[n], info.obs):
                    obs[agent_idx] = _obs
                ma_reward[n][agent_idx] = info.reward

                visited_ma_agent_ids[n].add(agent_id)

            for agent_id in terminal_steps:
                agent_idx = agent_id_to_index[agent_id]
                info = terminal_steps[agent_id]
                for obs, _obs in zip(ma_obs_list[n], info.obs):
                    obs[agent_idx] = _obs
                ma_reward[n][agent_idx] = info.reward
                ma_done[n][agent_idx] = True
                ma_max_step[n][agent_idx] = info.interrupted

                visited_ma_agent_ids[n].add(agent_id)

        while any([ma_decision_steps_len[n] != len(self._ma_agents_ids[n]) for n in self.behavior_names]):
            for n in self.behavior_names:
                self._env.set_actions(n, self._empty_action(ma_decision_steps_len[n]))

            self._env.step()

            for n in self.behavior_names:
                decision_steps, terminal_steps = self._env.get_steps(n)

                agent_id_to_index = self._ma_agent_id_to_index[n]
                ma_decision_steps_len[n] = len(decision_steps)

                for agent_id in decision_steps:
                    agent_idx = agent_id_to_index[agent_id]
                    info = decision_steps[agent_id]
                    for obs, _obs in zip(ma_obs_list[n], info.obs):
                        obs[agent_idx] = _obs

                    if agent_id not in visited_ma_agent_ids[n]:
                        ma_reward[n][agent_idx] = info.reward

                    visited_ma_agent_ids[n].add(agent_id)

                for agent_id in terminal_steps:
                    agent_idx = agent_id_to_index[agent_id]
                    info = terminal_steps[agent_id]
                    for obs, _obs in zip(ma_obs_list[n], info.obs):
                        obs[agent_idx] = _obs
                    ma_reward[n][agent_idx] = info.reward
                    ma_done[n][agent_idx] = True
                    ma_max_step[n][agent_idx] = info.interrupted

                    visited_ma_agent_ids[n].add(agent_id)

        for n in self.behavior_names:
            if self.ma_padding_index[n] != -1:
                mask = ma_obs_list[n][self.ma_padding_index[n]][:, 0] == 1
                ma_padding_mask[n][mask] = True

                del ma_obs_list[n][self.ma_padding_index[n]]

        for n in self.behavior_names:
            done = ma_done[n]
            max_step = ma_max_step[n]

        if self.group_aggregation:
            for n in self.behavior_names:
                group_ids = self._ma_group_ids[n]
                u_group_ids = self._ma_u_group_ids[n]
                u_group_id_counts = self._ma_u_group_id_counts[n]

                for i, obs in enumerate(ma_obs_list[n]):
                    aggr_obs = np.zeros((len(u_group_ids), u_group_id_counts.max(), *obs.shape[1:]), dtype=obs.dtype)
                    for j, (group_id, group_id_count) in enumerate(zip(u_group_ids, u_group_id_counts)):
                        aggr_obs[j, :group_id_count] = obs[group_ids == group_id]

                    ma_obs_list[n][i] = aggr_obs

                reward = ma_reward[n]
                done = ma_done[n]
                max_step = ma_max_step[n]
                padding_mask = ma_padding_mask[n]

                aggr_reward = np.zeros(len(u_group_ids), dtype=reward.dtype)
                aggr_done = np.zeros(len(u_group_ids), dtype=done.dtype)
                aggr_max_step = np.zeros(len(u_group_ids), dtype=max_step.dtype)
                aggr_padding_mask = np.zeros(len(u_group_ids), dtype=padding_mask.dtype)

                for j, (group_id, group_id_count) in enumerate(zip(u_group_ids, u_group_id_counts)):
                    aggr_reward[j] = reward[group_ids == group_id].sum()
                    # If all agents in the group is done, then the group is done
                    aggr_done[j] = done[group_ids == group_id].all()
                    # If all agents in the group reaches the max step, then the group reaches the max step
                    aggr_max_step[j] = max_step[group_ids == group_id].all()
                    aggr_padding_mask[j] = padding_mask[group_ids == group_id].all()

                ma_reward[n] = aggr_reward
                ma_done[n] = aggr_done
                ma_max_step[n] = aggr_max_step
                ma_padding_mask[n] = aggr_padding_mask

        return ma_obs_list, ma_reward, ma_done, ma_max_step, ma_padding_mask

    def close(self):
        self._env.close()
        self._logger.warning(f'Environment closed')


class UnityWrapper(EnvWrapper):
    def __init__(self,
                 train_mode: bool = True,
                 env_name: str = None,
                 n_envs: int = 1,

                 base_port=5005,
                 no_graphics=True,
                 time_scale=None,
                 seed=None,
                 scene=None,
                 additional_args=None,
                 group_aggregation=False,
                 force_seq=None):
        """
        Args:
            train_mode: If in train mode, Unity will run in the highest quality
            env_name: The executable path. The UnityEnvironment will run in editor if None
            n_envs: The env copies count

            base_port: The port that communicate to Unity. It will be set to 5004 automatically if in editor.
            no_graphics: If Unity runs in no graphic mode. It must be set to False if Unity has camera sensor.
            time_scale: Time scale of Unity. If None: time_scale = 20 if train_mode else 1
            seed: Random seed
            scene: The scene name
            group_aggregation: If aggregate group agents
        """
        super().__init__(train_mode, env_name, None, n_envs)
        self.base_port = base_port
        self.no_graphics = no_graphics
        self.time_scale = time_scale
        self.seed = seed
        self.scene = scene
        self.additional_args = additional_args
        self.group_aggregation = group_aggregation

        self._logger = logging.getLogger('UnityWrapper')

        # force_seq: Whether forcing use multiple processes
        # self._seq_envs: Whether use multiple processes
        if force_seq is None:
            self._seq_envs: bool = self.n_envs <= MAX_N_ENVS_PER_PROCESS
        else:
            self._seq_envs: bool = force_seq

        self._process_id = 0

        self.env_length = math.ceil(self.n_envs / MAX_N_ENVS_PER_PROCESS)

        if self._seq_envs:
            self._logger.info('Using sequential environments')

            # All environments are executed sequentially
            self._envs: List[UnityWrapperProcess] = []

            for i in range(self.env_length):
                self._envs.append(UnityWrapperProcess(conn=None,
                                                      train_mode=train_mode,
                                                      file_name=env_name,
                                                      worker_id=i,
                                                      base_port=base_port,
                                                      no_graphics=no_graphics,
                                                      time_scale=time_scale,
                                                      seed=seed,
                                                      scene=scene,
                                                      additional_args=additional_args,
                                                      n_envs=min(MAX_N_ENVS_PER_PROCESS, self.n_envs - i * MAX_N_ENVS_PER_PROCESS),
                                                      group_aggregation=group_aggregation))
        else:
            self._logger.info('Using multi-processing environments')

            # All environments are executed in parallel
            self._conns: List[multiprocessing.connection.Connection] = [None] * self.env_length
            self._processes: List[multiprocessing.Process] = [None] * self.env_length

            self._generate_processes()

        self._logger.info('Environments loaded')

    def _process_conn_receiving(self, conn: multiprocessing.connection.Connection, i: int):
        if not conn.poll(60):
            self._logger.error(f'Environment {i} timeout')
            raise TimeoutError()

        return conn.recv()

    def _generate_processes(self, force_init=False):
        if self._seq_envs:
            return

        while None in self._conns:
            i = self._conns.index(None)

            self._logger.info(f'Starting environment {i} ...')
            parent_conn, child_conn = multiprocessing.Pipe()
            self._conns[i] = parent_conn
            p = multiprocessing.Process(target=UnityWrapperProcess,
                                        args=(child_conn,
                                              self.train_mode,
                                              self.env_name,
                                              self._process_id,
                                              self.base_port,
                                              self.no_graphics,
                                              self.time_scale,
                                              self.seed,
                                              self.scene,
                                              self.additional_args,
                                              min(MAX_N_ENVS_PER_PROCESS, self.n_envs - i * MAX_N_ENVS_PER_PROCESS),
                                              self.group_aggregation),
                                        daemon=True)
            p.start()
            self._processes[i] = p

            if force_init:
                parent_conn.send((INIT, None))
                self._process_conn_receiving(parent_conn, i)

            self._logger.info(f'Environment {i} started with process {self._process_id}')
            self._process_id += 1

    def send_option(self, option: Dict[str, int]):
        # TODO multiple envs
        self._envs[0].option_channel.send_option(option)

    def init(self):
        self.ma_n_agents_list = defaultdict(list)

        if self._seq_envs:
            for env in self._envs:
                results = env.init()
                ma_obs_names, ma_obs_shapes, ma_d_action_sizes, ma_c_action_size, ma_n_agents = results
                for n, n_agents in ma_n_agents.items():
                    self.ma_n_agents_list[n].append(n_agents)
        else:
            for i, conn in enumerate(self._conns):
                conn.send((INIT, None))
                results = self._process_conn_receiving(conn, i)
                ma_obs_names, ma_obs_shapes, ma_d_action_sizes, ma_c_action_size, ma_n_agents = results
                for n, n_agents in ma_n_agents.items():
                    self.ma_n_agents_list[n].append(n_agents)

        self.behavior_names = list(ma_obs_shapes.keys())
        for n, n_agents_list in self.ma_n_agents_list.items():
            for i in range(1, len(n_agents_list)):
                n_agents_list[i] += n_agents_list[i - 1]
            n_agents_list.insert(0, 0)

        return ma_obs_names, ma_obs_shapes, ma_d_action_sizes, ma_c_action_size

    def reset(self, reset_config=None):
        if self._seq_envs:
            envs_ma_obs_list = [env.reset(reset_config) for env in self._envs]
        else:
            for conn in self._conns:
                conn.send((RESET, reset_config))

            envs_ma_obs_list = [self._process_conn_receiving(conn, i) for i, conn in enumerate(self._conns)]

        ma_obs_list = {n: [np.concatenate(env_obs_list)
                           for env_obs_list in zip(*[ma_obs_list[n] for ma_obs_list in envs_ma_obs_list])]
                       for n in self.behavior_names}

        return ma_obs_list

    def step(self,
             ma_d_action: Dict[str, np.ndarray],
             ma_c_action: Dict[str, np.ndarray]):
        ma_envs_obs_list = {n: [] for n in self.behavior_names}
        ma_envs_reward = {n: [] for n in self.behavior_names}
        ma_envs_done = {n: [] for n in self.behavior_names}
        ma_envs_max_step = {n: [] for n in self.behavior_names}
        ma_envs_padding_mask = {n: [] for n in self.behavior_names}

        for i in range(self.env_length):
            tmp_ma_d_actions = {}
            tmp_ma_c_actions = {}

            for n in self.behavior_names:
                d_action = ma_d_action[n]
                c_action = ma_c_action[n]
                n_agents_list = self.ma_n_agents_list[n]

                if d_action is not None:
                    tmp_ma_d_actions[n] = d_action[n_agents_list[i]:n_agents_list[i + 1]]
                if c_action is not None:
                    tmp_ma_c_actions[n] = c_action[n_agents_list[i]:n_agents_list[i + 1]]

            if self._seq_envs:
                (ma_obs_list,
                 ma_reward,
                 ma_done,
                 ma_max_step,
                 ma_padding_mask) = self._envs[i].step(tmp_ma_d_actions, tmp_ma_c_actions)

                for n in self.behavior_names:
                    ma_envs_obs_list[n].append(ma_obs_list[n])
                    ma_envs_reward[n].append(ma_reward[n])
                    ma_envs_done[n].append(ma_done[n])
                    ma_envs_max_step[n].append(ma_max_step[n])
                    ma_envs_padding_mask[n].append(ma_padding_mask[n])
            else:
                self._conns[i].send((STEP, (tmp_ma_d_actions, tmp_ma_c_actions)))

        if not self._seq_envs:
            failed_count = 0

            for i, conn in enumerate(self._conns):
                try:
                    (ma_obs_list,
                     ma_reward,
                     ma_done,
                     ma_max_step,
                     ma_padding_mask) = self._process_conn_receiving(conn, i)

                    for n in self.behavior_names:
                        ma_envs_obs_list[n].append(ma_obs_list[n])
                        ma_envs_reward[n].append(ma_reward[n])
                        ma_envs_done[n].append(ma_done[n])
                        ma_envs_max_step[n].append(ma_max_step[n])
                        ma_envs_padding_mask[n].append(ma_padding_mask[n])
                except Exception as e:
                    self._logger.error(f'Environment {i} error, {e.__class__.__name__} {e}')

                    self._processes[i].terminate()
                    while self._processes[i].is_alive():
                        self._logger.warning(f'Environment {i} still running...')
                        time.sleep(1)
                    self._logger.warning(f'Environment {i} terminated by error')

                    self._processes[i] = None
                    self._conns[i].close()
                    self._conns[i] = None
                    failed_count += 1

            if failed_count != 0:
                self._logger.warning(f'{failed_count} environments failed. Restarting...')
                self._generate_processes(force_init=True)
                self._logger.warning(f'{failed_count} environments restarted')

                return None, None, None, None, None

        ma_obs_list = {n: [np.concatenate(obs) for obs in zip(*ma_envs_obs_list[n])] for n in self.behavior_names}
        ma_reward = {n: np.concatenate(ma_envs_reward[n]) for n in self.behavior_names}
        ma_done = {n: np.concatenate(ma_envs_done[n]) for n in self.behavior_names}
        ma_max_step = {n: np.concatenate(ma_envs_max_step[n]) for n in self.behavior_names}
        ma_padding_mask = {n: np.concatenate(ma_envs_padding_mask[n]) for n in self.behavior_names}

        return ma_obs_list, ma_reward, ma_done, ma_max_step, ma_padding_mask

    def close(self):
        self._logger.warning('Closing environments')
        if self._seq_envs:
            for env in self._envs:
                env.close()
        else:
            for conn in self._conns:
                try:
                    conn.send((CLOSE, None))
                except:
                    pass
            for p in self._processes:
                try:
                    p.terminate()
                except:
                    pass
        self._logger.warning('Environments closed')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    N_ENVS = 2
    GROUP_AGGREGATION = True

    env = UnityWrapper(train_mode=True,
                       #    file_name=r'C:\Users\fisher\Documents\Unity\win-RL-Envs\RLEnvironments.exe',
                       #    scene='Roller',
                       n_envs=N_ENVS,
                       group_aggregation=GROUP_AGGREGATION)
    ma_obs_names, ma_obs_shapes, ma_d_action_sizes, ma_c_action_size = env.init()
    ma_names = list(ma_obs_shapes.keys())

    for i in range(100):
        ma_obs_list = env.reset()

        for j in range(100):
            ma_d_action = {}
            ma_c_action = {}
            for n in ma_names:
                d_action, c_action = None, None
                if ma_d_action_sizes[n]:
                    d_action_sizes = ma_d_action_sizes[n]
                    d_action_list = [np.random.randint(0, d_action_size, size=N_ENVS)
                                     for d_action_size in d_action_sizes]
                    d_action_list = [np.eye(d_action_size, dtype=np.int32)[d_action]
                                     for d_action, d_action_size in zip(d_action_list, d_action_sizes)]
                    d_action = np.concatenate(d_action_list, axis=-1)
                if ma_c_action_size[n]:
                    c_action = np.random.randn(N_ENVS, ma_c_action_size[n])

            ma_d_action[n] = d_action
            ma_c_action[n] = c_action

            ma_obs_list, ma_reward, ma_done, ma_max_step, ma_padding_mask = env.step(ma_d_action, ma_c_action)

    env.close()

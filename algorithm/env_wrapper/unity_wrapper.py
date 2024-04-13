import json
import logging
import math
import multiprocessing
import multiprocessing.connection
import os
import random
import time
import uuid
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from mlagents_envs.environment import ActionTuple, UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import \
    EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import \
    EnvironmentParametersChannel
from mlagents_envs.side_channel.side_channel import (IncomingMessage,
                                                     OutgoingMessage,
                                                     SideChannel)

from .env_wrapper import DecisionStep, EnvWrapper, TerminalStep

INIT = 0
RESET = 1
STEP = 2
CLOSE = 3
RESET_ENVS_DONE = 4

MAX_N_ENVS = 100


class EnvDoneChannel(SideChannel):
    def __init__(self, on_done: Callable[[], None]) -> None:
        super().__init__(uuid.UUID("823f1d9f-dcdf-433b-81ba-ac0512c649e4"))
        self.on_done = on_done

    def on_message_received(self, msg: IncomingMessage) -> None:
        if msg.read_bool(default_value=False):
            self.on_done()


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
    env_done = False  # Whether env process is done from the signal by Unity

    _ma_group_ids: Dict[str, np.ndarray] = {}  # the group id corresponding to each agent
    _ma_u_group_ids: Dict[str, np.ndarray] = {}  # all unique group ids
    _ma_u_group_id_counts: Dict[str, np.ndarray] = {}  # the agent count in each unique group

    def __init__(self,
                 conn: Optional[multiprocessing.connection.Connection] = None,
                 train_mode: bool = True,
                 file_name: Optional[str] = None,
                 worker_id: int = 0,
                 base_port: int = 5005,
                 no_graphics: bool = True,
                 force_vulkan: bool = False,
                 time_scale: Optional[float] = None,
                 seed: Optional[int] = None,
                 scene: Optional[str] = None,
                 env_args: Dict = None,
                 n_envs: int = 1):
        """
        Args:
            conn: Connection if run in multiprocessing mode
            train_mode: If in train mode, Unity will speed up
            file_name: The executable path. The UnityEnvironment will run in editor if None
            worker_id: Offset from base_port
            base_port: The port that communicate to Unity. It will be set to 5004 automatically if in editor.
            no_graphics: If Unity runs in no graphic mode. It must be set to False if Unity has camera sensor.
            force_vulkan: -force-vulkan
            time_scale: Time scale of Unity. If None: time_scale = 20 if train_mode else 1
            seed: Random seed
            scene: The scene name
            env_args: Additional environment arguments when initializing
            n_envs: The env copies count
        """
        self.scene = scene
        self.n_envs = n_envs

        seed = seed if seed is not None else random.randint(0, 65536)
        if env_args is None:
            env_args = {}

        self.engine_configuration_channel = EngineConfigurationChannel()
        self.environment_parameters_channel = EnvironmentParametersChannel()

        self.environment_parameters_channel.set_float_parameter('env_copys', float(n_envs))
        for k, v in env_args.items():
            self.environment_parameters_channel.set_float_parameter(k, float(v))

        def on_env_done():
            if not self.env_done:
                self.env_done = True
        self.env_done_channel = EnvDoneChannel(on_env_done)
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

        additional_args = ['--scene', scene]
        if force_vulkan:
            additional_args.append('-force-vulkan')

        self._env = UnityEnvironment(file_name=file_name,
                                     worker_id=worker_id,
                                     base_port=base_port if file_name else None,
                                     no_graphics=no_graphics and train_mode,
                                     seed=seed,
                                     additional_args=additional_args,
                                     side_channels=[self.engine_configuration_channel,
                                                    self.environment_parameters_channel,
                                                    self.env_done_channel,
                                                    self.option_channel])

        if time_scale is None:
            time_scale = 20 if train_mode else 1
        self.engine_configuration_channel.set_configuration_parameters(
            width=200 if train_mode else 1280,
            height=200 if train_mode else 720,
            quality_level=0,
            # 0: URP-Performant-Renderer
            # 1: URP-Balanced-Renderer
            # 2: URP-HighFidelity-Renderer
            time_scale=time_scale)

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
                        self._logger.warning('Received CLOSE')
                        break
                    elif cmd == RESET_ENVS_DONE:
                        self.env_done = False
            except Exception as e:
                self._logger.error(e)
            finally:
                self.close()
                conn.close()

    def init(self):
        """
        Returns:
            observation names: list[str]
            observation shapes: list[(o1, ), (o2, ), (o3_1, o3_2, o3_3), ...]
            discrete action sizes: list[int], list of all action branches
            continuous action size: int
        """
        self.ma_obs_names: Dict[str, List[str]] = {}
        self.ma_obs_shapes: Dict[str, Tuple[int, ...]] = {}
        self.ma_d_action_sizes: Dict[str, List[int]] = {}
        self.ma_c_action_size: Dict[str, int] = {}

        self._env.reset()
        self.behavior_names: List[str] = list(self._env.behavior_specs)

        for n in self.behavior_names:
            behavior_spec = self._env.behavior_specs[n]
            obs_names = [o.name for o in behavior_spec.observation_specs]
            self._logger.info(f'{n} Observation names: {obs_names}')
            self.ma_obs_names[n] = obs_names

            obs_shapes = [o.shape for o in behavior_spec.observation_specs]

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

            for o_name, o_shape in zip(obs_names, obs_shapes):
                if ('camera' in o_name.lower() or 'visual' in o_name.lower() or 'image' in o_name.lower()) \
                        and len(o_shape) >= 3:
                    self.engine_configuration_channel.set_configuration_parameters(quality_level=2)
                    break

        self._logger.info('Initialized')

        return (self.ma_obs_names,
                self.ma_obs_shapes,
                self.ma_d_action_sizes,
                self.ma_c_action_size)

    def reset(self, reset_config=None):
        """
        return:
            ma_agent_ids: dict[str, (NAgents, )]
            ma_obs_list: dict[str, list[(NAgents, o1), (NAgents, o2), (NAgents, o3_1, o3_2, o3_3)]]
        """
        reset_config = {} if reset_config is None else reset_config
        for k, v in reset_config.items():
            self.environment_parameters_channel.set_float_parameter(k, float(v))

        self._env.reset()

        ma_agent_ids = {}
        ma_obs_list = {}
        for n in self.behavior_names:
            decision_steps, terminal_steps = self._env.get_steps(n)
            ma_agent_ids[n] = decision_steps.agent_id
            ma_obs_list[n] = decision_steps.obs

        return ma_agent_ids, ma_obs_list

    def step(self, ma_d_action, ma_c_action):
        """
        Args:
            ma_d_action: dict[str, (NAgents, discrete_action_size)], one hot like action
            ma_c_action: dict[str, (NAgents, continuous_action_size)]

        Returns:
            DecisionStep: decision_ma_agent_ids
                          decision_ma_obs_list
                          decision_ma_last_reward
            TerminalStep: terminal_ma_agent_ids
                          terminal_ma_obs_list
                          terminal_ma_last_reward
                          terminal_ma_max_reached
        """
        for n in self.behavior_names:  # sending actions to the environment
            d_action = c_action = None

            if self.ma_d_action_sizes[n] and n in ma_d_action:
                d_action_list = np.split(ma_d_action[n], np.cumsum(self.ma_d_action_sizes[n]), axis=-1)[:-1]
                d_action_list = [np.argmax(d_action, axis=-1) for d_action in d_action_list]
                d_action = np.stack(d_action_list, axis=-1)

            if self.ma_c_action_size[n] and n in ma_c_action:
                c_action = ma_c_action[n]

            if d_action is not None or c_action is not None:
                self._env.set_actions(n,
                                      ActionTuple(continuous=c_action, discrete=d_action))

        self._env.step()

        decision_ma_agent_ids: Dict[str, np.ndarray] = {}
        decision_ma_obs_list: Dict[str, List[np.ndarray]] = {}
        decision_ma_last_reward: Dict[str, np.ndarray] = {}

        terminal_ma_agent_ids: Dict[str, List[int]] = {}
        terminal_ma_obs_list: Dict[str, List[np.ndarray]] = {}
        terminal_ma_last_reward: Dict[str, np.ndarray] = {}
        terminal_ma_max_reached: Dict[str, np.ndarray] = {}

        for n in self.behavior_names:  # receving data from the environment
            decision_steps, terminal_steps = self._env.get_steps(n)

            decision_ma_agent_ids[n] = decision_steps.agent_id
            decision_ma_obs_list[n] = decision_steps.obs
            decision_ma_last_reward[n] = decision_steps.reward

            terminal_ma_agent_ids[n] = terminal_steps.agent_id
            terminal_ma_obs_list[n] = terminal_steps.obs
            terminal_ma_last_reward[n] = terminal_steps.reward
            terminal_ma_max_reached[n] = terminal_steps.interrupted

        return (
            decision_ma_agent_ids,
            decision_ma_obs_list,
            decision_ma_last_reward
        ), (
            terminal_ma_agent_ids,
            terminal_ma_obs_list,
            terminal_ma_last_reward,
            terminal_ma_max_reached
        ), self.env_done

    def reset_env_done(self):
        self.env_done = False

    def close(self):
        self._env.close()
        self._logger.warning(f'Environment closed')


class UnityWrapper(EnvWrapper):
    def __init__(self,
                 train_mode: bool = True,
                 env_name: str = None,
                 n_envs: int = 1,

                 base_port: int = 5005,
                 max_n_envs_per_process: int = 10,
                 no_graphics: bool = True,
                 force_vulkan: bool = False,
                 time_scale: Optional[float] = None,
                 seed: Optional[int] = None,
                 scene: Optional[str] = None,
                 env_args: Optional[Union[List[str], Dict]] = None,
                 force_seq: Optional[bool] = None):
        """
        Args:
            train_mode: If in train mode, Unity will run in the highest quality
            env_name: The executable path. The UnityEnvironment will run in editor if None
            n_envs: The env copies count

            base_port: The port that communicate to Unity. It will be set to 5004 automatically if in editor.
            max_n_envs_per_process: The max env copies count in each process
            no_graphics: If Unity runs in no graphic mode. It must be set to False if Unity has camera sensor.
            force_vulkan: -force-vulkan
            time_scale: Time scale of Unity. If None: time_scale = 20 if train_mode else 1
            seed: Random seed
            scene: The scene name
        """
        super().__init__(train_mode, env_name, env_args, n_envs)
        self.base_port = base_port
        self.max_n_envs_per_process = max_n_envs_per_process
        self.no_graphics = no_graphics
        self.force_vulkan = force_vulkan
        self.time_scale = time_scale
        self.seed = seed
        self.scene = scene

        self._logger = logging.getLogger('UnityWrapper')

        # force_seq: Whether forcing use multiple processes
        # self._seq_processes: Whether using multiple processes
        if force_seq is None:
            self._seq_processes: bool = self.n_envs <= self.max_n_envs_per_process
        else:
            self._seq_processes: bool = force_seq

        self._process_id = 0

        # The size of Unity Environment Processes
        self._env_processes_size = math.ceil(self.n_envs / self.max_n_envs_per_process)
        # The list of whether process done
        self._all_env_processes_done = [False] * self._env_processes_size

        if self._seq_processes:
            self._logger.info('Using sequential environments')

            # All environments are executed sequentially
            self._env_processes: List[UnityWrapperProcess] = []

            for i in range(self._env_processes_size):
                self._env_processes.append(UnityWrapperProcess(conn=None,
                                                               train_mode=self.train_mode,
                                                               file_name=self.env_name,
                                                               worker_id=i,
                                                               base_port=base_port,
                                                               no_graphics=no_graphics,
                                                               force_vulkan=force_vulkan,
                                                               time_scale=time_scale,
                                                               seed=seed,
                                                               scene=scene,
                                                               env_args=self.env_args,
                                                               n_envs=min(self.max_n_envs_per_process, self.n_envs - i * self.max_n_envs_per_process)))
        else:
            self._logger.info('Using multi-processing environments')

            # All environments are executed in parallel
            self._conns: List[multiprocessing.connection.Connection] = [None] * self._env_processes_size
            self._processes: List[multiprocessing.Process] = [None] * self._env_processes_size

            self._generate_processes()

        self._logger.info('Environments loaded')

    def _process_conn_receiving(self, conn: multiprocessing.connection.Connection, i: int):
        if not conn.poll(60):
            self._logger.error(f'Environment {i} timeout')
            raise TimeoutError()

        return conn.recv()

    def _generate_processes(self, force_init=False):
        if self._seq_processes:
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
                                              self.force_vulkan,
                                              self.time_scale,
                                              self.seed,
                                              self.scene,
                                              self.env_args,
                                              min(self.max_n_envs_per_process, self.n_envs - i * self.max_n_envs_per_process)),
                                        daemon=True)
            p.start()
            self._processes[i] = p

            if force_init:
                parent_conn.send((INIT, None))
                self._process_conn_receiving(parent_conn, i)
                parent_conn.send((RESET, None))
                self._process_conn_receiving(parent_conn, i)

            self._logger.info(f'Environment {i} started with process {self._process_id}')
            self._process_id += 1

    def send_option(self, option: Dict[str, int]):
        # TODO multiple envs
        self._env_processes[0].option_channel.send_option(option)

    def init(self):
        if self._seq_processes:
            for env in self._env_processes:
                results = env.init()
                ma_obs_names, ma_obs_shapes, ma_d_action_sizes, ma_c_action_size = results
        else:
            for i, conn in enumerate(self._conns):
                conn.send((INIT, None))
                results = self._process_conn_receiving(conn, i)
                ma_obs_names, ma_obs_shapes, ma_d_action_sizes, ma_c_action_size = results

        self.behavior_names = list(ma_obs_names.keys())

        return ma_obs_names, ma_obs_shapes, ma_d_action_sizes, ma_c_action_size

    def _cumulate_ma_n_agents_list(self,
                                   ma_envs_agent_ids: Dict[str, List[int]]):
        """
        Indicating the cumulative agents counts of each env process in the next decision step
        """
        self.ma_cum_n_agents_list = {n: [] for n in self.behavior_names}
        envs = range(self._env_processes_size) if self._seq_processes else self._conns
        for i in range(len(envs)):
            for n in self.behavior_names:
                self.ma_cum_n_agents_list[n].append(len(ma_envs_agent_ids[n][i]))

        for n, n_agents_list in self.ma_cum_n_agents_list.items():
            for i in range(1, len(n_agents_list)):
                n_agents_list[i] += n_agents_list[i - 1]
            n_agents_list.insert(0, 0)

    def _map_agent_ids(self, agent_ids: np.ndarray, i: int) -> np.ndarray:
        return agent_ids * MAX_N_ENVS + i

    def reset(self, reset_config=None) -> Tuple[Dict[str, List[int]],
                                                Dict[str, List[np.ndarray]]]:
        ma_envs_agent_ids = {n: [] for n in self.behavior_names}
        ma_envs_obs_list = {n: [] for n in self.behavior_names}

        if self._seq_processes:
            for i in range(self._env_processes_size):
                ma_agent_ids, ma_obs_list = self._env_processes[i].reset(reset_config)
                for n in self.behavior_names:
                    ma_envs_agent_ids[n].append(self._map_agent_ids(ma_agent_ids[n], i))
                    ma_envs_obs_list[n].append(ma_obs_list[n])
        else:
            for conn in self._conns:
                conn.send((RESET, reset_config))

            for i, conn in enumerate(self._conns):
                ma_agent_ids, ma_obs_list = self._process_conn_receiving(conn, i)
                for n in self.behavior_names:
                    ma_envs_agent_ids[n].append(self._map_agent_ids(ma_agent_ids[n], i))
                    ma_envs_obs_list[n].append(ma_obs_list[n])

        self._cumulate_ma_n_agents_list(ma_envs_agent_ids)
        ma_agent_ids = {n: np.concatenate(ma_envs_agent_ids[n]) for n in self.behavior_names}
        ma_obs_list = {n: [np.concatenate(obs) for obs in zip(*ma_envs_obs_list[n])] for n in self.behavior_names}

        return ma_agent_ids, ma_obs_list

    def step(self,
             ma_d_action: Dict[str, np.ndarray],
             ma_c_action: Dict[str, np.ndarray]) -> Tuple[DecisionStep, TerminalStep]:

        decision_ma_envs_agent_ids = {n: [] for n in self.behavior_names}
        decision_ma_envs_obs_list = {n: [] for n in self.behavior_names}
        decision_ma_envs_last_reward = {n: [] for n in self.behavior_names}

        terminal_ma_envs_agent_ids = {n: [] for n in self.behavior_names}
        terminal_ma_envs_obs_list = {n: [] for n in self.behavior_names}
        terminal_ma_envs_last_reward = {n: [] for n in self.behavior_names}
        terminal_ma_envs_max_reached = {n: [] for n in self.behavior_names}

        for i in range(self._env_processes_size):
            tmp_ma_d_actions = {}
            tmp_ma_c_actions = {}

            for n in self.behavior_names:
                d_action = c_action = None
                if n in ma_d_action:
                    d_action = ma_d_action[n]
                if n in ma_c_action:
                    c_action = ma_c_action[n]
                cum_n_agents_list = self.ma_cum_n_agents_list[n]

                if d_action is not None:
                    tmp_ma_d_actions[n] = d_action[cum_n_agents_list[i]:cum_n_agents_list[i + 1]]
                if c_action is not None:
                    tmp_ma_c_actions[n] = c_action[cum_n_agents_list[i]:cum_n_agents_list[i + 1]]

            if self._seq_processes:
                (
                    decision_ma_agent_ids,
                    decision_ma_obs_list,
                    decision_ma_last_reward
                ), (
                    terminal_ma_agent_ids,
                    terminal_ma_obs_list,
                    terminal_ma_last_reward,
                    terminal_ma_max_reached
                ), envs_done = self._env_processes[i].step(tmp_ma_d_actions, tmp_ma_c_actions)

                for n in self.behavior_names:
                    decision_ma_envs_agent_ids[n].append(self._map_agent_ids(decision_ma_agent_ids[n], i))
                    decision_ma_envs_obs_list[n].append(decision_ma_obs_list[n])
                    decision_ma_envs_last_reward[n].append(decision_ma_last_reward[n])

                    terminal_ma_envs_agent_ids[n].append(self._map_agent_ids(terminal_ma_agent_ids[n], i))
                    terminal_ma_envs_obs_list[n].append(terminal_ma_obs_list[n])
                    terminal_ma_envs_last_reward[n].append(terminal_ma_last_reward[n])
                    terminal_ma_envs_max_reached[n].append(terminal_ma_max_reached[n])

                self._all_env_processes_done[i] = envs_done
            else:
                self._conns[i].send((STEP, (tmp_ma_d_actions, tmp_ma_c_actions)))

        if self._seq_processes:
            for i, env in enumerate(self._env_processes):
                self._all_env_processes_done[i] = env.env_done
        else:
            failed_count = 0

            for i, conn in enumerate(self._conns):
                try:
                    (
                        decision_ma_agent_ids,
                        decision_ma_obs_list,
                        decision_ma_last_reward
                    ), (
                        terminal_ma_agent_ids,
                        terminal_ma_obs_list,
                        terminal_ma_last_reward,
                        terminal_ma_max_reached
                    ), envs_done = self._process_conn_receiving(conn, i)

                    for n in self.behavior_names:
                        decision_ma_envs_agent_ids[n].append(self._map_agent_ids(decision_ma_agent_ids[n], i))
                        decision_ma_envs_obs_list[n].append(decision_ma_obs_list[n])
                        decision_ma_envs_last_reward[n].append(decision_ma_last_reward[n])

                        terminal_ma_envs_agent_ids[n].append(self._map_agent_ids(terminal_ma_agent_ids[n], i))
                        terminal_ma_envs_obs_list[n].append(terminal_ma_obs_list[n])
                        terminal_ma_envs_last_reward[n].append(terminal_ma_last_reward[n])
                        terminal_ma_envs_max_reached[n].append(terminal_ma_max_reached[n])

                    self._all_env_processes_done[i] = envs_done

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

                return None, None, None

        decision_ma_agent_ids = {n: np.concatenate(decision_ma_envs_agent_ids[n]) for n in self.behavior_names}
        decision_ma_obs_list = {n: [np.concatenate(obs) for obs in zip(*decision_ma_envs_obs_list[n])] for n in self.behavior_names}
        decision_ma_last_reward = {n: np.concatenate(decision_ma_envs_last_reward[n]) for n in self.behavior_names}

        terminal_ma_agent_ids = {n: np.concatenate(terminal_ma_envs_agent_ids[n]) for n in self.behavior_names}
        terminal_ma_obs_list = {n: [np.concatenate(obs) for obs in zip(*terminal_ma_envs_obs_list[n])] for n in self.behavior_names}
        terminal_ma_last_reward = {n: np.concatenate(terminal_ma_envs_last_reward[n]) for n in self.behavior_names}
        terminal_ma_max_reached = {n: np.concatenate(terminal_ma_envs_max_reached[n]) for n in self.behavior_names}

        all_envs_done = all(self._all_env_processes_done)
        if all_envs_done:
            if self._seq_processes:
                for env in self._env_processes:
                    env.env_done = False
            else:
                for conn in self._conns:
                    conn.send((RESET_ENVS_DONE, None))

        self._cumulate_ma_n_agents_list(decision_ma_envs_agent_ids)

        return (DecisionStep(decision_ma_agent_ids,
                             decision_ma_obs_list,
                             decision_ma_last_reward),
                TerminalStep(terminal_ma_agent_ids,
                             terminal_ma_obs_list,
                             terminal_ma_last_reward,
                             terminal_ma_max_reached),
                all_envs_done)

    def close(self):
        self._logger.warning('Closing environments')
        if self._seq_processes:
            for env in self._env_processes:
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

    env = UnityWrapper(train_mode=True,
                       env_name=r'C:\Users\fisher\Documents\Unity\win-RL-Envs\RLEnvironments.exe',
                       n_envs=N_ENVS,
                       max_n_envs_per_process=1,
                       scene='MLAgentsTest',
                       force_seq=True)
    ma_obs_names, ma_obs_shapes, ma_d_action_sizes, ma_c_action_size = env.init()
    ma_names = list(ma_obs_names.keys())

    for i in range(100):
        ma_agent_ids, ma_obs_list = env.reset()
        decision_ma_agent_ids = ma_agent_ids

        for j in range(100):
            ma_d_action = {}
            ma_c_action = {}
            for n in ma_names:
                n_agents = len(decision_ma_agent_ids[n])
                d_action, c_action = None, None
                if ma_d_action_sizes[n]:
                    d_action_sizes = ma_d_action_sizes[n]
                    d_action_list = [np.random.randint(0, d_action_size, size=n_agents)
                                     for d_action_size in d_action_sizes]
                    d_action_list = [np.eye(d_action_size, dtype=np.int32)[d_action]
                                     for d_action, d_action_size in zip(d_action_list, d_action_sizes)]
                    d_action = np.concatenate(d_action_list, axis=-1)
                if ma_c_action_size[n]:
                    c_action = np.random.randn(n_agents, ma_c_action_size[n])

            ma_d_action[n] = d_action
            ma_c_action[n] = c_action

            decision_step, terminal_step = env.step(ma_d_action, ma_c_action)
            decision_ma_agent_ids = decision_step.ma_agent_ids

    env.close()

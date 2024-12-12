from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class DecisionStep:
    def __init__(self,
                 ma_agent_ids: Dict[str, np.ndarray],
                 ma_obs_list: Dict[str, List[np.ndarray]],
                 ma_last_reward: Dict[str, np.ndarray],
                 ma_offline_action: Dict[str, np.ndarray] | None = None):
        self.ma_agent_ids = ma_agent_ids
        self.ma_obs_list = ma_obs_list
        self.ma_last_reward = ma_last_reward
        self.ma_offline_action = ma_offline_action


class TerminalStep:
    def __init__(self,
                 ma_agent_ids: Dict[str, np.ndarray],
                 ma_obs_list: Dict[str, List[np.ndarray]],
                 ma_last_reward: Dict[str, np.ndarray],
                 ma_max_reached: Dict[str, np.ndarray],
                 ma_offline_action: Dict[str, np.ndarray] | None = None):
        self.ma_agent_ids = ma_agent_ids
        self.ma_obs_list = ma_obs_list
        self.ma_last_reward = ma_last_reward
        self.ma_max_reached = ma_max_reached
        self.ma_offline_action = ma_offline_action


class EnvWrapper:
    def __init__(self,
                 train_mode: bool = True,
                 env_name: str = None,
                 env_args: Optional[List[str] | Dict] = None,
                 n_envs: int = 1,
                 model_abs_dir: Path | None = None):
        self.train_mode = train_mode
        self.env_name = env_name
        self.env_args = {}
        if env_args is not None:
            if isinstance(env_args, dict):
                self.env_args = env_args
            else:
                for kv in env_args:
                    if '=' in kv:
                        k, v = kv.split('=')
                    else:
                        k, v = kv, 1.
                    self.env_args[k] = v
        self.n_envs = n_envs
        self.model_abs_dir = model_abs_dir

    def init(self) -> Tuple[Dict[str, List[str]],
                            Dict[str, List[Tuple[int]]],
                            Dict[str, List[int]],
                            Dict[str, int]]:
        """
        Returns:
            ma_obs_names: dict[str, list[str]]
            ma_obs_shapes: dict[str, list[(o1, ), (o2, ), (o3_1, o3_2, o3_3), ...]]
            ma_d_action_sizes: dict[str, list[int]], list of all action branches
            ma_c_action_size: dict[str, int]
        """

        raise NotImplementedError()

    def reset(self, reset_config: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray],
                                                                  Dict[str, List[np.ndarray]]] | Tuple[Dict[str, np.ndarray],
                                                                                                       Dict[str, List[np.ndarray]], Dict[str, List[np.ndarray]]]:
        """
        Returns:
            ma_agent_ids: dict[str, (NAgents, )]
            ma_obs_list: dict[str, list[(NAgents, o1), (NAgents, o2), (NAgents, o3_1, o3_2, o3_3)]]
            ma_offline_action (optional): dict[str, (NAgents, )]
        """

        raise NotImplementedError()

    def step(self,
             ma_d_action: Dict[str, np.ndarray],
             ma_c_action: Dict[str, np.ndarray]) -> Tuple[DecisionStep, TerminalStep]:
        """
        Args:
            ma_d_action: dict[str, (NAgents, discrete_action_size)], one hot like action
            ma_c_action: dict[str, (NAgents, continuous_action_size)]

        Returns:
            DecisionStep: decision_ma_agent_ids
                          decision_ma_obs_list
                          decision_ma_last_reward
                          ma_offline_action (optional)
            TerminalStep: terminal_ma_agent_ids
                          terminal_ma_obs_list
                          terminal_ma_last_reward
                          terminal_ma_max_reached
                          ma_offline_action (optional)
            all_envs_done: bool
        """

        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()

    def send_option(self, option: Dict[str, int]):
        pass

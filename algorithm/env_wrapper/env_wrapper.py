from typing import Dict, Optional

import numpy as np


class EnvWrapper:
    def __init__(self,
                 train_mode: bool = True,
                 env_name: str = None,
                 n_envs: int = 1):
        self.train_mode = train_mode
        self.env_name = env_name
        self.n_envs = n_envs

    def init(self):
        """
        Returns:
            observation shapes: dict[str, tuple[(o1, ), (o2, ), (o3_1, o3_2, o3_3), ...]]
            discrete action sizes: dict[str, list[int]], list of all action branches
            continuous action size: dict[str, int]
        """

        raise NotImplementedError()

    def reset(self, reset_config: Optional[Dict] = None):
        """
        return:
            observation: dict[str, list[(NAgents, o1), (NAgents, o2), (NAgents, o3_1, o3_2, o3_3)]]
        """

        raise NotImplementedError()

    def step(self,
             ma_d_action: Dict[str, np.ndarray],
             ma_c_action: Dict[str, np.ndarray]):
        """
        Args:
            d_action: dict[str, (NAgents, discrete_action_size)], one hot like action
            c_action: dict[str, (NAgents, continuous_action_size)]

        Returns:
            observation: dict[str, list[(NAgents, o1), (NAgents, o2), (NAgents, o3_1, o3_2, o3_3)]]
            reward: dict[str, (NAgents, )]
            done: dict[str, (NAgents, )], bool
            max_step: dict[str, (NAgents, )], bool
            ma_padding_mask: dict[str, (NAgents, )], bool
        """

        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()

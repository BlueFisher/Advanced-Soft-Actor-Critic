from typing import Dict, Optional, Union

import numpy as np


class EnvWrapper:
    def __init__(self,
                 train_mode: bool = True,
                 env_name: str = None,
                 env_args: Optional[Union[str, Dict]] = None,
                 n_envs: int = 1):
        self.train_mode = train_mode
        self.env_name = env_name
        self.env_args = {}
        if env_args is not None:
            if isinstance(env_args, dict):
                self.env_args = env_args
            else:
                kv_pairs = env_args.split(' ')
                for kv in kv_pairs:
                    k, v = kv.split(',')
                    self.env_args[k] = v
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

    def send_option(self, option: Dict[str, int]):
        pass

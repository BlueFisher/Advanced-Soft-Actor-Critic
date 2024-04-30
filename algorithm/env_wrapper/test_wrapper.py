from typing import Dict, List, Optional, Tuple

import numpy as np

if __name__ in ('__main__', '__mp_main__'):
    from env_wrapper import DecisionStep, EnvWrapper, TerminalStep
else:
    from .env_wrapper import DecisionStep, EnvWrapper, TerminalStep


def get_ma_agent_ids(n_envs, ma_names):
    return {
        n: np.arange(n_envs)
        for n in ma_names
    }


def get_ma_obs_list(n_envs, ma_obs_shapes):
    return {
        n: [np.random.randn(n_envs, *obs_shape).astype(np.float32) for obs_shape in obs_shapes]
        for n, obs_shapes in ma_obs_shapes.items()
    }


class TestWrapper(EnvWrapper):
    def __init__(self, env_args, n_envs=1):
        super().__init__(True, 'TEST', env_args=env_args, n_envs=n_envs)

        self.env_args = {} if env_args is None else env_args

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

        if 'ma_obs_shapes' in self.env_args:
            self._ma_obs_names = {n: [str(i) for i in range(len(obs_shapes))]
                                  for n, obs_shapes in self.env_args['ma_obs_shapes'].items()}
            self._ma_obs_shapes = self.env_args['ma_obs_shapes']
        else:
            self._ma_obs_names = {
                'test0': ['vector'],
                'test1': ['vector']
            }
            self._ma_obs_shapes = {
                'test0': [(6,)],
                'test1': [(8,)]
            }

        if 'ma_d_action_sizes' in self.env_args:
            ma_d_action_sizes = self.env_args['ma_d_action_sizes']
        else:
            ma_d_action_sizes = {
                'test0': [2, 3, 4],
                'test1': [4, 3, 2]
            }

        if 'ma_c_action_size' in self.env_args:
            ma_c_action_size = self.env_args['ma_c_action_size']
        else:
            ma_c_action_size = {
                'test0': 3,
                'test1': 5
            }

        return self._ma_obs_names, self._ma_obs_shapes, ma_d_action_sizes, ma_c_action_size

    def reset(self, reset_config: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray],
                                                                  Dict[str, List[np.ndarray]]]:
        """
        return:
            ma_agent_ids: dict[str, (NAgents, )]
            ma_obs_list: dict[str, list[(NAgents, o1), (NAgents, o2), (NAgents, o3_1, o3_2, o3_3)]]
        """
        return get_ma_agent_ids(self.n_envs, self._ma_obs_names), get_ma_obs_list(self.n_envs, self._ma_obs_shapes)

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
            TerminalStep: terminal_ma_agent_ids
                          terminal_ma_obs_list
                          terminal_ma_last_reward
                          terminal_ma_max_reached
            all_envs_done: bool
        """

        ma_agent_ids = get_ma_agent_ids(self.n_envs, self._ma_obs_names)
        ma_obs_list = get_ma_obs_list(self.n_envs, self._ma_obs_shapes)
        ma_last_reward = {}

        terminal_ma_agent_ids = {}
        terminal_ma_obs_list = {}
        terminal_ma_last_reward = {}
        terminal_ma_max_reached = {}

        for n, agent_ids in ma_agent_ids.items():
            ma_last_reward[n] = np.random.randn(self.n_envs).astype(np.float32)

            done = np.random.rand(self.n_envs) < 0.1
            terminal_ma_agent_ids[n] = agent_ids[done]
            terminal_ma_obs_list[n] = [o[done] for o in ma_obs_list[n]]
            terminal_ma_last_reward[n] = ma_last_reward[n][done]
            terminal_ma_max_reached[n] = np.random.rand(sum(done)) < 0.1

        decision_step = DecisionStep(ma_agent_ids,
                                     ma_obs_list,
                                     ma_last_reward)
        terminal_step = TerminalStep(terminal_ma_agent_ids,
                                     terminal_ma_obs_list,
                                     terminal_ma_last_reward,
                                     terminal_ma_max_reached)

        return decision_step, terminal_step, False

    def close(self):
        pass

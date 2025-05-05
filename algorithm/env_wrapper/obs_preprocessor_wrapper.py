import numpy as np

if __name__ in ('__main__', '__mp_main__'):
    from env_wrapper import DecisionStep, EnvWrapper, TerminalStep
else:
    from .env_wrapper import DecisionStep, EnvWrapper, TerminalStep


class ObsPreprocessorWrapper(EnvWrapper):
    def __init__(self, env: EnvWrapper):
        self._env = env

    def init(self) -> tuple[dict[str, list[str]],
                            dict[str, list[tuple[int]]],
                            dict[str, list[int]],
                            dict[str, int]]:
        """
        Returns:
            ma_obs_names: dict[str, list[str]]
            ma_obs_shapes: dict[str, list[(o1, ), (o2, ), (o3_1, o3_2, o3_3), ...]]
            ma_d_action_sizes: dict[str, list[int]], list of all action branches
            ma_c_action_size: dict[str, int]
        """

        return self._env.init()

    def reset(self, reset_config: dict | None = None) -> tuple[dict[str, np.ndarray],
                                                               dict[str, list[np.ndarray]]]:
        """
        return:
            ma_agent_ids: dict[str, (NAgents, )]
            ma_obs_list: dict[str, list[(NAgents, o1), (NAgents, o2), (NAgents, o3_1, o3_2, o3_3)]]
        """
        return self._env.reset(reset_config)

    def step(self,
             ma_d_action: dict[str, np.ndarray],
             ma_c_action: dict[str, np.ndarray]) -> tuple[DecisionStep, TerminalStep]:
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

        return self._env.step(ma_d_action, ma_c_action)

    def close(self):
        self._env.close()

    def send_option(self, option):
        self._env.send_option(option)
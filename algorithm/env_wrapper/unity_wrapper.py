import logging

import numpy as np


from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfig, EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel


class UnityWrapper:
    def __init__(self,
                 train_mode=True,
                 file_name=None,
                 base_port=5005,
                 seed=None,
                 scene=None,
                 n_agents=1):

        seed = seed if seed is not None else np.random.randint(0, 65536)

        self._logger = logging.getLogger('UnityWrapper')

        engine_configuration_channel = EngineConfigurationChannel()
        self.environment_parameters_channel = EnvironmentParametersChannel()

        self._env = UnityEnvironment(file_name=file_name,
                                     base_port=base_port,
                                     seed=seed,
                                     args=['--scene', scene, '--n_agents', str(n_agents)],
                                     side_channels=[engine_configuration_channel,
                                                    self.environment_parameters_channel])

        if train_mode:
            engine_configuration_channel.set_configuration_parameters(time_scale=100)
        else:
            engine_configuration_channel.set_configuration_parameters(width=1028,
                                                                      height=720,
                                                                      quality_level=5,
                                                                      time_scale=0,
                                                                      target_frame_rate=60)

        self._env.reset()
        self.bahavior_name = self._env.get_behavior_names()[0]

    def init(self):
        behavior_spec = self._env.get_behavior_spec(self.bahavior_name)
        self._logger.info(f'Observation shapes: {behavior_spec.observation_shapes}')
        is_discrete = behavior_spec.is_action_discrete()
        self._logger.info(f'Action size: {behavior_spec.action_size}. Is discrete: {is_discrete}')

        return behavior_spec.observation_shapes, behavior_spec.action_size, is_discrete

    def reset(self, reset_config=None):
        reset_config = {} if reset_config is None else reset_config
        for k, v in reset_config.items():
            self.environment_parameters_channel.set_float_parameter(k, float(v))

        self._env.reset()
        decision_steps, terminal_steps = self._env.get_steps(self.bahavior_name)

        return len(decision_steps), [obs.astype(np.float32) for obs in decision_steps.obs]

    def step(self, action):
        self._env.set_actions(self.bahavior_name, action)
        self._env.step()
        decision_steps, terminal_steps = self._env.get_steps(self.bahavior_name)

        reward = decision_steps.reward
        reward[terminal_steps.agent_id] = terminal_steps.reward

        done = np.full([10, ], False, dtype=np.bool)
        done[terminal_steps.agent_id] = True

        max_step = np.full([10, ], False, dtype=np.bool)
        max_step[terminal_steps.agent_id] = terminal_steps.max_step

        return ([obs.astype(np.float32) for obs in decision_steps.obs],
                decision_steps.reward.astype(np.float32),
                done,
                max_step)

    def close(self):
        self._env.close()


if __name__ == "__main__":
    env = UnityWrapper(True, base_port=5004)
    n_agents, obs = env.reset()
    print('n_agents', n_agents)
    # print('obs', obs)

    for _ in range(10000):
        obs, reward, done, max_step = env.step(np.random.randn(n_agents, 2))

        print('done', done)
        input()

    env.close()

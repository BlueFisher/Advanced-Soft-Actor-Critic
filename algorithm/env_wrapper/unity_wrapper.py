import logging

import numpy as np


from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfig, EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

logger = logging.getLogger('UnityWrapper')
logger.setLevel(level=logging.INFO)


class UnityWrapper:
    def __init__(self,
                 train_mode=True,
                 file_name=None,
                 base_port=5005,
                 seed=None,
                 scene=None,
                 n_agents=1):

        seed = seed if seed is not None else np.random.randint(0, 65536)

        self.engine_configuration_channel = EngineConfigurationChannel()
        self.environment_parameters_channel = EnvironmentParametersChannel()

        self._env = UnityEnvironment(file_name=file_name,
                                     base_port=base_port,
                                     seed=seed,
                                     args=['--scene', scene, '--n_agents', str(n_agents)],
                                     side_channels=[self.engine_configuration_channel,
                                                    self.environment_parameters_channel])

        if train_mode:
            self.engine_configuration_channel.set_configuration_parameters(width=200,
                                                                           height=200,
                                                                           quality_level=0,
                                                                           time_scale=100)
        else:
            self.engine_configuration_channel.set_configuration_parameters(width=1028,
                                                                           height=720,
                                                                           quality_level=5,
                                                                           time_scale=5,
                                                                           target_frame_rate=60)

        self._env.reset()
        self.bahavior_name = self._env.get_behavior_names()[0]

    def init(self):
        behavior_spec = self._env.get_behavior_spec(self.bahavior_name)
        logger.info(f'Observation shapes: {behavior_spec.observation_shapes}')
        is_discrete = behavior_spec.is_action_discrete()
        logger.info(f'Action size: {behavior_spec.action_size}. Is discrete: {is_discrete}')

        for o in behavior_spec.observation_shapes:
            if len(o) >= 3:
                self.engine_configuration_channel.set_configuration_parameters(quality_level=5)
                break

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

        tmp_terminal_steps = terminal_steps

        while len(decision_steps) == 0:
            self._env.set_actions(self.bahavior_name, np.empty([0, action.shape[-1]]))
            self._env.step()
            decision_steps, terminal_steps = self._env.get_steps(self.bahavior_name)
            tmp_terminal_steps.agent_id = np.concatenate([tmp_terminal_steps.agent_id,
                                                          terminal_steps.agent_id])
            tmp_terminal_steps.reward = np.concatenate([tmp_terminal_steps.reward,
                                                        terminal_steps.reward])
            tmp_terminal_steps.max_step = np.concatenate([tmp_terminal_steps.max_step,
                                                          terminal_steps.max_step])

        reward = decision_steps.reward
        reward[tmp_terminal_steps.agent_id] = tmp_terminal_steps.reward

        done = np.full([len(decision_steps), ], False, dtype=np.bool)
        done[tmp_terminal_steps.agent_id] = True

        max_step = np.full([len(decision_steps), ], False, dtype=np.bool)
        max_step[tmp_terminal_steps.agent_id] = tmp_terminal_steps.max_step

        return ([obs.astype(np.float32) for obs in decision_steps.obs],
                decision_steps.reward.astype(np.float32),
                done,
                max_step)

    def close(self):
        self._env.close()


if __name__ == "__main__":
    env = UnityWrapper(train_mode=True, base_port=5004)

    for i in range(100):
        n_agents, obs = env.reset()
        for _ in range(100):
            obs, reward, done, max_step = env.step(np.random.randn(n_agents, 2))

    env.close()

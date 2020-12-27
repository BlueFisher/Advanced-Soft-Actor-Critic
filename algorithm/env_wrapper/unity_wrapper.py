import logging
import itertools

import numpy as np
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfig, EngineConfigurationChannel)
from mlagents_envs.side_channel.environment_parameters_channel import \
    EnvironmentParametersChannel

logger = logging.getLogger('UnityWrapper')
logger.setLevel(level=logging.INFO)


class UnityWrapper:
    def __init__(self,
                 train_mode=True,
                 file_name=None,
                 base_port=5005,
                 no_graphics=True,
                 seed=None,
                 scene=None,
                 n_agents=1):

        self.scene = scene

        seed = seed if seed is not None else np.random.randint(0, 65536)

        self.engine_configuration_channel = EngineConfigurationChannel()
        self.environment_parameters_channel = EnvironmentParametersChannel()

        self._env = UnityEnvironment(file_name=file_name,
                                     base_port=base_port,
                                     no_graphics=no_graphics and train_mode,
                                     seed=seed,
                                     additional_args=['--scene', scene, '--n_agents', str(n_agents)],
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
        self.bahavior_name = list(self._env.behavior_specs)[0]

    def init(self):
        behavior_spec = self._env.behavior_specs[self.bahavior_name]
        logger.info(f'Observation shapes: {behavior_spec.observation_shapes}')

        self._empty_action = behavior_spec.action_spec.empty_action

        discrete_action_size = 0
        if behavior_spec.action_spec.discrete_size > 0:
            discrete_action_size = 1
            action_product_list = []
            for action, branch_size in enumerate(behavior_spec.action_spec.discrete_branches):
                discrete_action_size *= branch_size
                action_product_list.append(range(branch_size))
                logger.info(f"Discrete action branch {action} has {branch_size} different actions")

            self.action_product = np.array(list(itertools.product(*action_product_list)))

        continuous_action_size = behavior_spec.action_spec.continuous_size

        logger.info(f'Continuous action size: {continuous_action_size}')

        self.d_action_dim = discrete_action_size
        self.c_action_dim = continuous_action_size

        for o in behavior_spec.observation_shapes:
            if len(o) >= 3:
                self.engine_configuration_channel.set_configuration_parameters(quality_level=5)
                break

        return behavior_spec.observation_shapes, discrete_action_size, continuous_action_size

    def reset(self, reset_config=None):
        reset_config = {} if reset_config is None else reset_config
        for k, v in reset_config.items():
            self.environment_parameters_channel.set_float_parameter(k, float(v))

        self._env.reset()
        decision_steps, terminal_steps = self._env.get_steps(self.bahavior_name)

        return [obs.astype(np.float32) for obs in decision_steps.obs]

    def step(self, d_action, c_action):
        if self.d_action_dim:
            d_action = np.argmax(d_action, axis=1)
            d_action = self.action_product[d_action]

        self._env.set_actions(self.bahavior_name,
                              ActionTuple(continuous=c_action, discrete=d_action))
        self._env.step()

        decision_steps, terminal_steps = self._env.get_steps(self.bahavior_name)

        tmp_terminal_steps = terminal_steps

        while len(decision_steps) == 0:
            self._env.set_actions(self.bahavior_name, self._empty_action(0))
            self._env.step()
            decision_steps, terminal_steps = self._env.get_steps(self.bahavior_name)
            tmp_terminal_steps.agent_id = np.concatenate([tmp_terminal_steps.agent_id,
                                                          terminal_steps.agent_id])
            tmp_terminal_steps.reward = np.concatenate([tmp_terminal_steps.reward,
                                                        terminal_steps.reward])
            tmp_terminal_steps.interrupted = np.concatenate([tmp_terminal_steps.interrupted,
                                                             terminal_steps.interrupted])

        reward = decision_steps.reward
        reward[tmp_terminal_steps.agent_id] = tmp_terminal_steps.reward

        done = np.full([len(decision_steps), ], False, dtype=np.bool)
        done[tmp_terminal_steps.agent_id] = True

        max_step = np.full([len(decision_steps), ], False, dtype=np.bool)
        max_step[tmp_terminal_steps.agent_id] = tmp_terminal_steps.interrupted

        return ([obs.astype(np.float32) for obs in decision_steps.obs],
                decision_steps.reward.astype(np.float32),
                done,
                max_step)

    def close(self):
        self._env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    env = UnityWrapper(train_mode=True, base_port=5004)
    obs_shape_list, d_action_dim, c_action_dim = env.init()

    for i in range(100):
        obs_list = env.reset()
        n_agents = obs_list[0].shape[0]
        for j in range(100):
            d_action, c_action = None, None
            if d_action_dim:
                d_action = np.random.randint(0, d_action_dim, size=n_agents)
                d_action = np.eye(d_action_dim, dtype=np.int32)[d_action]
            if c_action_dim:
                c_action = np.random.randn(n_agents, c_action_dim)

            print(i, j)
            obs_list, reward, done, max_step = env.step(d_action, c_action)

    env.close()

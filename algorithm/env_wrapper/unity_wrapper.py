import logging
import itertools

import numpy as np
from mlagents_envs.environment import UnityEnvironment
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
        self.is_discrete = behavior_spec.is_action_discrete()
        logger.info(f'Action size: {behavior_spec.action_size}. Is discrete: {self.is_discrete}')

        for o in behavior_spec.observation_shapes:
            if len(o) >= 3:
                self.engine_configuration_channel.set_configuration_parameters(quality_level=5)
                break

        if self.is_discrete:
            action_size = 1
            action_product_list = []
            for action, branch_size in enumerate(behavior_spec.discrete_action_branches):
                action_size *= branch_size
                action_product_list.append(range(branch_size))
                logger.info(f"Action number {action} has {branch_size} different options")

            self.action_product = np.array(list(itertools.product(*action_product_list)))

            return behavior_spec.observation_shapes, action_size, 0
        else:
            if self.scene == 'Antisubmarine2':
                return behavior_spec.observation_shapes, 2, 2
            return behavior_spec.observation_shapes, 0, behavior_spec.action_size

    def reset(self, reset_config=None):
        reset_config = {} if reset_config is None else reset_config
        for k, v in reset_config.items():
            self.environment_parameters_channel.set_float_parameter(k, float(v))

        self._env.reset()
        decision_steps, terminal_steps = self._env.get_steps(self.bahavior_name)

        return [obs.astype(np.float32) for obs in decision_steps.obs]

    def step(self, d_action, c_action):
        if self.is_discrete:
            d_action = np.argmax(d_action, axis=1)
            d_action = self.action_product[d_action]

            self._env.set_actions(self.bahavior_name, d_action)
        else:
            if self.scene == 'Antisubmarine2':
                d_action = np.argmax(d_action, axis=1).reshape([-1, 1])
                d_action = d_action * 2 - 1
                c_action = np.concatenate([c_action, d_action], axis=-1)
            self._env.set_actions(self.bahavior_name, c_action)
        self._env.step()
        decision_steps, terminal_steps = self._env.get_steps(self.bahavior_name)

        tmp_terminal_steps = terminal_steps

        while len(decision_steps) == 0:
            self._env.set_actions(self.bahavior_name, np.empty([0, c_action.shape[-1]]))
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
        for _ in range(100):
            action = np.random.randint(0, d_action_dim, size=n_agents)
            action = np.eye(d_action_dim, dtype=np.int32)[action]
            obs_list, reward, done, max_step = env.step(action, None)
            # print(action, obs_list)

    env.close()

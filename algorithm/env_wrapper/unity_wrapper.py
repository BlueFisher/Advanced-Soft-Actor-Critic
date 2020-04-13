import logging

import numpy as np


from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfig, EngineConfigurationChannel
from mlagents_envs.side_channel.float_properties_channel import FloatPropertiesChannel


class UnityWrapper:
    _agent_ids = None
    _n_agents = 1
    _addition_action_dim = 0

    def __init__(self,
                 train_mode=True,
                 file_name=None,
                 base_port=5005,
                 seed=None,
                 no_graphics=False,
                 scene=None,
                 n_agents=1):

        seed = seed if seed is not None else np.random.randint(0, 65536)

        self._logger = logging.getLogger('UnityWrapper')

        engine_configuration_channel = EngineConfigurationChannel()

        if train_mode:
            engine_configuration_channel.set_configuration_parameters(time_scale=100)
        else:
            engine_configuration_channel.set_configuration_parameters(width=1028,
                                                                      height=720,
                                                                      quality_level=5,
                                                                      time_scale=0,
                                                                      target_frame_rate=60)

        self.float_properties_channel = FloatPropertiesChannel()

        self._env = UnityEnvironment(file_name=file_name,
                                     base_port=base_port,
                                     seed=seed,
                                     no_graphics=no_graphics,
                                     args=['--scene', scene, '--n_agents', str(n_agents)],
                                     side_channels=[engine_configuration_channel,
                                                    self.float_properties_channel])

        self._env.reset()
        self.group_name = self._env.get_agent_groups()[0]
        step_result = self._env.get_step_result(self.group_name)
        self._n_agents = step_result.n_agents()

    def init(self):
        group_spec = self._env.get_agent_group_spec(self.group_name)
        self._logger.info(f'Observation shapes: {group_spec.observation_shapes}')
        is_discrete = group_spec.is_action_discrete()
        self._logger.info(f'Action size: {group_spec.action_size}. Is discrete: {is_discrete}')

        return group_spec.observation_shapes, group_spec.action_size, is_discrete

    def reset(self, reset_config=None):
        reset_config = {} if reset_config is None else reset_config
        for k, v in reset_config.items():
            self.float_properties_channel.set_property(k, v)

        self._env.reset()
        step_result = self._env.get_step_result(self.group_name)

        obs_list = step_result.obs

        if step_result.n_agents() == self._n_agents:
            self._agent_ids = step_result.agent_id
            self._addition_action_dim = 0
        elif step_result.n_agents() > self._n_agents:
            obs_list = [obs[-self._n_agents:] for obs in obs_list]
            self._agent_ids = step_result.agent_id[-self._n_agents:]
            self._addition_action_dim = step_result.n_agents() - self._n_agents
        else:
            self._logger.error('reset error')
            return self.reset(reset_config)

        return self._n_agents, [obs.astype(np.float32) for obs in obs_list]

    def step(self, action):
        if self._addition_action_dim != 0:
            action = np.concatenate([np.zeros([self._addition_action_dim, action.shape[-1]]), action], axis=0)
            self._addition_action_dim = 0

        self._env.set_actions(self.group_name, action)

        done_step_result = dict()
        while True:
            self._env.step()
            step_result = self._env.get_step_result(self.group_name)
            n = self._n_agents

            obs_list = step_result.obs

            if step_result.n_agents() < n:
                true_ids = np.where(np.isin(self._agent_ids, step_result.agent_id))[0]
                for i, true_id in enumerate(true_ids):
                    done_step_result[true_id] = (step_result.reward[i],
                                                 step_result.done[i],
                                                 step_result.max_step[i])
                continue

            elif step_result.n_agents() > n:
                true_ids = np.where(np.isin(self._agent_ids, step_result.agent_id[:-n]))[0]
                for i, true_id in enumerate(true_ids):
                    done_step_result[true_id] = (step_result.reward[i],
                                                 step_result.done[i],
                                                 step_result.max_step[i])

                obs_list = [obs[-n:] for obs in obs_list]
                reward = step_result.reward[-n:]
                done = step_result.done[-n:]
                max_step = step_result.max_step[-n:]
                self._agent_ids = step_result.agent_id[-n:]
                self._addition_action_dim = step_result.n_agents() - n
                break

            else:
                if np.all(step_result.done):
                    true_ids = np.where(np.isin(self._agent_ids, step_result.agent_id))[0]
                    for i, true_id in enumerate(true_ids):
                        done_step_result[true_id] = (step_result.reward[i],
                                                     step_result.done[i],
                                                     step_result.max_step[i])
                    continue

                else:
                    reward = step_result.reward
                    done = step_result.done
                    max_step = step_result.max_step
                    self._agent_ids = step_result.agent_id
                    break

        for key, value in done_step_result.items():
            reward[key], done[key], max_step[key] = value
        done_step_result.clear()

        return ([obs.astype(np.float32) for obs in obs_list],
                reward.astype(np.float32),
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

import numpy as np

import sys
sys.path.append('..')

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfig, EngineConfigurationChannel
from mlagents_envs.side_channel.float_properties_channel import FloatPropertiesChannel


class EnvWrapper:
    _agent_ids = None
    _addition_action_dim = 0

    def __init__(self,
                 train_mode=True,
                 file_name=None,
                 base_port=5005,
                 seed=None,
                 no_graphics=False,
                 args=None):

        seed = seed if seed is not None else np.random.randint(0, 65536)

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
                                     args=args,
                                     side_channels=[engine_configuration_channel, self.float_properties_channel])

        self._env.reset()

        self.group_name = self._env.get_agent_groups()[0]

    def init(self):
        group_spec = self._env.get_agent_group_spec(self.group_name)
        return group_spec.observation_shapes[0][0], group_spec.action_size

    def reset(self, reset_config=None):
        reset_config = {} if reset_config is None else reset_config
        for k, v in reset_config.items():
            self.float_properties_channel.set_property(k, v)

        self._env.reset()
        step_result = self._env.get_step_result(self.group_name)
        self._agent_ids = step_result.agent_id

        return step_result.n_agents(), step_result.obs[0]

    def step(self, action):
        if self._addition_action_dim != 0:
            action = np.concatenate([np.zeros([self._addition_action_dim, action.shape[-1]]), action], axis=0)
            self._addition_action_dim = 0

        self._env.set_actions(self.group_name, action)

        done_step_result = dict()
        while True:
            self._env.step()
            step_result = self._env.get_step_result(self.group_name)
            agents_len = len(self._agent_ids)

            if step_result.n_agents() < agents_len:
                true_ids = np.where(np.isin(self._agent_ids, step_result.agent_id))[0]
                for i, true_id in enumerate(true_ids):
                    done_step_result[true_id] = (step_result.reward[i],
                                                 step_result.done[i],
                                                 step_result.max_step[i])
                continue

            elif step_result.n_agents() > agents_len:
                true_ids = np.where(np.isin(self._agent_ids, step_result.agent_id[:-agents_len]))[0]
                for i, true_id in enumerate(true_ids):
                    done_step_result[true_id] = (step_result.reward[i],
                                                 step_result.done[i],
                                                 step_result.max_step[i])
                obs = step_result.obs[0][-agents_len:]
                reward = step_result.reward[-agents_len:]
                done = step_result.done[-agents_len:]
                max_step = step_result.max_step[-agents_len:]
                self._agent_ids = step_result.agent_id[-agents_len:]
                self._addition_action_dim = step_result.n_agents() - agents_len
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
                    obs = step_result.obs[0]
                    reward = step_result.reward
                    done = step_result.done
                    max_step = step_result.max_step
                    self._agent_ids = step_result.agent_id
                    break

        for key, value in done_step_result.items():
            reward[key], done[key], max_step[key] = value
        done_step_result.clear()

        return obs, reward, done, max_step

    def close(self):
        self._env.close()


if __name__ == "__main__":

    env = EnvWrapper(True, base_port=5004)
    n_agents, obs = env.reset({
        'copy': 4
    })
    print('n_agents', n_agents)
    print('obs', obs)

    for _ in range(10000):
        obs, reward, done, max_step = env.step(np.random.randn(n_agents, 2))

        print('obs', obs.shape)
        print('reward', reward)
        print('done', done)
        print('max_step', max_step)

    env.close()

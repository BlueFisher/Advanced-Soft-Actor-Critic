import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfig, EngineConfigurationChannel
from mlagents_envs.side_channel.float_properties_channel import FloatPropertiesChannel


class EnvWrapper:
    def __init__(self,
                 train_mode=True,
                 file_name=None,
                 base_port=5005,
                 seed=None,
                 no_graphics=False,
                 args=None,
                 init_config=None):

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

        return step_result.agent_id, step_result.obs[0]

    def step(self, action):
        self._env.set_actions(self.group_name, action)
        self._env.step()
        step_result = self._env.get_step_result(self.group_name)

        return step_result.obs[0], step_result.reward, step_result.done, step_result.max_step

    def close(self):
        self._env.close()

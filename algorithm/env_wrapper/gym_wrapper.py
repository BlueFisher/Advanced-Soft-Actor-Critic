import logging
from typing import Dict, Optional, Union

import gymnasium as gym
import numpy as np

if __name__ in ('__main__', '__mp_main__'):
    from env_wrapper import EnvWrapper
else:
    from .env_wrapper import EnvWrapper


try:
    import memory_corridor
except Exception as e:
    print('MemoryCorridor is not installed')


class GymWrapper(EnvWrapper):
    def __init__(self,
                 train_mode: bool = True,
                 env_name: str = None,
                 env_args: Optional[Union[str, Dict]] = None,
                 n_envs: int = 1,

                 render=False):
        super().__init__(train_mode, env_name, env_args, n_envs)

        self.render = render

        self._logger = logging.getLogger('GymWrapper')

        self._logger.info(', '.join(gym.registry.keys()))

    def init(self):
        self.env = env = gym.vector.make(self.env_name,
                                         render_mode='human' if self.render else None,
                                         num_envs=self.n_envs,
                                         **self.env_args)

        self._logger.info(f'Observation shapes: {env.observation_space}')
        self._logger.info(f'Action size: {env.action_space}')

        obs_shape = env.single_observation_space.shape
        self.is_discrete = isinstance(env.single_action_space, gym.spaces.Discrete)

        d_action_sizes, c_action_size = [], 0

        if self.is_discrete:
            d_action_sizes = [env.single_action_space.n]
        else:
            c_action_size = env.single_action_space.shape[0]

        return ({'gym': ['vector']},
                {'gym': [obs_shape]},
                {'gym': d_action_sizes},
                {'gym': c_action_size})

    def reset(self, reset_config: Optional[Dict] = None):
        obs, info = self.env.reset(options={**reset_config}
                                   if reset_config is not None else None)

        obs = obs.astype(np.float32)

        return {'gym': [obs]}

    def step(self,
             ma_d_action: Dict[str, np.ndarray],
             ma_c_action: Dict[str, np.ndarray]):
        if self.is_discrete:
            d_action = ma_d_action['gym']
            # Convert one-hot to label
            action = np.argmax(d_action, axis=1)
        else:
            c_action = ma_c_action['gym']
            action = c_action

        obs, reward, done, max_step, info = self.env.step(action)

        obs = obs.astype(np.float32)
        reward = reward.astype(np.float32)

        return {'gym': [obs]}, {'gym': reward}, {'gym': done}, {'gym': max_step}, {'gym': np.zeros_like(done)}

    def close(self):
        self.env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    N_ENVS = 2

    env = GymWrapper(True, 'MemoryCorridor-v0', render=True, n_envs=N_ENVS)
    ma_obs_names, ma_obs_shapes, ma_d_action_sizes, ma_c_action_size = env.init()
    ma_names = list(ma_obs_shapes.keys())

    for i in range(100):
        ma_obs_list = env.reset(reset_config={'block_layer_count': 1})

        for j in range(100):
            ma_d_action = {}
            ma_c_action = {}
            for n in ma_names:
                d_action, c_action = None, None
                if ma_d_action_sizes[n]:
                    d_action_sizes = ma_d_action_sizes[n]
                    d_action_list = [np.random.randint(0, d_action_size, size=N_ENVS)
                                     for d_action_size in d_action_sizes]
                    d_action_list = [np.eye(d_action_size, dtype=np.int32)[d_action]
                                     for d_action, d_action_size in zip(d_action_list, d_action_sizes)]
                    d_action = np.concatenate(d_action_list, axis=-1)
                if ma_c_action_size[n]:
                    c_action = np.random.randn(N_ENVS, ma_c_action_size[n])

            ma_d_action[n] = d_action
            ma_c_action[n] = c_action

            ma_obs_list, ma_reward, ma_done, ma_max_step, ma_padding_mask = env.step(ma_d_action, ma_c_action)

    env.close()

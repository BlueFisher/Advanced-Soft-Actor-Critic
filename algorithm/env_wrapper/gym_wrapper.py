import logging
import time

import gymnasium as gym
import numpy as np


class GymWrapper:
    def __init__(self,
                 train_mode=True,
                 env_name=None,
                 render=False,
                 n_copies=1):
        self.train_mode = train_mode
        self.env_name = env_name
        self.render = render
        self.n_copies = n_copies

        self._logger = logging.getLogger('GymWrapper')

        self._logger.info(list(gym.registry.keys()))

    def init(self):
        self.env = env = gym.vector.make(self.env_name,
                                         render_mode='human' if self.render else None,
                                         num_envs=self.n_copies)

        self._logger.info(f'Observation shapes: {env.observation_space}')
        self._logger.info(f'Action size: {env.action_space}')

        self._state_dim = env.single_observation_space.shape[0]
        self.is_discrete = isinstance(env.single_action_space, gym.spaces.Discrete)

        d_action_size, c_action_size = 0, 0

        if self.is_discrete:
            d_action_size = env.single_action_space.n
        else:
            c_action_size = env.single_action_space.shape[0]

        return ({'gym': [(self._state_dim, )]},
                {'gym': d_action_size},
                {'gym': c_action_size})

    def reset(self, reset_config=None):
        obs, info = self.env.reset()

        obs = obs.astype(np.float32)

        return {'gym': [obs]}

    def step(self, ma_d_action, ma_c_action):
        if self.is_discrete:
            d_action = ma_d_action['gym']
            # Convert one-hot to label
            action = np.argmax(d_action, axis=1)
        else:
            c_action = ma_c_action['gym']
            action = c_action

        obs, reward, done, max_step, info = self.env.step(action)

        return {'gym': [obs]}, {'gym': reward}, {'gym': done}, {'gym': max_step}

    def close(self):
        self.env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    N_COPIES = 2

    env = GymWrapper(True, 'Ant-v4', render=True, n_copies=2)
    ma_obs_shapes, ma_d_action_size, ma_c_action_size = env.init()
    ma_names = list(ma_obs_shapes.keys())

    for i in range(100):
        ma_obs_list = env.reset()

        for j in range(100):
            ma_d_action = {}
            ma_c_action = {}
            for n in ma_names:
                d_action, c_action = None, None
                if ma_d_action_size[n]:
                    d_action = np.random.randint(0, ma_d_action_size[n], size=N_COPIES)
                    d_action = np.eye(ma_d_action_size[n], dtype=np.int32)[d_action]
                if ma_c_action_size[n]:
                    c_action = np.random.randn(N_COPIES, ma_c_action_size[n])

            ma_d_action[n] = d_action
            ma_c_action[n] = c_action

            ma_obs_list, ma_reward, ma_done, ma_max_step = env.step(ma_d_action, ma_c_action)

    env.close()

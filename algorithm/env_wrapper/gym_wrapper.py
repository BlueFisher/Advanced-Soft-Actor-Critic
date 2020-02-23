import numpy as np
import gym


class GymWrapper:
    def __init__(self,
                 train_mode=True,
                 env_name=None):
        self.train_mode = train_mode
        self.env_name = env_name

        self._env = gym.make(env_name)

    def init(self):
        return self._env.observation_space.shape[0], self._env.action_space.shape[0]

    def reset(self, reset_config=None):
        obs = self._env.reset()

        return 1, np.expand_dims(obs, 0)

    def step(self, action):
        obs, reward, done, info = self._env.step(action[0])

        if not self.train_mode:
            self._env.render()

        if self.env_name == 'Pendulum-v0':
            reward = reward / 10

        # obs, reward, done, max_step
        return np.expand_dims(obs, 0), [reward], [done], [False]

    def close(self):
        self._env.close()

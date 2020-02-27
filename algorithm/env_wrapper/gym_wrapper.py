import numpy as np
import gym


class GymWrapper:
    def __init__(self,
                 train_mode=True,
                 env_name=None):
        self.train_mode = train_mode
        self.env_name = env_name
        self._envs = list()

    def init(self):
        env = gym.make(self.env_name)
        self._state_dim = env.observation_space.shape[0]
        self._action_dim = env.action_space.shape[0]

        return self._state_dim, self._action_dim

    def reset(self, reset_config=None):
        if reset_config['copy'] != len(self._envs):
            for env in self._envs:
                env.close()

            for _ in range(reset_config['copy']):
                self._envs.append(gym.make(self.env_name))

        obs = np.stack([e.reset() for e in self._envs], axis=0)

        return len(self._envs), obs.astype(np.float32)

    def step(self, action):
        obs = np.empty([len(self._envs), self._state_dim], dtype=np.float32)
        reward = np.empty(len(self._envs), dtype=np.float32)
        done = np.empty(len(self._envs), dtype=np.bool)
        max_step = np.full(len(self._envs), False)

        for i, env in enumerate(self._envs):
            tmp_obs, tmp_reward, tmp_done, info = env.step(action[i])
            obs[i] = tmp_obs
            reward[i] = tmp_reward
            done[i] = tmp_done

        if not self.train_mode:
            self._envs[0].render()

        for i in np.where(done)[0]:
            obs[i] = self._envs[i].reset()

        return obs, reward, done, max_step

    def close(self):
        for env in self._envs:
            env.close()

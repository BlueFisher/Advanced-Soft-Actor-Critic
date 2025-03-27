from pathlib import Path
import random

import numpy as np

from .test_wrapper import TestWrapper


class TestOfflineWrapper(TestWrapper):
    def __init__(self,
                 env_name: str,
                 env_args: dict | None = None,
                 n_envs: int = 1,
                 model_abs_dir: Path | None = None):
        super().__init__(env_args, n_envs, model_abs_dir)

    def get_episode(self):
        ma_ep_obs_list = {}
        ma_ep_action = {}
        ma_reward = {}
        ma_done = {}

        for n in self._ma_obs_names.keys():
            ep_len = random.randint(100, 200)

            ma_ep_obs_list[n] = [np.random.rand(1, ep_len, *obs_shape).astype(np.float32) for obs_shape in self._ma_obs_shapes[n]]

            action = np.zeros((1, ep_len, sum(self._ma_d_action_sizes[n]) + self._ma_c_action_size[n]), dtype=np.float32)
            if self._ma_d_action_sizes[n]:
                d_action_list = [np.random.randint(0, d_action_size, size=(1, ep_len))
                                 for d_action_size in self._ma_d_action_sizes[n]]
                d_action_list = [np.eye(d_action_size, dtype=np.int32)[d_action]
                                 for d_action, d_action_size in zip(d_action_list, self._ma_d_action_sizes[n])]
                d_action = np.concatenate(d_action_list, axis=-1)
                action[:, :, :sum(self._ma_d_action_sizes[n])] = d_action
            if self._ma_c_action_size[n]:
                c_action = np.tanh(np.random.randn(1, ep_len, self._ma_c_action_size[n]))
                # c_action = np.ones((n_agents, self.c_action_size), dtype=np.float32)
                action[:, :, sum(self._ma_d_action_sizes[n]):] = c_action
            ma_ep_action[n] = action

            ma_reward[n] = np.random.rand(1, ep_len).astype(np.float32)

            done = np.zeros((1, ep_len), dtype=bool)
            done[:, -2:] = True
            ma_done[n] = done

        return (ma_ep_obs_list,
                ma_ep_action,
                ma_reward,
                ma_done)

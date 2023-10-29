import numpy as np

if __name__ in ('__main__', '__mp_main__'):
    from env_wrapper import EnvWrapper
else:
    from .env_wrapper import EnvWrapper


def get_ma_obs_list(n_envs, ma_obs_shapes):
    return {
        n: [np.random.randn(n_envs, *obs_shape).astype(np.float32) for obs_shape in obs_shapes]
        for n, obs_shapes in ma_obs_shapes.items()
    }


class TestWrapper(EnvWrapper):
    def __init__(self, env_args, n_envs=1):
        self.env_args = {} if env_args is None else env_args

        super().__init__(True, 'TEST', n_envs)

    def init(self):
        if 'ma_obs_shapes' in self.env_args:
            self._ma_obs_names = {n: [str(i) for i in range(len(obs_shapes))]
                                  for n, obs_shapes in self.env_args['ma_obs_shapes'].items()}
            self._ma_obs_shapes = self.env_args['ma_obs_shapes']
        else:
            self._ma_obs_names = {
                'test0': ['vector'],
                'test1': ['vector']
            }
            self._ma_obs_shapes = {
                'test0': [(6,)],
                'test1': [(8,)]
            }

        if 'ma_d_action_sizes' in self.env_args:
            ma_d_action_sizes = self.env_args['ma_d_action_sizes']
        else:
            ma_d_action_sizes = {
                'test0': [2, 3, 4],
                'test1': [4, 3, 2]
            }

        if 'ma_c_action_size' in self.env_args:
            ma_c_action_size = self.env_args['ma_c_action_size']
        else:
            ma_c_action_size = {
                'test0': 3,
                'test1': 5
            }

        return self._ma_obs_names, self._ma_obs_shapes, ma_d_action_sizes, ma_c_action_size

    def reset(self, reset_config=None):
        return get_ma_obs_list(self.n_envs, self._ma_obs_shapes)

    def step(self, ma_d_action, ma_c_action):
        """
        Returns:
            ma_next_obs_list
            ma_reward
            ma_local_done
            ma_max_reached
            ma_next_padding_mask
        """

        ma_local_done = {
            'test0': np.random.rand(self.n_envs) < 0.1,
            'test1': np.random.rand(self.n_envs) < 0.1
        }

        ma_max_reached = {
            'test0': np.logical_and(np.random.rand(self.n_envs) < 0.1, ma_local_done['test0']),
            'test1': np.logical_and(np.random.rand(self.n_envs) < 0.1, ma_local_done['test1'])
        }

        ma_padding_mask = {
            'test0': np.logical_and(np.random.rand(self.n_envs) < 0.5, ~ma_local_done['test0']),
            'test1': np.logical_and(np.random.rand(self.n_envs) < 0.5, ~ma_local_done['test1'])
        }

        return (get_ma_obs_list(self.n_envs, self._ma_obs_shapes), {
            'test0': np.random.randn(self.n_envs).astype(np.float32),
            'test1': np.random.randn(self.n_envs).astype(np.float32),
        }, ma_local_done, ma_max_reached, ma_padding_mask)

    def close(self):
        pass

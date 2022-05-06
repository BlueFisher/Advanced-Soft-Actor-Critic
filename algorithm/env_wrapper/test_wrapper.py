import numpy as np


def get_ma_obs_list(n_agents, ma_obs_shapes):
    return {
        n: [np.random.randn(n_agents, *obs_shape).astype(np.float32) for obs_shape in obs_shapes]
        for n, obs_shapes in ma_obs_shapes.items()
    }


class TestWrapper:
    def __init__(self, env_args, n_agents=1):
        self.env_args = env_args
        self.n_agents = n_agents

    def init(self):
        if 'ma_obs_shapes' in self.env_args:
            self._ma_obs_shapes = self.env_args['ma_obs_shapes']
        else:
            self._ma_obs_shapes = {
                'test0': [(6,)],
                'test1': [(8,)]
            }

        if 'ma_d_action_size' in self.env_args:
            ma_d_action_size = self.env_args['ma_d_action_size']
        else:
            ma_d_action_size = {
                'test0': 2,
                'test1': 4
            }

        if 'ma_c_action_size' in self.env_args:
            ma_c_action_size = self.env_args['ma_c_action_size']
        else:
            ma_c_action_size = {
                'test0': 3,
                'test1': 5
            }

        return self._ma_obs_shapes, ma_d_action_size, ma_c_action_size

    def reset(self, reset_config=None):
        return get_ma_obs_list(self.n_agents, self._ma_obs_shapes)

    def step(self, ma_d_action, ma_c_action):
        return (get_ma_obs_list(self.n_agents, self._ma_obs_shapes), {
            'test0': np.random.randn(self.n_agents).astype(np.float32),
            'test1': np.random.randn(self.n_agents).astype(np.float32),
        }, {
            'test0': [False] * self.n_agents,
            'test1': [False] * self.n_agents
        }, {
            'test0': [False] * self.n_agents,
            'test1': [False] * self.n_agents
        })

    def close(self):
        pass

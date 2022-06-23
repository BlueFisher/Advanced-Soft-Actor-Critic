import logging
import multiprocessing
import time

import gym
import gym_minigrid
import numpy as np
from .gym_wrapper import *


class MiniGridWrapper(GymWrapper):
    def __init__(self,
                 train_mode=True,
                 env_name=None,
                 render=False,
                 n_agents=1,
                 force_seq=None):
        super().__init__(train_mode,
                         env_name,
                         render,
                         n_agents,
                         force_seq)

        self._logger = logging.getLogger('MiniGridWrapper')

    def init(self):
        if self._seq_envs:
            for i in range(self.n_agents):
                self._envs.append(gym.make(self.env_name))
            env = self._envs[0]

            if self.render or not self.train_mode:
                env.render()
        else:
            env = gym.make(self.env_name)

        self.obs_shapes = [env.observation_space['image'].shape, (1,)]

        self._logger.info(f'Observation shapes: {self.obs_shapes}')
        self._logger.info(f'Action size: {env.action_space}')

        d_action_size = env.action_space.n
        if 'Empty' in self.env_name:
            d_action_size = 3

        if not self._seq_envs:
            env.close()

            for i in range(self.n_agents):
                parent_conn, child_conn = multiprocessing.Pipe()
                self._conns.append(parent_conn)
                p = multiprocessing.Process(target=start_gym_process,
                                            args=(self.env_name,
                                                  (self.render or not self.train_mode) and i == 0,
                                                  child_conn),
                                            daemon=True)
                p.start()

        return ({'gym': self.obs_shapes},
                {'gym': d_action_size},
                {'gym': 0})

    def reset(self, reset_config=None):
        if self._seq_envs:
            raw_obs_dict = [e.reset() for e in self._envs]
            obs_list = [
                np.stack([o['image'] for o in raw_obs_dict], axis=0),
                np.stack([[o['direction']] for o in raw_obs_dict], axis=0)
            ]
            obs_list = [o.astype(np.float32) for o in obs_list]
        else:
            for conn in self._conns:
                conn.send((RESET, None))

            obs_list = [np.empty([self.n_agents, *o], dtype=np.float32) for o in self.obs_shapes]

            for i, conn in enumerate(self._conns):
                t_obs_dict = conn.recv()
                obs_list[0][i] = t_obs_dict['image']
                obs_list[1][i] = t_obs_dict['direction']

        return {'gym': obs_list}

    def step(self, ma_d_action, ma_c_action):
        d_action = ma_d_action['gym']

        obs_list = [np.empty([self.n_agents, *o], dtype=np.float32) for o in self.obs_shapes]
        reward = np.empty(self.n_agents, dtype=np.float32)
        done = np.empty(self.n_agents, dtype=bool)
        max_step = np.full(self.n_agents, False)

        # Convert one-hot to label
        action = np.argmax(d_action, axis=1)
        action = action.tolist()

        if self._seq_envs:
            for i, env in enumerate(self._envs):
                tmp_obs_dict, tmp_reward, tmp_done, info = env.step(action[i])
                obs_list[0][i] = tmp_obs_dict['image']
                obs_list[1][i] = tmp_obs_dict['direction']
                reward[i] = tmp_reward
                done[i] = tmp_done

            if self.render or not self.train_mode:
                self._envs[0].render()

            for i in np.where(done)[0]:
                tmp_obs_dict = self._envs[i].reset()
                obs_list[0][i] = tmp_obs_dict['image']
                obs_list[1][i] = tmp_obs_dict['direction']
        else:
            for i, conn in enumerate(self._conns):
                conn.send((STEP, action[i]))

            for i, conn in enumerate(self._conns):
                tmp_obs_dict, tmp_reward, tmp_done, info = conn.recv()
                obs_list[0][i] = tmp_obs_dict['image']
                obs_list[1][i] = tmp_obs_dict['direction']
                reward[i] = tmp_reward
                done[i] = tmp_done

            done_index = np.where(done)[0]
            for i in done_index:
                self._conns[i].send((RESET, None))

            for i in done_index:
                tmp_obs_dict = self._conns[i].recv()
                obs_list[0][i] = tmp_obs_dict['image']
                obs_list[1][i] = tmp_obs_dict['direction']

        return {'gym': obs_list}, {'gym': reward}, {'gym': done}, {'gym': max_step}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    n_agents = 3
    gym_wrapper = MiniGridWrapper(True, 'MiniGrid-Empty-16x16-v0', n_agents=n_agents, force_seq=False, render=True)
    obs_shapes, d_action_size, c_action_size = gym_wrapper.init()
    obs_shapes, d_action_size, c_action_size = obs_shapes['gym'], d_action_size['gym'], c_action_size['gym']

    tt = []
    for _ in range(3):
        t = time.time()
        obs_list = gym_wrapper.reset()
        for i in range(1000):
            gym_wrapper.step({'gym': np.eye(d_action_size)[np.random.rand(n_agents, d_action_size).argmax(axis=-1)]},
                             {'gym': None})
        tt.append(time.time() - t)

    # print(tt)
    # print(sum(tt) / len(tt))

    gym_wrapper.close()

import logging
import time
import multiprocessing
import os

import numpy as np
import gym

logger = logging.getLogger('GymWrapper')
logger.setLevel(level=logging.INFO)

try:
    import pybullet_envs
except:
    logger.warning('No PyBullet environments')

try:
    import gym_maze
except:
    logger.warning('No Gym Maze environments')


VANILLA_ENVS = ['CartPole-v1', 'MountainCarContinuous-v0']

RESET = 0
STEP = 1
CLOSE = 2


def start_gym(env_name, render, conn):
    logger.info(f'Process {os.getpid()} created')

    env = gym.make(env_name)
    if render:
        env.render()

    try:
        while True:
            cmd, data = conn.recv()
            if cmd == RESET:
                obs = env.reset()
                conn.send(obs)
            elif cmd == CLOSE:
                env.close()
                break
            elif cmd == STEP:
                conn.send(env.step(data))
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(e)

    logger.warning(f'Process {os.getpid()} exits')


class GymWrapper:
    def __init__(self,
                 train_mode=True,
                 env_name=None,
                 render=False,
                 n_agents=1,
                 force_seq=None):
        self.train_mode = train_mode
        self.env_name = env_name
        self.render = render
        self.n_agents = n_agents

        # If use multiple processes
        if force_seq is None:
            self._seq_envs = n_agents == 1 or env_name in VANILLA_ENVS
        else:
            self._seq_envs = force_seq

        if self._seq_envs:
            # All environments are executed sequentially
            self._envs = list()
        else:
            # All environments are executed in parallel
            self._conns = list()

    def init(self):
        if self._seq_envs:
            for i in range(self.n_agents):
                # TODO
                # spec = gym.envs.registry.spec('MinitaurBulletEnv-v0')
                # spec._kwargs['render'] = not self.train_mode and i == 0
                self._envs.append(gym.make(self.env_name))
            env = self._envs[0]

            if self.render or not self.train_mode:
                env.render()
        else:
            env = gym.make(self.env_name)

        logger.info(f'Observation shapes: {env.observation_space}')
        logger.info(f'Action size: {env.action_space}')

        self._state_dim = env.observation_space.shape[0]
        self.is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        if self.is_discrete:
            action_dim = env.action_space.n
        else:
            action_dim = env.action_space.shape[0]

        if not self._seq_envs:
            env.close()

            for i in range(self.n_agents):
                parent_conn, child_conn = multiprocessing.Pipe()
                self._conns.append(parent_conn)
                p = multiprocessing.Process(target=start_gym,
                                            args=(self.env_name,
                                                  (self.render or not self.train_mode) and i == 0,
                                                  child_conn),
                                            daemon=True)
                p.start()

        return [(self._state_dim, )], action_dim, self.is_discrete

    def reset(self, reset_config=None):
        if self._seq_envs:
            obs = np.stack([e.reset() for e in self._envs], axis=0)
            obs = obs.astype(np.float32)
        else:
            for conn in self._conns:
                conn.send((RESET, None))

            obs = np.empty([self.n_agents, self._state_dim], dtype=np.float32)

            for i, conn in enumerate(self._conns):
                t_obs = conn.recv()
                obs[i] = t_obs

        return [obs]

    def step(self, action):
        obs = np.empty([self.n_agents, self._state_dim], dtype=np.float32)
        reward = np.empty(self.n_agents, dtype=np.float32)
        done = np.empty(self.n_agents, dtype=np.bool)
        max_step = np.full(self.n_agents, False)

        if self.is_discrete:
            # Convert one-hot to label
            action = np.argmax(action, axis=1)
            action = action.tolist()

        if self._seq_envs:
            for i, env in enumerate(self._envs):
                tmp_obs, tmp_reward, tmp_done, info = env.step(action[i])
                obs[i] = tmp_obs
                reward[i] = tmp_reward
                done[i] = tmp_done

            if self.render or not self.train_mode:
                self._envs[0].render()

            for i in np.where(done)[0]:
                obs[i] = self._envs[i].reset()
        else:
            for i, conn in enumerate(self._conns):
                conn.send((STEP, action[i]))

            for i, conn in enumerate(self._conns):
                tmp_obs, tmp_reward, tmp_done, info = conn.recv()
                obs[i] = tmp_obs
                reward[i] = tmp_reward
                done[i] = tmp_done

            done_index = np.where(done)[0]
            for i in done_index:
                self._conns[i].send((RESET, None))

            for i in done_index:
                obs[i] = self._conns[i].recv()

        if not self.train_mode and self.n_agents == 1 and 'Bullet' in self.env_name:
            time.sleep(0.01)

        return [obs], reward, done, max_step

    def close(self):
        if self._seq_envs:
            for env in self._envs:
                env.close()
        else:
            for conn in self._conns:
                conn.send((CLOSE, None))


if __name__ == "__main__":
    gym_wrapper = GymWrapper(True, 'AntBulletEnv-v0', n_agents=3, force_seq=True)
    gym_wrapper.init()

    tt = []
    for _ in range(3):
        t = time.time()
        gym_wrapper.reset()
        for i in range(1000):
            gym_wrapper.step(np.random.randn(3, 8))
        tt.append(time.time() - t)

    print(tt)
    print(sum(tt) / len(tt))

    gym_wrapper.close()

import logging
import multiprocessing
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from dm_control import suite
from dm_env import TimeStep

RESET = 0
STEP = 1
CLOSE = 2

if __name__ in ('__main__', '__mp_main__'):
    from env_wrapper import EnvWrapper
else:
    from .env_wrapper import EnvWrapper


def start_dm_control_process(domain, task, render, conn):
    try:
        from algorithm import config_helper
        config_helper.set_logger()
    except:
        pass

    logger = logging.getLogger('DMControlWrapper.Process')

    logger.info(f'Process {os.getpid()} created')

    env = suite.load(domain_name=domain, task_name=task)
    # if render:
    #     env.render()

    try:
        while True:
            cmd, data = conn.recv()
            if cmd == RESET:
                time_step = env.reset()
                conn.send(list(time_step.observation.values()))
            elif cmd == CLOSE:
                env.close()
                break
            elif cmd == STEP:
                time_step = env.step(data)
                obs_list = list(time_step.observation.values())
                reward = time_step.reward
                done = time_step.last()
                conn.send((obs_list, reward, done))
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(e)

    logger.warning(f'Process {os.getpid()} exits')


class DMControlWrapper(EnvWrapper):
    def __init__(self,
                 train_mode=True,
                 env_name=None,
                 n_envs=1,

                 render=False,
                 force_seq=None):
        super().__init__(train_mode, env_name, n_envs)
        self.render = render

        # If use multiple processes
        if force_seq is None:
            self._seq_envs = n_envs == 1
        else:
            self._seq_envs = force_seq

        if self._seq_envs:
            # All environments are executed sequentially
            self._envs = []
        else:
            # All environments are executed in parallel
            self._conns = []

        self._logger = logging.getLogger('DMControlWrapper')

    def init(self):
        domain, task = self.env_name.split('/')
        if self._seq_envs:
            for i in range(self.n_envs):
                env = suite.load(domain_name=domain, task_name=task)
                self._envs.append(env)
            env = self._envs[0]

            if self.render or not self.train_mode:
                plt.ion()

                self.fig, self.ax = plt.subplots(figsize=(5, 5), squeeze=True)
                self.ax.axis('off')
                self.ax.grid(False)
        else:
            env = suite.load(domain_name=domain, task_name=task)

        observation_spec = ''
        for k, v in env.observation_spec().items():
            observation_spec += f'\n{k}: {v.shape}'
        self._logger.info(f'Observation shapes: {observation_spec}')
        self._logger.info(f'Action size: {env.action_spec()}')

        self.observation_shapes = [v.shape for v in env.observation_spec().values()]

        d_action_sizes, c_action_size = [], env.action_spec().shape[0]

        if not self._seq_envs:
            env.close()

            for i in range(self.n_envs):
                parent_conn, child_conn = multiprocessing.Pipe()
                self._conns.append(parent_conn)
                p = multiprocessing.Process(target=start_dm_control_process,
                                            args=(domain, task,
                                                  (self.render or not self.train_mode) and i == 0,
                                                  child_conn),
                                            daemon=True)
                p.start()

        return ({'gym': ['vector']},
                {'gym': self.observation_shapes},
                {'gym': d_action_sizes},
                {'gym': c_action_size})

    def reset(self, reset_config=None):
        if self._seq_envs:
            _time_steps = [e.reset() for e in self._envs]
            _obses = [t.observation.values() for t in _time_steps]
            obs_list = [np.stack(t, axis=0).astype(np.float32) for t in zip(*_obses)]

            if self.render or not self.train_mode:
                pixels = self._envs[0].physics.render(height=84, width=84, camera_id=0)
                self.im = self.ax.imshow(pixels)
        else:
            for conn in self._conns:
                conn.send((RESET, None))

            obs_list = [np.zeros([self.n_envs, *s], dtype=np.float32) for s in self.observation_shapes]

            for i, conn in enumerate(self._conns):
                _obs_list = conn.recv()
                for j in range(len(obs_list)):
                    obs_list[j][i] = _obs_list[j]

        return {'gym': obs_list}

    def step(self, ma_d_action, ma_c_action):
        d_action, c_action = ma_d_action['gym'], ma_c_action['gym']

        obs_list = [np.zeros([self.n_envs, *s], dtype=np.float32) for s in self.observation_shapes]
        reward = np.zeros(self.n_envs, dtype=np.float32)
        done = np.zeros(self.n_envs, dtype=bool)
        max_step = np.zeros(self.n_envs, dtype=bool)

        action = c_action

        if self._seq_envs:
            for i, env in enumerate(self._envs):
                time_step: TimeStep = env.step(action[i])
                _obs_list = list(time_step.observation.values())
                for j in range(len(obs_list)):
                    obs_list[j][i] = _obs_list[j]
                reward[i] = time_step.reward
                done[i] = time_step.last()

            for i in np.where(done)[0]:
                time_step: TimeStep = self._envs[i].reset()
                _obs_list = list(time_step.observation.values())
                for j in range(len(obs_list)):
                    obs_list[j][i] = _obs_list[j]

            if self.render or not self.train_mode:
                pixels = self._envs[0].physics.render(height=200, width=200, camera_id=0)
                self.im.set_data(pixels)
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()

        else:
            for i, conn in enumerate(self._conns):
                conn.send((STEP, action[i]))

            for i, conn in enumerate(self._conns):
                tmp_obs_list, tmp_reward, tmp_done = conn.recv()
                for j in range(len(obs_list)):
                    obs_list[j][i] = tmp_obs_list[j]
                reward[i] = tmp_reward
                done[i] = tmp_done

            done_index = np.where(done)[0]
            for i in done_index:
                self._conns[i].send((RESET, None))

            for i in done_index:
                tmp_obs_list = self._conns[i].recv()
                for j in range(len(obs_list)):
                    obs_list[j][i] = tmp_obs_list[j]

        return {'gym': obs_list}, {'gym': reward}, {'gym': done}, {'gym': max_step}, {'gym': np.zeros_like(max_step)}

    def close(self):
        if self._seq_envs:
            for env in self._envs:
                env.close()
        else:
            for conn in self._conns:
                conn.send((CLOSE, None))


if __name__ == "__main__":
    n_envs = 1
    logging.basicConfig(level=logging.INFO)
    wrapper = DMControlWrapper(True, 'cartpole/balance', n_envs=n_envs, force_seq=True, render=True)
    wrapper.init()

    tt = []
    for _ in range(3):
        t = time.time()
        wrapper.reset()
        for i in range(1000):
            obs_list, reward, done, max_step = wrapper.step(None, np.random.randn(n_envs, 1))
        tt.append(time.time() - t)

    print(sum(tt) / len(tt))

    wrapper.close()

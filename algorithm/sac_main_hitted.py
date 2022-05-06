import socket

import numpy as np

from algorithm.utils import format_global_step

from .agent import Agent
from .sac_main import Main


class AgentHitted(Agent):
    hitted = 0

    def _extra_log(self,
                   obs_list,
                   action,
                   reward,
                   local_done,
                   max_reached,
                   next_obs_list,
                   prob):

        if not self.done and reward >= 1:
            self.hitted += 1

    def clear(self):
        super().clear()
        self.hitted = 0

    def reset(self):
        super().reset()
        self.hitted = 0


class MainHitted(Main):
    _agent_class = AgentHitted

    def _run(self):
        self.ma_evaluation_data = {n: {
            'episodes': 0,
            'hitted': 0,
            'hitted_steps': 0,
            'failed_steps': 0
        } for n in self.ma_names}

        super()._run()

        if not self.train_mode:
            for n, ev in self.ma_evaluation_data.items():
                if len(self.ma_names) == 1:
                    result_path = self.model_abs_dir / 'result.txt'
                else:
                    result_path = self.model_abs_dir / f'{n}_result.txt'
                result_path.touch(exist_ok=True)
                hostname = socket.gethostname()
                log = f'{hostname}, {ev["episodes"]}, {ev["hitted"]}, {ev["hitted_steps"]}, {ev["failed_steps"]}'
                with open(result_path, 'a') as f:
                    f.write(log + '\n')
                self._logger.info(log)

    def _log_episode_summaries(self, ma_agents):
        for n, agents in ma_agents.items():
            rewards = np.array([a.reward for a in agents])
            hitted = sum([a.hitted for a in agents])

            self.ma_sac[n].write_constant_summaries([
                {'tag': 'reward/mean', 'simple_value': rewards.mean()},
                {'tag': 'reward/max', 'simple_value': rewards.max()},
                {'tag': 'reward/min', 'simple_value': rewards.min()},
                {'tag': 'reward/hitted', 'simple_value': hitted}
            ])

    def _log_episode_info(self, iteration, iter_time, ma_agents):
        for n, agents in ma_agents.items():
            global_step = format_global_step(self.ma_sac[n].get_global_step())
            rewards = [a.reward for a in agents]
            rewards = ", ".join([f"{i:6.1f}" for i in rewards])
            hitted = sum([a.hitted for a in agents])
            max_step = max([a.steps for a in agents])

            if not self.train_mode:
                for agent in agents:
                    if agent.steps > 10:
                        self.ma_evaluation_data[n]['episodes'] += 1
                        if agent.hitted:
                            self.ma_evaluation_data[n]['hitted'] += 1
                            self.ma_evaluation_data[n]['hitted_steps'] += agent.steps
                        else:
                            self.ma_evaluation_data[n]['failed_steps'] += agent.steps

            self._logger.info(f'{n} {iteration}({global_step}), T {iter_time:.2f}s, S {max_step}, R {rewards}, hitted {hitted}')

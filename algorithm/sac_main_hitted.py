import socket

import numpy as np

from algorithm.utils import format_global_step

from . import agent
from .agent import Agent
from .sac_main import Main


class AgentHitted(Agent):
    hitted = 0

    def _extra_log(self,
                   obs_list,
                   action,
                   reward,
                   done,
                   max_reached,
                   prob):

        if not self.done and reward >= 1:
            self.hitted += 1

    def reset(self):
        super().reset()
        self.hitted = 0


class MainHitted(Main):
    def _run(self):
        for n, mgr in self.ma_manager:
            mgr['evaluation_data'] = {
                'episodes': 0,
                'hitted': 0,
                'hitted_steps': 0,
                'failed_steps': 0
            }

        super()._run()

        if not self.train_mode:
            for n, mgr in self.ma_manager:
                ev = mgr['evaluation_data']

                if len(self.ma_manager) == 1:
                    result_path = self.model_abs_dir / 'result.txt'
                else:
                    result_path = self.model_abs_dir / f'{n}_result.txt'
                result_path.touch(exist_ok=True)
                hostname = socket.gethostname()
                log = f'{hostname}, {ev["episodes"]}, {ev["hitted"]}, {ev["hitted_steps"]}, {ev["failed_steps"]}'
                with open(result_path, 'a') as f:
                    f.write(log + '\n')
                self._logger.info(log)

    def _log_episode_summaries(self):
        for n, mgr in self.ma_manager:
            if n in self.inference_ma_names or len(mgr.non_empty_agents) == 0:
                continue

            rewards = np.array([a.reward for a in mgr.non_empty_agents])
            hitted = sum([a.hitted for a in mgr.non_empty_agents])

            mgr.rl.write_constant_summaries([
                {'tag': 'reward/mean', 'simple_value': rewards.mean()},
                {'tag': 'reward/max', 'simple_value': rewards.max()},
                {'tag': 'reward/min', 'simple_value': rewards.min()},
                {'tag': 'reward/hitted', 'simple_value': hitted}
            ])

            mgr.rl.write_histogram_summaries([
                {'tag': 'reward', 'histogram': rewards}
            ])

            steps = np.array([a.steps for a in mgr.non_empty_agents])

            mgr.rl.write_constant_summaries([
                {'tag': 'metric/steps', 'simple_value': steps.mean()},
            ])

    def _log_episode_info(self, iteration, iter_time):
        for n, mgr in self.ma_manager:
            if len(mgr.non_empty_agents) == 0:
                continue
            global_step = format_global_step(mgr.rl.get_global_step())
            rewards = [a.reward for a in mgr.non_empty_agents]
            rewards = ", ".join([f"{i:6.1f}" for i in rewards])
            hitted = sum([a.hitted for a in mgr.non_empty_agents])
            max_step = max([a.steps for a in mgr.non_empty_agents])

            if not self.train_mode:
                for agent in mgr.non_empty_agents:
                    if agent.steps > 10:
                        mgr['evaluation_data']['episodes'] += 1
                        if agent.hitted:
                            mgr['evaluation_data']['hitted'] += 1
                            mgr['evaluation_data']['hitted_steps'] += agent.steps
                        else:
                            mgr['evaluation_data']['failed_steps'] += agent.steps

            self._logger.info(f'{n} {iteration}({global_step}), T {iter_time:.2f}s, S {max_step}, R {rewards}, hitted {hitted}')


agent.Agent = AgentHitted

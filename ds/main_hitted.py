import time

import numpy as np

from algorithm.sac_main_hitted import AgentHitted

from .actor import Actor
from .learner import Learner


class LearnerHitted(Learner):
    _agent_class = AgentHitted

    def _log_episode_summaries(self):
        for n, mgr in self.ma_manager:
            rewards = np.array([a.reward for a in mgr.agents])
            hitted = sum([a.hitted for a in mgr.agents])

            mgr['cmd_pipe_client'].send(('LOG_EPISODE_SUMMARIES', [
                {'tag': 'reward/mean', 'simple_value': float(rewards.mean())},
                {'tag': 'reward/max', 'simple_value': float(rewards.max())},
                {'tag': 'reward/min', 'simple_value': float(rewards.min())},
                {'tag': 'reward/hitted', 'simple_value': hitted}
            ]))

    def _log_episode_info(self, iteration, start_time):
        for n, mgr in self.ma_manager:
            time_elapse = (time.time() - start_time) / 60
            rewards = [a.reward for a in mgr.agents]
            rewards = ", ".join([f"{i:6.1f}" for i in rewards])
            hitted = sum([a.hitted for a in mgr.agents])
            max_step = max([a.steps for a in mgr.agents])

            self._logger.info(f'{n} {iteration}, {time_elapse:.2f}m, S {max_step}, R {rewards}, hitted {hitted}')


class ActorHitted(Actor):
    _agent_class = AgentHitted

    def _log_episode_info(self, iteration):
        for n, mgr in self.ma_manager:
            rewards = [a.reward for a in mgr.agents]
            rewards = ", ".join([f"{i:6.1f}" for i in rewards])
            hitted = sum([a.hitted for a in mgr.agents])
            max_step = max([a.steps for a in mgr.agents])

            self._logger.info(f'{n} {iteration}, S {max_step}, R {rewards}, hitted {hitted}')

import time

import numpy as np

from algorithm import agent
from algorithm.sac_main_hit import AgentHit

from .actor import Actor
from .learner import Learner


class LearnerHit(Learner):
    def _log_episode_summaries(self):
        for n, mgr in self.ma_manager:
            if n in self.inference_ma_names or len(mgr.non_empty_agents) == 0:
                continue
            rewards = np.array([a.reward for a in mgr.non_empty_agents])
            hit = sum([a.hit for a in mgr.non_empty_agents])
            steps = np.array([a.steps for a in mgr.non_empty_agents])

            self._ma_agent_manager_buffer[n].cmd_pipe_client.send(('LOG_EPISODE_SUMMARIES', [
                {'tag': 'reward/mean', 'simple_value': float(rewards.mean())},
                {'tag': 'reward/max', 'simple_value': float(rewards.max())},
                {'tag': 'reward/min', 'simple_value': float(rewards.min())},
                {'tag': 'reward/hit', 'simple_value': hit},
                {'tag': 'metric/steps', 'simple_value': steps.mean()}
            ]))

    def _log_episode_info(self, iteration, iter_time):
        for n, mgr in self.ma_manager:
            if len(mgr.non_empty_agents) == 0:
                continue
            rewards = [a.reward for a in mgr.non_empty_agents]
            rewards = ", ".join([f"{i:6.1f}" for i in rewards])
            hit = sum([a.hit for a in mgr.non_empty_agents])
            max_step = max([a.steps for a in mgr.non_empty_agents])

            self._logger.info(f'{n} {iteration}, {iter_time:.2f}s, S {max_step}, R {rewards}, hit {hit}')


class ActorHit(Actor):
    def _log_episode_info(self, iteration, iter_time):
        for n, mgr in self.ma_manager:
            if len(mgr.non_empty_agents) == 0:
                continue
            rewards = [a.reward for a in mgr.non_empty_agents]
            rewards = ", ".join([f"{i:6.1f}" for i in rewards])
            hit = sum([a.hit for a in mgr.non_empty_agents])
            max_step = max([a.steps for a in mgr.non_empty_agents])

            self._logger.info(f'{n} {iteration}, T {iter_time:.2f}s, S {max_step}, R {rewards}, hit {hit}')


agent.Agent = AgentHit

import time

import numpy as np

from algorithm.sac_main_hitted import AgentHitted

from .actor import Actor
from .learner import Learner


class LearnerHitted(Learner):
    _agent_class = AgentHitted

    def _log_episode_summaries(self, agents):
        rewards = np.array([a.reward for a in agents])
        hitted = sum([a.hitted for a in agents])

        self.cmd_pipe_client.send(('LOG_EPISODE_SUMMARIES', [
            {'tag': 'reward/mean', 'simple_value': float(rewards.mean())},
            {'tag': 'reward/max', 'simple_value': float(rewards.max())},
            {'tag': 'reward/min', 'simple_value': float(rewards.min())},
            {'tag': 'reward/hitted', 'simple_value': hitted}
        ]))

    def _log_episode_info(self, iteration, start_time, agents):
        time_elapse = (time.time() - start_time) / 60
        rewards = [a.reward for a in agents]
        hitted = sum([a.hitted for a in agents])

        rewards = ", ".join([f"{i:6.1f}" for i in rewards])
        steps = [a.steps for a in agents]
        self._logger.info(f'{iteration}, {time_elapse:.2f}m, S {max(steps)}, R {rewards}, hitted {hitted}')


class ActorHitted(Actor):
    _agent_class = AgentHitted

    def _log_episode_info(self, iteration, agents):
        rewards = [a.reward for a in agents]
        hitted = sum([a.hitted for a in agents])

        rewards = ", ".join([f"{i:6.1f}" for i in rewards])
        steps = [a.steps for a in agents]
        self._logger.info(f'{iteration}, S {max(steps)}, R {rewards}, hitted {hitted}')

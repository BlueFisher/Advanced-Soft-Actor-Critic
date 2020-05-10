from pathlib import Path
import sys
import time

import numpy as np

from .learner import Learner
from .actor import Actor

from algorithm.agent import Agent
from algorithm.sac_main_hitted import AgentHitted


class LearnerHitted(Learner):
    _agent_class = AgentHitted

    def _log_episode_summaries(self, agents):
        rewards = np.array([a.reward for a in agents])
        hitted = sum([a.hitted for a in agents])

        self.sac.write_constant_summaries([
            {'tag': 'reward/mean', 'simple_value': rewards.mean()},
            {'tag': 'reward/max', 'simple_value': rewards.max()},
            {'tag': 'reward/min', 'simple_value': rewards.min()},
            {'tag': 'reward/hitted', 'simple_value': hitted}
        ])

    def _log_episode_info(self, iteration, start_time, agents):
        time_elapse = (time.time() - start_time) / 60
        rewards = [a.reward for a in agents]
        hitted = sum([a.hitted for a in agents])

        rewards = ", ".join([f"{i:6.1f}" for i in rewards])
        self.logger.info(f'{iteration}, {time_elapse:.2f}, rewards {rewards}, hitted {hitted}')


class ActorHitted(Actor):
    _agent_class = AgentHitted

    def _log_episode_info(self, iteration, agents):
        rewards = [a.reward for a in agents]
        hitted = sum([a.hitted for a in agents])

        rewards = ", ".join([f"{i:6.1f}" for i in rewards])
        self.logger.info(f'{iteration}, rewards {rewards}, hitted {hitted}')

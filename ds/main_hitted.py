from pathlib import Path
import sys
import time

import numpy as np

from .learner import Learner
from .actor import Actor

from algorithm.agent import Agent


class AgentHitted(Agent):
    hitted = 0

    def _extra_log(self,
                   state,
                   action,
                   reward,
                   local_done,
                   max_reached,
                   state_):
        if not self.done and reward >= 1:
            self.hitted += 1


class LearnerHitted(Learner):
    _agent_class = AgentHitted

    def _log_episode_info(self, iteration, start_time, agents):
        rewards = np.array([a.reward for a in agents])
        hitted = sum([a.hitted for a in agents])

        self.sac.write_constant_summaries([
            {'tag': 'reward/mean', 'simple_value': rewards.mean()},
            {'tag': 'reward/max', 'simple_value': rewards.max()},
            {'tag': 'reward/min', 'simple_value': rewards.min()},
            {'tag': 'reward/hitted', 'simple_value': hitted}
        ], iteration)

        time_elapse = (time.time() - start_time) / 60
        rewards_sorted = ", ".join([f"{i:.1f}" for i in sorted(rewards)])
        self.logger.info(f'{iteration}, {time_elapse:.2f}min, rewards {rewards_sorted}, hitted {hitted}')


class ActorHitted(Actor):
    _agent_class = AgentHitted

    def _log_episode_info(self, global_step, agents):
        rewards = [a.reward for a in agents]
        rewards_sorted = ", ".join([f"{i:.1f}" for i in sorted(rewards)])
        hitted = sum([a.hitted for a in agents])
        self.logger.info(f'{global_step}, rewards {rewards_sorted}, hitted {hitted}')

import time
import logging
import sys
from pathlib import Path

import numpy as np

from learner import Learner
from replay_learner import ReplayLearner
from actor import Actor

sys.path.append(str(Path(__file__).resolve().parent.parent))
from algorithm.agent import Agent

logger = logging.getLogger('sac.ds')


class HittedLearner_Custom(object):
    def _log_episode_info(self, eval_step, start_time, agents):
        rewards = np.array([a.reward for a in agents])
        hitted = sum([a.hitted for a in agents])

        self.sac.write_constant_summaries([
            {'tag': 'reward/mean', 'simple_value': rewards.mean()},
            {'tag': 'reward/max', 'simple_value': rewards.max()},
            {'tag': 'reward/min', 'simple_value': rewards.min()},
            {'tag': 'reward/hitted', 'simple_value': hitted}
        ], eval_step)

        time_elapse = (time.time() - start_time) / 60
        rewards_sorted = ", ".join([f"{i:.1f}" for i in sorted(rewards)])
        logger.info(f'{eval_step}, {time_elapse:.2f}min, rewards {rewards_sorted}, hitted {hitted}')


class LearnerHitted(HittedLearner_Custom, Learner):
    pass


class ReplayLearnerHitted(HittedLearner_Custom, ReplayLearner):
    pass


class AgentHitted(Agent):
    hitted = 0

    def _extra_log(self,
                   state,
                   action,
                   reward,
                   local_done,
                   max_reached,
                   state_):
        if reward >= 1:
            self.hitted += 1


class ActorHitted(Actor):
    def _get_agents(self, agent_ids):
        return [AgentHitted(i) for i in agent_ids]

    def _log_episode_info(self, global_step, agents):
        rewards = [a.reward for a in agents]
        rewards_sorted = ", ".join([f"{i:.1f}" for i in sorted(rewards)])
        hitted = sum([a.hitted for a in agents])
        logger.info(f'{global_step}, rewards {rewards_sorted}, hitted {hitted}')

import numpy as np

from .sac_main import Main
from .agent import Agent


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

    def clear(self):
        super().clear()
        self.hitted = 0

    def reset(self):
        super().reset()
        self.hitted = 0


class MainHitted(Main):
    _agent_class = AgentHitted

    def _log_episode_summaries(self, iteration, agents):
        rewards = np.array([a.reward for a in agents])
        hitted = sum([a.hitted for a in agents])

        self.sac.write_constant_summaries([
            {'tag': 'reward/mean', 'simple_value': rewards.mean()},
            {'tag': 'reward/max', 'simple_value': rewards.max()},
            {'tag': 'reward/min', 'simple_value': rewards.min()},
            {'tag': 'reward/hitted', 'simple_value': hitted}
        ], iteration)

    def _log_episode_info(self, iteration, agents):
        rewards = [a.reward for a in agents]
        hitted = sum([a.hitted for a in agents])

        rewards_sorted = ", ".join([f"{i:.1f}" for i in sorted(rewards)])
        self.logger.info(f'iter {iteration}, rewards {rewards_sorted}, hitted {hitted}')

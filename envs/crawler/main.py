import sys
import logging

import numpy as np

sys.path.append('../..')
from algorithm.sac_main import Main
from algorithm.agent import Agent

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] - [%(name)s] - %(message)s')

    _log = logging.getLogger('tensorflow')
    _log.setLevel(logging.ERROR)

    logger = logging.getLogger('sac')

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

    class MainHitted(Main):
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
            logger.info(f'iter {iteration}, rewards {rewards_sorted}, hitted {hitted}')

    MainHitted(sys.argv[1:], AgentHitted)

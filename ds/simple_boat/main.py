import time
import logging
import getopt
import sys
sys.path.append('..')

import numpy as np

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] - [%(name)s] - %(message)s')

_log = logging.getLogger('werkzeug')
_log.setLevel(logging.WARN)

_log = logging.getLogger('tensorflow')
_log.setLevel(logging.ERROR)

logger = logging.getLogger('sac.ds')

node = sys.argv[1]

# add hitted info


def start_policy_evaluation(self):
    eval_step = 0
    start_time = time.time()
    brain_info = self.env.reset(train_mode=True, config=self._reset_config)[self.default_brain_name]

    while True:
        if self.env.global_done:
            brain_info = self.env.reset(train_mode=True, config=self._reset_config)[self.default_brain_name]

        len_agents = len(brain_info.agents)

        all_done = [False] * len_agents
        all_cumulative_rewards = np.zeros(len_agents)
        hitted = 0

        states = brain_info.vector_observations

        while False in all_done:
            actions = self.sac.choose_action(states)
            brain_info = self.env.step({
                self.default_brain_name: actions
            })[self.default_brain_name]

            rewards = np.array(brain_info.rewards)
            local_dones = np.array(brain_info.local_done, dtype=bool)

            for i in range(len_agents):
                if not all_done[i]:
                    all_cumulative_rewards[i] += rewards[i]
                    if rewards[i] > 0:
                        hitted += 1

                all_done[i] = all_done[i] or local_dones[i]

            states = brain_info.vector_observations

        self.sac.write_constant_summaries([
            {'tag': 'reward/mean', 'simple_value': all_cumulative_rewards.mean()},
            {'tag': 'reward/max', 'simple_value': all_cumulative_rewards.max()},
            {'tag': 'reward/min', 'simple_value': all_cumulative_rewards.min()},
            {'tag': 'reward/hitted', 'simple_value': hitted}
        ], eval_step)

        time_elapse = (time.time() - start_time) / 60
        rewards_sorted = ", ".join([f"{i:.1f}" for i in sorted(all_cumulative_rewards)])
        logger.info(f'{eval_step}, {time_elapse:.2f}min, rewards {rewards_sorted}, hitted {hitted}')
        eval_step += 1


if __name__ == '__main__':
    if node == '-r':
        from replay import Replay
        Replay(sys.argv[2:])
    elif node == '-l':
        from learner import Learner

        class LearnerHitted(Learner):
            def _start_policy_evaluation(self):
                start_policy_evaluation(self)

        LearnerHitted(sys.argv[2:])
    elif node == '-rl':
        from replay_learner import ReplayLearner

        class ReplayLearnerHitted(ReplayLearner):
            def _start_policy_evaluation(self):
                start_policy_evaluation(self)

        ReplayLearnerHitted(sys.argv[2:])
    elif node == '-a':
        from actor import Actor

        class ActorHitted(Actor):
            def _run(self):
                global_step = 0

                brain_info = self.env.reset(train_mode=self._train_mode, config=self._reset_config)[self.default_brain_name]

                while True:
                    if self.env.global_done:
                        brain_info = self.env.reset(train_mode=self._train_mode, config=self._reset_config)[self.default_brain_name]

                    len_agents = len(brain_info.agents)

                    all_done = [False] * len_agents
                    all_cumulative_rewards = [0] * len_agents

                    hitted = 0
                    states = brain_info.vector_observations

                    while False in all_done and not self.env.global_done and not self._reset_signal:
                        if global_step % self._update_policy_variables_per_step == 0:
                            self._update_policy_variables()

                        actions = self.sac_actor.choose_action(states)
                        brain_info = self.env.step({
                            self.default_brain_name: actions
                        })[self.default_brain_name]

                        rewards = np.array(brain_info.rewards)
                        local_dones = np.array(brain_info.local_done, dtype=bool)
                        max_reached = np.array(brain_info.max_reached, dtype=bool)

                        for i in range(len_agents):
                            if not all_done[i]:
                                all_cumulative_rewards[i] += rewards[i]
                                if rewards[i] > 0:
                                    hitted += 1

                            all_done[i] = all_done[i] or local_dones[i]

                        states_ = brain_info.vector_observations

                        dones = np.logical_and(local_dones, np.logical_not(max_reached))
                        s, a, r, s_, done = states, actions, rewards[:, np.newaxis], states_, dones[:, np.newaxis]
                        self._add_trans(s, a, r, s_, done)

                        states = states_
                        global_step += 1

                    if self._reset_signal:
                        self._reset_signal = False

                        self._tmp_trans_buffer.clear()
                        brain_info = self.env.reset(train_mode=self._train_mode, config=self._reset_config)[self.default_brain_name]
                        global_step = 0

                        logger.info('reset')
                        continue

                    rewards_sorted = ", ".join([f"{i:.1f}" for i in sorted(all_cumulative_rewards)])
                    logger.info(f'{global_step}, rewards {rewards_sorted}, hitted {hitted}')

        ActorHitted(sys.argv[2:])
    else:
        logger.error('the first arg must be one of -r, -l, and -a')

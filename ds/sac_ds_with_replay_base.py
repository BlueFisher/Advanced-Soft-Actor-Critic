from pathlib import Path
import sys
import threading
import time

import numpy as np
import tensorflow as tf

from sac_ds_base import SAC_DS_Base

sys.path.append(str(Path(__file__).resolve().parent.parent))
from algorithm.replay_buffer import PrioritizedReplayBuffer


class SAC_DS_with_Replay_Base(SAC_DS_Base):
    def __init__(self,
                 state_dim,
                 action_dim,
                 model_root_path,
                 model,

                 burn_in_step=0,
                 n_step=1,
                 use_rnn=False,

                 seed=None,
                 save_model_per_step=5000,
                 write_summary_per_step=20,
                 tau=0.005,
                 update_target_per_step=1,
                 init_log_alpha=-2.3,
                 use_auto_alpha=True,
                 lr=3e-4,
                 gamma=0.99,

                 replay_config=None):

        replay_config = {} if replay_config is None else replay_config
        self.replay_buffer = PrioritizedReplayBuffer(**replay_config)

        super().__init__(state_dim,
                         action_dim,
                         model_root_path,
                         model,

                         burn_in_step,
                         n_step,
                         use_rnn,

                         seed,
                         save_model_per_step,
                         write_summary_per_step,
                         tau,
                         update_target_per_step,
                         init_log_alpha,
                         use_auto_alpha,
                         lr,
                         gamma)

    def add(self, n_states, n_actions, n_rewards, state_, done, mu_n_probs,
            rnn_state=None):
        if self.use_rnn:
            self.replay_buffer.add(n_states, n_actions, n_rewards, state_, done, mu_n_probs,
                                   rnn_state)
        else:
            self.replay_buffer.add(n_states, n_actions, n_rewards, state_, done, mu_n_probs)

    def add_with_td_errors(self, td_errors,
                           n_states, n_actions, n_rewards, state_, done, mu_n_probs,
                           rnn_state=None):
        if self.use_rnn:
            self.replay_buffer.add_with_td_errors(td_errors,
                                                  n_states, n_actions, n_rewards, state_, done, mu_n_probs,
                                                  rnn_state)
        else:
            self.replay_buffer.add_with_td_errors(td_errors,
                                                  n_states, n_actions, n_rewards, state_, done, mu_n_probs)

    def train(self):
        # sample from replay buffer
        sampled = self.replay_buffer.sample()
        if sampled is None:
            return

        pointers, trans, priority_is = sampled

        if self.use_rnn:
            n_states, n_actions, n_rewards, state_, done, mu_n_probs, rnn_state = trans
        else:
            n_states, n_actions, n_rewards, state_, done, mu_n_probs = trans

        reward = self._get_n_reward_gamma_sum(n_rewards[:, self.burn_in_step:])

        if self.use_rnn:
            pi_n_probs = self.get_n_step_probs(n_states, n_actions,  # TODO: only need [:, burn_in_step:, :]
                                               rnn_state).numpy()
        else:
            pi_n_probs = self.get_n_step_probs(n_states, n_actions).numpy()

        n_step_is = self._get_n_step_is(pi_n_probs[:, self.burn_in_step:, :],
                                        mu_n_probs[:, self.burn_in_step:, :])

        self._train(n_states, n_actions, reward, state_, done,
                    n_step_is=n_step_is,
                    priority_is=priority_is,
                    initial_rnn_state=rnn_state if self.use_rnn else None)

        # update td_error
        if self.use_rnn:
            td_error = self.get_td_error(n_states, n_actions, reward, state_, done,
                                         rnn_state).numpy()
        else:
            td_error = self.get_td_error(n_states, n_actions, reward, state_, done).numpy()

        self.replay_buffer.update(pointers, td_error.flatten())

        # update mu_n_probs
        if self.use_rnn:
            pi_n_probs = self.get_n_step_probs(n_states, n_actions,
                                               rnn_state).numpy()
        else:
            pi_n_probs = self.get_n_step_probs(n_states, n_actions).numpy()

        self.replay_buffer.update_transitions(pointers, 5, pi_n_probs)

        self.save_model()

        return True

from pathlib import Path
import sys
import threading
import time

import numpy as np
import tensorflow as tf

sys.path.append(str(Path(__file__).resolve().parent.parent))
from algorithm.sac_base import SAC_Base


class SAC_DS_Base(SAC_Base):
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
                 gamma=0.99):

        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.burn_in_step = burn_in_step
        self.n_step = n_step
        self.use_rnn = use_rnn

        self.save_model_per_step = save_model_per_step
        self.write_summary_per_step = write_summary_per_step
        self.tau = tau
        self.update_target_per_step = update_target_per_step
        self.use_auto_alpha = use_auto_alpha
        self.gamma = gamma
        self.use_priority = True
        self.use_n_step_is = True

        self.zero_init_states = False
        self.use_prediction = True

        if seed is not None:
            tf.random.set_seed(seed)

        self._build_model(lr, init_log_alpha, model)

        self.summary_writer = None
        if model_root_path is not None:
            self._init_or_restore(model_root_path)

            summary_path = f'{model_root_path}/log'
            self.summary_writer = tf.summary.create_file_writer(summary_path)

    def get_policy_variables(self):
        variables = self.model_policy.trainable_variables

        return [v.numpy().tolist() for v in variables]

    @tf.function
    def update_policy_variables(self, policy_variables):
        for v, n_v in zip(self.model_policy.trainable_variables, policy_variables):
            v.assign(n_v)

    def save_model(self):
        global_step = self.global_step + self.init_iteration
        if global_step % self.save_model_per_step == 0:
            self.ckpt_manager.save(global_step)

    def train(self, n_states, n_actions, n_rewards, state_, done,
              mu_n_probs,
              priority_is,
              rnn_state=None):

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

        # update mu_n_probs
        if self.use_rnn:
            pi_n_probs = self.get_n_step_probs(n_states, n_actions,
                                               rnn_state).numpy()
        else:
            pi_n_probs = self.get_n_step_probs(n_states, n_actions).numpy()

        self.save_model()

        return td_error.flatten(), pi_n_probs

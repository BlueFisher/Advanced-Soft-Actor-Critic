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
                 obs_dim,
                 action_dim,
                 model_root_path,  # None in actor
                 model,
                 train_mode=True,

                 burn_in_step=0,
                 n_step=1,
                 use_rnn=False,

                 seed=None,
                 write_summary_per_step=20,
                 tau=0.005,
                 update_target_per_step=1,
                 init_log_alpha=-2.3,
                 use_auto_alpha=True,
                 learning_rate=3e-4,
                 gamma=0.99,
                 _lambda=0.9,
                 use_prediction=False,
                 use_reward_normalization=False):

        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.train_mode = train_mode

        self.burn_in_step = burn_in_step
        self.n_step = n_step
        self.use_rnn = use_rnn

        self.write_summary_per_step = write_summary_per_step
        self.tau = tau
        self.update_target_per_step = update_target_per_step
        self.use_auto_alpha = use_auto_alpha
        self.gamma = gamma
        self._lambda = _lambda
        self.use_prediction = use_prediction
        self.use_reward_normalization = use_reward_normalization
        self.use_priority = True
        self.use_n_step_is = True

        if seed is not None:
            tf.random.set_seed(seed)

        self._build_model(model, init_log_alpha learning_rate)
        if model_root_path is not None:
            self._init_or_restore(model_root_path)

        if train_mode:
            summary_path = f'{model_root_path}/log'
            self.summary_writer = tf.summary.create_file_writer(summary_path)

        self._init_tf_function()

    # For learner to send variables to actors
    @tf.function
    def get_policy_variables(self):
        variables = self.model_policy.trainable_variables
        if self.use_rnn:
            variables += self.model_rnn.trainable_variables

        return variables

    # For actor to update its own network from learner
    @tf.function
    def update_policy_variables(self, policy_variables):
        variables = self.model_policy.trainable_variables
        if self.use_rnn:
            variables += self.model_rnn.trainable_variables

        for v, n_v in zip(variables, policy_variables):
            v.assign(n_v)

    def train(self, pointers,
              n_obses,
              n_actions,
              n_rewards,
              obs_,
              n_dones,
              n_mu_probs,
              priority_is,
              rnn_state=None):

        self._train(n_obses=n_obses,
                    n_actions=n_actions,
                    n_rewards=n_rewards,
                    obs_=obs_,
                    n_dones=n_dones,
                    n_mu_probs=n_mu_probs,
                    priority_is=priority_is,
                    initial_rnn_state=rnn_state if self.use_rnn else None)

        if self.use_rnn:
            n_pi_probs = self.get_rnn_n_step_probs(n_obses, n_actions,
                                                   rnn_state).numpy()
            td_error = self.get_td_error(n_obses, n_actions, n_rewards, obs_, n_dones,
                                         n_pi_probs, rnn_state).numpy()
        else:
            n_pi_probs = self.get_n_step_probs(n_obses, n_actions).numpy()
            td_error = self.get_td_error(n_obses, n_actions, n_rewards, obs_, n_dones,
                                         n_pi_probs).numpy()

        update_data = list()

        pointers_list = [pointers + i for i in range(0, self.burn_in_step + self.n_step)]
        tmp_pointers = np.stack(pointers_list, axis=1).reshape(-1)
        pi_probs = n_pi_probs.reshape(-1, n_pi_probs.shape[-1])
        update_data.append((tmp_pointers, 'mu_prob', pi_probs))

        if self.use_rnn:
            pointers_list = [pointers + i for i in range(1, self.burn_in_step + self.n_step + 1)]
            tmp_pointers = np.stack(pointers_list, axis=1).reshape(-1)
            n_rnn_states = self.get_n_rnn_states(n_obses, rnn_state).numpy()
            rnn_states = n_rnn_states.reshape(-1, n_rnn_states.shape[-1])
            update_data.append((tmp_pointers, 'rnn_state', rnn_states))

        return td_error.flatten(), update_data

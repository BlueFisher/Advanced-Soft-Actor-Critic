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
                 obs_dims,
                 action_dim,
                 is_discrete,
                 model_root_path,  # None in actor
                 model,
                 train_mode=True,

                 seed=None,
                 write_summary_per_step=20,

                 burn_in_step=0,
                 n_step=1,
                 use_rnn=False,

                 tau=0.005,
                 update_target_per_step=1,
                 init_log_alpha=-2.3,
                 use_auto_alpha=True,
                 learning_rate=3e-4,
                 gamma=0.99,
                 _lambda=0.9,
                 clip_epsilon=0.2,

                 use_prediction=False,
                 use_reward_normalization=False,
                 use_curiosity=False,
                 curiosity_strength=1,

                 noise=0.):

        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

        self.obs_dims = obs_dims
        self.action_dim = action_dim
        self.is_discrete = is_discrete
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
        self.clip_epsilon = clip_epsilon

        self.use_prediction = use_prediction
        self.use_reward_normalization = use_reward_normalization
        self.use_curiosity = use_curiosity
        self.curiosity_strength = curiosity_strength
        self.use_priority = True
        self.use_n_step_is = True

        self.noise = noise

        if seed is not None:
            tf.random.set_seed(seed)

        self._build_model(model, init_log_alpha, learning_rate)
        if model_root_path is not None:
            self._init_or_restore(model_root_path)

        if train_mode:
            summary_path = f'{model_root_path}/log'
            self.summary_writer = tf.summary.create_file_writer(summary_path)

        self._init_tf_function()

    @tf.function
    def choose_action(self, obs_list):
        """
        tf.function
        obs_list: list([None, obs_dim_i], ...)
        """
        state = self.model_rep(obs_list)
        policy = self.model_policy(state)
        if self.is_discrete:
            return tf.one_hot(policy.sample(), self.action_dim)
        else:
            action = tf.tanh(policy.sample())
            return tf.clip_by_value(action + tf.random.normal(tf.shape(action), stddev=self.noise), -1., 1.)

    # For learner to send variables to actors
    @tf.function
    def get_policy_variables(self):
        variables = self.model_rep.trainable_variables + self.model_policy.trainable_variables

        return variables

    # For actor to update its own network from learner
    @tf.function
    def update_policy_variables(self, policy_variables):
        variables = self.model_rep.trainable_variables + self.model_policy.trainable_variables

        for v, n_v in zip(variables, policy_variables):
            v.assign(n_v)

    def train(self,
              pointers,
              n_obses_list,
              n_actions,
              n_rewards,
              next_obs_list,
              n_dones,
              n_mu_probs,
              priority_is,
              rnn_state=None):

        summary = self._train(n_obses_list=n_obses_list,
                              n_actions=n_actions,
                              n_rewards=n_rewards,
                              next_obs_list=next_obs_list,
                              n_dones=n_dones,
                              n_mu_probs=n_mu_probs,
                              priority_is=priority_is,
                              initial_rnn_state=rnn_state if self.use_rnn else None)

        if self.summary_writer is not None and (self.global_step - 1) % self.write_summary_per_step == 0:
            with self.summary_writer.as_default():
                for k, v in summary['scalar'].items():
                    tf.summary.scalar(k, v, step=self.global_step)
                for k, v in summary['image'].items():
                    tf.summary.image(k, v, step=self.global_step)

            self.summary_writer.flush()

        n_pi_probs = self.get_n_probs(n_obses_list,
                                      n_actions,
                                      rnn_state=rnn_state if self.use_rnn else None).numpy()

        td_error = self.get_td_error(n_obses_list=n_obses_list,
                                     n_actions=n_actions,
                                     n_rewards=n_rewards,
                                     next_obs_list=next_obs_list,
                                     n_dones=n_dones,
                                     n_mu_probs=n_pi_probs,
                                     rnn_state=rnn_state if self.use_rnn else None).numpy()

        update_data = list()

        pointers_list = [pointers + i for i in range(0, self.burn_in_step + self.n_step)]
        tmp_pointers = np.stack(pointers_list, axis=1).reshape(-1)
        pi_probs = n_pi_probs.reshape(-1)
        update_data.append((tmp_pointers, 'mu_prob', pi_probs))

        if self.use_rnn:
            pointers_list = [pointers + i for i in range(1, self.burn_in_step + self.n_step + 1)]
            tmp_pointers = np.stack(pointers_list, axis=1).reshape(-1)
            n_rnn_states = self.get_n_rnn_states(n_obses_list, rnn_state).numpy()
            rnn_states = n_rnn_states.reshape(-1, n_rnn_states.shape[-1])
            update_data.append((tmp_pointers, 'rnn_state', rnn_states))

        return td_error, update_data

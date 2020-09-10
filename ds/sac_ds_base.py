import logging
import sys
import time
from pathlib import Path

import numpy as np
from numpy.core.fromnumeric import var
import tensorflow as tf

sys.path.append(str(Path(__file__).resolve().parent.parent))
from algorithm.sac_base import SAC_Base

logger = logging.getLogger('sac.base.ds')


class SAC_DS_Base(SAC_Base):
    def __init__(self,
                 obs_dims,
                 action_dim,
                 is_discrete,
                 model_abs_dir,  # None in actor
                 model,
                 train_mode=True,
                 last_ckpt=None,

                 seed=None,
                 write_summary_per_step=1e3,
                 save_model_per_step=1e5,
                 save_model_per_minute=5,

                 burn_in_step=0,
                 n_step=1,
                 use_rnn=False,

                 tau=0.005,
                 update_target_per_step=1,
                 init_log_alpha=-2.3,
                 use_auto_alpha=True,
                 learning_rate=3e-4,
                 gamma=0.99,
                 v_lambda=0.9,
                 v_rho=1.,
                 v_c=1.,
                 clip_epsilon=0.2,

                 use_prediction=False,
                 transition_kl=0.8,
                 use_extra_data=True,
                 use_curiosity=False,
                 curiosity_strength=1,
                 use_rnd=False,
                 rnd_n_sample=10,
                 use_normalization=False,

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

        self.write_summary_per_step = int(write_summary_per_step)
        self.save_model_per_step = int(save_model_per_step)
        self.save_model_per_minute = save_model_per_minute
        self.tau = tau
        self.update_target_per_step = update_target_per_step
        self.use_auto_alpha = use_auto_alpha
        self.gamma = gamma
        self.v_lambda = v_lambda
        self.v_rho = v_rho
        self.v_c = v_c
        self.clip_epsilon = clip_epsilon

        self.use_prediction = use_prediction
        self.transition_kl = transition_kl
        self.use_extra_data = use_extra_data
        self.use_curiosity = use_curiosity
        self.curiosity_strength = curiosity_strength
        self.use_rnd = use_rnd
        self.rnd_n_sample = rnd_n_sample
        self.use_normalization = use_normalization
        self.use_priority = True
        self.use_n_step_is = True

        self.noise = noise

        if seed is not None:
            tf.random.set_seed(seed)

        self._build_model(model, init_log_alpha, learning_rate)
        self._init_or_restore(model_abs_dir, last_ckpt)

        if train_mode:
            summary_path = f'{model_abs_dir}/log'
            self.summary_writer = tf.summary.create_file_writer(summary_path)

        self._init_tf_function()

    @tf.function
    def choose_action(self, obs_list):
        action = super().choose_action(obs_list)
        action = tf.clip_by_value(action + tf.random.normal(tf.shape(action),
                                                            stddev=self.noise), -1., 1.)
        return action

    @tf.function
    def choose_rnn_action(self, obs_list, pre_action, rnn_state):
        action, next_rnn_state = super().choose_rnn_action(obs_list, pre_action, rnn_state)
        action = tf.clip_by_value(action + tf.random.normal(tf.shape(action),
                                                            stddev=self.noise), -1., 1.)
        return action, next_rnn_state

    # For learner to send variables to actors
    # If use @tf.function, the function will return Tensors, not Variables
    def get_policy_variables(self):
        variables = self.model_rep.trainable_variables + self.model_policy.trainable_variables

        return variables

    # For actor to update its own network from learner
    @tf.function
    def update_policy_variables(self, t_variables):
        variables = self.get_policy_variables()

        for v, t_v in zip(variables, t_variables):
            v.assign(tf.cast(t_v, v.dtype))

    # For learner to send variables to evolver
    def get_nn_variables(self):
        variables = self.model_rep.trainable_variables +\
            self.model_target_rep.trainable_variables +\
            self.model_q1.trainable_variables +\
            self.model_target_q1.trainable_variables +\
            self.model_q2.trainable_variables +\
            self.model_target_q2.trainable_variables +\
            self.model_policy.trainable_variables +\
            [self.log_alpha]

        if self.use_prediction:
            variables += self.model_transition.trainable_variables +\
                self.model_reward.trainable_variables +\
                self.model_observation.trainable_variables

        if self.use_curiosity:
            variables += self.model_forward.trainable_variables +\
                self.optimizer_forward.weights[1:]

        if self.use_rnd:
            variables += self.model_rnd.trainable_variables +\
                self.model_target_rnd.trainable_variables +\
                self.optimizer_rnd.weights[1:]

        return variables

    # Update own network from evolver selection
    @tf.function
    def update_nn_variables(self, t_variables):
        variables = self.get_nn_variables()

        for v, t_v in zip(variables, t_variables):
            v.assign(tf.cast(t_v, v.dtype))

    def get_all_variables(self):
        variables = self.get_nn_variables()
        if self.use_normalization:
            variables += [self.normalizer_step] +\
                self.running_means +\
                self.running_variances

        return variables

    @tf.function
    def update_all_variables(self, t_variables):
        variables = self.get_all_variables()

        for v, t_v in zip(variables, t_variables):
            v.assign(t_v)

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

        self._train(n_obses_list=n_obses_list,
                    n_actions=n_actions,
                    n_rewards=n_rewards,
                    next_obs_list=next_obs_list,
                    n_dones=n_dones,
                    n_mu_probs=n_mu_probs,
                    priority_is=priority_is,
                    initial_rnn_state=rnn_state if self.use_rnn else None)

        step = self.global_step.numpy()

        if step % self.save_model_per_step == 0 \
                and (time.time() - self._last_save_time) / 60 >= self.save_model_per_minute:
            self.save_model()
            self._last_save_time = time.time()

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
            n_rnn_states = self.get_n_rnn_states(n_obses_list, n_actions, rnn_state).numpy()
            rnn_states = n_rnn_states.reshape(-1, n_rnn_states.shape[-1])
            update_data.append((tmp_pointers, 'rnn_state', rnn_states))

        self._increase_global_step()

        return step, td_error, update_data

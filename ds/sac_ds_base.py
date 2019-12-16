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
                 model_root_path,  # None in actor
                 model,
                 train_mode=True,

                 burn_in_step=0,
                 n_step=1,
                 use_rnn=False,
                 use_prediction=True,

                 seed=None,
                 save_model_per_step=5000,
                 write_summary_per_step=20,
                 tau=0.005,
                 update_target_per_step=1,
                 init_log_alpha=-2.3,
                 use_auto_alpha=True,
                 lr=3e-4,
                 gamma=0.99,
                 _lambda=0.9,

                 replay_batch_size=None):  # for concrete_function

        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.burn_in_step = burn_in_step
        self.n_step = n_step
        self.use_rnn = use_rnn
        self.use_prediction = use_prediction

        self.save_model_per_step = save_model_per_step
        self.write_summary_per_step = write_summary_per_step
        self.tau = tau
        self.update_target_per_step = update_target_per_step
        self.use_auto_alpha = use_auto_alpha
        self.gamma = gamma
        self._lambda = _lambda
        self.use_priority = True
        self.use_n_step_is = True

        self.zero_init_states = False

        if seed is not None:
            tf.random.set_seed(seed)

        self._build_model(lr, init_log_alpha, model)
        if model_root_path is not None:
            self._init_or_restore(model_root_path)

        if train_mode:
            assert replay_batch_size is not None

            summary_path = f'{model_root_path}/log'
            self.summary_writer = tf.summary.create_file_writer(summary_path)
            self._init_train_concrete_function(replay_batch_size, None)

    @tf.function
    def get_policy_variables(self):
        return self.model_policy.trainable_variables

    # for actor
    @tf.function
    def update_policy_variables(self, policy_variables):
        for v, n_v in zip(self.model_policy.trainable_variables, policy_variables):
            v.assign(n_v)

    def save_model(self):
        global_step = self.global_step.numpy() + self.init_iteration
        if global_step % self.save_model_per_step == 0:
            self.ckpt_manager.save(global_step)

    def train(self, n_states, n_actions, n_rewards, state_, done,
              mu_n_probs,
              priority_is,
              rnn_state=None,
              n_states_for_next_rnn_state_list=None,
              episode_trans=None):

        fn_args = {
            'n_states': n_states,
            'n_actions': n_actions,
            'n_rewards': n_rewards,
            'state_': state_,
            'done': done
        }
        if self.use_n_step_is:
            fn_args['mu_n_probs'] = mu_n_probs
        if self.use_priority:
            fn_args['priority_is'] = priority_is
        if self.use_rnn:
            fn_args['initial_rnn_state'] = rnn_state

            if self.use_rnn and self.use_prediction:
                ep_rnn_state = self.get_next_rnn_state(n_states_for_next_rnn_state_list)
                ep_m_states, ep_n_actions = episode_trans
                fn_args['ep_m_states'] = ep_m_states
                fn_args['ep_n_actions'] = ep_n_actions
                fn_args['ep_rnn_state'] = ep_rnn_state

        self._train_fn(**{k: tf.constant(fn_args[k], dtype=tf.float32) for k in fn_args})

        if self.use_rnn:
            pi_n_probs = self.get_rnn_n_step_probs(n_states, n_actions,
                                                   rnn_state).numpy()
            td_error = self.get_td_error(n_states, n_actions, n_rewards, state_, done,
                                         pi_n_probs if self.use_n_step_is else None,
                                         rnn_state).numpy()
        else:
            pi_n_probs = self.get_n_step_probs(n_states, n_actions).numpy()
            td_error = self.get_td_error(n_states, n_actions, n_rewards, state_, done,
                                         pi_n_probs if self.use_n_step_is else None).numpy()

        self.save_model()

        return td_error.flatten(), pi_n_probs

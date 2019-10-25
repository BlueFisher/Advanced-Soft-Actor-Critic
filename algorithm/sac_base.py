import functools
import logging
import time
import sys

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

logger = logging.getLogger('sac.base')
logger.setLevel(level=logging.INFO)

initializer_helper = {
    'kernel_initializer': tf.keras.initializers.TruncatedNormal(0, .1),
    'bias_initializer': tf.keras.initializers.Constant(0.1)
}


class SAC_Base(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 model_root_path,
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
                 lr=3e-4,
                 gamma=0.99,
                 use_priority=False,
                 use_n_step_is=True,

                 replay_config=None):
        """
        state_dim: dimension of state
        action_dim: dimension of action
        model_root_path: the path that saves summary, checkpoints, config etc.
        model: custom Model Class
        """

        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.burn_in_step = burn_in_step
        self.n_step = n_step
        self.use_rnn = use_rnn

        self.write_summary_per_step = write_summary_per_step
        self.tau = tau
        self.update_target_per_step = update_target_per_step
        self.use_auto_alpha = use_auto_alpha
        self.gamma = gamma
        self.use_priority = use_priority
        self.use_n_step_is = use_n_step_is

        self.zero_init_states = False
        self.use_prediction = True

        if seed is not None:
            tf.random.set_seed(seed)

        self._build_model(lr, init_log_alpha, model)
        self._init_or_restore(model_root_path)

        if train_mode:
            summary_path = f'{model_root_path}/log'
            self.summary_writer = tf.summary.create_file_writer(summary_path)

            replay_config = {} if replay_config is None else replay_config
            if self.use_priority:
                self.replay_buffer = PrioritizedReplayBuffer(**replay_config)
            else:
                self.replay_buffer = ReplayBuffer(**replay_config)

    def _build_model(self, lr, init_log_alpha, model):
        """
        Initialize variables, network models and optimizers
        """
        self.log_alpha = tf.Variable(init_log_alpha, dtype=tf.float32, name='log_alpha')
        self.global_step = tf.Variable(0, dtype=tf.int64, name='global_step')

        self.optimizer_q = tf.keras.optimizers.Adam(lr)
        self.optimizer_policy = tf.keras.optimizers.Adam(lr)
        if self.use_auto_alpha:
            self.optimizer_alpha = tf.keras.optimizers.Adam(lr)

        if self.use_rnn:
            self.model_rnn = model.ModelRNN(self.state_dim)
            self.model_target_rnn = model.ModelRNN(self.state_dim)

            tmp_state, tmp_rnn_state = self.model_rnn.get_call_result_tensors()
            state_dim = tmp_state.shape[-1]
            self._initial_rnn_state = np.zeros(tmp_rnn_state.shape[-1], dtype=np.float32)

            if self.use_prediction:
                self.model_prediction = model.ModelPrediction(self.state_dim, state_dim, self.action_dim)
        else:
            state_dim = self.state_dim

        self.model_q1 = model.ModelQ(state_dim, self.action_dim)
        self.model_target_q1 = model.ModelQ(state_dim, self.action_dim)
        self.model_q2 = model.ModelQ(state_dim, self.action_dim)
        self.model_target_q2 = model.ModelQ(state_dim, self.action_dim)
        self.model_policy = model.ModelPolicy(state_dim, self.action_dim)

    def _init_or_restore(self, model_path):
        """
        Initialize network weights from scratch or restore from model_root_path
        """
        ckpt = tf.train.Checkpoint(log_alpha=self.log_alpha,
                                   global_step=self.global_step,

                                   optimizer_q=self.optimizer_q,
                                   optimizer_policy=self.optimizer_policy,

                                   model_q1=self.model_q1,
                                   model_target_q1=self.model_target_q1,
                                   model_q2=self.model_q2,
                                   model_target_q2=self.model_target_q2,
                                   model_policy=self.model_policy)
        if self.use_rnn:
            ckpt.model_rnn = self.model_rnn
            ckpt.model_target_rnn = self.model_target_rnn
            if self.use_prediction:
                ckpt.model_prediction = self.model_prediction

        if self.use_auto_alpha:
            ckpt.optimizer_alpha = self.optimizer_alpha

        self.ckpt_manager = tf.train.CheckpointManager(ckpt, f'{model_path}/model', max_to_keep=10)

        ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            logger.info(f'Restored from {self.ckpt_manager.latest_checkpoint}')
            self.init_iteration = int(self.ckpt_manager.latest_checkpoint.split('-')[1].split('.')[0])
        else:
            logger.info('Initializing from scratch')
            self.init_iteration = 0
            self._update_target_variables()

    def get_initial_rnn_state(self, batch_size):
        assert self.use_rnn

        initial_rnn_state = np.repeat([self._initial_rnn_state], batch_size, axis=0)
        return initial_rnn_state

    @tf.function
    def _update_target_variables(self, tau=1.):
        """
        soft update target networks (default hard)
        """
        target_variables = self.model_target_q1.trainable_variables + self.model_target_q2.trainable_variables
        eval_variables = self.model_q1.trainable_variables + self.model_q2.trainable_variables
        if self.use_rnn:
            target_variables += self.model_target_rnn.trainable_variables
            eval_variables += self.model_rnn.trainable_variables

        [t.assign(tau * e + (1. - tau) * t) for t, e in zip(target_variables, eval_variables)]

    @tf.function
    def _get_y(self, reward, state_, done):
        """
        tf.function
        get target value
        """
        alpha = tf.exp(self.log_alpha)

        next_policy = self.model_policy(state_)
        next_action_sampled = next_policy.sample()
        next_action_log_prob = tf.reduce_prod(next_policy.log_prob(next_action_sampled), axis=1, keepdims=True)

        target_q1 = self.model_target_q1(state_, next_action_sampled)
        target_q2 = self.model_target_q2(state_, next_action_sampled)

        y = reward + self.gamma**self.n_step * (1. - done) * (tf.minimum(target_q1, target_q2) - alpha * next_action_log_prob)

        return y

    @tf.function
    def _train(self, n_states, n_actions, reward, state_, done,
               n_step_is=None, priority_is=None,
               initial_rnn_state=None):

        with tf.GradientTape(persistent=True) as tape:
            if self.use_rnn:
                m_states = tf.concat([n_states, tf.reshape(state_, (-1, 1, state_.shape[-1]))], axis=1)
                encoded_m_target_states, *_ = self.model_target_rnn(m_states, initial_rnn_state)
                state_ = encoded_m_target_states[:, -1, :]

                encoded_n_states, *_ = self.model_rnn(n_states, initial_rnn_state)
                state = encoded_n_states[:, self.burn_in_step, :]

                if self.use_prediction:
                    approx_next_states = self.model_prediction(encoded_n_states[:, self.burn_in_step:, :],
                                                               n_actions[:, self.burn_in_step:, :])
                    loss_prediction = tf.math.squared_difference(approx_next_states,
                                                                 m_states[:, self.burn_in_step + 1:, :])
            else:
                state = n_states[:, self.burn_in_step, :]

            action = n_actions[:, self.burn_in_step, :]

            alpha = tf.exp(self.log_alpha)

            policy = self.model_policy(state)
            action_sampled = policy.sample()

            q1 = self.model_q1(state, action)
            q1_for_gradient = self.model_q1(state, action_sampled)
            q2 = self.model_q2(state, action)

            y = tf.stop_gradient(self._get_y(reward, state_, done))

            loss_q1 = tf.math.squared_difference(q1, y)
            loss_q2 = tf.math.squared_difference(q2, y)

            loss_q = loss_q1 + loss_q2

            if self.use_n_step_is and n_step_is is not None:
                loss_q *= n_step_is

            if self.use_priority and priority_is is not None:
                loss_q *= priority_is

            if self.use_rnn and self.use_prediction:
                loss_prediction = tf.reshape(tf.reduce_mean(loss_prediction, axis=[1, 2]), [-1, 1])
                loss_q += loss_prediction

            loss_policy = alpha * policy.log_prob(action_sampled) - q1_for_gradient
            loss_alpha = -self.log_alpha * policy.log_prob(action_sampled) + self.log_alpha * self.action_dim

        q_variables = self.model_q1.trainable_variables + self.model_q2.trainable_variables
        if self.use_rnn:
            q_variables = q_variables + self.model_rnn.trainable_variables
            if self.use_prediction:
                q_variables = q_variables + self.model_prediction.trainable_variables

        grads_q = tape.gradient(loss_q, q_variables)
        self.optimizer_q.apply_gradients(zip(grads_q, q_variables))

        grads_policy = tape.gradient(loss_policy, self.model_policy.trainable_variables)
        self.optimizer_policy.apply_gradients(zip(grads_policy, self.model_policy.trainable_variables))

        if self.use_auto_alpha:
            grads_alpha = tape.gradient(loss_alpha, self.log_alpha)
            self.optimizer_alpha.apply_gradients([(grads_alpha, self.log_alpha)])

        del tape

        if self.summary_writer is not None and self.global_step % self.write_summary_per_step == 0:
            with self.summary_writer.as_default():
                tf.summary.scalar('loss/Q', tf.reduce_mean(loss_q), step=self.global_step)
                if self.use_rnn and self.use_prediction:
                    tf.summary.scalar('loss/prediction', tf.reduce_mean(loss_prediction), step=self.global_step)
                tf.summary.scalar('loss/policy', tf.reduce_mean(loss_policy), step=self.global_step)
                tf.summary.scalar('loss/entropy', tf.reduce_mean(policy.entropy()), step=self.global_step)
                tf.summary.scalar('loss/alpha', alpha, step=self.global_step)

        if self.global_step % self.update_target_per_step == 0:
            self._update_target_variables(tau=self.tau)

        self.global_step.assign_add(1)

    @tf.function()
    def choose_action(self, state):
        """
        tf.function
        """
        policy = self.model_policy(state)
        return tf.clip_by_value(policy.sample(), -1, 1)

    @tf.function
    def choose_rnn_action(self, state, rnn_state):
        """
        tf.function
        """
        state = tf.reshape(state, (-1, 1, state.shape[-1]))
        encoded_state, next_rnn_state, = self.model_rnn(state, rnn_state)
        policy = self.model_policy(encoded_state)
        action = policy.sample()
        action = tf.reshape(action, (-1, action.shape[-1]))
        return tf.clip_by_value(action, -1, 1), next_rnn_state

    @tf.function
    def get_td_error(self, n_states, n_actions, reward, state_, done,
                     rnn_state=None):
        """
        tf.function
        """
        if self.use_rnn:
            m_states = tf.concat([n_states, tf.reshape(state_, (-1, 1, state_.shape[-1]))], axis=1)
            m_states, *_ = self.model_target_rnn(m_states,
                                                 rnn_state)
            state_ = m_states[:, -1, :]

            n_states, *_ = self.model_rnn(n_states,
                                          rnn_state)

        state = n_states[:, self.burn_in_step, :]
        action = n_actions[:, self.burn_in_step, :]

        q1 = self.model_q1(state, action)
        q2 = self.model_q2(state, action)
        y = self._get_y(reward, state_, done)

        q1_td_error = tf.abs(q1 - y)
        q2_td_error = tf.abs(q2 - y)
        td_error = tf.reduce_mean(tf.concat([q1_td_error, q2_td_error], axis=1),
                                  axis=1, keepdims=True)

        return td_error

    @tf.function
    def get_n_step_probs(self, n_states, n_actions,
                         rnn_state=None):
        """
        tf.function
        """
        if self.use_rnn:
            n_states, *_ = self.model_rnn(n_states,
                                          rnn_state)
        policy = self.model_policy(n_states)
        policy_prob = policy.prob(n_actions)

        return policy_prob

    def save_model(self, iteration):
        self.ckpt_manager.save(iteration + self.init_iteration)

    def write_constant_summaries(self, constant_summaries, iteration):
        with self.summary_writer.as_default():
            for s in constant_summaries:
                tf.summary.scalar(s['tag'], s['simple_value'], step=iteration + self.init_iteration)
        self.summary_writer.flush()

    def _get_n_step_is(self, pi_n_probs, mu_n_probs):
        """
        get n-step importance sampling ratio
        pi_n_probs, mu_n_probs: [None, n_step, length of state space]
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            tmp_is = np.true_divide(pi_n_probs, mu_n_probs)
            tmp_is[~np.isfinite(tmp_is)] = 1.
            tmp_is = np.clip(tmp_is, 0, 1.)
            n_step_is = np.prod(tmp_is, axis=(1, 2)).reshape(-1, 1)
        return n_step_is  # [None, 1]

    def _get_n_reward_gamma_sum(self, n_rewards):
        # [None, n_step]
        assert n_rewards.shape[1] == self.n_step

        n_gamma = np.array([self.gamma**i for i in range(self.n_step)], dtype=np.float32)
        n_rewards_gamma = n_rewards * n_gamma
        return n_rewards_gamma.sum(axis=1, keepdims=True)  # [None, 1]

    def fill_replay_buffer(self, n_states, n_actions, n_rewards, state_, done,
                           rnn_state=None):
        """
        n_states: [None, burn_in_step + n_step, state_dim]
        n_states: [None, burn_in_step + n_step, state_dim]
        n_rewards: [None, burn_in_step + n_step]
        state_: [None, state_dim]
        done: [None, 1]
        """
        assert len(n_states) == len(n_actions) == len(n_rewards) == len(state_) == len(done)
        if not self.use_rnn:
            assert rnn_state is None

        if self.use_rnn:
            if self.zero_init_states:
                rnn_state = self.get_initial_rnn_state(len(n_states))
            mu_n_probs = self.get_n_step_probs(n_states, n_actions,  # TODO: only need [:, burn_in_step:, :]
                                               rnn_state).numpy()
            self.replay_buffer.add(n_states, n_actions, n_rewards, state_, done, mu_n_probs,
                                   rnn_state)
        else:
            mu_n_probs = self.get_n_step_probs(n_states, n_actions).numpy()
            self.replay_buffer.add(n_states, n_actions, n_rewards, state_, done, mu_n_probs)

    def train(self):
        # sample from replay buffer
        sampled = self.replay_buffer.sample()
        if sampled is None:
            return

        if self.use_priority:
            pointers, trans, priority_is = sampled
        else:
            pointers, trans = sampled

        if self.use_rnn:
            n_states, n_actions, n_rewards, state_, done, mu_n_probs, rnn_state = trans
        else:
            n_states, n_actions, n_rewards, state_, done, mu_n_probs = trans

        reward = self._get_n_reward_gamma_sum(n_rewards[:, self.burn_in_step:])

        if self.use_n_step_is:
            if self.use_rnn:
                pi_n_probs = self.get_n_step_probs(n_states, n_actions,  # TODO: only need [:, burn_in_step:, :]
                                                   rnn_state).numpy()
            else:
                pi_n_probs = self.get_n_step_probs(n_states, n_actions).numpy()

            n_step_is = self._get_n_step_is(pi_n_probs[:, self.burn_in_step:, :],
                                            mu_n_probs[:, self.burn_in_step:, :])

        self._train(n_states, n_actions, reward, state_, done,
                    n_step_is=n_step_is if self.use_n_step_is else None,
                    priority_is=priority_is if self.use_priority else None,
                    initial_rnn_state=rnn_state if self.use_rnn else None)

        self.summary_writer.flush()

        # update td_error
        if self.use_priority:
            if self.use_rnn:
                td_error = self.get_td_error(n_states, n_actions, reward, state_, done,
                                             rnn_state).numpy()
            else:
                td_error = self.get_td_error(n_states, n_actions, reward, state_, done).numpy()

            self.replay_buffer.update(pointers, td_error.flatten())

        # update mu_n_probs
        if self.use_n_step_is:
            if self.use_rnn:
                pi_n_probs = self.get_n_step_probs(n_states, n_actions,
                                                   rnn_state).numpy()
            else:
                pi_n_probs = self.get_n_step_probs(n_states, n_actions).numpy()

            self.replay_buffer.update_transitions(pointers, 5, pi_n_probs)

import functools
import logging
import time
import sys

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

# logger = logging.getLogger('sac.base')
# logger.propagate = False
# fh = logging.handlers.RotatingFileHandler('test.txt', maxBytes=1024 * 10240, backupCount=100000)
# fh.setLevel(logging.INFO)

# # add handler and formatter to logger
# fh.setFormatter(logging.Formatter('%(asctime)-15s [%(levelname)s] - [%(name)s] - %(message)s'))
# logger.addHandler(fh)

logger = logging.getLogger('sac')
logger.setLevel(level=logging.INFO)


class SAC_Base(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 ModelQ,
                 ModelPolicy,
                 model_root_path,
                 use_rnn=False,

                 write_summary_graph=False,
                 seed=None,
                 tau=0.005,
                 write_summary_per_step=20,
                 update_target_per_step=1,
                 init_log_alpha=-2.3,
                 use_auto_alpha=True,
                 lr=3e-4,
                 gamma=0.99,
                 burn_in_step=0,
                 n_step=1,
                 use_priority=False,
                 use_n_step_is=True,

                 replay_config=None):

        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

        self.tau = tau
        self.use_auto_alpha = use_auto_alpha
        self.gamma = gamma
        self.n_step = n_step
        self.burn_in_step = burn_in_step
        self.use_priority = use_priority
        self.use_n_step_is = use_n_step_is
        self.use_rnn = use_rnn

        replay_config = {} if replay_config is None else replay_config
        if self.use_priority:
            self.replay_buffer = PrioritizedReplayBuffer(**replay_config)
        else:
            self.replay_buffer = ReplayBuffer(**replay_config)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.write_summary_per_step = write_summary_per_step
        self.update_target_per_step = update_target_per_step

        if seed is not None:
            tf.random.set_seed(seed)

        self._build_model(ModelQ, ModelPolicy, lr, init_log_alpha)
        self._init_or_restore(model_root_path)

        summary_path = f'{model_root_path}/log'
        self.summary_writer = tf.summary.create_file_writer(summary_path)

    def _build_model(self, ModelQ, ModelPolicy, lr, init_log_alpha):
        self.log_alpha = tf.Variable(init_log_alpha, dtype=tf.float32, name='log_alpha')
        self.global_step = tf.Variable(0, dtype=tf.int64, name='global_step')

        self.optimizer_q1 = tf.keras.optimizers.Adam(lr)
        self.optimizer_q2 = tf.keras.optimizers.Adam(lr)
        self.optimizer_policy = tf.keras.optimizers.Adam(lr)
        self.optimizer_alpha = tf.keras.optimizers.Adam(lr)

        self.model_q1 = ModelQ(self.state_dim, self.action_dim)
        self.model_target_q1 = ModelQ(self.state_dim, self.action_dim)
        self.model_q2 = ModelQ(self.state_dim, self.action_dim)
        self.model_target_q2 = ModelQ(self.state_dim, self.action_dim)
        self.model_policy = ModelPolicy(self.state_dim, self.action_dim)

    def _init_or_restore(self, model_path):
        ckpt = tf.train.Checkpoint(log_alpha=self.log_alpha,
                                   global_step=self.global_step,

                                   optimizer_q1=self.optimizer_q1,
                                   optimizer_q2=self.optimizer_q2,
                                   optimizer_policy=self.optimizer_policy,
                                   optimizer_alpha=self.optimizer_alpha,

                                   model_q1=self.model_q1,
                                   model_target_q1=self.model_target_q1,
                                   model_q2=self.model_q2,
                                   model_target_q2=self.model_target_q2,
                                   model_policy=self.model_policy)

        self.ckpt_manager = tf.train.CheckpointManager(ckpt, f'{model_path}/model', max_to_keep=10)

        ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            logger.info(f'Restored from {self.ckpt_manager.latest_checkpoint}')
            self.init_iteration = int(self.ckpt_manager.latest_checkpoint.split('-')[1].split('.')[0])
        else:
            logger.info('Initializing from scratch')
            self.init_iteration = 0
            self._update_target_variables()

    @tf.function
    def _update_target_variables(self, tau=1.):
        target_variables = self.model_target_q1.trainable_variables + self.model_target_q2.trainable_variables
        eval_variables = self.model_q1.trainable_variables + self.model_q2.trainable_variables
        [t.assign(tau * e + (1. - tau) * t) for t, e in zip(target_variables, eval_variables)]

    @tf.function
    def _get_y(self, reward, state_, done):
        alpha = tf.exp(self.log_alpha)

        next_policy = tfp.distributions.Normal(*self.model_policy(state_))
        next_action_sampled = next_policy.sample()
        next_action_log_prob = tf.reduce_prod(next_policy.log_prob(next_action_sampled), axis=1, keepdims=True)

        target_q1 = self.model_target_q1(state_, next_action_sampled)
        target_q2 = self.model_target_q2(state_, next_action_sampled)

        y = reward + self.gamma**self.n_step * (1. - done) * (tf.minimum(target_q1, target_q2) - alpha * next_action_log_prob)
        y = tf.stop_gradient(y)

        return y

    @tf.function
    def _train(self, n_states, n_actions, reward, state_, done, n_step_is=None, priority_is=None):
        state = n_states[:, self.burn_in_step, :]
        action = n_actions[:, self.burn_in_step, :]

        with tf.GradientTape(persistent=True) as tape:
            alpha = tf.exp(self.log_alpha)

            policy = tfp.distributions.Normal(*self.model_policy(state))
            action_sampled = policy.sample()

            q1 = self.model_q1(state, action)
            q1_for_gradient = self.model_q1(state, action_sampled)
            q2 = self.model_q2(state, action)

            y = self._get_y(reward, state_, done)

            loss_q1 = tf.math.squared_difference(q1, y)
            loss_q2 = tf.math.squared_difference(q2, y)

            if self.use_n_step_is and n_step_is is not None:
                loss_q1 *= n_step_is
                loss_q2 *= n_step_is

            if self.use_priority and priority_is is not None:
                loss_q1 *= priority_is
                loss_q2 *= priority_is

            loss_policy = tf.reduce_mean(alpha * policy.log_prob(action_sampled) - q1_for_gradient)
            loss_alpha = -self.log_alpha * policy.log_prob(action_sampled) + self.log_alpha * self.action_dim

        if self.global_step % self.write_summary_per_step == 0:
            with self.summary_writer.as_default():
                tf.summary.scalar('loss/Q1', tf.reduce_mean(loss_q1), step=self.global_step)
                tf.summary.scalar('loss/Q2', tf.reduce_mean(loss_q2), step=self.global_step)
                tf.summary.scalar('loss/policy', loss_policy, step=self.global_step)
                tf.summary.scalar('loss/entropy', tf.reduce_mean(policy.entropy()), step=self.global_step)
                tf.summary.scalar('loss/alpha', alpha, step=self.global_step)

        grads_q1 = tape.gradient(loss_q1, self.model_q1.trainable_variables)
        grads_q2 = tape.gradient(loss_q2, self.model_q2.trainable_variables)
        grads_policy = tape.gradient(loss_policy, self.model_policy.trainable_variables)
        grads_alpha = tape.gradient(loss_alpha, self.log_alpha)

        self.optimizer_q1.apply_gradients(zip(grads_q1, self.model_q1.trainable_variables))
        self.optimizer_q2.apply_gradients(zip(grads_q2, self.model_q2.trainable_variables))
        self.optimizer_policy.apply_gradients(zip(grads_policy, self.model_policy.trainable_variables))
        self.optimizer_alpha.apply_gradients([(grads_alpha, self.log_alpha)])

        del tape

        if self.global_step % self.update_target_per_step == 0:
            self._update_target_variables(tau=self.tau)

        self.global_step.assign_add(1)

    @tf.function
    def choose_action(self, state):
        assert len(state.shape) == 2
        assert not self.use_rnn

        policy = tfp.distributions.Normal(*self.model_policy(state))

        return tf.clip_by_value(policy.sample(), -1, 1)

    @tf.function
    def get_td_error(self, state, action, reward, state_, done):
        if len(state.shape) == 3:
            state = state[:, self.burn_in_step, :]
        if len(action.shape) == 3:
            action = action[:, self.burn_in_step, :]

        q1 = self.model_q1(state, action)
        q2 = self.model_q2(state, action)
        y = self._get_y(reward, state_, done)

        q1_td_error = tf.abs(q1 - y)
        q2_td_error = tf.abs(q2 - y)
        td_error = tf.reduce_mean(tf.concat([q1_td_error, q2_td_error], axis=1),
                                  axis=1, keepdims=True)

        return td_error

    @tf.function
    def get_n_step_probs(self, n_states, n_actions):
        policy = tfp.distributions.Normal(*self.model_policy(n_states))
        policy_prob = policy.prob(n_actions)

        return policy_prob

    def _get_lstm_state_tuple(self, lstm_state_c, lstm_state_h):
        return tf.nn.rnn_cell.LSTMStateTuple(c=np.asarray(lstm_state_c),
                                             h=np.asarray(lstm_state_h))

    def save_model(self, iteration):
        self.ckpt_manager.save(iteration + self.init_iteration)

    def write_constant_summaries(self, constant_summaries, iteration):
        with self.summary_writer.as_default():
            for s in constant_summaries:
                tf.summary.scalar(s['tag'], s['simple_value'], step=iteration + self.init_iteration)
            self.summary_writer.flush()

    def _get_n_step_is(self, pi_n_probs, mu_n_probs):
        """
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
                           lstm_state_c=None, lstm_state_h=None):
        """
        n_states: [None, burn_in_step + n_step, state_dim]
        n_states: [None, burn_in_step + n_step, state_dim]
        n_rewards: [None, burn_in_step + n_step]
        state_: [None, state_dim]
        done: [None, 1]
        """
        assert len(n_states) == len(n_actions) == len(n_rewards) == len(state_) == len(done)
        if not self.use_rnn:
            assert lstm_state_c is None
            assert lstm_state_h is None

        if self.use_rnn:
            # lstm_state_before_burn_in = self.get_initial_lstm_state(len(n_states))
            lstm_state_before_burn_in = self._get_lstm_state_tuple(lstm_state_c, lstm_state_h)
            mu_n_probs = self.get_n_step_lstm_probs(lstm_state_before_burn_in, n_states, n_actions)  # TODO: only need [:, burn_in_step:, :]
            self.replay_buffer.add(n_states, n_actions, n_rewards, state_, done, mu_n_probs,
                                   lstm_state_c, lstm_state_h)
        else:
            mu_n_probs = self.get_n_step_probs(n_states, n_actions).numpy()
            self.replay_buffer.add(n_states, n_actions, n_rewards, state_, done, mu_n_probs)

    def train(self):
        # sample from replay buffer
        sampled = self.replay_buffer.sample()
        if sampled is None:
            return

        if self.use_priority:
            points, trans, priority_is = sampled
        else:
            points, trans = sampled

        n_states, n_actions, n_rewards, state_, done, mu_n_probs = trans
        reward = self._get_n_reward_gamma_sum(n_rewards[:, self.burn_in_step:])

        if self.use_n_step_is:
            pi_n_probs = self.get_n_step_probs(n_states, n_actions).numpy()

            n_step_is = self._get_n_step_is(pi_n_probs[:, self.burn_in_step:, :],
                                            mu_n_probs[:, self.burn_in_step:, :])

        self._train(n_states, n_actions, reward, state_, done,
                    n_step_is=n_step_is if self.use_n_step_is else None,
                    priority_is=priority_is if self.use_priority else None)

        self.summary_writer.flush()

        # update td_error
        if self.use_priority:
            if self.use_rnn:
                td_error = self.get_lstm_td_error(lstm_state_before_burn_in, n_states, n_actions, reward, state_, done)
            else:
                td_error = self.get_td_error(n_states, n_actions, reward, state_, done)

            self.replay_buffer.update(points, td_error.numpy().flatten())

        # update mu_n_probs
        if self.use_n_step_is:
            if self.use_rnn:
                pi_n_probs = self.get_n_step_lstm_probs(lstm_state_before_burn_in, n_states, n_actions)
            else:
                pi_n_probs = self.get_n_step_probs(n_states, n_actions).numpy()

            self.replay_buffer.update_transitions(5, points, pi_n_probs)

    def dispose(self):
        # self.sess.close()
        pass

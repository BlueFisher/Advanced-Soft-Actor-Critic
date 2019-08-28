import functools
import time
import sys

import numpy as np
import tensorflow as tf

from .saver import Saver
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class SAC_Base(object):
    def __init__(self,
                 state_dim,
                 action_dim,
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
                 use_priority=False,
                 use_n_step_is=True,

                 replay_config=None):
        self.graph = tf.Graph()
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options),
                               graph=self.graph)

        self.use_auto_alpha = use_auto_alpha
        self.use_priority = use_priority
        self.use_n_step_is = use_n_step_is
        self.use_rnn = use_rnn

        replay_config = {} if replay_config is None else replay_config
        if self.use_priority:
            self.replay_buffer = PrioritizedReplayBuffer(**replay_config)
        else:
            self.replay_buffer = ReplayBuffer(**replay_config)

        self.s_dim = state_dim
        self.a_dim = action_dim

        self.write_summary_per_step = write_summary_per_step
        self.update_target_per_step = update_target_per_step

        with self.graph.as_default():
            if seed is not None:
                tf.random.set_random_seed(seed)

            self._build_model(tau, lr, init_log_alpha)

            self.saver = Saver(f'{model_root_path}/model', self.sess)
            self.init_iteration = self.saver.restore_or_init()

            summary_path = f'{model_root_path}/log'
            if write_summary_graph:
                writer = tf.summary.FileWriter(summary_path, self.graph)
                writer.close()
            self.summary_writer = tf.summary.FileWriter(summary_path)

            self.sess.run(self.update_target_hard_op)

    def _build_model(self, tau, lr, init_log_alpha):
        self.pl_s = tf.placeholder(tf.float32, shape=(None, self.s_dim), name='state')
        self.pl_a = tf.placeholder(tf.float32, shape=(None, self.a_dim), name='action')

        self.pl_n_step_s = tf.placeholder(tf.float32, shape=(None, None, self.s_dim), name='n_step_state')
        self.pl_n_step_a = tf.placeholder(tf.float32, shape=(None, None, self.a_dim), name='n_step_action')

        self.pl_r = tf.placeholder(tf.float32, shape=(None, 1), name='reward')
        self.pl_s_ = tf.placeholder(tf.float32, shape=(None, self.s_dim), name='state_')
        self.pl_done = tf.placeholder(tf.float32, shape=(None, 1), name='done')
        self.pl_gamma = tf.placeholder(tf.float32, shape=(None, 1), name='gamma')

        self.pl_n_step_is = tf.placeholder(tf.float32, shape=(None, 1), name='n_step_is')
        self.pl_priority_is = tf.placeholder(tf.float32, shape=(None, 1), name='priority_is')

        log_alpha = tf.get_variable('alpha', shape=(), initializer=tf.constant_initializer(init_log_alpha))
        alpha = tf.exp(log_alpha)

        if self.use_rnn:
            self.pl_batch_size = tf.placeholder(tf.int32, shape=(), name='batch_size')

            encoded_pl_s, self.initial_lstm_state, self.lstm_state, lstm_variables = self._build_lstm_s_input(self.pl_s, 'lstm_state')
            encoded_pl_s_, self.next_initial_lstm_state, self.next_lstm_state, _ = self._build_lstm_s_input(self.pl_s_, 'lstm_state', reuse=True)
            encoded_pl_n_step_s, self.n_step_initial_lstm_state, self.n_step_lstm_state, _ = self._build_lstm_s_input(self.pl_n_step_s, 'lstm_state', reuse=True)

            pl_s = encoded_pl_s
            pl_s_ = encoded_pl_s_
            pl_n_step_s = encoded_pl_n_step_s
        else:
            pl_s = self.pl_s
            pl_s_ = self.pl_s_
            pl_n_step_s = self.pl_n_step_s

        self.policy, self.action_sampled, policy_variables = self._build_policy_net(pl_s, 'policy')
        policy_next, action_next_sampled, _ = self._build_policy_net(pl_s_, 'policy', reuse=True)
        self.policy_prob = self.policy.prob(self.pl_a)

        self.n_step_policy, self.n_step_action_sampled, _ = self._build_policy_net(pl_n_step_s, 'policy', reuse=True)
        self.n_step_policy_prob = self.n_step_policy.prob(self.pl_n_step_a)

        q1, q1_variables = self._build_q_net(pl_s, self.pl_a, 'q1')
        q1_for_gradient, _ = self._build_q_net(pl_s, self.action_sampled, 'q1', reuse=True)
        q1_target, q1_target_variables = self._build_q_net(pl_s_, action_next_sampled, 'q1_target', trainable=False)

        q2, q2_variables = self._build_q_net(pl_s, self.pl_a, 'q2')
        q2_target, q2_target_variables = self._build_q_net(pl_s_, action_next_sampled, 'q2_target', trainable=False)

        y = self.pl_r + self.pl_gamma * (1 - self.pl_done) * (tf.minimum(q1_target, q2_target) - alpha * policy_next.log_prob(action_next_sampled))
        y = tf.stop_gradient(y)

        L_q1 = tf.squared_difference(q1, y)
        L_q2 = tf.squared_difference(q2, y)
        if self.use_n_step_is:
            L_q1 *= self.pl_n_step_is
            L_q2 *= self.pl_n_step_is

        if self.use_priority:
            L_q1 *= self.pl_priority_is
            L_q2 *= self.pl_priority_is

        L_q1 = tf.reduce_mean(L_q1)
        L_q2 = tf.reduce_mean(L_q2)

        q1_td_error = tf.abs(q1 - y)
        q2_td_error = tf.abs(q2 - y)
        self.td_error = tf.reduce_mean(tf.concat([q1_td_error, q2_td_error], axis=1),
                                       axis=1, keepdims=True)

        L_policy = tf.reduce_mean(alpha * self.policy.log_prob(self.action_sampled) - q1_for_gradient)

        entropy = self.policy.entropy()
        L_alpha = -log_alpha * self.policy.log_prob(self.action_sampled) + log_alpha * self.a_dim

        with tf.name_scope('optimizer'):
            self.global_step = tf.get_variable('global_step', shape=(), dtype=tf.int32, initializer=tf.constant_initializer(0), trainable=False)

            self.train_q_ops = [tf.train.AdamOptimizer(lr).minimize(L_q1,
                                                                    var_list=q1_variables + lstm_variables),
                                tf.train.AdamOptimizer(lr).minimize(L_q2,
                                                                    var_list=q2_variables + lstm_variables)]
            self.train_policy_op = tf.train.AdamOptimizer(lr).minimize(L_policy,
                                                                       global_step=self.global_step,
                                                                       var_list=policy_variables)
            self.train_alpha_op = tf.train.AdamOptimizer(lr).minimize(L_alpha,
                                                                      var_list=[log_alpha])

        self.update_target_hard_op = [tf.assign(t, e) for t, e in
                                      zip(q1_target_variables + q2_target_variables, q1_variables + q2_variables)]
        self.update_target_op = [tf.assign(t, tau * e + (1 - tau) * t) for t, e in
                                 zip(q1_target_variables + q2_target_variables, q1_variables + q2_variables)]

        tf.summary.scalar('loss/Q1', L_q1)
        tf.summary.scalar('loss/Q2', L_q2)
        tf.summary.scalar('loss/policy', L_policy)
        tf.summary.scalar('loss/entropy', tf.reduce_mean(entropy))
        tf.summary.scalar('loss/alpha', alpha)
        self.summaries = tf.summary.merge_all()

    def _build_q_net(self, s_input, a_input, scope, trainable=True, reuse=False):
        raise Exception('SAC_Base._build_q_net not implemented')
        # return q, variables

    def _build_policy_net(self, s_inputs, scope, trainable=True, reuse=False):
        raise Exception('SAC_Base._build_policy_net not implemented')
        # return policy, action_sampled, variables

    def choose_action(self, s, lstm_state=None):
        assert len(s.shape) == 2

        if self.use_rnn:
            a, lstm_state = self.sess.run([self.action_sampled, self.lstm_state], {
                self.pl_s: s,
                self.initial_lstm_state: lstm_state,
                self.pl_batch_size: len(s)
            })

            return np.clip(a, -1, 1), lstm_state
        else:
            a = self.sess.run(self.action_sampled, {
                self.pl_s: s,
            })

            return np.clip(a, -1, 1)

    def get_initial_lstm_state(self, batch_size):
        return self.sess.run(self.initial_lstm_state, {
            self.pl_batch_size: batch_size
        })

    def get_n_step_lstm_state_(self, lstm_state, n_step_s):
        n_step_s = np.asarray(n_step_s)
        assert len(n_step_s.shape) == 3

        return self.sess.run(self.n_step_lstm_state, {
            self.initial_lstm_state: lstm_state,
            self.pl_n_step_s: n_step_s,
            self.pl_batch_size: len(n_step_s)
        })

    def get_td_error(self, s, a, r, s_, done, gamma):
        td_error = self.sess.run(self.td_error, {
            self.pl_s: s,
            self.pl_a: a,
            self.pl_r: r,
            self.pl_s_: s_,
            self.pl_done: done,
            self.pl_gamma: gamma
        })

        return td_error

    def get_lstm_td_error(self, lstm_state, s, a, r, lstm_state_, s_, done, gamma):
        td_error = self.sess.run(self.td_error, {
            self.initial_lstm_state: lstm_state,
            self.pl_s: s,
            self.pl_a: a,
            self.pl_r: r,
            self.next_initial_lstm_state: lstm_state_,
            self.pl_s_: s_,
            self.pl_done: done,
            self.pl_gamma: gamma,
            self.pl_batch_size: len(s)
        })

        return td_error

    def save_model(self, iteration):
        self.saver.save(iteration + self.init_iteration)

    def write_constant_summaries(self, constant_summaries, iteration):
        summaries = tf.Summary(value=[tf.Summary.Value(tag=i['tag'],
                                                       simple_value=i['simple_value'])
                                      for i in constant_summaries])
        self.summary_writer.add_summary(summaries, iteration + self.init_iteration)

    def get_n_step_probs(self, n_states, n_actions):
        """
        n_states: [None, n_step, length of state space]
        n_actions: [None, n_step, length of action space]
        """

        n_states = np.asarray(n_states)
        n_actions = np.asarray(n_actions)
        assert len(n_states.shape) == len(n_actions.shape) == 3

        n_step = n_states.shape[1]

        n_states_flatten = n_states.reshape(-1, self.s_dim)
        n_actions_flatten = n_actions.reshape(-1, self.a_dim)

        n_probs = self.sess.run(self.policy_prob, {
            self.pl_s: n_states_flatten,
            self.pl_a: n_actions_flatten
        })
        # [None, length of action space]

        n_probs = n_probs.reshape(-1, n_step, self.a_dim)

        return n_probs  # [None, n_step, length of action space]

    def get_n_step_lstm_probs(self, lstm_state, n_states, n_actions):
        """
        n_states: [None, n_step, length of state space]
        n_actions: [None, n_step, length of action space]
        """
        n_states = np.asarray(n_states)
        n_actions = np.asarray(n_actions)
        assert len(n_states.shape) == len(n_actions.shape) == 3

        n_probs = self.sess.run(self.n_step_policy_prob, {
            self.n_step_initial_lstm_state: lstm_state,
            self.pl_n_step_s: n_states,
            self.pl_n_step_a: n_actions,
            self.pl_batch_size: len(n_states)
        })
        # [None, n_step, length of action space]

        return n_probs  # [None, n_step, length of action space]

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

    def train(self, s, a, r, s_, done, gamma, n_states, n_actions, lstm_state_c=None, lstm_state_h=None):
        # n_states: [None, n_step, length of state space]
        assert len(s) == len(a) == len(r) == len(s_) == len(done) == len(gamma) == len(n_states) == len(n_actions)

        if len(s) == 0:
            return

        if self.use_rnn:
            lstm_state = tf.nn.rnn_cell.LSTMStateTuple(c=np.asarray(lstm_state_c),
                                                       h=np.asarray(lstm_state_h))
            mu_n_probs = self.get_n_step_lstm_probs(lstm_state, n_states, n_actions)
            self.replay_buffer.add(s, a, r, s_, done, gamma, n_states, n_actions, mu_n_probs,
                                   lstm_state_c, lstm_state_h)
        else:
            mu_n_probs = self.get_n_step_probs(n_states, n_actions)
            self.replay_buffer.add(s, a, r, s_, done, gamma, n_states, n_actions, mu_n_probs)

        sampled = self.replay_buffer.sample()
        if sampled is None:
            return

        global_step = self.sess.run(self.global_step)

        if self.use_priority:
            points, trans, priority_is = sampled
        else:
            points, trans = sampled

        if self.use_rnn:
            s, a, r, s_, done, gamma, n_states, n_actions, mu_n_probs, lstm_state_c, lstm_state_h = trans
            lstm_state = tf.nn.rnn_cell.LSTMStateTuple(c=np.asarray(lstm_state_c),
                                                       h=np.asarray(lstm_state_h))
            lstm_state_ = self.get_n_step_lstm_state_(lstm_state, np.asarray(n_states)[:, 0:2, :])
        else:
            s, a, r, s_, done, gamma, n_states, n_actions, mu_n_probs = trans

        if self.use_n_step_is:
            if self.use_rnn:
                pi_n_probs = self.get_n_step_lstm_probs(lstm_state, n_states, n_actions)
            else:
                pi_n_probs = self.get_n_step_probs(n_states, n_actions)
            n_step_is = self._get_n_step_is(pi_n_probs, mu_n_probs)

        # write summary, update target networks and train q function
        feed_dict = {
            self.pl_s: s,
            self.pl_a: a,
            self.pl_r: r,
            self.pl_s_: s_,
            self.pl_done: done,
            self.pl_gamma: gamma
        }
        if self.use_n_step_is:
            feed_dict.update({
                self.pl_n_step_is: n_step_is
            })
        if self.use_priority:
            feed_dict.update({
                self.pl_priority_is: priority_is
            })
        if self.use_rnn:
            feed_dict.update({
                self.initial_lstm_state: lstm_state,
                self.next_initial_lstm_state: lstm_state_,
                self.pl_batch_size: len(s)
            })

        if global_step % self.write_summary_per_step == 0:
            summaries = self.sess.run(self.summaries, feed_dict)
            self.summary_writer.add_summary(summaries, global_step)

        if global_step % self.update_target_per_step == 0:
            self.sess.run(self.update_target_op)

        self.sess.run(self.train_q_ops, feed_dict)

        # train policy
        feed_dict = {
            self.pl_s: s,
        }
        if self.use_rnn:
            feed_dict.update({
                self.initial_lstm_state: lstm_state,
                self.pl_batch_size: len(s)
            })
        self.sess.run(self.train_policy_op, feed_dict)

        if self.use_auto_alpha:
            self.sess.run(self.train_alpha_op, feed_dict)

        # update td_error
        if self.use_priority:
            if self.use_rnn:
                td_error = self.get_lstm_td_error(lstm_state, s, a, r, lstm_state_, s_, done, gamma)
            else:
                td_error = self.get_td_error(s, a, r, s_, done, gamma)

            self.replay_buffer.update(points, td_error.flatten())

        # update mu_n_probs
        if self.use_n_step_is:
            if self.use_rnn:
                pi_n_probs = self.get_n_step_lstm_probs(lstm_state, n_states, n_actions)
            else:
                pi_n_probs = self.get_n_step_probs(n_states, n_actions)

            self.replay_buffer.update_transitions(8, points, pi_n_probs)

    def dispose(self):
        self.sess.close()

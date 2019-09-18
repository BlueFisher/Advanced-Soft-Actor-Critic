import functools
import logging
import time
import sys

import numpy as np
import tensorflow as tf

from .saver import Saver
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

# logger = logging.getLogger('sac.base')
# logger.propagate = False
# fh = logging.handlers.RotatingFileHandler('test.txt', maxBytes=1024 * 10240, backupCount=100000)
# fh.setLevel(logging.INFO)

# # add handler and formatter to logger
# fh.setFormatter(logging.Formatter('%(asctime)-15s [%(levelname)s] - [%(name)s] - %(message)s'))
# logger.addHandler(fh)


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
                 gamma=0.99,
                 burn_in_step=0,
                 n_step=1,
                 use_priority=False,
                 use_n_step_is=True,

                 replay_config=None):
        self.graph = tf.Graph()
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options),
                               graph=self.graph)

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
        self.pl_n_step_s = tf.placeholder(tf.float32, shape=(None, None, self.s_dim), name='n_step_state')
        self.pl_n_step_s_ = tf.placeholder(tf.float32, shape=(None, None, self.s_dim), name='n_step_state_')
        self.pl_n_step_a = tf.placeholder(tf.float32, shape=(None, None, self.a_dim), name='n_step_action')

        self.pl_s = self.pl_n_step_s[:, self.burn_in_step, :]
        self.pl_s_ = self.pl_n_step_s_[:, -1, :]
        self.pl_a = self.pl_n_step_a[:, self.burn_in_step, :]

        self.pl_r = tf.placeholder(tf.float32, shape=(None, 1), name='reward')
        self.pl_done = tf.placeholder(tf.float32, shape=(None, 1), name='done')

        self.pl_n_step_is = tf.placeholder(tf.float32, shape=(None, 1), name='n_step_is')
        self.pl_priority_is = tf.placeholder(tf.float32, shape=(None, 1), name='priority_is')

        log_alpha = tf.get_variable('alpha', shape=(), initializer=tf.constant_initializer(init_log_alpha))
        alpha = tf.exp(log_alpha)

        if self.use_rnn:
            self.pl_batch_size = tf.placeholder(tf.int32, shape=(), name='batch_size')

            # only for choosing action
            (encoded_pl_plain_s,
             self.initial_lstm_state,
             self.lstm_state,
             lstm_variables) = self._build_lstm_s_input(self.pl_s, 'lstm_state')

            (encoded_pl_n_step_s,
             self.n_step_initial_lstm_state,
             n_step_lstm_state, _) = self._build_lstm_s_input(self.pl_n_step_s, 'lstm_state', reuse=True)
            pl_n_step_s = encoded_pl_n_step_s
            pl_s = encoded_pl_n_step_s[:, self.burn_in_step, :]

            pl_s_, *_ = self._build_lstm_s_input(self.pl_s_, 'lstm_state', initial_lstm_state=n_step_lstm_state, reuse=True)

            # only for generating target encoded state_
            (_, self.target_n_step_initial_lstm_state,
             target_n_step_lstm_state, target_lstm_variables) = self._build_lstm_s_input(self.pl_n_step_s, 'target_lstm_state', trainable=False)

            pl_target_s_, *_ = self._build_lstm_s_input(self.pl_s_, 'target_lstm_state', initial_lstm_state=target_n_step_lstm_state, trainable=False, reuse=True)

            # approx_n_step_s_, approx_variables = self._build_rnn_test_net(encoded_pl_n_step_s, self.pl_n_step_a, 'lstm_approx')
            # L_approx_lstm = tf.reduce_mean(tf.squared_difference(approx_n_step_s_, self.pl_n_step_s_))
        else:
            pl_s = self.pl_s
            pl_s_ = self.pl_s_
            pl_n_step_s = self.pl_n_step_s

        self.policy, self.action_sampled, policy_variables = self._build_policy_net(pl_s, 'policy')
        if self.use_rnn:
            _, self.plain_action_sampled, _ = self._build_policy_net(encoded_pl_plain_s, 'policy', reuse=True)
        self.policy_prob = self.policy.prob(self.pl_a)

        self.n_step_policy, self.n_step_action_sampled, _ = self._build_policy_net(pl_n_step_s, 'policy', reuse=True)
        self.n_step_policy_prob = self.n_step_policy.prob(self.pl_n_step_a)

        if self.use_rnn:
            next_policy, next_action_sampled, _ = self._build_policy_net(pl_target_s_, 'policy', reuse=True)
        else:
            next_policy, next_action_sampled, _ = self._build_policy_net(pl_s_, 'policy', reuse=True)

        q1, q1_variables = self._build_q_net(pl_s, self.pl_a, 'q1')
        q2, q2_variables = self._build_q_net(pl_s, self.pl_a, 'q2')

        if self.use_rnn:
            target_q1, target_q1_variables = self._build_q_net(pl_target_s_, next_action_sampled, 'target_q1', trainable=False)
            target_q2, target_q2_variables = self._build_q_net(pl_target_s_, next_action_sampled, 'target_q2', trainable=False)
        else:
            target_q1, target_q1_variables = self._build_q_net(pl_s_, next_action_sampled, 'target_q1', trainable=False)
            target_q2, target_q2_variables = self._build_q_net(pl_s_, next_action_sampled, 'target_q2', trainable=False)

        q1_for_gradient, _ = self._build_q_net(pl_s, self.action_sampled, 'q1', reuse=True)

        next_action_prob = tf.reduce_prod(next_policy.log_prob(next_action_sampled), axis=1, keep_dims=True)
        y = self.pl_r + self.gamma**self.n_step * (1 - self.pl_done) * (tf.minimum(target_q1, target_q2) - alpha * next_action_prob)
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

        target_variables = target_q1_variables + target_q2_variables
        eval_variables = q1_variables + q2_variables
        if self.use_rnn:
            target_variables += target_lstm_variables
            eval_variables += lstm_variables

        self.update_target_hard_op = [tf.assign(t, e) for t, e in
                                      zip(target_variables, eval_variables)]
        self.update_target_op = [tf.assign(t, tau * e + (1 - tau) * t) for t, e in
                                 zip(target_variables, eval_variables)]

        with tf.name_scope('optimizer'):
            self.global_step = tf.get_variable('global_step', shape=(), dtype=tf.int32, initializer=tf.constant_initializer(0), trainable=False)

            if self.use_rnn:
                q1_variables = q1_variables + lstm_variables  # + approx_variables
                q2_variables = q2_variables + lstm_variables  # + approx_variables

            self.train_q_ops = [tf.train.AdamOptimizer(lr).minimize(L_q1,  # + L_approx_lstm,
                                                                    var_list=q1_variables),
                                tf.train.AdamOptimizer(lr).minimize(L_q2,  # + L_approx_lstm,
                                                                    var_list=q2_variables)]
            self.train_policy_op = tf.train.AdamOptimizer(lr).minimize(L_policy,
                                                                       global_step=self.global_step,
                                                                       var_list=policy_variables)
            self.train_alpha_op = tf.train.AdamOptimizer(lr).minimize(L_alpha,
                                                                      var_list=[log_alpha])

        tf.summary.scalar('loss/Q1', L_q1)
        tf.summary.scalar('loss/Q2', L_q2)
        tf.summary.scalar('loss/policy', L_policy)
        tf.summary.scalar('loss/entropy', tf.reduce_mean(entropy))
        tf.summary.scalar('loss/alpha', alpha)
        # tf.summary.scalar('loss/approx_lstm', L_approx_lstm)
        self.summaries = tf.summary.merge_all()

    def _build_rnn_test_net(self, s_input, a_input, scope, trainable=True, reuse=False):
        initializer_helper = {
            'kernel_initializer': tf.truncated_normal_initializer(0, .1),
            'bias_initializer': tf.constant_initializer(.1)
        }
        ls = tf.layers.dense(
            s_input, 64, activation=tf.nn.relu,
            trainable=trainable, **initializer_helper
        )
        la = tf.layers.dense(
            a_input, 64, activation=tf.nn.relu,
            trainable=trainable, **initializer_helper
        )
        l = tf.concat([ls, la], -1)
        l = tf.layers.dense(l, 32, activation=tf.nn.relu, trainable=trainable, **initializer_helper)
        l = tf.layers.dense(l, 32, activation=tf.nn.relu, trainable=trainable, **initializer_helper)
        s_ = tf.layers.dense(l, self.s_dim, trainable=trainable, **initializer_helper)

        variables = tf.get_variable_scope().global_variables()

        return s_, variables

    def _build_q_net(self, s_input, a_input, scope, trainable=True, reuse=False):
        raise Exception('SAC_Base._build_q_net not implemented')
        # return q, variables

    def _build_policy_net(self, s_input, scope, trainable=True, reuse=False):
        raise Exception('SAC_Base._build_policy_net not implemented')
        # return policy, action_sampled, variables

    def _build_lstm_s_input(self, s_input, scope, trainable=True, reuse=False):
        raise Exception('SAC_Base._build_lstm_s_input not implemented')
        # return encoded_s, initial_lstm_state, lstm_state, variables

    def choose_action(self, s):
        assert len(s.shape) == 2
        assert not self.use_rnn

        a = self.sess.run(self.action_sampled, {
            self.pl_s: s,
        })

        return np.clip(a, -1, 1)

    def choose_lstm_action(self, lstm_state, s):
        assert len(s.shape) == 2
        assert self.use_rnn

        a, lstm_state_ = self.sess.run([self.plain_action_sampled, self.lstm_state], {
            self.pl_s: s,
            self.initial_lstm_state: lstm_state,
            self.pl_batch_size: len(s)
        })

        return np.clip(a, -1, 1), lstm_state_

    def get_initial_lstm_state(self, batch_size):
        return self.sess.run(self.initial_lstm_state, {
            self.pl_batch_size: batch_size
        })

    def get_td_error(self, n_states, n_actions, rewards, state_, done):
        td_error = self.sess.run(self.td_error, {
            self.pl_n_step_s: n_states,
            self.pl_n_step_a: n_actions,
            self.pl_r: rewards,
            self.pl_s_: state_,
            self.pl_done: done,
        })

        return td_error

    def get_lstm_td_error(self, lstm_state, n_states, n_actions, rewards, state_, done):
        td_error = self.sess.run(self.td_error, {
            self.initial_lstm_state: lstm_state,
            self.pl_n_step_s: n_states,
            self.pl_n_step_a: n_actions,
            self.pl_r: rewards,
            self.pl_s_: state_,
            self.pl_done: done,
            self.pl_batch_size: len(n_states)
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
        n_states: [None, None, length of state space]
        n_actions: [None, None, length of action space]
        """

        n_states = np.asarray(n_states)
        n_actions = np.asarray(n_actions)
        assert len(n_states.shape) == len(n_actions.shape) == 3
        assert n_states.shape[1] == n_actions.shape[1]

        n_probs = self.sess.run(self.n_step_policy_prob, {
            self.pl_n_step_s: n_states,
            self.pl_n_step_a: n_actions,
        })
        # [None, length of action space]

        return n_probs  # [None, n_step, length of action space]

    def get_n_step_lstm_probs(self, lstm_state, n_states, n_actions):
        """
        n_states: [None, None, length of state space]
        n_actions: [None, None, length of action space]
        """
        n_states = np.asarray(n_states)
        n_actions = np.asarray(n_actions)
        assert len(n_states.shape) == len(n_actions.shape) == 3
        assert n_states.shape[1] == n_actions.shape[1]

        n_probs = self.sess.run(self.n_step_policy_prob, {
            self.n_step_initial_lstm_state: lstm_state,
            self.pl_n_step_s: n_states,
            self.pl_n_step_a: n_actions,
            self.pl_batch_size: len(n_states)
        })
        # [None, n_step, length of action space]

        return n_probs  # [None, n_step, length of action space]

    def _get_lstm_state_tuple(self, lstm_state_c, lstm_state_h):
        return tf.nn.rnn_cell.LSTMStateTuple(c=np.asarray(lstm_state_c),
                                             h=np.asarray(lstm_state_h))

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

        n_gamma = np.array([self.gamma**i for i in range(self.n_step)])
        n_rewards_gamma = n_rewards * n_gamma
        return n_rewards_gamma.sum(axis=1, keepdims=True)  # [None, 1]

    def fill_replay_buffer(self, n_states, n_actions, n_rewards, state_, done,
                           lstm_state_c=None, lstm_state_h=None):
        """
        n_states: [None, burn_in_step + n_step, s_dim]
        n_states: [None, burn_in_step + n_step, s_dim]
        n_rewards: [None, burn_in_step + n_step]
        state_: [None, s_dim]
        done: [None, 1]
        """
        assert len(n_states) == len(n_actions) == len(n_rewards) == len(state_) == len(done)
        if not self.use_rnn:
            assert lstm_state_c is None
            assert lstm_state_h is None

        if len(n_states) == 0:
            return

        if self.use_rnn:
            # lstm_state_before_burn_in = self.get_initial_lstm_state(len(n_states))
            lstm_state_before_burn_in = self._get_lstm_state_tuple(lstm_state_c, lstm_state_h)
            mu_n_probs = self.get_n_step_lstm_probs(lstm_state_before_burn_in, n_states, n_actions)  # TODO: only need [:, burn_in_step:, :]
            self.replay_buffer.add(n_states, n_actions, n_rewards, state_, done, mu_n_probs,
                                   lstm_state_c, lstm_state_h)
        else:
            mu_n_probs = self.get_n_step_probs(n_states, n_actions)
            self.replay_buffer.add(n_states, n_actions, n_rewards, state_, done, mu_n_probs)

    def train(self):
        # sample from replay buffer
        sampled = self.replay_buffer.sample()
        if sampled is None:
            return

        global_step = self.sess.run(self.global_step)

        if self.use_priority:
            points, trans, priority_is = sampled
        else:
            points, trans = sampled

        trans = [np.asarray(t) for t in trans]

        if self.use_rnn:
            n_states, n_actions, n_rewards, state_, done, mu_n_probs, lstm_state_c, lstm_state_h = trans
            # lstm_state_before_burn_in = self.get_initial_lstm_state(len(n_states))
            lstm_state_before_burn_in = self._get_lstm_state_tuple(lstm_state_c, lstm_state_h)
        else:
            n_states, n_actions, n_rewards, state_, done, mu_n_probs = trans

        if self.use_n_step_is:
            if self.use_rnn:
                pi_n_probs = self.get_n_step_lstm_probs(lstm_state_before_burn_in, n_states, n_actions)
            else:
                pi_n_probs = self.get_n_step_probs(n_states, n_actions)

            n_step_is = self._get_n_step_is(pi_n_probs[:, self.burn_in_step:, :],
                                            mu_n_probs[:, self.burn_in_step:, :])

        rewards = self._get_n_reward_gamma_sum(n_rewards[:, self.burn_in_step:])

        # write summary, update target networks and train q function
        feed_dict = {
            self.pl_n_step_s: n_states,
            self.pl_n_step_a: n_actions,
            self.pl_r: rewards,
            self.pl_s_: state_,
            self.pl_done: done
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
                self.n_step_initial_lstm_state: lstm_state_before_burn_in,
                # self.pl_n_step_s_: n_states_,
                self.target_n_step_initial_lstm_state: lstm_state_before_burn_in,
                self.pl_batch_size: len(n_states)
            })

        if global_step % self.write_summary_per_step == 0:
            summaries = self.sess.run(self.summaries, feed_dict)
            self.summary_writer.add_summary(summaries, global_step)

        if global_step % self.update_target_per_step == 0:
            self.sess.run(self.update_target_op)

        self.sess.run(self.train_q_ops, feed_dict)

        # train policy
        feed_dict = {
            self.pl_n_step_s: n_states,
        }
        if self.use_rnn:
            feed_dict.update({
                self.initial_lstm_state: lstm_state_before_burn_in,
                self.pl_batch_size: len(n_states)
            })
        self.sess.run(self.train_policy_op, feed_dict)

        if self.use_auto_alpha:
            self.sess.run(self.train_alpha_op, feed_dict)

        # update td_error
        if self.use_priority:
            if self.use_rnn:
                td_error = self.get_lstm_td_error(lstm_state_before_burn_in, n_states, n_actions, rewards, state_, done)
            else:
                td_error = self.get_td_error(n_states, n_actions, rewards, state_, done)

            self.replay_buffer.update(points, td_error.flatten())

        # update mu_n_probs
        if self.use_n_step_is:
            if self.use_rnn:
                pi_n_probs = self.get_n_step_lstm_probs(lstm_state_before_burn_in, n_states, n_actions)
            else:
                pi_n_probs = self.get_n_step_probs(n_states, n_actions)

            self.replay_buffer.update_transitions(5, points, pi_n_probs)

    def dispose(self):
        self.sess.close()

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

                 write_summary_graph=False,
                 seed=None,
                 tau=0.005,
                 write_summary_per_step=20,
                 update_target_per_step=1,
                 init_log_alpha=-4.6,
                 use_auto_alpha=True,
                 lr=3e-4,
                 use_priority=False,

                 replay_config=None):
        self.graph = tf.Graph()
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options),
                               graph=self.graph)

        self.use_auto_alpha = use_auto_alpha
        self.use_priority = use_priority

        default_replay_config = {
            'batch_size': 256,
            'capacity': 1e6,
            'alpha': 0.9
        }
        replay_config = dict(default_replay_config, **({} if replay_config is None else replay_config))
        if self.use_priority:
            self.replay_buffer = PrioritizedReplayBuffer(replay_config['batch_size'],
                                                         replay_config['capacity'],
                                                         replay_config['alpha'])
        else:
            self.replay_buffer = ReplayBuffer(replay_config['batch_size'],
                                              replay_config['capacity'])

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
        self.pl_r = tf.placeholder(tf.float32, shape=(None, 1), name='reward')
        self.pl_s_ = tf.placeholder(tf.float32, shape=(None, self.s_dim), name='state_')
        self.pl_done = tf.placeholder(tf.float32, shape=(None, 1), name='done')
        self.pl_gamma = tf.placeholder(tf.float32, shape=(None, 1), name='gamma')

        self.pl_is = tf.placeholder(tf.float32, shape=(None, 1), name='is_weight')

        log_alpha = tf.get_variable('alpha', shape=(), initializer=tf.constant_initializer(init_log_alpha))
        alpha = tf.exp(log_alpha)

        policy, self.action_sampled, self.policy_variables = self._build_policy_net(self.pl_s, 'policy')
        policy_next, action_next_sampled, policy_next_variables = self._build_policy_net(self.pl_s_, 'policy', reuse=True)

        q1, q1_variables = self._build_q_net(self.pl_s, self.pl_a, 'q1')
        q1_for_gradient, _ = self._build_q_net(self.pl_s, self.action_sampled, 'q1', reuse=True)
        q1_target, q1_target_variables = self._build_q_net(self.pl_s_, action_next_sampled, 'q1_target', trainable=False)

        q2, q2_variables = self._build_q_net(self.pl_s, self.pl_a, 'q2')
        q2_target, q2_target_variables = self._build_q_net(self.pl_s_, action_next_sampled, 'q2_target', trainable=False)

        y = self.pl_r + self.pl_gamma * (1 - self.pl_done) * (tf.minimum(q1_target, q2_target) - alpha * policy_next.log_prob(action_next_sampled))
        y = tf.stop_gradient(y)

        if self.use_priority:
            L_q1 = tf.reduce_mean(tf.squared_difference(q1, y) * self.pl_is)
            L_q2 = tf.reduce_mean(tf.squared_difference(q2, y) * self.pl_is)
        else:
            L_q1 = tf.reduce_mean(tf.squared_difference(q1, y))
            L_q2 = tf.reduce_mean(tf.squared_difference(q2, y))

        q1_td_error = tf.abs(q1 - y)
        q2_td_error = tf.abs(q2 - y)
        self.td_error = tf.reduce_mean(tf.concat([q1_td_error, q2_td_error], axis=1),
                                       axis=1, keepdims=True)

        L_policy = tf.reduce_mean(alpha * policy.log_prob(self.action_sampled) - q1_for_gradient)

        entropy = policy.entropy()
        L_alpha = -log_alpha * policy.log_prob(self.action_sampled) + log_alpha * self.a_dim

        with tf.name_scope('optimizer'):
            self.global_step = tf.get_variable('global_step', shape=(), dtype=tf.int32, initializer=tf.constant_initializer(0), trainable=False)

            self.train_q_ops = [tf.train.AdamOptimizer(lr).minimize(L_q1,
                                                                    var_list=q1_variables),
                                tf.train.AdamOptimizer(lr).minimize(L_q2,
                                                                    var_list=q2_variables)]
            self.train_policy_op = tf.train.AdamOptimizer(lr).minimize(L_policy,
                                                                       global_step=self.global_step,
                                                                       var_list=self.policy_variables)
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

    def choose_action(self, s):
        assert len(s.shape) == 2

        a = self.sess.run(self.action_sampled, {
            self.pl_s: s,
        })

        return np.clip(a, -1, 1)

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

    def save_model(self, iteration):
        self.saver.save(iteration + self.init_iteration)

    def write_constant_summaries(self, constant_summaries, iteration):
        summaries = tf.Summary(value=[tf.Summary.Value(tag=i['tag'],
                                                       simple_value=i['simple_value'])
                                      for i in constant_summaries])
        self.summary_writer.add_summary(summaries, iteration + self.init_iteration)

    def train(self, s, a, r, s_, done, gamma):
        assert len(s.shape) == 2

        global_step = self.sess.run(self.global_step)

        self.replay_buffer.add(s, a, r, s_, done, gamma)

        if self.use_priority:
            points, (s, a, r, s_, done, gamma), is_weight = self.replay_buffer.sample()
        else:
            s, a, r, s_, done, gamma = self.replay_buffer.sample()

        if global_step % self.update_target_per_step == 0:
            self.sess.run(self.update_target_op)

        if global_step % self.write_summary_per_step == 0:
            summaries = self.sess.run(self.summaries, {
                self.pl_s: s,
                self.pl_a: a,
                self.pl_r: r,
                self.pl_s_: s_,
                self.pl_done: done,
                self.pl_gamma: gamma,
                self.pl_is: np.zeros((1, 1)) if not self.use_priority else is_weight
            })
            self.summary_writer.add_summary(summaries, global_step)

        if self.replay_buffer.is_lg_batch_size:
            self.sess.run(self.train_q_ops, {
                self.pl_s: s,
                self.pl_a: a,
                self.pl_r: r,
                self.pl_s_: s_,
                self.pl_done: done,
                self.pl_gamma: gamma,
                self.pl_is: np.zeros((1, 1)) if not self.use_priority else is_weight
            })

            self.sess.run(self.train_policy_op, {
                self.pl_s: s,
            })

            if self.use_auto_alpha:
                self.sess.run(self.train_alpha_op, {
                    self.pl_s: s,
                })

            if self.use_priority:
                td_error = self.sess.run(self.td_error, {
                    self.pl_s: s,
                    self.pl_a: a,
                    self.pl_r: r,
                    self.pl_s_: s_,
                    self.pl_done: done,
                    self.pl_gamma: gamma
                })

                self.replay_buffer.update(points, td_error.flatten())

    def dispose(self):
        self.sess.close()


if __name__ == '__main__':
    sac = SAC(3, 2, np.array([1, 1]))

    sac.train(np.array([[1., 1., 2.]]),
              np.array([[2., 2.]]),
              np.array([[1.]]),
              np.array([[1., 1., 3.]]),
              np.array([[0]]),
              10)

import functools
import logging
import time
import sys

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from .trans_cache import TransCache

logger = logging.getLogger('sac.base')
logger.setLevel(level=logging.INFO)


def squash_correction_log_prob(dist, x):
    return dist.log_prob(x) - tf.math.log(1 - tf.pow(tf.tanh(x), 2) + 1e-6)


def debug(x):
    return tf.print(tf.reduce_min(x), tf.reduce_mean(x), tf.reduce_max(x))


class SAC_Base(object):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 model_root_path,
                 model,
                 train_mode=True,

                 seed=None,
                 write_summary_per_step=20,

                 burn_in_step=0,
                 n_step=1,
                 use_rnn=False,
                 use_prediction=True,

                 tau=0.005,
                 update_target_per_step=1,
                 init_log_alpha=-2.3,
                 use_auto_alpha=True,
                 q_lr=3e-4,
                 policy_lr=3e-4,
                 alpha_lr=3e-4,
                 rnn_lr=3e-4,
                 gamma=0.99,
                 _lambda=0.9,
                 use_priority=False,
                 use_n_step_is=True,
                 use_q_clip=False,

                 replay_config=None):
        """
        obs_dim: dimension of observation
        action_dim: dimension of action
        model_root_path: the path that saves summary, checkpoints, config etc.
        model: custom Model Class
        """

        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.train_mode = train_mode

        self.burn_in_step = burn_in_step
        self.n_step = n_step
        self.use_rnn = use_rnn
        self.use_prediction = use_prediction

        self.write_summary_per_step = write_summary_per_step
        self.tau = tau
        self.update_target_per_step = update_target_per_step
        self.use_auto_alpha = use_auto_alpha
        self.gamma = gamma
        self._lambda = _lambda
        self.use_priority = use_priority
        self.use_n_step_is = use_n_step_is
        self.use_q_clip = use_q_clip

        if seed is not None:
            tf.random.set_seed(seed)

        self._build_model(model, init_log_alpha,
                          q_lr, policy_lr, alpha_lr, rnn_lr)
        self._init_or_restore(model_root_path)

        if self.train_mode:
            summary_path = f'{model_root_path}/log'
            self.summary_writer = tf.summary.create_file_writer(summary_path)

            replay_config = {} if replay_config is None else replay_config
            if self.use_priority:
                self.replay_buffer = PrioritizedReplayBuffer(**replay_config)
            else:
                self.replay_buffer = ReplayBuffer(**replay_config)

        self._init_tf_function()

    def _build_model(self, model, init_log_alpha,
                     q_lr, policy_lr, alpha_lr,
                     rnn_lr=None):
        """
        Initialize variables, network models and optimizers
        """
        self.log_alpha = tf.Variable(init_log_alpha, dtype=tf.float32, name='log_alpha')
        if self.use_q_clip:
            self.min_q = tf.Variable(0, dtype=tf.float32, name='min_q')
            self.max_q = tf.Variable(0, dtype=tf.float32, name='max_q')
        self.global_step = tf.Variable(0, dtype=tf.int64, name='global_step')

        self.optimizer_q1 = tf.keras.optimizers.Adam(q_lr)
        self.optimizer_q2 = tf.keras.optimizers.Adam(q_lr)
        self.optimizer_policy = tf.keras.optimizers.Adam(policy_lr)
        if self.use_auto_alpha:
            self.optimizer_alpha = tf.keras.optimizers.Adam(alpha_lr)

        if self.use_rnn:
            self.model_rnn = model.ModelRNN(self.obs_dim)
            self.model_target_rnn = model.ModelRNN(self.obs_dim)

            # get state dimension and initial_rnn_state
            tmp_state, tmp_rnn_state, _ = self.model_rnn.get_call_result_tensors()
            state_dim = tmp_state.shape[-1]
            self.rnn_state_dim = tmp_rnn_state.shape[-1]

            self.optimizer_rnn = tf.keras.optimizers.Adam(rnn_lr)

            if self.use_prediction:
                self.model_transition = model.ModelTransition(state_dim, self.action_dim)
                self.model_reward = model.ModelReward(state_dim)
        else:
            state_dim = self.obs_dim
            self.rnn_state_dim = 1

        self.model_q1 = model.ModelQ(state_dim, self.action_dim)
        self.model_target_q1 = model.ModelQ(state_dim, self.action_dim)
        self.model_q2 = model.ModelQ(state_dim, self.action_dim)
        self.model_target_q2 = model.ModelQ(state_dim, self.action_dim)
        self.model_policy = model.ModelPolicy(state_dim, self.action_dim)

    def _init_tf_function(self):
        def _np_to_tensor(fn):
            def c(*args, **kwargs):
                return fn(*[tf.constant(k) if not isinstance(k, tf.Tensor) else k for k in args if k is not None],
                          **{k: tf.constant(v) if not isinstance(v, tf.Tensor) else v for k, v in kwargs.items() if v is not None})

            return c

        if self.use_rnn:
            tmp_get_rnn_n_step_probs = self.get_rnn_n_step_probs.get_concrete_function(
                n_obses=tf.TensorSpec(shape=(None, None, self.obs_dim), dtype=tf.float32),
                n_selected_actions=tf.TensorSpec(shape=(None, None, self.action_dim), dtype=tf.float32),
                rnn_state=tf.TensorSpec(shape=(None, self.rnn_state_dim), dtype=tf.float32),
            )
            self.get_rnn_n_step_probs = _np_to_tensor(tmp_get_rnn_n_step_probs)
        else:
            tmp_get_n_step_probs = self.get_n_step_probs.get_concrete_function(
                n_obses=tf.TensorSpec(shape=(None, None, self.obs_dim), dtype=tf.float32),
                n_selected_actions=tf.TensorSpec(shape=(None, None, self.action_dim), dtype=tf.float32)
            )
            self.get_n_step_probs = _np_to_tensor(tmp_get_n_step_probs)

        if self.train_mode:
            step_size = self.burn_in_step + self.n_step
            # get_td_error
            kwargs = {
                'n_obses': tf.TensorSpec(shape=[None, step_size, self.obs_dim], dtype=tf.float32),
                'n_actions': tf.TensorSpec(shape=[None, step_size, self.action_dim], dtype=tf.float32),
                'n_rewards': tf.TensorSpec(shape=[None, step_size], dtype=tf.float32),
                'obs_': tf.TensorSpec(shape=[None, self.obs_dim], dtype=tf.float32),
                'n_dones': tf.TensorSpec(shape=[None, step_size], dtype=tf.float32)
            }
            if self.use_n_step_is:
                kwargs['n_mu_probs'] = tf.TensorSpec(shape=[None, step_size, self.action_dim], dtype=tf.float32)
            if self.use_rnn:
                kwargs['rnn_state'] = tf.TensorSpec(shape=[None, self.rnn_state_dim], dtype=tf.float32)
            tmp_get_td_error = self.get_td_error.get_concrete_function(**kwargs)
            self.get_td_error = _np_to_tensor(tmp_get_td_error)

            # _train
            kwargs = {
                'n_obses': tf.TensorSpec(shape=[None, step_size, self.obs_dim], dtype=tf.float32),
                'n_actions': tf.TensorSpec(shape=[None, step_size, self.action_dim], dtype=tf.float32),
                'n_rewards': tf.TensorSpec(shape=[None, step_size], dtype=tf.float32),
                'obs_': tf.TensorSpec(shape=[None, self.obs_dim], dtype=tf.float32),
                'n_dones': tf.TensorSpec(shape=[None, step_size], dtype=tf.float32)
            }
            if self.use_n_step_is:
                kwargs['n_mu_probs'] = tf.TensorSpec(shape=[None, step_size, self.action_dim], dtype=tf.float32)
            if self.use_priority:
                kwargs['priority_is'] = tf.TensorSpec(shape=[None, 1], dtype=tf.float32)
            if self.use_rnn:
                kwargs['initial_rnn_state'] = tf.TensorSpec(shape=[None, self.rnn_state_dim], dtype=tf.float32)
            tmp_train = self._train.get_concrete_function(**kwargs)
            self._train = _np_to_tensor(tmp_train)

    def _init_or_restore(self, model_path):
        """
        Initialize network weights from scratch or restore from model_root_path
        """
        ckpt = tf.train.Checkpoint(log_alpha=self.log_alpha,
                                   global_step=self.global_step,

                                   optimizer_q1=self.optimizer_q1,
                                   optimizer_q2=self.optimizer_q2,
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
                ckpt.model_transition = self.model_transition
                ckpt.model_reward = self.model_reward

        if self.use_auto_alpha:
            ckpt.optimizer_alpha = self.optimizer_alpha

        if self.use_q_clip:
            ckpt.min_q = self.min_q
            ckpt.max_q = self.max_q

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

        return np.zeros([batch_size, self.rnn_state_dim], dtype=np.float32)

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
    def get_n_step_probs(self, n_obses, n_selected_actions):
        """
        tf.function
        """
        policy = self.model_policy(n_obses)
        policy_prob = tf.exp(squash_correction_log_prob(policy, tf.atanh(n_selected_actions)))

        return policy_prob

    @tf.function
    def get_rnn_n_step_probs(self, n_obses, n_selected_actions, rnn_state):
        """
        tf.function
        """
        n_states, *_ = self.model_rnn(n_obses, [rnn_state])
        return self.get_n_step_probs(n_states, n_selected_actions)

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32),
                                  tf.TensorSpec(shape=None, dtype=tf.float32)])
    def _update_q_bound(self, min_reward, max_reward):
        self.min_q.assign(tf.minimum(self.min_q, min_reward))
        self.max_q.assign(tf.maximum(self.max_q, max_reward))

    def update_q_bound(self, n_rewards):
        # n_rewards [1, episode_len]
        ep_len = n_rewards.shape[1]
        gamma = np.array([[np.power(self.gamma, i)] for i in range(ep_len)], dtype=np.float32)

        cum_rewards = np.empty(ep_len, dtype=np.float32)
        tmp_rewards = n_rewards * gamma

        for i in range(ep_len):
            cum_rewards[i] = np.sum(tmp_rewards.diagonal(i))

        cum_n_rewards = np.cumsum(tmp_rewards.diagonal(0))
        cum_rewards = np.concatenate([cum_rewards, cum_n_rewards])

        self._update_q_bound(np.min(cum_rewards), np.max(cum_rewards))

    def _get_y(self, n_states, n_actions, n_rewards, state_, n_dones,
               n_mu_probs=None):
        """
        tf.function
        get target value
        """
        # if self.use_q_clip:
        #     n_rewards = n_rewards / self.max_q

        gamma_ratio = [[tf.pow(self.gamma, i) for i in range(self.n_step)]]
        lambda_ratio = [[tf.pow(self._lambda, i) for i in range(self.n_step)]]
        alpha = tf.exp(self.log_alpha)

        policy = self.model_policy(n_states)
        n_actions_sampled = policy.sample()
        n_actions_log_prob = tf.reduce_sum(squash_correction_log_prob(policy, n_actions_sampled), axis=2)

        next_n_states = tf.concat([n_states[:, 1:, :], tf.reshape(state_, (-1, 1, state_.shape[-1]))], axis=1)
        next_policy = self.model_policy(next_n_states)
        next_n_actions_sampled = next_policy.sample()
        next_n_actions_log_prob = tf.reduce_sum(squash_correction_log_prob(next_policy, next_n_actions_sampled), axis=2)

        q1 = self.model_target_q1(n_states, tf.tanh(n_actions_sampled))
        q2 = self.model_target_q2(n_states, tf.tanh(n_actions_sampled))
        q1 = tf.squeeze(q1, [-1])
        q2 = tf.squeeze(q2, [-1])

        next_q1 = self.model_target_q1(next_n_states, tf.tanh(next_n_actions_sampled))
        next_q2 = self.model_target_q2(next_n_states, tf.tanh(next_n_actions_sampled))
        next_q1 = tf.squeeze(next_q1, [-1])
        next_q2 = tf.squeeze(next_q2, [-1])

        min_next_q = tf.minimum(next_q1, next_q2)
        min_q = tf.minimum(q1, q2)
        if self.use_q_clip:
            min_next_q = tf.clip_by_value(min_next_q, self.min_q - alpha * next_n_actions_log_prob, self.max_q - alpha * next_n_actions_log_prob)
            min_q = tf.clip_by_value(min_q, self.min_q - alpha * n_actions_log_prob, self.max_q - alpha * n_actions_log_prob)

        td_error = n_rewards + self.gamma * (1 - n_dones) * (min_next_q - alpha * next_n_actions_log_prob) \
            - (min_q - alpha * n_actions_log_prob)

        td_error = gamma_ratio * td_error
        if self.use_n_step_is:
            td_error = lambda_ratio * td_error
            n_pi_probs = self.get_n_step_probs(n_states, n_actions)
            # ρ_t
            n_step_is = tf.clip_by_value(n_pi_probs / n_mu_probs, 0, 1.)  # [None, None, action_dim]
            n_step_is = tf.reduce_prod(n_step_is, axis=2)  # [None, None]

            # \prod{c_i}
            cumulative_n_step_is = tf.math.cumprod(n_step_is, axis=1)
            td_error = n_step_is * cumulative_n_step_is * td_error

        # \sum{td_error}
        r = tf.reduce_sum(td_error, axis=1, keepdims=True)

        # V_s + \sum{td_error}
        min_q = tf.minimum(q1[:, 0:1], q2[:, 0:1])
        if self.use_q_clip:
            min_q = tf.clip_by_value(min_q, self.min_q - alpha * n_actions_log_prob[:, 0:1], self.max_q - alpha * n_actions_log_prob[:, 0:1])
        y = min_q - alpha * n_actions_log_prob[:, 0:1] + r

        # NO V-TRACE
        # policy = self.model_policy(state_)
        # action_sampled = policy.sample()
        # next_log_prob = tf.reduce_sum(policy.log_prob(action_sampled), axis=1, keepdims=True)
        # next_q1 = self.model_target_q1(state_, action_sampled)
        # next_q2 = self.model_target_q2(state_, action_sampled)
        # reward = tf.reduce_sum(n_rewards * gamma_ratio, axis=1, keepdims=True)
        # y = reward + tf.pow(self.gamma, self.n_step) * (1 - done) * (tf.minimum(next_q1, next_q2) - alpha * next_log_prob)

        return y  # [None, 1]

    @tf.function
    def _train(self, n_obses, n_actions, n_rewards, obs_, n_dones,
               n_mu_probs=None, priority_is=None,
               initial_rnn_state=None):
        """
        tf.function
        """
        if self.global_step % self.update_target_per_step == 0:
            self._update_target_variables(tau=self.tau)

        with tf.GradientTape(persistent=True) as tape:
            if self.use_rnn:
                m_obses = tf.concat([n_obses, tf.reshape(obs_, (-1, 1, obs_.shape[-1]))], axis=1)
                m_states, *_ = self.model_rnn(m_obses, initial_rnn_state)
                n_states = m_states[:, :-1, :]
                state = n_states[:, self.burn_in_step, :]

                m_target_states, *_ = self.model_target_rnn(m_obses, initial_rnn_state)
                state_ = m_target_states[:, -1, :]
            else:
                n_states = n_obses
                state = n_obses[:, 0, :]
                state_ = obs_

            action = n_actions[:, self.burn_in_step, :]

            alpha = tf.exp(self.log_alpha)

            policy = self.model_policy(state)
            action_sampled = policy.sample()

            q1 = self.model_q1(state, action)
            q1_for_gradient = self.model_q1(state, tf.tanh(action_sampled))
            q2 = self.model_q2(state, action)
            q2_for_gradient = self.model_q2(state, tf.tanh(action_sampled))

            y = tf.stop_gradient(self._get_y(n_states[:, self.burn_in_step:, :],
                                             n_actions[:, self.burn_in_step:, :],
                                             n_rewards[:, self.burn_in_step:],
                                             state_, n_dones[:, self.burn_in_step:],
                                             n_mu_probs[:, self.burn_in_step:] if self.use_n_step_is else None))

            loss_mse = tf.keras.losses.MeanSquaredError()

            loss_q1 = loss_mse(q1, y)
            loss_q2 = loss_mse(q2, y)

            if self.use_priority:
                loss_q1 *= priority_is
                loss_q2 *= priority_is

            if self.use_rnn:
                loss_rnn = loss_q1 + loss_q2

                if self.use_prediction:
                    approx_next_state_dist = self.model_transition(n_states[:, self.burn_in_step:, :],
                                                                   n_actions[:, self.burn_in_step:, :])
                    loss_transition = -approx_next_state_dist.log_prob(m_states[:, self.burn_in_step + 1:, :])
                    std_normal = tfp.distributions.Normal(tf.zeros_like(approx_next_state_dist.loc),
                                                          tf.ones_like(approx_next_state_dist.scale))
                    loss_transition += tfp.distributions.kl_divergence(approx_next_state_dist, std_normal)
                    loss_transition = tf.reduce_mean(loss_transition)

                    approx_reward = self.model_reward(m_states[:, self.burn_in_step + 1:, :])
                    loss_reward = loss_mse(approx_reward, tf.expand_dims(n_rewards[:, self.burn_in_step:], 2))

                    loss_rnn = loss_rnn + loss_transition + loss_reward

            log_prob = tf.reduce_sum(squash_correction_log_prob(policy, action_sampled), axis=1, keepdims=True)
            loss_policy = alpha * log_prob - tf.minimum(q1_for_gradient, q2_for_gradient)
            loss_alpha = -self.log_alpha * log_prob + self.log_alpha * self.action_dim

        grads_q1 = tape.gradient(loss_q1, self.model_q1.trainable_variables)
        self.optimizer_q1.apply_gradients(zip(grads_q1, self.model_q1.trainable_variables))

        grads_q2 = tape.gradient(loss_q2, self.model_q2.trainable_variables)
        self.optimizer_q2.apply_gradients(zip(grads_q2, self.model_q2.trainable_variables))

        if self.use_rnn:
            rnn_variables = self.model_rnn.trainable_variables
            if self.use_prediction:
                rnn_variables += self.model_transition.trainable_variables
                rnn_variables += self.model_reward.trainable_variables
            grads_rnn = tape.gradient(loss_rnn, rnn_variables)
            self.optimizer_rnn.apply_gradients(zip(grads_rnn, rnn_variables))

        grads_policy = tape.gradient(loss_policy, self.model_policy.trainable_variables)
        self.optimizer_policy.apply_gradients(zip(grads_policy, self.model_policy.trainable_variables))

        if self.use_auto_alpha:
            grads_alpha = tape.gradient(loss_alpha, self.log_alpha)
            self.optimizer_alpha.apply_gradients([(grads_alpha, self.log_alpha)])

        del tape

        if self.summary_writer is not None and self.global_step % self.write_summary_per_step == 0:
            with self.summary_writer.as_default():
                tf.summary.scalar('loss/y', tf.reduce_mean(y), step=self.global_step)
                if self.use_rnn and self.use_prediction:
                    tf.summary.scalar('loss/transition', -tf.reduce_mean(loss_transition), step=self.global_step)
                    tf.summary.scalar('loss/reward', tf.reduce_mean(loss_reward), step=self.global_step)
                tf.summary.scalar('loss/Q1', tf.reduce_mean(loss_q1), step=self.global_step)
                tf.summary.scalar('loss/Q2', tf.reduce_mean(loss_q2), step=self.global_step)
                tf.summary.scalar('loss/policy', tf.reduce_mean(loss_policy), step=self.global_step)
                tf.summary.scalar('loss/entropy', tf.reduce_mean(policy.entropy()), step=self.global_step)
                tf.summary.scalar('loss/alpha', alpha, step=self.global_step)

        self.global_step.assign_add(1)

    @tf.function
    def get_n_rnn_states(self, n_obses, rnn_state):
        *_, n_rnn_states = self.model_rnn(n_obses, [rnn_state])
        return n_rnn_states

    @tf.function
    def choose_action(self, obs):
        """
        tf.function
        """
        policy = self.model_policy(obs)
        return tf.tanh(policy.sample())

    @tf.function
    def choose_rnn_action(self, obs, rnn_state):
        """
        tf.function
        """
        obs = tf.reshape(obs, (-1, 1, obs.shape[-1]))
        state, next_rnn_state, _ = self.model_rnn(obs, [rnn_state])
        policy = self.model_policy(state)
        action = policy.sample()
        action = tf.reshape(action, (-1, action.shape[-1]))
        return tf.tanh(action), next_rnn_state

    @tf.function
    def get_td_error(self, n_obses, n_actions, n_rewards, obs_, n_dones,
                     n_mu_probs=None, rnn_state=None):
        """
        tf.function
        """
        if self.use_rnn:
            m_obses = tf.concat([n_obses, tf.reshape(obs_, (-1, 1, obs_.shape[-1]))], axis=1)
            m_target_states, *_ = self.model_target_rnn(m_obses, [rnn_state])
            state_ = m_target_states[:, -1, :]

            n_states, *_ = self.model_rnn(n_obses,
                                          [rnn_state])
        else:
            n_states = n_obses
            state_ = obs_

        state = n_states[:, self.burn_in_step, :]
        action = n_actions[:, self.burn_in_step, :]

        q1 = self.model_q1(state, action)
        q2 = self.model_q2(state, action)
        y = self._get_y(n_states[:, self.burn_in_step:, :],
                        n_actions[:, self.burn_in_step:, :],
                        n_rewards[:, self.burn_in_step:],
                        state_, n_dones[:, self.burn_in_step:],
                        n_mu_probs[:, self.burn_in_step:] if self.use_n_step_is else None)

        q1_td_error = tf.abs(q1 - y)
        q2_td_error = tf.abs(q2 - y)
        td_error = tf.reduce_mean(tf.concat([q1_td_error, q2_td_error], axis=1),
                                  axis=1, keepdims=True)

        return td_error

    def save_model(self, iteration):
        self.ckpt_manager.save(iteration + self.init_iteration)

    def write_constant_summaries(self, constant_summaries, iteration):
        with self.summary_writer.as_default():
            for s in constant_summaries:
                tf.summary.scalar(s['tag'], s['simple_value'], step=iteration + self.init_iteration)
        self.summary_writer.flush()

    def fill_replay_buffer(self, n_obses, n_actions, n_rewards, obs_, n_dones,
                           n_rnn_states=None):
        """
        n_obses: [1, episode_len, obs_dim]
        n_actions: [1, episode_len, action_dim]
        n_rewards: [1, episode_len]
        obs_: [1, obs_dim]
        n_dones: [1, episode_len]
        """
        if self.use_q_clip:
            self.update_q_bound(n_rewards)

        if n_obses.shape[1] < self.burn_in_step + self.n_step:
            return

        obs = n_obses.reshape([-1, n_obses.shape[-1]])
        action = n_actions.reshape([-1, n_actions.shape[-1]])
        reward = n_rewards.reshape([-1])
        done = n_dones.reshape([-1])

        # padding obs_
        obs = np.concatenate([obs, obs_])
        action = np.concatenate([action,
                                 np.empty([1, action.shape[-1]], dtype=np.float32)])
        reward = np.concatenate([reward,
                                 np.zeros([1], dtype=np.float32)])
        done = np.concatenate([done,
                               np.zeros([1], dtype=np.float32)])

        storage_data = {
            'obs': obs,
            'action': action,
            'reward': reward,
            'done': done,
        }

        if self.use_n_step_is:
            if self.use_rnn:
                n_mu_probs = self.get_rnn_n_step_probs(n_obses, n_actions,
                                                       n_rnn_states[:, 0, :]).numpy()
            else:
                n_mu_probs = self.get_n_step_probs(n_obses, n_actions).numpy()

            mu_prob = n_mu_probs.reshape([-1, n_mu_probs.shape[-1]])
            mu_prob = np.concatenate([mu_prob,
                                      np.empty([1, mu_prob.shape[-1]], dtype=np.float32)])
            storage_data['mu_prob'] = mu_prob

        if self.use_rnn:
            rnn_state = n_rnn_states.reshape([-1, n_rnn_states.shape[-1]])
            rnn_state = np.concatenate([rnn_state,
                                        np.empty([1, rnn_state.shape[-1]], dtype=np.float32)])
            storage_data['rnn_state'] = rnn_state

        """
        obs: [episode_len + 1, obs_dim]
        action: [episode_len + 1, action_dim]
        reward: [episode_len + 1, ]
        done: [episode_len + 1, ]
        mu_prob: [episode_len + 1, action_dim]
        """

        # n_step transitions except the first one and the last obs_, n_step - 1 + 1
        if self.use_priority:
            self.replay_buffer.add(storage_data, ignore_size=self.burn_in_step + self.n_step)
        else:
            self.replay_buffer.add(storage_data)  # TODO

    def train(self):
        # sample from replay buffer
        sampled = self.replay_buffer.sample()
        if sampled is None:
            return

        """
        trans:
            obs: [None, obs_dim]
            action: [None, action_dim]
            reward: [None, ]
            done: [None, ]
            mu_prob: [None, action_dim]
        """
        if self.use_priority:
            pointers, trans, priority_is = sampled
        else:
            pointers, trans = sampled

        # get n_step transitions
        trans = {k: [v] for k, v in trans.items()}
        # k: [v, v, ...]
        for i in range(1, self.burn_in_step + self.n_step + 1):
            t_trans = self.replay_buffer.get_storage_data(pointers + i).items()
            for k, v in t_trans:
                trans[k].append(v)

        for k, v in trans.items():
            trans[k] = np.concatenate([np.expand_dims(t, 1) for t in v], axis=1)

        """
        m_obses: [None, episode_len + 1, obs_dim]
        m_actions: [None, episode_len + 1, action_dim]
        m_rewards: [None, episode_len + 1]
        m_dones: [None, episode_len + 1]
        m_mu_probs: [None, episode_len + 1, action_dim]
        """
        m_obses = trans['obs']
        m_actions = trans['action']
        m_rewards = trans['reward']
        m_dones = trans['done']
        m_mu_probs = trans['mu_prob']

        n_obses = m_obses[:, :-1, :]
        n_actions = m_actions[:, :-1, :]
        n_rewards = m_rewards[:, :-1]
        obs_ = m_obses[:, -1, :]
        n_dones = m_dones[:, :-1]
        n_mu_probs = m_mu_probs[:, :-1, :]

        if self.use_rnn:
            m_rnn_states = trans['rnn_state']
            rnn_state = m_rnn_states[:, 0, :]

        self._train(n_obses=n_obses,
                    n_actions=n_actions,
                    n_rewards=n_rewards,
                    obs_=obs_,
                    n_dones=n_dones,
                    n_mu_probs=n_mu_probs if self.use_n_step_is else None,
                    priority_is=priority_is if self.use_priority else None,
                    initial_rnn_state=rnn_state if self.use_rnn else None)

        self.summary_writer.flush()

        if self.use_n_step_is or self.use_priority:
            if self.use_rnn:
                n_pi_probs = self.get_rnn_n_step_probs(n_obses, n_actions,
                                                       rnn_state).numpy()
            else:
                n_pi_probs = self.get_n_step_probs(n_obses, n_actions).numpy()

        # update td_error
        if self.use_priority:
            td_error = self.get_td_error(n_obses, n_actions, n_rewards, obs_, n_dones,
                                         n_mu_probs=n_pi_probs if self.use_n_step_is else None,
                                         rnn_state=rnn_state if self.use_rnn else None).numpy()

            self.replay_buffer.update(pointers, td_error)

        # update rnn_state
        if self.use_rnn:
            pointers_list = [pointers + i for i in range(1, self.burn_in_step + self.n_step + 1)]
            tmp_pointers = np.stack(pointers_list, axis=1).reshape(-1)
            n_rnn_states = self.get_n_rnn_states(n_obses, rnn_state).numpy()
            # n_rnn_states = n_rnn_states[:, self.burn_in_step - 1:, :]
            rnn_states = n_rnn_states.reshape(-1, n_rnn_states.shape[-1])
            self.replay_buffer.update_transitions(tmp_pointers, 'rnn_state', rnn_states)

        # update n_mu_probs
        if self.use_n_step_is:
            pointers_list = [pointers + i for i in range(0, self.burn_in_step + self.n_step)]
            tmp_pointers = np.stack(pointers_list, axis=1).reshape(-1)
            # n_pi_probs = n_pi_probs[:, self.burn_in_step:]
            pi_probs = n_pi_probs.reshape(-1, n_pi_probs.shape[-1])
            self.replay_buffer.update_transitions(tmp_pointers, 'mu_prob', pi_probs)

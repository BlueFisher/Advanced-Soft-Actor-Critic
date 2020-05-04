import functools
import logging
import time
import sys

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .replay_buffer import PrioritizedReplayBuffer
from .trans_cache import TransCache

logger = logging.getLogger('sac.base')
logger.setLevel(level=logging.INFO)


def squash_correction_log_prob(dist, x):
    return dist.log_prob(x) - tf.math.log(tf.maximum(1 - tf.square(tf.tanh(x)), 1e-6))


def debug(x):
    tf.print(tf.reduce_min(x), tf.reduce_mean(x), tf.reduce_max(x))


def debug_grad(grads):
    for grad in grads:
        tf.print(grad.name)
        debug(grad)


def debug_grad_com(grads, grads1):
    for i, grad in enumerate(grads):
        tf.print(grad.name)
        debug(grad - grads1[i])


def _np_to_tensor(fn):
    def c(*args, **kwargs):
        return fn(*[k for k in args if k is not None],
                  **{k: v for k, v in kwargs.items() if v is not None})

    return c


class SAC_Base(object):
    def __init__(self,
                 obs_dims,
                 action_dim,
                 is_discrete,
                 model_root_path,
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

                 use_priority=True,
                 use_n_step_is=True,
                 use_prediction=False,
                 use_reward_normalization=False,
                 use_curiosity=False,
                 curiosity_strength=1,

                 replay_config=None):
        """
        obs_dims: List of dimensions of observations
        action_dim: Dimension of action
        model_root_path: The path that saves summary, checkpoints, config etc.
        model: Custom Model Class
        train_mode: Is training or inference

        seed: Random seed
        write_summary_per_step: Write summaries in TensorBoard every `write_summary_per_step` steps

        burn_in_step: Burn-in steps in R2D2
        n_step: Update Q function by `n_step` steps
        use_rnn: If use RNN

        tau: Coefficient of updating target network
        update_target_per_step: Update target network every 'update_target_per_step' steps
        init_log_alpha: The initial log_alpha
        use_auto_alpha: If use automating entropy adjustment
        learning_rate: Learning rate of all optimizers
        gamma: Discount factor
        _lambda: Discount factor for V-trace
        clip_epsilon: Epsilon for q and policy clip

        use_priority: If use PER importance ratio
        use_n_step_is: If use importance sampling
        use_prediction: If train a transition model
        use_reward_normalization: If use reward normalization
        use_curiosity: If use curiosity
        curiosity_strength: Curiosity strength if use curiosity
        """

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

        self.use_priority = use_priority
        self.use_n_step_is = use_n_step_is
        self.use_prediction = use_prediction
        self.use_reward_normalization = use_reward_normalization
        self.use_curiosity = use_curiosity
        self.curiosity_strength = curiosity_strength

        self.use_add_with_td = False

        if seed is not None:
            tf.random.set_seed(seed)

        if self.train_mode:
            summary_path = f'{model_root_path}/log'
            self.summary_writer = tf.summary.create_file_writer(summary_path)

            replay_config = {} if replay_config is None else replay_config
            self.replay_buffer = PrioritizedReplayBuffer(**replay_config)

        self._build_model(model, init_log_alpha, learning_rate)
        self._init_or_restore(model_root_path)

        self._init_tf_function()

    def _build_model(self, model, init_log_alpha, learning_rate):
        """
        Initialize variables, network models and optimizers
        """
        self.log_alpha = tf.Variable(init_log_alpha, dtype=tf.float32, name='log_alpha')
        if self.use_reward_normalization:
            self.min_cum_reward = tf.Variable(0, dtype=tf.float32, name='min_q')
            self.max_cum_reward = tf.Variable(0, dtype=tf.float32, name='max_q')
        self.global_step = tf.Variable(0, dtype=tf.int64, name='global_step')

        def adam_optimizer(): return tf.keras.optimizers.Adam(learning_rate)

        self.optimizer_rep = adam_optimizer()
        self.optimizer_q1 = adam_optimizer()
        self.optimizer_q2 = adam_optimizer()
        self.optimizer_policy = adam_optimizer()

        if self.use_auto_alpha:
            self.optimizer_alpha = adam_optimizer()

        # Get represented state dimension
        self.model_rep = model.ModelRep(self.obs_dims)
        self.model_target_rep = model.ModelRep(self.obs_dims)
        if self.use_rnn:
            # Get rnn_state dimension
            state, next_rnn_state, _ = self.model_rep.get_call_result_tensors()
            self.rnn_state_dim = next_rnn_state.shape[-1]
        else:
            state = self.model_rep.get_call_result_tensors()
            self.rnn_state_dim = 1
        state_dim = state.shape[-1]
        logger.info(f'State Dimension: {state_dim}')

        if self.use_prediction:
            self.model_transition = model.ModelTransition(state_dim, self.action_dim)
            self.model_reward = model.ModelReward(state_dim)
            self.model_observation = model.ModelObservation(state_dim, self.obs_dims)

        if self.use_curiosity:
            self.model_forward = model.ModelForward(state_dim, self.action_dim)
            self.optimizer_forward = adam_optimizer()

        self.model_q1 = model.ModelQ(state_dim, self.action_dim)
        self.model_target_q1 = model.ModelQ(state_dim, self.action_dim)
        self.model_q2 = model.ModelQ(state_dim, self.action_dim)
        self.model_target_q2 = model.ModelQ(state_dim, self.action_dim)

        self.model_policy = model.ModelPolicy(state_dim, self.action_dim)
        self.model_target_policy = model.ModelPolicy(state_dim, self.action_dim)

        self.cem_rewards = None

    def _init_or_restore(self, model_path):
        """
        Initialize network weights from scratch or restore from model_root_path
        """
        ckpt = tf.train.Checkpoint(log_alpha=self.log_alpha,
                                   global_step=self.global_step,

                                   optimizer_rep=self.optimizer_rep,
                                   optimizer_q1=self.optimizer_q1,
                                   optimizer_q2=self.optimizer_q2,
                                   optimizer_policy=self.optimizer_policy,

                                   model_rep=self.model_rep,
                                   model_target_rep=self.model_target_rep,
                                   model_q1=self.model_q1,
                                   model_target_q1=self.model_target_q1,
                                   model_q2=self.model_q2,
                                   model_target_q2=self.model_target_q2,
                                   model_policy=self.model_policy)
        if self.use_prediction:
            ckpt.model_transition = self.model_transition
            ckpt.model_reward = self.model_reward
            ckpt.model_observation = self.model_observation

        if self.use_auto_alpha:
            ckpt.optimizer_alpha = self.optimizer_alpha

        if self.use_reward_normalization:
            ckpt.min_cum_reward = self.min_cum_reward
            ckpt.max_cum_reward = self.max_cum_reward

        if self.use_curiosity:
            ckpt.model_forward = self.model_forward
            ckpt.optimizer_forward = self.optimizer_forward

        self.ckpt_manager = tf.train.CheckpointManager(ckpt, f'{model_path}/model', max_to_keep=10)

        ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            logger.info(f'Restored from {self.ckpt_manager.latest_checkpoint}')
            self.init_iteration = int(self.ckpt_manager.latest_checkpoint.split('-')[1].split('.')[0])
        else:
            logger.info('Initializing from scratch')
            self.init_iteration = 0
            self._update_target_variables()

    def _init_tf_function(self):
        """
        Initialize some @tf.function and specify tf.TensorSpec
        """

        """ get_n_probs
        n_obses_list, n_selected_actions, rnn_state=None """
        if self.use_rnn:
            tmp_get_n_probs = tf.function(self.get_n_probs.python_function, input_signature=[
                [tf.TensorSpec(shape=(None, None, *t)) for t in self.obs_dims],
                tf.TensorSpec(shape=(None, None, self.action_dim)),
                tf.TensorSpec(shape=(None, self.rnn_state_dim)),
            ])
        else:
            tmp_get_n_probs = tf.function(self.get_n_probs.python_function, input_signature=[
                [tf.TensorSpec(shape=(None, None, *t)) for t in self.obs_dims],
                tf.TensorSpec(shape=(None, None, self.action_dim)),
            ])
        self.get_n_probs = _np_to_tensor(tmp_get_n_probs)

        if self.train_mode:
            step_size = self.burn_in_step + self.n_step
            """ get_td_error
            n_obses_list, n_actions, n_rewards, next_obs_list, n_dones,
            n_mu_probs=None,
            rnn_state=None """
            signature = [
                [tf.TensorSpec(shape=(None, step_size, *t)) for t in self.obs_dims],
                tf.TensorSpec(shape=(None, step_size, self.action_dim)),
                tf.TensorSpec(shape=(None, step_size)),
                [tf.TensorSpec(shape=(None, *t)) for t in self.obs_dims],
                tf.TensorSpec(shape=(None, step_size))
            ]
            if self.use_n_step_is:
                signature.append(tf.TensorSpec(shape=(None, step_size)))
            if self.use_rnn:
                signature.append(tf.TensorSpec(shape=(None, self.rnn_state_dim)))
            self.get_td_error = _np_to_tensor(tf.function(self.get_td_error.python_function,
                                                          input_signature=signature))

            """ _train
            n_obses_list, n_actions, n_rewards, next_obs_list, n_dones,
            n_mu_probs=None, priority_is=None,
            initial_rnn_state=None """
            signature = [
                [tf.TensorSpec(shape=(None, step_size, *t)) for t in self.obs_dims],
                tf.TensorSpec(shape=(None, step_size, self.action_dim)),
                tf.TensorSpec(shape=(None, step_size)),
                [tf.TensorSpec(shape=(None, *t)) for t in self.obs_dims],
                tf.TensorSpec(shape=(None, step_size))
            ]
            if self.use_n_step_is:
                signature.append(tf.TensorSpec(shape=(None, step_size)))
            if self.use_priority:
                signature.append(tf.TensorSpec(shape=(None, 1)))
            if self.use_rnn:
                signature.append(tf.TensorSpec(shape=(None, self.rnn_state_dim)))
            self._train = _np_to_tensor(tf.function(self._train.python_function,
                                                    input_signature=signature))

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

        target_variables += self.model_target_rep.trainable_variables
        eval_variables += self.model_rep.trainable_variables

        target_variables += self.model_target_policy.trainable_variables
        eval_variables += self.model_policy.trainable_variables

        [t.assign(tau * e + (1. - tau) * t) for t, e in zip(target_variables, eval_variables)]

    @tf.function
    def get_n_probs(self, n_obses_list, n_selected_actions, rnn_state=None):
        """
        tf.function
        """
        if self.use_rnn:
            n_states, *_ = self.model_rep(n_obses_list, rnn_state)
        else:
            n_states = self.model_rep(n_obses_list)

        policy = self.model_policy(n_states)

        if self.is_discrete:
            policy_prob = policy.prob(tf.argmax(n_selected_actions, axis=-1))  # [Batch, n]
        else:
            policy_prob = tf.exp(squash_correction_log_prob(policy, tf.atanh(n_selected_actions)))
            # [Batch, n, action_dim]
            policy_prob = tf.reduce_prod(policy_prob, axis=-1)  # [Batch, n]

        return policy_prob

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32),
                                  tf.TensorSpec(shape=None, dtype=tf.float32)])
    def _update_reward_bound(self, min_reward, max_reward):
        self.min_cum_reward.assign(tf.minimum(self.min_cum_reward, min_reward))
        self.max_cum_reward.assign(tf.maximum(self.max_cum_reward, max_reward))

    def update_reward_bound(self, n_rewards):
        # n_rewards [1, episode_len]
        ep_len = n_rewards.shape[1]
        gamma = np.array([[np.power(self.gamma, i)] for i in range(ep_len)], dtype=np.float32)

        cum_rewards = np.empty(ep_len, dtype=np.float32)
        tmp_rewards = n_rewards * gamma

        for i in range(ep_len):
            cum_rewards[i] = np.sum(tmp_rewards.diagonal(i))

        cum_n_rewards = np.cumsum(tmp_rewards.diagonal(0))
        cum_rewards = np.concatenate([cum_rewards, cum_n_rewards])

        self._update_reward_bound(np.min(cum_rewards), np.max(cum_rewards))

    @tf.function
    def _get_y(self, n_states, n_actions, n_rewards, state_, n_dones,
               n_mu_probs=None):
        """
        tf.function
        Get target value
        """
        if self.use_reward_normalization:
            n_rewards = n_rewards / tf.maximum(self.max_cum_reward, tf.abs(self.min_cum_reward))

        gamma_ratio = [[tf.pow(self.gamma, i) for i in range(self.n_step)]]
        lambda_ratio = [[tf.pow(self._lambda, i) for i in range(self.n_step)]]
        alpha = tf.exp(self.log_alpha)

        next_n_states = tf.concat([n_states[:, 1:, ...], tf.reshape(state_, (-1, 1, state_.shape[-1]))], axis=1)

        policy = self.model_policy(n_states)
        next_policy = self.model_policy(next_n_states)

        if self.use_curiosity:
            approx_next_n_states = self.model_forward(n_states, n_actions)
            in_n_rewards = tf.reduce_sum(tf.math.squared_difference(approx_next_n_states, next_n_states), axis=2) * 0.5
            in_n_rewards = in_n_rewards * self.curiosity_strength
            n_rewards = n_rewards + in_n_rewards

        if self.is_discrete:
            q1 = self.model_target_q1(n_states)  # [Batch, n, action_dim]
            q2 = self.model_target_q2(n_states)

            next_q1 = self.model_target_q1(next_n_states)  # [Batch, n, action_dim]
            next_q2 = self.model_target_q2(next_n_states)

            min_q = tf.minimum(q1, q2)
            min_next_q = tf.minimum(next_q1, next_q2)

            probs = tf.nn.softmax(policy.logits)
            next_probs = tf.nn.softmax(next_policy.logits)
            clipped_probs = tf.maximum(probs, 1e-8)
            clipped_next_probs = tf.maximum(next_probs, 1e-8)
            tmp_v = min_q - alpha * tf.math.log(clipped_probs)  # [Batch, n, action_dim]
            tmp_next_v = min_next_q - alpha * tf.math.log(clipped_next_probs)

            v = tf.reduce_sum(probs * tmp_v, axis=-1)  # [Batch, n]
            next_v = tf.reduce_sum(next_probs * tmp_next_v, axis=-1)
        else:
            n_actions_sampled = policy.sample()  # [Batch, n, action_dim]
            n_actions_log_prob = tf.reduce_sum(squash_correction_log_prob(policy, n_actions_sampled), axis=2)  # [Batch, n]
            next_n_actions_sampled = next_policy.sample()
            next_n_actions_log_prob = tf.reduce_sum(squash_correction_log_prob(next_policy, next_n_actions_sampled), axis=2)

            q1 = self.model_target_q1(n_states, tf.tanh(n_actions_sampled))  # [Batch, n, 1]
            q2 = self.model_target_q2(n_states, tf.tanh(n_actions_sampled))
            q1 = tf.squeeze(q1, [-1])  # [Batch, n]
            q2 = tf.squeeze(q2, [-1])

            next_q1 = self.model_target_q1(next_n_states, tf.tanh(next_n_actions_sampled))  # [Batch, n, 1]
            next_q2 = self.model_target_q2(next_n_states, tf.tanh(next_n_actions_sampled))
            next_q1 = tf.squeeze(next_q1, [-1])  # [Batch, n]
            next_q2 = tf.squeeze(next_q2, [-1])

            min_q = tf.minimum(q1, q2)
            min_next_q = tf.minimum(next_q1, next_q2)

            v = min_q - alpha * n_actions_log_prob  # [Batch, n]
            next_v = min_next_q - alpha * next_n_actions_log_prob

        td_error = n_rewards + self.gamma * (1 - n_dones) * next_v - v  # [Batch, n]
        td_error = gamma_ratio * td_error
        if self.use_n_step_is:
            td_error = lambda_ratio * td_error

            _policy = self.model_policy(n_states)
            if self.is_discrete:
                n_pi_probs = _policy.prob(tf.argmax(n_actions, axis=-1))  # [Batch, n]
            else:
                n_pi_probs = tf.exp(squash_correction_log_prob(_policy, tf.atanh(n_actions)))
                # [Batch, n, action_dim]
                n_pi_probs = tf.reduce_prod(n_pi_probs, axis=-1)
                # [Batch, n]
            # ρ_t
            n_step_is = tf.clip_by_value(n_pi_probs / tf.maximum(n_mu_probs, 1e-8), 0, 1.)  # [Batch, n]

            # \prod{c_i}
            cumulative_n_step_is = tf.math.cumprod(n_step_is, axis=1)
            td_error = n_step_is * cumulative_n_step_is * td_error

        # \sum{td_error}
        r = tf.reduce_sum(td_error, axis=1, keepdims=True)  # [Batch, 1]

        # V_s + \sum{td_error}
        y = v[:, 0:1] + r  # [Batch, 1]

        return y  # [None, 1]

    @tf.function
    def _train(self, n_obses_list, n_actions, n_rewards, next_obs_list, n_dones,
               n_mu_probs=None, priority_is=None,
               initial_rnn_state=None):
        """
        tf.function
        """
        if self.global_step % self.update_target_per_step == 0:
            self._update_target_variables(tau=self.tau)

        with tf.GradientTape(persistent=True) as tape:
            m_obses_list = [tf.concat([n_obses, tf.reshape(next_obs, (-1, 1, *next_obs.shape[1:]))], axis=1)
                            for n_obses, next_obs in zip(n_obses_list, next_obs_list)]
            if self.use_rnn:
                m_states, *_ = self.model_rep(m_obses_list, initial_rnn_state)
                m_target_states, *_ = self.model_target_rep(m_obses_list, initial_rnn_state)
            else:
                m_states = self.model_rep(m_obses_list)
                m_target_states = self.model_target_rep(m_obses_list)

            n_states = m_states[:, :-1, ...]
            state = m_states[:, self.burn_in_step, ...]

            action = n_actions[:, self.burn_in_step, ...]

            alpha = tf.exp(self.log_alpha)

            policy = self.model_policy(state)
            target_policy = self.model_target_policy(state)
            if self.is_discrete:
                q1 = self.model_q1(state)  # [Batch, action_dim]
                q2 = self.model_q2(state)  # [Batch, action_dim]
            else:
                action_sampled = policy.sample()  # [Batch, action_dim]
                q1 = self.model_q1(state, action)  # [Batch, 1]
                q1_for_gradient = self.model_q1(state, tf.tanh(action_sampled))
                target_q1 = self.model_target_q1(state, action)
                q2 = self.model_q2(state, action)  # [Batch, 1]
                q2_for_gradient = self.model_q2(state, tf.tanh(action_sampled))
                target_q2 = self.model_target_q2(state, action)

            y = tf.stop_gradient(self._get_y(m_target_states[:, self.burn_in_step:-1, ...],
                                             n_actions[:, self.burn_in_step:, ...],
                                             n_rewards[:, self.burn_in_step:],
                                             m_target_states[:, -1, ...],
                                             n_dones[:, self.burn_in_step:],
                                             n_mu_probs[:, self.burn_in_step:] if self.use_n_step_is else None))

            if self.is_discrete:
                q1_single = tf.reduce_sum(action * q1, axis=-1, keepdims=True)  # [Batch, 1]
                q2_single = tf.reduce_sum(action * q2, axis=-1, keepdims=True)
                loss_q1 = tf.square(q1_single - y)
                loss_q2 = tf.square(q2_single - y)
            else:
                clipped_q1 = target_q1 + tf.clip_by_value(
                    q1 - target_q1,
                    -self.clip_epsilon,
                    self.clip_epsilon,
                )
                clipped_q2 = target_q2 + tf.clip_by_value(
                    q2 - target_q2,
                    -self.clip_epsilon,
                    self.clip_epsilon,
                )

                loss_q1_a = tf.square(clipped_q1 - y)
                loss_q2_a = tf.square(clipped_q2 - y)

                loss_q1_b = tf.square(q1 - y)
                loss_q2_b = tf.square(q2 - y)

                loss_q1 = tf.maximum(loss_q1_a, loss_q1_b)
                loss_q2 = tf.maximum(loss_q2_a, loss_q2_b)

                # loss_q1 = tf.square(q1 - y)
                # loss_q2 = tf.square(q2 - y)

            if self.use_priority:
                loss_q1 *= priority_is
                loss_q2 *= priority_is

            loss_q1 = 0.5 * tf.reduce_mean(loss_q1)
            loss_q2 = 0.5 * tf.reduce_mean(loss_q2)

            loss_rep = loss_rep_q = loss_q1 + loss_q2

            loss_mse = tf.keras.losses.MeanSquaredError()
            if self.use_prediction:
                extra_obs = self.model_transition.extra_obs(n_obses_list)[:, self.burn_in_step:, ...]
                approx_next_state_dist = self.model_transition(tf.concat([n_states[:, self.burn_in_step:, ...],
                                                                          extra_obs], axis=-1),
                                                               n_actions[:, self.burn_in_step:, ...])
                loss_transition = -approx_next_state_dist.log_prob(m_target_states[:, self.burn_in_step + 1:, ...])
                # loss_transition = -tf.maximum(loss_transition, -2.)
                std_normal = tfp.distributions.Normal(tf.zeros_like(approx_next_state_dist.loc),
                                                      tf.ones_like(approx_next_state_dist.scale))
                loss_transition += 0.8 * tfp.distributions.kl_divergence(approx_next_state_dist, std_normal)
                loss_transition = tf.reduce_mean(loss_transition)

                approx_n_rewards = self.model_reward(m_states[:, self.burn_in_step + 1:, ...])
                loss_reward = 0.5 * loss_mse(approx_n_rewards, tf.expand_dims(n_rewards[:, self.burn_in_step:], 2))

                loss_obs = 0.5 * self.model_observation.get_loss(m_states[:, self.burn_in_step:, ...],
                                                                 [m_obses[:, self.burn_in_step:, ...] for m_obses in m_obses_list])

                loss_rep = loss_rep_q + loss_transition + loss_reward + loss_obs

            if self.use_curiosity:
                approx_next_n_states = self.model_forward(n_states[:, self.burn_in_step:, ...],
                                                          n_actions[:, self.burn_in_step:, ...])
                next_n_states = m_states[:, self.burn_in_step + 1:, ...]
                loss_forward = loss_mse(approx_next_n_states, next_n_states)

            if self.is_discrete:
                # policy, q1, q2: [Batch, action_dim]
                probs = tf.nn.softmax(policy.logits)
                clipped_probs = tf.maximum(probs, 1e-8)
                loss_policy = alpha * tf.math.log(clipped_probs) - tf.minimum(q1, q2)
                loss_policy = tf.reduce_sum(probs * loss_policy, axis=1, keepdims=True)
                # [Batch, 1]

                loss_alpha = -alpha * (tf.math.log(clipped_probs) - self.action_dim)  # [Batch, action_dim]
                loss_alpha = tf.reduce_sum(probs * loss_alpha, axis=1, keepdims=True)  # [Batch, 1]
            else:
                log_prob = tf.reduce_sum(squash_correction_log_prob(policy, action_sampled), axis=1, keepdims=True)
                loss_policy = alpha * log_prob - tf.minimum(q1_for_gradient, q2_for_gradient)
                # [Batch, 1]

                loss_alpha = -alpha * (log_prob - self.action_dim)  # [Batch, 1]

            loss_policy = tf.reduce_mean(loss_policy)
            loss_alpha = tf.reduce_mean(loss_alpha)

        # Compute gradients and optimize loss
        grads_q1 = tape.gradient(loss_q1, self.model_q1.trainable_variables)
        self.optimizer_q1.apply_gradients(zip(grads_q1, self.model_q1.trainable_variables))

        grads_q2 = tape.gradient(loss_q2, self.model_q2.trainable_variables)
        self.optimizer_q2.apply_gradients(zip(grads_q2, self.model_q2.trainable_variables))

        rep_variables = self.model_rep.trainable_variables
        if self.use_prediction:
            rep_variables += self.model_transition.trainable_variables
            rep_variables += self.model_reward.trainable_variables
            rep_variables += self.model_observation.trainable_variables
        grads_rep = tape.gradient(loss_rep, rep_variables)
        self.optimizer_rep.apply_gradients(zip(grads_rep, rep_variables))

        if self.use_curiosity:
            grads_forward = tape.gradient(loss_forward, self.model_forward.trainable_variables)
            self.optimizer_forward.apply_gradients(zip(grads_forward, self.model_forward.trainable_variables))

        grads_policy = tape.gradient(loss_policy, self.model_policy.trainable_variables)
        self.optimizer_policy.apply_gradients(zip(grads_policy, self.model_policy.trainable_variables))

        if self.use_auto_alpha:
            grads_alpha = tape.gradient(loss_alpha, self.log_alpha)
            self.optimizer_alpha.apply_gradients([(grads_alpha, self.log_alpha)])

        del tape

        self.global_step.assign_add(1)

        summary = {
            'scalar': {
                'loss/y': tf.reduce_mean(y),
                'loss/rep_q': loss_rep_q,
                'loss/Q1': loss_q1,
                'loss/Q2': loss_q2,
                'loss/policy': loss_policy,
                'loss/entropy': tf.reduce_mean(policy.entropy()),
                'loss/alpha': alpha,
            },
            'image': {}
        }
        if self.use_curiosity:
            summary['scalar']['loss/forward'] = loss_forward

        if self.use_prediction:
            summary['scalar']['loss/transition'] = tf.reduce_mean(approx_next_state_dist.entropy())
            summary['scalar']['loss/reward'] = loss_reward
            summary['scalar']['loss/observation'] = loss_obs

            approx_obs_list = self.model_observation(m_states[0:1, self.burn_in_step:, ...])
            assert isinstance(approx_obs_list, list)
            for approx_obs in approx_obs_list:
                if len(approx_obs.shape) > 3:
                    summary['image']['observation'] = tf.reshape(approx_obs, [-1, *approx_obs.shape[2:]])

        return summary

    @tf.function
    def get_n_rnn_states(self, n_obses_list, rnn_state):
        """
        tf.function
        """
        *_, n_rnn_states = self.model_rep(n_obses_list, rnn_state)
        return n_rnn_states

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
            return tf.tanh(policy.sample())

    @tf.function
    def choose_rnn_action(self, obs_list, rnn_state):
        """
        tf.function
        obs_list: list([None, obs_dim_i], ...)
        rnn_state: [None, rnn_state]
        """
        obs_list = [tf.reshape(obs, (-1, 1, *obs.shape[1:])) for obs in obs_list]
        state, next_rnn_state, _ = self.model_rep(obs_list, rnn_state)
        policy = self.model_policy(state)
        if self.is_discrete:
            action = policy.sample()
            action = tf.one_hot(action, self.action_dim)
        else:
            action = tf.tanh(policy.sample())

        action = tf.reshape(action, (-1, action.shape[-1]))
        return action, next_rnn_state

    @tf.function
    def _cal_cem_reward(self, state, action):
        cem_horizon = 12

        if self.cem_rewards is None:
            self.cem_rewards = tf.Variable(tf.zeros([state.shape[0], cem_horizon]))

        for j in range(cem_horizon):
            state_ = self.model_transition(state, action).sample()
            self.cem_rewards[:, j:j + 1].assign(self.model_reward(state_))
            state = state_
            action = tf.tanh(self.model_policy(state).sample())

        return self.cem_rewards

    @tf.function
    def choose_action_by_cem(self, obs_list, rnn_state):
        """
        tf.function
        obs_list: list([None, obs_dim_i], ...)
        rnn_state: [None, rnn_state]
        """
        if self.use_rnn:
            obs_list = [tf.reshape(obs, (-1, 1, obs.shape[-1])) for obs in obs_list]
            state, next_rnn_state, _ = self.model_rep(obs_list, rnn_state)
        else:
            state = self.model_rep(obs_list)

        state = tf.reshape(state, (-1, state.shape[-1]))

        repeat = 1000
        top = 100
        iteration = 10

        batch = state.shape[0]
        dist = self.model_policy(state)
        mean, std = dist.loc, dist.scale

        for i in range(iteration):
            state_repeated = tf.repeat(state, repeat, axis=0)
            mean_repeated = tf.repeat(mean, repeat, axis=0)
            std_repeated = tf.repeat(std, repeat, axis=0)

            action_repeated = tfp.distributions.Normal(mean_repeated, std_repeated)
            action_repeated = tf.tanh(action_repeated.sample())

            rewards = self._cal_cem_reward(state_repeated, action_repeated)

            cum_reward = tf.reshape(tf.reduce_sum(rewards, axis=1), [batch, repeat])
            sorted_index = tf.argsort(cum_reward, axis=1)
            sorted_index = sorted_index[..., -top:]
            sorted_index = tf.reshape(sorted_index, [-1])
            tmp_index = tf.repeat(tf.range(batch), top, axis=0)

            action_repeated = tf.reshape(action_repeated, [batch, repeat, 2])
            action_repeated = tf.gather_nd(action_repeated, tf.unstack([tmp_index, sorted_index], axis=1))
            action_repeated = tf.reshape(action_repeated, [batch, top, 2])
            mean = tf.reduce_mean(tf.atanh(action_repeated * 0.9999), axis=1)
            std = tf.math.reduce_std(tf.atanh(action_repeated * 0.9999), axis=1)

        action = tfp.distributions.Normal(mean, std)
        action = tf.tanh(action.sample())

        if self.use_rnn:
            return action, next_rnn_state
        else:
            return action

    @tf.function
    def get_td_error(self,
                     n_obses_list,
                     n_actions,
                     n_rewards,
                     next_obs_list,
                     n_dones,
                     n_mu_probs=None,
                     rnn_state=None):
        """
        tf.function
        """
        m_obses_list = [tf.concat([n_obses, tf.reshape(next_obs, (-1, 1, *next_obs.shape[1:]))], axis=1)
                        for n_obses, next_obs in zip(n_obses_list, next_obs_list)]
        if self.use_rnn:
            tmp_states, *_ = self.model_rep([m_obses[:, :self.burn_in_step + 1, ...] for m_obses in m_obses_list],
                                            rnn_state)
            state = tmp_states[:, self.burn_in_step, ...]
            m_target_states, *_ = self.model_target_rep(m_obses_list, rnn_state)
        else:
            state = self.model_rep([m_obses[:, self.burn_in_step, ...] for m_obses in m_obses_list])
            m_target_states = self.model_target_rep(m_obses_list)

        action = n_actions[:, self.burn_in_step, ...]

        if self.is_discrete:
            q1 = self.model_q1(state)  # [Batch, action_dim]
            q2 = self.model_q2(state)
            q1 = tf.reduce_sum(action * q1, axis=-1, keepdims=True)  # [Batch, 1]
            q2 = tf.reduce_sum(action * q2, axis=-1, keepdims=True)
        else:
            q1 = self.model_q1(state, action)  # [Batch, 1]
            q2 = self.model_q2(state, action)

        y = self._get_y(m_target_states[:, self.burn_in_step:-1, ...],
                        n_actions[:, self.burn_in_step:, ...],
                        n_rewards[:, self.burn_in_step:],
                        m_target_states[:, -1, ...],
                        n_dones[:, self.burn_in_step:],
                        n_mu_probs[:, self.burn_in_step:] if self.use_n_step_is else None)
        # [Batch, 1]
        q1_td_error = tf.abs(q1 - y)
        q2_td_error = tf.abs(q2 - y)

        td_error = tf.reduce_mean(tf.concat([q1_td_error, q2_td_error], axis=1),
                                  axis=1, keepdims=True)
        return td_error

    def get_episode_td_error(self,
                             n_obses_list,
                             n_actions,
                             n_rewards,
                             next_obs_list,
                             n_dones,
                             n_mu_probs=None,
                             n_rnn_states=None):
        """
        n_obses_list: list([1, episode_len, obs_dim_i], ...)
        n_actions: [1, episode_len, action_dim]
        n_rewards: [1, episode_len]
        next_obs_list: list([1, obs_dim_i], ...)
        n_dones: [1, episode_len]
        n_rnn_states: [1, episode_len, rnn_state_dim]
        """
        ignore_size = self.burn_in_step + self.n_step

        tmp_n_obses_list = [None] * len(n_obses_list)
        for i, n_obses in enumerate(n_obses_list):
            tmp_n_obses_list[i] = np.concatenate([n_obses[:, i:i + ignore_size]
                                                  for i in range(n_obses.shape[1] - ignore_size + 1)], axis=0)
        n_actions = np.concatenate([n_actions[:, i:i + ignore_size]
                                    for i in range(n_actions.shape[1] - ignore_size + 1)], axis=0)
        n_rewards = np.concatenate([n_rewards[:, i:i + ignore_size]
                                    for i in range(n_rewards.shape[1] - ignore_size + 1)], axis=0)
        tmp_next_obs_list = [None] * len(next_obs_list)
        for i, n_obses in enumerate(n_obses_list):
            tmp_next_obs_list[i] = np.concatenate([n_obses[:, i + ignore_size]
                                                   for i in range(n_obses.shape[1] - ignore_size)]
                                                  + [next_obs_list[i]],
                                                  axis=0)
        n_dones = np.concatenate([n_dones[:, i:i + ignore_size]
                                  for i in range(n_dones.shape[1] - ignore_size + 1)], axis=0)

        if self.use_n_step_is:
            n_mu_probs = np.concatenate([n_mu_probs[:, i:i + ignore_size]
                                         for i in range(n_mu_probs.shape[1] - ignore_size + 1)], axis=0)
        if self.use_rnn:
            rnn_state = np.concatenate([n_rnn_states[:, i]
                                        for i in range(n_rnn_states.shape[1] - ignore_size + 1)], axis=0)

        td_error = self.get_td_error(n_obses_list=tmp_n_obses_list,
                                     n_actions=n_actions,
                                     n_rewards=n_rewards,
                                     next_obs_list=tmp_next_obs_list,
                                     n_dones=n_dones,
                                     n_mu_probs=n_mu_probs if self.use_n_step_is else None,
                                     rnn_state=rnn_state if self.use_rnn else None).numpy()
        td_error = td_error.flatten()
        td_error = np.concatenate([td_error,
                                   np.zeros(ignore_size, dtype=np.float32)])
        return td_error

    def save_model(self, iteration):
        self.ckpt_manager.save(iteration + self.init_iteration)

    def write_constant_summaries(self, constant_summaries, iteration):
        """
        Write constant information like reward, iteration from sac_main.py
        """
        with self.summary_writer.as_default():
            for s in constant_summaries:
                tf.summary.scalar(s['tag'], s['simple_value'], step=iteration + self.init_iteration)
        self.summary_writer.flush()

    def fill_replay_buffer(self,
                           n_obses_list,
                           n_actions,
                           n_rewards,
                           next_obs_list,
                           n_dones,
                           n_rnn_states=None):
        """
        n_obses_list: list([1, episode_len, obs_dim_i], ...)
        n_actions: [1, episode_len, action_dim]
        n_rewards: [1, episode_len]
        next_obs_list: list([1, obs_dim_i], ...)
        n_dones: [1, episode_len]
        n_rnn_states: [1, episode_len, rnn_state_dim]
        """
        if self.use_reward_normalization:
            self.update_reward_bound(n_rewards)

        # Ignore episodes whose length is too short
        if n_obses_list[0].shape[1] < self.burn_in_step + self.n_step:
            return

        # Reshape [1, episode_len, ...] to [episode_len, ...]
        obs_list = [n_obses.reshape([-1, *n_obses.shape[2:]]) for n_obses in n_obses_list]
        action = n_actions.reshape([-1, n_actions.shape[-1]])
        reward = n_rewards.reshape([-1])
        done = n_dones.reshape([-1])

        # Padding next_obs for episode experience replay
        obs_list = [np.concatenate([obs, next_obs]) for obs, next_obs in zip(obs_list, next_obs_list)]
        action = np.concatenate([action,
                                 np.empty([1, action.shape[-1]], dtype=np.float32)])
        reward = np.concatenate([reward,
                                 np.zeros([1], dtype=np.float32)])
        done = np.concatenate([done,
                               np.zeros([1], dtype=np.float32)])

        storage_data = {f'obs_{i}': obs for i, obs in enumerate(obs_list)}
        storage_data = {
            **storage_data,
            'action': action,
            'reward': reward,
            'done': done,
        }

        if self.use_n_step_is:
            if self.use_rnn:
                n_mu_probs = self.get_n_probs(n_obses_list, n_actions,
                                              n_rnn_states[:, 0, ...]).numpy()
            else:
                n_mu_probs = self.get_n_probs(n_obses_list, n_actions).numpy()

            mu_prob = n_mu_probs.reshape([-1])
            mu_prob = np.concatenate([mu_prob,
                                      np.empty([1], dtype=np.float32)])
            storage_data['mu_prob'] = mu_prob

        if self.use_rnn:
            rnn_state = n_rnn_states.reshape([-1, n_rnn_states.shape[-1]])
            rnn_state = np.concatenate([rnn_state,
                                        np.empty([1, rnn_state.shape[-1]], dtype=np.float32)])
            storage_data['rnn_state'] = rnn_state

        # n_step transitions except the first one and the last obs_, n_step - 1 + 1
        if self.use_add_with_td:
            td_error = self.get_episode_td_error(n_obses_list=n_obses_list,
                                                 n_actions=n_actions,
                                                 n_rewards=n_rewards,
                                                 next_obs_list=next_obs_list,
                                                 n_dones=n_dones,
                                                 n_mu_probs=n_mu_probs if self.use_n_step_is else None,
                                                 n_rnn_states=n_rnn_states if self.use_rnn else None)
            self.replay_buffer.add_with_td_error(td_error, storage_data, self.burn_in_step + self.n_step)
        else:
            self.replay_buffer.add(storage_data, ignore_size=self.burn_in_step + self.n_step)

    def train(self):
        # Sample from replay buffer
        sampled = self.replay_buffer.sample()
        if sampled is None:
            return

        """
        trans:
            obs_i: [Batch, obs_dim_i]
            action: [Batch, action_dim]
            reward: [Batch, ]
            done: [Batch, ]
            mu_prob: [Batch, ]
            rnn_state: [Batch, rnn_state_dim]
        """
        pointers, trans, priority_is = sampled

        # Get n_step transitions
        trans = {k: [v] for k, v in trans.items()}
        # k: [v, v, ...]
        for i in range(1, self.burn_in_step + self.n_step + 1):
            t_trans = self.replay_buffer.get_storage_data(pointers + i).items()
            for k, v in t_trans:
                trans[k].append(v)

        for k, v in trans.items():
            trans[k] = np.concatenate([np.expand_dims(t, 1) for t in v], axis=1)

        """
        m_obses_list: list([Batch, N + 1, obs_dim_i])
        m_actions: [Batch, N + 1, action_dim]
        m_rewards: [Batch, N + 1]
        m_dones: [Batch, N + 1]
        m_mu_probs: [Batch, N + 1]
        m_rnn_states: [Batch, N + 1, rnn_state_dim]
        """
        m_obses_list = [trans[f'obs_{i}'] for i in range(len(self.obs_dims))]
        m_actions = trans['action']
        m_rewards = trans['reward']
        m_dones = trans['done']

        n_obses_list = [m_obses[:, :-1, ...] for m_obses in m_obses_list]
        n_actions = m_actions[:, :-1, ...]
        n_rewards = m_rewards[:, :-1]
        next_obs_list = [m_obses[:, -1, ...] for m_obses in m_obses_list]
        n_dones = m_dones[:, :-1]

        if self.use_n_step_is:
            m_mu_probs = trans['mu_prob']
            n_mu_probs = m_mu_probs[:, :-1]

        if self.use_rnn:
            m_rnn_states = trans['rnn_state']
            rnn_state = m_rnn_states[:, 0, ...]

        summary = self._train(n_obses_list=n_obses_list,
                              n_actions=n_actions,
                              n_rewards=n_rewards,
                              next_obs_list=next_obs_list,
                              n_dones=n_dones,
                              n_mu_probs=n_mu_probs if self.use_n_step_is else None,
                              priority_is=priority_is if self.use_priority else None,
                              initial_rnn_state=rnn_state if self.use_rnn else None)

        step = self.global_step - 1
        if self.summary_writer is not None and step % self.write_summary_per_step == 0:
            with self.summary_writer.as_default():
                for k, v in summary['scalar'].items():
                    tf.summary.scalar(k, v, step=step)
                for k, v in summary['image'].items():
                    tf.summary.image(k, v, max_outputs=self.n_step, step=step)

            self.summary_writer.flush()

        if self.use_n_step_is:
            n_pi_probs = self.get_n_probs(n_obses_list,
                                          n_actions,
                                          rnn_state=rnn_state if self.use_rnn else None).numpy()

        # Update td_error
        if self.use_priority:
            td_error = self.get_td_error(n_obses_list=n_obses_list,
                                         n_actions=n_actions,
                                         n_rewards=n_rewards,
                                         next_obs_list=next_obs_list,
                                         n_dones=n_dones,
                                         n_mu_probs=n_pi_probs if self.use_n_step_is else None,
                                         rnn_state=rnn_state if self.use_rnn else None).numpy()
            self.replay_buffer.update(pointers, td_error)

        # Update rnn_state
        if self.use_rnn:
            pointers_list = [pointers + i for i in range(1, self.burn_in_step + self.n_step + 1)]
            tmp_pointers = np.stack(pointers_list, axis=1).reshape(-1)
            n_rnn_states = self.get_n_rnn_states(n_obses_list, rnn_state).numpy()
            rnn_states = n_rnn_states.reshape(-1, n_rnn_states.shape[-1])
            self.replay_buffer.update_transitions(tmp_pointers, 'rnn_state', rnn_states)

        # Update n_mu_probs
        if self.use_n_step_is:
            pointers_list = [pointers + i for i in range(0, self.burn_in_step + self.n_step)]
            tmp_pointers = np.stack(pointers_list, axis=1).reshape(-1)
            pi_probs = n_pi_probs.reshape(-1)
            self.replay_buffer.update_transitions(tmp_pointers, 'mu_prob', pi_probs)

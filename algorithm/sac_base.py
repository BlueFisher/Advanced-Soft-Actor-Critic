import functools
import logging
import time
import sys

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer, EpisodeBuffer
from .trans_cache import TransCache

logger = logging.getLogger('sac.base')
logger.setLevel(level=logging.INFO)


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
                 use_prediction=True,

                 seed=None,
                 write_summary_per_step=20,
                 tau=0.005,
                 update_target_per_step=1,
                 init_log_alpha=-2.3,
                 use_auto_alpha=True,
                 lr=3e-4,
                 gamma=0.99,
                 _lambda=0.9,
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
        self.use_prediction = use_prediction

        self.write_summary_per_step = write_summary_per_step
        self.tau = tau
        self.update_target_per_step = update_target_per_step
        self.use_auto_alpha = use_auto_alpha
        self.gamma = gamma
        self._lambda = _lambda
        self.use_priority = use_priority
        self.use_n_step_is = use_n_step_is

        self.zero_init_states = False

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

            if self.use_rnn and self.use_prediction:
                self.episode_buffer = EpisodeBuffer(16, 32, 50)  # TODO

            self._trans_cache = TransCache(56)

            self._init_train_concrete_function(self.replay_buffer.batch_size,
                                               self.episode_buffer.batch_size if self.use_rnn and self.use_prediction else None)

    def _build_model(self, lr, init_log_alpha, model):
        """
        Initialize variables, network models and optimizers
        """
        self.log_alpha = tf.Variable(init_log_alpha, dtype=tf.float32, name='log_alpha')
        self.global_step = tf.Variable(0, dtype=tf.int64, name='global_step')

        self.optimizer_q1 = tf.keras.optimizers.Adam(lr)
        self.optimizer_q2 = tf.keras.optimizers.Adam(lr)
        self.optimizer_policy = tf.keras.optimizers.Adam(lr)
        if self.use_auto_alpha:
            self.optimizer_alpha = tf.keras.optimizers.Adam(lr)

        if self.use_rnn:
            self.model_rnn = model.ModelRNN(self.state_dim)
            self.model_target_rnn = model.ModelRNN(self.state_dim)

            tmp_state, tmp_rnn_state = self.model_rnn.get_call_result_tensors()
            state_dim = tmp_state.shape[-1]
            self._initial_rnn_state = np.zeros(tmp_rnn_state.shape[-1], dtype=np.float32)

            self.optimizer_rnn = tf.keras.optimizers.Adam(lr)

            if self.use_prediction:
                self.model_prediction = model.ModelPrediction(self.state_dim, state_dim, self.action_dim)
                self.optimizer_prediction = tf.keras.optimizers.Adam(lr)
                # # avoid ValueError: tf.function-decorated function tried to create variables on non-first call.
                # zero_grads = [np.zeros_like(v.numpy()) for v in self.model_prediction.trainable_variables]
                # self.optimizer_prediction.apply_gradients(zip(zero_grads, self.model_prediction.trainable_variables))
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
                ckpt.optimizer_prediction = self.optimizer_prediction
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

    def _init_train_concrete_function(self, batch_size, episode_batch_size=None):
        batch_size = None
        step_size = self.burn_in_step + self.n_step
        args = {
            'n_states': tf.TensorSpec(shape=[batch_size, step_size, self.state_dim], dtype=tf.float32),
            'n_actions': tf.TensorSpec(shape=[batch_size, step_size, self.action_dim], dtype=tf.float32),
            'n_rewards': tf.TensorSpec(shape=[batch_size, step_size], dtype=tf.float32),
            'state_': tf.TensorSpec(shape=[batch_size, self.state_dim], dtype=tf.float32),
            'done': tf.TensorSpec(shape=[batch_size, 1], dtype=tf.float32)
        }
        if self.use_n_step_is:
            args['mu_n_probs'] = tf.TensorSpec(shape=[batch_size, step_size, self.action_dim], dtype=tf.float32)
        if self.use_priority:
            args['priority_is'] = tf.TensorSpec(shape=[batch_size, 1], dtype=tf.float32)
        if self.use_rnn:
            args['initial_rnn_state'] = tf.TensorSpec(shape=[batch_size, self._initial_rnn_state.shape[-1]], dtype=tf.float32)

            if self.use_prediction:
                args['ep_m_states'] = tf.TensorSpec(shape=[episode_batch_size, batch_size, self.state_dim], dtype=tf.float32)
                args['ep_n_actions'] = tf.TensorSpec(shape=[episode_batch_size, batch_size, self.action_dim], dtype=tf.float32)
                args['ep_rnn_state'] = tf.TensorSpec(shape=[episode_batch_size, self._initial_rnn_state.shape[-1]], dtype=tf.float32)

        self._train_fn = self._train.get_concrete_function(**args)

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
    def _get_y(self, n_states, n_actions, n_rewards, state_, done,
               mu_n_probs=None):
        """
        tf.function
        get target value
        """

        ratio = [[(self.gamma * self._lambda)**i for i in range(self.n_step)]]
        n_dones = tf.concat([tf.zeros([tf.shape(n_states)[0], self.n_step - 1]), done], axis=1)
        alpha = tf.exp(self.log_alpha)

        policy = self.model_policy(n_states)
        n_actions_sampled = policy.sample()
        n_actions_log_prob = tf.reduce_prod(policy.log_prob(n_actions_sampled), axis=2)

        next_n_states = tf.concat([n_states[:, 1:, :], tf.reshape(state_, (-1, 1, state_.shape[-1]))], axis=1)
        next_policy = self.model_policy(next_n_states)
        next_n_actions_sampled = next_policy.sample()
        next_n_actions_log_prob = tf.reduce_prod(next_policy.log_prob(next_n_actions_sampled), axis=2)

        q1 = self.model_target_q1(n_states, n_actions_sampled)
        q2 = self.model_target_q2(n_states, n_actions_sampled)
        q1 = tf.squeeze(q1, [-1])
        q2 = tf.squeeze(q2, [-1])

        next_q1 = self.model_target_q1(next_n_states, next_n_actions_sampled)
        next_q2 = self.model_target_q2(next_n_states, next_n_actions_sampled)
        next_q1 = tf.squeeze(next_q1, [-1])
        next_q2 = tf.squeeze(next_q2, [-1])

        td_error = n_rewards + self.gamma * (1 - n_dones) * (tf.minimum(next_q1, next_q2) - alpha * next_n_actions_log_prob) - (tf.minimum(q1, q2) - alpha * n_actions_log_prob)
        td_error = ratio * td_error
        if self.use_n_step_is:
            pi_n_probs = self.get_n_step_probs(n_states, n_actions)
            n_step_is = tf.clip_by_value(pi_n_probs / mu_n_probs, 0, 1.)
            n_step_is = tf.reduce_prod(n_step_is, axis=2)

            cumulative_n_step_is = tf.concat([tf.reduce_prod(n_step_is[:, 0:i + 1], axis=1, keepdims=True) for i in range(n_step_is.shape[1])], axis=1)
            td_error = n_step_is * cumulative_n_step_is * td_error
        r = tf.reduce_sum(td_error, axis=1, keepdims=True)

        y = tf.minimum(q1[:, 0:1], q2[:, 0:1]) - alpha * n_actions_log_prob[:, 0:1] + r

        return y  # [None, 1]

    @tf.function
    def _train(self, n_states, n_actions, n_rewards, state_, done,
               mu_n_probs=None, priority_is=None,
               initial_rnn_state=None,
               ep_m_states=None, ep_n_actions=None, ep_rnn_state=None):
        """
        tf.function
        """

        with tf.GradientTape(persistent=True) as tape:
            if self.use_rnn:
                m_states = tf.concat([n_states, tf.reshape(state_, (-1, 1, state_.shape[-1]))], axis=1)
                encoded_m_target_states, *_ = self.model_target_rnn(m_states, initial_rnn_state)
                state_ = encoded_m_target_states[:, -1, :]

                encoded_n_states, *_ = self.model_rnn(n_states, initial_rnn_state)
                n_states = encoded_n_states
                state = encoded_n_states[:, self.burn_in_step, :]
            else:
                state = n_states[:, self.burn_in_step, :]

            action = n_actions[:, self.burn_in_step, :]

            alpha = tf.exp(self.log_alpha)

            policy = self.model_policy(state)
            action_sampled = policy.sample()

            q1 = self.model_q1(state, action)
            q1_for_gradient = self.model_q1(state, action_sampled)
            q2 = self.model_q2(state, action)

            y = tf.stop_gradient(self._get_y(n_states[:, self.burn_in_step:, :],
                                             n_actions[:, self.burn_in_step:, :],
                                             n_rewards[:, self.burn_in_step:],
                                             state_, done,
                                             mu_n_probs[:, self.burn_in_step:]))

            loss_q1 = tf.math.squared_difference(q1, y)
            loss_q2 = tf.math.squared_difference(q2, y)

            if self.use_priority:
                loss_q1 *= priority_is
                loss_q2 *= priority_is

            if self.use_rnn:
                loss_rnn = loss_q1 + loss_q2

                if self.use_prediction:
                    encoded_ep_n_states, *_ = self.model_rnn(ep_m_states[:, :-1, :], ep_rnn_state)
                    approx_ep_next_states = self.model_prediction(encoded_ep_n_states,
                                                                  ep_n_actions)
                    loss_prediction = tf.math.squared_difference(approx_ep_next_states,
                                                                 ep_m_states[:, 1:, :])

                    loss_rnn = tf.reduce_mean(loss_rnn) + tf.reduce_mean(loss_prediction)

            loss_policy = alpha * policy.log_prob(action_sampled) - q1_for_gradient
            loss_alpha = -self.log_alpha * policy.log_prob(action_sampled) + self.log_alpha * self.action_dim

        grads_q1 = tape.gradient(loss_q1, self.model_q1.trainable_variables)
        self.optimizer_q1.apply_gradients(zip(grads_q1, self.model_q1.trainable_variables))

        grads_q2 = tape.gradient(loss_q2, self.model_q2.trainable_variables)
        self.optimizer_q1.apply_gradients(zip(grads_q2, self.model_q2.trainable_variables))

        if self.use_rnn:
            grads_rnn = tape.gradient(loss_rnn, self.model_rnn.trainable_variables)
            self.optimizer_rnn.apply_gradients(zip(grads_rnn, self.model_rnn.trainable_variables))

            if self.use_prediction:
                grads_pred = tape.gradient(loss_prediction, self.model_prediction.trainable_variables)
                self.optimizer_prediction.apply_gradients(zip(grads_pred, self.model_prediction.trainable_variables))

        grads_policy = tape.gradient(loss_policy, self.model_policy.trainable_variables)
        self.optimizer_policy.apply_gradients(zip(grads_policy, self.model_policy.trainable_variables))

        if self.use_auto_alpha:
            grads_alpha = tape.gradient(loss_alpha, self.log_alpha)
            self.optimizer_alpha.apply_gradients([(grads_alpha, self.log_alpha)])

        del tape

        if self.summary_writer is not None and self.global_step % self.write_summary_per_step == 0:
            with self.summary_writer.as_default():
                tf.summary.scalar('loss/Q1', tf.reduce_mean(loss_q1), step=self.global_step)
                tf.summary.scalar('loss/Q2', tf.reduce_mean(loss_q2), step=self.global_step)
                if self.use_rnn and self.use_prediction:
                    tf.summary.scalar('loss/prediction', tf.reduce_mean(loss_prediction), step=self.global_step)
                tf.summary.scalar('loss/policy', tf.reduce_mean(loss_policy), step=self.global_step)
                tf.summary.scalar('loss/entropy', tf.reduce_mean(policy.entropy()), step=self.global_step)
                tf.summary.scalar('loss/alpha', alpha, step=self.global_step)

        if self.global_step % self.update_target_per_step == 0:
            self._update_target_variables(tau=self.tau)

        self.global_step.assign_add(1)

    @tf.function()
    def _get_next_rnn_state(self, n_states, rnn_state):
        """
        tf.function
        """
        _, next_rnn_state = self.model_rnn(n_states, [rnn_state])
        return next_rnn_state

    def get_next_rnn_state(self, n_states_list):
        fn = self._get_next_rnn_state.get_concrete_function(tf.TensorSpec(shape=[None, None, self.state_dim], dtype=tf.float32),
                                                            tf.TensorSpec(shape=[None, len(self._initial_rnn_state)], dtype=tf.float32))

        init_rnn_state = self.get_initial_rnn_state(1)
        next_rnn_state = np.empty((len(n_states_list), init_rnn_state.shape[1]))
        for i, n_states in enumerate(n_states_list):
            if n_states.shape[1] == 0:
                next_rnn_state[i] = init_rnn_state
            else:
                next_rnn_state[i] = fn(tf.constant(n_states), tf.constant(init_rnn_state)).numpy()

        return next_rnn_state


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
        encoded_state, next_rnn_state, = self.model_rnn(state, [rnn_state])
        policy = self.model_policy(encoded_state)
        action = policy.sample()
        action = tf.reshape(action, (-1, action.shape[-1]))
        return tf.clip_by_value(action, -1, 1), next_rnn_state

    @tf.function
    def get_td_error(self, n_states, n_actions, n_rewards, state_, done,
                     mu_n_probs=None, rnn_state=None):
        """
        tf.function
        """
        if self.use_rnn:
            m_states = tf.concat([n_states, tf.reshape(state_, (-1, 1, state_.shape[-1]))], axis=1)
            m_states, *_ = self.model_target_rnn(m_states,
                                                 [rnn_state])
            state_ = m_states[:, -1, :]

            n_states, *_ = self.model_rnn(n_states,
                                          [rnn_state])

        state = n_states[:, self.burn_in_step, :]
        action = n_actions[:, self.burn_in_step, :]

        q1 = self.model_q1(state, action)
        q2 = self.model_q2(state, action)
        y = self._get_y(n_states[:, self.burn_in_step:, :],
                        n_actions[:, self.burn_in_step:, :],
                        n_rewards[:, self.burn_in_step:],
                        state_, done,
                        mu_n_probs[:, self.burn_in_step:])

        q1_td_error = tf.abs(q1 - y)
        q2_td_error = tf.abs(q2 - y)
        td_error = tf.reduce_mean(tf.concat([q1_td_error, q2_td_error], axis=1),
                                  axis=1, keepdims=True)

        return td_error

    @tf.function
    def get_n_step_probs(self, n_states, n_actions):
        """
        tf.function
        """
        policy = self.model_policy(n_states)
        policy_prob = policy.prob(n_actions)

        return policy_prob

    @tf.function
    def get_rnn_n_step_probs(self, n_states, n_actions, rnn_state):
        """
        tf.function
        """
        n_states, *_ = self.model_rnn(n_states, [rnn_state])
        return self.get_n_step_probs(n_states, n_actions)

    def save_model(self, iteration):
        self.ckpt_manager.save(iteration + self.init_iteration)

    def write_constant_summaries(self, constant_summaries, iteration):
        with self.summary_writer.as_default():
            for s in constant_summaries:
                tf.summary.scalar(s['tag'], s['simple_value'], step=iteration + self.init_iteration)
        self.summary_writer.flush()

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
            self._trans_cache.add(n_states, n_actions, n_rewards, state_, done, rnn_state)
        else:
            self._trans_cache.add(n_states, n_actions, n_rewards, state_, done)

        trans = self._trans_cache.get_batch_trans()
        if trans is None:
            return

        if self.use_rnn:
            n_states, n_actions, n_rewards, state_, done, rnn_state = trans

            if self.zero_init_states:
                rnn_state = self.get_initial_rnn_state(len(n_states))
            mu_n_probs = self.get_rnn_n_step_probs(n_states, n_actions,  # TODO: only need [:, burn_in_step:, :]
                                                   rnn_state).numpy()
            self.replay_buffer.add(n_states, n_actions, n_rewards, state_, done, mu_n_probs, rnn_state)
        else:
            n_states, n_actions, n_rewards, state_, done = trans

            mu_n_probs = self.get_n_step_probs(n_states, n_actions).numpy()
            self.replay_buffer.add(n_states, n_actions, n_rewards, state_, done, mu_n_probs)

    def fill_episode_replay_buffer(self, n_states, n_actions, n_rewards, state_, done):
        """
        n_states: [1, burn_in_step + n_step, state_dim]
        n_states: [1, burn_in_step + n_step, state_dim]
        n_rewards: [1, burn_in_step + n_step]
        state_: [1, state_dim]
        done: [1, 1]
        """
        assert self.use_rnn and self.use_prediction
        assert len(n_states) == len(n_actions) == len(n_rewards) == len(state_) == len(done)

        self.episode_buffer.add(n_states, n_actions, n_rewards, state_, done)

    def train(self):
        # sample from replay buffer
        sampled = self.replay_buffer.sample()
        if sampled is None:
            return

        if self.use_rnn and self.use_prediction:
            ep_trans = self.episode_buffer.sample(self.get_next_rnn_state)
            if ep_trans is None:
                return

        if self.use_priority:
            pointers, trans, priority_is = sampled
        else:
            pointers, trans = sampled

        if self.use_rnn:
            n_states, n_actions, n_rewards, state_, done, mu_n_probs, rnn_state = trans
        else:
            n_states, n_actions, n_rewards, state_, done, mu_n_probs = trans

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
                ep_m_states, ep_n_actions, ep_rnn_state = ep_trans
                fn_args['ep_m_states'] = ep_m_states
                fn_args['ep_n_actions'] = ep_n_actions
                fn_args['ep_rnn_state'] = ep_rnn_state

        self._train_fn(**{k: tf.constant(fn_args[k], dtype=tf.float32) for k in fn_args})

        self.summary_writer.flush()

        if self.use_n_step_is or self.use_priority:
            if self.use_rnn:
                pi_n_probs = self.get_rnn_n_step_probs(n_states, n_actions,
                                                       rnn_state).numpy()
            else:
                pi_n_probs = self.get_n_step_probs(n_states, n_actions).numpy()

        # update td_error
        if self.use_priority:
            if self.use_rnn:
                td_error = self.get_td_error(n_states, n_actions, n_rewards, state_, done,
                                             pi_n_probs if self.use_n_step_is else None,
                                             rnn_state).numpy()
            else:
                td_error = self.get_td_error(n_states, n_actions, n_rewards, state_, done,
                                             pi_n_probs if self.use_n_step_is else None).numpy()

            self.replay_buffer.update(pointers, td_error.flatten())

        # update mu_n_probs
        if self.use_n_step_is:
            self.replay_buffer.update_transitions(pointers, 5, pi_n_probs)

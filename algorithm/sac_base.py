import logging
import math
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import algorithm.constants as C

from .cpu2gpu_buffer import CPU2GPUBuffer
from .replay_buffer import PrioritizedReplayBuffer
from .utils import *

logger = logging.getLogger('sac.base')


class SAC_Base(object):
    _last_save_time = 0

    def __init__(self,
                 obs_dims,
                 d_action_dim,
                 c_action_dim,
                 model_abs_dir,
                 model,
                 summary_path='log',
                 train_mode=True,
                 last_ckpt=None,

                 seed=None,
                 write_summary_per_step=1e3,
                 save_model_per_step=1e5,
                 save_model_per_minute=5,

                 ensemble_q_num=2,
                 ensemble_q_sample=2,

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

                 discrete_dqn_like=False,
                 use_priority=True,
                 use_n_step_is=True,
                 use_prediction=False,
                 transition_kl=0.8,
                 use_extra_data=True,
                 use_curiosity=False,
                 curiosity_strength=1,
                 use_rnd=False,
                 rnd_n_sample=10,
                 use_normalization=False,

                 replay_config=None):
        """
        obs_dims: List of dimensions of observations
        action_dim: Dimension of action
        model_abs_dir: The directory that saves summary, checkpoints, config etc.
        model: Custom Model Class
        train_mode: Is training or inference
        last_ckpt: The checkpoint to restore

        seed: Random seed
        write_summary_per_step: Write summaries in TensorBoard every `write_summary_per_step` steps
        save_model_per_step: Save model every N steps
        save_model_per_minute: Save model every N minutes

        ensemble_q_num: 2 # Number of Qs
        ensemble_q_sample: 2 # Number of min Qs

        burn_in_step: Burn-in steps in R2D2
        n_step: Update Q function by `n_step` steps
        use_rnn: If use RNN

        tau: Coefficient of updating target network
        update_target_per_step: Update target network every 'update_target_per_step' steps
        init_log_alpha: The initial log_alpha
        use_auto_alpha: If use automating entropy adjustment
        learning_rate: Learning rate of all optimizers
        gamma: Discount factor
        v_lambda: Discount factor for V-trace
        v_rho: Rho for V-trace
        v_c: C for V-trace
        clip_epsilon: Epsilon for q and policy clip

        use_priority: If use PER importance ratio
        use_n_step_is: If use importance sampling
        use_prediction: If train a transition model
        transition_kl: The coefficient of KL of transition and standard normal
        use_extra_data: If use extra data to train prediction model
        use_curiosity: If use curiosity
        curiosity_strength: Curiosity strength if use curiosity
        use_rnd: If use RND
        rnd_n_sample: RND sample times
        use_normalization: If use observation normalization
        """

        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

        self.obs_dims = obs_dims
        self.d_action_dim = d_action_dim
        self.c_action_dim = c_action_dim
        self.train_mode = train_mode

        self.ensemble_q_num = ensemble_q_num
        self.ensemble_q_sample = ensemble_q_sample

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

        self.discrete_dqn_like = discrete_dqn_like
        self.use_priority = use_priority
        self.use_n_step_is = use_n_step_is
        self.use_prediction = use_prediction
        self.transition_kl = transition_kl
        self.use_extra_data = use_extra_data
        self.use_curiosity = use_curiosity
        self.curiosity_strength = curiosity_strength
        self.use_rnd = use_rnd
        self.rnd_n_sample = rnd_n_sample
        self.use_normalization = use_normalization

        self.action_dim = self.d_action_dim + self.c_action_dim

        self.use_add_with_td = False

        if seed is not None:
            tf.random.set_seed(seed)

        if model_abs_dir:
            summary_path = Path(model_abs_dir).joinpath(summary_path)
            self.summary_writer = tf.summary.create_file_writer(str(summary_path))

        if self.train_mode:
            replay_config = {} if replay_config is None else replay_config
            self.replay_buffer = PrioritizedReplayBuffer(**replay_config)

        self._build_model(model, init_log_alpha, learning_rate)
        self._init_or_restore(model_abs_dir, last_ckpt)

        self._init_tf_function()
        self._train_data_buffer = CPU2GPUBuffer(self._sample,
                                                self.get_train_input_signature(self.replay_buffer.batch_size),
                                                can_return_None=True)

    def _build_model(self, model, init_log_alpha, learning_rate):
        """
        Initialize variables, network models and optimizers
        """
        self.log_alpha_d = tf.Variable(init_log_alpha, dtype=tf.float32, name='log_alpha_d')
        self.log_alpha_c = tf.Variable(init_log_alpha, dtype=tf.float32, name='log_alpha_c')

        self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step')

        def adam_optimizer(): return tf.keras.optimizers.Adam(learning_rate)

        if self.use_auto_alpha:
            self.optimizer_alpha = adam_optimizer()

        if self.use_normalization:
            self._create_normalizer()

            p_self = self

            # When tensorflow executes ModelRep.call, it will add a kwarg 'training'
            # automatically. But if the subclass does not specify the 'training' argument,
            # it will throw exception when calling super().call(obs_list, *args, **kwargs)

            import inspect

            has_training = 'training' in inspect.signature(model.ModelRep.call).parameters.keys()

            class ModelRep(model.ModelRep):
                def call(self, obs_list, *args, **kwargs):
                    obs_list = [
                        tf.clip_by_value(
                            (obs - p_self.running_means)
                            / tf.sqrt(
                                p_self.running_variances / (tf.cast(p_self.normalizer_step, tf.float32) + 1)
                            ),
                            -5, 5
                        ) for obs in obs_list
                    ]

                    if 'training' in kwargs and not has_training:
                        del kwargs['training']

                    return super().call(obs_list, *args, **kwargs)
        else:
            ModelRep = model.ModelRep

        if self.use_rnn:
            # Get represented state dimension
            self.model_rep = ModelRep(self.obs_dims, self.d_action_dim, self.c_action_dim)
            self.model_target_rep = ModelRep(self.obs_dims, self.d_action_dim, self.c_action_dim)
            # Get state and rnn_state dimension
            state, next_rnn_state = self.model_rep.init()
            self.rnn_state_dim = next_rnn_state.shape[-1]
        else:
            # Get represented state dimension
            self.model_rep = ModelRep(self.obs_dims)
            self.model_target_rep = ModelRep(self.obs_dims)
            # Get state dimension
            state = self.model_rep.init()
            self.rnn_state_dim = 1
        state_dim = state.shape[-1]
        logger.info(f'State Dimension: {state_dim}')
        self.optimizer_rep = adam_optimizer()

        # PRMs
        if self.use_prediction:
            self.model_transition = model.ModelTransition(state_dim,
                                                          self.d_action_dim,
                                                          self.c_action_dim,
                                                          self.use_extra_data)
            self.model_reward = model.ModelReward(state_dim,
                                                  self.use_extra_data)
            self.model_observation = model.ModelObservation(state_dim, self.obs_dims,
                                                            self.use_extra_data)

            self.optimizer_prediction = adam_optimizer()

        if self.use_curiosity:
            self.model_forward = model.ModelForward(state_dim, self.action_dim)
            self.optimizer_forward = adam_optimizer()

        if self.use_rnd:
            self.model_rnd = model.ModelRND(state_dim, self.d_action_dim + self.c_action_dim)
            self.model_target_rnd = model.ModelRND(state_dim, self.d_action_dim + self.c_action_dim)
            self.optimizer_rnd = tf.keras.optimizers.Adam(learning_rate)

        self.model_q_list = [model.ModelQ(state_dim, self.d_action_dim, self.c_action_dim, f'q{i}')
                             for i in range(self.ensemble_q_num)]
        self.model_target_q_list = [model.ModelQ(state_dim, self.d_action_dim, self.c_action_dim, f'target_q{i}')
                                    for i in range(self.ensemble_q_num)]
        self.optimizer_q_list = [adam_optimizer() for _ in range(self.ensemble_q_num)]

        self.model_policy = model.ModelPolicy(state_dim, self.d_action_dim, self.c_action_dim, 'policy')
        self.optimizer_policy = adam_optimizer()

    def _create_normalizer(self):
        self.normalizer_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='normalizer_step')
        self.running_means = []
        self.running_variances = []
        for dim in self.obs_dims:
            self.running_means.append(
                tf.Variable(tf.zeros(dim), trainable=False, name='running_means'))
            self.running_variances.append(
                tf.Variable(tf.ones(dim), trainable=False, name='running_variances'))

    def _init_or_restore(self, model_abs_dir, last_ckpt):
        """
        Initialize network weights from scratch or restore from model_abs_dir
        """
        ckpt_saved = {
            'log_alpha_d': self.log_alpha_d,
            'log_alpha_c': self.log_alpha_c,
            'global_step': self.global_step,

            'model_rep': self.model_rep,
            'model_target_rep': self.model_target_rep,
            'model_policy': self.model_policy,

            'optimizer_rep': self.optimizer_rep,
            'optimizer_policy': self.optimizer_policy
        }

        for i in range(self.ensemble_q_num):
            ckpt_saved[f'model_q{i}'] = self.model_q_list[i]
            ckpt_saved[f'model_target_q{i}'] = self.model_target_q_list[i]
            ckpt_saved[f'optimizer_q{i}'] = self.optimizer_q_list[i]

        if self.use_normalization:
            ckpt_saved['normalizer_step'] = self.normalizer_step
            for i, v in enumerate(self.running_means):
                ckpt_saved[f'running_means_{i}'] = v
            for i, v in enumerate(self.running_variances):
                ckpt_saved[f'running_variances_{i}'] = v

        if self.use_prediction:
            ckpt_saved['model_transition'] = self.model_transition
            ckpt_saved['model_reward'] = self.model_reward
            ckpt_saved['model_observation'] = self.model_observation
            ckpt_saved['optimizer_prediction'] = self.optimizer_prediction

        if self.use_auto_alpha:
            ckpt_saved['optimizer_alpha'] = self.optimizer_alpha

        if self.use_curiosity:
            ckpt_saved['model_forward'] = self.model_forward
            ckpt_saved['optimizer_forward'] = self.optimizer_forward

        if self.use_rnd:
            ckpt_saved['model_rnd'] = self.model_rnd
            ckpt_saved['optimizer_rnd'] = self.optimizer_rnd

        # Execute init() of all models from nn_models
        for m in ckpt_saved.values():
            if isinstance(m, tf.keras.Model):
                m.init()

        if model_abs_dir:
            ckpt = tf.train.Checkpoint(**ckpt_saved)
            ckpt_dir = Path(model_abs_dir).joinpath('model')
            self.ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=10)

            if self.ckpt_manager.latest_checkpoint:
                if last_ckpt is None:
                    latest_checkpoint = self.ckpt_manager.latest_checkpoint
                else:
                    i = str.rindex(self.ckpt_manager.latest_checkpoint, '-')
                    latest_checkpoint = self.ckpt_manager.latest_checkpoint[:i] + f'-{last_ckpt}'
                ckpt.restore(latest_checkpoint)
                logger.info(f'Restored from {latest_checkpoint}')
                self.init_iteration = int(latest_checkpoint.split('-')[-1])
            else:
                ckpt.restore(None)
                logger.info('Initializing from scratch')
                self.init_iteration = 0
                self._update_target_variables()

    def get_train_input_signature(self, batch_size=None):
        None_tensor = tf.TensorSpec((0, ))

        step_size = self.burn_in_step + self.n_step
        """ _train
        n_obses_list, n_actions, n_rewards, next_obs_list, n_dones,
        n_mu_probs=None, priority_is=None,
        initial_rnn_state=None """
        signature = [
            [tf.TensorSpec(shape=(batch_size, step_size, *t)) for t in self.obs_dims],
            tf.TensorSpec(shape=(batch_size, step_size, self.action_dim)),
            tf.TensorSpec(shape=(batch_size, step_size)),
            [tf.TensorSpec(shape=(batch_size, *t)) for t in self.obs_dims],
            tf.TensorSpec(shape=(batch_size, step_size)),
            tf.TensorSpec(shape=(batch_size, step_size)) if self.use_n_step_is else None_tensor,
            tf.TensorSpec(shape=(batch_size, 1)) if self.use_priority else None_tensor,
            tf.TensorSpec(shape=(batch_size, self.rnn_state_dim)) if self.use_rnn else None_tensor,
        ]

        return signature

    def _init_tf_function(self):
        """
        Initialize some @tf.function and specify tf.TensorSpec
        """

        None_tensor = tf.TensorSpec((0, ))

        """ _udpate_normalizer
        obses_list """
        if self.use_normalization:
            self._udpate_normalizer = np_to_tensor(
                tf.function(self._udpate_normalizer.python_function, input_signature=[
                    [tf.TensorSpec(shape=(None, *t)) for t in self.obs_dims]
                ]))

        """ get_n_probs
        n_obses_list, n_selected_actions, rnn_state=None """
        tmp_get_n_probs = tf.function(self.get_n_probs.python_function, input_signature=[
            [tf.TensorSpec(shape=(None, None, *t)) for t in self.obs_dims],
            tf.TensorSpec(shape=(None, None, self.action_dim)),
            tf.TensorSpec(shape=(None, self.rnn_state_dim)) if self.use_rnn else None_tensor,
        ])
        self.get_n_probs = np_to_tensor(tmp_get_n_probs)

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
            tf.TensorSpec(shape=(None, step_size)),
            tf.TensorSpec(shape=(None, step_size)) if self.use_n_step_is else None_tensor,
            tf.TensorSpec(shape=(None, self.rnn_state_dim)) if self.use_rnn else None_tensor,
        ]
        self.get_td_error = np_to_tensor(tf.function(self.get_td_error.python_function,
                                                     input_signature=signature))

        if self.train_mode:
            signature = self.get_train_input_signature(None)
            self._train = np_to_tensor(tf.function(self._train.python_function,
                                                   input_signature=signature))

    def get_initial_rnn_state(self, batch_size):
        assert self.use_rnn

        return np.zeros([batch_size, self.rnn_state_dim], dtype=np.float32)

    @tf.function
    def _update_target_variables(self, tau=1.):
        """
        soft update target networks (default hard)
        """
        target_variables, eval_variables = [], []

        for i in range(self.ensemble_q_num):
            target_variables += self.model_target_q_list[i].trainable_variables
            eval_variables += self.model_q_list[i].trainable_variables

        target_variables += self.model_target_rep.trainable_variables
        eval_variables += self.model_rep.trainable_variables

        [t.assign(tau * e + (1. - tau) * t) for t, e in zip(target_variables, eval_variables)]

    @tf.function
    def _udpate_normalizer(self, obs_list):
        self.normalizer_step.assign(self.normalizer_step + tf.shape(obs_list[0])[0])

        input_to_old_means = [tf.subtract(obs_list[i], self.running_means[i]) for i in range(len(obs_list))]
        new_means = [self.running_means[i] + tf.reduce_sum(
            input_to_old_means[i] / tf.cast(self.normalizer_step, dtype=tf.float32), axis=0
        ) for i in range(len(obs_list))]

        input_to_new_means = [tf.subtract(obs_list[i], new_means[i]) for i in range(len(obs_list))]
        new_variance = [self.running_variances[i] + tf.reduce_sum(
            input_to_new_means[i] * input_to_old_means[i], axis=0
        ) for i in range(len(obs_list))]

        [self.running_means[i].assign(new_means[i]) for i in range(len(obs_list))]
        [self.running_variances[i].assign(new_variance[i]) for i in range(len(obs_list))]

    @tf.function
    def get_n_probs(self, n_obses_list, n_selected_actions, rnn_state=None):
        """
        tf.function
        """
        if self.use_rnn:
            n_states, _ = self.model_rep(n_obses_list,
                                         gen_pre_n_actions(n_selected_actions),
                                         rnn_state)
        else:
            n_states = self.model_rep(n_obses_list)

        d_policy, c_policy = self.model_policy(n_states)

        policy_prob = tf.ones((tf.shape(n_states)[:2]))  # [Batch, n]

        if self.d_action_dim:
            n_selected_d_actions = n_selected_actions[..., :self.d_action_dim]
            policy_prob *= d_policy.prob(tf.argmax(n_selected_d_actions, axis=-1))  # [Batch, n]

        if self.c_action_dim:
            n_selected_c_actions = n_selected_actions[..., self.d_action_dim:]
            c_policy_prob = squash_correction_prob(c_policy, tf.atanh(n_selected_c_actions))
            # [Batch, n, action_dim]
            policy_prob *= tf.reduce_prod(c_policy_prob, axis=-1)  # [Batch, n]

        return policy_prob

    @tf.function
    def get_dqn_like_d_y(self, n_rewards, n_dones,
                         stacked_next_q, stacked_next_target_q):
        """
        n_rewards: [Batch, n]
        n_dones: [Batch, n]
        stacked_next_q: [ensemble_q_sample, Batch, n, d_action_dim]
        stacked_next_target_q: [ensemble_q_sample, Batch, n, d_action_dim]
        """
        gamma_ratio = [[tf.pow(self.gamma, i) for i in range(self.n_step)]]

        stacked_next_q = stacked_next_q[..., -1, :]  # [ensemble_q_sample, Batch, d_action_dim]
        stacked_next_target_q = stacked_next_target_q[..., -1, :]  # [ensemble_q_sample, Batch, d_action_dim]

        done = n_dones[:, -1:]  # [Batch, 1]

        mask_stacked_q = tf.one_hot(tf.argmax(stacked_next_q, axis=-1), self.d_action_dim)
        # [ensemble_q_sample, Batch, d_action_dim]

        stacked_max_next_target_q = tf.reduce_sum(stacked_next_target_q * mask_stacked_q,
                                                  axis=-1,
                                                  keepdims=True)
        # [ensemble_q_sample, Batch, 1]

        next_q = tf.reduce_min(stacked_max_next_target_q, axis=0)
        # [Batch, 1]

        g = tf.reduce_sum(gamma_ratio * n_rewards, axis=-1, keepdims=True)  # [Batch, 1]
        y = g + tf.pow(self.gamma, self.n_step) * next_q * (1 - done)  # [Batch, 1]

        return y

    @tf.function
    def _v_trace(self, n_rewards, n_dones,
                 n_mu_probs, n_pi_probs,
                 v, next_v):
        """
        n_rewards: [Batch, n]
        n_dones: [Batch, n]
        n_mu_probs: [Batch, n]
        n_pi_probs: [Batch, n]
        v: [Batch, n]
        next_v: [Batch, n],
        """

        gamma_ratio = [[tf.pow(self.gamma, i) for i in range(self.n_step)]]
        lambda_ratio = [[tf.pow(self.v_lambda, i) for i in range(self.n_step)]]

        td_error = n_rewards + self.gamma * (1 - n_dones) * next_v - v  # [Batch, n]
        td_error = gamma_ratio * td_error

        if self.use_n_step_is:
            td_error = lambda_ratio * td_error

            n_step_is = n_pi_probs / tf.maximum(n_mu_probs, 1e-8)

            # ρ_t, t \in [s, s+n-1]
            rho = tf.minimum(n_step_is, self.v_rho)  # [Batch, n]

            # \prod{c_i}, i \in [s, t-1]
            c = tf.minimum(n_step_is, self.v_c)
            c = tf.concat([tf.ones((tf.shape(n_step_is)[0], 1)), c[..., :-1]], axis=-1)
            c = tf.math.cumprod(c, axis=1)

            # \prod{c_i} * ρ_t * td_error
            td_error = c * rho * td_error

        # \sum{td_error}
        r = tf.reduce_sum(td_error, axis=1, keepdims=True)  # [Batch, 1]

        # V_s + \sum{td_error}
        y = v[:, 0:1] + r  # [Batch, 1]

        return y

    @tf.function
    def _get_y(self, n_states, n_actions, n_rewards, state_, n_dones,
               n_mu_probs=None):
        """
        tf.function
        Get target value
        """

        alpha_d = tf.exp(self.log_alpha_d)
        alpha_c = tf.exp(self.log_alpha_c)

        next_n_states = tf.concat([n_states[:, 1:, ...], tf.reshape(state_, (-1, 1, state_.shape[-1]))], axis=1)

        d_policy, c_policy = self.model_policy(n_states)
        next_d_policy, next_c_policy = self.model_policy(next_n_states)

        if self.use_curiosity:
            approx_next_n_states = self.model_forward(n_states, n_actions)
            in_n_rewards = tf.reduce_sum(tf.math.squared_difference(approx_next_n_states, next_n_states), axis=2) * 0.5
            in_n_rewards = in_n_rewards * self.curiosity_strength
            n_rewards += in_n_rewards

        if self.c_action_dim:
            n_c_actions_sampled = c_policy.sample()  # [Batch, n, action_dim]
            next_n_c_actions_sampled = next_c_policy.sample()
        else:
            n_c_actions_sampled = tf.zeros((0,))
            next_n_c_actions_sampled = tf.zeros((0,))

        # ([Batch, n, action_dim], [Batch, n, 1])
        q_list = [q(n_states, tf.tanh(n_c_actions_sampled)) for q in self.model_target_q_list]
        next_q_list = [q(next_n_states, tf.tanh(next_n_c_actions_sampled)) for q in self.model_target_q_list]

        d_q_list = [q[0] for q in q_list]  # [Batch, n, action_dim]
        c_q_list = [q[1] for q in q_list]  # [Batch, n, 1]

        next_d_q_list = [q[0] for q in next_q_list]  # [Batch, n, action_dim]
        next_c_q_list = [q[1] for q in next_q_list]  # [Batch, n, 1]

        d_y, c_y = tf.zeros((0,)), tf.zeros((0,))

        if self.d_action_dim:
            stacked_next_d_q = tf.gather(next_d_q_list,
                                         tf.random.shuffle(tf.range(self.ensemble_q_num))[:self.ensemble_q_sample])
            # [ensemble_q_num, Batch, n, d_action_dim] -> [ensemble_q_sample, Batch, n, d_action_dim]

            if self.discrete_dqn_like:
                next_d_eval_q_list = [q(next_n_states, tf.tanh(next_n_c_actions_sampled))[0] for q in self.model_q_list]
                stacked_next_d_eval_q = tf.gather(next_d_eval_q_list,
                                                  tf.random.shuffle(tf.range(self.ensemble_q_num))[:self.ensemble_q_sample])
                # [ensemble_q_num, Batch, n, d_action_dim] -> [ensemble_q_sample, Batch, n, d_action_dim]

                d_y = self.get_dqn_like_d_y(n_rewards, n_dones,
                                            stacked_next_d_eval_q,
                                            stacked_next_d_q)
            else:
                stacked_d_q = tf.gather(d_q_list,
                                        tf.random.shuffle(tf.range(self.ensemble_q_num))[:self.ensemble_q_sample])
                # [ensemble_q_num, Batch, n, d_action_dim] -> [ensemble_q_sample, Batch, n, d_action_dim]

                min_q = tf.reduce_min(stacked_d_q, axis=0)  # [Batch, n, d_action_dim]
                min_next_q = tf.reduce_min(stacked_next_d_q, axis=0)  # [Batch, n, d_action_dim]

                probs = tf.nn.softmax(d_policy.logits)  # [Batch, n, action_dim]
                next_probs = tf.nn.softmax(next_d_policy.logits)  # [Batch, n, action_dim]
                clipped_probs = tf.maximum(probs, 1e-8)
                clipped_next_probs = tf.maximum(next_probs, 1e-8)
                tmp_v = min_q - alpha_d * tf.math.log(clipped_probs)  # [Batch, n, action_dim]
                tmp_next_v = min_next_q - alpha_d * tf.math.log(clipped_next_probs)  # [Batch, n, action_dim]

                v = tf.reduce_sum(probs * tmp_v, axis=-1)  # [Batch, n]
                next_v = tf.reduce_sum(next_probs * tmp_next_v, axis=-1)  # [Batch, n]

                if self.use_n_step_is:
                    n_d_actions = n_actions[..., :self.d_action_dim]
                    n_pi_probs = d_policy.prob(tf.argmax(n_d_actions, axis=-1))  # [Batch, n]

                d_y = self._v_trace(n_rewards, n_dones,
                                    n_mu_probs,
                                    n_pi_probs if self.use_n_step_is else tf.zeros((0,)),
                                    v, next_v)

        if self.c_action_dim:
            n_actions_log_prob = tf.reduce_sum(squash_correction_log_prob(c_policy, n_c_actions_sampled), axis=-1)  # [Batch, n]
            next_n_actions_log_prob = tf.reduce_sum(squash_correction_log_prob(next_c_policy, next_n_c_actions_sampled), axis=-1)

            stacked_c_q = tf.gather(c_q_list,
                                    tf.random.shuffle(tf.range(self.ensemble_q_num))[:self.ensemble_q_sample])
            # [ensemble_q_num, Batch, n, 1] -> [ensemble_q_sample, Batch, n, 1]
            stacked_next_c_q = tf.gather(next_c_q_list,
                                         tf.random.shuffle(tf.range(self.ensemble_q_num))[:self.ensemble_q_sample])
            # [ensemble_q_num, Batch, n, 1] -> [ensemble_q_sample, Batch, n, 1]

            min_q = tf.squeeze(tf.reduce_min(stacked_c_q, axis=0), axis=-1)  # [Batch, n]
            min_next_q = tf.squeeze(tf.reduce_min(stacked_next_c_q, axis=0), axis=-1)  # [Batch, n]

            v = min_q - alpha_c * n_actions_log_prob  # [Batch, n]
            next_v = min_next_q - alpha_c * next_n_actions_log_prob  # [Batch, n]

            # v = scale_inverse_h(v)
            # next_v = scale_inverse_h(next_v)

            if self.use_n_step_is:
                n_c_actions = n_actions[..., self.d_action_dim:]
                n_pi_probs = squash_correction_prob(c_policy, tf.atanh(n_c_actions))
                # [Batch, n, action_dim]
                n_pi_probs = tf.reduce_prod(n_pi_probs, axis=-1)  # [Batch, n]

            c_y = self._v_trace(n_rewards, n_dones,
                                n_mu_probs,
                                n_pi_probs if self.use_n_step_is else tf.zeros((0,)),
                                v, next_v)

        return d_y, c_y  # [None, 1]

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
                m_states, _ = self.model_rep(m_obses_list,
                                             gen_pre_n_actions(n_actions, keep_last_action=True),
                                             initial_rnn_state)
                m_target_states, _ = self.model_target_rep(m_obses_list,
                                                           gen_pre_n_actions(n_actions, keep_last_action=True),
                                                           initial_rnn_state)
            else:
                m_states = self.model_rep(m_obses_list)
                m_target_states = self.model_target_rep(m_obses_list)

            n_states = m_states[:, :-1, ...]
            state = m_states[:, self.burn_in_step, ...]

            action = n_actions[:, self.burn_in_step, ...]
            d_action = action[..., :self.d_action_dim]
            c_action = action[..., self.d_action_dim:]

            q_list = [q(state, c_action) for q in self.model_q_list]
            # ([Batch, action_dim], [Batch, 1])
            d_q_list = [q[0] for q in q_list]  # [Batch, action_dim]
            c_q_list = [q[1] for q in q_list]  # [Batch, 1]

            d_y, c_y = self._get_y(m_target_states[:, self.burn_in_step:-1, ...],
                                   n_actions[:, self.burn_in_step:, ...],
                                   n_rewards[:, self.burn_in_step:],
                                   m_target_states[:, -1, ...],
                                   n_dones[:, self.burn_in_step:],
                                   n_mu_probs[:, self.burn_in_step:] if self.use_n_step_is else None)

            d_y, c_y = tf.stop_gradient(d_y), tf.stop_gradient(c_y)
            #  [Batch, 1], [Batch, 1]

            d_policy, c_policy = self.model_policy(state)

            ##### Q LOSS #####

            loss_q_list = [tf.zeros((tf.shape(state)[0], 1)) for _ in range(self.ensemble_q_num)]

            if self.d_action_dim:
                for i in range(self.ensemble_q_num):
                    q_single = tf.reduce_sum(d_action * d_q_list[i], axis=-1, keepdims=True)  # [Batch, 1]
                    loss_q_list[i] += tf.square(q_single - d_y)

            if self.c_action_dim:
                if self.clip_epsilon > 0:
                    target_c_q_list = [q(state, c_action)[1] for q in self.model_target_q_list]

                    clipped_q_list = [target_c_q_list[i] + tf.clip_by_value(
                        c_q_list[i] - target_c_q_list[i],
                        -self.clip_epsilon,
                        self.clip_epsilon,
                    ) for i in range(self.ensemble_q_num)]

                    loss_q_a_list = [tf.square(clipped_q - c_y) for clipped_q in clipped_q_list]
                    loss_q_b_list = [tf.square(q - c_y) for q in c_q_list]

                    for i in range(self.ensemble_q_num):
                        loss_q_list[i] += tf.maximum(loss_q_a_list[i], loss_q_b_list[i])
                else:
                    for i in range(self.ensemble_q_num):
                        loss_q_list[i] += tf.square(c_q_list[i] - c_y)

            if self.use_priority:
                loss_q_list = [loss * priority_is for loss in loss_q_list]

            loss_q_list = [tf.reduce_mean(loss) for loss in loss_q_list]

            loss_rep_q = sum(loss_q_list)

            loss_mse = tf.keras.losses.MeanSquaredError()
            if self.use_prediction:
                if self.use_extra_data:
                    extra_obs = self.model_transition.extra_obs(n_obses_list)[:, self.burn_in_step:, ...]
                    extra_state = tf.concat([n_states[:, self.burn_in_step:, ...], extra_obs], axis=-1)
                    approx_next_state_dist = self.model_transition(extra_state,
                                                                   n_actions[:, self.burn_in_step:, ...])
                else:
                    approx_next_state_dist = self.model_transition(n_states[:, self.burn_in_step:, ...],
                                                                   n_actions[:, self.burn_in_step:, ...])
                loss_transition = -approx_next_state_dist.log_prob(m_target_states[:, self.burn_in_step + 1:, ...])
                # loss_transition = -tf.maximum(loss_transition, -2.)
                std_normal = tfp.distributions.Normal(tf.zeros_like(approx_next_state_dist.loc),
                                                      tf.ones_like(approx_next_state_dist.scale))
                loss_transition += self.transition_kl * tfp.distributions.kl_divergence(approx_next_state_dist, std_normal)
                loss_transition = tf.reduce_mean(loss_transition)

                approx_n_rewards = self.model_reward(m_states[:, self.burn_in_step + 1:, ...])
                loss_reward = loss_mse(approx_n_rewards, tf.expand_dims(n_rewards[:, self.burn_in_step:], 2))

                loss_obs = self.model_observation.get_loss(m_states[:, self.burn_in_step:, ...],
                                                           [m_obses[:, self.burn_in_step:, ...] for m_obses in m_obses_list])

                loss_prediction = loss_transition + loss_reward + loss_obs

            if self.use_curiosity:
                approx_next_n_states = self.model_forward(n_states[:, self.burn_in_step:, ...],
                                                          n_actions[:, self.burn_in_step:, ...])
                next_n_states = m_states[:, self.burn_in_step + 1:, ...]
                loss_forward = loss_mse(approx_next_n_states, next_n_states)

            if self.use_rnd:
                approx_f = self.model_rnd(n_states[:, self.burn_in_step:, ...],
                                          n_actions[:, self.burn_in_step:, ...])
                f = self.model_target_rnd(n_states[:, self.burn_in_step:, ...],
                                          n_actions[:, self.burn_in_step:, ...])
                loss_rnd = tf.reduce_mean(tf.math.squared_difference(f, approx_f))

            ##### ALPHA LOSS & POLICY LOSS #####

            loss_d_policy = tf.zeros((tf.shape(state)[0], 1))
            loss_c_policy = tf.zeros((tf.shape(state)[0], 1))

            loss_d_alpha = tf.zeros((tf.shape(state)[0], 1))
            loss_c_alpha = tf.zeros((tf.shape(state)[0], 1))

            alpha_d = tf.exp(self.log_alpha_d)
            alpha_c = tf.exp(self.log_alpha_c)

            if self.d_action_dim and not self.discrete_dqn_like:
                probs = tf.nn.softmax(d_policy.logits)   # [Batch, action_dim]
                clipped_probs = tf.maximum(probs, 1e-8)

                stacked_d_q = tf.gather(d_q_list,
                                        tf.random.shuffle(tf.range(self.ensemble_q_num))[:self.ensemble_q_sample])
                # [ensemble_q_num, Batch, d_action_dim] -> [ensemble_q_sample, Batch, d_action_dim]
                min_d_q = tf.reduce_min(stacked_d_q, axis=0)
                # [ensemble_q_sample, Batch, d_action_dim] -> [Batch, d_action_dim]

                _loss_policy = alpha_d * tf.math.log(clipped_probs) - min_d_q  # [Batch, d_action_dim]
                loss_d_policy = tf.reduce_sum(probs * _loss_policy, axis=1, keepdims=True)  # [Batch, 1]

                _loss_alpha = -alpha_d * (tf.math.log(clipped_probs) - self.d_action_dim)  # [Batch, action_dim]
                loss_d_alpha = tf.reduce_sum(probs * _loss_alpha, axis=1, keepdims=True)  # [Batch, 1]

            if self.c_action_dim:
                action_sampled = c_policy.sample()
                c_q_for_gradient_list = [q(state, tf.tanh(action_sampled))[1] for q in self.model_q_list]
                # [[Batch, 1], ...]

                stacked_c_q_for_gradient = tf.gather(c_q_for_gradient_list,
                                                     tf.random.shuffle(tf.range(self.ensemble_q_num))[:self.ensemble_q_sample])
                # [ensemble_q_num, Batch, 1] -> [ensemble_q_sample, Batch, 1]

                log_prob = tf.reduce_sum(squash_correction_log_prob(c_policy, action_sampled), axis=1, keepdims=True)
                # [Batch, 1]

                min_c_q_for_gradient = tf.reduce_min(stacked_c_q_for_gradient, axis=0)
                # [ensemble_q_sample, Batch, 1] -> [Batch, 1]

                loss_c_policy = alpha_c * log_prob - min_c_q_for_gradient
                # [Batch, 1]

                loss_c_alpha = -alpha_c * (log_prob - self.c_action_dim)  # [Batch, 1]

            loss_policy = tf.reduce_mean(loss_d_policy + loss_c_policy)
            loss_alpha = tf.reduce_mean(loss_d_alpha + loss_c_alpha)

        # Compute gradients and optimize loss
        for i in range(self.ensemble_q_num):
            grads_q = tape.gradient(loss_q_list[i], self.model_q_list[i].trainable_variables)
            self.optimizer_q_list[i].apply_gradients(zip(grads_q, self.model_q_list[i].trainable_variables))

        rep_variables = self.model_rep.trainable_variables
        grads_rep = tape.gradient(loss_rep_q, rep_variables)

        if self.use_prediction:
            grads_rep_preds = [tape.gradient(loss_transition, rep_variables),
                               tape.gradient(loss_reward, rep_variables),
                               tape.gradient(loss_obs, rep_variables)]

            # for i in range(len(grads_rep)):
            #     grad_rep = grads_rep[i]
            #     grad_rep_norm = tf.norm(grad_rep)
            #     for grads_rep_pred in grads_rep_preds:
            #         grad_rep_pred = grads_rep_pred[i]
            #         cos = tf.reduce_sum(grad_rep * grad_rep_pred) / (grad_rep_norm * tf.norm(grad_rep_pred))
            #         grads_rep[i] += tf.maximum(cos, 0) * grad_rep_pred

            _grads_rep_main = tf.concat([tf.reshape(g, [-1]) for g in grads_rep], axis=0)
            _grads_rep_preds = [tf.concat([tf.reshape(g, [-1]) for g in grads_rep_pred], axis=0)
                                for grads_rep_pred in grads_rep_preds]

            coses = [tf.reduce_sum(_grads_rep_main * grads_rep_pred) / (tf.norm(_grads_rep_main) * tf.norm(grads_rep_pred))
                     for grads_rep_pred in _grads_rep_preds]
            coses = [tf.maximum(0., tf.sign(cos)) for cos in coses]

            for grads_rep_pred, cos in zip(grads_rep_preds, coses):
                for i in range(len(grads_rep_pred)):
                    grads_rep[i] += cos * grads_rep_pred[i]

        self.optimizer_rep.apply_gradients(zip(grads_rep, rep_variables))

        if self.use_prediction:
            prediction_variables = self.model_transition.trainable_variables
            prediction_variables += self.model_reward.trainable_variables
            prediction_variables += self.model_observation.trainable_variables
            grads_prediction = tape.gradient(loss_prediction, prediction_variables)
            self.optimizer_prediction.apply_gradients(zip(grads_prediction, prediction_variables))

        if self.use_curiosity:
            grads_forward = tape.gradient(loss_forward, self.model_forward.trainable_variables)
            self.optimizer_forward.apply_gradients(zip(grads_forward, self.model_forward.trainable_variables))

        if self.use_rnd:
            grads_rnd = tape.gradient(loss_rnd, self.model_rnd.trainable_variables)
            self.optimizer_rnd.apply_gradients(zip(grads_rnd, self.model_rnd.trainable_variables))

        if (self.d_action_dim and not self.discrete_dqn_like) or self.c_action_dim:
            grads_policy = tape.gradient(loss_policy, self.model_policy.trainable_variables)
            self.optimizer_policy.apply_gradients(zip(grads_policy, self.model_policy.trainable_variables))

        if self.use_auto_alpha and ((self.d_action_dim and not self.discrete_dqn_like) or self.c_action_dim):
            grads_alpha = tape.gradient(loss_alpha, [self.log_alpha_d, self.log_alpha_c])
            self.optimizer_alpha.apply_gradients(zip(grads_alpha, [self.log_alpha_d, self.log_alpha_c]))

        del tape

        if self.summary_writer is not None and self.global_step % self.write_summary_per_step == 0:
            with self.summary_writer.as_default():
                tf.summary.scalar('loss/q', tf.reduce_mean(loss_q_list), step=self.global_step)
                if self.d_action_dim:
                    tf.summary.scalar('loss/d_entropy', tf.reduce_mean(d_policy.entropy()), step=self.global_step)
                    tf.summary.scalar('loss/alpha_d', alpha_d, step=self.global_step)
                if self.c_action_dim:
                    tf.summary.scalar('loss/c_entropy', tf.reduce_mean(c_policy.entropy()), step=self.global_step)
                    tf.summary.scalar('loss/alpha_c', alpha_c, step=self.global_step)

                if self.use_curiosity:
                    tf.summary.scalar('loss/forward', loss_forward, step=self.global_step)

                if self.use_rnd:
                    tf.summary.scalar('loss/rnd', loss_rnd, step=self.global_step)

                if self.use_prediction:
                    tf.summary.scalar('loss/transition',
                                      tf.reduce_mean(approx_next_state_dist.entropy()),
                                      step=self.global_step)
                    tf.summary.scalar('loss/reward', loss_reward, step=self.global_step)
                    tf.summary.scalar('loss/observation', loss_obs, step=self.global_step)

                    approx_obs_list = self.model_observation(m_states[0:1, self.burn_in_step:, ...])
                    if not isinstance(approx_obs_list, (list, tuple)):
                        approx_obs_list = [approx_obs_list]
                    for approx_obs in approx_obs_list:
                        if len(approx_obs.shape) > 3:
                            tf.summary.image('observation',
                                             tf.reshape(approx_obs, [-1, *approx_obs.shape[2:]]),
                                             max_outputs=self.n_step, step=self.global_step)

            self.summary_writer.flush()

    @tf.function
    def get_n_rnn_states(self, n_obses_list, n_actions, rnn_state):
        """
        tf.function
        """
        n_rnn_states = list()
        n_actions = gen_pre_n_actions(n_actions)
        for i in range(n_obses_list[0].shape[1]):
            _, rnn_state = self.model_rep([o[:, i:i + 1, ...] for o in n_obses_list],
                                          n_actions[:, i:i + 1, ...],
                                          rnn_state)
            n_rnn_states.append(rnn_state)

        return tf.stack(n_rnn_states, axis=1)

    @tf.function
    def rnd_sample(self, state, d_policy, c_policy):
        n_sample = self.rnd_n_sample

        d_action = tf.one_hot(d_policy.sample(n_sample), self.d_action_dim) if self.d_action_dim \
            else tf.zeros((n_sample, tf.shape(state)[0], self.d_action_dim))
        c_action = tf.tanh(c_policy.sample(n_sample)) if self.c_action_dim \
            else tf.zeros((n_sample, tf.shape(state)[0], self.c_action_dim))

        actions = tf.concat([d_action, c_action], axis=-1)  # [n_sample, batch, action_dim]

        actions = tf.transpose(actions, [1, 0, 2])  # [batch, n_sample, action_dim]
        states = tf.repeat(tf.expand_dims(state, 1), n_sample, axis=1)  # [batch, n_sample, state_dim]
        approx_f = self.model_rnd(states, actions)
        f = self.model_target_rnd(states, actions)  # [batch, n_sample, f]
        loss = tf.reduce_sum(tf.math.squared_difference(f, approx_f), axis=2)  # [batch, n_sample]

        idx = tf.argmax(loss, axis=1, output_type=tf.int32)  # [batch, ]
        idx = tf.stack([tf.range(states.shape[0]), idx], axis=1)  # [batch, 2]

        return tf.gather_nd(actions, idx)

    @tf.function
    def _choose_action(self, state):
        d_policy, c_policy = self.model_policy(state)
        if self.use_rnd:
            return self.rnd_sample(state, d_policy, c_policy)
        else:
            if self.d_action_dim:
                if self.discrete_dqn_like:
                    if tf.random.uniform((1,)) < 0.2:
                        d_action = tf.random.categorical(tf.ones((1, self.d_action_dim)),
                                                         tf.shape(state)[0])[0]
                        d_action = tf.one_hot(d_action, self.d_action_dim)
                    else:
                        d_q, _ = self.model_q_list[0](state, c_policy.sample() if self.c_action_dim else tf.zeros((0,)))
                        d_action = tf.argmax(d_q, axis=-1)
                        d_action = tf.one_hot(d_action, self.d_action_dim)
                else:
                    d_action = tf.one_hot(d_policy.sample(), self.d_action_dim)
            else:
                d_action = tf.zeros((tf.shape(state)[0], 0))

            c_action = tf.tanh(c_policy.sample()) if self.c_action_dim else tf.zeros((tf.shape(state)[0], 0))

            return tf.concat([d_action, c_action], axis=-1)

    @tf.function
    def choose_action(self, obs_list):
        """
        tf.function
        obs_list: list([None, obs_dim_i], ...)
        """
        state = self.model_rep(obs_list)
        return self._choose_action(state)

    @tf.function
    def choose_rnn_action(self, obs_list, pre_action, rnn_state):
        """
        tf.function
        obs_list: list([None, obs_dim_i], ...)
        rnn_state: [None, rnn_state]
        """
        obs_list = [tf.reshape(obs, (-1, 1, *obs.shape[1:])) for obs in obs_list]
        pre_action = tf.reshape(pre_action, (-1, 1, *pre_action.shape[1:]))
        state, next_rnn_state = self.model_rep(obs_list, pre_action, rnn_state)
        state = tf.reshape(state, (-1, state.shape[-1]))

        action = self._choose_action(state)

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
    def choose_action_by_cem(self, obs, rnn_state):
        obs = tf.reshape(obs, (-1, 1, obs.shape[-1]))
        state, next_rnn_state, _ = self.model_rnn(obs, [rnn_state])

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
        return action, next_rnn_state

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
        Return the td-error of (burn-in + n-step) observations (sampled from replay buffer)
        """
        m_obses_list = [tf.concat([n_obses, tf.reshape(next_obs, (-1, 1, *next_obs.shape[1:]))], axis=1)
                        for n_obses, next_obs in zip(n_obses_list, next_obs_list)]
        if self.use_rnn:
            tmp_states, _ = self.model_rep([m_obses[:, :self.burn_in_step + 1, ...] for m_obses in m_obses_list],
                                           gen_pre_n_actions(n_actions[:, :self.burn_in_step + 1, ...]),
                                           rnn_state)
            state = tmp_states[:, self.burn_in_step, ...]
            m_target_states, *_ = self.model_target_rep(m_obses_list,
                                                        gen_pre_n_actions(n_actions,
                                                                          keep_last_action=True),
                                                        rnn_state)
        else:
            state = self.model_rep([m_obses[:, self.burn_in_step, ...] for m_obses in m_obses_list])
            m_target_states = self.model_target_rep(m_obses_list)

        action = n_actions[:, self.burn_in_step, ...]
        d_action = action[..., :self.d_action_dim]
        c_action = action[..., self.d_action_dim:]

        # ([Batch, action_dim], [Batch, 1])
        q_list = [q(state, c_action) for q in self.model_q_list]
        d_q_list = [q[0] for q in q_list]  # [Batch, action_dim]
        c_q_list = [q[1] for q in q_list]  # [Batch, 1]

        if self.d_action_dim:
            d_q_list = [tf.reduce_sum(d_action * q, axis=-1, keepdims=True) for q in d_q_list]
            # [Batch, 1]

        d_y, c_y = self._get_y(m_target_states[:, self.burn_in_step:-1, ...],
                               n_actions[:, self.burn_in_step:, ...],
                               n_rewards[:, self.burn_in_step:],
                               m_target_states[:, -1, ...],
                               n_dones[:, self.burn_in_step:],
                               n_mu_probs[:, self.burn_in_step:] if self.use_n_step_is else None)

        # [Batch, 1]
        q_td_error_list = [tf.zeros((tf.shape(state)[0], 1)) for _ in range(self.ensemble_q_num)]
        if self.d_action_dim:
            for i in range(self.ensemble_q_num):
                q_td_error_list[i] += tf.abs(d_q_list[i] - d_y)

        if self.c_action_dim:
            for i in range(self.ensemble_q_num):
                q_td_error_list[i] += tf.abs(c_q_list[i] - c_y)

        td_error = tf.reduce_mean(tf.concat(q_td_error_list, axis=-1),
                                  axis=-1, keepdims=True)
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

        Return the td-error of raw episode observations
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

        td_error_list = []
        all_batch = tmp_n_obses_list[0].shape[0]
        for i in range(math.ceil(all_batch / C.GET_EPISODE_TD_ERROR_SEG)):
            b_i, b_j = i * C.GET_EPISODE_TD_ERROR_SEG, (i + 1) * C.GET_EPISODE_TD_ERROR_SEG
            td_error = self.get_td_error(n_obses_list=[o[b_i:b_j, :] for o in tmp_n_obses_list],
                                         n_actions=n_actions[b_i:b_j, :],
                                         n_rewards=n_rewards[b_i:b_j, :],
                                         next_obs_list=[o[b_i:b_j, :] for o in tmp_next_obs_list],
                                         n_dones=n_dones[b_i:b_j, :],
                                         n_mu_probs=n_mu_probs[b_i:b_j, :] if self.use_n_step_is else None,
                                         rnn_state=rnn_state[b_i:b_j, :] if self.use_rnn else None).numpy()
            td_error_list.append(td_error.flatten())

        td_error = np.concatenate([*td_error_list,
                                   np.zeros(ignore_size, dtype=np.float32)])
        return td_error

    def write_constant_summaries(self, constant_summaries, iteration=None):
        """
        Write constant information like reward, iteration from sac_main.py
        """
        with self.summary_writer.as_default():
            for s in constant_summaries:
                tf.summary.scalar(s['tag'], s['simple_value'],
                                  step=self.global_step if iteration is None else iteration)

        self.summary_writer.flush()

    def save_model(self):
        self.ckpt_manager.save(self.global_step)
        logger.info(f"Model saved at {self.global_step.numpy()}")

    @tf.function
    def _increase_global_step(self):
        self.global_step.assign_add(1)

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

        # Ignore episodes whose length is too short
        if n_obses_list[0].shape[1] < self.burn_in_step + self.n_step:
            return

        # Reshape [1, episode_len, ...] to [episode_len, ...]
        obs_list = [n_obses.reshape([-1, *n_obses.shape[2:]]) for n_obses in n_obses_list]
        if self.use_normalization:
            self._udpate_normalizer(obs_list)
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
            n_mu_probs = self.get_n_probs(n_obses_list, n_actions,
                                          n_rnn_states[:, 0, ...] if self.use_rnn else None).numpy()

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
            self.replay_buffer.add_with_td_error(td_error, storage_data,
                                                 ignore_size=self.burn_in_step + self.n_step)
        else:
            self.replay_buffer.add(storage_data,
                                   ignore_size=self.burn_in_step + self.n_step)

    def _sample(self):
        # Sample from replay buffer
        sampled = self.replay_buffer.sample()
        if sampled is None:
            return None

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

        # Get n_step transitions TODO: could be faster
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

        return pointers, (n_obses_list,
                          n_actions,
                          n_rewards,
                          next_obs_list,
                          n_dones,
                          n_mu_probs if self.use_n_step_is else None,
                          priority_is if self.use_priority else None,
                          rnn_state if self.use_rnn else None)

    def train(self):
        train_data = self._train_data_buffer.get_data()
        if train_data is None:
            return 0

        pointers, (n_obses_list, n_actions, n_rewards, next_obs_list, n_dones,
                   n_mu_probs,
                   priority_is,
                   rnn_state) = train_data

        self._train(n_obses_list=n_obses_list,
                    n_actions=n_actions,
                    n_rewards=n_rewards,
                    next_obs_list=next_obs_list,
                    n_dones=n_dones,
                    n_mu_probs=n_mu_probs if self.use_n_step_is else None,
                    priority_is=priority_is if self.use_priority else None,
                    initial_rnn_state=rnn_state if self.use_rnn else None)

        step = self.global_step.numpy()

        if step % self.save_model_per_step == 0 \
                and (time.time() - self._last_save_time) / 60 >= self.save_model_per_minute:
            self.save_model()
            self._last_save_time = time.time()

        if self.use_n_step_is:
            n_pi_probs_tensor = self.get_n_probs(n_obses_list,
                                                 n_actions,
                                                 rnn_state=rnn_state if self.use_rnn else None)

        # Update td_error
        if self.use_priority:
            td_error = self.get_td_error(n_obses_list=n_obses_list,
                                         n_actions=n_actions,
                                         n_rewards=n_rewards,
                                         next_obs_list=next_obs_list,
                                         n_dones=n_dones,
                                         n_mu_probs=n_pi_probs_tensor if self.use_n_step_is else None,
                                         rnn_state=rnn_state if self.use_rnn else None).numpy()
            self.replay_buffer.update(pointers, td_error)

        # Update rnn_state
        if self.use_rnn:
            pointers_list = [pointers + i for i in range(1, self.burn_in_step + self.n_step + 1)]
            tmp_pointers = np.stack(pointers_list, axis=1).reshape(-1)
            n_rnn_states = self.get_n_rnn_states(n_obses_list, n_actions, rnn_state).numpy()
            rnn_states = n_rnn_states.reshape(-1, n_rnn_states.shape[-1])
            self.replay_buffer.update_transitions(tmp_pointers, 'rnn_state', rnn_states)

        # Update n_mu_probs
        if self.use_n_step_is:
            pointers_list = [pointers + i for i in range(0, self.burn_in_step + self.n_step)]
            tmp_pointers = np.stack(pointers_list, axis=1).reshape(-1)
            pi_probs = n_pi_probs_tensor.numpy().reshape(-1)
            self.replay_buffer.update_transitions(tmp_pointers, 'mu_prob', pi_probs)

        self._increase_global_step()

        return step + 1

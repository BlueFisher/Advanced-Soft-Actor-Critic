import logging
import sys
import time
from itertools import chain
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.append(str(Path(__file__).resolve().parent.parent))
from algorithm.sac_base import SAC_Base

logger = logging.getLogger('sac.base.ds')


class SAC_DS_Base(SAC_Base):
    def __init__(self,
                 obs_shapes: Tuple,
                 d_action_size: int,
                 c_action_size: int,
                 model_abs_dir: Union[str, None],
                 model,
                 device: Union[str, None] = None,
                 summary_path: str = 'log',
                 train_mode: bool = True,
                 last_ckpt: Union[str, None] = None,

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
                 use_prediction=False,
                 transition_kl=0.8,
                 use_extra_data=True,
                 use_curiosity=False,
                 curiosity_strength=1,
                 use_rnd=False,
                 rnd_n_sample=10,
                 use_normalization=False,

                 noise=0.):

        self.obs_shapes = obs_shapes
        self.d_action_size = d_action_size
        self.c_action_size = c_action_size
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
        self.use_prediction = use_prediction
        self.transition_kl = transition_kl
        self.use_extra_data = use_extra_data
        self.use_curiosity = use_curiosity
        self.curiosity_strength = curiosity_strength
        self.use_rnd = use_rnd
        self.rnd_n_sample = rnd_n_sample
        self.use_normalization = use_normalization
        self.use_priority = True
        self.use_n_step_is = True

        self.noise = noise

        self.device = device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        self.summary_writer = None
        if model_abs_dir:
            summary_path = Path(model_abs_dir).joinpath(summary_path)
            self.summary_writer = SummaryWriter(str(summary_path))

        self._build_model(model, init_log_alpha, learning_rate)
        self._init_or_restore(model_abs_dir, int(last_ckpt) if last_ckpt is not None else None)

    def _random_action(self, action):
        batch = action.shape[0]
        d_action = action[..., :self.d_action_size]
        c_action = action[..., self.d_action_size:]

        if self.d_action_size:
            action_random = np.eye(self.d_action_size)[np.random.randint(0, self.d_action_size, size=batch)]
            cond = np.random.rand(batch) < self.noise
            d_action[cond] = action_random[cond]

        if self.c_action_size:
            c_action = np.tanh(np.arctanh(c_action) + np.random.randn(batch, self.c_action_size) * self.noise)

        return np.concatenate([d_action, c_action], axis=-1)

    def choose_action(self, obs_list):
        action = super().choose_action(obs_list)

        return self._random_action(action)

    def choose_rnn_action(self, obs_list, pre_action, rnn_state):
        action, next_rnn_state = super().choose_rnn_action(obs_list, pre_action, rnn_state)

        return self._random_action(action), next_rnn_state

    def get_policy_variables(self, get_numpy=True):
        """
        For learner to send variables to actors
        """
        variables = chain(self.model_rep.parameters(), self.model_policy.parameters())

        if get_numpy:
            return [v.detach().cpu().numpy() for v in variables]
        else:
            return variables

    def update_policy_variables(self, t_variables: List[np.ndarray]):
        """
        For actor to update its own network from learner
        """
        variables = self.get_policy_variables(get_numpy=False)

        for v, t_v in zip(variables, t_variables):
            v.data.copy_(torch.from_numpy(t_v).to(self.device))

    def get_nn_variables(self, get_numpy=True):
        """
        For learner to send variables to evolver
        """
        variables = chain(self.model_rep.parameters(),
                          self.model_policy.parameters(),
                          [self.log_d_alpha, self.log_c_alpha])

        for model_q in self.model_q_list:
            variables = chain(variables,
                              model_q.parameters())

        if self.use_prediction:
            variables = chain(variables,
                              self.model_transition.parameters(),
                              self.model_reward.parameters(),
                              self.model_observation.parameters())

        if self.use_curiosity:
            variables = chain(variables,
                              self.model_forward.parameters())

        if self.use_rnd:
            variables = chain(variables,
                              self.model_rnd.parameters(),
                              self.model_target_rnd.parameters())

        if get_numpy:
            return [v.detach().cpu().numpy() for v in variables]
        else:
            return variables

    def update_nn_variables(self, t_variables: List[np.ndarray]):
        """
        Update own network from evolver selection
        """
        variables = self.get_nn_variables(get_numpy=False)

        for v, t_v in zip(variables, t_variables):
            v.data.copy_(torch.from_numpy(t_v).to(self.device))

        self._update_target_variables()

    def get_all_variables(self, get_numpy=True):
        variables = self.get_nn_variables(get_numpy=False)
        variables = chain(variables, self.model_target_rep.parameters())

        for model_target_q in self.model_target_q_list:
            variables = chain(variables, model_target_q.parameters())

        if self.use_normalization:
            variables = chain(variables,
                              [self.normalizer_step],
                              self.running_means,
                              self.running_variances)

        if get_numpy:
            return [v.detach().cpu().numpy() for v in variables]
        else:
            return variables

    def update_all_variables(self, t_variables: List[np.ndarray]):
        if any([np.isnan(v.sum()) for v in t_variables]):
            return False

        variables = self.get_all_variables(get_numpy=False)

        for v, t_v in zip(variables, t_variables):
            v.data.copy_(torch.from_numpy(t_v).to(self.device))

        return True

    def train(self,
              pointers,
              n_obses_list,
              n_actions,
              n_rewards,
              next_obs_list,
              n_dones,
              n_mu_probs,
              priority_is,
              rnn_state=None):

        n_obses_list = [torch.from_numpy(t).to(self.device) for t in n_obses_list]
        n_actions = torch.from_numpy(n_actions).to(self.device)
        n_rewards = torch.from_numpy(n_rewards).to(self.device)
        next_obs_list = [torch.from_numpy(t).to(self.device) for t in next_obs_list]
        n_dones = torch.from_numpy(n_dones).to(self.device)
        n_mu_probs = torch.from_numpy(n_mu_probs).to(self.device)
        priority_is = torch.from_numpy(priority_is).to(self.device)
        if self.use_rnn:
            rnn_state = torch.from_numpy(rnn_state).to(self.device)

        self._train(n_obses_list=n_obses_list,
                    n_actions=n_actions,
                    n_rewards=n_rewards,
                    next_obs_list=next_obs_list,
                    n_dones=n_dones,
                    n_mu_probs=n_mu_probs,
                    priority_is=priority_is,
                    initial_rnn_state=rnn_state if self.use_rnn else None)

        step = self.global_step.item()

        if step % self.save_model_per_step == 0 \
                and (time.time() - self._last_save_time) / 60 >= self.save_model_per_minute:
            self.save_model()
            self._last_save_time = time.time()

        n_pi_probs_tensor = self.get_n_probs(n_obses_list,
                                             n_actions,
                                             rnn_state=rnn_state if self.use_rnn else None)

        td_error = self.get_td_error(n_obses_list=n_obses_list,
                                     n_actions=n_actions,
                                     n_rewards=n_rewards,
                                     next_obs_list=next_obs_list,
                                     n_dones=n_dones,
                                     n_mu_probs=n_pi_probs_tensor,
                                     rnn_state=rnn_state if self.use_rnn else None).detach().cpu().numpy()

        update_data = []

        pointers_list = [pointers + i for i in range(0, self.burn_in_step + self.n_step)]
        tmp_pointers = np.stack(pointers_list, axis=1).reshape(-1)
        pi_probs = n_pi_probs_tensor.detach().cpu().numpy().reshape(-1)
        update_data.append((tmp_pointers, 'mu_prob', pi_probs))

        if self.use_rnn:
            pointers_list = [pointers + i for i in range(1, self.burn_in_step + self.n_step + 1)]
            tmp_pointers = np.stack(pointers_list, axis=1).reshape(-1)
            n_rnn_states = self.get_n_rnn_states(n_obses_list, n_actions, rnn_state).detach().cpu().numpy()
            rnn_states = n_rnn_states.reshape(-1, n_rnn_states.shape[-1])
            update_data.append((tmp_pointers, 'rnn_state', rnn_states))

        self._increase_global_step()

        return step + 1, td_error, update_data

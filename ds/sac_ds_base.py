import logging
import sys
import time
from itertools import chain
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from algorithm.utils.enums import CURIOSITY, SIAMESE

sys.path.append(str(Path(__file__).resolve().parent.parent))
from algorithm.sac_base import SAC_Base


class SAC_DS_Base(SAC_Base):
    def __init__(self,
                 obs_shapes: List[Tuple],
                 d_action_size: int,
                 c_action_size: int,
                 model_abs_dir: Optional[str],
                 device: Optional[str] = None,
                 ma_name: Optional[str] = None,
                 summary_path: str = 'log',
                 train_mode: bool = True,
                 last_ckpt: Optional[str] = None,

                 nn_config: Optional[dict] = None,

                 nn = None,
                 seed=None,
                 write_summary_per_step=1e3,
                 save_model_per_step=1e5,

                 ensemble_q_num=2,
                 ensemble_q_sample=2,

                 burn_in_step=0,
                 n_step=1,
                 seq_encoder=None,

                 batch_size=256,
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
                 siamese: Optional[str] = None,
                 siamese_use_q=False,
                 siamese_use_adaptive=False,
                 use_prediction=False,
                 transition_kl=0.8,
                 use_extra_data=True,
                 curiosity: Optional[str] = None,
                 curiosity_strength=1,
                 use_rnd=False,
                 rnd_n_sample=10,
                 use_normalization=False,
                 action_noise: Optional[List[float]] = None):

        self.obs_shapes = obs_shapes
        self.d_action_size = d_action_size
        self.c_action_size = c_action_size
        self.model_abs_dir = model_abs_dir
        self.train_mode = train_mode

        self.ensemble_q_num = ensemble_q_num
        self.ensemble_q_sample = ensemble_q_sample

        self.burn_in_step = burn_in_step
        self.n_step = n_step
        self.seq_encoder = seq_encoder

        self.write_summary_per_step = int(write_summary_per_step)
        self.save_model_per_step = int(save_model_per_step)
        self.batch_size = batch_size
        self.tau = tau
        self.update_target_per_step = update_target_per_step
        self.use_auto_alpha = use_auto_alpha
        self.gamma = gamma
        self.v_lambda = v_lambda
        self.v_rho = v_rho
        self.v_c = v_c
        self.clip_epsilon = clip_epsilon

        self.discrete_dqn_like = discrete_dqn_like
        self.siamese = siamese
        self.siamese_use_q = siamese_use_q
        self.siamese_use_adaptive = siamese_use_adaptive
        self.use_prediction = use_prediction
        self.transition_kl = transition_kl
        self.use_extra_data = use_extra_data
        self.curiosity = curiosity
        self.curiosity_strength = curiosity_strength
        self.use_rnd = use_rnd
        self.rnd_n_sample = rnd_n_sample
        self.use_normalization = use_normalization
        self.action_noise = action_noise

        self.use_replay_buffer = False
        self.use_priority = False
        self.use_n_step_is = True

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        self.summary_writer = None
        if self.model_abs_dir:
            if ma_name is not None:
                self.model_abs_dir = self.model_abs_dir / ma_name.replace('?', '-')
            summary_path = Path(self.model_abs_dir).joinpath(summary_path)
            self.summary_writer = SummaryWriter(str(summary_path))

        if ma_name is None:
            self._logger = logging.getLogger('sac.base.ds')
        else:
            self._logger = logging.getLogger(f'sac.base.ds.{ma_name}')

        self._build_model(nn, nn_config, init_log_alpha, learning_rate)
        self._init_or_restore(int(last_ckpt) if last_ckpt is not None else None)

    def get_policy_variables(self, get_numpy=True):
        """
        For learner to send variables to actors
        """
        variables = chain(self.model_rep.parameters(), self.model_policy.parameters())

        if self.use_rnd:
            variables = chain(variables,
                              self.model_rnd.parameters(),
                              self.model_target_rnd.parameters())

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
        variables = chain(self.model_rep.parameters(),
                          self.model_policy.parameters(),
                          [self.log_d_alpha, self.log_c_alpha])

        for model_q in self.model_q_list:
            variables = chain(variables,
                              model_q.parameters())

        if self.siamese == SIAMESE.ATC:
            variables = chain(variables, self.contrastive_weight_list)
        elif self.siamese == SIAMESE.BYOL:
            variables = chain(variables,
                              *[pro.parameters() for pro in self.model_rep_projection_list],
                              *[pre.parameters() for pre in self.model_rep_prediction_list])

        if self.use_prediction:
            variables = chain(variables,
                              self.model_transition.parameters(),
                              self.model_reward.parameters(),
                              self.model_observation.parameters())

        if self.curiosity == CURIOSITY.FORWARD:
            variables = chain(variables,
                              self.model_forward_dynamic.parameters())
        elif self.curiosity == CURIOSITY.INVERSE:
            variables = chain(variables,
                              self.model_inverse_dynamice.parameters())

        if get_numpy:
            return [v.detach().cpu().numpy() for v in variables]
        else:
            return variables

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

        if self.siamese == 'BYOL':
            variables = chain(variables,
                              *[t_pro.parameters() for t_pro in self.model_target_rep_projection_list])

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
              bn_indexes,
              bn_padding_masks,
              bn_obses_list,
              bn_actions,
              bn_rewards,
              next_obs_list,
              bn_dones,
              bn_mu_probs,
              f_seq_hidden_states=None):

        bn_indexes = torch.from_numpy(bn_indexes).to(self.device)
        bn_padding_masks = torch.from_numpy(bn_padding_masks).to(self.device)
        bn_obses_list = [torch.from_numpy(t).to(self.device) for t in bn_obses_list]
        bn_actions = torch.from_numpy(bn_actions).to(self.device)
        bn_rewards = torch.from_numpy(bn_rewards).to(self.device)
        next_obs_list = [torch.from_numpy(t).to(self.device) for t in next_obs_list]
        bn_dones = torch.from_numpy(bn_dones).to(self.device)
        bn_mu_probs = torch.from_numpy(bn_mu_probs).to(self.device)
        if self.seq_encoder is not None:
            f_seq_hidden_states = torch.from_numpy(f_seq_hidden_states).to(self.device)

        self._train(bn_indexes=bn_indexes,
                    bn_padding_masks=bn_padding_masks,
                    bn_obses_list=bn_obses_list,
                    bn_actions=bn_actions,
                    bn_rewards=bn_rewards,
                    next_obs_list=next_obs_list,
                    bn_dones=bn_dones,
                    bn_mu_probs=bn_mu_probs,
                    priority_is=None,
                    f_seq_hidden_states=f_seq_hidden_states if self.seq_encoder is not None else None)

        step = self.global_step.item()

        if step % self.save_model_per_step == 0:
            self.save_model()

        self._increase_global_step()

        return step + 1

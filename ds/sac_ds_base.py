import logging
import sys
from itertools import chain
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from algorithm.utils import *

sys.path.append(str(Path(__file__).resolve().parent.parent))
from algorithm.sac_base import SAC_Base


class SAC_DS_Base(SAC_Base):
    def __init__(self,
                 obs_names: list[str],
                 obs_shapes: list[tuple[int]],
                 d_action_sizes: list[int],
                 c_action_size: int,
                 model_abs_dir: Path | None,
                 nn,

                 device: str | None = None,
                 ma_name: str | None = None,
                 summary_path: str | None = 'log',
                 train_mode: bool = True,
                 last_ckpt: str | None = None,

                 nn_config: dict | None = None,

                 seed: float | None = None,
                 write_summary_per_step: float = 1e3,
                 save_model_per_step: float = 1e5,

                 use_replay_buffer: bool = True,
                 use_priority: bool = True,

                 ensemble_q_num: int = 2,
                 ensemble_q_sample: int = 2,

                 burn_in_step: int = 0,
                 n_step: int = 1,
                 seq_encoder: SEQ_ENCODER | None = None,

                 batch_size: int = 256,
                 tau: float = 0.005,
                 update_target_per_step: int = 1,
                 init_log_alpha: float = -2.3,
                 use_auto_alpha: bool = True,
                 target_d_alpha: float = 0.98,
                 target_c_alpha: float = 1.,
                 d_policy_entropy_penalty: float = 0.5,

                 learning_rate: float = 3e-4,

                 gamma: float = 0.99,
                 v_lambda: float = 1.,
                 v_rho: float = 1.,
                 v_c: float = 1.,
                 clip_epsilon: float = 0.2,

                 discrete_dqn_like: bool = False,
                 discrete_dqn_epsilon: float = 0.2,

                 siamese: SIAMESE | None = None,
                 siamese_use_q: bool = False,
                 siamese_use_adaptive: bool = False,

                 use_prediction: bool = False,
                 transition_kl: float = 0.8,
                 use_extra_data: bool = True,

                 curiosity: CURIOSITY | None = None,
                 curiosity_strength: float = 1.,
                 use_rnd: bool = False,
                 rnd_n_sample: int = 10,

                 use_normalization: bool = False,

                 action_noise: list[float] | None = None,

                 replay_config: dict | None = None):

        self.obs_names = obs_names
        self.obs_shapes = obs_shapes
        self.d_action_sizes = d_action_sizes
        self.d_action_summed_size = sum(d_action_sizes)
        self.d_action_branch_size = len(d_action_sizes)
        self.c_action_size = c_action_size
        self.model_abs_dir = model_abs_dir
        self.ma_name = ma_name
        self.train_mode = train_mode

        self.use_replay_buffer = use_replay_buffer
        self.use_priority = use_priority

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
        self.target_d_alpha = target_d_alpha
        self.target_c_alpha = target_c_alpha
        self.d_policy_entropy_penalty = d_policy_entropy_penalty

        self.learning_rate = learning_rate

        self.gamma = gamma
        self.v_lambda = v_lambda
        self.v_rho = v_rho
        self.v_c = v_c
        self.clip_epsilon = clip_epsilon

        self.discrete_dqn_like = discrete_dqn_like
        self.discrete_dqn_epsilon = discrete_dqn_epsilon

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

        self.use_n_step_is = True

        self._set_logger()

        if self.use_n_step_is and c_action_size == 0 and len(d_action_sizes) != 0 and discrete_dqn_like:
            self.use_n_step_is = False
            self._logger.warning('use_n_step_is is disabled because of discrete DQN-like')

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self._logger.info(f'Device: {self.device.type}:{self.device.index}')

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        self._profiler = UnifiedElapsedTimer(self._logger)

        self.summary_writer = None
        if self.model_abs_dir:
            summary_path = Path(self.model_abs_dir).joinpath(summary_path)
            self.summary_writer = SummaryWriter(str(summary_path))

        self._build_model(nn, nn_config, init_log_alpha, learning_rate)
        self._build_ckpt()
        self._init_replay_buffer(replay_config)
        self._init_or_restore(int(last_ckpt) if last_ckpt is not None else None)

    def _set_logger(self):
        if self.ma_name is None:
            self._logger = logging.getLogger('sac.base.ds')
        else:
            self._logger = logging.getLogger(f'sac.base.ds.{self.ma_name}')

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

    def update_policy_variables(self, t_variables: list[np.ndarray]):
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

    def update_all_variables(self, t_variables: list[np.ndarray]):
        if any([np.isnan(v.sum()) for v in t_variables]):
            return False

        variables = self.get_all_variables(get_numpy=False)

        for v, t_v in zip(variables, t_variables):
            v.data.copy_(torch.from_numpy(t_v).to(self.device))

        return True

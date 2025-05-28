from collections import defaultdict
from itertools import chain

import numpy as np
import torch
from torch import distributions, nn, optim
from torch.nn import functional

from algorithm.oc.option_base import OptionBase

from .. import sac_base
from ..nn_models import *
from ..sac_base import SAC_Base
from ..utils import *
from .oc_batch_buffer import BatchBuffer


class OptionSelectorBase(SAC_Base):
    option_list = None

    def __init__(self,
                 obs_names: list[str],
                 obs_shapes: list[tuple],
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
                 use_n_step_is: bool = True,

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

                 use_dilation: bool = False,
                 option_burn_in_step: int = -1,
                 option_seq_encoder: SEQ_ENCODER | None = None,
                 option_epsilon: float = 0.2,
                 terminal_entropy: float = 0.01,
                 key_max_length: int = 200,
                 option_nn_config: dict | None = None,
                 option_configs: list[dict] = [],

                 replay_config: dict | None = None):

        sac_base.BatchBuffer = BatchBuffer

        self.use_dilation = use_dilation
        self.option_burn_in_step = option_burn_in_step
        self.option_seq_encoder = option_seq_encoder
        self.option_epsilon = option_epsilon
        self.terminal_entropy = terminal_entropy
        self.key_max_length = key_max_length
        self.option_nn_config = option_nn_config

        if len(option_configs) == 0:
            # Default option configs
            for i in range(2):
                option_configs.append({
                    'name': f'option_{i}',
                    'fix_policy': False,
                    'random_q': False
                })
        self.option_configs = option_configs
        self.num_options = len(option_configs)

        super().__init__(obs_names,
                         obs_shapes,
                         d_action_sizes,
                         c_action_size,
                         model_abs_dir,
                         nn,
                         device, ma_name,
                         summary_path,
                         train_mode,
                         last_ckpt,
                         nn_config,
                         seed,
                         write_summary_per_step,
                         save_model_per_step,
                         use_replay_buffer,
                         use_priority,
                         ensemble_q_num,
                         ensemble_q_sample,
                         burn_in_step,
                         n_step,
                         seq_encoder,
                         batch_size,
                         tau,
                         update_target_per_step,
                         init_log_alpha,
                         use_auto_alpha,
                         target_d_alpha,
                         target_c_alpha,
                         d_policy_entropy_penalty,
                         learning_rate,
                         gamma,
                         v_lambda,
                         v_rho,
                         v_c,
                         clip_epsilon,
                         discrete_dqn_like,
                         discrete_dqn_epsilon,
                         use_n_step_is,
                         siamese,
                         siamese_use_q,
                         siamese_use_adaptive,
                         use_prediction,
                         transition_kl,
                         use_extra_data,
                         curiosity,
                         curiosity_strength,
                         use_rnd,
                         rnd_n_sample,
                         use_normalization,
                         action_noise,
                         replay_config)

    def _set_logger(self):
        if self.ma_name is None:
            self._logger = logging.getLogger('option_selector')
        else:
            self._logger = logging.getLogger(f'option_selector.{self.ma_name}')

    def _build_model(self, nn, nn_config: dict | None, init_log_alpha: float, learning_rate: float) -> None:
        """
        Initialize variables, network models and optimizers
        """
        if nn_config is None:
            nn_config = {}
        nn_config = defaultdict(dict, nn_config)
        if nn_config['rep'] is None:
            nn_config['rep'] = {}

        self.global_step = torch.tensor(0, dtype=torch.int64, requires_grad=False, device='cpu')

        self._gamma_ratio = torch.logspace(0, self.n_step - 1, self.n_step, self.gamma, device=self.device)
        self._lambda_ratio = torch.logspace(0, self.n_step - 1, self.n_step, self.v_lambda, device=self.device)

        self.v_rho = torch.tensor(self.v_rho, device=self.device)
        self.v_c = torch.tensor(self.v_c, device=self.device)

        self._option_eye = torch.eye(self.num_options + 1, dtype=torch.float32, device=self.device)

        d_action_list = [np.eye(d_action_size, dtype=np.float32)[0]
                         for d_action_size in self.d_action_sizes]
        self._padding_action = np.concatenate(d_action_list + [np.zeros(self.c_action_size, dtype=np.float32)], axis=-1)

        def adam_optimizer(params) -> optim.Adam | None:
            params = list(params)
            if len(params) == 0:
                return
            return optim.Adam(params, lr=learning_rate)

        """ NORMALIZATION """
        if self.use_normalization:
            self.normalizer_step = torch.tensor(0, dtype=torch.int32, device=self.device, requires_grad=False)
            self.running_means = []
            self.running_variances = []
            for shape in self.obs_shapes:
                self.running_means.append(torch.zeros(shape, device=self.device))
                self.running_variances.append(torch.ones(shape, device=self.device))

            p_self = self

            class ModelRep(nn.ModelOptionSelectorRep):
                def forward(self, obs_list, *args, **kwargs):
                    obs_list = [
                        torch.clamp(
                            (obs - mean) / torch.sqrt(variance / (p_self.normalizer_step + 1)),
                            -5, 5
                        ) for obs, mean, variance in zip(obs_list,
                                                         p_self.running_means,
                                                         p_self.running_variances)
                    ]

                    return super().forward(obs_list, *args, **kwargs)
        else:
            ModelRep = nn.ModelOptionSelectorRep

        """ REPRESENTATION """
        if self.seq_encoder in (None, SEQ_ENCODER.RNN):
            self.model_rep: ModelBaseOptionSelectorRep = ModelRep(self.obs_names,
                                                                  self.obs_shapes,
                                                                  self.d_action_sizes, self.c_action_size,
                                                                  False,
                                                                  self.use_dilation,
                                                                  self.model_abs_dir,
                                                                  **nn_config['rep']).to(self.device)
            self.model_target_rep: ModelBaseOptionSelectorRep = ModelRep(self.obs_names,
                                                                         self.obs_shapes,
                                                                         self.d_action_sizes, self.c_action_size,
                                                                         True,
                                                                         self.use_dilation,
                                                                         self.model_abs_dir,
                                                                         **nn_config['rep']).to(self.device)
            # Get represented state and seq_hidden_state_shape
            test_obs_list = [torch.rand(self.batch_size, 1, *obs_shape, device=self.device) for obs_shape in self.obs_shapes]
            test_pre_action = torch.rand(self.batch_size, 1, self.d_action_summed_size + self.c_action_size, device=self.device)
            test_state, test_seq_hidden_states = self.model_rep(test_obs_list,
                                                                test_pre_action,
                                                                None)
            state_size = test_state.shape[-1]
            if self.seq_encoder is None:
                seq_hidden_state_shape = test_seq_hidden_states.shape[2:]  # [batch, 1, *seq_hidden_state_shape]
            else:
                seq_hidden_state_shape = test_seq_hidden_states.shape[1:]  # [batch, *seq_hidden_state_shape]

        elif self.seq_encoder == SEQ_ENCODER.ATTN:
            obs_names = self.obs_names.copy()
            obs_shapes = self.obs_shapes.copy()
            if self.use_dilation:
                obs_names = ['_OPTION_INDEX'] + obs_names
                obs_shapes = [(self.num_options + 1, )] + obs_shapes

            self.model_rep: ModelBaseOptionSelectorAttentionRep = ModelRep(obs_names,
                                                                           obs_shapes,
                                                                           self.d_action_sizes, self.c_action_size,
                                                                           False,
                                                                           self.use_dilation,
                                                                           self.model_abs_dir,
                                                                           **nn_config['rep']).to(self.device)
            self.model_target_rep: ModelBaseOptionSelectorAttentionRep = ModelRep(obs_names,
                                                                                  obs_shapes,
                                                                                  self.d_action_sizes, self.c_action_size,
                                                                                  True,
                                                                                  self.use_dilation,
                                                                                  self.model_abs_dir,
                                                                                  **nn_config['rep']).to(self.device)
            # Get represented state and seq_hidden_state_shape
            test_index = torch.zeros((self.batch_size, 1), dtype=torch.int32, device=self.device)
            test_obs_list = [torch.rand(self.batch_size, 1, *obs_shape, device=self.device) for obs_shape in obs_shapes]
            test_pre_action = torch.rand(self.batch_size, 1, self.d_action_summed_size + self.c_action_size, device=self.device)
            test_state, test_attn_state, _ = self.model_rep(1,
                                                            test_index,
                                                            test_obs_list,
                                                            test_pre_action,
                                                            None)
            state_size, seq_hidden_state_shape = test_state.shape[-1], test_attn_state.shape[2:]

        for param in self.model_target_rep.parameters():
            param.requires_grad = False

        self.state_size = state_size
        self.seq_hidden_state_shape = seq_hidden_state_shape
        self._logger.info(f'State size: {state_size}')
        self._logger.info(f'Seq hidden state shape: {tuple(seq_hidden_state_shape)}')

        self.optimizer_rep = adam_optimizer(self.model_rep.parameters())

        """ RANDOM NETWORK DISTILLATION """
        if self.use_rnd:
            self.model_rnd: ModelOptionSelectorRND = nn.ModelOptionSelectorRND(state_size, self.num_options).to(self.device)
            self.model_target_rnd: ModelOptionSelectorRND = nn.ModelOptionSelectorRND(state_size, self.num_options).to(self.device)
            for param in self.model_target_rnd.parameters():
                param.requires_grad = False
            self.optimizer_rnd = adam_optimizer(self.model_rnd.parameters())

        """ V OVER OPTIONS """
        self.model_v_over_options_list = [nn.ModelVOverOptions(state_size,
                                                               self.num_options,
                                                               False).to(self.device)
                                          for _ in range(self.ensemble_q_num)]
        self.model_target_v_over_options_list = [nn.ModelVOverOptions(state_size,
                                                                      self.num_options,
                                                                      True).to(self.device)
                                                 for _ in range(self.ensemble_q_num)]
        self.optimizer_v_list = [adam_optimizer(self.model_v_over_options_list[i].parameters()) for i in range(self.ensemble_q_num)]

        """ SIAMESE REPRESENTATION LEARNING """
        if self.siamese in (SIAMESE.ATC, SIAMESE.BYOL):
            test_encoder_list = self.model_rep.get_augmented_encoders(test_obs_list)
            if not isinstance(test_encoder_list, tuple):
                test_encoder_list = [test_encoder_list, ]

            if self.siamese == SIAMESE.ATC:
                self.contrastive_weight_list = [torch.randn((test_encoder.shape[-1], test_encoder.shape[-1]),
                                                            requires_grad=True,
                                                            device=self.device) for test_encoder in test_encoder_list]
                self.optimizer_siamese = adam_optimizer(self.contrastive_weight_list)

            elif self.siamese == SIAMESE.BYOL:
                self.model_rep_projection_list: list[ModelBaseRepProjection] = [
                    nn.ModelRepProjection(test_encoder.shape[-1]).to(self.device) for test_encoder in test_encoder_list]
                self.model_target_rep_projection_list: list[ModelBaseRepProjection] = [
                    nn.ModelRepProjection(test_encoder.shape[-1]).to(self.device) for test_encoder in test_encoder_list]

                test_projection_list = [pro(test_encoder) for pro, test_encoder in zip(self.model_rep_projection_list, test_encoder_list)]
                self.model_rep_prediction_list: list[ModelBaseRepPrediction] = [
                    nn.ModelRepPrediction(test_projection.shape[-1]).to(self.device) for test_projection in test_projection_list]
                self.optimizer_siamese = adam_optimizer(chain(*[pro.parameters() for pro in self.model_rep_projection_list],
                                                              *[pre.parameters() for pre in self.model_rep_prediction_list]))

    def _build_ckpt(self) -> None:
        self.ckpt_dict = ckpt_dict = {
            'global_step': self.global_step
        }

        """ NORMALIZATION & REPRESENTATION """
        if self.use_normalization:
            ckpt_dict['normalizer_step'] = self.normalizer_step
            for i, v in enumerate(self.running_means):
                ckpt_dict[f'running_means_{i}'] = v
            for i, v in enumerate(self.running_variances):
                ckpt_dict[f'running_variances_{i}'] = v

        if self.optimizer_rep is not None:
            ckpt_dict['model_rep'] = self.model_rep
            ckpt_dict['model_target_rep'] = self.model_target_rep
            ckpt_dict['optimizer_rep'] = self.optimizer_rep

        """ RANDOM NETWORK DISTILLATION """
        if self.use_rnd:
            ckpt_dict['model_rnd'] = self.model_rnd
            ckpt_dict['model_target_rnd'] = self.model_target_rnd
            ckpt_dict['optimizer_rnd'] = self.optimizer_rnd

        """ V OVER OPTIONS """
        for i in range(self.ensemble_q_num):
            if self.optimizer_v_list[i] is None:
                continue
            ckpt_dict[f'model_v_over_options_{i}'] = self.model_v_over_options_list[i]
            ckpt_dict[f'model_target_v_over_options_{i}'] = self.model_target_v_over_options_list[i]
            ckpt_dict[f'optimizer_v_over_options_{i}'] = self.optimizer_v_list[i]

        total_parameter_num = 0
        for m in ckpt_dict.values():
            if isinstance(m, nn.Module):
                total_parameter_num += sum([p.numel() for p in m.parameters()])
        self._logger.info(f'Parameters: {total_parameter_num}')

    def _init_or_restore(self, last_ckpt):
        super()._init_or_restore(last_ckpt)

        """
        Initialize each option
        """
        if self.option_burn_in_step == -1:
            self.option_burn_in_step = self.burn_in_step

        self.option_burn_in_from = self.burn_in_step - self.option_burn_in_step

        option_kwargs = self._kwargs  # SAC_BASE's kwargs
        del option_kwargs['self']
        option_kwargs['obs_names'] = ['state', *self.obs_names]
        option_kwargs['obs_shapes'] = [(self.state_size, ), *self.obs_shapes]
        option_kwargs['device'] = self.device

        assert self.option_seq_encoder in (None, SEQ_ENCODER.RNN)  # seq_encoder of option can ONLY be RNN or VANILLA

        option_kwargs['seq_encoder'] = self.option_seq_encoder
        option_kwargs['burn_in_step'] = self.option_burn_in_step
        option_kwargs['use_replay_buffer'] = False

        if self.option_nn_config is None:
            self.option_nn_config = {}
        self.option_nn_config = defaultdict(dict, self.option_nn_config)
        option_kwargs['nn_config'] = self.option_nn_config

        self.option_list: list[OptionBase] = [None] * self.num_options
        for i, option_config in enumerate(self.option_configs):
            if self.model_abs_dir is not None:
                option_kwargs['model_abs_dir'] = self.model_abs_dir / option_config['name']
                option_kwargs['model_abs_dir'].mkdir(parents=True, exist_ok=True)
            if self.ma_name is not None:
                option_kwargs['ma_name'] = f'{self.ma_name}_{option_config["name"]}'
            else:
                option_kwargs['ma_name'] = option_config['name']

            self.option_list[i] = OptionBase(option=i,
                                             fix_policy=option_config['fix_policy'],
                                             random_q=option_config['random_q'],
                                             **option_kwargs)
            self._logger.info(f'[{i}] - {self.option_list[i].ma_name} initialized')

        if self.train_mode:
            for i, option in enumerate(self.option_list):
                global_step = self.get_global_step()
                option.set_global_step(global_step)
                option.remove_models(global_step)

        self.low_seq_hidden_state_shape = self.option_list[0].seq_hidden_state_shape

    def _init_replay_buffer(self, replay_config=None):
        self._key_batch = None

        return super()._init_replay_buffer(replay_config)

    def set_train_mode(self, train_mode=True):
        super().set_train_mode(train_mode)

        for option in self.option_list:
            option.set_train_mode(train_mode)

    def save_model(self, save_replay_buffer=False) -> None:
        super().save_model(save_replay_buffer)

        for option in self.option_list:
            option.save_model(save_replay_buffer)

    def get_initial_option_index(self, batch_size: int) -> np.ndarray:
        return np.full([batch_size, ], -1, dtype=np.int8)

    def get_initial_low_seq_hidden_state(self, batch_size, get_numpy=True) -> np.ndarray | torch.Tensor:
        return self.option_list[0].get_initial_seq_hidden_state(batch_size,
                                                                get_numpy)

    def get_option_names(self) -> list[str]:
        return [option.ma_name for option in self.option_list]

    @torch.no_grad()
    def _update_target_variables(self, tau=1.) -> None:
        target = self.model_target_rep.parameters()
        source = self.model_rep.parameters()

        for i in range(self.ensemble_q_num):
            target = chain(target, self.model_target_v_over_options_list[i].parameters())
            source = chain(source, self.model_v_over_options_list[i].parameters())

        for target_param, param in zip(target, source):
            target_param.data.copy_(
                target_param.data * (1. - tau) + param.data * tau
            )

        if self.option_list is None:
            self._logger.warning('option_list is None, abort updating option target variables')
            return

        for option in self.option_list:
            option._update_target_variables(tau)

    def _get_termination_mask(self,
                              termination: torch.Tensor,
                              disable_sample: bool = False) -> torch.Tensor:
        if disable_sample:
            termination_mask = termination > 0.5
        else:
            termination_dist = torch.distributions.Categorical(probs=torch.stack([termination, 1 - termination],
                                                                                 dim=-1))
            termination_mask = termination_dist.sample() == 0

        return termination_mask

    #################### ! GET ACTION ####################

    def _choose_option_index(self,
                             pre_option_index: torch.Tensor,
                             state: torch.Tensor,
                             pre_termination_mask: torch.Tensor) -> tuple[torch.Tensor,
                                                                          torch.Tensor]:
        """
        Args:
            pre_option_index (torch.int64): [batch, ]
            state: [batch, state_size]
            pre_termination_mask (bool): [batch, ]

        Returns:
            new_option_index (torch.int64): [batch, ]
            new_option_mask (torch.bool): [batch, ]
        """
        batch = pre_option_index.shape[0]

        option_index = pre_option_index.clone()

        v_over_options_list = [v(state) for v in self.model_v_over_options_list]  # list([batch, num_options], ...)
        stacked_v_over_options = torch.stack(v_over_options_list)  # [ensemble, batch, num_options]
        v_over_options, _ = stacked_v_over_options.min(dim=0)  # [batch, num_options]

        none_option_mask = option_index == -1

        new_option_index = v_over_options.argmax(dim=-1)  # [batch, ]

        option_index[none_option_mask] = new_option_index[none_option_mask]
        option_index[pre_termination_mask] = new_option_index[pre_termination_mask]

        if self.train_mode:
            random_mask = torch.rand_like(option_index, dtype=torch.float32) < self.option_epsilon
            if self.use_rnd:
                rnd = self.model_rnd.cal_rnd(state)  # [batch, num_options, f]
                t_rnd = self.model_target_rnd.cal_rnd(state)  # [batch, num_options, f]
                loss = torch.sum(torch.abs(rnd - t_rnd), dim=-1)  # [batch, num_options]
                random_option_index = loss.argmax(dim=-1)  # [batch, ]
            else:
                dist = distributions.Categorical(logits=torch.ones((batch, self.num_options),
                                                                   device=self.device))
                random_option_index = dist.sample()  # [batch, ]

            option_index[random_mask] = random_option_index[random_mask]

        return option_index

    def _choose_option_action(self,
                              obs_list: list[torch.Tensor],
                              state: torch.Tensor,
                              pre_option_index: torch.Tensor,
                              pre_action: torch.Tensor,
                              pre_low_seq_hidden_state: torch.Tensor,
                              pre_termination_mask: torch.Tensor,

                              disable_sample: bool = False,
                              force_rnd_if_available: bool = False) -> tuple[torch.Tensor,
                                                                             torch.Tensor,
                                                                             torch.Tensor,
                                                                             torch.Tensor,
                                                                             torch.Tensor]:
        """
        Args:
            obs_list: list([batch, *obs_shapes_i], ...)
            state: [batch, d_action_summed_size + c_action_size]
            pre_option_index (torch.int64): [batch, ]
            pre_action: [batch, action_size]
            pre_low_seq_hidden_state: [batch, *low_seq_hidden_state_shape]
            pre_termination_mask (bool): [batch, ]

        Returns:
            new_option_index (torch.int64): [batch, ]
            action: [batch, action_size]
            prob: [batch, action_size]
            low_seq_hidden_state: [batch, *low_seq_hidden_state_shape]
            termination: [batch, ]
        """

        low_obs_list = self.get_l_low_obses_list(obs_list, state)

        option_index = self._choose_option_index(pre_option_index=pre_option_index,
                                                 state=state,
                                                 pre_termination_mask=pre_termination_mask)

        batch = state.shape[0]
        initial_low_seq_hidden_state = self.get_initial_low_seq_hidden_state(batch, get_numpy=False)

        new_option_index_mask = pre_option_index != option_index
        pre_low_seq_hidden_state[new_option_index_mask] = initial_low_seq_hidden_state[new_option_index_mask]

        action = torch.zeros(batch, self.d_action_summed_size + self.c_action_size, device=self.device)
        prob = torch.ones((batch,
                           self.d_action_summed_size + self.c_action_size),
                          device=self.device)
        low_seq_hidden_state = torch.zeros_like(pre_low_seq_hidden_state)
        termination = torch.zeros_like(pre_termination_mask, dtype=torch.float32)

        for i, option in enumerate(self.option_list):
            mask = (option_index == i)
            if not torch.any(mask):
                continue

            o_low_obs_list = [low_obs[mask] for low_obs in low_obs_list]

            (o_action,
             o_prob,
             o_low_seq_hidden_state,
             o_termination) = option.choose_action(o_low_obs_list,
                                                   pre_action[mask],
                                                   pre_low_seq_hidden_state[mask],
                                                   disable_sample=disable_sample,
                                                   force_rnd_if_available=force_rnd_if_available)
            action[mask] = o_action
            prob[mask] = o_prob
            low_seq_hidden_state[mask] = o_low_seq_hidden_state
            termination[mask] = o_termination

        return option_index, action, prob, low_seq_hidden_state, termination

    @torch.no_grad()
    def choose_action(self,
                      obs_list: list[np.ndarray],
                      pre_option_index: np.ndarray,
                      pre_action: np.ndarray,
                      pre_seq_hidden_state: np.ndarray,
                      pre_low_seq_hidden_state: np.ndarray,
                      pre_termination: np.ndarray,

                      disable_sample: bool = False,
                      force_rnd_if_available: bool = False) -> tuple[np.ndarray,
                                                                     np.ndarray,
                                                                     np.ndarray,
                                                                     np.ndarray,
                                                                     np.ndarray,
                                                                     np.ndarray]:
        """
        Args:
            obs_list (np): list([batch, *obs_shapes_i], ...)
            pre_option_index (np.int8): [batch, ]
            pre_action (np): [batch, d_action_summed_size + c_action_size]
            pre_seq_hidden_state (np): [batch, *seq_hidden_state_shape]
            pre_low_seq_hidden_state (np): [batch, *low_seq_hidden_state_shape]
            pre_termination: [batch, ]

        Returns:
            option_index (np.int8): [batch, ]
            action (np): [batch, d_action_summed_size + c_action_size]
            prob (np): [batch, action_size]
            seq_hidden_state (np): [batch, *seq_hidden_state_shape]
            low_seq_hidden_state (np): [batch, *low_seq_hidden_state_shape]
            termination: [batch, ]
        """
        obs_list = [torch.from_numpy(obs).to(self.device) for obs in obs_list]
        pre_option_index = torch.from_numpy(pre_option_index).type(torch.int64).to(self.device)
        pre_action = torch.from_numpy(pre_action).to(self.device)
        pre_seq_hidden_state = torch.from_numpy(pre_seq_hidden_state).to(self.device)
        pre_low_seq_hidden_state = torch.from_numpy(pre_low_seq_hidden_state).to(self.device)
        pre_termination = torch.from_numpy(pre_termination).to(self.device)

        obs_list = [obs.unsqueeze(1) for obs in obs_list]
        pre_action = pre_action.unsqueeze(1)
        pre_seq_hidden_state = pre_seq_hidden_state.unsqueeze(1)
        pre_termination_mask = self._get_termination_mask(pre_termination, disable_sample)

        state, seq_hidden_state = self.model_rep(obs_list,
                                                 pre_action,
                                                 pre_seq_hidden_state,
                                                 pre_termination_mask=pre_termination_mask)
        # state: [batch, 1, state_size]
        # seq_hidden_state: [batch, 1, *seq_hidden_state_shape] | [batch, *seq_hidden_state_shape]

        state = state.squeeze(1)
        obs_list = [obs.squeeze(1) for obs in obs_list]
        if self.seq_encoder is None:
            seq_hidden_state = seq_hidden_state.squeeze(1)
        pre_action = pre_action.squeeze(1)

        (option_index,
         action,
         prob,
         low_seq_hidden_state,
         termination) = self._choose_option_action(obs_list,
                                                   state,
                                                   pre_option_index,
                                                   pre_action,
                                                   pre_low_seq_hidden_state,
                                                   pre_termination_mask,
                                                   disable_sample=disable_sample,
                                                   force_rnd_if_available=force_rnd_if_available)

        return (option_index.detach().cpu().numpy().astype(np.int8),
                action.detach().cpu().numpy(),
                prob.detach().cpu().numpy(),
                seq_hidden_state.detach().cpu().numpy(),
                low_seq_hidden_state.detach().cpu().numpy(),
                termination.detach().cpu().numpy())

    @torch.no_grad()
    def choose_attn_action(self,
                           ep_indexes: np.ndarray,
                           ep_padding_masks: np.ndarray,
                           ep_obses_list: list[np.ndarray],
                           ep_pre_actions: np.ndarray,
                           ep_pre_attn_states: np.ndarray,

                           pre_option_index: np.ndarray,
                           pre_low_seq_hidden_state: np.ndarray,
                           pre_termination: np.ndarray,

                           disable_sample: bool = False,
                           force_rnd_if_available: bool = False) -> tuple[np.ndarray,
                                                                          np.ndarray,
                                                                          np.ndarray,
                                                                          np.ndarray,
                                                                          np.ndarray,
                                                                          np.ndarray]:
        """
        Args:
            ep_indexes (np.int32): [batch, episode_len]
            ep_padding_masks (bool): [batch, episode_len]
            ep_obses_list (np): list([batch, episode_len, *obs_shapes_i], ...)
            ep_pre_actions (np): [batch, episode_len, d_action_summed_size + c_action_size]
            ep_pre_attn_states (np): [batch, episode_len, *seq_hidden_state_shape]

            pre_option_index (np.int8): [batch, ]
            pre_low_seq_hidden_state (np): [batch, *low_seq_hidden_state_shape]
            pre_termination: [batch, ]

        Returns:
            option_index (np.int8): [batch, ]
            action (np): [batch, d_action_summed_size + c_action_size]
            prob (np): [batch, action_size]
            attn_state (np): [batch, *attn_state_shape]
            low_seq_hidden_state (np): [batch, *low_rnn_state_shape]
            termination: [batch, ]
        """
        ep_indexes = torch.from_numpy(ep_indexes).to(self.device)
        ep_padding_masks = torch.from_numpy(ep_padding_masks).to(self.device)
        ep_obses_list = [torch.from_numpy(obs).to(self.device) for obs in ep_obses_list]
        ep_pre_actions = torch.from_numpy(ep_pre_actions).to(self.device)
        ep_pre_attn_states = torch.from_numpy(ep_pre_attn_states).to(self.device)

        pre_option_index = torch.from_numpy(pre_option_index).type(torch.int64).to(self.device)
        pre_low_seq_hidden_state = torch.from_numpy(pre_low_seq_hidden_state).to(self.device)
        pre_termination = torch.from_numpy(pre_termination).to(self.device)

        pre_termination_mask = self._get_termination_mask(pre_termination, disable_sample)

        state, attn_state, _ = self.model_rep(1,
                                              ep_indexes,
                                              ep_obses_list,
                                              ep_pre_actions,
                                              None,
                                              pre_termination_mask=pre_termination_mask,
                                              padding_mask=ep_padding_masks)
        # state: [batch, 1, state_size]
        # attn_state: [batch, 1, *attn_state_shape]

        obs_list = [ep_obses[:, -1, ...] for ep_obses in ep_obses_list]
        state = state.squeeze(1)
        attn_state = attn_state.squeeze(1)
        pre_action = ep_pre_actions[:, -1, ...]

        (option_index,
         action,
         prob,
         low_seq_hidden_state,
         termination) = self._choose_option_action(obs_list,
                                                   state,
                                                   pre_option_index,
                                                   pre_action,
                                                   pre_low_seq_hidden_state,
                                                   pre_termination_mask,
                                                   disable_sample=disable_sample,
                                                   force_rnd_if_available=force_rnd_if_available)

        return (option_index.detach().cpu().numpy().astype(np.int8),
                action.detach().cpu().numpy(),
                prob.detach().cpu().numpy(),
                attn_state.detach().cpu().numpy(),
                low_seq_hidden_state.detach().cpu().numpy(),
                termination.detach().cpu().numpy())

    @torch.no_grad()
    def choose_dilated_attn_action(self,
                                   key_indexes: np.ndarray,
                                   key_padding_masks: np.ndarray,
                                   key_obses_list: list[np.ndarray],
                                   key_option_indexes: np.ndarray,
                                   key_attn_states: np.ndarray,

                                   pre_option_index: np.ndarray,
                                   pre_action: np.ndarray,
                                   pre_low_seq_hidden_state: np.ndarray,
                                   pre_termination: np.ndarray,

                                   disable_sample: bool = False,
                                   force_rnd_if_available: bool = False) -> tuple[np.ndarray,
                                                                                  np.ndarray,
                                                                                  np.ndarray,
                                                                                  np.ndarray,
                                                                                  np.ndarray]:
        """
        Args:
            key_indexes (np.int32): [batch, key_len]z
            key_padding_masks (np.bool): [batch, key_len]
            key_obses_list (np): list([batch, key_len, *obs_shapes_i], ...)
            key_option_indexes (np.int8): [batch, key_len]
            key_attn_states (np): [batch, key_len, *seq_hidden_state_shape]
            # The last key transition is the current transition

            pre_option_index (np.int8): [batch, ]
            pre_action (np): [batch, action_size]
            pre_low_seq_hidden_state (np): [batch, *low_seq_hidden_state_shape]
            pre_termination (np): [batch, ]

        Returns:
            option_index (np.int8): [batch]
            action (np): [batch, d_action_summed_size + c_action_size]
            prob (np): [batch, action_size]
            attn_state (np): [batch, *attn_state_shape]
            low_rnn_state (np): [batch, *low_rnn_state_shape]
        """

        key_indexes = torch.from_numpy(key_indexes).to(self.device)
        key_padding_masks = torch.from_numpy(key_padding_masks).to(self.device)
        key_obses_list = [torch.from_numpy(obs).to(self.device) for obs in key_obses_list]
        key_option_indexes = key_option_indexes.astype(np.int32)
        key_option_indexes = torch.from_numpy(key_option_indexes).to(self.device)
        key_attn_states = torch.from_numpy(key_attn_states).to(self.device)

        pre_option_index = torch.from_numpy(pre_option_index).type(torch.int64).to(self.device)
        pre_action = torch.from_numpy(pre_action).to(self.device)
        pre_low_seq_hidden_state = torch.from_numpy(pre_low_seq_hidden_state).to(self.device)
        pre_termination = torch.from_numpy(pre_termination).to(self.device)

        pre_termination_mask = self.get_termination_mask(pre_termination, disable_sample)

        _key_option_indexes = self._option_eye[key_option_indexes]
        state, attn_state, _ = self.model_rep(1,
                                              key_indexes,
                                              [_key_option_indexes] + key_obses_list,
                                              None,
                                              None,
                                              pre_termination_mask=pre_termination_mask,
                                              query_only_attend_to_rest_key=True,
                                              padding_mask=key_padding_masks)
        # state: [batch, 1, state_size]
        # attn_state: [batch, 1, *attn_state_shape]

        obs_list = [key_obses[:, -1, ...] for key_obses in key_obses_list]
        state = state.squeeze(1)
        attn_state = attn_state.squeeze(1)

        (option_index,
         action,
         prob,
         low_seq_hidden_state,
         termination) = self._choose_option_action(obs_list,
                                                   state,
                                                   pre_option_index,
                                                   pre_action,
                                                   pre_low_seq_hidden_state,
                                                   pre_termination_mask,
                                                   disable_sample=disable_sample,
                                                   force_rnd_if_available=force_rnd_if_available)

        return (option_index.detach().cpu().numpy().astype(np.int8),
                action.detach().cpu().numpy(),
                prob.detach().cpu().numpy(),
                attn_state.detach().cpu().numpy(),
                low_seq_hidden_state.detach().cpu().numpy(),
                termination.detach().cpu().numpy())

    #################### ! GET STATES ####################

    def get_l_states(self,
                     l_indexes: torch.Tensor,
                     l_padding_masks: torch.Tensor,
                     l_obses_list: list[torch.Tensor],
                     l_pre_actions: torch.Tensor,
                     l_pre_seq_hidden_states: torch.Tensor,

                     key_batch: tuple[torch.Tensor,
                                      torch.Tensor,
                                      list[torch.Tensor],
                                      torch.Tensor] | None = None,

                     is_target=False) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            l_indexes: [batch, l]
            l_padding_masks: [batch, l]
            l_obses_list: list([batch, l, *obs_shapes_i], ...)
            l_pre_actions: [batch, l, action_size]
            l_pre_seq_hidden_states: [batch, l, *seq_hidden_state_shape]

            key_batch:
                key_indexes: [batch, key_len]
                key_padding_masks: [batch, key_len]
                key_obses_list: list([batch, key_len, *obs_shapes_i], ...)
                key_option_indexes: [batch, key_len]
                key_pre_seq_hidden_states: [batch, key_len, *seq_hidden_state_shape]

        Returns:
            l_states: [batch, l, state_size]
            l_seq_hidden_states (optional): [batch, l, *seq_hidden_state_shape]
            f_rnn_states (optional): [batch, 1, *seq_hidden_state_shape]
        """
        model_rep = self.model_target_rep if is_target else self.model_rep

        if self.seq_encoder == SEQ_ENCODER.RNN and self.use_dilation:
            (key_indexes,
             key_padding_masks,
             key_obses_list,
             key_option_indexes,
             key_pre_seq_hidden_states) = key_batch

            _, next_key_rnn_state = model_rep(key_obses_list,
                                              None,
                                              key_pre_seq_hidden_states,
                                              padding_mask=key_padding_masks)

            batch, l, *_ = l_indexes.shape

            l_states = None

            for t in range(l):
                f_states, next_rnn_state = model_rep([l_obses[:, t:t + 1, ...] for l_obses in l_obses_list],
                                                     None,
                                                     next_key_rnn_state.unsqueeze(1),
                                                     padding_mask=l_padding_masks[:, t:t + 1, ...])
                if l_states is None:
                    l_states = torch.zeros((batch, l, *f_states.shape[2:]), device=self.device)
                l_states[:, t:t + 1] = f_states

            next_f_rnn_states = next_rnn_state.unsqueeze(dim=1)

            return l_states, next_f_rnn_states

        elif self.seq_encoder == SEQ_ENCODER.ATTN and self.use_dilation:
            seq_q_len = l_indexes.shape[1]

            (key_indexes,
             key_padding_masks,
             key_obses_list,
             key_option_indexes,
             key_pre_seq_hidden_states) = key_batch

            l_indexes = torch.concat([key_indexes, l_indexes], dim=1)
            l_padding_masks = torch.concat([key_padding_masks, l_padding_masks], dim=1)
            l_obses_list = [torch.concat([key_obses, l_obses], dim=1)
                            for key_obses, l_obses in zip(key_obses_list, l_obses_list)]
            l_option_indexes = torch.concat([
                key_option_indexes,
                -torch.ones((key_option_indexes.shape[0], seq_q_len),
                            dtype=torch.int64,
                            device=self.device)
            ], dim=1)

            _l_option_indexes = self._option_eye[l_option_indexes]
            l_states, l_attn_states, _ = model_rep(seq_q_len,
                                                   l_indexes,
                                                   [_l_option_indexes] + l_obses_list,
                                                   None,
                                                   key_pre_seq_hidden_states[:, :1],
                                                   is_prev_hidden_state=True,
                                                   query_only_attend_to_rest_key=True,
                                                   padding_mask=l_padding_masks)

            return l_states, l_attn_states

        else:
            return super().get_l_states(
                l_indexes=l_indexes,
                l_padding_masks=l_padding_masks,
                l_obses_list=l_obses_list,
                l_pre_actions=l_pre_actions,
                l_pre_seq_hidden_states=l_pre_seq_hidden_states,
                is_target=is_target
            )

    def get_l_states_with_seq_hidden_states(
        self,
        l_indexes: torch.Tensor,
        l_padding_masks: torch.Tensor,
        l_obses_list: list[torch.Tensor],
        l_pre_actions: torch.Tensor,
        l_pre_seq_hidden_states: torch.Tensor,

        key_batch: tuple[torch.Tensor,
                         torch.Tensor,
                         list[torch.Tensor],
                         torch.Tensor] | None = None,

    ) -> tuple[torch.Tensor,
               torch.Tensor | None]:
        """
        Args:
            l_indexes (torch.int32): [batch, l]
            l_padding_masks (torch.bool): [batch, l]
            l_obses_list: list([batch, l, *obs_shapes_i], ...)
            l_pre_actions: [batch, l, action_size]
            l_pre_seq_hidden_states: [batch, l, *seq_hidden_state_shape]

            key_batch:
                key_indexes: [batch, key_len]
                key_padding_masks: [batch, key_len]
                key_obses_list: list([batch, key_len, *obs_shapes_i], ...)
                key_option_indexes: [batch, key_len]
                key_pre_seq_hidden_states: [batch, key_len, *seq_hidden_state_shape]

        Returns:
            l_states: [batch, l, state_size]
            l_seq_hidden_state: [batch, l, *seq_hidden_state_shape]
        """
        if self.seq_encoder == SEQ_ENCODER.RNN and self.use_dilation:
            (key_indexes,
             key_padding_masks,
             key_obses_list,
             key_option_indexes,
             key_pre_seq_hidden_states) = key_batch

            key_padding_masks = key_padding_masks[:, :-1]
            key_obses_list = [key_obses[:, :-1] for key_obses in key_obses_list]
            _, key_rnn_state = self.model_rep(key_obses_list,
                                              None,
                                              key_pre_seq_hidden_states[:, 0],
                                              padding_mask=key_padding_masks)

            batch, l, *_ = l_indexes.shape

            l_states = None
            l_rnn_states = torch.zeros_like(l_pre_seq_hidden_states)

            for t in range(l):
                f_states, rnn_state = self.model_rep([l_obses[:, t:t + 1, ...] for l_obses in l_obses_list],
                                                     None,
                                                     key_rnn_state,
                                                     padding_mask=l_padding_masks[:, t:t + 1, ...])

                if l_states is None:
                    l_states = torch.zeros((batch, l, *f_states.shape[2:]), device=self.device)
                l_states[:, t:t + 1] = f_states

                l_rnn_states[:, t] = rnn_state

            return l_states, l_rnn_states

        elif self.seq_encoder == SEQ_ENCODER.ATTN and self.use_dilation:
            return self.get_l_states(l_indexes=l_indexes,
                                     l_padding_masks=l_padding_masks,
                                     l_obses_list=l_obses_list,
                                     l_pre_actions=None,
                                     l_pre_seq_hidden_states=l_pre_seq_hidden_states,
                                     key_batch=key_batch,
                                     is_target=False)

        else:
            return super().get_l_states_with_seq_hidden_states(
                l_indexes=l_indexes,
                l_padding_masks=l_padding_masks,
                l_obses_list=l_obses_list,
                l_pre_actions=l_pre_actions,
                l_pre_seq_hidden_states=l_pre_seq_hidden_states
            )

    def get_l_low_obses_list(self,
                             l_obses_list: list[torch.Tensor],
                             l_states: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            l_obses_list: list([batch, l, *obs_shapes_i], ...)
            l_states: [batch, l, state_size]

        Returns:
            l_low_obses_list: list([batch, l, *low_obs_shapes_i], ...)
        """

        l_low_obses_list = [l_states] + l_obses_list

        return l_low_obses_list

    def get_l_low_states(self,
                         l_indexes: torch.Tensor,
                         l_padding_masks: torch.Tensor,
                         l_low_obses_list: list[torch.Tensor],
                         l_option_indexes: torch.Tensor,
                         l_pre_actions: torch.Tensor,
                         l_pre_low_seq_hidden_states: torch.Tensor = None,
                         is_target=False) -> tuple[torch.Tensor,
                                                   torch.Tensor]:
        """
        Args:
            l_indexes (torch.int32): [batch, l]
            l_padding_masks (torch.bool): [batch, l]
            l_low_obses_list: list([batch, l, *low_obs_shapes_i], ...)
            l_option_indexes (torch.int64): [batch, l]
            l_pre_actions: [batch, l, action_size]
            l_pre_low_seq_hidden_states: [batch, l, *low_seq_hidden_state_shape]
            is_target: bool

        Returns:
            l_low_states: [batch, l, low_state_size]
            l_seq_hidden_states (optional): [batch, 1, *low_seq_hidden_state_shape]
            l_low_attn_states (optional): [batch, l, *low_seq_hidden_state_shape]
        """

        l_low_states = None
        f_or_l_low_seq_hidden_states = None

        for i, option in enumerate(self.option_list):
            mask = (l_option_indexes == i)
            if not torch.any(mask):
                continue

            o_l_states, o_f_or_l_low_seq_hidden_states = option.get_l_states(
                l_indexes=l_indexes,
                l_padding_masks=torch.logical_or(l_padding_masks,
                                                 l_option_indexes != i),
                l_obses_list=l_low_obses_list,
                l_pre_actions=l_pre_actions,
                l_pre_seq_hidden_states=l_pre_low_seq_hidden_states,
                is_target=is_target
            )

            if l_low_states is None:
                l_low_states = o_l_states
            else:
                l_low_states[mask] = o_l_states[mask]

            if f_or_l_low_seq_hidden_states is None:
                f_or_l_low_seq_hidden_states = o_f_or_l_low_seq_hidden_states
            else:
                if f_or_l_low_seq_hidden_states.shape[1] == 1:
                    f_or_l_low_seq_hidden_states[mask[:, -1:]] = o_f_or_l_low_seq_hidden_states[mask[:, -1:]]
                else:
                    f_or_l_low_seq_hidden_states[mask] = o_f_or_l_low_seq_hidden_states[mask]

        return l_low_states, f_or_l_low_seq_hidden_states

    def get_l_low_states_with_seq_hidden_states(self,
                                                l_indexes: torch.Tensor,
                                                l_padding_masks: torch.Tensor,
                                                l_low_obses_list: list[torch.Tensor],
                                                l_option_indexes: torch.Tensor,
                                                l_pre_actions: torch.Tensor,
                                                l_pre_low_seq_hidden_states: torch.Tensor = None) -> tuple[torch.Tensor,
                                                                                                           torch.Tensor]:
        """
        Args:
            l_indexes (torch.int32): [batch, l]
            l_padding_masks (torch.bool): [batch, l]
            l_low_obses_list: list([batch, l, *low_obs_shapes_i], ...)
            l_option_indexes (torch.int64): [batch, l]
            l_pre_actions: [batch, l, action_size]
            l_pre_low_seq_hidden_states: [batch, l, *low_seq_hidden_state_shape]

        Returns:
            l_low_states: [batch, l, low_state_size]
            l_low_seq_hidden_states: [batch, l, *low_seq_hidden_state_shape]
        """
        l_low_states = None
        l_low_seq_hidden_states = None

        for i, option in enumerate(self.option_list):
            mask = (l_option_indexes == i)
            if not torch.any(mask):
                continue

            o_l_states, o_l_low_seq_hidden_states = option.get_l_states_with_seq_hidden_states(
                l_indexes=l_indexes,
                l_padding_masks=torch.logical_or(l_padding_masks,
                                                 l_option_indexes != i),
                l_obses_list=l_low_obses_list,
                l_pre_actions=l_pre_actions,
                l_pre_seq_hidden_states=l_pre_low_seq_hidden_states
            )

            if l_low_states is None:
                l_low_states = o_l_states
            else:
                l_low_states[mask] = o_l_states[mask]

            if l_low_seq_hidden_states is None:
                l_low_seq_hidden_states = o_l_low_seq_hidden_states
            else:
                l_low_seq_hidden_states[mask] = o_l_low_seq_hidden_states[mask]

        return l_low_states, l_low_seq_hidden_states

    @torch.no_grad()
    def get_l_probs(self,
                    l_low_obses_list: list[torch.Tensor],
                    l_low_states: torch.Tensor,
                    l_option_indexes: torch.Tensor,
                    l_actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            l_low_obses_list: list([batch, l, *low_obs_shapes_i], ...)
            l_low_states: [batch, l, low_state_size]
            l_option_indexes (torch.int64): [batch, l]
            l_actions: [batch, l, action_size]

        Returns:
            l_low_probs: [batch, l, action_size]
        """
        l_low_probs = None

        for i, option in enumerate(self.option_list):
            mask = (l_option_indexes == i)
            if not torch.any(mask):
                continue

            o_l_probs = option.get_l_probs(l_obses_list=l_low_obses_list,
                                           l_states=l_low_states,
                                           l_actions=l_actions)

            if l_low_probs is None:
                l_low_probs = o_l_probs
            else:
                l_low_probs[mask] = o_l_probs[mask]

        return l_low_probs

    #################### ! COMPUTE LOSS ####################

    @torch.no_grad()
    def _get_td_error(self,
                      next_n_vs_over_options: torch.Tensor,

                      bn_padding_masks: torch.Tensor,
                      bn_states: torch.Tensor,
                      bn_option_indexes: torch.Tensor,
                      obn_low_obses_list: list[torch.Tensor],
                      obnx_low_target_obses_list: list[torch.Tensor],
                      low_state: torch.Tensor,
                      obnx_low_target_states: torch.Tensor,
                      bn_actions: torch.Tensor,
                      bn_rewards: torch.Tensor,
                      bn_dones: torch.Tensor,
                      bn_mu_probs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            next_n_vs_over_options: [batch, n, num_options]

            bn_padding_masks (torch.bool): [batch, b + n]
            bn_states: [batch, b + n, state_size]
            bn_option_indexes (torch.int64): [batch, b + n]
            obn_low_obses_list: list([batch, ob + n, *low_obs_shapes_i], ...)
            obnx_low_target_obses_list: list([batch, ob + n + 1, *low_obs_shapes_i], ...)
            low_state: [batch, low_state_size]
            obnx_low_target_states: [batch, ob + n + 1, low_state_size]
            bn_actions: [batch, b + n, action_size]
            bn_rewards: [batch, b + n]
            bn_dones (torch.bool): [batch, b + n]
            bn_mu_probs: [batch, b + n, action_size]

        Returns:
            The td-error of observations, [batch, 1]
        """

        n_option_indexes = bn_option_indexes[:, self.burn_in_step:]  # [batch, n]
        option_index = n_option_indexes[:, 0]  # [batch, ]

        batch = bn_states.shape[0]
        td_error = torch.zeros((batch, 1), device=self.device)

        for i, option in enumerate(self.option_list):
            mask = (option_index == i)
            if not torch.any(mask):
                continue

            o_obnx_obses_list = [torch.concat([obn_low_obses[mask], obnx_low_target_obses[mask, -1:]], dim=1)
                                 for obn_low_obses, obnx_low_target_obses
                                 in zip(obn_low_obses_list, obnx_low_target_obses_list)]

            o_td_error = option._get_td_error(
                next_n_vs_over_options=next_n_vs_over_options[mask],

                bn_padding_masks=bn_padding_masks[mask, self.option_burn_in_from:],
                bnx_obses_list=o_obnx_obses_list,
                bnx_target_obses_list=[obnx_low_target_obses[mask] for obnx_low_target_obses in obnx_low_target_obses_list],
                state=low_state[mask],
                bnx_target_states=obnx_low_target_states[mask],
                bn_actions=bn_actions[mask, self.option_burn_in_from:],
                bn_rewards=bn_rewards[mask, self.option_burn_in_from:],
                bn_dones=bn_dones[mask, self.option_burn_in_from:],
                bn_mu_probs=bn_mu_probs[mask, self.option_burn_in_from:])

            td_error[mask] = o_td_error

        return td_error

    def _train_q(self,
                 bn_indexes: torch.Tensor,
                 bn_padding_masks: torch.Tensor,
                 bnx_states: torch.Tensor,
                 bn_option_indexes: torch.Tensor,
                 obnx_low_obses_list: list[torch.Tensor],
                 obnx_low_target_obses_list: list[torch.Tensor],
                 bn_actions: torch.Tensor,
                 bn_rewards: torch.Tensor,
                 bn_dones: torch.Tensor,
                 bn_mu_probs: torch.Tensor,
                 obnx_pre_low_seq_hidden_states: torch.Tensor | None = None,
                 priority_is: torch.Tensor | None = None) -> tuple[torch.Tensor,
                                                                   torch.Tensor,
                                                                   list[torch.Tensor],
                                                                   list[torch.Tensor],
                                                                   torch.Tensor,
                                                                   torch.Tensor]:
        """
        Args:
            bn_indexes (torch.int32): [batch, b + n],
            bn_padding_masks (torch.bool): [batch, b + n],
            bnx_states: [batch, b + n + 1]
            bn_option_indexes (torch.int64): [batch, b + n],
            obnx_low_obses_list: list([batch, ob + n + 1, *low_obs_shapes_i], ...)
            obnx_low_target_obses_list: list([batch, ob + n + 1, *low_obs_shapes_i], ...)
            bn_actions: [batch, b + n, action_size],
            bn_rewards: [batch, b + n],
            bn_dones (torch.bool): [batch, b + n]
            bn_mu_probs: [batch, b + n, action_size]
            obnx_pre_low_seq_hidden_states: [batch, ob + n + 1, *low_seq_hidden_state_shape]
            priority_is: [batch, 1]

        Returns:
            obnx_low_states: [batch, ob + n + 1, low_state_size]
            obnx_low_target_states: [batch, ob + n + 1, low_state_size]
            next_n_vs_over_options: [batch, n, num_options]
            y: [batch, 1]
        """

        batch = bn_indexes.shape[0]
        option_index = bn_option_indexes[:, self.burn_in_step]

        next_n_states = bnx_states[:, self.burn_in_step + 1:]
        next_n_vs_over_options_list = [v(next_n_states) for v in self.model_target_v_over_options_list]  # list([batch, n, num_options], ...)
        next_n_vs_over_options, _ = torch.min(torch.stack(next_n_vs_over_options_list, -1), dim=-1)

        d_y = torch.zeros((batch, 1), device=self.device)
        c_y = torch.zeros((batch, 1), device=self.device)

        obnx_low_states = None
        obnx_low_target_states = None

        # Get all states of all options, then replace them according to option_indexes
        for i, option in enumerate(self.option_list):
            mask = (option_index == i)
            if not torch.any(mask):
                continue

            (o_obnx_indexes,
             o_obnx_padding_masks,
             o_obnx_pre_actions) = option.get_bnx_data(
                bn_indexes=bn_indexes[mask, self.option_burn_in_from:],
                bn_padding_masks=bn_padding_masks[mask, self.option_burn_in_from:],
                bn_actions=bn_actions[mask, self.option_burn_in_from:],
            )

            o_obnx_low_states, _ = option.get_l_states(
                l_indexes=o_obnx_indexes,
                l_padding_masks=o_obnx_padding_masks,
                l_obses_list=[o[mask] for o in obnx_low_obses_list],
                l_pre_actions=o_obnx_pre_actions,
                l_pre_seq_hidden_states=obnx_pre_low_seq_hidden_states[mask],
                is_target=False
            )

            with torch.no_grad():
                o_obnx_low_target_states, _ = option.get_l_states(
                    l_indexes=o_obnx_indexes,
                    l_padding_masks=o_obnx_padding_masks,
                    l_obses_list=[o[mask] for o in obnx_low_target_obses_list],
                    l_pre_actions=o_obnx_pre_actions,
                    l_pre_seq_hidden_states=obnx_pre_low_seq_hidden_states[mask],
                    is_target=True
                )

            if obnx_low_states is None:
                obnx_low_states = torch.zeros((batch, *o_obnx_low_states.shape[1:]), device=o_obnx_low_states.device)
                obnx_low_target_states = torch.zeros((batch, *o_obnx_low_target_states.shape[1:]), device=o_obnx_low_target_states.device)
            obnx_low_states[mask] = o_obnx_low_states
            obnx_low_target_states[mask] = o_obnx_low_target_states

            o_obn_padding_masks = torch.logical_or(bn_padding_masks[mask, self.option_burn_in_from:],
                                                   bn_option_indexes[mask, self.option_burn_in_from:] != i)

            o_d_y, o_c_y = option.compute_rep_q_grads(
                next_n_vs_over_options=next_n_vs_over_options[mask],

                bn_indexes=bn_indexes[mask, self.option_burn_in_from:],
                bn_padding_masks=o_obn_padding_masks,
                bnx_obses_list=[obnx_low_obses[mask]
                                for obnx_low_obses in obnx_low_obses_list],
                bnx_target_obses_list=[obnx_low_target_obses[mask]
                                       for obnx_low_target_obses in obnx_low_target_obses_list],
                bnx_states=o_obnx_low_states,
                bnx_target_states=o_obnx_low_target_states,
                bn_actions=bn_actions[mask, self.option_burn_in_from:],
                bn_rewards=bn_rewards[mask, self.option_burn_in_from:],
                bn_dones=bn_dones[mask, self.option_burn_in_from:],
                bn_mu_probs=bn_mu_probs[mask, self.option_burn_in_from:],
                priority_is=priority_is[mask] if self.use_replay_buffer and self.use_priority else None
            )

            if o_d_y is not None:
                d_y[mask] = o_d_y

            if o_c_y is not None:
                c_y[mask] = o_c_y

        for i, option in enumerate(self.option_list):
            option.train_rep_q()

        return (obnx_low_states, obnx_low_target_states,
                next_n_vs_over_options,
                d_y + c_y)

    def _train_v(self,
                 y: torch.Tensor,

                 bn_indexes: torch.Tensor,
                 bn_padding_masks: torch.Tensor,
                 bnx_obses_list: list[torch.Tensor],
                 bnx_states: torch.Tensor,
                 bn_option_indexes: torch.Tensor,
                 bn_actions: torch.Tensor,
                 priority_is: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            y: [batch, 1]

            bn_indexes (torch.int32): [batch, b + n]
            bn_padding_masks (torch.bool): [batch, b + n]
            bnx_obses_list: list([batch, b + n + 1, *obs_shapes_i], ...)
            bnx_states: [batch, b + n + 1, state_size]
            bn_option_indexes (torch.int64): [batch, b + n]
            bn_actions: [batch, b + n, action_size]
            priority_is: [batch, 1]

        Returns:
            loss_v_list: list(torch.float32, ...)
        """

        state = bnx_states[:, self.burn_in_step, ...]
        option_index = bn_option_indexes[:, self.burn_in_step]

        option_index_one_hot = nn.functional.one_hot(option_index, self.num_options)  # [batch, num_options]

        loss_v_list: list[torch.Tensor] = []
        loss_none_mse = nn.MSELoss(reduction='none')

        for i, model_v_over_options in enumerate(self.model_v_over_options_list):
            v_over_options = model_v_over_options(state)  # [batch, num_options]
            v = (v_over_options * option_index_one_hot).sum(-1, keepdim=True)  # [batch, 1]

            if self.clip_epsilon > 0:
                target_v_over_options = self.model_target_v_over_options_list[i](state.detach())  # [batch, num_options]
                target_v = (target_v_over_options * option_index_one_hot).sum(-1, keepdim=True)  # [batch, 1]

                clipped_v = target_v + torch.clamp(
                    v - target_v,
                    -self.clip_epsilon,
                    self.clip_epsilon,
                )

                loss_v_a = loss_none_mse(clipped_v, y)  # [batch, 1]
                loss_v_b = loss_none_mse(v, y)  # [batch, 1]

                loss_v = torch.maximum(loss_v_a, loss_v_b)
            else:
                loss_v = loss_none_mse(v, y)

            if priority_is is not None:
                loss_v = loss_v * priority_is  # [batch, 1]

            loss_v = torch.mean(loss_v)
            loss_v_list.append(loss_v)

            optimizer = self.optimizer_v_list[i]

            if optimizer is not None:
                optimizer.zero_grad()

            if loss_v.requires_grad:
                loss_v.backward(retain_graph=True)

        grads_rep_main = [m.grad.detach() if m.grad is not None else None
                          for m in self.model_rep.parameters()]

        grads_v_main_list = [[m.grad.detach() if m.grad is not None else None for m in v.parameters()]
                             for v in self.model_v_over_options_list]

        """ Siamese Representation Learning """
        loss_siamese, loss_siamese_q = None, None
        if self.siamese is not None:
            loss_siamese, loss_siamese_q = self._train_siamese_representation_learning(
                grads_rep_main=grads_rep_main,
                grads_v_main_list=grads_v_main_list,
                bn_indexes=bn_indexes,
                bn_padding_masks=bn_padding_masks,
                bn_obses_list=[bnx_obses[:, :-1] for bnx_obses in bnx_obses_list],
                bn_actions=bn_actions)

        for opt_v in self.optimizer_v_list:
            if opt_v is not None:
                opt_v.step()

        return loss_v_list, loss_siamese, loss_siamese_q

    def _train_siamese_representation_learning(self,
                                               grads_rep_main: list[torch.Tensor],
                                               grads_v_main_list: list[list[torch.Tensor]],
                                               bn_indexes: torch.Tensor,
                                               bn_padding_masks: torch.Tensor,
                                               bn_obses_list: list[torch.Tensor],
                                               bn_actions: torch.Tensor) -> tuple[torch.Tensor,
                                                                                  torch.Tensor | None]:
        """
        Args:
            grads_rep_main list(torch.Tensor)
            grads_v_main_list list(list(torch.Tensor))
            bn_indexes (torch.int32): [batch, b + n]
            bn_padding_masks (torch.bool): [batch, b + n]
            bn_obses_list: list([batch, b + n, *obs_shapes_i], ...)
            bn_actions: [batch, b + n, action_size]

        Returns:
            loss_siamese
            loss_siamese_q
        """

        if not any([p.requires_grad for p in self.model_rep.parameters()]):
            return None, None

        n_padding_masks = bn_padding_masks[:, self.burn_in_step:]
        n_obses_list = [bn_obses[:, self.burn_in_step:, ...] for bn_obses in bn_obses_list]
        encoder_list = self.model_rep.get_augmented_encoders(n_obses_list)  # [batch, n, f], ...
        target_encoder_list = self.model_target_rep.get_augmented_encoders(n_obses_list)  # [batch, n, f], ...

        if not isinstance(encoder_list, tuple):
            encoder_list = (encoder_list, )
            target_encoder_list = (target_encoder_list, )

        batch, n, *_ = encoder_list[0].shape

        if self.siamese == SIAMESE.ATC:
            _encoder_list = [e.reshape(batch * n, -1) for e in encoder_list]  # [batch * n, f], ...
            _target_encoder_list = [t_e.reshape(batch * n, -1) for t_e in target_encoder_list]  # [batch * n, f], ...
            logits_list = [torch.mm(e, weight) for e, weight in zip(_encoder_list, self.contrastive_weight_list)]
            logits_list = [torch.mm(logits, t_e.t()) for logits, t_e in zip(logits_list, _target_encoder_list)]  # [batch * n, batch * n], ...
            contrastive_labels = torch.block_diag(*torch.ones(batch, n, n, device=self.device))  # [batch * n, batch * n]

            padding_mask = n_padding_masks.reshape(batch * n, 1)  # [batch * n, 1]

            loss_siamese_list = [functional.binary_cross_entropy_with_logits(logits,
                                                                             contrastive_labels,
                                                                             reduction='none')
                                 for logits in logits_list]
            loss_siamese_list = [(loss * padding_mask).mean() for loss in loss_siamese_list]

        elif self.siamese == SIAMESE.BYOL:
            _encoder_list = [e.reshape(batch * n, -1) for e in encoder_list]  # [batch * n, f], ...
            projection_list = [pro(encoder) for pro, encoder in zip(self.model_rep_projection_list, _encoder_list)]
            prediction_list = [pre(projection) for pre, projection in zip(self.model_rep_prediction_list, projection_list)]
            _target_encoder_list = [t_e.reshape(batch * n, -1) for t_e in target_encoder_list]  # [batch * n, f], ...
            t_projection_list = [t_pro(t_e) for t_pro, t_e in zip(self.model_target_rep_projection_list, _target_encoder_list)]

            padding_mask = n_padding_masks.reshape(batch * n)  # [batch * n, ]

            loss_siamese_list = [(functional.cosine_similarity(prediction, t_projection) * padding_mask).mean()  # [batch * n, ] -> [1, ]
                                 for prediction, t_projection in zip(prediction_list, t_projection_list)]

        if self.siamese_use_q:
            obses_list_at_n = [n_obses[:, 0:1, ...] for n_obses in n_obses_list]

            _encoder = [e[:, 0:1, ...] for e in encoder_list]
            _target_encoder = [t_e[:, 0:1, ...] for t_e in target_encoder_list]

            pre_actions_at_n = bn_actions[:, self.burn_in_step - 1:self.burn_in_step, ...]

            if self.seq_encoder in (None, SEQ_ENCODER.RNN):
                padding_masks_at_n = bn_padding_masks[:, self.burn_in_step:self.burn_in_step + 1]
                state = self.model_rep.get_state_from_encoders(_encoder if len(_encoder) > 1 else _encoder[0],
                                                               obses_list_at_n,
                                                               pre_actions_at_n,
                                                               self.get_initial_seq_hidden_state(batch, False).unsqueeze(1),
                                                               padding_mask=padding_masks_at_n)
                target_state = self.model_target_rep.get_state_from_encoders(_target_encoder if len(_target_encoder) > 1 else _target_encoder[0],
                                                                             obses_list_at_n,
                                                                             pre_actions_at_n,
                                                                             self.get_initial_seq_hidden_state(batch, False).unsqueeze(1),
                                                                             padding_mask=padding_masks_at_n)
                state = state[:, 0, ...]
                target_state = target_state[:, 0, ...]

            elif self.seq_encoder == SEQ_ENCODER.ATTN:
                indexes_at_n = bn_indexes[:, self.burn_in_step:self.burn_in_step + 1]
                padding_masks_at_n = bn_padding_masks[:, self.burn_in_step:self.burn_in_step + 1]
                state = self.model_rep.get_state_from_encoders(1,
                                                               _encoder if len(_encoder) > 1 else _encoder[0],
                                                               indexes_at_n,
                                                               obses_list_at_n,
                                                               pre_actions_at_n,
                                                               None,
                                                               padding_mask=padding_masks_at_n)
                target_state = self.model_target_rep.get_state_from_encoders(1,
                                                                             _encoder if len(_encoder) > 1 else _encoder[0],
                                                                             indexes_at_n,
                                                                             obses_list_at_n,
                                                                             pre_actions_at_n,
                                                                             None,
                                                                             padding_mask=padding_masks_at_n)
                state = state[:, 0, ...]
                target_state = target_state[:, 0, ...]

            v_loss_list = []

            v_list = [v(state)
                      for v in self.model_v_over_options_list]  # ([batch, d_action_summed_size], [batch, 1]), ...
            target_v_list = [v(target_state)
                             for v in self.model_target_v_over_options_list]  # ([batch, d_action_summed_size], [batch, 1]), ...

            v_loss_list += [functional.mse_loss(v, t_v)
                            for v, t_v in zip(v_list, target_v_list)]

            loss_list = loss_siamese_list + v_loss_list
        else:
            loss_list = loss_siamese_list

        if self.siamese_use_q:
            if self.siamese_use_adaptive:
                for grads_v_main, v_loss, v in zip(grads_v_main_list, v_loss_list, self.model_v_over_options_list):
                    self.calculate_adaptive_weights(grads_v_main, [v_loss], v)
            else:
                for v_loss, v in zip(v_loss_list, self.model_v_over_options_list):
                    v_loss.backward(inputs=list(v.parameters()), retain_graph=True)

        loss = sum(loss_list)

        if self.siamese_use_adaptive:
            self.calculate_adaptive_weights(grads_rep_main, loss_list, self.model_rep)
        else:
            loss.backward(inputs=list(self.model_rep.parameters()), retain_graph=True)

        self.optimizer_siamese.zero_grad()
        if self.siamese == SIAMESE.ATC:
            loss.backward(inputs=self.contrastive_weight_list, retain_graph=True)
        elif self.siamese == SIAMESE.BYOL:
            loss.backward(inputs=list(chain(*[pro.parameters() for pro in self.model_rep_projection_list],
                                            *[pre.parameters() for pre in self.model_rep_prediction_list])), retain_graph=True)
        self.optimizer_siamese.step()

        return sum(loss_siamese_list), sum(v_loss_list) if self.siamese_use_q else None

    def _train_terminations(self,
                            y: torch.Tensor,

                            bnx_states: torch.Tensor,
                            bn_option_indexes: torch.Tensor,
                            obnx_low_target_obses_list: list[torch.Tensor],
                            obnx_low_target_states: list[torch.Tensor],
                            bn_dones: torch.Tensor,
                            priority_is: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            y: [batch, 1]

            bnx_states: [batch, b + n + 1, state_size]
            bn_option_indexes (torch.int64): [batch, b + n]s
            obnx_low_target_obses_list: list([batch, ob + n + 1, *low_obs_shapes_i], ...)
            obnx_low_target_states: [batch, ob + n + 1, low_state_size]
            bn_dones (torch.bool): [batch, b + n]
            priority_is: [batch, 1]

        Returns:
            loss_v_list: list(torch.float32, ...)
        """

        state = bnx_states[:, self.burn_in_step, ...]
        option_index = bn_option_indexes[:, self.burn_in_step]

        low_obs_list = [obnx_low_obses[:, self.option_burn_in_step]
                        for obnx_low_obses in obnx_low_target_obses_list]
        done = bn_dones[:, self.burn_in_step]

        with torch.no_grad():
            target_v_over_options_list = [v(state) for v in self.model_target_v_over_options_list]  # list([batch, num_options], ...)
            stacked_target_v_over_options = torch.stack(target_v_over_options_list)  # [ensemble, batch, num_options]
            target_v_over_options, _ = stacked_target_v_over_options.min(dim=0)  # [batch, num_options]

        for i, option in enumerate(self.option_list):
            mask = (option_index == i)
            if not torch.any(mask):
                continue

            low_target_state = obnx_low_target_states[:, self.option_burn_in_step]

            option.compute_termination_grads(
                terminal_entropy=self.terminal_entropy,
                obs_list=[o[mask] for o in low_obs_list],
                state=low_target_state[mask],
                y=y[mask],
                v_over_options=target_v_over_options[mask],
                done=done[mask],
                priority_is=priority_is[mask]
            )

        for i, option in enumerate(self.option_list):
            option.train_termination()

    def _train_rnd(self,
                   bn_padding_masks: torch.Tensor,
                   bn_option_indexes: torch.Tensor,
                   bn_states: torch.Tensor) -> torch.Tensor:
        n_padding_masks = bn_padding_masks[:, self.burn_in_step:]  # [batch, n]
        n_option_indexes = bn_option_indexes[:, self.burn_in_step:]  # [batch, n]
        n_states = bn_states[:, self.burn_in_step:]  # [batch, n, state_size]

        rnd = self.model_rnd.cal_rnd(n_states)  # [batch, n, num_options, f]
        with torch.no_grad():
            t_rnd = self.model_target_rnd.cal_rnd(n_states)  # [batch, n, num_options, f]

        n_option_indexes = n_option_indexes * ~n_padding_masks  # preventing -1 error in onehot
        _i = nn.functional.one_hot(n_option_indexes, self.num_options).float().unsqueeze(-1)  # [batch, n, num_options, 1]
        rnd = (_i * rnd).sum(-2)
        # [batch, n, d_action_summed_size, f] -> [batch, n, f]
        t_rnd = (_i * t_rnd).sum(-2)
        # [batch, n, d_action_summed_size, f] -> [batch, n, f]

        _loss = nn.functional.mse_loss(rnd, t_rnd, reduction='none')
        _loss = _loss * ~n_padding_masks.unsqueeze(-1)
        loss = torch.mean(_loss)

        self.optimizer_rnd.zero_grad()
        loss.backward(inputs=list(self.model_rnd.parameters()))
        self.optimizer_rnd.step()

        return loss

    def _train(self,
               bn_indexes: torch.Tensor,
               bn_padding_masks: torch.Tensor,
               bnx_obses_list: list[torch.Tensor],
               bn_option_indexes: torch.Tensor,
               bn_actions: torch.Tensor,
               bn_rewards: torch.Tensor,
               bn_dones: torch.Tensor,
               bn_mu_probs: torch.Tensor,
               bnx_pre_seq_hidden_states: torch.Tensor,
               bnx_pre_low_seq_hidden_states: torch.Tensor,
               priority_is: torch.Tensor = None,
               key_batch: tuple[torch.Tensor | list[torch.Tensor], ...] | None = None) -> tuple[torch.Tensor,
                                                                                                list[torch.Tensor],
                                                                                                torch.Tensor]:
        """
        Args:
            bn_indexes (torch.int32): [batch, b + n]
            bn_padding_masks (torch.bool): [batch, b + n]
            bnx_obses_list: list([batch, b + n + 1, *obs_shapes_i], ...)
            bn_option_indexes (torch.int64): [batch, b + n]
            bn_actions: [batch, b + n, action_size]
            bn_rewards: [batch, b + n]
            bn_dones (torch.bool): [batch, b + n]
            bn_mu_probs: [batch, b + n, action_size]
            bnx_pre_seq_hidden_states: [batch, b + n + 1, *seq_hidden_state_shape]
            bnx_pre_low_seq_hidden_states: [batch, b + n + 1, *low_seq_hidden_state_shape]
                (start from self.option_burn_in_from)
            priority_is: [batch, 1]

            key_batch:
                key_indexes: [batch, key_len]
                key_padding_masks: [batch, key_len]
                key_obses_list: list([batch, key_len, *obs_shapes_i], ...)
                key_option_indexes: [batch, key_len]
                key_pre_seq_hidden_states: [batch, key_len, *seq_hidden_state_shape]

        Returns:
            bnx_target_states: [batch, b + n + 1, state_size]
            obnx_low_target_obses_list: list([batch, ob + n + 1, *low_obs_shapes_i], ...)
            obnx_low_target_states: [batch, ob + n + 1, low_state_size]
        """

        if self.global_step % self.update_target_per_step == 0:
            self._update_target_variables(tau=self.tau)

        (bnx_indexes,
         bnx_padding_masks,
         bnx_pre_actions) = self.get_bnx_data(bn_indexes=bn_indexes,
                                              bn_padding_masks=bn_padding_masks,
                                              bn_actions=bn_actions)

        bnx_states, _ = self.get_l_states(l_indexes=bnx_indexes,
                                          l_padding_masks=bnx_padding_masks,
                                          l_obses_list=bnx_obses_list,
                                          l_pre_actions=bnx_pre_actions,
                                          l_pre_seq_hidden_states=bnx_pre_seq_hidden_states,

                                          key_batch=key_batch,

                                          is_target=False)
        # [batch, b + n + 1, state_size]

        with torch.no_grad():
            bnx_target_states, _ = self.get_l_states(l_indexes=bnx_indexes,
                                                     l_padding_masks=bnx_padding_masks,
                                                     l_obses_list=bnx_obses_list,
                                                     l_pre_actions=bnx_pre_actions,
                                                     l_pre_seq_hidden_states=bnx_pre_seq_hidden_states,

                                                     key_batch=key_batch,

                                                     is_target=True)
            # [batch, b + n + 1, state_size]

        n_option_indexes = bn_option_indexes[:, self.burn_in_step:]  # [batch, n]
        option_index = n_option_indexes[:, 0]  # [batch, ]

        obnx_low_obses_list = self.get_l_low_obses_list(l_obses_list=[bnx_obses[:, self.option_burn_in_from:] for bnx_obses in bnx_obses_list],
                                                        l_states=bnx_states[:, self.option_burn_in_from:])
        obnx_low_target_obses_list = self.get_l_low_obses_list(l_obses_list=[bnx_obses[:, self.option_burn_in_from:] for bnx_obses in bnx_obses_list],
                                                               l_states=bnx_target_states[:, self.option_burn_in_from:])

        if self.optimizer_rep:
            self.optimizer_rep.zero_grad()

        (obnx_low_states,
         obnx_low_target_states,
         next_n_vs_over_options,
         y) = self._train_q(bn_indexes=bn_indexes,
                            bn_padding_masks=bn_padding_masks,
                            bnx_states=bnx_states,
                            bn_option_indexes=bn_option_indexes,
                            obnx_low_obses_list=[o for o in obnx_low_obses_list],
                            obnx_low_target_obses_list=obnx_low_target_obses_list,
                            bn_actions=bn_actions,
                            bn_rewards=bn_rewards,
                            bn_dones=bn_dones,
                            bn_mu_probs=bn_mu_probs,
                            obnx_pre_low_seq_hidden_states=bnx_pre_low_seq_hidden_states[:, self.option_burn_in_from:],
                            priority_is=priority_is)
        obnx_low_states = obnx_low_states.detach()  # [batch, ob + n + 1, low_state_size]
        obnx_low_target_states = obnx_low_target_states.detach()  # [batch, ob + n + 1, low_state_size]
        next_n_vs_over_options = next_n_vs_over_options.detach()
        # [batch, n, num_options]

        (loss_v_list,
         loss_siamese,
         loss_siamese_q) = self._train_v(y=y,

                                         bn_indexes=bn_indexes,
                                         bn_padding_masks=bn_padding_masks,
                                         bnx_obses_list=bnx_obses_list,
                                         bnx_states=bnx_states,
                                         bn_option_indexes=bn_option_indexes,
                                         bn_actions=bn_actions,
                                         priority_is=priority_is)

        if self.optimizer_rep:
            self.optimizer_rep.step()

        with torch.no_grad():
            bnx_states, _ = self.get_l_states(l_indexes=bnx_indexes,
                                              l_padding_masks=bnx_padding_masks,
                                              l_obses_list=bnx_obses_list,
                                              l_pre_actions=bnx_pre_actions,
                                              l_pre_seq_hidden_states=bnx_pre_seq_hidden_states,

                                              key_batch=key_batch,

                                              is_target=False)
            # [batch, b + n + 1, state_size]

            obnx_low_obses_list = self.get_l_low_obses_list(l_obses_list=[bnx_obses[:, self.option_burn_in_from:] for bnx_obses in bnx_obses_list],
                                                            l_states=bnx_states[:, self.option_burn_in_from:])
            # list([batch, ob + n + 1, *low_obs_shapes_i], ...)

        self._train_terminations(y=y,

                                 bnx_states=bnx_states,
                                 bn_option_indexes=bn_option_indexes,
                                 obnx_low_target_obses_list=obnx_low_target_obses_list,
                                 obnx_low_target_states=obnx_low_target_states,
                                 bn_dones=bn_dones,
                                 priority_is=priority_is)

        for i, option in enumerate(self.option_list):
            mask = (option_index == i)
            if not torch.any(mask):
                continue

            (o_obnx_indexes,
             o_obnx_padding_masks,
             o_obnx_pre_actions) = option.get_bnx_data(
                bn_indexes=bn_indexes[mask, self.option_burn_in_from:],
                bn_padding_masks=bn_padding_masks[mask, self.option_burn_in_from:],
                bn_actions=bn_actions[mask, self.option_burn_in_from:],
            )

            with torch.no_grad():
                o_obnx_low_states, _ = option.get_l_states(
                    l_indexes=o_obnx_indexes,
                    l_padding_masks=o_obnx_padding_masks,
                    l_obses_list=[o[mask] for o in obnx_low_obses_list],
                    l_pre_actions=o_obnx_pre_actions,
                    l_pre_seq_hidden_states=bnx_pre_low_seq_hidden_states[mask, self.option_burn_in_from:],
                    is_target=False
                )
                obnx_low_states[mask] = o_obnx_low_states

            option.train_policy_alpha(bn_padding_masks=bn_padding_masks[mask, self.option_burn_in_from:],
                                      bn_obses_list=[obnx_low_obses[mask, :-1, ...]
                                                     for obnx_low_obses in obnx_low_obses_list],
                                      bnx_states=o_obnx_low_states,
                                      bn_actions=bn_actions[mask, self.option_burn_in_from:],
                                      bn_mu_probs=bn_mu_probs[mask, self.option_burn_in_from:])

        if self.use_rnd:
            loss_rnd = self._train_rnd(
                bn_padding_masks=bn_padding_masks,
                bn_option_indexes=bn_option_indexes,
                bn_states=bnx_states[:, :-1]
            )

        if self.summary_writer is not None and self.global_step % self.write_summary_per_step == 0:
            self.summary_available = True

            if self.use_replay_buffer:
                curr_rb_id = self.replay_buffer.get_curr_id()
                self.summary_writer.add_scalar('metric/replay_id', curr_rb_id, self.global_step)

            with torch.no_grad():
                self.summary_writer.add_scalar('loss/v', loss_v_list[0], self.global_step)

                if self.siamese is not None and loss_siamese is not None:
                    self.summary_writer.add_scalar('loss/siamese',
                                                   loss_siamese,
                                                   self.global_step)
                    if self.siamese_use_q and loss_siamese_q is not None:
                        self.summary_writer.add_scalar('loss/siamese_q',
                                                       loss_siamese_q,
                                                       self.global_step)

                if self.use_rnd:
                    self.summary_writer.add_scalar('loss/rnd', loss_rnd, self.global_step)

            self.summary_writer.flush()

        return (obnx_low_target_obses_list,
                obnx_low_target_states,
                next_n_vs_over_options)

    _avg_logged_episode_reward = None
    _avg_logged_episode_count = 1

    def log_episode(self, **episode_trans: np.ndarray) -> None:
        ep_rewards = episode_trans['ep_rewards']
        reward = ep_rewards.sum()
        self._avg_logged_episode_count += 1
        if self._avg_logged_episode_reward is None:
            self._avg_logged_episode_reward = reward
        else:
            self._avg_logged_episode_reward += (reward - self._avg_logged_episode_reward) \
                / self._avg_logged_episode_count

        if reward < self._avg_logged_episode_reward:
            return

        if self.summary_writer is None or not self.summary_available:
            return

        ep_indexes = episode_trans['ep_indexes']
        ep_obses_list = episode_trans['ep_obses_list']
        ep_actions = episode_trans['ep_actions']

        ep_option_indexes = episode_trans['ep_option_indexes']
        ep_option_changed_indexes = episode_trans['ep_option_changed_indexes']

        image = plot_episode_option_indexes(ep_option_indexes, ep_option_changed_indexes, self.num_options)
        self.summary_writer.add_figure(f'option_index', image, self.global_step)

        if self.seq_encoder == SEQ_ENCODER.ATTN:
            with torch.no_grad():
                if self.use_dilation:
                    summary_name = 'key_attn_weight'
                    key_option_changed_indexes = np.unique(ep_option_changed_indexes)  # [key_len, ]
                    self.summary_writer.add_scalar('metric/key_len', len(key_option_changed_indexes), self.global_step)

                    key_indexes = ep_indexes[:, key_option_changed_indexes]  # [1, key_len]
                    key_obses_list = [ep_obses[:, key_option_changed_indexes]
                                      for ep_obses in ep_obses_list]  # [1, key_len, state_size]
                    key_option_indexes = ep_option_indexes[:, key_option_changed_indexes]  # [1, key_len]

                    ep_len = ep_indexes.shape[1]
                    l_indexes = np.concatenate([key_indexes, ep_indexes], axis=1)
                    l_obses_list = [np.concatenate([key_obses, ep_obses], axis=1)
                                    for key_obses, ep_obses in zip(key_obses_list, ep_obses_list)]

                    l_indexes = torch.from_numpy(l_indexes).to(self.device)
                    l_obses_list = [torch.from_numpy(o).to(self.device) for o in l_obses_list]
                    l_option_indexes = torch.concat([
                        torch.from_numpy(key_option_indexes).to(self.device),
                        -torch.ones((key_option_indexes.shape[0], ep_len), dtype=torch.int64, device=self.device)
                    ], dim=1)

                    _l_option_indexes = self._option_eye[l_option_indexes]
                    *_, attn_weights_list = self.model_rep(ep_len,
                                                           l_indexes,
                                                           [_l_option_indexes] + l_obses_list,
                                                           None,
                                                           None,
                                                           query_only_attend_to_rest_key=True)

                else:
                    summary_name = 'attn_weight'
                    pre_l_actions = gen_pre_n_actions(ep_actions)
                    *_, attn_weights_list = self.model_rep(ep_indexes.shape[1],
                                                           torch.from_numpy(ep_indexes).to(self.device),
                                                           [torch.from_numpy(o).to(self.device) for o in ep_obses_list],
                                                           torch.from_numpy(pre_l_actions).to(self.device),
                                                           None)

                for i, attn_weight in enumerate(attn_weights_list):
                    image = plot_attn_weight(attn_weight[0].cpu().numpy())
                    self.summary_writer.add_figure(f'{summary_name}/{i}', image, self.global_step)

        self.summary_available = False

    def put_episode(self,
                    ep_indexes: np.ndarray,
                    ep_obses_list: list[np.ndarray],
                    ep_option_indexes: np.ndarray,
                    ep_option_changed_indexes: np.ndarray,
                    ep_actions: np.ndarray,
                    ep_rewards: np.ndarray,
                    ep_dones: np.ndarray,
                    ep_probs: list[np.ndarray],
                    ep_pre_seq_hidden_states: np.ndarray,
                    ep_pre_low_seq_hidden_states: np.ndarray) -> None:
        # Ignore episodes which length is too short
        # if ep_indexes.shape[1] < self.n_step:
        #     return

        assert ep_indexes.dtype == np.int32
        assert ep_option_indexes.dtype == np.int8
        assert ep_option_changed_indexes.dtype == np.int32

        ep_padding_masks = np.zeros_like(ep_indexes, dtype=bool)
        ep_padding_masks[:, -1] = True  # The last step is next_step
        ep_padding_masks[ep_indexes == -1] = True

        if self.use_replay_buffer:
            self._fill_replay_buffer(ep_indexes=ep_indexes,
                                     ep_padding_masks=ep_padding_masks,
                                     ep_obses_list=ep_obses_list,
                                     ep_option_indexes=ep_option_indexes,
                                     ep_option_changed_indexes=ep_option_changed_indexes,
                                     ep_actions=ep_actions,
                                     ep_rewards=ep_rewards,
                                     ep_dones=ep_dones,
                                     ep_probs=ep_probs,
                                     ep_pre_seq_hidden_states=ep_pre_seq_hidden_states,
                                     ep_pre_low_seq_hidden_states=ep_pre_low_seq_hidden_states)
        else:
            self.batch_buffer.put_episode(ep_indexes=ep_indexes,
                                          ep_padding_masks=ep_padding_masks,
                                          ep_obses_list=ep_obses_list,
                                          ep_option_indexes=ep_option_indexes,
                                          ep_option_changed_indexes=ep_option_changed_indexes,
                                          ep_actions=ep_actions,
                                          ep_rewards=ep_rewards,
                                          ep_dones=ep_dones,
                                          ep_probs=ep_probs,
                                          ep_pre_seq_hidden_states=ep_pre_seq_hidden_states,
                                          ep_pre_low_seq_hidden_states=ep_pre_low_seq_hidden_states)

    def _fill_replay_buffer(self,
                            ep_indexes: np.ndarray,
                            ep_padding_masks: np.ndarray,
                            ep_obses_list: list[np.ndarray],
                            ep_option_indexes: np.ndarray,
                            ep_option_changed_indexes: np.ndarray,
                            ep_actions: np.ndarray,
                            ep_rewards: np.ndarray,
                            ep_dones: np.ndarray,
                            ep_probs: np.ndarray,
                            ep_pre_seq_hidden_states: np.ndarray,
                            ep_pre_low_seq_hidden_states: np.ndarray) -> None:
        """
        Args:
            ep_indexes (np.int32): [1, ep_len]
            ep_padding_masks: (bool): [1, ep_len]
            ep_obses_list (np): list([1, ep_len, *obs_shapes_i], ...)
            ep_option_indexes (np.int8): [1, ep_len]
            ep_option_changed_indexes (np.int32): [1, ep_len]
            ep_actions (np): [1, ep_len, action_size]
            ep_rewards (np): [1, ep_len]
            ep_dones (bool): [1, ep_len]
            ep_probs (np): [1, ep_len, action_size]
            ep_pre_seq_hidden_states (np): [1, ep_len, *seq_hidden_state_shape]
            ep_pre_low_seq_hidden_states (np): [1, ep_len, *low_seq_hidden_state_shape]
        """

        # Reshape [1, ep_len, ...] to [ep_len, ...]
        index = ep_indexes.squeeze(0)
        padding_mask = ep_padding_masks.squeeze(0)
        obs_list = [ep_obses.squeeze(0) for ep_obses in ep_obses_list]
        if self.use_normalization:
            self._udpate_normalizer([torch.from_numpy(obs).to(self.device) for obs in obs_list])
        option_index = ep_option_indexes.squeeze(0)
        option_changed_index = ep_option_changed_indexes.squeeze(0)
        action = ep_actions.squeeze(0)
        reward = ep_rewards.squeeze(0)
        done = ep_dones.squeeze(0)
        mu_prob = ep_probs.squeeze(0)
        pre_seq_hidden_state = ep_pre_seq_hidden_states.squeeze(0)
        pre_low_seq_hidden_state = ep_pre_low_seq_hidden_states.squeeze(0)

        storage_data = {
            'index': index,
            'padding_mask': padding_mask,
            **{f'obs_{name}': obs for name, obs in zip(self.obs_names, obs_list)},
            'option_index': option_index,
            'option_changed_index': option_changed_index,
            'action': action,
            'reward': reward,
            'done': done,
            'mu_prob': mu_prob,
            'pre_seq_hidden_state': pre_seq_hidden_state,
            'pre_low_seq_hidden_state': pre_low_seq_hidden_state
        }

        # n_step transitions except the first one and the last obs
        self.replay_buffer.add(storage_data, ignore_size=1)

    def _sample_from_replay_buffer(self) -> tuple[np.ndarray,
                                                  tuple[np.ndarray | list[np.ndarray], ...],
                                                  tuple[np.ndarray | list[np.ndarray], ...]]:
        """
        Sample from replay buffer

        Returns:
            pointers: [batch, ]
            (
                bn_indexes (np.int32): [batch, b + n]
                bnx_obses_list (np): list([batch, b + n + 1, *obs_shapes_i], ...)
                bn_option_indexes (np.int8): [batch, b + n]
                bn_actions (np): [batch, b + n, action_size]
                bn_rewards (np): [batch, b + n]
                bn_dones (np): [batch, b + n]
                bn_mu_probs (np): [batch, b + n, action_size]
                bnx_pre_seq_hidden_states (np): [batch, b + n + 1, *seq_hidden_state_shape],
                bnx_pre_low_seq_hidden_states (np): [batch, b + n + 1, *low_seq_hidden_state_shape],
                priority_is (np): [batch, 1]
            )
        """
        sampled = self.replay_buffer.sample(prev_n=self.burn_in_step,
                                            post_n=self.n_step)
        if sampled is None:
            return None

        """
        trans:
            index (np.int32): [batch, bn + 1]
            padding_mask (bool): [batch, bn + 1]
            obs_i: [batch, bn + 1, *obs_shapes_i]
            option_index (np.int8): [batch, bn + 1]
            option_changed_index (np.int32): [batch, bn + 1]
            action: [batch, bn + 1,action_size]
            reward: [batch, bn + 1]
            done (bool): [batch, bn + 1]
            mu_prob: [batch, bn + 1, action_size]
            pre_seq_hidden_state: [batch, bn + 1, *seq_hidden_state_shape]
            pre_low_seq_hidden_state: [batch, bn + 1, *low_seq_hidden_state_shape]
        """
        pointers, batch, priority_is = sampled

        def set_padding(t, mask):
            t['index'][mask] = -1
            t['padding_mask'][mask] = True
            for n in self.obs_names:
                t[f'obs_{n}'][mask] = 0.
            t['option_index'][mask] = -1
            t['option_changed_index'][mask] = 0
            t['action'][mask] = self._padding_action
            t['reward'][mask] = 0.
            t['done'][mask] = True
            t['mu_prob'][mask] = 1.
            t['pre_seq_hidden_state'][mask] = 0.
            t['pre_low_seq_hidden_state'][mask] = 0.

        trans_index = batch['index'][:, self.burn_in_step]

        # Padding next n_step data
        for i in range(1, self.n_step + 1):
            t_trans_index = batch['index'][:, self.burn_in_step + i]

            mask = (t_trans_index - trans_index) != i
            set_padding({k: v[:, self.burn_in_step + i] for k, v in batch.items()}, mask)

        # Padding previous burn_in_step data
        for i in range(self.burn_in_step):
            t_trans_index = batch['index'][:, self.burn_in_step - i - 1]

            mask = (trans_index - t_trans_index) != i + 1
            set_padding({k: v[:, self.burn_in_step - i - 1] for k, v in batch.items()}, mask)

        """
        bnx_indexes (np.int32): [batch, bn + 1]
        bnx_padding_masks (bool): [batch, bn + 1]
        bnx_obses_list: list([batch, bn + 1, *obs_shapes_i], ...)
        bnx_option_indexes (np.int8): [batch, bn + 1]
        bnx_actions: [batch, bn + 1, action_size]
        bnx_rewards: [batch, bn + 1]
        bnx_dones (bool): [batch, bn + 1]
        bnx_mu_probs: [batch, bn + 1, action_size]
        bnx_pre_seq_hidden_states: [batch, bn + 1, *seq_hidden_state_shape]
        bnx_pre_low_seq_hidden_states: [batch, bn + 1, *low_seq_hidden_state_shape]
        """
        bnx_indexes = batch['index']
        bnx_padding_masks = batch['padding_mask']
        bnx_obses_list = [batch[f'obs_{name}'] for name in self.obs_names]
        bnx_option_indexes = batch['option_index']
        bnx_actions = batch['action']
        bnx_rewards = batch['reward']
        bnx_dones = batch['done']
        bnx_mu_probs = batch['mu_prob']
        bnx_pre_seq_hidden_states = batch['pre_seq_hidden_state']
        bnx_pre_low_seq_hidden_states = batch['pre_low_seq_hidden_state']

        bn_indexes = bnx_indexes[:, :-1]
        bn_padding_masks = bnx_padding_masks[:, :-1]
        bn_option_indexes = bnx_option_indexes[:, :-1]
        bn_actions = bnx_actions[:, :-1, ...]
        bn_rewards = bnx_rewards[:, :-1]
        bn_dones = bnx_dones[:, :-1]
        bn_mu_probs = bnx_mu_probs[:, :-1]

        key_batch = None
        if self.use_dilation:
            # TODO: multiple threads risks
            key_trans = {k: np.zeros((v.shape[0],
                                      self.key_max_length + 1,
                                      *v.shape[2:]), dtype=v.dtype)
                         for k, v in batch.items()}
            # From the state that need to be trained
            for k, v in batch.items():
                key_trans[k][:, -1] = v[:, self.burn_in_step]

            tmp_pointers = pointers  # From the state that need to be trained

            for i in range(self.key_max_length, 0, -1):  # All keys are the first keys in episodes
                tmp_tran_index = key_trans['index'][:, i]  # The current key tran index in an episode
                tmp_tran_padding_mask = key_trans['padding_mask'][:, i]  # The current key tran padding mask in an episode
                tmp_pre_tran = self.replay_buffer.get_storage_data(tmp_pointers - 1)  # The previous tran of the current key tran
                # The previous option changed key index in an episode
                tmp_option_changed_index = tmp_pre_tran['option_changed_index']
                delta = tmp_tran_index - tmp_option_changed_index

                # The current key tran is the first key in an episode OR is padding already
                padding_mask = np.logical_or(tmp_tran_index == 0, tmp_tran_padding_mask)
                # The previous tran is not actually the previous tran of the current key tran
                padding_mask = np.logical_or(padding_mask, tmp_tran_index - tmp_pre_tran['index'] != 1)

                if np.all(padding_mask):  # Early stop
                    for k, v in key_trans.items():
                        key_trans[k] = v[:, min(i, self.key_max_length - 1):]
                    break

                delta[padding_mask] = 0
                tmp_pointers = (tmp_pointers - delta).astype(pointers.dtype)
                tmp_tran = self.replay_buffer.get_storage_data(tmp_pointers)
                set_padding(tmp_tran, padding_mask)
                for k, v in tmp_tran.items():
                    key_trans[k][:, i - 1] = v

            # Remove the last element, which is the state that need to be trained
            for k, v in key_trans.items():
                key_trans[k] = v[:, :-1]

            key_batch = (
                key_trans['index'],
                key_trans['padding_mask'],
                [key_trans[f'obs_{name}'] for name in self.obs_names],
                key_trans['option_index'],
                key_trans['pre_seq_hidden_state']
            )

        return (pointers,
                (bn_indexes,
                 bn_padding_masks,
                 bnx_obses_list,
                 bn_option_indexes,
                 bn_actions,
                 bn_rewards,
                 bn_dones,
                 bn_mu_probs,
                 bnx_pre_seq_hidden_states,
                 bnx_pre_low_seq_hidden_states,
                 priority_is if self.use_replay_buffer and self.use_priority else None),
                key_batch)

    def _sample_thread(self):
        self._batch_obtained_event.set()

        while not self._closed:
            if self.use_replay_buffer:
                with self._profiler(f'thread_{threading.get_ident()}.sample_from_replay_buffer', repeat=10) as profiler:
                    train_data = self._sample_from_replay_buffer()
                    if train_data is None:
                        profiler.ignore()
                        self._batch = None
                        self._batch_available_event.set()
                        time.sleep(1)
                        continue

                pointers, batch, key_batch = train_data
                batch_list = [batch]
                key_batch_list = [key_batch]
                self._pointers = pointers
            else:
                batch_list, key_batch_list = self.batch_buffer.get_batch()
                if len(batch_list) == 0:
                    self._batch = None
                    self._batch_available_event.set()
                    time.sleep(1)
                    continue

                batch_list = [(*batch, None) for batch in batch_list]  # None is priority_is

            for batch, key_batch in zip(batch_list, key_batch_list):
                self._batch_obtained_event.wait()
                self._batch_obtained_event.clear()

                (bn_indexes,
                 bn_padding_masks,
                 bnx_obses_list,
                 bn_option_indexes,
                 bn_actions,
                 bn_rewards,
                 bn_dones,
                 bn_mu_probs,
                 bnx_pre_seq_hidden_states,
                 bnx_pre_low_seq_hidden_states,
                 priority_is) = batch
                """
                bn_indexes (np.int32): [batch, b + n]
                bn_padding_masks (bool): [batch, b + n]
                bnx_obses_list: list([batch, b + n + 1, *obs_shapes_i], ...)
                bn_option_indexes (np.int8): [batch, b + n]
                bn_actions: [batch, b + n, action_size]
                bn_rewards: [batch, b + n]
                bn_dones (bool): [batch, b + n]
                bn_mu_probs: [batch, b + n, action_size]
                bnx_pre_seq_hidden_states: [batch, b + n + 1, *seq_hidden_state_shape]
                bnx_pre_low_seq_hidden_states: [batch, b + n + 1, *low_seq_hidden_state_shape]
                priority_is: [batch, 1]
                """
                assert bn_indexes.dtype == np.int32
                assert bn_option_indexes.dtype == np.int8

                with self._profiler(f'thread_{threading.get_ident()}.to_gpu', repeat=10):
                    bn_indexes = torch.from_numpy(bn_indexes).to(self.device)
                    bn_padding_masks = torch.from_numpy(bn_padding_masks).to(self.device)
                    bnx_obses_list = [torch.from_numpy(t).to(self.device) for t in bnx_obses_list]
                    for i, bnx_obses in enumerate(bnx_obses_list):
                        # obs is image. It is much faster to convert uint8 to float32 in GPU
                        if bnx_obses.dtype == torch.uint8:
                            bnx_obses_list[i] = bnx_obses.type(torch.float32) / 255.
                    bn_option_indexes = torch.from_numpy(bn_option_indexes).type(torch.int64).to(self.device)
                    bn_actions = torch.from_numpy(bn_actions).to(self.device)
                    bn_rewards = torch.from_numpy(bn_rewards).to(self.device)
                    bn_dones = torch.from_numpy(bn_dones).to(self.device)
                    bn_mu_probs = torch.from_numpy(bn_mu_probs).to(self.device)
                    bnx_pre_seq_hidden_states = torch.from_numpy(bnx_pre_seq_hidden_states).to(self.device)
                    bnx_pre_low_seq_hidden_states = torch.from_numpy(bnx_pre_low_seq_hidden_states).to(self.device)
                    if self.use_replay_buffer and self.use_priority:
                        priority_is = torch.from_numpy(priority_is).to(self.device)

                    if key_batch is not None:
                        (key_indexes,
                         key_padding_masks,
                         key_obses_list,
                         key_option_indexes,
                         key_pre_seq_hidden_states) = key_batch

                        key_indexes = torch.from_numpy(key_indexes).to(self.device)
                        key_padding_masks = torch.from_numpy(key_padding_masks).to(self.device)
                        key_obses_list = [torch.from_numpy(t).to(self.device) for t in key_obses_list]
                        for i, key_obses in enumerate(key_obses_list):
                            # obs is image
                            if key_obses.dtype == torch.uint8:
                                key_obses_list[i] = key_obses.type(torch.float32) / 255.
                        key_option_indexes = torch.from_numpy(key_option_indexes).to(self.device)
                        key_pre_seq_hidden_states = torch.from_numpy(key_pre_seq_hidden_states).to(self.device)

                        key_batch = (key_indexes,
                                     key_padding_masks,
                                     key_obses_list,
                                     key_option_indexes,
                                     key_pre_seq_hidden_states)

                self._batch = (bn_indexes,
                               bn_padding_masks,
                               bnx_obses_list,
                               bn_option_indexes,
                               bn_actions,
                               bn_rewards,
                               bn_dones,
                               bn_mu_probs,
                               bnx_pre_seq_hidden_states,
                               bnx_pre_low_seq_hidden_states,
                               priority_is)

                self._key_batch = key_batch

                self._batch_available_event.set()

    @unified_elapsed_timer('train a step', 10)
    def train(self) -> int:
        step = self.get_global_step()

        if self._batch is None:
            self._profiler('train a step').ignore()
            self._batch_obtained_event.set()
            return step

        with self._profiler('waiting_batch_available', repeat=10):
            self._batch_available_event.wait()
            self._batch_available_event.clear()

        (bn_indexes,
         bn_padding_masks,
         bnx_obses_list,
         bn_option_indexes,
         bn_actions,
         bn_rewards,
         bn_dones,
         bn_mu_probs,
         bnx_pre_seq_hidden_states,
         bnx_pre_low_seq_hidden_states,
         priority_is) = self._batch

        """
        bn_indexes (np.int32): [batch, b + n]
        bn_padding_masks (bool): [batch, b + n]
        bnx_obses_list: list([batch, b + n + 1, *obs_shapes_i], ...)
        bn_option_indexes (np.int8): [batch, b + n]
        bn_actions: [batch, b + n, action_size]
        bn_rewards: [batch, b + n]
        bn_dones (bool): [batch, b + n]
        bn_mu_probs: [batch, b + n, action_size]
        bnx_pre_seq_hidden_states: [batch, b + n + 1, *seq_hidden_state_shape]
        bnx_pre_low_seq_hidden_states: [batch, b + n + 1, *low_seq_hidden_state_shape]
        priority_is: [batch, 1]
        """
        pointers = self._pointers  # Could be None if NOT use_replay_buffer
        key_batch = self._key_batch

        self._batch_obtained_event.set()

        with self._profiler('train', repeat=10):
            (obnx_low_target_obses_list,
             obnx_low_target_states,
             next_n_vs_over_options) = self._train(
                bn_indexes=bn_indexes,
                bn_padding_masks=bn_padding_masks,
                bnx_obses_list=bnx_obses_list,
                bn_option_indexes=bn_option_indexes,
                bn_actions=bn_actions,
                bn_rewards=bn_rewards,
                bn_dones=bn_dones,
                bn_mu_probs=bn_mu_probs,
                bnx_pre_seq_hidden_states=bnx_pre_seq_hidden_states,
                bnx_pre_low_seq_hidden_states=bnx_pre_low_seq_hidden_states,
                priority_is=priority_is if self.use_replay_buffer and self.use_priority else None,

                key_batch=key_batch)

        if step % self.save_model_per_step == 0:
            self.save_model()

        if self.use_replay_buffer:
            bn_obses_list = [bnx_obses[:, :-1, ...] for bnx_obses in bnx_obses_list]
            bn_pre_actions = gen_pre_n_actions(bn_actions)  # [batch, b + n, action_size]
            bn_pre_seq_hidden_states = bnx_pre_seq_hidden_states[:, :-1, ...]  # [batch, b + n, *seq_hidden_state_shape]
            bn_pre_low_seq_hidden_states = bnx_pre_low_seq_hidden_states[:, :-1, ...]  # [batch, b + n, *seq_hidden_state_shape]

            with self._profiler('get_l_states_with_seq_hidden_states', repeat=10):
                bn_states, next_bn_seq_hidden_states = self.get_l_states_with_seq_hidden_states(
                    l_indexes=bn_indexes,
                    l_padding_masks=bn_padding_masks,
                    l_obses_list=bn_obses_list,
                    l_pre_actions=bn_pre_actions,
                    l_pre_seq_hidden_states=bn_pre_seq_hidden_states,

                    key_batch=key_batch)

            bn_low_obses_list = self.get_l_low_obses_list(l_obses_list=bn_obses_list,
                                                          l_states=bn_states)

            with self._profiler('get_l_low_states_with_seq_hidden_states', repeat=10):
                (obn_low_states,
                 next_obn_low_seq_hidden_states) = self.get_l_low_states_with_seq_hidden_states(
                    l_indexes=bn_indexes[:, self.option_burn_in_from:],
                    l_padding_masks=bn_padding_masks[:, self.option_burn_in_from:],
                    l_low_obses_list=[bn_low_obses[:, self.option_burn_in_from:] for bn_low_obses in bn_low_obses_list],
                    l_option_indexes=bn_option_indexes[:, self.option_burn_in_from:],
                    l_pre_actions=bn_pre_actions[:, self.option_burn_in_from:],
                    l_pre_low_seq_hidden_states=bn_pre_low_seq_hidden_states[:, self.option_burn_in_from:]
                )  # TODO: SHOULD NOT option_burn_in_from

            if self.use_n_step_is or (self.d_action_sizes and not self.discrete_dqn_like):
                with self._profiler('get_l_probs', repeat=10):
                    obn_pi_probs_tensor = self.get_l_probs(
                        l_low_obses_list=[bn_low_obses[:, self.option_burn_in_from:] for bn_low_obses in bn_low_obses_list],
                        l_low_states=obn_low_states,
                        l_option_indexes=bn_option_indexes[:, self.option_burn_in_from:],
                        l_actions=bn_actions[:, self.option_burn_in_from:])

            # Update td_error
            if self.use_priority:
                with self._profiler('get_td_error', repeat=10):
                    td_error = self._get_td_error(
                        next_n_vs_over_options=next_n_vs_over_options,

                        bn_padding_masks=bn_padding_masks,
                        bn_states=bn_states,
                        bn_option_indexes=bn_option_indexes,
                        obn_low_obses_list=[bn_low_obses[:, self.option_burn_in_from:] for bn_low_obses in bn_low_obses_list],
                        obnx_low_target_obses_list=obnx_low_target_obses_list,
                        low_state=obn_low_states[:, self.option_burn_in_step],
                        obnx_low_target_states=obnx_low_target_states,
                        bn_actions=bn_actions,
                        bn_rewards=bn_rewards,
                        bn_dones=bn_dones,
                        bn_mu_probs=bn_mu_probs
                    ).detach().cpu().numpy()
                self.replay_buffer.update(pointers, td_error)

            bn_padding_masks = bn_padding_masks.detach().cpu().numpy()
            padding_mask = bn_padding_masks.reshape(-1)
            low_padding_mask = bn_padding_masks[:, self.option_burn_in_from:].reshape(-1)

            # Update seq_hidden_states
            if self.seq_hidden_state_shape[-1] != 0:
                pointers_list = [pointers + 1 + i for i in range(-self.burn_in_step, self.n_step)]
                tmp_pointers = np.stack(pointers_list, axis=1).reshape(-1)

                next_bn_seq_hidden_states = next_bn_seq_hidden_states.detach().cpu().numpy()
                seq_hidden_state = next_bn_seq_hidden_states.reshape(-1, *next_bn_seq_hidden_states.shape[2:])
                self.replay_buffer.update_transitions(tmp_pointers[~padding_mask], 'pre_seq_hidden_state', seq_hidden_state[~padding_mask])

            if self.low_seq_hidden_state_shape[-1] != 0:
                pointers_list = [pointers + 1 + self.option_burn_in_from + i for i in range(-self.option_burn_in_step, self.n_step)]
                tmp_pointers = np.stack(pointers_list, axis=1).reshape(-1)

                next_obn_low_seq_hidden_states = next_obn_low_seq_hidden_states.detach().cpu().numpy()
                low_seq_hidden_state = next_obn_low_seq_hidden_states.reshape(-1, *next_obn_low_seq_hidden_states.shape[2:])
                self.replay_buffer.update_transitions(tmp_pointers[~low_padding_mask], 'pre_low_seq_hidden_state', low_seq_hidden_state[~low_padding_mask])

            # Update n_mu_probs
            if self.use_n_step_is or (self.d_action_sizes and not self.discrete_dqn_like):
                pointers_list = [pointers + self.option_burn_in_from + i for i in range(-self.option_burn_in_step, self.n_step)]
                tmp_pointers = np.stack(pointers_list, axis=1).reshape(-1)

                pi_probs = obn_pi_probs_tensor.detach().cpu().numpy()
                pi_prob = pi_probs.reshape(-1, *pi_probs.shape[2:])
                self.replay_buffer.update_transitions(tmp_pointers[~low_padding_mask], 'mu_prob', pi_prob[~low_padding_mask])

        step = self.increase_global_step()
        for option in self.option_list:
            option.increase_global_step()

        return step

from collections import defaultdict
from itertools import chain
from typing import List, Optional

import numpy as np
import torch
from torch import distributions, nn, optim

from algorithm.oc.option_base import OptionBase

from .. import sac_base
from ..nn_models import *
from ..sac_base import SAC_Base
from ..utils import *
from .oc_batch_buffer import BatchBuffer

sac_base.BatchBuffer = BatchBuffer


class OptionSelectorBase(SAC_Base):
    def __init__(self,
                 num_options: int,
                 use_dilation: bool,
                 option_burn_in_step: int,
                 option_nn_config: dict,

                 obs_names: List[str],
                 obs_shapes: List[Tuple],
                 d_action_sizes: List[int],
                 c_action_size: int,
                 model_abs_dir: Optional[Path],
                 device: Optional[str] = None,
                 ma_name: Optional[str] = None,
                 summary_path: Optional[str] = 'log',
                 train_mode: bool = True,
                 last_ckpt: Optional[str] = None,

                 nn_config: Optional[dict] = None,

                 nn=None,
                 seed: Optional[float] = None,
                 write_summary_per_step: float = 1e3,
                 save_model_per_step: float = 1e5,

                 use_replay_buffer: bool = True,
                 use_priority: bool = True,

                 ensemble_q_num: int = 2,
                 ensemble_q_sample: int = 2,

                 burn_in_step: int = 0,
                 n_step: int = 1,
                 seq_encoder: Optional[SEQ_ENCODER] = None,

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
                 use_n_step_is: bool = True,
                 siamese: Optional[SIAMESE] = None,
                 siamese_use_q: bool = False,
                 siamese_use_adaptive: bool = False,
                 use_prediction: bool = False,
                 transition_kl: float = 0.8,
                 use_extra_data: bool = True,
                 curiosity: Optional[CURIOSITY] = None,
                 curiosity_strength: float = 1.,
                 use_rnd: bool = False,
                 rnd_n_sample: int = 10,
                 use_normalization: bool = False,
                 action_noise: Optional[List[float]] = None,

                 replay_config: Optional[dict] = None):

        self.num_options = num_options
        self.use_dilation = use_dilation
        self.option_burn_in_step = option_burn_in_step
        self.option_nn_config = option_nn_config

        super().__init__(obs_names,
                         obs_shapes,
                         d_action_sizes,
                         c_action_size,
                         model_abs_dir,
                         device, ma_name,
                         summary_path,
                         train_mode,
                         last_ckpt,
                         nn_config,
                         nn,
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

    def _build_model(self, nn, nn_config: Optional[dict], init_log_alpha: float, learning_rate: float) -> None:
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

        d_action_list = [np.eye(d_action_size, dtype=np.float32)[0]
                         for d_action_size in self.d_action_sizes]
        self._padding_action = np.concatenate(d_action_list + [np.zeros(self.c_action_size, dtype=np.float32)], axis=-1)

        def adam_optimizer(params):
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

            class ModelRep(nn.ModelRep):
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
            ModelRep = nn.ModelRep

        """ REPRESENTATION """
        if self.seq_encoder == SEQ_ENCODER.RNN:
            self.model_rep: ModelBaseRNNRep = ModelRep(self.obs_shapes,
                                                       self.d_action_sizes, self.c_action_size,
                                                       False, self.train_mode,
                                                       self.model_abs_dir,
                                                       use_dilation=self.use_dilation,
                                                       **nn_config['rep']).to(self.device)
            self.model_target_rep: ModelBaseRNNRep = ModelRep(self.obs_shapes,
                                                              self.d_action_sizes, self.c_action_size,
                                                              True, self.train_mode,
                                                              self.model_abs_dir,
                                                              use_dilation=self.use_dilation,
                                                              **nn_config['rep']).to(self.device)
            # Get represented state and seq_hidden_state_shape
            test_obs_list = [torch.rand(self.batch_size, 1, *obs_shape, device=self.device) for obs_shape in self.obs_shapes]
            test_pre_action = torch.rand(self.batch_size, 1, self.d_action_summed_size + self.c_action_size, device=self.device)
            test_state, test_rnn_state = self.model_rep(test_obs_list,
                                                        test_pre_action)
            state_size, self.seq_hidden_state_shape = test_state.shape[-1], test_rnn_state.shape[1:]

        elif self.seq_encoder == SEQ_ENCODER.ATTN:
            self.model_rep: ModelBaseAttentionRep = ModelRep(self.obs_shapes,
                                                             self.d_action_sizes, self.c_action_size,
                                                             False, self.train_mode,
                                                             self.model_abs_dir,
                                                             use_dilation=self.use_dilation,
                                                             **nn_config['rep']).to(self.device)
            self.model_target_rep: ModelBaseAttentionRep = ModelRep(self.obs_shapes,
                                                                    self.d_action_sizes, self.c_action_size,
                                                                    True, self.train_mode,
                                                                    self.model_abs_dir,
                                                                    use_dilation=self.use_dilation,
                                                                    **nn_config['rep']).to(self.device)
            # Get represented state and seq_hidden_state_shape
            test_index = torch.zeros((self.batch_size, 1), dtype=torch.int32, device=self.device)
            test_obs_list = [torch.rand(self.batch_size, 1, *obs_shape, device=self.device) for obs_shape in self.obs_shapes]
            test_pre_action = torch.rand(self.batch_size, 1, self.d_action_summed_size + self.c_action_size, device=self.device)
            test_state, test_attn_state, _ = self.model_rep(test_index,
                                                            test_obs_list,
                                                            test_pre_action)
            state_size, self.seq_hidden_state_shape = test_state.shape[-1], test_attn_state.shape[2:]

        else:
            self.model_rep: ModelBaseSimpleRep = ModelRep(self.obs_shapes,
                                                          False, self.train_mode,
                                                          self.model_abs_dir,
                                                          **nn_config['rep']).to(self.device)
            self.model_target_rep: ModelBaseSimpleRep = ModelRep(self.obs_shapes,
                                                                 True, self.train_mode,
                                                                 self.model_abs_dir,
                                                                 **nn_config['rep']).to(self.device)
            # Get represented state dimension
            test_obs_list = [torch.rand(self.batch_size, *obs_shape, device=self.device) for obs_shape in self.obs_shapes]
            test_state = self.model_rep(test_obs_list)
            state_size = test_state.shape[-1]

        for param in self.model_target_rep.parameters():
            param.requires_grad = False

        self.state_size = state_size
        self._logger.info(f'State size: {state_size}')

        if len(list(self.model_rep.parameters())) > 0:
            self.optimizer_rep = adam_optimizer(self.model_rep.parameters())
        else:
            self.optimizer_rep = None

        self.model_v_over_options_list = [nn.ModelVOverOption(state_size, self.num_options).to(self.device)
                                          for _ in range(self.ensemble_q_num)]
        self.model_target_v_over_options_list = [nn.ModelVOverOption(state_size, self.num_options).to(self.device)
                                                 for _ in range(self.ensemble_q_num)]
        for model_target_v_over_options in self.model_target_v_over_options_list:
            for param in model_target_v_over_options.parameters():
                param.requires_grad = False
        self.optimizer_v_list = [adam_optimizer(self.model_v_over_options_list[i].parameters()) for i in range(self.ensemble_q_num)]

        """
        Initialize each option
        """
        if self.option_burn_in_step == -1:
            self.option_burn_in_step = self.burn_in_step

        self.option_burn_in_from = self.burn_in_step - self.option_burn_in_step

        option_kwargs = self._kwargs
        del option_kwargs['self']
        option_kwargs['obs_names'] = ['state', *self.obs_names]
        option_kwargs['obs_shapes'] = [(self.state_size, ), *self.obs_shapes]
        # seq_encoder of option can only be RNN or VANILLA
        option_kwargs['seq_encoder'] = SEQ_ENCODER.RNN if self.seq_encoder == SEQ_ENCODER.ATTN else self.seq_encoder
        option_kwargs['burn_in_step'] = self.option_burn_in_step
        option_kwargs['use_replay_buffer'] = False

        if self.option_nn_config is None:
            self.option_nn_config = {}
        self.option_nn_config = defaultdict(dict, self.option_nn_config)
        option_kwargs['nn_config'] = self.option_nn_config

        _tmp_ModelRep, option_kwargs['nn'].ModelRep = option_kwargs['nn'].ModelRep, option_kwargs['nn'].ModelOptionRep

        self.option_list: List[OptionBase] = [None] * self.num_options
        for i in range(self.num_options):
            if self.model_abs_dir is not None:
                option_kwargs['model_abs_dir'] = self.model_abs_dir / f'option_{i}'
                option_kwargs['model_abs_dir'].mkdir(parents=True, exist_ok=True)
            if self.ma_name is not None:
                option_kwargs['ma_name'] = f'{self.ma_name}_option_{i}'
            else:
                option_kwargs['ma_name'] = f'option_{i}'

            self.option_list[i] = OptionBase(**option_kwargs)

        option_kwargs['nn'].ModelRep = _tmp_ModelRep

        if self.seq_encoder is not None:
            self.low_seq_hidden_state_shape = self.option_list[0].seq_hidden_state_shape

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

        if len(list(self.model_rep.parameters())) > 0:
            ckpt_dict['model_rep'] = self.model_rep
            ckpt_dict['model_target_rep'] = self.model_target_rep
            ckpt_dict['optimizer_rep'] = self.optimizer_rep

        for i in range(self.ensemble_q_num):
            ckpt_dict[f'model_v_over_options_{i}'] = self.model_v_over_options_list[i]
            ckpt_dict[f'model_target_v_over_options_{i}'] = self.model_target_v_over_options_list[i]
            ckpt_dict[f'optimizer_v_over_options_{i}'] = self.optimizer_v_list[i]

    def _init_or_restore(self, last_ckpt: int) -> None:
        """
        Initialize network weights from scratch or restore from model_abs_dir
        """
        self.ckpt_dir = None
        if self.model_abs_dir:
            self.ckpt_dir = ckpt_dir = self.model_abs_dir.joinpath('model')

            ckpts = []
            if ckpt_dir.exists():
                for ckpt_path in ckpt_dir.glob('*.pth'):
                    ckpts.append(int(ckpt_path.stem))
                ckpts.sort()
            else:
                ckpt_dir.mkdir()

            if ckpts:
                if last_ckpt is None:
                    last_ckpt = ckpts[-1]
                else:
                    assert last_ckpt in ckpts

                ckpt_restore_path = ckpt_dir.joinpath(f'{last_ckpt}.pth')
                ckpt_restore = torch.load(ckpt_restore_path, map_location=self.device)
                self.global_step = self.global_step.to('cpu')
                for name, model in self.ckpt_dict.items():
                    if name not in ckpt_restore:
                        self._logger.warning(f'{name} not in {last_ckpt}.pth')
                        continue

                    if isinstance(model, torch.Tensor):
                        model.data = ckpt_restore[name]
                    else:
                        try:
                            model.load_state_dict(ckpt_restore[name])
                        except RuntimeError as e:
                            self._logger.error(e)
                        if isinstance(model, nn.Module):
                            if self.train_mode:
                                model.train()
                            else:
                                model.eval()

                self._logger.info(f'Restored from {ckpt_restore_path}')

                if self.train_mode and self.use_replay_buffer:
                    self.replay_buffer.load(ckpt_dir, last_ckpt)

                    self._logger.info(f'Replay buffer restored')
            else:
                self._logger.info('Initializing from scratch')
                self._update_target_variables()

    def set_train_mode(self, train_mode=True):
        self.train_mode = train_mode
        for option in self.option_list:
            option.set_train_mode(train_mode)

    def save_model(self, save_replay_buffer=False) -> None:
        super().save_model(save_replay_buffer)

        for option in self.option_list:
            option.save_model(save_replay_buffer)

    def get_initial_option_index(self, batch_size: int) -> np.ndarray:
        return np.full([batch_size, ], -1, dtype=np.int8)

    def get_initial_low_seq_hidden_state(self, batch_size, get_numpy=True) -> Union[np.ndarray, torch.Tensor]:
        return self.option_list[0].get_initial_seq_hidden_state(batch_size,
                                                                get_numpy)

    @torch.no_grad()
    def _update_target_variables(self, tau=1.) -> None:
        """
        Soft (momentum) update target networks (default hard)
        """

        target = self.model_target_rep.parameters()
        source = self.model_rep.parameters()

        for i in range(self.ensemble_q_num):
            target = chain(target, self.model_target_v_over_options_list[i].parameters())
            source = chain(source, self.model_v_over_options_list[i].parameters())

        for target_param, param in zip(target, source):
            target_param.data.copy_(
                target_param.data * (1. - tau) + param.data * tau
            )

        for option in self.option_list:
            option._update_target_variables(tau)

    def _choose_option_index(self,
                             option_index: torch.Tensor,
                             state: torch.Tensor,
                             low_obs_list: List[torch.Tensor],
                             pre_action: Optional[torch.Tensor] = None,
                             low_seq_state: Optional[torch.Tensor] = None,
                             disable_sample: bool = False) -> Tuple[torch.Tensor,
                                                                    torch.Tensor]:
        """
        Args:
            option_index (torch.int64): [batch, ]
            state: [batch, state_size]
            low_obs_list: list([batch, *low_obs_shapes_i], ...)
            pre_action: [batch, action_size]
            low_seq_state: [batch, low_seq_state_size]
            disable_sample (bool)

        Returns:
            new_option_index (torch.int64): [batch, ]
            new_option_mask (torch.bool): [batch, ]
        """
        batch = option_index.shape[0]

        pre_option_index = option_index.clone()

        v_over_options = self.model_v_over_options_list[0](state)  # [batch, num_options]
        new_option_index = v_over_options.argmax(dim=-1)  # [batch, ]

        none_option_mask = option_index == -1
        option_index[none_option_mask] = new_option_index[none_option_mask]

        termination = torch.zeros((batch, 1), device=self.device)  # [batch, 1]
        for i, option in enumerate(self.option_list):
            mask = (option_index == i)
            if not torch.any(mask):
                continue

            o_low_f_state, _ = option.get_l_states(l_indexes=None,  # option is not ATTN
                                                   l_padding_masks=None,
                                                   l_obses_list=[low_obs[mask].unsqueeze(1) for low_obs in low_obs_list],
                                                   l_pre_actions=pre_action[mask].unsqueeze(1) if pre_action is not None else None,
                                                   f_seq_hidden_states=low_seq_state[mask].unsqueeze(1) if low_seq_state is not None else None)
            o_low_state = o_low_f_state.squeeze(1)

            termination[mask] = option.model_termination(o_low_state)

        termination = termination.squeeze(-1)
        termination_mask = termination > .5
        option_index[termination_mask] = new_option_index[termination_mask]

        if self.train_mode:
            random_mask = torch.rand_like(option_index, dtype=torch.float32) < 0.2  # TODO HYPERPARAMETER
            dist = distributions.Categorical(logits=torch.ones((batch, self.num_options),
                                                               device=self.device))
            random_option_index = dist.sample()  # [batch, ]

            option_index[random_mask] = random_option_index[random_mask]

        return option_index, pre_option_index != option_index

    def _choose_option_action(self,
                              obs_list: List[torch.Tensor],
                              state: torch.Tensor,
                              pre_option_index: torch.Tensor,

                              disable_sample: bool = False,
                              force_rnd_if_available: bool = False) -> Tuple[torch.Tensor,
                                                                             torch.Tensor,
                                                                             torch.Tensor]:
        """
        Args:
            obs_list: list([batch, 1, *obs_shapes_i], ...)
            state: [batch, 1, d_action_summed_size + c_action_size]
            pre_option_index (torch.int64): [batch, ]

        Returns:
            new_option_index (torch.int64): [batch, ]
            action: [batch, action_size]
            prob: [batch, action_size]
        """

        low_obs_list = self.get_l_low_obses_list(obs_list, state)
        option_index, _ = self._choose_option_index(option_index=pre_option_index,
                                                    state=state,
                                                    low_obs_list=low_obs_list,
                                                    pre_action=None,
                                                    low_seq_state=None,
                                                    disable_sample=disable_sample)

        batch = state.shape[0]
        action = torch.zeros((batch,
                              self.d_action_summed_size + self.c_action_size),
                             device=self.device)
        prob = torch.ones((batch,
                           self.d_action_summed_size + self.c_action_size),
                          device=self.device)

        for i, option in enumerate(self.option_list):
            mask = (option_index == i)
            if not torch.any(mask):
                continue

            o_low_obs_list = [low_obs[mask] for low_obs in low_obs_list]

            o_action, o_prob = option.choose_action(o_low_obs_list,
                                                    disable_sample=disable_sample,
                                                    force_rnd_if_available=force_rnd_if_available)
            action[mask] = o_action
            prob[mask] = o_prob

        return option_index, action, prob

    def _choose_option_rnn_action(self,
                                  obs_list: List[torch.Tensor],
                                  state: torch.Tensor,
                                  pre_option_index: torch.Tensor,
                                  pre_action: torch.Tensor,
                                  low_rnn_state: torch.Tensor,

                                  disable_sample: bool = False,
                                  force_rnd_if_available: bool = False) -> Tuple[torch.Tensor,
                                                                                 torch.Tensor,
                                                                                 torch.Tensor,
                                                                                 torch.Tensor]:
        """
        Args:
            obs_list: list([batch, *obs_shapes_i], ...)
            state: [batch, d_action_summed_size + c_action_size]
            pre_option_index (torch.int64): [batch, ]
            pre_action: [batch, action_size]
            low_rnn_state: [batch, *low_seq_hidden_state_shape]

        Returns:
            new_option_index (torch.int64): [batch, ]
            action: [batch, action_size]
            prob: [batch, action_size]
            next_low_rnn_state: [batch, *low_seq_hidden_state_shape]
        """

        low_obs_list = self.get_l_low_obses_list(obs_list, state)

        option_index, new_option_index_mask = self._choose_option_index(option_index=pre_option_index,
                                                                        state=state,
                                                                        low_obs_list=low_obs_list,
                                                                        pre_action=pre_action,
                                                                        low_seq_state=low_rnn_state,
                                                                        disable_sample=disable_sample)

        batch = state.shape[0]
        initial_low_seq_hidden_state = self.get_initial_low_seq_hidden_state(batch, get_numpy=False)

        low_rnn_state[new_option_index_mask] = initial_low_seq_hidden_state[new_option_index_mask]

        action = torch.zeros(batch, self.d_action_summed_size + self.c_action_size, device=self.device)
        prob = torch.ones((batch,
                           self.d_action_summed_size + self.c_action_size),
                          device=self.device)
        next_low_rnn_state = torch.zeros_like(low_rnn_state)

        for i, option in enumerate(self.option_list):
            mask = (option_index == i)
            if not torch.any(mask):
                continue

            o_low_obs_list = [low_obs[mask] for low_obs in low_obs_list]

            o_action, o_prob, o_next_low_rnn_state = option.choose_rnn_action(o_low_obs_list,
                                                                              pre_action[mask],
                                                                              low_rnn_state[mask],
                                                                              disable_sample=disable_sample,
                                                                              force_rnd_if_available=force_rnd_if_available)
            action[mask] = o_action
            prob[mask] = o_prob
            next_low_rnn_state[mask] = o_next_low_rnn_state

        return option_index, action, prob, next_low_rnn_state

    @torch.no_grad()
    def choose_action(self,
                      obs_list: List[np.ndarray],
                      pre_option_index: np.ndarray,

                      disable_sample: bool = False,
                      force_rnd_if_available: bool = False) -> Tuple[np.ndarray,
                                                                     np.ndarray,
                                                                     np.ndarray]:
        """
        Args:
            obs_list (np): list([batch, *obs_shapes_i], ...)
            pre_option_index (np.int8): [batch, ]

        Returns:
            option_index (np.int8): [batch, ]
            action (np): [batch, d_action_summed_size + c_action_size]
            prob (np): [batch, action_size]
        """
        obs_list = [torch.from_numpy(obs).to(self.device) for obs in obs_list]
        pre_option_index = torch.from_numpy(pre_option_index).type(torch.int64).to(self.device)

        state = self.model_rep(obs_list)

        option_index, action, prob = self._choose_option_action(obs_list,
                                                                state,
                                                                pre_option_index,
                                                                disable_sample=disable_sample,
                                                                force_rnd_if_available=force_rnd_if_available)

        return (option_index.detach().cpu().numpy().astype(np.int8),
                action.detach().cpu().numpy(),
                prob.detach().cpu().numpy())

    @torch.no_grad()
    def choose_rnn_action(self,
                          obs_list: List[np.ndarray],
                          pre_option_index: np.ndarray,
                          pre_action: np.ndarray,
                          rnn_state: np.ndarray,
                          low_rnn_state: np.ndarray,

                          disable_sample: bool = False,
                          force_rnd_if_available: bool = False) -> Tuple[np.ndarray,
                                                                         np.ndarray,
                                                                         np.ndarray,
                                                                         np.ndarray,
                                                                         np.ndarray]:
        """
        Args:
            obs_list (np): list([batch, *obs_shapes_i], ...)
            pre_option_index (np.int8): [batch, ]
            pre_action (np): [batch, d_action_summed_size + c_action_size]
            rnn_state (np): [batch, *seq_hidden_state_shape]
            low_rnn_state (np): [batch, *low_seq_hidden_state_shape]

        Returns:
            option_index (np.int8): [batch, ]
            action (np): [batch, d_action_summed_size + c_action_size]
            prob (np): [batch, action_size]
            next_rnn_state (np): [batch, *seq_hidden_state_shape]
            next_low_rnn_state (np): [batch, *low_seq_hidden_state_shape]
        """
        obs_list = [torch.from_numpy(obs).to(self.device) for obs in obs_list]
        pre_option_index = torch.from_numpy(pre_option_index).type(torch.int64).to(self.device)
        pre_action = torch.from_numpy(pre_action).to(self.device)
        rnn_state = torch.from_numpy(rnn_state).to(self.device)
        low_rnn_state = torch.from_numpy(low_rnn_state).to(self.device)

        obs_list = [obs.unsqueeze(1) for obs in obs_list]
        pre_action = pre_action.unsqueeze(1)

        state, next_rnn_state = self.model_rep(obs_list, pre_action, rnn_state)
        # state: [batch, 1, state_size]

        obs_list = [obs.squeeze(1) for obs in obs_list]
        state = state.squeeze(1)
        pre_action = pre_action.squeeze(1)

        (option_index,
         action,
         prob,
         next_low_rnn_state) = self._choose_option_rnn_action(obs_list,
                                                              state,
                                                              pre_option_index,
                                                              pre_action,
                                                              low_rnn_state,
                                                              disable_sample=disable_sample,
                                                              force_rnd_if_available=force_rnd_if_available)

        return (option_index.detach().cpu().numpy().astype(np.int8),
                action.detach().cpu().numpy(),
                prob.detach().cpu().numpy(),
                next_rnn_state.detach().cpu().numpy(),
                next_low_rnn_state.detach().cpu().numpy())

    @torch.no_grad()
    def choose_attn_action(self,
                           ep_indexes: np.ndarray,
                           ep_padding_masks: np.ndarray,
                           ep_obses_list: List[np.ndarray],
                           ep_pre_actions: np.ndarray,
                           ep_attn_states: np.ndarray,

                           pre_option_index: np.ndarray,
                           low_rnn_state: np.ndarray,

                           disable_sample: bool = False,
                           force_rnd_if_available: bool = False) -> Tuple[np.ndarray,
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
            ep_attn_states (np): [batch, episode_len, *seq_hidden_state_shape]

            pre_option_index (np.int8): [batch, ]
            low_rnn_state (np): [batch, *low_seq_hidden_state_shape]

        Returns:
            option_index (np.int8): [batch, ]
            action (np): [batch, d_action_summed_size + c_action_size]
            prob (np): [batch, action_size]
            next_attn_state (np): [batch, *attn_state_shape]
            next_low_rnn_state (np): [batch, *low_rnn_state_shape]
        """
        ep_indexes = torch.from_numpy(ep_indexes).to(self.device)
        ep_padding_masks = torch.from_numpy(ep_padding_masks).to(self.device)
        ep_obses_list = [torch.from_numpy(obs).to(self.device) for obs in ep_obses_list]
        ep_pre_actions = torch.from_numpy(ep_pre_actions).to(self.device)
        ep_attn_states = torch.from_numpy(ep_attn_states).to(self.device)

        pre_option_index = torch.from_numpy(pre_option_index).type(torch.int64).to(self.device)
        low_rnn_state = torch.from_numpy(low_rnn_state).to(self.device)

        state, next_attn_state, _ = self.model_rep(ep_indexes,
                                                   ep_obses_list, ep_pre_actions,
                                                   query_length=1,
                                                   #    hidden_state=ep_attn_states,
                                                   #    is_prev_hidden_state=False,
                                                   padding_mask=ep_padding_masks)
        # state: [batch, 1, state_size]
        # next_attn_state: [batch, 1, *attn_state_shape]

        obs_list = [ep_obses[:, -1, ...] for ep_obses in ep_obses_list]
        state = state.squeeze(1)
        pre_action = ep_pre_actions[:, -1, ...]
        next_attn_state = next_attn_state.squeeze(1)

        (option_index,
         action,
         prob,
         next_low_rnn_state) = self._choose_option_rnn_action(obs_list,
                                                              state,
                                                              pre_option_index,
                                                              pre_action,
                                                              low_rnn_state,
                                                              disable_sample=disable_sample,
                                                              force_rnd_if_available=force_rnd_if_available)

        return (option_index.detach().cpu().numpy().astype(np.int8),
                action.detach().cpu().numpy(),
                prob.detach().cpu().numpy(),
                next_attn_state.detach().cpu().numpy(),
                next_low_rnn_state.detach().cpu().numpy())

    @torch.no_grad()
    def choose_dilated_attn_action(self,
                                   key_indexes: np.ndarray,
                                   key_padding_masks: np.ndarray,
                                   key_obses_list: List[np.ndarray],
                                   key_attn_states: np.ndarray,

                                   pre_option_index: np.ndarray,
                                   pre_action: np.ndarray,
                                   low_rnn_state: np.ndarray,

                                   disable_sample: bool = False,
                                   force_rnd_if_available: bool = False) -> Tuple[np.ndarray,
                                                                                  np.ndarray,
                                                                                  np.ndarray,
                                                                                  np.ndarray,
                                                                                  np.ndarray]:
        """
        Args:
            key_indexes (np.int32): [batch, key_len]
            key_padding_masks (np.bool): [batch, key_len]
            key_obses_list (np): list([batch, key_len, *obs_shapes_i], ...)
            key_attn_states (np): [batch, key_len, *seq_hidden_state_shape]

            pre_option_index (np.int8): [batch, ]
            pre_action (np): [batch, action_size]
            low_rnn_state (np): [batch, *low_seq_hidden_state_shape]

        Returns:
            option_index (np.int8): [batch]
            action (np): [batch, d_action_summed_size + c_action_size]
            prob (np): [batch, action_size]
            next_attn_state (np): [batch, *attn_state_shape]
            next_low_rnn_state (np): [batch, *low_rnn_state_shape]
        """

        key_indexes = torch.from_numpy(key_indexes).to(self.device)
        key_padding_masks = torch.from_numpy(key_padding_masks).to(self.device)
        key_obses_list = [torch.from_numpy(obs).to(self.device) for obs in key_obses_list]
        key_attn_states = torch.from_numpy(key_attn_states).to(self.device)

        pre_option_index = torch.from_numpy(pre_option_index).type(torch.int64).to(self.device)
        pre_action = torch.from_numpy(pre_action).to(self.device)
        low_rnn_state = torch.from_numpy(low_rnn_state).to(self.device)

        # self._logger.debug(f'choose {key_indexes.shape}')

        state, next_attn_state, _ = self.model_rep(key_indexes,
                                                   key_obses_list,
                                                   pre_action=None,
                                                   query_length=1,
                                                   hidden_state=key_attn_states,
                                                   is_prev_hidden_state=False,
                                                   padding_mask=key_padding_masks)
        # state: [batch, 1, state_size]
        # next_attn_state: [batch, 1, *attn_state_shape]

        obs_list = [key_obses[:, -1, ...] for key_obses in key_obses_list]
        state = state.squeeze(1)
        next_attn_state = next_attn_state.squeeze(1)

        (option_index,
         action,
         prob,
         next_low_rnn_state) = self._choose_option_rnn_action(obs_list,
                                                              state,
                                                              pre_option_index,
                                                              pre_action,
                                                              low_rnn_state,
                                                              disable_sample=disable_sample,
                                                              force_rnd_if_available=force_rnd_if_available)

        return (option_index.detach().cpu().numpy().astype(np.int8),
                action.detach().cpu().numpy(),
                prob.detach().cpu().numpy(),
                next_attn_state.detach().cpu().numpy(),
                next_low_rnn_state.detach().cpu().numpy())

    def get_m_data(self,
                   bn_indexes: torch.Tensor,
                   bn_padding_masks: torch.Tensor,
                   bn_obses_list: List[torch.Tensor],
                   bn_actions: torch.Tensor,
                   next_obs_list: torch.Tensor) -> Tuple[torch.Tensor,
                                                         torch.Tensor,
                                                         List[torch.Tensor],
                                                         torch.Tensor]:
        m_indexes = torch.concat([bn_indexes, bn_indexes[:, -1:] + 1], dim=1)
        m_padding_masks = torch.concat([bn_padding_masks,
                                        torch.zeros_like(bn_padding_masks[:, -1:], dtype=torch.bool)], dim=1)
        m_obses_list = [torch.cat([n_obses, next_obs.unsqueeze(1)], dim=1)
                        for n_obses, next_obs in zip(bn_obses_list, next_obs_list)]
        m_pre_actions = gen_pre_n_actions(bn_actions, keep_last_action=True)

        return m_indexes, m_padding_masks, m_obses_list, m_pre_actions

    def get_l_states(self,
                     l_indexes: torch.Tensor,
                     l_padding_masks: torch.Tensor,
                     l_obses_list: List[torch.Tensor],
                     l_pre_actions: Optional[torch.Tensor] = None,
                     f_seq_hidden_states: Optional[torch.Tensor] = None,

                     key_batch: Optional[Tuple[torch.Tensor,
                                               torch.Tensor,
                                               List[torch.Tensor],
                                               torch.Tensor]] = None,

                     is_target=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            l_indexes: [batch, l]
            l_padding_masks: [batch, l]
            l_obses_list: list([batch, l, *obs_shapes_i], ...)
            l_pre_actions: [batch, l, action_size]
            f_seq_hidden_states: [batch, 1, *seq_hidden_state_shape]

            key_batch:
                key_indexes: [batch, key_len]
                key_padding_masks: [batch, key_len]
                key_obses_list: list([batch, key_len, *obs_shapes_i], ...)
                key_option_indexes: [batch, key_len]
                key_seq_hidden_states: [batch, key_len, *seq_hidden_state_shape]

        Returns:
            l_states: [batch, l, state_size]
            next_f_rnn_states (optional): [batch, 1, rnn_state_size]
            l_attn_states (optional): [batch, l, attn_state_size]
        """
        model_rep = self.model_target_rep if is_target else self.model_rep

        if self.seq_encoder == SEQ_ENCODER.RNN and self.use_dilation:
            (key_indexes,
             key_padding_masks,
             key_obses_list,
             key_option_indexes,
             key_seq_hidden_states) = key_batch

            _, next_key_rnn_state = model_rep(key_obses_list,
                                              None,
                                              key_seq_hidden_states[:, 0],
                                              padding_mask=key_padding_masks)

            batch, l, *_ = l_indexes.shape

            l_states = None

            for t in range(l):
                f_states, next_rnn_state = model_rep([l_obses[:, t:t + 1, ...] for l_obses in l_obses_list],
                                                     l_pre_actions[:, t:t + 1, ...] if l_pre_actions is not None else None,
                                                     next_key_rnn_state,
                                                     padding_mask=l_padding_masks[:, t:t + 1, ...])
                if l_states is None:
                    l_states = torch.zeros((batch, l, *f_states.shape[2:]), device=self.device)
                l_states[:, t:t + 1] = f_states

            next_f_rnn_states = next_rnn_state.unsqueeze(dim=1)

            return l_states, next_f_rnn_states

        elif self.seq_encoder == SEQ_ENCODER.ATTN and self.use_dilation:
            query_length = l_indexes.shape[1]

            (key_indexes,
             key_padding_masks,
             key_obses_list,
             key_option_indexes,
             key_seq_hidden_states) = key_batch

            l_indexes = torch.concat([key_indexes, l_indexes], dim=1)
            l_padding_masks = torch.concat([key_padding_masks, l_padding_masks], dim=1)
            l_obses_list = [torch.concat([key_obses, l_obses], dim=1)
                            for key_obses, l_obses in zip(key_obses_list, l_obses_list)]

            l_states, l_attn_states, _ = model_rep(l_indexes,
                                                   l_obses_list,
                                                   pre_action=None,
                                                   query_length=query_length,
                                                   hidden_state=key_seq_hidden_states,
                                                   is_prev_hidden_state=True,
                                                   query_only_attend_to_reset_key=True,
                                                   padding_mask=l_padding_masks)

            return l_states, l_attn_states

        else:
            return super().get_l_states(
                l_indexes=l_indexes,
                l_padding_masks=l_padding_masks,
                l_obses_list=l_obses_list,
                l_pre_actions=l_pre_actions,
                f_seq_hidden_states=f_seq_hidden_states,
                is_target=is_target
            )

    def get_l_states_with_seq_hidden_states(
        self,
        l_indexes: torch.Tensor,
        l_padding_masks: torch.Tensor,
        l_obses_list: List[torch.Tensor],
        l_pre_actions: Optional[torch.Tensor] = None,

        key_batch: Optional[Tuple[torch.Tensor,
                                  torch.Tensor,
                                  List[torch.Tensor],
                                  torch.Tensor]] = None,

        f_seq_hidden_states: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor,
               Optional[torch.Tensor]]:
        """
        Args:
            l_indexes (torch.int32): [batch, l]
            l_padding_masks (torch.bool): [batch, l]
            l_obses_list: list([batch, l, *obs_shapes_i], ...)
            l_pre_actions: [batch, l, action_size]
            f_seq_hidden_states: [batch, 1, *seq_hidden_state_shape]

            key_batch:
                key_indexes: [batch, key_len]
                key_padding_masks: [batch, key_len]
                key_obses_list: list([batch, key_len, *obs_shapes_i], ...)
                key_option_indexes: [batch, key_len]
                key_seq_hidden_states: [batch, key_len, *seq_hidden_state_shape]

        Returns:
            l_states: [batch, l, state_size]
            l_seq_hidden_state: [batch, l, *seq_hidden_state_shape]
        """
        if self.seq_encoder == SEQ_ENCODER.RNN and self.use_dilation:
            (key_indexes,
             key_padding_masks,
             key_obses_list,
             key_option_indexes,
             key_seq_hidden_states) = key_batch

            key_padding_masks = key_padding_masks[:, :-1]
            key_obses_list = [key_obses[:, :-1] for key_obses in key_obses_list]
            _, next_key_rnn_state = self.model_rep(key_obses_list,
                                                   None,
                                                   key_seq_hidden_states[:, 0],
                                                   padding_mask=key_padding_masks)

            batch, l, *_ = l_indexes.shape

            l_states = None
            next_l_rnn_states = torch.zeros((batch, l, *f_seq_hidden_states.shape[2:]), device=self.device)

            for t in range(l):
                f_states, rnn_state = self.model_rep([l_obses[:, t:t + 1, ...] for l_obses in l_obses_list],
                                                     l_pre_actions[:, t:t + 1, ...] if l_pre_actions is not None else None,
                                                     next_key_rnn_state,
                                                     padding_mask=l_padding_masks[:, t:t + 1, ...])

                if l_states is None:
                    l_states = torch.zeros((batch, l, *f_states.shape[2:]), device=self.device)
                l_states[:, t:t + 1] = f_states

                next_l_rnn_states[:, t] = rnn_state

            return l_states, next_l_rnn_states

        elif self.seq_encoder == SEQ_ENCODER.ATTN and self.use_dilation:
            return self.get_l_states(l_indexes=l_indexes,
                                     l_padding_masks=l_padding_masks,
                                     l_obses_list=l_obses_list,
                                     l_pre_actions=None,
                                     f_seq_hidden_states=f_seq_hidden_states,
                                     key_batch=key_batch,
                                     is_target=False)
        else:
            return super().get_l_states_with_seq_hidden_states(
                l_indexes=l_indexes,
                l_padding_masks=l_padding_masks,
                l_obses_list=l_obses_list,
                l_pre_actions=l_pre_actions,
                f_seq_hidden_states=f_seq_hidden_states
            )

    def get_l_low_obses_list(self,
                             l_obses_list: List[torch.Tensor],
                             l_states: torch.Tensor) -> List[torch.Tensor]:
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
                         l_low_obses_list: List[torch.Tensor],
                         l_option_indexes: torch.Tensor,
                         l_pre_actions: torch.Tensor,
                         f_low_seq_hidden_states: torch.Tensor = None,
                         is_target=False) -> Tuple[torch.Tensor,
                                                   torch.Tensor]:
        """
        Args:
            l_indexes (torch.int32): [batch, l]
            l_padding_masks (torch.bool): [batch, l]
            l_low_obses_list: list([batch, l, *low_obs_shapes_i], ...)
            l_option_indexes (torch.int64): [batch, l]
            l_pre_actions: [batch, l, action_size]
            f_low_seq_hidden_states: [batch, 1, *low_seq_hidden_state_shape]
            is_target: bool

        Returns:
            l_low_states: [batch, l, low_state_size]
            next_l_low_rnn_states (optional): [batch, 1, *low_seq_hidden_state_shape]
            next_l_low_attn_states (optional): [batch, l, *low_seq_hidden_state_shape]
        """

        l_low_states = None
        next_f_or_l_low_seq_hidden_states = None

        for i, option in enumerate(self.option_list):
            mask = (l_option_indexes == i)
            if not torch.any(mask):
                continue

            o_l_states, o_next_f_or_l_low_seq_hidden_states = option.get_l_states(
                l_indexes=l_indexes,
                l_padding_masks=l_padding_masks,
                l_obses_list=l_low_obses_list,
                l_pre_actions=l_pre_actions,
                f_seq_hidden_states=f_low_seq_hidden_states,
                is_target=is_target
            )

            if l_low_states is None:
                l_low_states = o_l_states
            else:
                l_low_states[mask] = o_l_states[mask]

            if next_f_or_l_low_seq_hidden_states is None:
                next_f_or_l_low_seq_hidden_states = o_next_f_or_l_low_seq_hidden_states
            else:
                if next_f_or_l_low_seq_hidden_states.shape[1] == 1:
                    next_f_or_l_low_seq_hidden_states[mask[:, -1:]] = o_next_f_or_l_low_seq_hidden_states[mask[:, -1:]]
                else:
                    next_f_or_l_low_seq_hidden_states[mask] = o_next_f_or_l_low_seq_hidden_states[mask]

        return l_low_states, next_f_or_l_low_seq_hidden_states

    def get_l_low_states_with_seq_hidden_states(self,
                                                l_indexes: torch.Tensor,
                                                l_padding_masks: torch.Tensor,
                                                l_low_obses_list: List[torch.Tensor],
                                                l_option_indexes: torch.Tensor,
                                                l_pre_actions: torch.Tensor,
                                                f_low_seq_hidden_states: torch.Tensor = None) -> Tuple[torch.Tensor,
                                                                                                       torch.Tensor]:
        """
        Args:
            l_indexes (torch.int32): [batch, l]
            l_padding_masks (torch.bool): [batch, l]
            l_low_obses_list: list([batch, l, *low_obs_shapes_i], ...)
            l_option_indexes (torch.int64): [batch, l]
            l_pre_actions: [batch, l, action_size]
            f_low_seq_hidden_states: [batch, 1, *low_seq_hidden_state_shape]

        Returns:
            l_low_states: [batch, l, low_state_size]
            next_l_low_seq_hidden_states: [batch, l, *low_seq_hidden_state_shape]
        """
        l_low_states = None
        next_l_low_seq_hidden_states = None

        for i, option in enumerate(self.option_list):
            mask = (l_option_indexes == i)
            if not torch.any(mask):
                continue

            o_l_states, o_next_l_low_seq_hidden_states = option.get_l_states_with_seq_hidden_states(
                l_indexes=l_indexes,
                l_padding_masks=l_padding_masks,
                l_obses_list=l_low_obses_list,
                l_pre_actions=l_pre_actions,
                f_seq_hidden_states=f_low_seq_hidden_states
            )

            if l_low_states is None:
                l_low_states = o_l_states
            else:
                l_low_states[mask] = o_l_states[mask]

            if next_l_low_seq_hidden_states is None:
                next_l_low_seq_hidden_states = o_next_l_low_seq_hidden_states
            else:
                next_l_low_seq_hidden_states[mask] = o_next_l_low_seq_hidden_states[mask]

        return l_low_states, next_l_low_seq_hidden_states

    @torch.no_grad()
    def get_l_probs(self,
                    l_low_obses_list: List[torch.Tensor],
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

    @torch.no_grad()
    def _get_td_error(self,
                      bn_padding_masks: torch.Tensor,
                      bn_states: torch.Tensor,
                      bn_target_states: torch.Tensor,
                      bn_option_indexes: torch.Tensor,
                      obn_low_obses_list: List[torch.Tensor],
                      obn_low_states: torch.Tensor,
                      obn_low_target_states: torch.Tensor,
                      bn_actions: torch.Tensor,
                      bn_rewards: torch.Tensor,
                      next_target_state: torch.Tensor,
                      next_low_obs_list: List[torch.Tensor],
                      next_low_target_state: torch.Tensor,
                      bn_dones: torch.Tensor,
                      bn_mu_probs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bn_padding_masks (torch.bool): [batch, b + n]
            bn_states: [batch, b + n, state_size]
            bn_target_states: [batch, b + n, state_size]
            bn_option_indexes (torch.int64): [batch, b + n]
            obn_low_obses_list: list([batch, ob + n, *low_obs_shapes_i], ...)
            obn_low_states: [batch, ob + n, low_state_size]
            obn_low_target_states: [batch, ob + n, low_state_size]
            bn_actions: [batch, b + n, action_size]
            bn_rewards: [batch, b + n]
            next_target_state: [batch, state_size]
            next_low_obs_list: list([batch, *low_obs_shapes_i], ...)
            next_low_target_state: [batch, low_state_size]
            bn_dones (torch.bool): [batch, b + n]
            bn_mu_probs: [batch, b + n, action_size]

        Returns:
            The td-error of observations, [batch, 1]
        """

        next_n_states = torch.concat([bn_target_states[:, self.burn_in_step + 1:, ...],
                                      next_target_state.unsqueeze(1)], dim=1)  # [batch, n, state_size]
        n_option_indexes = bn_option_indexes[:, self.burn_in_step:]  # [batch, n]
        option_index = n_option_indexes[:, 0]  # [batch, ]

        next_n_v_over_options_list = [v(next_n_states) for v in self.model_target_v_over_options_list]  # [batch, n, num_options]

        batch = bn_states.shape[0]
        td_error = torch.zeros((batch, 1), device=self.device)

        for i, option in enumerate(self.option_list):
            mask = (option_index == i)
            if not torch.any(mask):
                continue

            o_td_error = option._get_td_error(
                next_n_v_over_options_list=[next_n_v_over_options[mask]
                                            for next_n_v_over_options in next_n_v_over_options_list],

                bn_padding_masks=bn_padding_masks[mask, self.option_burn_in_from:],
                bn_obses_list=[obn_low_obses[mask] for obn_low_obses in obn_low_obses_list],
                bn_states=obn_low_states[mask],
                bn_target_states=obn_low_target_states[mask],
                bn_actions=bn_actions[mask, self.option_burn_in_from:],
                bn_rewards=bn_rewards[mask, self.option_burn_in_from:],
                next_obs_list=[next_low_obs[mask] for next_low_obs in next_low_obs_list],
                next_target_state=next_low_target_state[mask],
                bn_dones=bn_dones[mask, self.option_burn_in_from:],
                bn_mu_probs=bn_mu_probs[mask, self.option_burn_in_from:])

            td_error[mask] = o_td_error

        return td_error

    def _train_rep_q(self,
                     next_n_v_over_options_list: List[torch.Tensor],

                     bn_indexes: torch.Tensor,
                     bn_padding_masks: torch.Tensor,
                     bn_option_indexes: torch.Tensor,
                     obn_low_obses_list: List[torch.Tensor],
                     obn_low_target_obses_list: List[torch.Tensor],
                     bn_actions: torch.Tensor,
                     bn_rewards: torch.Tensor,
                     next_low_obs_list: List[torch.Tensor],
                     next_low_target_obs_list: List[torch.Tensor],
                     bn_dones: torch.Tensor,
                     bn_mu_probs: torch.Tensor,
                     f_low_seq_hidden_states: Optional[torch.Tensor] = None,
                     priority_is: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            next_n_v_over_options_list: list([batch, n, num_options], ...),

            bn_indexes (torch.int32): [batch, b + n],
            bn_padding_masks (torch.bool): [batch, b + n],
            bn_option_indexes (torch.int64): [batch, b + n],
            obn_low_obses_list: list([batch, ob + n, *low_obs_shapes_i], ...)
            obn_low_target_obses_list: list([batch, ob + n, *low_obs_shapes_i], ...)
            bn_actions: [batch, b + n, action_size],
            bn_rewards: [batch, b + n],
            next_low_obs_list: list([batch, *low_obs_shapes_i], ...)
            next_low_target_obs_list: list([batch, *low_obs_shapes_i], ...)
            bn_dones (torch.bool): [batch, b + n]
            bn_mu_probs: [batch, b + n, action_size]
            f_low_seq_hidden_states: [batch, 1, *low_seq_hidden_state_shape]
            priority_is: [batch, 1]

        Returns:
            om_low_states: [batch, ob + n + 1, low_state_size]
            om_low_target_states: [batch, ob + n + 1, low_state_size]
        """

        if self.optimizer_rep:
            self.optimizer_rep.zero_grad()

        batch = bn_indexes.shape[0]

        option_index = bn_option_indexes[:, self.burn_in_step]

        om_low_states = None
        om_low_target_states = None

        for i, option in enumerate(self.option_list):
            mask = (option_index == i)
            if not torch.any(mask):
                continue

            (o_m_indexes,
             o_m_padding_masks,
             o_m_low_obses_list,
             o_m_pre_actions) = option.get_m_data(
                bn_indexes=bn_indexes[mask, self.option_burn_in_from:],
                bn_padding_masks=bn_padding_masks[mask, self.option_burn_in_from:],
                bn_obses_list=[obn_low_obses[mask] for obn_low_obses in obn_low_obses_list],
                bn_actions=bn_actions[mask, self.option_burn_in_from:],
                next_obs_list=[next_low_obs[mask] for next_low_obs in next_low_obs_list]
            )

            (_, _, o_m_low_target_obses_list, _) = option.get_m_data(
                bn_indexes=bn_indexes[mask, self.option_burn_in_from:],
                bn_padding_masks=bn_padding_masks[mask, self.option_burn_in_from:],
                bn_obses_list=[obn_low_target_obses[mask] for obn_low_target_obses in obn_low_target_obses_list],
                bn_actions=bn_actions[mask, self.option_burn_in_from:],
                next_obs_list=[next_low_target_obs[mask] for next_low_target_obs in next_low_target_obs_list]
            )

            if self.seq_encoder is not None:
                o_f_low_seq_hidden_states = f_low_seq_hidden_states[mask]

            o_m_low_states, _ = option.get_l_states(
                l_indexes=o_m_indexes,
                l_padding_masks=o_m_padding_masks,
                l_obses_list=o_m_low_obses_list,
                l_pre_actions=o_m_pre_actions,
                f_seq_hidden_states=o_f_low_seq_hidden_states if self.seq_encoder is not None else None,
                is_target=False
            )

            with torch.no_grad():
                o_m_low_target_states, _ = option.get_l_states(
                    l_indexes=o_m_indexes,
                    l_padding_masks=o_m_padding_masks,
                    l_obses_list=o_m_low_target_obses_list,
                    l_pre_actions=o_m_pre_actions,
                    f_seq_hidden_states=o_f_low_seq_hidden_states if self.seq_encoder is not None else None,
                    is_target=True
                )

            if om_low_states is None:
                om_low_states = torch.zeros((batch, *o_m_low_states.shape[1:]), device=self.device)
                om_low_target_states = torch.zeros((batch, *o_m_low_target_states.shape[1:]), device=self.device)
            om_low_states[mask] = o_m_low_states
            om_low_target_states[mask] = o_m_low_target_states

            option.train_rep_q(next_n_v_over_options_list=[next_n_v_over_options[mask] for next_n_v_over_options in next_n_v_over_options_list],

                               bn_indexes=bn_indexes[mask, self.option_burn_in_from:],
                               bn_padding_masks=bn_padding_masks[mask, self.option_burn_in_from:],
                               bn_obses_list=[o_m_low_obses[:, :-1, ...]
                                              for o_m_low_obses in o_m_low_obses_list],
                               bn_target_obses_list=[o_m_low_target_obses[:, :-1, ...]
                                                     for o_m_low_target_obses in o_m_low_target_obses_list],
                               bn_states=o_m_low_states[:, :-1, ...],
                               bn_target_states=o_m_low_target_states[:, :-1, ...],
                               bn_actions=bn_actions[mask, self.option_burn_in_from:],
                               bn_rewards=bn_rewards[mask, self.option_burn_in_from:],
                               next_obs_list=[o_m_low_obses[:, -1, ...]
                                              for o_m_low_obses in o_m_low_obses_list],
                               next_target_obs_list=[o_m_low_target_obses[:, -1, ...]
                                                     for o_m_low_target_obses in o_m_low_target_obses_list],
                               next_state=o_m_low_states[:, -1, ...],
                               next_target_state=o_m_low_target_states[:, -1, ...],
                               bn_dones=bn_dones[mask, self.option_burn_in_from:],
                               bn_mu_probs=bn_mu_probs[mask, self.option_burn_in_from:],
                               priority_is=priority_is[mask] if self.use_replay_buffer and self.use_priority else None)

        if self.optimizer_rep:
            self.optimizer_rep.step()

        return om_low_states, om_low_target_states

    def _train_v_terminations(self,
                              bn_padding_masks: torch.Tensor,
                              m_states: torch.Tensor,
                              bn_option_indexes: torch.Tensor,
                              om_low_obses_list: List[torch.Tensor],
                              om_low_states: torch.Tensor,
                              bn_dones: torch.Tensor,
                              priority_is: torch.Tensor) -> Tuple[torch.Tensor,
                                                                  torch.Tensor]:
        """
        Args:
            bn_padding_masks (torch.bool): [batch, b + n]
            m_states: [batch, b + n + 1, state_size]
            bn_option_indexes (torch.int64): [batch, b + n]
            om_low_obses_list: list([batch, ob + n + 1, *low_obs_shapes_i], ...)
            om_low_states: [batch, ob + n + 1, state_size]
            bn_dones (torch.bool): [batch, b + n]
            priority_is: [batch, 1]

        Returns:
            loss_v: torch.float32
        """

        batch = m_states.shape[0]
        n_padding_masks = bn_padding_masks[:, self.burn_in_step:, ...]
        state = m_states[:, self.burn_in_step, ...]
        next_n_states = m_states[:, self.burn_in_step + 1:, ...]
        option_index = bn_option_indexes[:, self.burn_in_step]
        n_option_indexes = bn_option_indexes[:, self.burn_in_step:]
        low_obs_list = [om_low_obses[:, self.option_burn_in_step, ...]
                        for om_low_obses in om_low_obses_list]
        low_state = om_low_states[:, self.option_burn_in_step, ...]
        next_n_low_states = om_low_states[:, self.option_burn_in_step + 1:]
        n_dones = bn_dones[:, self.burn_in_step:]

        batch_tensor = torch.arange(batch, device=self.device)
        last_solid_index = get_last_false_indexes(n_padding_masks, dim=1)  # [batch, ]

        next_state = next_n_states[batch_tensor, last_solid_index]  # [batch, state_size]
        last_option_index = n_option_indexes[batch_tensor, last_solid_index]
        next_low_state = next_n_low_states[batch_tensor, last_solid_index]
        done = n_dones[batch_tensor, last_solid_index]

        y_for_v = torch.zeros((batch, 1), device=self.device)

        for i, option in enumerate(self.option_list):
            mask = (option_index == i)
            if not torch.any(mask):
                continue

            o_state = low_state[mask]
            o_low_obs_list = [low_obs[mask] for low_obs in low_obs_list]

            y_for_v[mask] = option.get_v(obs_list=o_low_obs_list,
                                         state=o_state)

        loss_none_mse = nn.MSELoss(reduction='none')
        for i, model_v_over_options in enumerate(self.model_v_over_options_list):
            v_over_options = model_v_over_options(state)  # [batch, num_options]
            v = v_over_options.gather(1, option_index.unsqueeze(-1))  # [batch, 1]

            loss_v = loss_none_mse(v, y_for_v)  # [batch, 1]
            if priority_is is not None:
                loss_v = loss_v * priority_is  # [batch, 1]

            loss_v = torch.mean(loss_v)

            optimizer = self.optimizer_v_list[i]

            optimizer.zero_grad()
            loss_v.backward()
            optimizer.step()

        with torch.no_grad():
            next_v_over_options = self.model_target_v_over_options_list[0](next_state)  # [batch, num_options]
            next_v = next_v_over_options.gather(1, last_option_index.unsqueeze(-1))  # [batch, 1]
        for i, option in enumerate(self.option_list):
            mask = (option_index == i)
            if not torch.any(mask):
                continue

            option.train_termination(next_state=next_low_state[mask],
                                     next_v_over_options=next_v_over_options[mask],
                                     next_v=next_v[mask],
                                     done=done[mask])

        return loss_v

    def _train(self,
               bn_indexes: torch.Tensor,
               bn_padding_masks: torch.Tensor,
               bn_obses_list: List[torch.Tensor],
               bn_option_indexes: torch.Tensor,
               bn_actions: torch.Tensor,
               bn_rewards: torch.Tensor,
               next_obs_list: List[torch.Tensor],
               bn_dones: torch.Tensor,
               bn_mu_probs: torch.Tensor,
               f_seq_hidden_states: torch.Tensor = None,
               f_low_seq_hidden_states: torch.Tensor = None,
               priority_is: torch.Tensor = None,
               key_batch: Optional[Tuple[Union[torch.Tensor, List[torch.Tensor]], ...]] = None) -> Tuple[torch.Tensor,
                                                                                                         List[torch.Tensor],
                                                                                                         torch.Tensor]:
        """
        Args:
            bn_indexes (torch.int32): [batch, b + n]
            bn_padding_masks (torch.bool): [batch, b + n]
            bn_obses_list: list([batch, b + n, *obs_shapes_i], ...)
            bn_option_indexes (torch.int64): [batch, b + n]
            bn_actions: [batch, b + n, action_size]
            bn_rewards: [batch, b + n]
            next_obs_list: list([batch, *obs_shapes_i], ...)
            bn_dones (torch.bool): [batch, b + n]
            bn_mu_probs: [batch, b + n, action_size]
            f_seq_hidden_states: [batch, 1, *seq_hidden_state_shape]
            f_low_seq_hidden_states: [batch, 1, *low_seq_hidden_state_shape]
                (start from self.option_burn_in_from)
            priority_is: [batch, 1]

            key_batch:
                key_indexes: [batch, key_len]
                key_padding_masks: [batch, key_len]
                key_obses_list: list([batch, key_len, *obs_shapes_i], ...)
                key_option_indexes: [batch, key_len]
                key_seq_hidden_states: [batch, key_len, *seq_hidden_state_shape]

        Returns:
            m_target_states: [batch, b + n + 1, state_size]
            om_low_target_obses_list: list([batch, ob + n + 1, *low_obs_shapes_i], ...)
            om_low_target_states: [batch, ob + n + 1, low_state_size]
        """

        if self.global_step % self.update_target_per_step == 0:
            self._update_target_variables(tau=self.tau)

        (m_indexes,
         m_padding_masks,
         m_obses_list,
         m_pre_actions) = self.get_m_data(bn_indexes=bn_indexes,
                                          bn_padding_masks=bn_padding_masks,
                                          bn_obses_list=bn_obses_list,
                                          bn_actions=bn_actions,
                                          next_obs_list=next_obs_list)

        m_states, _ = self.get_l_states(l_indexes=m_indexes,
                                        l_padding_masks=m_padding_masks,
                                        l_obses_list=m_obses_list,
                                        l_pre_actions=m_pre_actions,
                                        f_seq_hidden_states=f_seq_hidden_states,

                                        key_batch=key_batch,

                                        is_target=False)
        # [batch, b + n + 1, state_size]

        m_target_states, _ = self.get_l_states(l_indexes=m_indexes,
                                               l_padding_masks=m_padding_masks,
                                               l_obses_list=m_obses_list,
                                               l_pre_actions=m_pre_actions,
                                               f_seq_hidden_states=f_seq_hidden_states,

                                               key_batch=key_batch,

                                               is_target=True)
        # [batch, b + n + 1, state_size]

        next_n_states = m_target_states[:, self.burn_in_step + 1:, ...]  # [batch, n, state_size]
        n_option_indexes = bn_option_indexes[:, self.burn_in_step:]  # [batch, n]
        option_index = n_option_indexes[:, 0]  # [batch, ]

        next_n_v_over_options_list = [v(next_n_states) for v in self.model_target_v_over_options_list]  # [batch, n, num_options]

        om_low_obses_list = self.get_l_low_obses_list(l_obses_list=[m_obses[:, self.option_burn_in_from:] for m_obses in m_obses_list],
                                                      l_states=m_states[:, self.option_burn_in_from:])
        om_low_target_obses_list = self.get_l_low_obses_list(l_obses_list=[m_obses[:, self.option_burn_in_from:] for m_obses in m_obses_list],
                                                             l_states=m_target_states[:, self.option_burn_in_from:])

        (om_low_states,
         om_low_target_states) = self._train_rep_q(next_n_v_over_options_list=next_n_v_over_options_list,

                                                   bn_indexes=bn_indexes,
                                                   bn_padding_masks=bn_padding_masks,
                                                   bn_option_indexes=bn_option_indexes,
                                                   obn_low_obses_list=[om_low_obses[:, :-1, ...] for om_low_obses in om_low_obses_list],
                                                   obn_low_target_obses_list=[om_low_target_obses[:, :-1, ...] for om_low_target_obses in om_low_target_obses_list],
                                                   bn_actions=bn_actions,
                                                   bn_rewards=bn_rewards,
                                                   next_low_obs_list=[om_low_obses[:, -1, ...] for om_low_obses in om_low_obses_list],
                                                   next_low_target_obs_list=[om_low_target_obses[:, -1, ...] for om_low_target_obses in om_low_target_obses_list],
                                                   bn_dones=bn_dones,
                                                   bn_mu_probs=bn_mu_probs,
                                                   f_low_seq_hidden_states=f_low_seq_hidden_states,
                                                   priority_is=priority_is)
        # om_low_states: [batch, ob + n + 1, low_state_size]
        # om_low_target_states: [batch, ob + n + 1, low_state_size]

        with torch.no_grad():
            m_states, _ = self.get_l_states(l_indexes=m_indexes,
                                            l_padding_masks=m_padding_masks,
                                            l_obses_list=m_obses_list,
                                            l_pre_actions=m_pre_actions,
                                            f_seq_hidden_states=f_seq_hidden_states if self.seq_encoder is not None else None,

                                            key_batch=key_batch,

                                            is_target=False)
            # [batch, b + n + 1, state_size]

        om_low_obses_list = self.get_l_low_obses_list(l_obses_list=[m_obses[:, self.option_burn_in_from:] for m_obses in m_obses_list],
                                                      l_states=m_states[:, self.option_burn_in_from:])
        # list([batch, ob + n + 1, *low_obs_shapes_i], ...)

        for i, option in enumerate(self.option_list):
            mask = (option_index == i)
            if not torch.any(mask):
                continue

            (o_m_indexes,
             o_m_padding_masks,
             o_m_low_obses_list,
             o_m_pre_actions) = option.get_m_data(bn_indexes=bn_indexes[mask, self.option_burn_in_from:],
                                                  bn_padding_masks=bn_padding_masks[mask, self.option_burn_in_from:],
                                                  bn_obses_list=[om_low_obses[mask, :-1, ...] for om_low_obses in om_low_obses_list],
                                                  bn_actions=bn_actions[mask, self.option_burn_in_from:],
                                                  next_obs_list=[om_obses[mask, -1, ...] for om_obses in om_low_obses_list])

            if self.seq_encoder is not None:
                o_f_low_seq_hidden_states = f_low_seq_hidden_states[mask]

            o_m_low_states, _ = option.get_l_states(l_indexes=o_m_indexes,
                                                    l_padding_masks=o_m_padding_masks,
                                                    l_obses_list=o_m_low_obses_list,
                                                    l_pre_actions=o_m_pre_actions,
                                                    f_seq_hidden_states=o_f_low_seq_hidden_states if self.seq_encoder is not None else None,
                                                    is_target=False)
            om_low_states[mask] = o_m_low_states

            option.train_policy_alpha(bn_padding_masks=bn_padding_masks[mask, self.option_burn_in_from:],
                                      bn_obses_list=[o_m_obses[:, :-1, ...]
                                                     for o_m_obses in o_m_low_obses_list],
                                      m_states=o_m_low_states,
                                      bn_actions=bn_actions[mask, self.option_burn_in_from:],
                                      bn_mu_probs=bn_mu_probs[mask, self.option_burn_in_from:])

        loss_v = self._train_v_terminations(bn_padding_masks=bn_padding_masks,
                                            m_states=m_states,
                                            bn_option_indexes=bn_option_indexes,
                                            om_low_obses_list=om_low_obses_list,
                                            om_low_states=om_low_states,
                                            bn_dones=bn_dones,
                                            priority_is=priority_is)

        if self.summary_writer is not None and self.global_step % self.write_summary_per_step == 0:
            self.summary_available = True

            self.summary_writer.add_scalar('loss/v', loss_v, self.global_step)

            self.summary_writer.flush()

        return m_target_states, om_low_target_obses_list, om_low_target_states

    def put_episode(self, **episode_trans: np.ndarray) -> None:
        # Ignore episodes which length is too short
        if episode_trans['l_indexes'].shape[1] < self.n_step:
            return

        episodes = self._padding_next_obs_list(**episode_trans)

        if self.use_replay_buffer:
            self._fill_replay_buffer(*episodes)
        else:
            self.batch_buffer.put_episode(*episodes)

    def _padding_next_obs_list(self,
                               l_indexes: np.ndarray,
                               l_obses_list: List[np.ndarray],
                               l_option_indexes: np.ndarray,
                               l_option_changed_indexes: np.ndarray,
                               l_actions: np.ndarray,
                               l_rewards: np.ndarray,
                               next_obs_list: List[np.ndarray],
                               l_dones: np.ndarray,
                               l_probs: List[np.ndarray],
                               l_seq_hidden_states: Optional[np.ndarray] = None,
                               l_low_seq_hidden_states: Optional[np.ndarray] = None) -> Tuple[np.ndarray, ...]:
        """
        Padding next_obs_list

        Args:
            l_indexes (np.int32): [1, ep_len]
            l_obses_list (np): list([1, ep_len, *obs_shapes_i], ...)
            l_option_indexes (np.int8): [1, ep_len]
            l_option_changed_indexes (np.int32): [1, ep_len]
            l_actions (np): [1, ep_len, action_size]
            l_rewards (np): [1, ep_len]
            next_obs_list (np): list([1, *obs_shapes_i], ...)
            l_dones (bool): [1, ep_len]
            l_probs (np): [1, ep_len, action_size]
            l_seq_hidden_states (np): [1, ep_len, *seq_hidden_state_shape]
            l_low_seq_hidden_states (np): [1, ep_len, *low_seq_hidden_state_shape]

        Returns:
            ep_indexes (np.int32): [1, ep_len + 1]
            ep_padding_masks: (bool): [1, ep_len + 1]
            ep_obses_list (np): list([1, ep_len + 1, *obs_shapes_i], ...)
            ep_option_indexes (np.int8): [1, ep_len + 1]
            ep_option_changed_indexes (np.int32): [1, ep_len + 1]
            ep_actions (np): [1, ep_len + 1, action_size]
            ep_rewards (np): [1, ep_len + 1]
            ep_dones (bool): [1, ep_len + 1]
            ep_probs (np): [1, ep_len + 1, action_size]
            ep_seq_hidden_states (np): [1, ep_len + 1, *seq_hidden_state_shape]
            ep_low_seq_hidden_states (np): [1, ep_len + 1, *low_seq_hidden_state_shape]
        """
        assert l_indexes.dtype == np.int32, l_indexes.dtype
        assert l_option_indexes.dtype == np.int8, l_option_indexes.dtype
        assert l_option_changed_indexes.dtype == np.int32, l_option_changed_indexes.dtype

        (ep_indexes,
         ep_padding_masks,
         ep_obses_list,
         ep_actions,
         ep_rewards,
         ep_dones,
         ep_probs,
         ep_seq_hidden_states) = super()._padding_next_obs_list(l_indexes=l_indexes,
                                                                l_obses_list=l_obses_list,
                                                                l_actions=l_actions,
                                                                l_rewards=l_rewards,
                                                                next_obs_list=next_obs_list,
                                                                l_dones=l_dones,
                                                                l_probs=l_probs,
                                                                l_seq_hidden_states=l_seq_hidden_states)

        ep_option_indexes = np.concatenate([l_option_indexes,
                                            np.full([1, 1], 0, dtype=np.int8)], axis=1)
        ep_option_changed_indexes = np.concatenate([l_option_changed_indexes,
                                                    l_option_changed_indexes[:, -1:] + 1], axis=1)
        if l_low_seq_hidden_states is not None:
            ep_low_seq_hidden_states = np.concatenate([l_low_seq_hidden_states,
                                                       np.zeros([1, 1, *l_low_seq_hidden_states.shape[2:]], dtype=np.float32)], axis=1)

        return (ep_indexes,
                ep_padding_masks,
                ep_obses_list,
                ep_option_indexes,
                ep_option_changed_indexes,
                ep_actions,
                ep_rewards,
                ep_dones,
                ep_probs,
                ep_seq_hidden_states if l_seq_hidden_states is not None else None,
                ep_low_seq_hidden_states if l_low_seq_hidden_states is not None else None)

    def _fill_replay_buffer(self,
                            ep_indexes: np.ndarray,
                            ep_padding_masks: np.ndarray,
                            ep_obses_list: List[np.ndarray],
                            ep_option_indexes: np.ndarray,
                            ep_option_changed_indexes: np.ndarray,
                            ep_actions: np.ndarray,
                            ep_rewards: np.ndarray,
                            ep_dones: np.ndarray,
                            ep_probs: List[np.ndarray],
                            ep_seq_hidden_states: Optional[np.ndarray] = None,
                            ep_low_seq_hidden_states: Optional[np.ndarray] = None) -> None:
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
            ep_seq_hidden_states (np): [1, ep_len, *seq_hidden_state_shape]
            ep_low_seq_hidden_states (np): [1, ep_len, *low_seq_hidden_state_shape]
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

        storage_data = {
            'index': index,
            'padding_mask': padding_mask,
            **{f'obs_{name}': obs for name, obs in zip(self.obs_names, obs_list)},
            'option_index': option_index,
            'option_changed_index': option_changed_index,
            'action': action,
            'reward': reward,
            'done': done,
            'mu_prob': mu_prob
        }

        if ep_seq_hidden_states is not None:
            seq_hidden_state = ep_seq_hidden_states.squeeze(0)
            storage_data['seq_hidden_state'] = seq_hidden_state

        if ep_low_seq_hidden_states is not None:
            low_seq_hidden_state = ep_low_seq_hidden_states.squeeze(0)
            storage_data['low_seq_hidden_state'] = low_seq_hidden_state

        # n_step transitions except the first one and the last obs
        self.replay_buffer.add(storage_data, ignore_size=1)

        if self.seq_encoder == SEQ_ENCODER.ATTN \
                and self.summary_writer is not None \
                and self.summary_available:
            self.summary_available = False
            with torch.no_grad():
                if self.use_dilation:
                    key_option_changed_indexes = np.unique(ep_option_changed_indexes)  # [key_len, ]
                    key_indexes = ep_indexes[:, key_option_changed_indexes]  # [1, key_len]
                    key_obses_list = [l_obses[:, key_option_changed_indexes] for l_obses in ep_obses_list]  # [1, key_len, state_size]
                    *_, attn_weights_list = self.model_rep(torch.from_numpy(key_indexes).to(self.device),
                                                           [torch.from_numpy(o).to(self.device) for o in key_obses_list],
                                                           pre_action=None,
                                                           query_length=key_indexes.shape[1])

                else:
                    pre_l_actions = gen_pre_n_actions(ep_actions)
                    *_, attn_weights_list = self.model_rep(torch.from_numpy(ep_indexes).to(self.device),
                                                           [torch.from_numpy(o).to(self.device) for o in ep_obses_list],
                                                           pre_action=torch.from_numpy(pre_l_actions).to(self.device),
                                                           query_length=ep_indexes.shape[1])

                for i, attn_weight in enumerate(attn_weights_list):
                    image = plot_attn_weight(attn_weight[0].cpu().numpy())
                    self.summary_writer.add_figure(f'attn_weight/{i}', image, self.global_step)

    def _sample_from_replay_buffer(self) -> Tuple[np.ndarray,
                                                  Tuple[Union[np.ndarray, List[np.ndarray]], ...],
                                                  Tuple[Union[np.ndarray, List[np.ndarray]], ...]]:
        """
        Sample from replay buffer

        Returns:
            pointers: [batch, ]
            (
                bn_indexes (np.int32): [batch, b + n]
                bn_obses_list (np): list([batch, b + n, *obs_shapes_i], ...)
                bn_option_indexes (np.int8): [batch, b + n]
                bn_actions (np): [batch, b + n, action_size]
                bn_rewards (np): [batch, b + n]
                next_obs_list (np): list([batch, *obs_shapes_i], ...)
                bn_dones (np): [batch, b + n]
                bn_mu_probs (np): [batch, b + n, action_size]
                bn_seq_hidden_states (np): [batch, b + n, *seq_hidden_state_shape],
                bn_low_seq_hidden_states (np): [batch, b + n, *low_seq_hidden_state_shape],
                priority_is (np): [batch, 1]
            )
        """
        sampled = self.replay_buffer.sample()
        if sampled is None:
            return None

        """
        trans:
            index (np.int32): [batch, ]
            padding_mask (bool): [batch, ]
            obs_i: [batch, *obs_shapes_i]
            option_index (np.int8): [batch, ]
            option_changed_index (np.int32): [batch, ]
            action: [batch, action_size]
            reward: [batch, ]
            done (bool): [batch, ]
            mu_prob: [batch, action_size]
            seq_hidden_state: [batch, *seq_hidden_state_shape]
            low_seq_hidden_state: [batch, *low_seq_hidden_state_shape]
        """
        pointers, trans, priority_is = sampled

        # Get burn_in_step + n_step transitions
        # TODO: could be faster, no need get all data
        batch = {k: np.zeros((v.shape[0],
                              self.burn_in_step + self.n_step + 1,
                              *v.shape[1:]), dtype=v.dtype)
                 for k, v in trans.items()}

        for k, v in trans.items():
            batch[k][:, self.burn_in_step] = v

        def set_padding(t, mask):
            t['index'][mask] = -1
            t['padding_mask'][mask] = True
            for n in self.obs_names:
                t[f'obs_{n}'][mask] = 0.
            t['option_index'][mask] = 0
            t['option_changed_index'][mask] = 0
            t['action'][mask] = self._padding_action
            t['reward'][mask] = 0.
            t['done'][mask] = True
            t['mu_prob'][mask] = 1.
            if 'seq_hidden_state' in t:
                t['seq_hidden_state'][mask] = 0.
            if 'low_seq_hidden_state' in t:
                t['low_seq_hidden_state'][mask] = 0.

        # Get next n_step data
        for i in range(1, self.n_step + 1):
            t_trans = self.replay_buffer.get_storage_data(pointers + i)

            mask = (t_trans['index'] - trans['index']) != i
            set_padding(t_trans, mask)

            for k, v in t_trans.items():
                batch[k][:, self.burn_in_step + i] = v

        # Get previous burn_in_step data
        for i in range(self.burn_in_step):  # TODO: option_burn_in_step is enough in dilated attn
            t_trans = self.replay_buffer.get_storage_data(pointers - i - 1)

            mask = (t_trans['index'] - trans['index']) != i
            set_padding(t_trans, mask)

            for k, v in t_trans.items():
                batch[k][:, self.burn_in_step - i - 1] = v

        """
        m_indexes (np.int32): [batch, N + 1]
        m_padding_masks (bool): [batch, N + 1]
        m_obses_list: list([batch, N + 1, *obs_shapes_i], ...)
        m_option_indexes (np.int8): [batch, N + 1]
        m_actions: [batch, N + 1, action_size]
        m_rewards: [batch, N + 1]
        m_dones (bool): [batch, N + 1]
        m_mu_probs: [batch, N + 1, action_size]
        m_seq_hidden_states: [batch, N + 1, *seq_hidden_state_shape]
        m_low_seq_hidden_states: [batch, N + 1, *low_seq_hidden_state_shape]
        """
        m_indexes = batch['index']
        m_padding_masks = batch['padding_mask']
        m_obses_list = [batch[f'obs_{name}'] for name in self.obs_names]
        m_option_indexes = batch['option_index']
        m_actions = batch['action']
        m_rewards = batch['reward']
        m_dones = batch['done']
        m_mu_probs = batch['mu_prob']

        bn_indexes = m_indexes[:, :-1]
        bn_padding_masks = m_padding_masks[:, :-1]
        bn_obses_list = [m_obses[:, :-1, ...] for m_obses in m_obses_list]
        bn_option_indexes = m_option_indexes[:, :-1]
        bn_actions = m_actions[:, :-1, ...]
        bn_rewards = m_rewards[:, :-1]
        next_obs_list = [m_obses[:, -1, ...] for m_obses in m_obses_list]
        bn_dones = m_dones[:, :-1]
        bn_mu_probs = m_mu_probs[:, :-1]

        if self.seq_encoder is not None:
            m_seq_hidden_states = batch['seq_hidden_state']
            bn_seq_hidden_states = m_seq_hidden_states[:, :-1, ...]

            m_low_seq_hidden_states = batch['low_seq_hidden_state']
            bn_low_seq_hidden_states = m_low_seq_hidden_states[:, :-1, ...]

        key_batch = None
        if self.use_dilation:
            tmp_pointers = pointers
            key_tran = self.replay_buffer.get_storage_data(tmp_pointers)
            key_trans = {k: [v] for k, v in key_tran.items()}
            key_trans['padding_mask'] = [np.zeros_like(key_tran['index'], dtype=bool)]

            for _ in range(self.burn_in_step):  # All keys are the first keys in episodes
                tmp_tran_index = key_trans['index'][0]  # The current key tran index in an episode
                tmp_pre_tran = self.replay_buffer.get_storage_data(tmp_pointers - 1)  # The previous tran of the current key tran
                tmp_option_changed_index = tmp_pre_tran['option_changed_index']
                # The previous option changed key index in an episode
                delta = tmp_tran_index - tmp_option_changed_index

                padding_mask = tmp_tran_index == 0  # The current key tran is the first key in an episode
                padding_mask = np.logical_or(padding_mask, tmp_tran_index - tmp_pre_tran['index'] != 1)
                # The previous tran is not actually the previous tran of the current key tran

                delta[padding_mask] = 0
                tmp_pointers = (tmp_pointers - delta).astype(pointers.dtype)
                tmp_tran = self.replay_buffer.get_storage_data(tmp_pointers)
                tmp_tran['padding_mask'] = np.zeros_like(tmp_tran_index, dtype=bool)
                tmp_tran['padding_mask'][padding_mask] = True
                for name in self.obs_names:
                    tmp_tran[f'obs_{name}'][padding_mask] = 0.
                tmp_tran['option_index'][padding_mask] = -1
                tmp_tran['seq_hidden_state'][padding_mask] = 0.
                for k, v in tmp_tran.items():
                    key_trans[k].insert(0, v)

            for k, v in key_trans.items():
                del v[-1]
                key_trans[k] = np.concatenate([np.expand_dims(t, 1) for t in v], axis=1)

            key_batch = (
                key_trans['index'],
                key_trans['padding_mask'],
                [key_trans[f'obs_{name}'] for name in self.obs_names],
                key_trans['option_index'],
                key_trans['seq_hidden_state']
            )

        return (pointers,
                (bn_indexes,
                 bn_padding_masks,
                 bn_obses_list,
                 bn_option_indexes,
                 bn_actions,
                 bn_rewards,
                 next_obs_list,
                 bn_dones,
                 bn_mu_probs,
                 bn_seq_hidden_states if self.seq_encoder is not None else None,
                 bn_low_seq_hidden_states if self.seq_encoder is not None else None,
                 priority_is if self.use_replay_buffer and self.use_priority else None),
                key_batch)

    @unified_elapsed_timer('train_all', 10)
    def train(self, train_all_profiler) -> int:
        step = self.get_global_step()

        if self.use_replay_buffer:
            with self._profiler('sample_from_replay_buffer', repeat=10) as profiler:
                train_data = self._sample_from_replay_buffer()
            if train_data is None:
                profiler.ignore()
                train_all_profiler.ignore()
                return step

            pointers, batch, key_batch = train_data
            batch_list = [batch]
            key_batch_list = [key_batch]
        else:
            assert not self.use_dilation
            batch_list = self.batch_buffer.get_batch()
            batch_list = [(*batch, None) for batch in batch_list]  # None is priority_is
            key_batch_list = [None] * len(batch_list)

        for batch, key_batch in zip(batch_list, key_batch_list):
            (bn_indexes,
             bn_padding_masks,
             bn_obses_list,
             bn_option_indexes,
             bn_actions,
             bn_rewards,
             next_obs_list,
             bn_dones,
             bn_mu_probs,
             bn_seq_hidden_states,
             bn_low_seq_hidden_states,
             priority_is) = batch

            """
            bn_indexes (np.int32): [batch, b + n]
            bn_padding_masks (bool): [batch, b + n]
            bn_obses_list: list([batch, b + n, *obs_shapes_i], ...)
            bn_option_indexes (np.int8): [batch, b + n]
            bn_actions: [batch, b + n, action_size]
            bn_rewards: [batch, b + n]
            next_obs_list: list([batch, *obs_shapes_i], ...)
            bn_dones (bool): [batch, b + n]
            bn_mu_probs: [batch, b + n, action_size]
            bn_seq_hidden_states: [batch, b + n, *seq_hidden_state_shape]
            bn_low_seq_hidden_states: [batch, b + n, *low_seq_hidden_state_shape]
            priority_is: [batch, 1]
            """
            assert bn_indexes.dtype == np.int32
            assert bn_option_indexes.dtype == np.int8

            with self._profiler('to gpu', repeat=10):
                bn_indexes = torch.from_numpy(bn_indexes).to(self.device)
                bn_padding_masks = torch.from_numpy(bn_padding_masks).to(self.device)
                bn_obses_list = [torch.from_numpy(t).to(self.device) for t in bn_obses_list]
                for i, bn_obses in enumerate(bn_obses_list):
                    # obs is image. It is much faster to convert uint8 to float32 in GPU
                    if bn_obses.dtype == torch.uint8:
                        bn_obses_list[i] = bn_obses.type(torch.float32) / 255.
                bn_option_indexes = torch.from_numpy(bn_option_indexes).type(torch.int64).to(self.device)
                bn_actions = torch.from_numpy(bn_actions).to(self.device)
                bn_rewards = torch.from_numpy(bn_rewards).to(self.device)
                next_obs_list = [torch.from_numpy(t).to(self.device) for t in next_obs_list]
                for i, next_obs in enumerate(next_obs_list):
                    # obs is image
                    if next_obs.dtype == torch.uint8:
                        next_obs_list[i] = next_obs.type(torch.float32) / 255.
                bn_dones = torch.from_numpy(bn_dones).to(self.device)
                bn_mu_probs = torch.from_numpy(bn_mu_probs).to(self.device)
                if self.seq_encoder is not None:
                    f_seq_hidden_states = bn_seq_hidden_states[:, :1]
                    f_seq_hidden_states = torch.from_numpy(f_seq_hidden_states).to(self.device)

                    f_low_seq_hidden_states = bn_low_seq_hidden_states[:, self.option_burn_in_from:self.option_burn_in_from + 1]
                    f_low_seq_hidden_states = torch.from_numpy(f_low_seq_hidden_states).to(self.device)
                if self.use_replay_buffer and self.use_priority:
                    priority_is = torch.from_numpy(priority_is).to(self.device)

                if key_batch is not None:
                    (key_indexes,
                     key_padding_masks,
                     key_obses_list,
                     key_option_indexes,
                     key_seq_hidden_states) = key_batch

                    key_indexes = torch.from_numpy(key_indexes).to(self.device)
                    key_padding_masks = torch.from_numpy(key_padding_masks).to(self.device)
                    key_obses_list = [torch.from_numpy(t).to(self.device) for t in key_obses_list]
                    for i, key_obses in enumerate(key_obses_list):
                        # obs is image
                        if key_obses.dtype == torch.uint8:
                            key_obses_list[i] = key_obses.type(torch.float32) / 255.
                    key_option_indexes = torch.from_numpy(key_option_indexes).to(self.device)
                    key_seq_hidden_states = torch.from_numpy(key_seq_hidden_states).to(self.device)

                    # self._logger.debug(f'train {key_indexes.shape}')

                    key_batch = (key_indexes,
                                 key_padding_masks,
                                 key_obses_list,
                                 key_option_indexes,
                                 key_seq_hidden_states)

            with self._profiler('train', repeat=10):
                (m_target_states,
                 om_low_target_obses_list,
                 om_low_target_states) = self._train(
                    bn_indexes=bn_indexes,
                    bn_padding_masks=bn_padding_masks,
                    bn_obses_list=bn_obses_list,
                    bn_option_indexes=bn_option_indexes,
                    bn_actions=bn_actions,
                    bn_rewards=bn_rewards,
                    next_obs_list=next_obs_list,
                    bn_dones=bn_dones,
                    bn_mu_probs=bn_mu_probs,
                    f_seq_hidden_states=f_seq_hidden_states if self.seq_encoder is not None else None,
                    f_low_seq_hidden_states=f_low_seq_hidden_states if self.seq_encoder is not None else None,
                    priority_is=priority_is if self.use_replay_buffer and self.use_priority else None,

                    key_batch=key_batch)

            if step % self.save_model_per_step == 0:
                self.save_model()

            if self.use_replay_buffer:
                bn_pre_actions = gen_pre_n_actions(bn_actions)  # [batch, b + n, action_size]

                with self._profiler('get_l_states_with_seq_hidden_states', repeat=10):
                    bn_states, next_bn_seq_hidden_states = self.get_l_states_with_seq_hidden_states(
                        l_indexes=bn_indexes,
                        l_padding_masks=bn_padding_masks,
                        l_obses_list=bn_obses_list,
                        l_pre_actions=bn_pre_actions,

                        key_batch=key_batch,

                        f_seq_hidden_states=f_seq_hidden_states if self.seq_encoder is not None else None)

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
                        f_low_seq_hidden_states=f_low_seq_hidden_states if self.seq_encoder is not None else None
                    )

                if self.use_n_step_is or (self.d_action_sizes and not self.discrete_dqn_like):
                    with self._profiler('get_l_probs', repeat=10):
                        obn_pi_probs_tensor = self.get_l_probs(
                            l_low_obses_list=bn_low_obses_list,
                            l_low_states=obn_low_states,
                            l_option_indexes=bn_option_indexes[:, self.option_burn_in_from:],
                            l_actions=bn_actions[:, self.option_burn_in_from:])

                # Update td_error
                if self.use_priority:
                    with self._profiler('get_td_error', repeat=10):
                        td_error = self._get_td_error(
                            bn_padding_masks=bn_padding_masks,
                            bn_states=bn_states,
                            bn_target_states=m_target_states[:, :-1, ...],
                            bn_option_indexes=bn_option_indexes,
                            obn_low_obses_list=[bn_low_obses[:, self.option_burn_in_from:] for bn_low_obses in bn_low_obses_list],
                            obn_low_states=obn_low_states,
                            obn_low_target_states=om_low_target_states[:, :-1, ...],
                            bn_actions=bn_actions,
                            bn_rewards=bn_rewards,
                            next_target_state=m_target_states[:, -1, ...],
                            next_low_obs_list=[m_low_target_obses[:, -1, ...]
                                               for m_low_target_obses in om_low_target_obses_list],
                            next_low_target_state=om_low_target_states[:, -1, ...],
                            bn_dones=bn_dones,
                            bn_mu_probs=bn_mu_probs
                        ).detach().cpu().numpy()
                    self.replay_buffer.update(pointers, td_error)

                bn_padding_masks = bn_padding_masks.detach().cpu().numpy()
                padding_mask = bn_padding_masks.reshape(-1)
                low_padding_mask = bn_padding_masks[:, self.option_burn_in_from:].reshape(-1)

                # Update seq_hidden_states
                if self.seq_encoder is not None:
                    pointers_list = [pointers + 1 + i for i in range(-self.burn_in_step, self.n_step)]
                    tmp_pointers = np.stack(pointers_list, axis=1).reshape(-1)

                    next_bn_seq_hidden_states = next_bn_seq_hidden_states.detach().cpu().numpy()
                    seq_hidden_state = next_bn_seq_hidden_states.reshape(-1, *next_bn_seq_hidden_states.shape[2:])
                    self.replay_buffer.update_transitions(tmp_pointers[~padding_mask], 'seq_hidden_state', seq_hidden_state[~padding_mask])

                    pointers_list = [pointers + 1 + self.option_burn_in_from + i for i in range(-self.option_burn_in_step, self.n_step)]
                    tmp_pointers = np.stack(pointers_list, axis=1).reshape(-1)

                    next_obn_low_seq_hidden_states = next_obn_low_seq_hidden_states.detach().cpu().numpy()
                    low_seq_hidden_state = next_obn_low_seq_hidden_states.reshape(-1, *next_obn_low_seq_hidden_states.shape[2:])
                    self.replay_buffer.update_transitions(tmp_pointers[~low_padding_mask], 'low_seq_hidden_state', low_seq_hidden_state[~low_padding_mask])

                # Update n_mu_probs
                if self.use_n_step_is or (self.d_action_sizes and not self.discrete_dqn_like):
                    pointers_list = [pointers + self.option_burn_in_from + i for i in range(-self.option_burn_in_step, self.n_step)]
                    tmp_pointers = np.stack(pointers_list, axis=1).reshape(-1)

                    pi_probs = obn_pi_probs_tensor.detach().cpu().numpy()
                    pi_prob = pi_probs.reshape(-1, *pi_probs.shape[2:])
                    self.replay_buffer.update_transitions(tmp_pointers[~low_padding_mask], 'mu_prob', pi_prob[~low_padding_mask])

            step = self._increase_global_step()
            for option in self.option_list:
                option._increase_global_step()

        return step

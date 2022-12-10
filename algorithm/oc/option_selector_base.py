from collections import defaultdict
from itertools import chain
from typing import List, Optional

import numpy as np
import torch
from torch import nn, optim

from algorithm.oc.option_base import OptionBase

from .. import sac_base
from ..nn_models import *
from ..sac_base import SAC_Base
from ..utils import *
from .oc_batch_buffer import BatchBuffer, episode_to_batch

sac_base.BatchBuffer = BatchBuffer


class OptionSelectorBase(SAC_Base):
    def __init__(self,
                 num_options: int,
                 option_nn_config: dict,

                 obs_shapes: List[Tuple],
                 d_action_size: int,
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
                 use_add_with_td: bool = False,
                 action_noise: Optional[List[float]] = None,

                 replay_config: Optional[dict] = None):

        self.num_options = num_options
        self.option_nn_config = option_nn_config

        super().__init__(obs_shapes,
                         d_action_size,
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
                         use_add_with_td,
                         action_noise,
                         replay_config)

    def _build_model(self, nn, nn_config: Optional[dict], init_log_alpha: float, learning_rate: float):
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
                                                       self.d_action_size, self.c_action_size,
                                                       False, self.train_mode,
                                                       self.model_abs_dir,
                                                       **nn_config['rep']).to(self.device)
            self.model_target_rep: ModelBaseRNNRep = ModelRep(self.obs_shapes,
                                                              self.d_action_size, self.c_action_size,
                                                              True, self.train_mode,
                                                              self.model_abs_dir,
                                                              **nn_config['rep']).to(self.device)
            # Get represented state and seq_hidden_state_shape
            test_obs_list = [torch.rand(self.batch_size, 1, *obs_shape, device=self.device) for obs_shape in self.obs_shapes]
            test_pre_action = torch.rand(self.batch_size, 1, self.d_action_size + self.c_action_size, device=self.device)
            test_state, test_rnn_state = self.model_rep(test_obs_list,
                                                        test_pre_action)
            state_size, self.seq_hidden_state_shape = test_state.shape[-1], test_rnn_state.shape[1:]

        elif self.seq_encoder == SEQ_ENCODER.ATTN:
            self.model_rep: ModelBaseAttentionRep = ModelRep(self.obs_shapes,
                                                             self.d_action_size, self.c_action_size,
                                                             False, self.train_mode,
                                                             self.model_abs_dir,
                                                             **nn_config['rep']).to(self.device)
            self.model_target_rep: ModelBaseAttentionRep = ModelRep(self.obs_shapes,
                                                                    self.d_action_size, self.c_action_size,
                                                                    True, self.train_mode,
                                                                    self.model_abs_dir,
                                                                    **nn_config['rep']).to(self.device)
            # Get represented state and seq_hidden_state_shape
            test_index = torch.zeros((self.batch_size, 1), dtype=torch.int32, device=self.device)
            test_obs_list = [torch.rand(self.batch_size, 1, *obs_shape, device=self.device) for obs_shape in self.obs_shapes]
            test_pre_action = torch.rand(self.batch_size, 1, self.d_action_size + self.c_action_size, device=self.device)
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

        self.model_v_over_options_list = [ModelQOverOption(state_size, self.num_options).to(self.device) for _ in range(self.ensemble_q_num)]
        self.model_target_v_over_options_list = [ModelQOverOption(state_size, self.num_options).to(self.device) for _ in range(self.ensemble_q_num)]
        for model_target_v_over_options in self.model_target_v_over_options_list:
            for param in model_target_v_over_options.parameters():
                param.requires_grad = False
        self.optimizer_v_list = [adam_optimizer(self.model_v_over_options_list[i].parameters()) for i in range(self.ensemble_q_num)]

        self.model_termination_over_options = ModelTerminationOverOption(state_size, self.num_options).to(self.device)
        self.optimizer_termination = adam_optimizer(self.model_termination_over_options.parameters())

        """
        Initialize each Option
        """
        option_kwargs = self._kwargs
        del option_kwargs['self']
        option_kwargs['obs_shapes'] = [(self.state_size, ), *self.obs_shapes]
        option_kwargs['seq_encoder'] = SEQ_ENCODER.RNN if self.seq_encoder == SEQ_ENCODER.ATTN else self.seq_encoder

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

    def _init_or_restore(self, last_ckpt: int):
        """
        Initialize network weights from scratch or restore from model_abs_dir
        """
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

        ckpt_dict[f'model_termination_over_options_{i}'] = self.model_termination_over_options

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
                for name, model in ckpt_dict.items():
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

    def save_model(self, save_replay_buffer=False):
        super().save_model(save_replay_buffer)

        for option in self.option_list:
            option.save_model(save_replay_buffer)

    def get_initial_option_index(self, batch_size: int):
        return np.full([batch_size, ], -1, dtype=np.int8)

    def get_initial_low_seq_hidden_state(self, batch_size, get_numpy=True):
        return self.option_list[0].get_initial_seq_hidden_state(batch_size,
                                                                get_numpy)

    @torch.no_grad()
    def _update_target_variables(self, tau=1.):
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
                             state: torch.Tensor,
                             option_index: torch.Tensor,
                             disable_sample: bool = False):
        """
        Args:
            state: [Batch, state_size]
            option_index (torch.int64): [Batch, ]

        Returns:
            new_option_index (torch.int64): [Batch, ]
            new_option_mask (torch.bool): [Batch, ]
        """
        v_over_options = self.model_v_over_options_list[0](state)  # [Batch, num_options]
        new_option_index = v_over_options.argmax(dim=-1)  # [Batch, ]

        none_option_mask = option_index == -1
        option_index[none_option_mask] = new_option_index[none_option_mask]

        termination_over_options = self.model_termination_over_options(state)  # [Batch, num_options]
        termination = termination_over_options.gather(1, option_index.unsqueeze(-1))  # [Batch, 1]
        termination = termination.squeeze(-1)
        termination_mask = termination > .5
        option_index[termination_mask] = new_option_index[termination_mask]

        mask = torch.logical_or(none_option_mask, termination_mask)
        if not disable_sample:
            random_mask = torch.rand_like(option_index, dtype=torch.float32) < 0.2
            option_index[random_mask] = new_option_index[random_mask]
            mask = torch.logical_or(mask, random_mask)

        return option_index, mask

    def _choose_action(self,
                       obs_list: List[torch.Tensor],
                       state: torch.Tensor,
                       option_index: torch.Tensor,

                       disable_sample: bool = False,
                       force_rnd_if_available: bool = False):
        """
        Args:
            obs_list: list([Batch, 1, *obs_shapes_i], ...)
            state: [Batch, 1, d_action_size + c_action_size]
            option_index (torch.int64): [Batch, ]

        Returns:
            new_option_index (torch.int64): [Batch, ]
            action: [Batch, action_size]
            policy_prob: [Batch, ]
        """

        option_index, _ = self._choose_option_index(state=state,
                                                    option_index=option_index,
                                                    disable_sample=disable_sample)

        batch = state.shape[0]
        action = torch.zeros(batch, self.d_action_size + self.c_action_size, device=self.device)
        policy_prob = torch.ones(batch, device=self.device)
        obs_list = [state] + obs_list

        for i, option in enumerate(self.option_list):
            mask = (option_index == i)
            if not torch.any(mask):
                continue

            o_obs_list = [obs[mask] for obs in obs_list]

            o_action, o_prob = option.choose_action(o_obs_list,
                                                    disable_sample=disable_sample,
                                                    force_rnd_if_available=force_rnd_if_available)
            action[mask] = o_action
            policy_prob[mask] = o_prob

        return option_index, action, policy_prob

    def _choose_rnn_action(self,
                           obs_list: List[torch.Tensor],
                           state: torch.Tensor,
                           option_index: torch.Tensor,
                           pre_action: torch.Tensor,
                           low_rnn_state: torch.Tensor,

                           disable_sample: bool = False,
                           force_rnd_if_available: bool = False):
        """
        Args:
            obs_list: list([Batch, 1, *obs_shapes_i], ...)
            state: [Batch, 1, d_action_size + c_action_size]
            option_index (torch.int64): [Batch, ]
            pre_action: [Batch, 1, action_size]
            low_rnn_state: [Batch, *low_seq_hidden_state_shape]

        Returns:
            new_option_index (torch.int64): [Batch, ]
            action: [Batch, action_size]
            policy_prob: [Batch, ]
            next_low_rnn_state: [Batch, *low_seq_hidden_state_shape]
        """

        option_index, new_option_index_mask = self._choose_option_index(state=state,
                                                                        option_index=option_index,
                                                                        disable_sample=disable_sample)

        batch = state.shape[0]
        initial_low_seq_hidden_state = self.get_initial_low_seq_hidden_state(batch, get_numpy=False)

        low_rnn_state[new_option_index_mask] = initial_low_seq_hidden_state[new_option_index_mask]

        action = torch.zeros(batch, self.d_action_size + self.c_action_size, device=self.device)
        policy_prob = torch.ones(batch, device=self.device)
        next_low_rnn_state = torch.zeros_like(low_rnn_state)
        obs_list = [state] + obs_list

        for i, option in enumerate(self.option_list):
            mask = (option_index == i)
            if not torch.any(mask):
                continue

            o_obs_list = [obs[mask] for obs in obs_list]

            o_action, o_prob, o_next_low_rnn_state = option.choose_rnn_action(o_obs_list,
                                                                              pre_action[mask],
                                                                              low_rnn_state[mask],
                                                                              disable_sample=disable_sample,
                                                                              force_rnd_if_available=force_rnd_if_available)
            action[mask] = o_action
            policy_prob[mask] = o_prob
            next_low_rnn_state[mask] = o_next_low_rnn_state

        return option_index, action, policy_prob, next_low_rnn_state

    def choose_action(self,
                      obs_list: List[np.ndarray],
                      pre_option_index: np.ndarray,

                      disable_sample: bool = False,
                      force_rnd_if_available: bool = False):
        """
        Args:
            obs_list (np): list([Batch, *obs_shapes_i], ...)
            pre_option_index (np.int8): [Batch, ]

        Returns:
            option_index (np.int8): [Batch, ]
            action (np): [Batch, d_action_size + c_action_size]
            prob (np): [Batch, ]
        """
        obs_list = [torch.from_numpy(obs).to(self.device) for obs in obs_list]
        pre_option_index = torch.from_numpy(pre_option_index).type(torch.int64).to(self.device)

        state = self.model_rep(obs_list)

        option_index, action, prob = self._choose_action(obs_list,
                                                         state,
                                                         pre_option_index,
                                                         disable_sample=disable_sample,
                                                         force_rnd_if_available=force_rnd_if_available)

        return (option_index.detach().cpu().numpy().astype(np.int8),
                action.detach().cpu().numpy(),
                prob.detach().cpu().numpy())

    def choose_rnn_action(self,
                          obs_list: List[np.ndarray],
                          pre_option_index: np.ndarray,
                          pre_action: np.ndarray,
                          rnn_state: np.ndarray,
                          low_rnn_state: np.ndarray,

                          disable_sample: bool = False,
                          force_rnd_if_available: bool = False):
        """
        Args:
            obs_list (np): list([Batch, *obs_shapes_i], ...)
            pre_option_index (np.int8): [Batch, ]
            pre_action (np): [Batch, d_action_size + c_action_size]
            rnn_state (np): [Batch, *seq_hidden_state_shape]
            low_rnn_state (np): [Batch, *low_seq_hidden_state_shape]

        Returns:
            option_index (np.int8): [Batch, ]
            action (np): [Batch, d_action_size + c_action_size]
            prob (np): [Batch, ]
            next_rnn_state (np): [Batch, *seq_hidden_state_shape]
            next_low_rnn_state (np): [Batch, *low_seq_hidden_state_shape]
        """
        obs_list = [torch.from_numpy(obs).to(self.device) for obs in obs_list]
        pre_option_index = torch.from_numpy(pre_option_index).type(torch.int64).to(self.device)
        pre_action = torch.from_numpy(pre_action).to(self.device)
        rnn_state = torch.from_numpy(rnn_state).to(self.device)
        low_rnn_state = torch.from_numpy(low_rnn_state).to(self.device)

        obs_list = [obs.unsqueeze(1) for obs in obs_list]
        pre_action = pre_action.unsqueeze(1)

        state, next_rnn_state = self.model_rep(obs_list, pre_action, rnn_state)
        # state: [Batch, 1, state_size]

        obs_list = [obs.squeeze(1) for obs in obs_list]
        state = state.squeeze(1)
        pre_action = pre_action.squeeze(1)

        (option_index,
         action,
         prob,
         next_low_rnn_state) = self._choose_rnn_action(obs_list,
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
                           force_rnd_if_available: bool = False):
        """
        Args:
            ep_indexes (np.int32): [Batch, episode_len]
            ep_padding_masks (np.bool): [Batch, episode_len]
            ep_obses_list (np): list([Batch, episode_len, *obs_shapes_i], ...)
            ep_pre_actions (np): [Batch, episode_len, d_action_size + c_action_size]
            ep_attn_states (np): [Batch, episode_len, *seq_hidden_state_shape]

            pre_option_index (np.int8): [Batch, ]
            low_rnn_state (np): [Batch, *low_seq_hidden_state_shape]

        Returns:
            action (np): [Batch, d_action_size + c_action_size]
            prob (np): [Batch, ]
            next_attn_state (np): [Batch, *attn_state_shape]
            next_low_rnn_state (np): [Batch, *low_rnn_state_shape]
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
                                                   hidden_state=ep_attn_states,
                                                   is_prev_hidden_state=False,
                                                   padding_mask=ep_padding_masks)
        # state: [Batch, 1, state_size]
        # next_attn_state: [Batch, 1, *attn_state_shape]

        obs_list = [ep_obses[:, -1, ...] for ep_obses in ep_obses_list]
        state = state.squeeze(1)
        pre_action = ep_pre_actions[:, -1, ...]
        next_attn_state = next_attn_state.squeeze(1)

        (option_index,
         action,
         prob,
         next_low_rnn_state) = self._choose_rnn_action(obs_list,
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

    def get_l_low_obses_list(self,
                             l_obses_list: List[torch.Tensor],
                             l_states: torch.Tensor):
        """
        Args:
            l_obses_list: list([Batch, l, *obs_shapes_i], ...)
            l_states: [Batch, l, state_size]

        Returns:
            l_low_obses_list: list([Batch, l, *low_obs_shapes_i], ...)
        """

        l_low_obses_list = [l_states] + l_obses_list

        return l_low_obses_list

    def get_m_data(self,
                   bn_indexes: torch.Tensor,
                   bn_padding_masks: torch.Tensor,
                   bn_obses_list: List[torch.Tensor],
                   bn_actions: torch.Tensor,
                   next_obs_list: torch.Tensor):
        m_indexes = torch.concat([bn_indexes, bn_indexes[:, -1:] + 1], dim=1)
        m_padding_masks = torch.concat([bn_padding_masks,
                                        torch.zeros_like(bn_padding_masks[:, -1:], dtype=torch.bool)], dim=1)
        m_obses_list = [torch.cat([n_obses, next_obs.unsqueeze(1)], dim=1)
                        for n_obses, next_obs in zip(bn_obses_list, next_obs_list)]
        m_pre_actions = gen_pre_n_actions(bn_actions, keep_last_action=True)

        return m_indexes, m_padding_masks, m_obses_list, m_pre_actions

    def get_l_low_states_with_seq_hidden_states(self,
                                                l_indexes: torch.Tensor,
                                                l_padding_masks: torch.Tensor,
                                                l_low_obses_list: List[torch.Tensor],
                                                l_option_indexes: torch.Tensor,
                                                l_pre_actions: torch.Tensor,
                                                f_low_seq_hidden_states: torch.Tensor = None,
                                                is_target=False):
        batch, l, *_ = l_indexes.shape

        l_low_states = None
        next_l_low_seq_hidden_states = None

        for t in range(l):
            f_indexes = l_indexes[:, t:t + 1]
            f_padding_masks = l_padding_masks[:, t:t + 1]
            f_low_obses_list = [l_obses[:, t:t + 1, ...] for l_obses in l_low_obses_list]  # list([Batch, 1, *obs_shapes_i])
            f_pre_actions = l_pre_actions[:, t:t + 1, ...]  # [Batch, 1, action_size]
            option_index = l_option_indexes[:, t]  # [Batch, ]

            if self.seq_encoder is not None:
                next_l_low_seq_hidden_states = torch.zeros((batch, l, *f_low_seq_hidden_states.shape[2:]), device=self.device)

            if t > 0 and self.seq_encoder is not None:
                mask = l_option_indexes[:, t - 1] != option_index
                f_low_seq_hidden_states[mask, 0] = self.get_initial_low_seq_hidden_state(batch, False)[mask]

            for i, option in enumerate(self.option_list):
                mask = (option_index == i)
                if not torch.any(mask):
                    continue

                o_f_indexes = f_indexes[mask]
                o_f_padding_masks = f_padding_masks[mask]
                o_f_obses_list = [f_low_obses[mask] for f_low_obses in f_low_obses_list]
                if self.seq_encoder is not None:
                    o_f_pre_actions = f_pre_actions[mask]
                    o_f_low_seq_hidden_states = f_low_seq_hidden_states[mask]
                else:
                    o_f_pre_actions = None
                    o_f_low_seq_hidden_states = None

                o_f_states, o_tmp_f_low_seq_hidden_states = option.get_l_states(
                    l_indexes=o_f_indexes,
                    l_padding_masks=o_f_padding_masks,
                    l_obses_list=o_f_obses_list,
                    l_pre_actions=o_f_pre_actions,
                    f_seq_hidden_states=o_f_low_seq_hidden_states,
                    is_target=is_target
                )

                if l_low_states is None:
                    l_low_states = torch.zeros((batch, l, *o_f_states.shape[2:]), device=self.device)
                l_low_states[mask, t:t + 1] = o_f_states

                if self.seq_encoder is not None:
                    f_low_seq_hidden_states[mask] = o_tmp_f_low_seq_hidden_states
                    next_l_low_seq_hidden_states[mask, t:t + 1] = o_tmp_f_low_seq_hidden_states

        return l_low_states, next_l_low_seq_hidden_states

    @torch.no_grad()
    def get_l_probs(self,
                    l_low_obses_list: List[torch.Tensor],
                    l_low_states: torch.Tensor,
                    l_option_indexes: torch.Tensor,
                    l_actions: torch.Tensor):
        """
        Args:
            l_low_obses_list: list([Batch, l, *low_obs_shapes_i], ...)
            l_low_states: [Batch, l, low_state_size]
            l_option_indexes (torch.int64): [Batch, l]
            l_actions: [Batch, l, action_size]

        Returns:
            l_low_probs: [Batch, l]
        """
        batch, l, *_ = l_low_states.shape

        l_low_probs = None

        for t in range(l):
            f_low_obses_list = [l_obses[:, t:t + 1, ...] for l_obses in l_low_obses_list]  # list([Batch, 1, *obs_shapes_i])
            f_low_states = l_low_states[:, t:t + 1, ...]  # [Batch, 1, low_state_size]
            f_actions = l_actions[:, t:t + 1, ...]  # [Batch, 1, action_size]
            option_index = l_option_indexes[:, t]  # [Batch, ]

            for i, option in enumerate(self.option_list):
                mask = (option_index == i)
                if not torch.any(mask):
                    continue

                o_f_obses_list = [f_low_obses[mask] for f_low_obses in f_low_obses_list]
                o_f_states = f_low_states[mask]
                o_f_actions = f_actions[mask]

                o_f_probs = option.get_l_probs(l_obses_list=o_f_obses_list,
                                               l_states=o_f_states,
                                               l_actions=o_f_actions)
                if l_low_probs is None:
                    l_low_probs = torch.zeros((batch, l, *o_f_probs.shape[2:]), device=self.device)
                l_low_probs[mask, t:t + 1] = o_f_probs

        return l_low_probs

    @torch.no_grad()
    def _get_td_error(self,
                      bn_states: torch.Tensor,
                      bn_target_states: torch.Tensor,
                      bn_option_indexes: torch.Tensor,
                      bn_low_obses_list: List[torch.Tensor],
                      bn_low_states: torch.Tensor,
                      bn_low_target_states: torch.Tensor,
                      bn_actions: torch.Tensor,
                      bn_rewards: torch.Tensor,
                      next_target_state: torch.Tensor,
                      next_low_obs_list: List[torch.Tensor],
                      next_low_target_state: torch.Tensor,
                      bn_dones: torch.Tensor,
                      bn_mu_probs: torch.Tensor = None):
        """
        Args:
            bn_states: [Batch, b + n, state_size]
            bn_target_states: [Batch, b + n, state_size]
            bn_option_indexes (torch.int64): [Batch, b + n]
            bn_low_obses_list: list([Batch, b + n, *low_obs_shapes_i], ...)
            bn_low_states: [Batch, b + n, low_state_size]
            bn_low_target_states: [Batch, b + n, low_state_size]
            bn_actions: [Batch, b + n, action_size]
            bn_rewards: [Batch, b + n]
            next_target_state: [Batch, state_size]
            next_low_obs_list: list([Batch, *low_obs_shapes_i], ...)
            next_low_target_state: [Batch, low_state_size]
            bn_dones (torch.bool): [Batch, b + n]
            bn_mu_probs: [Batch, b + n]

        Returns:
            The td-error of observations, [Batch, 1]
        """

        next_n_states = torch.concat([bn_target_states[:, self.burn_in_step + 1:, ...],
                                      next_target_state.unsqueeze(1)], dim=1)  # [Batch, n, state_size]
        n_option_indexes = bn_option_indexes[:, self.burn_in_step:]  # [Batch, n]
        option_index = n_option_indexes[:, 0]  # [Batch, ]

        next_n_terminations_over_options = self.model_termination_over_options(next_n_states)  # [Batch, n, num_options]
        next_n_terminations = next_n_terminations_over_options.gather(2, n_option_indexes.unsqueeze(-1))  # [Batch, n, 1]
        next_n_terminations = next_n_terminations.squeeze(-1)  # [Batch, n]

        next_n_v_over_options_list = [v(next_n_states) for v in self.model_target_v_over_options_list]  # [Batch, n, num_options]

        batch = bn_states.shape[0]
        td_error = torch.zeros((batch, 1), device=self.device)

        for i, option in enumerate(self.option_list):
            mask = (option_index == i)
            if not torch.any(mask):
                continue

            o_td_error = option._get_td_error(
                next_n_terminations=next_n_terminations[mask],

                next_n_v_over_options_list=[next_n_v_over_options[mask]
                                            for next_n_v_over_options in next_n_v_over_options_list],

                bn_obses_list=[bn_low_obses[mask] for bn_low_obses in bn_low_obses_list],
                bn_states=bn_low_states[mask],
                bn_target_states=bn_low_target_states[mask],
                bn_actions=bn_actions[mask],
                bn_rewards=bn_rewards[mask],
                next_obs_list=[next_low_obs[mask] for next_low_obs in next_low_obs_list],
                next_target_state=next_low_target_state[mask],
                bn_dones=bn_dones[mask],
                bn_mu_probs=bn_mu_probs[mask] if self.use_n_step_is else None)

            td_error[mask] = o_td_error

        return td_error

    def get_episode_td_error(self,
                             l_indexes: np.ndarray,
                             l_padding_masks: np.ndarray,
                             l_obses_list: List[np.ndarray],
                             l_option_indexes: np.ndarray,
                             l_actions: np.ndarray,
                             l_rewards: np.ndarray,
                             next_obs_list: List[np.ndarray],
                             l_dones: np.ndarray,
                             l_mu_probs: np.ndarray = None,
                             l_seq_hidden_states: np.ndarray = None,
                             l_low_seq_hidden_states: np.ndarray = None):
        """
        Args:
            l_indexes: [1, episode_len]
            l_padding_masks: [1, episode_len]
            l_obses_list: list([1, episode_len, *obs_shapes_i], ...)
            l_option_indexes: [1, episode_len]
            l_actions: [1, episode_len, action_size]
            l_rewards: [1, episode_len]
            next_obs_list: list([1, *obs_shapes_i], ...)
            l_dones: [1, episode_len]
            l_mu_probs: [1, episode_len]
            l_seq_hidden_states: [1, episode_len, *seq_hidden_state_shape]
            l_low_seq_hidden_states: [1, episode_len, *low_seq_hidden_state_shape]

        Returns:
            The td-error of raw episode observations
            [episode_len, ]
        """
        ignore_size = self.burn_in_step + self.n_step

        (bn_indexes,
         bn_padding_masks,
         bn_obses_list,
         bn_options,
         bn_actions,
         bn_rewards,
         next_obs_list,
         bn_dones,
         bn_mu_probs,
         f_seq_hidden_states,
         f_low_seq_hidden_states) = episode_to_batch(bn=self.burn_in_step + self.n_step,
                                                     episode_length=l_indexes.shape[1],
                                                     l_indexes=l_indexes,
                                                     l_padding_masks=l_padding_masks,
                                                     l_obses_list=l_obses_list,
                                                     l_option_indexes=l_option_indexes,
                                                     l_actions=l_actions,
                                                     l_rewards=l_rewards,
                                                     next_obs_list=next_obs_list,
                                                     l_dones=l_dones,
                                                     l_probs=l_mu_probs,
                                                     l_seq_hidden_states=l_seq_hidden_states,
                                                     l_low_seq_hidden_states=l_low_seq_hidden_states)

        """
        bn_indexes: [episode_len - bn + 1, bn]
        bn_padding_masks: [episode_len - bn + 1, bn]
        bn_obses_list: list([episode_len - bn + 1, bn, *obs_shapes_i], ...)
        bn_options: [episode_len - bn + 1, bn]
        bn_actions: [episode_len - bn + 1, bn, action_size]
        bn_rewards: [episode_len - bn + 1, bn]
        next_obs_list: list([episode_len - bn + 1, *obs_shapes_i], ...)
        bn_dones: [episode_len - bn + 1, bn]
        bn_mu_probs: [episode_len - bn + 1, bn]
        f_seq_hidden_states: [episode_len - bn + 1, 1, *seq_hidden_state_shape]
        f_low_seq_hidden_states: [episode_len - bn + 1, 1, *low_seq_hidden_state_shape]
        """

        td_error_list = []
        all_batch = bn_obses_list[0].shape[0]
        batch_size = self.batch_size
        for i in range(math.ceil(all_batch / batch_size)):
            b_i, b_j = i * batch_size, (i + 1) * batch_size

            _bn_indexes = torch.from_numpy(bn_indexes[b_i:b_j]).to(self.device)
            _bn_padding_masks = torch.from_numpy(bn_padding_masks[b_i:b_j]).to(self.device)
            _bn_obses_list = [torch.from_numpy(o[b_i:b_j]).to(self.device) for o in bn_obses_list]
            _bn_options = torch.from_numpy(bn_options[b_i:b_j]).type(torch.int64).to(self.device)
            _bn_actions = torch.from_numpy(bn_actions[b_i:b_j]).to(self.device)
            _bn_rewards = torch.from_numpy(bn_rewards[b_i:b_j]).to(self.device)
            _next_obs_list = [torch.from_numpy(o[b_i:b_j]).to(self.device) for o in next_obs_list]
            _bn_dones = torch.from_numpy(bn_dones[b_i:b_j]).to(self.device)
            _bn_mu_probs = torch.from_numpy(bn_mu_probs[b_i:b_j]).to(self.device) if self.use_n_step_is else None
            _f_seq_hidden_states = torch.from_numpy(f_seq_hidden_states[b_i:b_j]).to(self.device) if self.seq_encoder is not None else None
            _f_low_seq_hidden_states = torch.from_numpy(f_low_seq_hidden_states[b_i:b_j]).to(self.device) if self.seq_encoder is not None else None

            (_m_indexes,
             _m_padding_masks,
             _m_obses_list,
             _m_pre_actions) = self.get_m_data(bn_indexes=_bn_indexes,
                                               bn_padding_masks=_bn_padding_masks,
                                               bn_obses_list=_bn_obses_list,
                                               bn_actions=_bn_actions,
                                               next_obs_list=_next_obs_list)
            _m_states, _ = self.get_l_states(l_indexes=_m_indexes,
                                             l_padding_masks=_m_padding_masks,
                                             l_obses_list=_m_obses_list,
                                             l_pre_actions=_m_pre_actions,
                                             f_seq_hidden_states=_f_seq_hidden_states if self.seq_encoder is not None else None,
                                             is_target=False)

            _m_target_states, _ = self.get_l_states(l_indexes=_m_indexes,
                                                    l_padding_masks=_m_padding_masks,
                                                    l_obses_list=_m_obses_list,
                                                    l_pre_actions=_m_pre_actions,
                                                    f_seq_hidden_states=_f_seq_hidden_states if self.seq_encoder is not None else None,
                                                    is_target=True)

            _m_low_obses_list = self.get_l_low_obses_list(l_obses_list=_m_obses_list,
                                                          l_states=_m_states)

            _m_low_target_obses_list = self.get_l_low_obses_list(l_obses_list=_m_obses_list,
                                                                 l_states=_m_target_states)

            _next_option, _ = self._choose_option_index(state=_m_states[:, -1, ...],
                                                        option_index=_bn_options[:, -1])
            _m_option_indexes = torch.concat([_bn_options, _next_option.unsqueeze(1)], dim=1)

            _m_low_states, _ = self.get_l_low_states_with_seq_hidden_states(
                l_indexes=_m_indexes,
                l_padding_masks=_m_padding_masks,
                l_low_obses_list=_m_low_obses_list,
                l_option_indexes=_m_option_indexes,
                l_pre_actions=_m_pre_actions,
                f_low_seq_hidden_states=_f_low_seq_hidden_states if self.seq_encoder is not None else None,
                is_target=False)

            _m_low_target_states, _ = self.get_l_low_states_with_seq_hidden_states(
                l_indexes=_m_indexes,
                l_padding_masks=_m_padding_masks,
                l_low_obses_list=_m_low_target_obses_list,
                l_option_indexes=_m_option_indexes,
                l_pre_actions=_m_pre_actions,
                f_low_seq_hidden_states=_f_low_seq_hidden_states if self.seq_encoder is not None else None,
                is_target=True)

            td_error = self._get_td_error(bn_states=_m_states[:, :-1, ...],
                                          bn_target_states=_m_target_states[:, :-1, ...],
                                          bn_option_indexes=_bn_options,
                                          bn_low_obses_list=[m_low_obses[:, :-1, ...] for m_low_obses in _m_low_obses_list],
                                          bn_low_states=_m_low_states[:, :-1, ...],
                                          bn_low_target_states=_m_low_target_states[:, :-1, ...],
                                          bn_actions=_bn_actions,
                                          bn_rewards=_bn_rewards,
                                          next_target_state=_m_target_states[:, -1, ...],
                                          next_low_obs_list=[m_low_target_obses[:, -1, ...]
                                                             for m_low_target_obses in _m_low_target_obses_list],
                                          next_low_target_state=_m_low_target_states[:, -1, ...],
                                          bn_dones=_bn_dones,
                                          bn_mu_probs=_bn_mu_probs if self.use_n_step_is else None).detach().cpu().numpy()
            td_error_list.append(td_error.flatten())

        td_error = np.concatenate([*td_error_list,
                                   np.zeros(ignore_size, dtype=np.float32)])
        return td_error

    def _train_rep_q(self,
                     next_n_terminations: torch.Tensor,

                     next_n_v_over_options_list: List[torch.Tensor],

                     bn_indexes: torch.Tensor,
                     bn_padding_masks: torch.Tensor,
                     bn_option_indexes: torch.Tensor,
                     bn_low_obses_list: List[torch.Tensor],
                     bn_low_target_obses_list: List[torch.Tensor],
                     bn_actions: torch.Tensor,
                     bn_rewards: torch.Tensor,
                     next_low_obs_list: List[torch.Tensor],
                     next_low_target_obs_list: List[torch.Tensor],
                     bn_dones: torch.Tensor,
                     bn_mu_probs: Optional[torch.Tensor] = None,
                     f_low_seq_hidden_states: Optional[torch.Tensor] = None,
                     priority_is: Optional[torch.Tensor] = None):
        """
        Args:
            next_n_terminations: [Batch, n],

            next_n_v_over_options_list: list([Batch, n, num_options], ...),

            bn_indexes (torch.int32): [Batch, b + n],
            bn_padding_masks (torch.bool): [Batch, b + n],
            bn_option_indexes (torch.int64): [Batch, b + n],
            bn_low_obses_list: list([Batch, b + n, *low_obs_shapes_i], ...)
            bn_low_target_obses_list: list([Batch, b + n, *low_obs_shapes_i], ...)
            bn_actions: [Batch, b + n, action_size],
            bn_rewards: [Batch, b + n],
            next_low_obs_list: list([Batch, *low_obs_shapes_i], ...)
            next_low_target_obs_list: list([Batch, *low_obs_shapes_i], ...)
            bn_dones (torch.bool): [Batch, b + n]
            bn_mu_probs: [Batch, b + n]
            f_low_seq_hidden_states: [Batch, 1, *low_seq_hidden_state_shape]
            priority_is: [Batch, 1]

        Returns:
            m_low_states: [Batch, N + 1, low_state_size]
            m_low_target_states: [Batch, N + 1, low_state_size]
        """

        if self.optimizer_rep:
            self.optimizer_rep.zero_grad()

        batch = bn_indexes.shape[0]

        option_index = bn_option_indexes[:, self.burn_in_step]

        m_low_states = None
        m_low_target_states = None

        for i, option in enumerate(self.option_list):
            mask = (option_index == i)
            if not torch.any(mask):
                continue

            (o_m_indexes,
             o_m_padding_masks,
             o_m_low_obses_list,
             o_m_pre_actions) = option.get_m_data(bn_indexes=bn_indexes[mask],
                                                  bn_padding_masks=bn_padding_masks[mask],
                                                  bn_obses_list=[bn_low_obses[mask] for bn_low_obses in bn_low_obses_list],
                                                  bn_actions=bn_actions[mask],
                                                  next_obs_list=[next_low_obs[mask] for next_low_obs in next_low_obs_list])

            (_,
             _,
             o_m_low_target_obses_list,
             _) = option.get_m_data(bn_indexes=bn_indexes[mask],
                                    bn_padding_masks=bn_padding_masks[mask],
                                    bn_obses_list=[bn_low_target_obses[mask] for bn_low_target_obses in bn_low_target_obses_list],
                                    bn_actions=bn_actions[mask],
                                    next_obs_list=[next_low_target_obs[mask] for next_low_target_obs in next_low_target_obs_list])

            if self.seq_encoder is not None:
                o_f_low_seq_hidden_states = f_low_seq_hidden_states[mask]

            o_m_low_states, _ = option.get_l_states(l_indexes=o_m_indexes,
                                                    l_padding_masks=o_m_padding_masks,
                                                    l_obses_list=o_m_low_obses_list,
                                                    l_pre_actions=o_m_pre_actions,
                                                    f_seq_hidden_states=o_f_low_seq_hidden_states if self.seq_encoder is not None else None,
                                                    is_target=False)

            with torch.no_grad():
                o_m_low_target_states, _ = option.get_l_states(l_indexes=o_m_indexes,
                                                               l_padding_masks=o_m_padding_masks,
                                                               l_obses_list=o_m_low_target_obses_list,
                                                               l_pre_actions=o_m_pre_actions,
                                                               f_seq_hidden_states=o_f_low_seq_hidden_states if self.seq_encoder is not None else None,
                                                               is_target=True)

            if m_low_states is None:
                m_low_states = torch.zeros((batch, *o_m_low_states.shape[1:]), device=self.device)
                m_low_target_states = torch.zeros((batch, *o_m_low_target_states.shape[1:]), device=self.device)
            m_low_states[mask] = o_m_low_states
            m_low_target_states[mask] = o_m_low_target_states

            option.train_rep_q(next_n_terminations=next_n_terminations[mask],

                               next_n_v_over_options_list=[next_n_v_over_options[mask] for next_n_v_over_options in next_n_v_over_options_list],

                               bn_indexes=bn_indexes[mask],
                               bn_padding_masks=bn_padding_masks[mask],
                               bn_obses_list=[o_m_low_obses[:, :-1, ...]
                                              for o_m_low_obses in o_m_low_obses_list],
                               bn_target_obses_list=[o_m_low_target_obses[:, :-1, ...]
                                                     for o_m_low_target_obses in o_m_low_target_obses_list],
                               bn_states=o_m_low_states[:, :-1, ...],
                               bn_target_states=o_m_low_target_states[:, :-1, ...],
                               bn_actions=bn_actions[mask],
                               bn_rewards=bn_rewards[mask],
                               next_obs_list=[o_m_low_obses[:, -1, ...]
                                              for o_m_low_obses in o_m_low_obses_list],
                               next_target_obs_list=[o_m_low_target_obses[:, -1, ...]
                                                     for o_m_low_target_obses in o_m_low_target_obses_list],
                               next_state=o_m_low_states[:, -1, ...],
                               next_target_state=o_m_low_target_states[:, -1, ...],
                               bn_dones=bn_dones[mask],
                               bn_mu_probs=bn_mu_probs[mask] if self.use_n_step_is else None,
                               priority_is=priority_is[mask] if self.use_replay_buffer and self.use_priority else None)

        if self.optimizer_rep:
            self.optimizer_rep.step()

        return m_low_states, m_low_target_states

    def _train_v_terminations(self,
                              state: torch.Tensor,
                              option_index: torch.Tensor,
                              last_option_index: torch.Tensor,
                              low_obs_list: List[torch.Tensor],
                              low_state: torch.Tensor,
                              next_state: torch.Tensor,
                              done: torch.Tensor):
        """
        Args:
            state: [Batch, state_size]
            option_index (torch.int64): [Batch, ]
            last_option_index (torch.int64): [Batch, ]
            low_obs_list: list([Batch, *low_obs_shapes_i], ...)
            low_state: [Batch, low_state_size]
            next_state: [Batch, state_size]
            done (torch.bool): [Batch, ]

        Returns:
            loss_v: torch.float32
            loss_termination: torch.float32
        """

        batch = state.shape[0]

        y_for_v = torch.zeros((batch, 1), device=self.device)

        for i, option in enumerate(self.option_list):
            mask = (option_index == i)
            if not torch.any(mask):
                continue

            o_state = low_state[mask]
            o_low_obs_list = [low_obs[mask] for low_obs in low_obs_list]

            y_for_v[mask] = option.get_v(obs_list=o_low_obs_list,
                                         state=o_state)

        loss_mse = nn.MSELoss()
        for i, model_v_over_options in enumerate(self.model_v_over_options_list):
            v_over_options = model_v_over_options(state)  # [Batch, num_options]
            v = v_over_options.gather(1, option_index.unsqueeze(-1))  # [Batch, 1]

            loss_v = loss_mse(v, y_for_v)

            optimizer = self.optimizer_v_list[i]

            optimizer.zero_grad()
            loss_v.backward()
            optimizer.step()

        next_termination_over_options = self.model_termination_over_options(next_state)  # [Batch, num_options]
        next_termination = next_termination_over_options.gather(1, last_option_index.unsqueeze(-1))  # [Batch, 1]
        with torch.no_grad():
            next_v_over_options = self.model_target_v_over_options_list[0](next_state)  # [Batch, num_options]
            next_v = next_v_over_options.gather(1, last_option_index.unsqueeze(-1))  # [Batch, 1]
        loss_termination = next_termination * (next_v - next_v_over_options.max(-1, keepdims=True)[0] + 0.01) * ~done
        loss_termination = torch.mean(loss_termination)

        self.optimizer_termination.zero_grad()
        loss_termination.backward()
        self.optimizer_termination.step()

        return loss_v, loss_termination

    def _train(self,
               bn_indexes: torch.Tensor,
               bn_padding_masks: torch.Tensor,
               bn_obses_list: List[torch.Tensor],
               bn_option_indexes: torch.Tensor,
               bn_actions: torch.Tensor,
               bn_rewards: torch.Tensor,
               next_obs_list: List[torch.Tensor],
               bn_dones: torch.Tensor,
               bn_mu_probs: torch.Tensor = None,
               f_seq_hidden_states: torch.Tensor = None,
               f_low_seq_hidden_states: torch.Tensor = None,
               priority_is: torch.Tensor = None):
        """
        Args:
            bn_indexes (torch.int32): [Batch, b + n]
            bn_padding_masks (torch.bool): [Batch, b + n]
            bn_obses_list: list([Batch, b + n, *obs_shapes_i], ...)
            bn_option_indexes: [Batch, b + n]
            bn_actions: [Batch, b + n, action_size]
            bn_rewards: [Batch, b + n]
            next_obs_list: list([Batch, *obs_shapes_i], ...)
            bn_dones (torch.bool): [Batch, b + n]
            bn_mu_probs: [Batch, b + n]
            f_seq_hidden_states: [Batch, 1, *seq_hidden_state_shape]
            f_low_seq_hidden_states: [Batch, 1, *low_seq_hidden_state_shape]
            priority_is: [Batch, 1]
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
                                        is_target=False)

        m_target_states, _ = self.get_l_states(l_indexes=m_indexes,
                                               l_padding_masks=m_padding_masks,
                                               l_obses_list=m_obses_list,
                                               l_pre_actions=m_pre_actions,
                                               f_seq_hidden_states=f_seq_hidden_states,
                                               is_target=True)

        next_n_states = m_target_states[:, self.burn_in_step + 1:, ...]  # [Batch, n, state_size]
        n_option_indexes = bn_option_indexes[:, self.burn_in_step:]  # [Batch, n]
        option_index = n_option_indexes[:, 0]  # [Batch, ]

        next_n_terminations_over_options = self.model_termination_over_options(next_n_states)  # [Batch, n, num_options]
        next_n_terminations = next_n_terminations_over_options.gather(2, n_option_indexes.unsqueeze(-1))  # [Batch, n, 1]
        next_n_terminations = next_n_terminations.squeeze(-1)  # [Batch, n]

        next_n_v_over_options_list = [v(next_n_states) for v in self.model_target_v_over_options_list]  # [Batch, n, num_options]

        m_low_obses_list = self.get_l_low_obses_list(l_obses_list=m_obses_list,
                                                     l_states=m_states)
        m_low_target_obses_list = self.get_l_low_obses_list(l_obses_list=m_obses_list,
                                                            l_states=m_target_states)

        (m_low_states,
         m_low_target_states) = self._train_rep_q(next_n_terminations=next_n_terminations,
                                                  next_n_v_over_options_list=next_n_v_over_options_list,

                                                  bn_indexes=bn_indexes,
                                                  bn_padding_masks=bn_padding_masks,
                                                  bn_option_indexes=bn_option_indexes,
                                                  bn_low_obses_list=[m_low_obses[:, :-1, ...] for m_low_obses in m_low_obses_list],
                                                  bn_low_target_obses_list=[m_low_target_obses[:, :-1, ...] for m_low_target_obses in m_low_target_obses_list],
                                                  bn_actions=bn_actions,
                                                  bn_rewards=bn_rewards,
                                                  next_low_obs_list=[m_low_obses[:, -1, ...] for m_low_obses in m_low_obses_list],
                                                  next_low_target_obs_list=[m_low_target_obses[:, -1, ...] for m_low_target_obses in m_low_target_obses_list],
                                                  bn_dones=bn_dones,
                                                  bn_mu_probs=bn_mu_probs,
                                                  f_low_seq_hidden_states=f_low_seq_hidden_states,
                                                  priority_is=priority_is)

        with torch.no_grad():
            m_states, _ = self.get_l_states(l_indexes=m_indexes,
                                            l_padding_masks=m_padding_masks,
                                            l_obses_list=m_obses_list,
                                            l_pre_actions=m_pre_actions,
                                            f_seq_hidden_states=f_seq_hidden_states if self.seq_encoder is not None else None,
                                            is_target=False)

        m_low_obses_list = self.get_l_low_obses_list(l_obses_list=m_obses_list,
                                                     l_states=m_states)

        for i, option in enumerate(self.option_list):
            mask = (option_index == i)
            if not torch.any(mask):
                continue

            (o_m_indexes,
             o_m_padding_masks,
             o_m_low_obses_list,
             o_m_pre_actions) = option.get_m_data(bn_indexes=bn_indexes[mask],
                                                  bn_padding_masks=bn_padding_masks[mask],
                                                  bn_obses_list=[m_low_obses[mask, :-1, ...] for m_low_obses in m_low_obses_list],
                                                  bn_actions=bn_actions[mask],
                                                  next_obs_list=[m_obses[mask, -1, ...] for m_obses in m_low_obses_list])

            o_bn_actions = bn_actions[mask]
            if self.seq_encoder is not None:
                o_f_low_seq_hidden_states = f_low_seq_hidden_states[mask]

            o_m_low_states, _ = option.get_l_states(l_indexes=o_m_indexes,
                                                    l_padding_masks=o_m_padding_masks,
                                                    l_obses_list=o_m_low_obses_list,
                                                    l_pre_actions=o_m_pre_actions,
                                                    f_seq_hidden_states=o_f_low_seq_hidden_states if self.seq_encoder is not None else None,
                                                    is_target=False)
            m_low_states[mask] = o_m_low_states

            option.train_policy_alpha(bn_obses_list=[o_m_obses[:, :-1, ...]
                                                     for o_m_obses in o_m_low_obses_list],
                                      bn_states=o_m_low_states[:, :-1, ...],
                                      bn_actions=o_bn_actions)

        loss_v, loss_termination = self._train_v_terminations(state=m_states[:, self.burn_in_step, ...],
                                                              option_index=bn_option_indexes[:, self.burn_in_step],
                                                              last_option_index=bn_option_indexes[:, -1],
                                                              low_obs_list=[m_low_obses[:, self.burn_in_step, ...]
                                                                            for m_low_obses in m_low_obses_list],
                                                              low_state=m_low_states[:, self.burn_in_step, ...],
                                                              next_state=m_states[:, -1, ...],
                                                              done=bn_dones[:, self.burn_in_step])

        if self.summary_writer is not None and self.global_step % self.write_summary_per_step == 0:
            self.summary_available = True

            self.summary_writer.add_scalar('loss/v', loss_v, self.global_step)
            self.summary_writer.add_scalar('loss/loss_termination', loss_termination, self.global_step)

            self.summary_writer.flush()

        return m_target_states, m_low_target_obses_list, m_low_target_states

    def put_episode(self, **episode_trans):
        # Ignore episodes which length is too short
        if episode_trans['l_indexes'].shape[1] < self.burn_in_step + self.n_step:
            return

        if self.use_replay_buffer:
            self._fill_replay_buffer(**episode_trans)
        else:
            self.batch_buffer.put_episode(**episode_trans)

    def _fill_replay_buffer(self,
                            l_indexes: np.ndarray,
                            l_padding_masks: np.ndarray,
                            l_obses_list: List[np.ndarray],
                            l_option_indexes: np.ndarray,
                            l_actions: np.ndarray,
                            l_rewards: np.ndarray,
                            next_obs_list: List[np.ndarray],
                            l_dones: np.ndarray,
                            l_probs: List[np.ndarray],
                            l_seq_hidden_states: Optional[np.ndarray] = None,
                            l_low_seq_hidden_states: Optional[np.ndarray] = None):
        """
        Args:
            l_indexes (np.int32): [1, episode_len]
            l_padding_masks (bool): [1, episode_len]
            l_obses_list (np): list([1, episode_len, *obs_shapes_i], ...)
            l_option_indexes (np.int8): [1, episode_len]
            l_actions (np): [1, episode_len, action_size]
            l_rewards (np): [1, episode_len]
            next_obs_list (np): list([1, *obs_shapes_i], ...)
            l_dones (bool): [1, episode_len]
            l_probs (np): [1, episode_len]
            l_seq_hidden_states (np): [1, episode_len, *seq_hidden_state_shape]
            l_low_seq_hidden_states (np): [1, episode_len, *low_seq_hidden_state_shape]
        """
        # Reshape [1, episode_len, ...] to [episode_len, ...]
        index = l_indexes.squeeze(0)
        padding_mask = l_padding_masks.squeeze(0)
        obs_list = [l_obses.squeeze(0) for l_obses in l_obses_list]
        if self.use_normalization:
            self._udpate_normalizer([torch.from_numpy(obs).to(self.device) for obs in obs_list])
        action = l_actions.squeeze(0)
        option_index = l_option_indexes.squeeze(0)
        reward = l_rewards.squeeze(0)
        done = l_dones.squeeze(0)

        # Padding next_obs for episode experience replay
        index = np.concatenate([index,
                                index[-1:] + 1])
        padding_mask = np.concatenate([padding_mask,
                                       np.zeros([1], dtype=bool)])
        obs_list = [np.concatenate([obs, next_obs]) for obs, next_obs in zip(obs_list, next_obs_list)]
        action = np.concatenate([action,
                                 np.zeros([1, action.shape[-1]], dtype=np.float32)])
        option_index = np.concatenate([option_index,
                                       np.full([1], -1, dtype=np.int8)])
        reward = np.concatenate([reward,
                                 np.zeros([1], dtype=np.float32)])
        done = np.concatenate([done,
                               np.zeros([1], dtype=bool)])

        storage_data = {
            'index': index,
            'padding_mask': padding_mask,
            **{f'obs_{i}': obs for i, obs in enumerate(obs_list)},
            'option_index': option_index,
            'action': action,
            'reward': reward,
            'done': done,
        }

        if self.use_n_step_is:
            l_mu_probs = l_probs
            mu_prob = l_mu_probs.squeeze(0)
            mu_prob = np.concatenate([mu_prob,
                                      np.zeros([1], dtype=np.float32)])
            storage_data['mu_prob'] = mu_prob

        if self.seq_encoder is not None:
            seq_hidden_state = l_seq_hidden_states.squeeze(0)
            seq_hidden_state = np.concatenate([seq_hidden_state,
                                               np.zeros([1, *seq_hidden_state.shape[1:]], dtype=np.float32)])
            storage_data['seq_hidden_state'] = seq_hidden_state

            low_seq_hidden_state = l_low_seq_hidden_states.squeeze(0)
            low_seq_hidden_state = np.concatenate([low_seq_hidden_state,
                                                   np.zeros([1, *low_seq_hidden_state.shape[1:]], dtype=np.float32)])
            storage_data['low_seq_hidden_state'] = low_seq_hidden_state

        # n_step transitions except the first one and the last obs_, n_step - 1 + 1
        if self.use_add_with_td:
            td_error = self.get_episode_td_error(l_indexes=l_indexes,
                                                 l_padding_masks=l_padding_masks,
                                                 l_obses_list=l_obses_list,
                                                 l_option_indexes=l_option_indexes,
                                                 l_actions=l_actions,
                                                 l_rewards=l_rewards,
                                                 next_obs_list=next_obs_list,
                                                 l_dones=l_dones,
                                                 l_mu_probs=l_mu_probs if self.use_n_step_is else None,
                                                 l_seq_hidden_states=l_seq_hidden_states if self.seq_encoder is not None else None,
                                                 l_low_seq_hidden_states=l_low_seq_hidden_states if self.seq_encoder is not None else None)
            self.replay_buffer.add_with_td_error(td_error, storage_data,
                                                 ignore_size=self.burn_in_step + self.n_step)
        else:
            self.replay_buffer.add(storage_data,
                                   ignore_size=self.burn_in_step + self.n_step)

        if self.seq_encoder == SEQ_ENCODER.ATTN and self.summary_writer is not None and self.summary_available:
            self.summary_available = False
            with torch.no_grad():
                l_indexes = l_indexes[:, self.burn_in_step:]
                l_obses_list = [o[:, self.burn_in_step:] for o in l_obses_list]
                l_pre_l_actions = gen_pre_n_actions(l_actions[:, self.burn_in_step:])
                l_padding_masks = l_padding_masks[:, self.burn_in_step:]
                * _, attn_weights_list = self.model_rep(torch.from_numpy(l_indexes).to(self.device),
                                                        [torch.from_numpy(o).to(self.device) for o in l_obses_list],
                                                        torch.from_numpy(l_pre_l_actions).to(self.device),
                                                        query_length=l_indexes.shape[1],
                                                        padding_mask=torch.from_numpy(l_padding_masks).to(self.device))

                for i, attn_weight in enumerate(attn_weights_list):
                    image = plot_attn_weight(attn_weight[0].cpu().numpy())
                    self.summary_writer.add_images(f'attn_weight/{i}', image, self.global_step)

    def _sample_from_replay_buffer(self):
        """
        Sample from replay buffer

        Returns:
            pointers: [Batch, ]
            (
                bn_indexes (np.int32): [Batch, b + n]
                bn_padding_masks (bool): [Batch, b + n]
                bn_obses_list (np): list([Batch, b + n, *obs_shapes_i], ...)
                bn_option_indexes (np.int8): [Batch, b + n]
                bn_actions (np): [Batch, b + n, action_size]
                bn_rewards (np): [Batch, b + n]
                next_obs_list (np): list([Batch, *obs_shapes_i], ...)
                bn_dones (np): [Batch, b + n]
                bn_mu_probs (np): [Batch, b + n]
                bn_seq_hidden_states (np): [Batch, b + n, *seq_hidden_state_shape],
                bn_low_seq_hidden_states (np): [Batch, b + n, *low_seq_hidden_state_shape],
                priority_is (np): [Batch, 1]
            )
        """
        sampled = self.replay_buffer.sample()
        if sampled is None:
            return None

        """
        trans:
            index (np.int32): [Batch, ]
            padding_mask (bool): [Batch, ]
            obs_i: [Batch, *obs_shapes_i]
            option_index (np.int8): [Batch, ]
            action: [Batch, action_size]
            reward: [Batch, ]
            done (bool): [Batch, ]
            mu_prob: [Batch, ]
            seq_hidden_state: [Batch, *seq_hidden_state_shape]
            low_seq_hidden_state: [Batch, *low_seq_hidden_state_shape]
        """
        pointers, trans, priority_is = sampled

        # Get n_step transitions TODO: could be faster, no need get all data
        trans = {k: [v] for k, v in trans.items()}
        # k: [v, v, ...]
        for i in range(1, self.burn_in_step + self.n_step + 1):
            t_trans = self.replay_buffer.get_storage_data(pointers + i).items()
            for k, v in t_trans:
                trans[k].append(v)

        for k, v in trans.items():
            trans[k] = np.concatenate([np.expand_dims(t, 1) for t in v], axis=1)

        """
        m_indexes (np.int32): [Batch, N + 1]
        m_padding_masks (bool): [Batch, N + 1]
        m_obses_list: list([Batch, N + 1, *obs_shapes_i], ...)
        m_option_indexes (np.int8): [Batch, N + 1]
        m_actions: [Batch, N + 1, action_size]
        m_rewards: [Batch, N + 1]
        m_dones (bool): [Batch, N + 1]
        m_mu_probs: [Batch, N + 1]
        m_seq_hidden_state: [Batch, N + 1, *seq_hidden_state_shape]
        m_low_seq_hidden_states: [Batch, N + 1, *low_seq_hidden_state_shape]
        """
        m_indexes = trans['index']
        m_padding_masks = trans['padding_mask']
        m_obses_list = [trans[f'obs_{i}'] for i in range(len(self.obs_shapes))]
        m_option_indexes = trans['option_index']
        m_actions = trans['action']
        m_rewards = trans['reward']
        m_dones = trans['done']

        bn_indexes = m_indexes[:, :-1]
        bn_padding_masks = m_padding_masks[:, :-1]
        bn_obses_list = [m_obses[:, :-1, ...] for m_obses in m_obses_list]
        bn_option_indexes = m_option_indexes[:, :-1]
        bn_actions = m_actions[:, :-1, ...]
        bn_rewards = m_rewards[:, :-1]
        next_obs_list = [m_obses[:, -1, ...] for m_obses in m_obses_list]
        bn_dones = m_dones[:, :-1]

        if self.use_n_step_is:
            m_mu_probs = trans['mu_prob']
            bn_mu_probs = m_mu_probs[:, :-1]

        if self.seq_encoder is not None:
            m_seq_hidden_states = trans['seq_hidden_state']
            bn_seq_hidden_states = m_seq_hidden_states[:, :-1, ...]

            m_low_seq_hidden_states = trans['low_seq_hidden_state']
            bn_low_seq_hidden_states = m_low_seq_hidden_states[:, :-1, ...]

        return pointers, (bn_indexes,
                          bn_padding_masks,
                          bn_obses_list,
                          bn_option_indexes,
                          bn_actions,
                          bn_rewards,
                          next_obs_list,
                          bn_dones,
                          bn_mu_probs if self.use_n_step_is else None,
                          bn_seq_hidden_states if self.seq_encoder is not None else None,
                          bn_low_seq_hidden_states if self.seq_encoder is not None else None,
                          priority_is if self.use_replay_buffer and self.use_priority else None)

    def train(self):
        step = self.get_global_step()

        if self.use_replay_buffer:
            train_data = self._sample_from_replay_buffer()
            if train_data is None:
                return step

            pointers, batch = train_data
            batch_list = [batch]
        else:
            batch_list = self.batch_buffer.get_batch()
            batch_list = [(*batch, None) for batch in batch_list]

        for batch in batch_list:
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
            bn_indexes (np.int32): [Batch, b + n]
            bn_padding_masks (bool): [Batch, b + n]
            bn_obses_list: list([Batch, b + n, *obs_shapes_i], ...)
            bn_option_indexes (np.int8): [Batch, b + n]
            bn_actions: [Batch, b + n, action_size]
            bn_rewards: [Batch, b + n]
            next_obs_list: list([Batch, *obs_shapes_i], ...)
            bn_dones (bool): [Batch, b + n]
            bn_mu_probs: [Batch, b + n]
            bn_seq_hidden_states: [Batch, b + n, *seq_hidden_state_shape]
            bn_low_seq_hidden_states: [Batch, b + n, *low_seq_hidden_state_shape]
            priority_is: [Batch, 1]
            """
            bn_indexes = torch.from_numpy(bn_indexes).to(self.device)
            bn_padding_masks = torch.from_numpy(bn_padding_masks).to(self.device)
            bn_obses_list = [torch.from_numpy(t).to(self.device) for t in bn_obses_list]
            bn_option_indexes = torch.from_numpy(bn_option_indexes).type(torch.int64).to(self.device)
            bn_actions = torch.from_numpy(bn_actions).to(self.device)
            bn_rewards = torch.from_numpy(bn_rewards).to(self.device)
            next_obs_list = [torch.from_numpy(t).to(self.device) for t in next_obs_list]
            bn_dones = torch.from_numpy(bn_dones).to(self.device)
            if self.use_n_step_is:
                bn_mu_probs = torch.from_numpy(bn_mu_probs).to(self.device)
            if self.seq_encoder is not None:
                f_seq_hidden_states = bn_seq_hidden_states[:, :1]
                f_seq_hidden_states = torch.from_numpy(f_seq_hidden_states).to(self.device)

                f_low_seq_hidden_states = bn_low_seq_hidden_states[:, :1]
                f_low_seq_hidden_states = torch.from_numpy(f_low_seq_hidden_states).to(self.device)
            if self.use_replay_buffer and self.use_priority:
                priority_is = torch.from_numpy(priority_is).to(self.device)

            (m_target_states,
             m_low_target_obses_list,
             m_low_target_states) = self._train(
                bn_indexes=bn_indexes,
                bn_padding_masks=bn_padding_masks,
                bn_obses_list=bn_obses_list,
                bn_option_indexes=bn_option_indexes,
                bn_actions=bn_actions,
                bn_rewards=bn_rewards,
                next_obs_list=next_obs_list,
                bn_dones=bn_dones,
                bn_mu_probs=bn_mu_probs if self.use_n_step_is else None,
                f_seq_hidden_states=f_seq_hidden_states if self.seq_encoder is not None else None,
                f_low_seq_hidden_states=f_low_seq_hidden_states if self.seq_encoder is not None else None,
                priority_is=priority_is if self.use_replay_buffer and self.use_priority else None)

            if step % self.save_model_per_step == 0:
                self.save_model()

            if self.use_replay_buffer:
                bn_pre_actions = gen_pre_n_actions(bn_actions)  # [Batch, b + n, action_size]

                bn_states, next_bn_seq_hidden_states = self.get_l_states_with_seq_hidden_states(
                    l_indexes=bn_indexes,
                    l_padding_masks=bn_padding_masks,
                    l_obses_list=bn_obses_list,
                    l_pre_actions=bn_pre_actions,
                    f_seq_hidden_states=f_seq_hidden_states if self.seq_encoder is not None else None)

                bn_low_obses_list = self.get_l_low_obses_list(l_obses_list=bn_obses_list,
                                                              l_states=bn_states)

                (bn_low_states,
                 next_bn_low_seq_hidden_states) = self.get_l_low_states_with_seq_hidden_states(
                    l_indexes=bn_indexes,
                    l_padding_masks=bn_padding_masks,
                    l_low_obses_list=bn_low_obses_list,
                    l_option_indexes=bn_option_indexes,
                    l_pre_actions=bn_pre_actions,
                    f_low_seq_hidden_states=f_low_seq_hidden_states if self.seq_encoder is not None else None,
                    is_target=False)

                if self.use_n_step_is:
                    bn_pi_probs_tensor = self.get_l_probs(l_low_obses_list=bn_low_obses_list,
                                                          l_low_states=bn_low_states,
                                                          l_option_indexes=bn_option_indexes,
                                                          l_actions=bn_actions)

                # Update td_error
                if self.use_priority:
                    td_error = self._get_td_error(
                        bn_states=bn_states,
                        bn_target_states=m_target_states[:, :-1, ...],
                        bn_option_indexes=bn_option_indexes,
                        bn_low_obses_list=bn_low_obses_list,
                        bn_low_states=bn_low_states,
                        bn_low_target_states=m_low_target_states[:, :-1, ...],
                        bn_actions=bn_actions,
                        bn_rewards=bn_rewards,
                        next_target_state=m_target_states[:, -1, ...],
                        next_low_obs_list=[m_low_target_obses[:, -1, ...]
                                           for m_low_target_obses in m_low_target_obses_list],
                        next_low_target_state=m_low_target_states[:, -1, ...],
                        bn_dones=bn_dones,
                        bn_mu_probs=bn_mu_probs if self.use_n_step_is else None).detach().cpu().numpy()
                    self.replay_buffer.update(pointers, td_error)

                # Update seq_hidden_states
                if self.seq_encoder is not None:
                    pointers_list = [pointers + i for i in range(1, self.burn_in_step + self.n_step + 1)]
                    tmp_pointers = np.stack(pointers_list, axis=1).reshape(-1)

                    next_bn_seq_hidden_states = next_bn_seq_hidden_states.detach().cpu().numpy()
                    bn_seq_hidden_states[:, 1:, ...] = next_bn_seq_hidden_states[:, :-1, ...]
                    seq_hidden_state = bn_seq_hidden_states.reshape(-1, *bn_seq_hidden_states.shape[2:])
                    self.replay_buffer.update_transitions(tmp_pointers, 'seq_hidden_state', seq_hidden_state)

                    next_bn_low_seq_hidden_states = next_bn_low_seq_hidden_states.detach().cpu().numpy()
                    bn_low_seq_hidden_states[:, 1:, ...] = next_bn_low_seq_hidden_states[:, :-1, ...]
                    low_seq_hidden_state = bn_low_seq_hidden_states.reshape(-1, *bn_low_seq_hidden_states.shape[2:])
                    self.replay_buffer.update_transitions(tmp_pointers, 'low_seq_hidden_state', low_seq_hidden_state)

                # Update n_mu_probs
                if self.use_n_step_is:
                    pointers_list = [pointers + i for i in range(0, self.burn_in_step + self.n_step)]
                    tmp_pointers = np.stack(pointers_list, axis=1).reshape(-1)
                    pi_probs = bn_pi_probs_tensor.detach().cpu().numpy().reshape(-1)
                    self.replay_buffer.update_transitions(tmp_pointers, 'mu_prob', pi_probs)

            step = self._increase_global_step()
            for option in self.option_list:
                option._increase_global_step()

        return step

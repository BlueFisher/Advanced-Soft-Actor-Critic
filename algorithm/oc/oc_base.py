from collections import defaultdict
from itertools import chain
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn, optim

from ..nn_models import *
from ..sac_base import SAC_Base
from ..utils import *

NUM_OPTIONS = 3


class OC_Base(SAC_Base):
    def _build_model(self, nn, nn_config: Optional[dict], init_log_alpha: float, learning_rate: float):
        """
        Initialize variables, network models and optimizers
        """
        if nn_config is None:
            nn_config = {}
        nn_config = defaultdict(dict, nn_config)
        if nn_config['rep'] is None:
            nn_config['rep'] = {}
        if nn_config['policy'] is None:
            nn_config['policy'] = {}

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

        self.model_v_over_options_list = [ModelQOverOption(state_size, NUM_OPTIONS).to(self.device) for _ in range(self.ensemble_q_num)]
        self.model_target_v_over_options_list = [ModelQOverOption(state_size, NUM_OPTIONS).to(self.device) for _ in range(self.ensemble_q_num)]
        for model_target_v_over_options in self.model_target_v_over_options_list:
            for param in model_target_v_over_options.parameters():
                param.requires_grad = False
        self.optimizer_v_list = [adam_optimizer(self.model_v_over_options_list[i].parameters()) for i in range(self.ensemble_q_num)]

        self.model_termination_over_options = ModelTerminationOverOption(state_size, NUM_OPTIONS).to(self.device)
        self.model_q_list = [nn.ModelQ(state_size, self.d_action_size, self.c_action_size,
                                       False, self.train_mode).to(self.device) for _ in range(NUM_OPTIONS)]
        self.model_target_q_list = [nn.ModelQ(state_size, self.d_action_size, self.c_action_size,
                                              True, self.train_mode).to(self.device) for _ in range(NUM_OPTIONS)]
        self.optimizer_q_list = [adam_optimizer(m.parameters()) for m in self.model_q_list]
        self.model_option_list = [nn.ModelPolicy(state_size, self.d_action_size, self.c_action_size,
                                                 self.train_mode,
                                                 self.model_abs_dir,
                                                 **nn_config['policy']).to(self.device) for _ in range(NUM_OPTIONS)]

        self.optimizer_option = adam_optimizer(chain(self.model_termination_over_options.parameters(),
                                                     *[m.parameters() for m in self.model_option_list]))

        """ ALPHA """
        self.log_d_alpha_list = [torch.tensor(init_log_alpha, dtype=torch.float32, requires_grad=True, device=self.device) for _ in range(NUM_OPTIONS)]
        self.log_c_alpha_list = [torch.tensor(init_log_alpha, dtype=torch.float32, requires_grad=True, device=self.device) for _ in range(NUM_OPTIONS)]

        if self.use_auto_alpha:
            self.optimizer_alpha = adam_optimizer(self.log_d_alpha_list + self.log_c_alpha_list)

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

        for i, option in enumerate(self.model_option_list):
            ckpt_dict[f'model_option_{i}'] = option

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

    def get_initial_option_index(self, batch_size: int):
        return np.full([batch_size, ], -1, dtype=np.int8)

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

        for i in range(len(self.model_q_list)):
            target = chain(target, self.model_target_q_list[i].parameters())
            source = chain(source, self.model_q_list[i].parameters())

        for target_param, param in zip(target, source):
            target_param.data.copy_(
                target_param.data * (1. - tau) + param.data * tau
            )

    @torch.no_grad()
    def get_l_probs(self,
                    l_indexes: torch.Tensor,
                    l_padding_masks: torch.Tensor,
                    l_obses_list: List[torch.Tensor],
                    l_option_indexes: torch.Tensor,
                    l_actions: torch.Tensor,
                    f_seq_hidden_states: torch.Tensor = None):
        """
        Args:
            l_indexes: [Batch, l]
            l_padding_masks: [Batch, l]
            l_obses_list: list([Batch, l, *obs_shapes_i], ...)
            l_option_indexes: [Batch, l]
            l_actions: [Batch, l, action_size]
            f_seq_hidden_states: [Batch, 1, *seq_hidden_state_shape]

        Returns:
            l_probs: [Batch, l]
        """
        if self.seq_encoder == SEQ_ENCODER.RNN:
            l_states, _ = self.model_rep(l_obses_list,
                                         gen_pre_n_actions(l_actions),
                                         f_seq_hidden_states[:, 0])
        elif self.seq_encoder == SEQ_ENCODER.ATTN:
            l_states, *_ = self.model_rep(l_indexes,
                                          l_obses_list,
                                          gen_pre_n_actions(l_actions),
                                          query_length=l_indexes.shape[1],
                                          hidden_state=f_seq_hidden_states,
                                          is_prev_hidden_state=True,
                                          padding_mask=l_padding_masks)
        else:
            l_states = self.model_rep(l_obses_list)
        #  l_states: [Batch, l, state_size]

        batch, l, *_ = l_states.shape
        flat_l_states = l_states.reshape(-1, self.state_size)
        flat_l_option_indexes = l_option_indexes.reshape(-1)
        flat_l_actions = l_actions.reshape(-1, self.d_action_size + self.c_action_size)  # [Batch * l, action_size]

        flat_l_policy_prob = torch.ones((flat_l_states.shape[0]), device=self.device)  # [Batch * l]
        for i, model_option in enumerate(self.model_option_list):
            mask = (flat_l_option_indexes == i)
            if not torch.any(mask):
                continue

            d_policy, c_policy = model_option(flat_l_states[mask], None)

            policy_prob = torch.ones_like(flat_l_policy_prob[mask])

            if self.d_action_size:
                n_selected_d_actions = flat_l_actions[mask][..., :self.d_action_size]
                policy_prob *= torch.exp(d_policy.log_prob(n_selected_d_actions))  # [Batch * l]

            if self.c_action_size:
                l_selected_c_actions = flat_l_actions[mask][..., self.d_action_size:]
                c_policy_prob = squash_correction_prob(c_policy, torch.atanh(l_selected_c_actions))
                # [Batch, l, c_action_size]
                policy_prob *= torch.prod(c_policy_prob, dim=-1)  # [Batch * l]

            flat_l_policy_prob[mask] = policy_prob

        return flat_l_policy_prob.reshape(batch, l)

    @torch.no_grad()
    def _get_y(self,
               n_obses_list: List[torch.Tensor],
               n_states: torch.Tensor,
               n_option_indexes: torch.Tensor,
               n_actions: torch.Tensor,
               n_rewards: torch.Tensor,
               next_obs_list: List[torch.Tensor],
               next_state: torch.Tensor,
               n_dones: torch.Tensor,
               n_mu_probs: torch.Tensor = None):
        """
        Args:
            n_obses_list: list([Batch, n, *obs_shapes_i], ...)
            n_states: [Batch, n, state_size]
            n_option_indexes: [Batch, n]
            n_actions: [Batch, n, action_size]
            n_rewards: [Batch, n]
            state_: [Batch, state_size]
            n_dones: [Batch, n], dtype=torch.bool
            n_mu_probs: [Batch, n]

        Returns:
            y: [Batch, 1]
        """

        next_n_states = torch.cat([n_states[:, 1:, ...], next_state.unsqueeze(1)], dim=1)  # [Batch, n, state_size]

        next_n_termination_over_options = self.model_termination_over_options(next_n_states)  # [Batch, n, num_options]
        next_n_termination = next_n_termination_over_options.gather(2, n_option_indexes.unsqueeze(-1))  # [Batch, n, 1]
        next_n_termination = next_n_termination.squeeze(-1)  # [Batch, n]

        n_v_over_options_list = [v(n_states) for v in self.model_v_over_options_list]  # [Batch, n, num_options]
        n_v_list = [n_v_over_options.gather(2, n_option_indexes.unsqueeze(-1)) for n_v_over_options in n_v_over_options_list]  # [Batch, n, 1]
        n_v_list = [n_v.squeeze(-1) for n_v in n_v_list]  # [Batch, n]

        next_n_v_over_options_list = [v(next_n_states) for v in self.model_target_v_over_options_list]  # [Batch, n, num_options]
        next_n_v_list = [next_n_v_over_options.gather(2, n_option_indexes.unsqueeze(-1)) for next_n_v_over_options in next_n_v_over_options_list]  # [Batch, n, 1]
        next_n_v_list = [next_n_v.squeeze(-1) for next_n_v in next_n_v_list]  # [Batch, n]

        stacked_next_n_v_over_options = torch.stack(next_n_v_over_options_list)[torch.randperm(self.ensemble_q_num)[:self.ensemble_q_sample]]
        # [ensemble_q_num, Batch, n, num_options] -> [ensemble_q_sample, Batch, n, num_options]
        stacked_n_v = torch.stack(n_v_list)[torch.randperm(self.ensemble_q_num)[:self.ensemble_q_sample]]
        # [ensemble_q_num, Batch, n] -> [ensemble_q_sample, Batch, n]
        stacked_next_n_v = torch.stack(next_n_v_list)[torch.randperm(self.ensemble_q_num)[:self.ensemble_q_sample]]
        # [ensemble_q_num, Batch, n] -> [ensemble_q_sample, Batch, n]

        min_next_n_v_over_options, _ = stacked_next_n_v_over_options.min(dim=0)  # [Batch, n, num_options]
        min_n_v, _ = stacked_n_v.min(dim=0)  # [Batch, n]
        min_next_n_v, _ = stacked_next_n_v.min(dim=0)  # [Batch, n]

        if self.use_n_step_is:
            batch = n_states.shape[0]

            flat_n_states = n_states.reshape(-1, self.state_size)  # [Batch * n, state_size]
            flat_n_c_actions = n_actions[..., self.d_action_size:].reshape(-1, self.c_action_size)  # [Batch * n, c_action_size]
            flat_n_option_indexes = n_option_indexes.reshape(-1)

            flat_n_pi_probs = torch.ones(flat_n_states.shape[0], device=self.device)

            for i, model_option in enumerate(self.model_option_list):
                mask = (flat_n_option_indexes == i)
                if not torch.any(mask):
                    continue

                o_d_policy, o_c_policy = model_option(flat_n_states[mask], None)
                _flat_n_pi_probs = squash_correction_prob(o_c_policy, torch.atanh(flat_n_c_actions[mask]))  # [Batch * n, c_action_size]

                flat_n_pi_probs[mask] = _flat_n_pi_probs.prod(axis=-1)  # [Batch * n]

            n_pi_probs = flat_n_pi_probs.reshape(batch, self.n_step)

        td_error = n_rewards + self.gamma * ~n_dones * \
            ((1 - next_n_termination) * min_next_n_v +
             next_n_termination * min_next_n_v_over_options.max(-1)[0]) - min_n_v  # [Batch, n]

        td_error = self._gamma_ratio * td_error  # [Batch, n]

        if self.use_n_step_is:
            td_error = self._lambda_ratio * td_error

            n_step_is = n_pi_probs / n_mu_probs.clamp(min=1e-8)

            # \rho_t, t \in [s, s+n-1]
            rho = torch.minimum(n_step_is, torch.tensor(self.v_rho, device=self.device))  # [Batch, n]

            # \prod{c_i}, i \in [s, t-1]
            c = torch.minimum(n_step_is, torch.tensor(self.v_c, device=self.device))
            c = torch.cat([torch.ones((n_step_is.shape[0], 1), device=self.device), c[..., :-1]], dim=-1)
            c = torch.cumprod(c, dim=1)

            # \prod{c_i} * \rho_t * td_error
            td_error = c * rho * td_error

        r = torch.sum(td_error, dim=1, keepdim=True)  # [Batch, 1]

        y = min_n_v[:, 0:1] + r  # [Batch, 1]

        # next_termination = next_n_termination[:, -1:]
        # next_v_over_options = min_next_n_v_over_options[:, -1, ...]
        # next_v = min_next_n_v[:, -1:]

        # done = n_dones[:, -1:]
        # y = torch.sum(self._gamma_ratio * n_rewards, dim=-1, keepdim=True)
        # y = y + self.gamma ** self.n_step * ~done * \
        #     ((1 - next_termination) * next_v +
        #      next_termination * next_v_over_options.max(-1, keepdim=True)[0])

        return y

    def _choose_action(self,
                       state: torch.Tensor,
                       option_index: np.ndarray):
        v_over_options = self.model_v_over_options_list[0](state)  # [Batch, num_options]
        new_option_index = v_over_options.argmax(dim=-1)  # [Batch, ]

        none_option_mask = option_index == -1
        option_index[none_option_mask] = new_option_index[none_option_mask]

        termination_over_options = self.model_termination_over_options(state)  # [Batch, num_options]
        termination = termination_over_options.gather(1, option_index.unsqueeze(-1))  # [Batch, 1]
        termination = termination.squeeze(-1)
        termination_mask = termination > .5
        option_index[termination_mask] = new_option_index[termination_mask]

        batch = state.shape[0]
        policy_prob = torch.ones(batch, device=self.device)
        d_action = torch.zeros(batch, self.d_action_size, device=self.device)
        c_action = torch.zeros(batch, self.c_action_size, device=self.device)

        for i, model_option in enumerate(self.model_option_list):
            mask = (option_index == i)
            if not torch.any(mask):
                continue

            o_d_policy, o_c_policy = model_option(state[mask], None)
            if self.d_action_size:
                o_d_action = o_d_policy.sample()

                d_action[mask] = o_d_action
                policy_prob[mask] *= torch.exp(o_d_policy.log_prob(o_d_action))  # [Batch, ]

            if self.c_action_size:
                o_c_action = torch.tanh(o_c_policy.sample())
                c_action[mask] = o_c_action

                o_c_policy_prob = squash_correction_prob(o_c_policy, torch.atanh(o_c_action))
                policy_prob[mask] *= torch.prod(o_c_policy_prob, dim=-1)  # [Batch, ]

        return option_index, torch.cat([d_action, c_action], dim=-1), policy_prob

    def choose_action(self,
                      obs_list: List[np.ndarray],
                      pre_option_index: np.ndarray):
        """
        Args:
            obs_list: list([Batch, *obs_shapes_i], ...)
            pre_action: [Batch, d_action_size + c_action_size]
            rnn_state: [Batch, *seq_hidden_state_shape]
        Returns:
            action: [Batch, d_action_size + c_action_size] (numpy)
            rnn_state: [Batch, *seq_hidden_state_shape] (numpy)
        """
        obs_list = [torch.from_numpy(obs).to(self.device) for obs in obs_list]
        option_index = torch.from_numpy(pre_option_index).type(torch.int64).to(self.device)  # [Batch, ]
        option_index = torch.ones_like(option_index)

        state = self.model_rep(obs_list)

        option_index, action, prob = self._choose_action(state, option_index)

        return (option_index.detach().cpu().numpy(),
                action.detach().cpu().numpy(),
                prob.detach().cpu().numpy())

    def choose_rnn_action(self,
                          obs_list: List[np.ndarray],
                          pre_option_index: np.ndarray,
                          pre_action: np.ndarray,
                          rnn_state: np.ndarray,):
        obs_list = [torch.from_numpy(obs).to(self.device) for obs in obs_list]
        option_index = torch.from_numpy(pre_option_index).type(torch.int64).to(self.device)  # [Batch, ]
        pre_action = torch.from_numpy(pre_action).to(self.device)
        rnn_state = torch.from_numpy(rnn_state).to(self.device)

        obs_list = [obs.unsqueeze(1) for obs in obs_list]
        pre_action = pre_action.unsqueeze(1)
        state, next_rnn_state = self.model_rep(obs_list, pre_action, rnn_state)
        state = state.squeeze(1)
        obs_list = [obs.squeeze(1) for obs in obs_list]

        option_index, action, prob = self._choose_action(state, option_index)

        return (option_index.detach().cpu().numpy(),
                action.detach().cpu().numpy(),
                prob.detach().cpu().numpy(),
                next_rnn_state.detach().cpu().numpy())

    @torch.no_grad()
    def _get_td_error(self,
                      bn_indexes: torch.Tensor,
                      bn_padding_masks: torch.Tensor,
                      bn_obses_list: List[torch.Tensor],
                      bn_option_indexes: torch.Tensor,
                      bn_actions: torch.Tensor,
                      bn_rewards: torch.Tensor,
                      next_obs_list: List[torch.Tensor],
                      bn_dones: torch.Tensor,
                      bn_mu_probs: torch.Tensor = None,
                      f_seq_hidden_states: torch.Tensor = None):
        """
        Args:
            bn_indexes: [Batch, b + n]
            bn_padding_masks: [Batch, b + n]
            bn_obses_list: list([Batch, b + n, *obs_shapes_i], ...)
            bn_option_indexes: [Batch, b + n]
            bn_actions: [Batch, b + n, action_size]
            bn_rewards: [Batch, b + n]
            next_obs_list: list([Batch, *obs_shapes_i], ...)
            bn_dones: [Batch, b + n]
            bn_mu_probs: [Batch, b + n]
            f_seq_hidden_states: [Batch, 1, *seq_hidden_state_shape]

        Returns:
            The td-error of observations, [Batch, 1]
        """
        m_obses_list = [torch.cat([bn_obses, next_obs.unsqueeze(1)], dim=1)
                        for bn_obses, next_obs in zip(bn_obses_list, next_obs_list)]

        if self.seq_encoder == SEQ_ENCODER.RNN:
            rnn_state = f_seq_hidden_states[:, 0]
            tmp_states, _ = self.model_rep([m_obses[:, :self.burn_in_step + 1, ...] for m_obses in m_obses_list],
                                           gen_pre_n_actions(bn_actions[:, :self.burn_in_step + 1, ...]),
                                           rnn_state)
            state = tmp_states[:, self.burn_in_step, ...]
            m_target_states, *_ = self.model_target_rep(m_obses_list,
                                                        gen_pre_n_actions(bn_actions,
                                                                          keep_last_action=True),
                                                        rnn_state)
        else:
            state = self.model_rep([m_obses[:, self.burn_in_step, ...] for m_obses in m_obses_list])
            m_target_states = self.model_target_rep(m_obses_list)

        action = bn_actions[:, self.burn_in_step, ...]
        d_action = action[..., :self.d_action_size]
        c_action = action[..., self.d_action_size:]
        option_index = bn_option_indexes[:, self.burn_in_step]  # [Batch, ]

        y = self._get_y([m_obses[:, self.burn_in_step:-1, ...] for m_obses in m_obses_list],
                        m_target_states[:, self.burn_in_step:-1, ...],
                        bn_option_indexes[:, self.burn_in_step:, ...],
                        bn_actions[:, self.burn_in_step:, ...],
                        bn_rewards[:, self.burn_in_step:],
                        [m_obses[:, -1, ...] for m_obses in m_obses_list],
                        m_target_states[:, -1, ...],
                        bn_dones[:, self.burn_in_step:],
                        bn_mu_probs[:, self.burn_in_step:] if self.use_n_step_is else None)

        batch = state.shape[0]
        td_error = torch.zeros((batch, 1), device=self.device)

        for i, model_q in enumerate(self.model_q_list):
            mask = (option_index == i)
            if not torch.any(mask):
                continue

            o_state = state[mask]

            o_d_action = d_action[mask]
            o_c_action = c_action[mask]
            o_d_q, o_c_q = model_q(o_state, o_c_action)

            td_error[mask] = torch.abs(o_c_q - y[mask])

        return td_error

    def _train_policy(self,
                      obs_list: List[torch.Tensor],
                      state: torch.Tensor,
                      option_index: torch.Tensor,
                      last_option_index: torch.Tensor,
                      action: torch.Tensor,
                      done: torch.Tensor,
                      next_state: torch.Tensor):
        batch = state.shape[0]

        next_termination_over_options = self.model_termination_over_options(next_state)  # [Batch, num_options]
        next_termination = next_termination_over_options.gather(1, last_option_index.unsqueeze(-1))  # [Batch, 1]
        with torch.no_grad():
            next_v_over_options = self.model_target_v_over_options_list[0](next_state)  # [Batch, num_options]
            next_v = next_v_over_options.gather(1, last_option_index.unsqueeze(-1))  # [Batch, 1]
        termination_loss = next_termination * (next_v - next_v_over_options.max(-1, keepdims=True)[0] + 0.01) * ~done

        # d_policy_0, c_policy_0 = self.model_option_list[0](state, None)
        # d_policy_1, c_policy_1 = self.model_option_list[1](state, None)
        # loss_ent = distributions.kl_divergence(c_policy_0, c_policy_1)
        # loss_ent = -torch.mean(loss_ent, dim=-1, keepdim=True)

        policy_loss = torch.zeros((batch, 1), device=self.device)
        option_c_entropy_list = [None] * NUM_OPTIONS

        for i, model_option in enumerate(self.model_option_list):
            mask = (option_index == i)
            if not torch.any(mask):
                continue

            o_state = state[mask]
            o_d_policy, o_c_policy = model_option(o_state, None)

            with torch.no_grad():
                o_d_alpha = torch.exp(self.log_d_alpha_list[i])
                o_c_alpha = torch.exp(self.log_c_alpha_list[i])

            # if self.d_action_size:
            #     o_d_action = d_action[mask]  # [Batch, d_action_size]
            #     log_policy_prob[mask] *= o_d_policy.log_prob(o_d_action).unsqueeze(-1)  # [Batch, 1]

            if self.c_action_size:
                o_c_action_sampled = o_c_policy.rsample()
                o_c_log_prob = squash_correction_log_prob(o_c_policy, o_c_action_sampled)
                o_c_log_prob = torch.sum(o_c_log_prob, dim=-1, keepdim=True)  # [Batch, 1]
                o_c_q_for_gradient = self.model_q_list[i](o_state, torch.tanh(o_c_action_sampled))[1]
                policy_loss[mask] += o_c_alpha * o_c_log_prob - o_c_q_for_gradient

                option_c_entropy_list[i] = torch.mean(o_c_policy.entropy())

        self.optimizer_option.zero_grad()
        actor_loss = torch.mean(termination_loss + policy_loss)
        actor_loss.backward(inputs=list(chain(self.model_termination_over_options.parameters(),
                                              *[m.parameters() for m in self.model_option_list])))
        self.optimizer_option.step()

        return option_c_entropy_list

    def _train_alpha(self,
                     obs_list: torch.Tensor,
                     state: torch.Tensor,
                     option_index: torch.Tensor):
        batch = state.shape[0]

        loss_d_alpha = torch.zeros((batch, 1), device=self.device)
        loss_c_alpha = torch.zeros((batch, 1), device=self.device)

        for i, model_option in enumerate(self.model_option_list):
            mask = (option_index == i)
            if not torch.any(mask):
                continue

            o_state = state[mask]
            o_d_policy, o_c_policy = model_option(o_state, None)

            o_d_alpha = torch.exp(self.log_d_alpha_list[i])
            o_c_alpha = torch.exp(self.log_c_alpha_list[i])

            if self.c_action_size:
                o_action_sampled = o_c_policy.sample()
                log_prob = torch.sum(squash_correction_log_prob(o_c_policy, o_action_sampled), dim=1, keepdim=True)
                # [Batch, 1]

                loss_c_alpha[mask] = -o_c_alpha * (log_prob - self.c_action_size)  # [Batch, 1]

        loss_alpha = torch.mean(loss_d_alpha + loss_c_alpha)
        self.optimizer_alpha.zero_grad()
        loss_alpha.backward(inputs=self.log_d_alpha_list + self.log_c_alpha_list)
        self.optimizer_alpha.step()

        return [torch.exp(a) for a in self.log_d_alpha_list], [torch.exp(a) for a in self.log_c_alpha_list]

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
               priority_is: torch.Tensor = None):
        """
        Args:
            bn_indexes: [Batch, b + n], dtype=torch.int32
            bn_padding_masks: [Batch, b + n], dtype=torch.bool
            bn_obses_list: list([Batch, b + n, *obs_shapes_i], ...)
            bn_option_indexes: [Batch, b + n]
            bn_actions: [Batch, b + n, action_size]
            bn_rewards: [Batch, b + n]
            next_obs_list: list([Batch, *obs_shapes_i], ...)
            bn_dones: [Batch, b + n], dtype=torch.bool
            bn_mu_probs: [Batch, b + n]
            f_seq_hidden_states: [Batch, 1, *seq_hidden_state_shape]
            priority_is: [Batch, 1]
        """

        if self.global_step % self.update_target_per_step == 0:
            self._update_target_variables(tau=self.tau)

        m_obses_list = [torch.cat([n_obses, next_obs.unsqueeze(1)], dim=1)
                        for n_obses, next_obs in zip(bn_obses_list, next_obs_list)]

        if self.seq_encoder is None:
            m_states = self.model_rep(m_obses_list)
            m_target_states = self.model_target_rep(m_obses_list)
        else:
            m_pre_actions = gen_pre_n_actions(bn_actions, keep_last_action=True)

            if self.seq_encoder == SEQ_ENCODER.RNN:
                rnn_state = f_seq_hidden_states[:, 0]
                m_states, _ = self.model_rep(m_obses_list,
                                             m_pre_actions,
                                             rnn_state)
                m_target_states, _ = self.model_target_rep(m_obses_list,
                                                           m_pre_actions,
                                                           rnn_state)

        state = m_states[:, self.burn_in_step, ...]
        next_state = m_target_states[:, -1, ...]

        option_index = bn_option_indexes[:, self.burn_in_step]  # [Batch, ]
        last_option_index = bn_option_indexes[:, -1]  # [Batch, ]

        action = bn_actions[:, self.burn_in_step, ...]
        d_action = action[:, :self.d_action_size]
        c_action = action[:, self.d_action_size:]

        done = bn_dones[:, -1:]  # [Batch, 1]

        batch = state.shape[0]

        loss_none_mse = nn.MSELoss(reduction='none')
        mse = nn.MSELoss()

        # Q in option #
        q_loss_list = []

        if self.optimizer_rep:
            self.optimizer_rep.zero_grad()

        y = self._get_y([m_obses[:, self.burn_in_step:-1, ...] for m_obses in m_obses_list],
                        m_target_states[:, self.burn_in_step:-1, ...],
                        bn_option_indexes[:, self.burn_in_step:, ...],
                        bn_actions[:, self.burn_in_step:, ...],
                        bn_rewards[:, self.burn_in_step:],
                        [m_obses[:, -1, ...] for m_obses in m_obses_list],
                        m_target_states[:, -1, ...],
                        bn_dones[:, self.burn_in_step:],
                        bn_mu_probs[:, self.burn_in_step:] if self.use_n_step_is else None)

        for i, model_q in enumerate(self.model_q_list):
            mask = (option_index == i)
            if not torch.any(mask):
                continue

            o_state = state[mask]

            o_d_action = d_action[mask]
            o_c_action = c_action[mask]
            o_d_q, o_c_q = model_q(o_state, o_c_action)

            q_loss = loss_none_mse(o_c_q, y[mask]) * priority_is[mask]
            q_loss = torch.mean(q_loss)
            q_loss_list.append(q_loss)
            self.optimizer_q_list[i].zero_grad()
            q_loss.backward(retain_graph=True)
            self.optimizer_q_list[i].step()

        if self.optimizer_rep:
            self.optimizer_rep.step()

        # ACTOR & V #
        with torch.no_grad():
            if self.seq_encoder is None:
                m_states = self.model_rep(m_obses_list)
            else:
                m_pre_actions = gen_pre_n_actions(bn_actions, keep_last_action=True)

                if self.seq_encoder == SEQ_ENCODER.RNN:
                    rnn_state = f_seq_hidden_states[:, 0]
                    m_states, _ = self.model_rep(m_obses_list,
                                                 m_pre_actions,
                                                 rnn_state)

            state = m_states[:, self.burn_in_step, ...]
            next_state = next_state.detach()

        y_for_v = torch.zeros((batch, 1), device=self.device)

        for i, model_target_q in enumerate(self.model_target_q_list):
            mask = (option_index == i)
            if not torch.any(mask):
                continue

            o_state = state[mask]

            with torch.no_grad():
                model_option = self.model_option_list[i]
                o_d_policy, o_c_policy = model_option(o_state, None)
                o_c_action_sampled = o_c_policy.rsample()  # [Batch, c_action_size]
                o_c_action_log_prob = squash_correction_log_prob(o_c_policy, o_c_action_sampled)  # [Batch, c_action_size]
                o_c_action_log_prob = torch.sum(o_c_action_log_prob, dim=-1, keepdim=True)  # [Batch, 1]
                o_d_q, o_c_q = model_target_q(o_state, torch.tanh(o_c_action_sampled))

                y_for_v[mask] = o_c_q - torch.exp(self.log_c_alpha_list[i]) * o_c_action_log_prob

        for i, model_v_over_options in enumerate(self.model_v_over_options_list):
            v_over_options = model_v_over_options(state)  # [Batch, num_options]
            v = v_over_options.gather(1, option_index.unsqueeze(-1))  # [Batch, 1]

            v_loss = mse(v, y_for_v)

            optimizer = self.optimizer_v_list[i]

            optimizer.zero_grad()
            v_loss.backward()
            optimizer.step()

        obs_list = [m_obses[:, self.burn_in_step, ...] for m_obses in m_obses_list]

        option_c_entropy_list = self._train_policy(obs_list,
                                                   state,
                                                   option_index,
                                                   last_option_index,
                                                   action,
                                                   done,
                                                   next_state)

        if self.use_auto_alpha and ((self.d_action_size and not self.discrete_dqn_like) or self.c_action_size):
            d_alpha_list, c_alpha_list = self._train_alpha(obs_list, state, option_index)

        if self.summary_writer is not None and self.global_step % self.write_summary_per_step == 0:
            self.summary_available = True

            self.summary_writer.add_scalar('loss/v', v_loss, self.global_step)
            for i, ent in enumerate(option_c_entropy_list):
                if ent is not None:
                    self.summary_writer.add_scalar(f'loss/ent_{i}', ent, self.global_step)
            for i, c_alpha in enumerate(c_alpha_list):
                self.summary_writer.add_scalar(f'loss/c_alpha_{i}', c_alpha, self.global_step)
            for i, q_loss in enumerate(q_loss_list):
                self.summary_writer.add_scalar(f'loss/q_{i}', q_loss, self.global_step)

            self.summary_writer.flush()

    def _fill_replay_buffer(self,
                            l_indexes: np.ndarray,
                            l_padding_masks: np.ndarray,
                            l_obses_list: List[np.ndarray],
                            l_options: np.ndarray,
                            l_actions: np.ndarray,
                            l_rewards: np.ndarray,
                            next_obs_list: List[np.ndarray],
                            l_dones: np.ndarray,
                            l_probs: List[np.ndarray],
                            l_seq_hidden_states: np.ndarray = None):
        """
        Args:
            l_indexes: [1, episode_len]
            l_padding_masks: [1, episode_len]
            l_obses_list: list([1, episode_len, *obs_shapes_i], ...)
            l_actions: [1, episode_len, action_size]
            l_rewards: [1, episode_len]
            next_obs_list: list([1, *obs_shapes_i], ...)
            l_dones: [1, episode_len]
            l_probs: [1, episode_len]
            l_seq_hidden_states: [1, episode_len, *seq_hidden_state_shape]
        """
        # Reshape [1, episode_len, ...] to [episode_len, ...]
        index = l_indexes.squeeze(0)
        padding_mask = l_padding_masks.squeeze(0)
        obs_list = [l_obses.squeeze(0) for l_obses in l_obses_list]
        if self.use_normalization:
            self._udpate_normalizer([torch.from_numpy(obs).to(self.device) for obs in obs_list])
        action = l_actions.squeeze(0)
        option_index = l_options.squeeze(0)
        reward = l_rewards.squeeze(0)
        done = l_dones.squeeze(0)

        # Padding next_obs for episode experience replay
        index = np.concatenate([index,
                                index[-1:] + 1])
        padding_mask = np.concatenate([padding_mask,
                                       np.zeros([1], dtype=bool)])
        obs_list = [np.concatenate([obs, next_obs]) for obs, next_obs in zip(obs_list, next_obs_list)]
        action = np.concatenate([action,
                                 np.empty([1, action.shape[-1]], dtype=np.float32)])
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
                                      np.empty([1], dtype=np.float32)])
            storage_data['mu_prob'] = mu_prob

        if self.seq_encoder is not None:
            seq_hidden_state = l_seq_hidden_states.squeeze(0)
            seq_hidden_state = np.concatenate([seq_hidden_state,
                                               np.empty([1, *seq_hidden_state.shape[1:]], dtype=np.float32)])
            storage_data['seq_hidden_state'] = seq_hidden_state

        # n_step transitions except the first one and the last obs_, n_step - 1 + 1
        if self.use_add_with_td:
            td_error = self.get_episode_td_error(l_indexes=l_indexes,
                                                 l_padding_masks=l_padding_masks,
                                                 l_obses_list=l_obses_list,  # TODO
                                                 l_actions=l_actions,
                                                 l_rewards=l_rewards,
                                                 next_obs_list=next_obs_list,
                                                 l_dones=l_dones,
                                                 l_mu_probs=l_mu_probs if self.use_n_step_is else None,
                                                 l_seq_hidden_states=l_seq_hidden_states if self.seq_encoder is not None else None)
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
                *_, attn_weights_list = self.model_rep(torch.from_numpy(l_indexes).to(self.device),
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
                bn_indexes: [Batch, b + n]
                bn_padding_masks: [Batch, b + n]
                bn_obses_list: list([Batch, b + n, *obs_shapes_i], ...)
                bn_actions: [Batch, b + n, action_size]
                bn_rewards: [Batch, b + n]
                next_obs_list: list([Batch, *obs_shapes_i], ...)
                bn_dones: [Batch, b + n]
                bn_mu_probs: [Batch, b + n]
                bn_seq_hidden_states: [Batch, b + n, *seq_hidden_state_shape],
                priority_is: [Batch, 1]
            )
        """
        sampled = self.replay_buffer.sample()
        if sampled is None:
            return None

        """
        trans:
            index: [Batch, ]
            padding_mask: [Batch, ]
            obs_i: [Batch, *obs_shapes_i]
            action: [Batch, action_size]
            reward: [Batch, ]
            done: [Batch, ]
            mu_prob: [Batch, ]
            seq_hidden_state: [Batch, *seq_hidden_state_shape],
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
        m_indexes: [Batch, N + 1]
        m_padding_masks: [Batch, N + 1]
        m_obses_list: list([Batch, N + 1, *obs_shapes_i], ...)
        m_actions: [Batch, N + 1, action_size]
        m_rewards: [Batch, N + 1]
        m_dones: [Batch, N + 1]
        m_mu_probs: [Batch, N + 1]
        m_seq_hidden_state: [Batch, N + 1, *seq_hidden_state_shape]
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
                          priority_is if self.use_priority else None)

    def train(self):
        step = self.get_global_step()
        train_data = self._sample_from_replay_buffer()
        if train_data is None:
            return step

        pointers, batch = train_data
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
         priority_is) = batch

        bn_indexes = torch.from_numpy(bn_indexes).to(self.device)
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
        if self.use_replay_buffer and self.use_priority:
            priority_is = torch.from_numpy(priority_is).to(self.device)

        self._train(bn_indexes=bn_indexes,
                    bn_padding_masks=bn_padding_masks,
                    bn_obses_list=bn_obses_list,
                    bn_option_indexes=bn_option_indexes,
                    bn_actions=bn_actions,
                    bn_rewards=bn_rewards,
                    next_obs_list=next_obs_list,
                    bn_dones=bn_dones,
                    bn_mu_probs=bn_mu_probs if self.use_n_step_is else None,
                    f_seq_hidden_states=f_seq_hidden_states if self.seq_encoder is not None else None,
                    priority_is=priority_is if self.use_replay_buffer and self.use_priority else None)

        if step % self.save_model_per_step == 0:
            self.save_model()

        if self.use_replay_buffer:
            if self.use_n_step_is:
                bn_pi_probs_tensor = self.get_l_probs(bn_indexes,
                                                      bn_padding_masks,
                                                      bn_obses_list,
                                                      bn_option_indexes,
                                                      bn_actions,
                                                      f_seq_hidden_states=f_seq_hidden_states if self.seq_encoder is not None else None)

            # Update td_error
            if self.use_priority:
                td_error = self._get_td_error(bn_indexes=bn_indexes,
                                              bn_padding_masks=bn_padding_masks,
                                              bn_obses_list=bn_obses_list,
                                              bn_option_indexes=bn_option_indexes,
                                              bn_actions=bn_actions,
                                              bn_rewards=bn_rewards,
                                              next_obs_list=next_obs_list,
                                              bn_dones=bn_dones,
                                              bn_mu_probs=bn_pi_probs_tensor if self.use_n_step_is else None,
                                              f_seq_hidden_states=f_seq_hidden_states if self.seq_encoder is not None else None).detach().cpu().numpy()
                self.replay_buffer.update(pointers, td_error)

            # Update n_mu_probs
            if self.use_n_step_is:
                pointers_list = [pointers + i for i in range(0, self.burn_in_step + self.n_step)]
                tmp_pointers = np.stack(pointers_list, axis=1).reshape(-1)
                pi_probs = bn_pi_probs_tensor.detach().cpu().numpy().reshape(-1)
                self.replay_buffer.update_transitions(tmp_pointers, 'mu_prob', pi_probs)

        step = self._increase_global_step()

        return step

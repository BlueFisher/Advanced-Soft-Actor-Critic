import torch
from torch import nn, optim
from torch.nn import functional

from ..nn_models import *
from ..sac_base import SAC_Base
from ..utils import *


class OptionBase(SAC_Base):
    def __init__(self,
                 option: int,
                 fix_policy: bool,
                 random_q: bool,
                 *args, **kwargs):
        self.option = option
        self.fix_policy = fix_policy
        self.random_q = random_q
        super().__init__(*args, **kwargs)

    def _sample_thread(self):
        # Disable sample thread
        pass

    def _set_logger(self):
        if self.ma_name is None:
            self._logger = logging.getLogger('option')
        else:
            self._logger = logging.getLogger(f'option.{self.ma_name}')

    def _init_replay_buffer(self, replay_config):
        return

    def _build_model(self, nn, nn_config: dict | None, init_log_alpha: float, learning_rate: float) -> None:
        super()._build_model(nn, nn_config, init_log_alpha, learning_rate)

        if self.fix_policy:
            for param in self.model_rep.parameters():
                param.requires_grad = False
            for param in self.model_policy.parameters():
                param.requires_grad = False

        self.model_termination = nn.ModelTermination(self.state_size).to(self.device)
        self.optimizer_termination = optim.Adam(self.model_termination.parameters(), lr=learning_rate)

    def _build_ckpt(self) -> None:
        super()._build_ckpt()

        self.ckpt_dict['model_termination'] = self.model_termination

    def _init_or_restore(self, last_ckpt: int | None) -> None:
        super()._init_or_restore(last_ckpt)

        if self.train_mode and self.random_q:
            for model_q in self.model_q_list + self.model_target_q_list:
                for p in model_q.parameters():
                    if p.dim() > 1:
                        torch.nn.init.kaiming_normal_(p)
                    else:
                        torch.nn.init.normal_(p)
            self._logger.warning('Model Q randomized')

    def remove_models(self, gt: int):
        if self.ckpt_dir is None:
            return

        for ckpt_file in self.ckpt_dir.glob("*.pth"):
            if int(ckpt_file.stem) <= gt:
                continue

            try:
                ckpt_file.unlink()
                self._logger.warning(f"{ckpt_file.name} deleted")
            except Exception as e:
                self._logger.error(f"Failed to delete {ckpt_file}: {e}")

    @torch.no_grad()
    def choose_action(self,
                      obs_list: list[torch.Tensor],
                      pre_action: torch.Tensor,
                      pre_seq_hidden_state: torch.Tensor,

                      offline_action: torch.Tensor | None = None,
                      disable_sample: bool = False,
                      force_rnd_if_available: bool = False) -> tuple[torch.Tensor,
                                                                     torch.Tensor,
                                                                     torch.Tensor]:
        """
        Args:
            obs_list: list([batch, *obs_shapes_i], ...)
            pre_action: [batch, action_size]
            pre_seq_hidden_state: [batch, *seq_hidden_state_shape]

            offline_action: [batch, action_size]

        Returns:
            action: [batch, action_size]
            prob: [batch, action_size]
            seq_hidden_state: [batch, *seq_hidden_state_shape]
            termination (bool): [batch, ]
        """
        obs_list = [obs.unsqueeze(1) for obs in obs_list]
        pre_action = pre_action.unsqueeze(1)
        pre_seq_hidden_state = pre_seq_hidden_state.unsqueeze(1)

        state, seq_hidden_state = self.model_rep(obs_list, pre_action, pre_seq_hidden_state)
        # state: [batch, 1, state_size]
        # seq_hidden_state: [batch, 1, *seq_hidden_state_shape]
        state = state.squeeze(1)
        obs_list = [obs.squeeze(1) for obs in obs_list]
        seq_hidden_state = seq_hidden_state.squeeze(1)

        action, prob = self._choose_action(obs_list,
                                           state,
                                           offline_action,
                                           disable_sample or self.fix_policy,
                                           force_rnd_if_available)

        termination = self.model_termination(state, obs_list)

        return action, prob, seq_hidden_state, termination.squeeze(-1)

    #################### ! GET STATES ####################

    def get_l_states(
        self,
        l_indexes: torch.Tensor,
        l_padding_masks: torch.Tensor,
        l_obses_list: list[torch.Tensor],
        l_pre_actions: torch.Tensor,
        l_pre_seq_hidden_states: torch.Tensor,
        is_target=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            l_indexes: [batch, l]
            l_padding_masks: [batch, l]
            l_obses_list: list([batch, l, *obs_shapes_i], ...)
            l_pre_actions: [batch, l, action_size]
            l_pre_seq_hidden_states: [batch, l, *seq_hidden_state_shape]

        Returns:
            l_states: [batch, l, state_size]
            l_seq_hidden_states: [batch, l, *seq_hidden_state_shape]
        """

        model_rep = self.model_target_rep if is_target else self.model_rep

        if self.seq_encoder in (None, SEQ_ENCODER.RNN):
            l_states, l_hidden_states = model_rep(l_obses_list,
                                                  l_pre_actions,
                                                  l_pre_seq_hidden_states,
                                                  padding_mask=l_padding_masks)

            return l_states, l_hidden_states  # [batch, l, state_size], [batch, l, *seq_hidden_state_shape]

        elif self.seq_encoder == SEQ_ENCODER.ATTN:
            raise Exception('seq_encoder in option cannot be ATTN')

    #################### ! COMPUTE LOSS ####################

    @torch.no_grad()
    def get_dqn_like_d_y(self,
                         n_terminations: torch.Tensor,

                         next_n_vs: torch.Tensor,

                         n_padding_masks: torch.Tensor,
                         n_rewards: torch.Tensor,
                         n_dones: torch.Tensor,
                         stacked_next_n_d_qs: torch.Tensor,
                         stacked_next_target_n_d_qs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            n_terminations: [batch, n]
            next_n_vs: [batch, n]
            n_padding_masks (torch.bool): [batch, n]
            n_rewards: [batch, n]
            n_dones (torch.bool): [batch, n]
            stacked_next_n_d_qs: [ensemble_q_sample, batch, n, d_action_summed_size]
            stacked_next_target_n_d_qs: [ensemble_q_sample, batch, n, d_action_summed_size]

        Returns:
            y: [batch, 1]
        """
        batch_tensor = torch.arange(n_padding_masks.shape[0], device=self.device)
        last_solid_index = get_last_false_indexes(n_padding_masks, dim=1)  # [batch, ]

        done = n_dones[batch_tensor, last_solid_index].unsqueeze(-1)  # [batch, 1]
        stacked_next_q = stacked_next_n_d_qs[:, batch_tensor, last_solid_index, :]
        # [ensemble_q_sample, batch, d_action_summed_size]
        stacked_next_target_q = stacked_next_target_n_d_qs[:, batch_tensor, last_solid_index, :]
        # [ensemble_q_sample, batch, d_action_summed_size]

        stacked_next_q_list = stacked_next_q.split(self.d_action_sizes, dim=-1)

        mask_stacked_q_list = [functional.one_hot(torch.argmax(stacked_next_q, dim=-1),
                                                  d_action_size)
                               for stacked_next_q, d_action_size in zip(stacked_next_q_list, self.d_action_sizes)]
        mask_stacked_q = torch.concat(mask_stacked_q_list, dim=-1)
        # [ensemble_q_sample, batch, d_action_summed_size]

        stacked_max_next_target_q = torch.sum(stacked_next_target_q * mask_stacked_q,
                                              dim=-1,
                                              keepdim=True)
        # [ensemble_q_sample, batch, 1]
        stacked_max_next_target_q = stacked_max_next_target_q / self.d_action_branch_size

        next_q, _ = torch.min(stacked_max_next_target_q, dim=0)
        # [batch, 1]

        termination = n_terminations[batch_tensor, last_solid_index].unsqueeze(-1)  # [batch, 1]
        next_v = next_n_vs[batch_tensor, last_solid_index].unsqueeze(-1)  # [batch, 1]

        next_q = (1 - termination) * next_q + \
            termination * next_v  # [batch, 1]

        g = torch.sum(self._gamma_ratio * n_rewards, dim=-1, keepdim=True)  # [batch, 1]
        y = g + torch.pow(self.gamma, last_solid_index.unsqueeze(-1) + 1) * next_q * ~done  # [batch, 1]

        return y

    @torch.no_grad()
    def _get_y(self,
               next_n_vs_over_options: torch.Tensor,

               n_terminations: torch.Tensor,

               n_padding_masks: torch.Tensor,
               nx_obses_list: list[torch.Tensor],
               nx_states: torch.Tensor,
               nx_actions: torch.Tensor,
               n_rewards: torch.Tensor,
               n_dones: torch.Tensor,
               n_mu_probs: torch.Tensor | None) -> tuple[torch.Tensor | None,
                                                         torch.Tensor | None]:
        """
        Args:
            next_n_vs_over_options: [batch, n, num_options]

            n_terminations: [batch, n]

            n_padding_masks (torch.bool): [batch, n]
            nx_obses_list: list([batch, n + 1, *obs_shapes_i], ...)
            nx_states: [batch, n + 1, state_size]
            nx_actions: [batch, n + 1, action_size]
            n_rewards: [batch, n]
            n_dones (torch.bool): [batch, n]
            n_mu_probs: [batch, n, action_size]

        Returns:
            y: [batch, 1]
        """

        d_alpha = torch.exp(self.log_d_alpha)
        c_alpha = torch.exp(self.log_c_alpha)

        next_n_states = nx_states[:, 1:, ...]  # [batch, n, state_size]

        n_actions = nx_actions[:, :-1]  # [batch, n, action_size]

        next_n_vs, _ = next_n_vs_over_options.max(-1)  # [batch, n]

        nx_d_policy, nx_c_policy = self.model_policy(nx_states, nx_obses_list)

        if self.c_action_size:
            nx_c_action_sampled = nx_c_policy.rsample()  # [batch, n + 1, c_action_size]
        else:
            nx_c_action_sampled = torch.zeros(0, device=self.device)

        nx_qs_list = [q(nx_states, torch.tanh(nx_c_action_sampled), nx_obses_list) for q in self.model_target_q_list]
        # ([batch, n + 1, d_action_summed_size], [batch, n + 1, 1])

        d_y, c_y = None, None

        if self.d_action_sizes:
            n_d_qs_list = [q[0][:, :-1] for q in nx_qs_list]
            # [batch, n + 1, d_action_summed_size] -> [batch, n, d_action_summed_size]

            next_n_d_qs_list = [q[0][:, 1:] for q in nx_qs_list]
            # [batch, n + 1, d_action_summed_size] -> [batch, n, d_action_summed_size]

            stacked_next_n_d_qs = torch.stack(next_n_d_qs_list)[torch.randperm(self.ensemble_q_num)[:self.ensemble_q_sample]]
            # [ensemble_q_num, batch, n, d_action_summed_size] -> [ensemble_q_sample, batch, n, d_action_summed_size]

            if self.discrete_dqn_like:
                next_n_obses_list = [nx_obses[:, 1:, ...] for nx_obses in nx_obses_list]  # list([batch, n, *obs_shapes_i], ...)
                if self.c_action_size:
                    next_n_c_actions_sampled = nx_c_action_sampled[:, 1:, :]  # [batch, n, c_action_size]
                else:
                    next_n_c_actions_sampled = nx_c_action_sampled

                next_n_d_eval_qs_list = [q(next_n_states, torch.tanh(next_n_c_actions_sampled), next_n_obses_list)[0] for q in self.model_q_list]
                stacked_next_n_d_eval_qs = torch.stack(next_n_d_eval_qs_list)[torch.randperm(self.ensemble_q_num)[:self.ensemble_q_sample]]
                # [ensemble_q_num, batch, n, d_action_summed_size] -> [ensemble_q_sample, batch, n, d_action_summed_size]

                d_y = self.get_dqn_like_d_y(n_terminations=n_terminations,
                                            next_n_vs=next_n_vs,
                                            n_padding_masks=n_padding_masks,
                                            n_rewards=n_rewards,
                                            n_dones=n_dones,
                                            stacked_next_n_d_qs=stacked_next_n_d_eval_qs,
                                            stacked_next_target_n_d_qs=stacked_next_n_d_qs)
            else:
                stacked_n_d_qs = torch.stack(n_d_qs_list)[torch.randperm(self.ensemble_q_num)[:self.ensemble_q_sample]]
                # [ensemble_q_num, batch, n, d_action_summed_size] -> [ensemble_q_sample, batch, n, d_action_summed_size]

                min_n_d_qs, _ = torch.min(stacked_n_d_qs, dim=0)  # [batch, n, d_action_summed_size]
                min_next_n_d_qs, _ = torch.min(stacked_next_n_d_qs, dim=0)  # [batch, n, d_action_summed_size]

                nx_probs = nx_d_policy.probs  # [batch, n + 1, d_action_summed_size]

                n_probs = nx_probs[:, :-1]  # [batch, n, d_action_summed_size]
                next_n_probs = nx_probs[:, 1:]  # [batch, n, d_action_summed_size]
                # ! Note that the probs here is not strict probabilities
                # ! sum(probs) == self.d_action_branch_size
                clipped_n_probs = n_probs.clamp(min=1e-8)  # [batch, n, d_action_summed_size]
                clipped_next_n_probs = next_n_probs.clamp(min=1e-8)  # [batch, n, d_action_summed_size]
                tmp_n_vs = min_n_d_qs - d_alpha * torch.log(clipped_n_probs)  # [batch, n, d_action_summed_size]

                tmp_next_n_vs = (1 - n_terminations.unsqueeze(-1)) * (min_next_n_d_qs - d_alpha * torch.log(clipped_next_n_probs)) + \
                    n_terminations.unsqueeze(-1) * next_n_vs.unsqueeze(-1)  # [batch, n, d_action_summed_size]

                n_vs = torch.sum(n_probs * tmp_n_vs, dim=-1) / self.d_action_branch_size  # [batch, n]
                next_n_vs = torch.sum(next_n_probs * tmp_next_n_vs, dim=-1) / self.d_action_branch_size  # [batch, n]

                if self.use_n_step_is:
                    n_d_actions = n_actions[..., :self.d_action_summed_size]  # [batch, n, d_action_summed_size]
                    n_d_mu_probs = n_mu_probs[..., :self.d_action_summed_size]  # [batch, n, d_action_summed_size]
                    n_d_mu_probs = n_d_mu_probs * n_d_actions  # [batch, n]
                    n_d_mu_probs[n_d_mu_probs == 0.] = 1.
                    n_d_mu_probs = n_d_mu_probs.prod(-1)  # [batch, n]

                    nx_d_pi_probs = torch.exp(nx_d_policy.log_prob(nx_actions[..., :self.d_action_summed_size]).sum(-1))  # [batch, n + 1]
                    n_d_pi_probs = nx_d_pi_probs[:, :-1]  # [batch, n]

                d_y = self._v_trace(n_padding_masks=n_padding_masks,
                                    n_rewards=n_rewards,
                                    n_dones=n_dones,
                                    n_mu_probs=n_d_mu_probs if self.use_n_step_is else None,
                                    n_pi_probs=n_d_pi_probs if self.use_n_step_is else None,
                                    n_vs=n_vs,
                                    next_n_vs=next_n_vs)

        if self.c_action_size:
            n_c_qs_list = [q[1][:, :-1] for q in nx_qs_list]
            # [batch, n + 1, 1] -> [batch, n, 1]

            next_n_c_qs_list = [q[1][:, 1:] for q in nx_qs_list]
            # [batch, n + 1, 1] -> [batch, n, 1]

            nx_actions_log_prob = sum_log_prob(squash_correction_log_prob(nx_c_policy, nx_c_action_sampled))  # [batch, n + 1]
            n_actions_log_prob = nx_actions_log_prob[:, :-1]  # [batch, n]
            next_n_actions_log_prob = nx_actions_log_prob[:, 1:]  # [batch, n]

            stacked_n_c_qs = torch.stack(n_c_qs_list)[torch.randperm(self.ensemble_q_num)[:self.ensemble_q_sample]]
            # [ensemble_q_num, batch, n, 1] -> [ensemble_q_sample, batch, n, 1]
            stacked_next_n_c_qs = torch.stack(next_n_c_qs_list)[torch.randperm(self.ensemble_q_num)[:self.ensemble_q_sample]]
            # [ensemble_q_num, batch, n, 1] -> [ensemble_q_sample, batch, n, 1]

            min_n_c_qs, _ = stacked_n_c_qs.min(dim=0)
            min_n_c_qs = min_n_c_qs.squeeze(dim=-1)  # [batch, n]
            min_next_n_c_qs, _ = stacked_next_n_c_qs.min(dim=0)
            min_next_n_c_qs = min_next_n_c_qs.squeeze(dim=-1)  # [batch, n]

            """NORMAL"""
            # termination = n_terminations[:, -1:]  # [batch, 1]
            # min_next_max_v = min_next_n_vs[:, -1:]  # [batch, 1]
            # next_action_log_prob = next_n_actions_log_prob[:, -1:]  # [batch, 1]

            # min_next_c_q = min_next_n_c_qs[:, -1:]  # [batch, 1]
            # done = n_dones[:, -1:]  # [batch, 1]
            # c_y = torch.sum(self._gamma_ratio * n_rewards, dim=-1, keepdim=True)  # [batch, 1]
            # c_y = c_y + self.gamma ** self.n_step * ~done * \
            #     ((1 - termination) * (min_next_c_q - c_alpha * next_action_log_prob) +
            #      termination * min_next_max_v)

            """V-TRACE"""
            n_vs = min_n_c_qs - c_alpha * n_actions_log_prob  # [batch, n]
            next_n_vs = (1 - n_terminations) * (min_next_n_c_qs - c_alpha * next_n_actions_log_prob) + \
                n_terminations * next_n_vs  # [batch, n]

            if self.use_n_step_is:
                n_c_mu_probs = n_mu_probs[..., self.d_action_summed_size:]  # [batch, n, c_action_size]
                nx_c_pi_probs = squash_correction_prob(nx_c_policy, torch.atanh(nx_actions[..., self.d_action_summed_size:]))
                # [batch, n + 1, c_action_size]
                n_c_pi_probs = nx_c_pi_probs[:, :-1]
                # [batch, n, c_action_size]

            c_y = self._v_trace(n_padding_masks=n_padding_masks,
                                n_rewards=n_rewards,
                                n_dones=n_dones,
                                n_mu_probs=n_c_mu_probs.prod(-1) if self.use_n_step_is else None,
                                n_pi_probs=n_c_pi_probs.prod(-1) if self.use_n_step_is else None,
                                n_vs=n_vs,
                                next_n_vs=next_n_vs)

        return d_y, c_y

    def compute_rep_q_grads(self,
                            next_n_vs_over_options: torch.Tensor,

                            bn_indexes: torch.Tensor,
                            bn_padding_masks: torch.Tensor,
                            bnx_obses_list: list[torch.Tensor],
                            bnx_target_obses_list: list[torch.Tensor],
                            bnx_states: torch.Tensor,
                            bnx_target_states: torch.Tensor,
                            bnx_actions: torch.Tensor,
                            bn_rewards: torch.Tensor,
                            bn_dones: torch.Tensor,
                            bn_mu_probs: torch.Tensor,
                            priority_is: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            next_n_vs_over_options: [batch, n, num_options]

            bn_indexes (torch.int32): [batch, b + n],
            bn_padding_masks (torch.bool): [batch, b + n],
            bnx_obses_list: list([batch, b + n + 1, *obs_shapes_i], ...)
            bnx_target_obses_list: list([batch, b + n + 1, *obs_shapes_i], ...)
            bnx_states: [batch, b + n + 1, state_size]
            bnx_target_states: [batch, b + n + 1, state_size]
            bnx_actions: [batch, b + n + 1, action_size]
            bn_rewards: [batch, b + n]
            bn_dones (torch.bool): [batch, b + n]
            bn_mu_probs: [batch, b + n, action_size]
            priority_is: [batch, 1]
        """

        n_target_states = bnx_target_states[:, self.burn_in_step:-1]  # [batch, n, state_size]
        n_target_obses_list = [bnx_obses[:, self.burn_in_step:-1, ...] for bnx_obses in bnx_target_obses_list]  # list([batch, n, *obs_shapes_i], ...)
        with torch.no_grad():
            n_target_terminations = self.model_termination(n_target_states, n_target_obses_list)  # [batch, n, 1]
            n_target_terminations = n_target_terminations.squeeze(-1)  # [batch, n]

        obs_list = [bnx_obses[:, self.burn_in_step, ...] for bnx_obses in bnx_obses_list]
        state = bnx_states[:, self.burn_in_step, ...]

        batch = state.shape[0]

        action = bnx_actions[:, self.burn_in_step, ...]
        d_action = action[..., :self.d_action_summed_size]
        c_action = action[..., self.d_action_summed_size:]

        q_list = [q(state, c_action, obs_list) for q in self.model_q_list]
        # ([batch, d_action_summed_size], [batch, 1])
        d_q_list = [q[0] for q in q_list]  # [batch, action_size]
        c_q_list = [q[1] for q in q_list]  # [batch, 1]

        d_y, c_y = self._get_y(next_n_vs_over_options=next_n_vs_over_options,

                               n_terminations=n_target_terminations,

                               n_padding_masks=bn_padding_masks[:, self.burn_in_step:, ...],
                               nx_obses_list=[bnx_obses[:, self.burn_in_step:, ...] for bnx_obses in bnx_target_obses_list],
                               nx_states=bnx_target_states[:, self.burn_in_step:, ...],
                               nx_actions=bnx_actions[:, self.burn_in_step:, ...],
                               n_rewards=bn_rewards[:, self.burn_in_step:],
                               n_dones=bn_dones[:, self.burn_in_step:],
                               n_mu_probs=bn_mu_probs[:, self.burn_in_step:])
        #  [batch, 1], [batch, 1]

        loss_q_list = [torch.zeros((batch, 1), device=self.device) for _ in range(self.ensemble_q_num)]
        loss_none_mse = nn.MSELoss(reduction='none')

        if self.d_action_sizes:
            for i in range(self.ensemble_q_num):
                q_single = torch.sum(d_action * d_q_list[i], dim=-1, keepdim=True) / self.d_action_branch_size  # [batch, 1]
                loss_q_list[i] = loss_q_list[i] + loss_none_mse(q_single, d_y)

        if self.c_action_size:
            if self.clip_epsilon > 0:
                target_c_q_list = [q(state.detach(), c_action, obs_list)[1] for q in self.model_target_q_list]

                clipped_q_list = [target_c_q_list[i] + torch.clamp(
                    c_q_list[i] - target_c_q_list[i],
                    -self.clip_epsilon,
                    self.clip_epsilon,
                ) for i in range(self.ensemble_q_num)]

                loss_q_a_list = [loss_none_mse(clipped_q, c_y) for clipped_q in clipped_q_list]  # [batch, 1]
                loss_q_b_list = [loss_none_mse(q, c_y) for q in c_q_list]  # [batch, 1]

                for i in range(self.ensemble_q_num):
                    loss_q_list[i] = loss_q_list[i] + torch.maximum(loss_q_a_list[i], loss_q_b_list[i])  # [batch, 1]
            else:
                for i in range(self.ensemble_q_num):
                    loss_q_list[i] = loss_q_list[i] + loss_none_mse(c_q_list[i], c_y)  # [batch, 1]

        if priority_is is not None:
            loss_q_list = [loss_q * priority_is for loss_q in loss_q_list]

        loss_q_list = [torch.mean(loss) for loss in loss_q_list]

        if self.optimizer_rep:
            self.optimizer_rep.zero_grad()

        for loss_q, opt_q in zip(loss_q_list, self.optimizer_q_list):
            opt_q.zero_grad()
            loss_q.backward(retain_graph=True)

        grads_rep_main = [m.grad.detach() if m.grad is not None else None
                          for m in self.model_rep.parameters()]
        grads_q_main_list = [[m.grad.detach() if m.grad is not None else None for m in q.parameters()]
                             for q in self.model_q_list]

        """ Siamese Representation Learning """
        loss_siamese, loss_siamese_q = None, None
        if self.siamese is not None:
            loss_siamese, loss_siamese_q = self._train_siamese_representation_learning(
                grads_rep_main=grads_rep_main,
                grads_q_main_list=grads_q_main_list,
                bn_indexes=bn_indexes,
                bn_padding_masks=bn_padding_masks,
                bn_obses_list=[bnx_obses[:, :-1, ...] for bnx_obses in bnx_obses_list],
                bn_actions=bnx_actions[:, :-1])

        """ Recurrent Prediction Model """
        loss_predictions = None
        if self.use_prediction:
            loss_predictions = self._train_rpm(grads_rep_main=grads_rep_main,
                                               bnx_obses_list=bnx_obses_list,
                                               bnx_states=bnx_states,
                                               bnx_target_states=bnx_target_states,
                                               bn_actions=bnx_actions[:, :-1],
                                               bn_rewards=bn_rewards)

        loss_q = loss_q_list[0]

        if self.summary_writer is not None and self.global_step % self.write_summary_per_step == 0:
            self.summary_available = True

            self.summary_writer.add_scalar('loss/q', loss_q, self.global_step)
            if self.siamese is not None and loss_siamese is not None:
                self.summary_writer.add_scalar('loss/siamese',
                                               loss_siamese,
                                               self.global_step)
                if self.siamese_use_q and loss_siamese_q is not None:
                    self.summary_writer.add_scalar('loss/siamese_q',
                                                   loss_siamese_q,
                                                   self.global_step)

            if self.use_prediction:
                approx_next_state_dist_entropy, loss_reward, loss_obs = loss_predictions
                self.summary_writer.add_scalar('loss/transition',
                                               approx_next_state_dist_entropy,
                                               self.global_step)
                self.summary_writer.add_scalar('loss/reward', loss_reward, self.global_step)
                self.summary_writer.add_scalar('loss/observation', loss_obs, self.global_step)

                approx_obs_list = self.model_observation(bnx_states[0:1, 0, ...])
                if not isinstance(approx_obs_list, (list, tuple)):
                    approx_obs_list = [approx_obs_list]
                for approx_obs in approx_obs_list:
                    if approx_obs.dim() > 3:
                        self.summary_writer.add_images('observation',
                                                       approx_obs,
                                                       self.global_step)
            self.summary_writer.flush()

        return d_y, c_y

    def train_rep_q(self):
        for opt_q in self.optimizer_q_list:
            opt_q.step()

        if self.optimizer_rep:
            self.optimizer_rep.step()

    def train_policy_alpha(self,
                           bn_padding_masks: torch.Tensor,
                           bn_obses_list: list[torch.Tensor],
                           bnx_states: torch.Tensor,
                           bn_actions: torch.Tensor,
                           bn_mu_probs: torch.Tensor) -> None:
        """
        Args:
            bn_padding_masks (bool): [batch, b + n]
            bn_obses_list: list([batch, b + n, *obs_shapes_i], ...)
            bnx_states: [batch, b + n + 1, state_size]
            bn_actions: [batch, b + n, action_size]
            bn_mu_probs: [batch, b + n, action_size]
        """

        if self.fix_policy:
            return

        obs_list = [bn_obses[:, self.burn_in_step, ...] for bn_obses in bn_obses_list]
        bn_states = bnx_states[:, :-1, ...]
        state = bn_states[:, self.burn_in_step, ...]
        action = bn_actions[:, self.burn_in_step, ...]
        mu_d_policy_probs = bn_mu_probs[:, self.burn_in_step, :self.d_action_summed_size]

        d_policy_entropy, c_policy_entropy = self._train_policy(obs_list=obs_list,
                                                                state=state,
                                                                action=action,
                                                                mu_d_policy_probs=mu_d_policy_probs)

        if self.use_auto_alpha and ((self.d_action_sizes and not self.discrete_dqn_like) or self.c_action_size):
            d_alpha, c_alpha = self._train_alpha(obs_list, state)

        if self.curiosity is not None:
            loss_curiosity = self._train_curiosity(bn_padding_masks, bnx_states, bn_actions)

        if self.use_rnd:
            bn_states = bnx_states[:, :-1, ...]
            loss_rnd = self._train_rnd(bn_padding_masks, bn_states, bn_actions)

        if self.summary_writer is not None and self.global_step % self.write_summary_per_step == 0:
            self.summary_available = True

            if self.d_action_sizes:
                if self.d_action_sizes and not self.discrete_dqn_like:
                    self.summary_writer.add_scalar('loss/d_entropy', d_policy_entropy, self.global_step)
                    if self.use_auto_alpha and not self.discrete_dqn_like:
                        self.summary_writer.add_scalar('loss/d_alpha', d_alpha, self.global_step)
            if self.c_action_size:
                self.summary_writer.add_scalar('loss/c_entropy', c_policy_entropy, self.global_step)
                if self.use_auto_alpha:
                    self.summary_writer.add_scalar('loss/c_alpha', c_alpha, self.global_step)

            if self.curiosity is not None:
                self.summary_writer.add_scalar('loss/curiosity', loss_curiosity, self.global_step)

            if self.use_rnd:
                self.summary_writer.add_scalar('loss/rnd', loss_rnd, self.global_step)

            self.summary_writer.flush()

    def compute_termination_grads(self,
                                  terminal_entropy: float,
                                  obs_list: list[torch.Tensor],
                                  state: torch.Tensor,
                                  y: torch.Tensor,
                                  v_over_options: torch.Tensor,
                                  done: torch.Tensor,
                                  priority_is: torch.Tensor):
        """
        Args:
            terminal_entropy: float, tending not to terminate >0, tending to terminate <0
            obs_list: list([batch, *obs_shapes_i], ...)
            state: [batch, state_size]
            y: [batch, 1]
            v_over_options: [batch, num_options]
            done (torch.bool): [batch, ]
            priority_is: [batch, 1]
        """
        termination = self.model_termination(state, obs_list).squeeze(-1)  # [batch, ]

        # max_v_over_options, _ = v_over_options.max(-1)  # [batch, ]
        mean_v_over_options = v_over_options.mean(-1)  # [batch, ]

        adv = y.squeeze(-1) - mean_v_over_options + terminal_entropy
        # adv = functional.normalize(adv, dim=0)

        loss_termination = termination * adv * ~done * priority_is  # [batch, ]

        loss_termination = torch.mean(loss_termination)

        self.optimizer_termination.zero_grad()
        loss_termination.backward(retain_graph=True)

        if self.summary_writer is not None and self.global_step % self.write_summary_per_step == 0:
            self.summary_available = True

            self.summary_writer.add_scalar('loss/termination', loss_termination, self.global_step)
            self.summary_writer.add_scalar('metric/termination', torch.mean(termination), self.global_step)

            self.summary_writer.flush()

    def train_termination(self):
        self.optimizer_termination.step()

    @torch.no_grad()
    def _get_td_error(self,
                      next_n_vs_over_options: torch.Tensor,

                      bn_padding_masks: torch.Tensor,
                      bnx_obses_list: list[torch.Tensor],
                      bnx_target_obses_list: list[torch.Tensor],
                      state: torch.Tensor,
                      bnx_target_states: torch.Tensor,
                      bnx_actions: torch.Tensor,
                      bn_rewards: torch.Tensor,
                      bn_dones: torch.Tensor,
                      bn_mu_probs: torch.Tensor | None) -> torch.Tensor:
        """
        Args:
            next_n_vs_over_options: [batch, n, num_options]

            bn_padding_masks (torch.bool): [batch, b + n]
            bnx_obses_list: list([batch, b + n + 1, *obs_shapes_i], ...)
            bnx_target_obses_list: list([batch, b + n + 1, *obs_shapes_i], ...)
            state: [batch, state_size]
            bnx_target_states: [batch, b + n + 1, state_size]
            bnx_actions: [batch, b + n + 1, action_size]
            bn_rewards: [batch, b + n]
            bn_dones: [batch, b + n]
            bn_mu_probs: [batch, b + n, action_size]

        Returns:
            The td-error of observations, [batch, 1]
        """

        n_target_states = bnx_target_states[:, self.burn_in_step:-1]  # [batch, n, state_size]
        n_target_obses_list = [bnx_obses[:, self.burn_in_step:-1, ...] for bnx_obses in bnx_target_obses_list]  # list([batch, n, *obs_shapes_i], ...)
        n_target_terminations = self.model_termination(n_target_states, n_target_obses_list)  # [batch, n, 1]
        n_target_terminations = n_target_terminations.squeeze(-1)  # [batch, n]

        obs_list = [bnx_obses[:, self.burn_in_step, ...] for bnx_obses in bnx_obses_list]
        action = bnx_actions[:, self.burn_in_step, ...]
        d_action = action[..., :self.d_action_summed_size]
        c_action = action[..., self.d_action_summed_size:]

        batch = state.shape[0]

        q_list = [q(state, c_action, obs_list) for q in self.model_q_list]
        # ([batch, action_size], [batch, 1])
        d_q_list = [q[0] for q in q_list]  # [batch, action_size]
        c_q_list = [q[1] for q in q_list]  # [batch, 1]

        if self.d_action_sizes:
            d_q_list = [torch.sum(d_action * q, dim=-1, keepdim=True) / self.d_action_branch_size
                        for q in d_q_list]
            # [batch, 1]

        d_y, c_y = self._get_y(next_n_vs_over_options=next_n_vs_over_options,

                               n_terminations=n_target_terminations,

                               n_padding_masks=bn_padding_masks[:, self.burn_in_step:, ...],
                               nx_obses_list=[bnx_obses[:, self.burn_in_step:, ...] for bnx_obses in bnx_obses_list],
                               nx_states=bnx_target_states[:, self.burn_in_step:, ...],
                               nx_actions=bnx_actions[:, self.burn_in_step:, ...],
                               n_rewards=bn_rewards[:, self.burn_in_step:],
                               n_dones=bn_dones[:, self.burn_in_step:],
                               n_mu_probs=bn_mu_probs[:, self.burn_in_step:] if self.use_n_step_is else None)
        # [batch, 1], [batch, 1]

        q_td_error_list = [torch.zeros((batch, 1), device=self.device) for _ in range(self.ensemble_q_num)]
        # [batch, 1]

        if self.d_action_sizes:
            for i in range(self.ensemble_q_num):
                q_td_error_list[i] += torch.abs(d_q_list[i] - d_y)

        if self.c_action_size:
            for i in range(self.ensemble_q_num):
                q_td_error_list[i] += torch.abs(c_q_list[i] - c_y)

        td_error = torch.mean(torch.cat(q_td_error_list, dim=-1),
                              dim=-1, keepdim=True)
        return td_error

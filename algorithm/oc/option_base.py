from typing import List

import torch
from torch import nn
from torch.nn import functional

from ..nn_models import *
from ..sac_base import SAC_Base
from ..utils import *


class OptionBase(SAC_Base):
    @torch.no_grad()
    def choose_action(self,
                      obs_list: List[torch.Tensor],
                      disable_sample: bool = False,
                      force_rnd_if_available: bool = False):
        """
        Args:
            obs_list: list([Batch, *obs_shapes_i], ...)

        Returns:
            action: [Batch, d_action_summed_size + c_action_size]
        """
        state = self.model_rep(obs_list)

        action, prob = self._choose_action(obs_list, state, disable_sample, force_rnd_if_available)
        return action, prob

    @torch.no_grad()
    def choose_rnn_action(self,
                          obs_list: List[torch.Tensor],
                          pre_action: torch.Tensor,
                          rnn_state: torch.Tensor,
                          disable_sample: bool = False,
                          force_rnd_if_available: bool = False):
        """
        Args:
            obs_list: list([Batch, *obs_shapes_i], ...)
            pre_action: [Batch, d_action_summed_size + c_action_size]
            rnn_state: [Batch, *seq_hidden_state_shape]

        Returns:
            action: [Batch, d_action_summed_size + c_action_size]
            rnn_state: [Batch, *seq_hidden_state_shape]
        """
        obs_list = [obs.unsqueeze(1) for obs in obs_list]
        pre_action = pre_action.unsqueeze(1)
        state, next_rnn_state = self.model_rep(obs_list, pre_action, rnn_state)
        state = state.squeeze(1)
        obs_list = [obs.squeeze(1) for obs in obs_list]

        action, prob = self._choose_action(obs_list, state, disable_sample, force_rnd_if_available)

        return action, prob, next_rnn_state

    @torch.no_grad()
    def choose_attn_action(self,
                           ep_indexes: torch.Tensor,
                           ep_padding_masks: torch.Tensor,
                           ep_obses_list: List[torch.Tensor],
                           ep_pre_actions: torch.Tensor,
                           ep_attn_hidden_states: torch.Tensor,

                           disable_sample: bool = False,
                           force_rnd_if_available: bool = False):
        """
        Args:
            ep_indexes: [Batch, episode_len]
            ep_padding_masks: [Batch, episode_len]
            ep_obses_list: list([Batch, episode_len, *obs_shapes_i], ...)
            ep_pre_actions: [Batch, episode_len, d_action_summed_size + c_action_size]
            ep_attn_hidden_states: [Batch, episode_len, *seq_hidden_state_shape]

        Returns:
            action: [Batch, d_action_summed_size + c_action_size]
            prob: [Batch, ]
            attn_hidden_state: [Batch, *rnn_state_shape]
        """
        state, next_attn_hidden_state, _ = self.model_rep(ep_indexes,
                                                          ep_obses_list, ep_pre_actions,
                                                          query_length=1,
                                                          hidden_state=ep_attn_hidden_states,
                                                          is_prev_hidden_state=False,
                                                          padding_mask=ep_padding_masks)
        state = state.squeeze(1)
        next_attn_hidden_state = next_attn_hidden_state.squeeze(1)

        action, prob = self._choose_action([o[:, -1] for o in ep_obses_list],
                                           state,
                                           disable_sample,
                                           force_rnd_if_available)

        return (action.detach().cpu().numpy(),
                prob.detach().cpu().numpy(),
                next_attn_hidden_state.detach().cpu().numpy())

    @torch.no_grad()
    def get_dqn_like_d_y(self,
                         next_n_terminations: torch.Tensor,

                         min_next_n_max_vs: torch.Tensor,

                         n_rewards: torch.Tensor,
                         n_dones: torch.Tensor,
                         stacked_next_n_d_qs: torch.Tensor,
                         stacked_next_target_n_d_qs: torch.Tensor):
        """
        Args:
            next_n_terminations: [Batch, n]
            min_next_n_max_vs: [Batch, n]
            n_rewards: [Batch, n]
            n_dones (torch.bool): [Batch, n]
            stacked_next_n_d_qs: [ensemble_q_sample, Batch, n, d_action_summed_size]
            stacked_next_target_n_d_qs: [ensemble_q_sample, Batch, n, d_action_summed_size]

        Returns:
            y: [Batch, 1]
        """

        stacked_next_q = stacked_next_n_d_qs[..., -1, :]  # [ensemble_q_sample, Batch, d_action_summed_size]
        stacked_next_target_q = stacked_next_target_n_d_qs[..., -1, :]  # [ensemble_q_sample, Batch, d_action_summed_size]

        done = n_dones[:, -1:]  # [Batch, 1]

        stacked_next_q_list = stacked_next_q.split(self.d_action_sizes, dim=-1)

        mask_stacked_q_list = [functional.one_hot(torch.argmax(stacked_next_q, dim=-1),
                                                  d_action_size)
                               for stacked_next_q, d_action_size in zip(stacked_next_q_list, self.d_action_sizes)]
        mask_stacked_q = torch.concat(mask_stacked_q_list, dim=-1)
        # [ensemble_q_sample, Batch, d_action_summed_size]

        stacked_max_next_target_q = torch.sum(stacked_next_target_q * mask_stacked_q,
                                              dim=-1,
                                              keepdim=True)
        # [ensemble_q_sample, Batch, 1]
        stacked_max_next_target_q = stacked_max_next_target_q / self.d_action_branch_size

        next_q, _ = torch.min(stacked_max_next_target_q, dim=0)
        # [Batch, 1]

        next_termination = next_n_terminations[:, -1:]
        min_next_max_v = min_next_n_max_vs[:, -1:]

        next_q = (1 - next_termination) * next_q + \
            next_termination * min_next_max_v  # [Batch, 1]

        g = torch.sum(self._gamma_ratio * n_rewards, dim=-1, keepdim=True)  # [Batch, 1]
        y = g + self.gamma**self.n_step * next_q * ~done  # [Batch, 1]

        return y

    @torch.no_grad()
    def _get_y(self,
               next_n_terminations,

               next_n_v_over_options_list,

               n_obses_list,
               n_states,
               n_actions,
               n_rewards,
               next_obs_list,
               next_state,
               n_dones,
               n_mu_probs):
        """
        Args:
            next_n_terminations: [Batch, n]

            next_n_v_over_options_list: [Batch, n, num_options], ...

            n_obses_list: list([Batch, n, *obs_shapes_i], ...)
            n_states: [Batch, n, state_size]
            n_option_indexes (torch.int64): [Batch, n]
            n_actions: [Batch, n, action_size]
            n_rewards: [Batch, n]
            next_state: [Batch, state_size]
            n_dones (torch.bool): [Batch, n]
            n_mu_probs: [Batch, n]

        """

        d_alpha = torch.exp(self.log_d_alpha)
        c_alpha = torch.exp(self.log_c_alpha)

        next_n_obses_list = [torch.cat([n_obses[:, 1:, ...], next_obs.unsqueeze(1)], dim=1)
                             for n_obses, next_obs in zip(n_obses_list, next_obs_list)]  # list([Batch, n, *obs_shapes_i], ...)
        next_n_states = torch.cat([n_states[:, 1:, ...], next_state.unsqueeze(1)], dim=1)  # [Batch, n, state_size]

        next_n_max_vs_list = [next_n_v_over_options.max(-1)[0] for next_n_v_over_options in next_n_v_over_options_list]  # [Batch, n]

        stacked_next_n_max_vs = torch.stack(next_n_max_vs_list)[torch.randperm(self.ensemble_q_num)[:self.ensemble_q_sample]]
        # [ensemble_q_num, Batch, n] -> [ensemble_q_sample, Batch, n]

        min_next_n_max_vs, _ = stacked_next_n_max_vs.min(dim=0)  # [Batch, n]

        d_policy, c_policy = self.model_policy(n_states, n_obses_list)
        next_d_policy, next_c_policy = self.model_policy(next_n_states, next_n_obses_list)

        if self.c_action_size:
            n_c_actions_sampled = c_policy.rsample()  # [Batch, n, action_size]
            next_n_c_actions_sampled = next_c_policy.rsample()  # [Batch, n, action_size]
        else:
            n_c_actions_sampled = torch.zeros(0, device=self.device)
            next_n_c_actions_sampled = torch.zeros(0, device=self.device)

        # ([Batch, n, action_size], [Batch, n, 1])
        n_qs_list = [q(n_states, torch.tanh(n_c_actions_sampled)) for q in self.model_target_q_list]
        next_n_qs_list = [q(next_n_states, torch.tanh(next_n_c_actions_sampled)) for q in self.model_target_q_list]

        n_d_qs_list = [q[0] for q in n_qs_list]  # [Batch, n, action_size]
        n_c_qs_list = [q[1] for q in n_qs_list]  # [Batch, n, 1]

        next_n_d_qs_list = [q[0] for q in next_n_qs_list]  # [Batch, n, action_size]
        next_n_c_qs_list = [q[1] for q in next_n_qs_list]  # [Batch, n, 1]

        d_y, c_y = None, None

        if self.d_action_sizes:
            stacked_next_n_d_qs = torch.stack(next_n_d_qs_list)[torch.randperm(self.ensemble_q_num)[:self.ensemble_q_sample]]
            # [ensemble_q_num, Batch, n, d_action_summed_size] -> [ensemble_q_sample, Batch, n, d_action_summed_size]

            if self.discrete_dqn_like:
                next_n_d_eval_qs_list = [q(next_n_states, torch.tanh(next_n_c_actions_sampled))[0] for q in self.model_q_list]
                stacked_next_n_d_eval_qs = torch.stack(next_n_d_eval_qs_list)[torch.randperm(self.ensemble_q_num)[:self.ensemble_q_sample]]
                # [ensemble_q_num, Batch, n, d_action_summed_size] -> [ensemble_q_sample, Batch, n, d_action_summed_size]

                d_y = self.get_dqn_like_d_y(next_n_terminations=next_n_terminations,
                                            min_next_n_max_vs=min_next_n_max_vs,
                                            n_rewards=n_rewards,
                                            n_dones=n_dones,
                                            stacked_next_n_d_qs=stacked_next_n_d_eval_qs,
                                            stacked_next_target_n_d_qs=stacked_next_n_d_qs)
            else:
                stacked_n_d_qs = torch.stack(n_d_qs_list)[torch.randperm(self.ensemble_q_num)[:self.ensemble_q_sample]]
                # [ensemble_q_num, Batch, n, d_action_summed_size] -> [ensemble_q_sample, Batch, n, d_action_summed_size]

                min_n_d_qs, _ = torch.min(stacked_n_d_qs, dim=0)  # [Batch, n, d_action_summed_size]
                min_next_n_d_qs, _ = torch.min(stacked_next_n_d_qs, dim=0)  # [Batch, n, d_action_summed_size]

                n_probs = d_policy.probs  # [Batch, n, d_action_summed_size]
                next_n_probs = next_d_policy.probs  # [Batch, n, d_action_summed_size]
                # ! Note that the probs here is not strict probabilities
                # ! sum(probs) == self.d_action_branch_size
                clipped_n_probs = n_probs.clamp(min=1e-8)  # [Batch, n, d_action_summed_size]
                clipped_next_n_probs = next_n_probs.clamp(min=1e-8)  # [Batch, n, d_action_summed_size]
                tmp_n_vs = min_n_d_qs - d_alpha * torch.log(clipped_n_probs)  # [Batch, n, d_action_summed_size]

                tmp_next_n_vs = (1 - next_n_terminations.unsqueeze(-1)) * (min_next_n_d_qs - d_alpha * torch.log(clipped_next_n_probs)) + \
                    next_n_terminations.unsqueeze(-1) * min_next_n_max_vs.unsqueeze(-1)  # [Batch, n, d_action_summed_size]

                n_vs = torch.sum(n_probs * tmp_n_vs, dim=-1) / self.d_action_branch_size  # [Batch, n]
                next_n_vs = torch.sum(next_n_probs * tmp_next_n_vs, dim=-1) / self.d_action_branch_size  # [Batch, n]

                if self.use_n_step_is:
                    n_d_actions = n_actions[..., :self.d_action_summed_size]
                    n_pi_probs = torch.exp(d_policy.log_prob(n_d_actions))  # [Batch, n]

                d_y = self._v_trace(n_rewards=n_rewards,
                                    n_dones=n_dones,
                                    n_mu_probs=n_mu_probs if self.use_n_step_is else None,
                                    n_pi_probs=n_pi_probs if self.use_n_step_is else None,
                                    n_vs=n_vs,
                                    next_n_vs=next_n_vs)

        if self.c_action_size:
            n_actions_log_prob = torch.sum(squash_correction_log_prob(c_policy, n_c_actions_sampled), dim=-1)  # [Batch, n]
            next_n_actions_log_prob = torch.sum(squash_correction_log_prob(next_c_policy, next_n_c_actions_sampled), dim=-1)  # [Batch, n]

            stacked_n_c_qs = torch.stack(n_c_qs_list)[torch.randperm(self.ensemble_q_num)[:self.ensemble_q_sample]]
            # [ensemble_q_num, Batch, n, 1] -> [ensemble_q_sample, Batch, n, 1]
            stacked_next_n_c_qs = torch.stack(next_n_c_qs_list)[torch.randperm(self.ensemble_q_num)[:self.ensemble_q_sample]]
            # [ensemble_q_num, Batch, n, 1] -> [ensemble_q_sample, Batch, n, 1]

            min_n_c_qs, _ = stacked_n_c_qs.min(dim=0)
            min_n_c_qs = min_n_c_qs.squeeze(dim=-1)  # [Batch, n]
            min_next_n_c_qs, _ = stacked_next_n_c_qs.min(dim=0)
            min_next_n_c_qs = min_next_n_c_qs.squeeze(dim=-1)  # [Batch, n]

            """NORMAL"""
            # next_termination = next_n_terminations[:, -1:]  # [Batch, 1]
            # min_next_max_v = min_next_n_max_vs[:, -1:]  # [Batch, 1]
            # next_action_log_prob = next_n_actions_log_prob[:, -1:]  # [Batch, 1]

            # min_next_c_q = min_next_n_c_qs[:, -1:]  # [Batch, 1]
            # done = n_dones[:, -1:]  # [Batch, 1]
            # c_y = torch.sum(self._gamma_ratio * n_rewards, dim=-1, keepdim=True)  # [Batch, 1]
            # c_y = c_y + self.gamma ** self.n_step * ~done * \
            #     ((1 - next_termination) * (min_next_c_q - c_alpha * next_action_log_prob) +
            #      next_termination * min_next_max_v)

            """V-TRACE"""
            n_vs = min_n_c_qs - c_alpha * n_actions_log_prob  # [Batch, n]
            next_n_vs = (1 - next_n_terminations) * (min_next_n_c_qs - c_alpha * next_n_actions_log_prob) + \
                next_n_terminations * min_next_n_max_vs  # [Batch, n]

            if self.use_n_step_is:
                n_c_actions = n_actions[..., -self.c_action_size:]
                n_pi_probs = squash_correction_prob(c_policy, torch.atanh(n_c_actions))
                # [Batch, n, action_size]
                n_pi_probs = n_pi_probs.prod(axis=-1)  # [Batch, n]

            c_y = self._v_trace(n_rewards=n_rewards,
                                n_dones=n_dones,
                                n_mu_probs=n_mu_probs if self.use_n_step_is else None,
                                n_pi_probs=n_pi_probs if self.use_n_step_is else None,
                                n_vs=n_vs,
                                next_n_vs=next_n_vs)

        return d_y, c_y

    @torch.no_grad()
    def get_v(self,
              obs_list: List[torch.Tensor],
              state: torch.Tensor):
        o_d_policy, o_c_policy = self.model_policy(state, obs_list)
        o_c_action_sampled = o_c_policy.rsample()  # [Batch, c_action_size]
        o_c_action_log_prob = squash_correction_log_prob(o_c_policy, o_c_action_sampled)  # [Batch, c_action_size]
        o_c_action_log_prob = torch.sum(o_c_action_log_prob, dim=-1, keepdim=True)  # [Batch, 1]
        o_c_q_list = [q(state, torch.tanh(o_c_action_sampled))[1] for q in self.model_q_list]  # [[Batch, 1], ...]
        stacked_o_c_q = torch.stack(o_c_q_list)[torch.randperm(self.ensemble_q_num)[:self.ensemble_q_sample]]
        # [ensemble_q_num, Batch, 1] -> [ensemble_q_sample, Batch, 1]

        min_o_c_q, _ = torch.min(stacked_o_c_q, dim=0)  # [ensemble_q_sample, Batch, 1] -> [Batch, 1]

        v = min_o_c_q - torch.exp(self.log_c_alpha) * o_c_action_log_prob

        return v

    def train_rep_q(self,
                    next_n_terminations,

                    next_n_v_over_options_list,

                    bn_indexes: torch.Tensor,
                    bn_padding_masks: torch.Tensor,
                    bn_obses_list: List[torch.Tensor],
                    bn_target_obses_list: List[torch.Tensor],
                    bn_states: torch.Tensor,
                    bn_target_states: torch.Tensor,
                    bn_actions: torch.Tensor,
                    bn_rewards: torch.Tensor,
                    next_obs_list: List[torch.Tensor],
                    next_target_obs_list: List[torch.Tensor],
                    next_state: torch.Tensor,
                    next_target_state: torch.Tensor,
                    bn_dones: torch.Tensor,
                    bn_mu_probs: torch.Tensor = None,
                    priority_is: torch.Tensor = None):

        state = bn_states[:, self.burn_in_step, ...]

        batch = state.shape[0]

        action = bn_actions[:, self.burn_in_step, ...]
        d_action = action[..., :self.d_action_summed_size]
        c_action = action[..., -self.c_action_size:]

        q_list = [q(state, c_action) for q in self.model_q_list]
        # ([Batch, d_action_summed_size], [Batch, 1])
        d_q_list = [q[0] for q in q_list]  # [Batch, action_size]
        c_q_list = [q[1] for q in q_list]  # [Batch, 1]

        d_y, c_y = self._get_y(next_n_terminations=next_n_terminations,

                               next_n_v_over_options_list=next_n_v_over_options_list,

                               n_obses_list=[bn_target_obses[:, self.burn_in_step:, ...] for bn_target_obses in bn_target_obses_list],
                               n_states=bn_target_states[:, self.burn_in_step:, ...],
                               n_actions=bn_actions[:, self.burn_in_step:, ...],
                               n_rewards=bn_rewards[:, self.burn_in_step:],
                               next_obs_list=next_target_obs_list,
                               next_state=next_target_state,
                               n_dones=bn_dones[:, self.burn_in_step:],
                               n_mu_probs=bn_mu_probs[:, self.burn_in_step:] if self.use_n_step_is else None)
        #  [Batch, 1], [Batch, 1]

        loss_q_list = [torch.zeros((batch, 1), device=self.device) for _ in range(self.ensemble_q_num)]
        loss_none_mse = nn.MSELoss(reduction='none')

        if self.d_action_sizes:
            for i in range(self.ensemble_q_num):
                q_single = torch.sum(d_action * d_q_list[i], dim=-1, keepdim=True) / self.d_action_branch_size  # [Batch, 1]
                loss_q_list[i] += loss_none_mse(q_single, d_y)

        if self.c_action_size:
            if self.clip_epsilon > 0:
                target_c_q_list = [q(state, c_action)[1] for q in self.model_target_q_list]

                clipped_q_list = [target_c_q_list[i] + torch.clamp(
                    c_q_list[i] - target_c_q_list[i],
                    -self.clip_epsilon,
                    self.clip_epsilon,
                ) for i in range(self.ensemble_q_num)]

                loss_q_a_list = [loss_none_mse(clipped_q, c_y) for clipped_q in clipped_q_list]  # [Batch, 1]
                loss_q_b_list = [loss_none_mse(q, c_y) for q in c_q_list]  # [Batch, 1]

                for i in range(self.ensemble_q_num):
                    loss_q_list[i] += torch.maximum(loss_q_a_list[i], loss_q_b_list[i])  # [Batch, 1]
            else:
                for i in range(self.ensemble_q_num):
                    loss_q_list[i] += loss_none_mse(c_q_list[i], c_y)  # [Batch, 1]

        if self.use_replay_buffer and self.use_priority:
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
                bn_obses_list=bn_obses_list,
                bn_actions=bn_actions)

        for opt_q in self.optimizer_q_list:
            opt_q.step()

        """ Recurrent Prediction Model """
        loss_predictions = None
        if self.use_prediction:
            m_obses_list = [torch.cat([n_obses, next_obs.unsqueeze(1)], dim=1)
                            for n_obses, next_obs in zip(bn_obses_list, next_obs_list)]
            m_states = torch.cat([bn_states, next_state.unsqueeze(1)], dim=1)
            m_target_states = torch.cat([bn_target_states, next_target_state.unsqueeze(1)], dim=1)

            loss_predictions = self._train_rpm(grads_rep_main=grads_rep_main,
                                               m_obses_list=m_obses_list,
                                               m_states=m_states,
                                               m_target_states=m_target_states,
                                               bn_actions=bn_actions,
                                               bn_rewards=bn_rewards)

        if self.optimizer_rep:
            self.optimizer_rep.step()

        loss_q = loss_q_list[0]

        if self.summary_writer is not None and self.global_step % self.write_summary_per_step == 0:
            self.summary_available = True

            self.summary_writer.add_scalar('loss/q', loss_q, self.global_step)
            if self.siamese is not None:
                self.summary_writer.add_scalar('loss/siamese',
                                               loss_siamese,
                                               self.global_step)
                if self.siamese_use_q:
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

                approx_obs_list = self.model_observation(m_states[0:1, 0, ...])
                if not isinstance(approx_obs_list, (list, tuple)):
                    approx_obs_list = [approx_obs_list]
                for approx_obs in approx_obs_list:
                    if len(approx_obs.shape) > 3:
                        self.summary_writer.add_images('observation',
                                                       approx_obs.permute([0, 3, 1, 2]),
                                                       self.global_step)
            self.summary_writer.flush()

    def train_policy_alpha(self,
                           bn_obses_list: List[torch.Tensor],
                           bn_states: torch.Tensor,
                           bn_actions: torch.Tensor):

        obs_list = [bn_obses[:, self.burn_in_step, ...] for bn_obses in bn_obses_list]
        state = bn_states[:, self.burn_in_step, ...]
        action = bn_actions[:, self.burn_in_step, ...]

        d_policy_entropy, c_policy_entropy = self._train_policy(obs_list=obs_list,
                                                                state=state,
                                                                action=action)

        if self.use_auto_alpha and ((self.d_action_sizes and not self.discrete_dqn_like) or self.c_action_size):
            d_alpha, c_alpha = self._train_alpha(obs_list, state)

        if self.summary_writer is not None and self.global_step % self.write_summary_per_step == 0:
            self.summary_available = True

            if self.d_action_sizes:
                self.summary_writer.add_scalar('loss/d_entropy', d_policy_entropy, self.global_step)
                if self.use_auto_alpha and not self.discrete_dqn_like:
                    self.summary_writer.add_scalar('loss/d_alpha', d_alpha, self.global_step)
            if self.c_action_size:
                self.summary_writer.add_scalar('loss/c_entropy', c_policy_entropy, self.global_step)
                if self.use_auto_alpha:
                    self.summary_writer.add_scalar('loss/c_alpha', c_alpha, self.global_step)

            self.summary_writer.flush()

    @torch.no_grad()
    def _get_td_error(self,
                      next_n_terminations: torch.Tensor,

                      next_n_v_over_options_list: List[torch.Tensor],

                      bn_obses_list: List[torch.Tensor],
                      bn_states: torch.Tensor,
                      bn_target_states: torch.Tensor,
                      bn_actions: torch.Tensor,
                      bn_rewards: torch.Tensor,
                      next_obs_list: List[torch.Tensor],
                      next_target_state: torch.Tensor,
                      bn_dones: torch.Tensor,
                      bn_mu_probs: torch.Tensor = None):
        """
        Args:
            next_n_terminations: [Batch, n]

            next_n_v_over_options_list: list([Batch, n, num_options], ...)

            bn_obses_list: list([Batch, b + n, *obs_shapes_i], ...)
            bn_states: [Batch, b + n, state_size]
            bn_target_states: [Batch, b + n, state_size]
            bn_actions: [Batch, b + n, action_size]
            bn_rewards: [Batch, b + n]
            next_obs_list: list([Batch, *obs_shapes_i], ...)
            next_target_state: [Batch, state_size]
            bn_dones: [Batch, b + n]
            bn_mu_probs: [Batch, b + n]

        Returns:
            The td-error of observations, [Batch, 1]
        """

        state = bn_states[:, self.burn_in_step, ...]
        action = bn_actions[:, self.burn_in_step, ...]
        d_action = action[..., :self.d_action_summed_size]
        c_action = action[..., -self.c_action_size:]

        batch = state.shape[0]

        q_list = [q(state, c_action) for q in self.model_q_list]
        # ([Batch, action_size], [Batch, 1])
        d_q_list = [q[0] for q in q_list]  # [Batch, action_size]
        c_q_list = [q[1] for q in q_list]  # [Batch, 1]

        if self.d_action_sizes:
            d_q_list = [torch.sum(d_action * q, dim=-1, keepdim=True) / self.d_action_branch_size
                        for q in d_q_list]
            # [Batch, 1]

        d_y, c_y = self._get_y(next_n_terminations=next_n_terminations,

                               next_n_v_over_options_list=next_n_v_over_options_list,

                               n_obses_list=[bn_obses[:, self.burn_in_step:, ...] for bn_obses in bn_obses_list],
                               n_states=bn_target_states[:, self.burn_in_step:, ...],
                               n_actions=bn_actions[:, self.burn_in_step:, ...],
                               n_rewards=bn_rewards[:, self.burn_in_step:],
                               next_obs_list=next_obs_list,
                               next_state=next_target_state,
                               n_dones=bn_dones[:, self.burn_in_step:],
                               n_mu_probs=bn_mu_probs[:, self.burn_in_step:] if self.use_n_step_is else None)
        # [Batch, 1], [Batch, 1]

        q_td_error_list = [torch.zeros((batch, 1), device=self.device) for _ in range(self.ensemble_q_num)]
        # [Batch, 1]

        if self.d_action_sizes:
            for i in range(self.ensemble_q_num):
                q_td_error_list[i] += torch.abs(d_q_list[i] - d_y)

        if self.c_action_size:
            for i in range(self.ensemble_q_num):
                q_td_error_list[i] += torch.abs(c_q_list[i] - c_y)

        td_error = torch.mean(torch.cat(q_td_error_list, dim=-1),
                              dim=-1, keepdim=True)
        return td_error

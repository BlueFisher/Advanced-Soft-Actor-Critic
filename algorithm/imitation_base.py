from itertools import chain

import numpy as np
import torch
from torch import optim

from .nn_models import *
from .sac_base import SAC_Base, gen_pre_n_actions
from .utils import *


class ImitationBase:
    def __init__(self, sac_base: SAC_Base):
        self._sac = sac_base

        self.opt = optim.Adam(chain(self._sac.model_rep.parameters(),
                                    self._sac.model_policy.parameters()),
                              lr=self._sac.learning_rate)

    def train(self,
              ep_obses_list: list[np.ndarray],
              ep_actions: np.ndarray,
              ep_rewards: np.ndarray,
              ep_dones: np.ndarray):
        """
        Args:
            ep_obses_list (np): list([1, ep_len, *obs_shapes_i], ...)
            ep_actions (np): [1, ep_len, action_size]
            ep_rewards (np): [1, ep_len]
            ep_dones (bool): [1, ep_len]
        """
        ep_len = ep_actions.shape[1]

        ep_indexes = torch.arange(0, ep_len, dtype=torch.int32, device=self._sac.device).unsqueeze(0)

        ep_padding_masks = torch.zeros_like(ep_indexes, dtype=bool)
        ep_padding_masks[:, -1] = True  # The last step is next_step

        ep_obses_list = [torch.from_numpy(t).to(self._sac.device) for t in ep_obses_list]

        ep_actions = torch.from_numpy(ep_actions).to(self._sac.device)
        ep_pre_actions = gen_pre_n_actions(ep_actions, keep_last_action=False)

        initial_seq_hidden_state = self._sac.get_initial_seq_hidden_state(1, get_numpy=False)
        ep_pre_seq_hidden_states = initial_seq_hidden_state.unsqueeze(1).repeat_interleave(ep_len, axis=1)

        ep_states, _ = self._sac.get_l_states(l_indexes=ep_indexes,
                                              l_padding_masks=ep_padding_masks,
                                              l_obses_list=ep_obses_list,
                                              l_pre_actions=ep_pre_actions,
                                              l_pre_seq_hidden_states=ep_pre_seq_hidden_states,
                                              is_target=False)

        d_policy, c_policy = self._sac.model_policy(ep_states, ep_obses_list)

        ep_c_actions = ep_actions[:, :, self._sac.d_action_summed_size:]
        loss = -c_policy.log_prob(ep_c_actions) - 0.1 * c_policy.entropy()
        loss = torch.mean(loss)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        if self._sac.global_step % self._sac.write_summary_per_step == 0:
            self._sac.write_constant_summaries([{'tag': 'offline/loss',
                                                 'simple_value': loss}])

        if self._sac.get_global_step() % self._sac.save_model_per_step == 0:
            self._sac.save_model()

        self._sac.increase_global_step()

        return self._sac.get_global_step()

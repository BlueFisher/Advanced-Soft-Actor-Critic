from typing import List
import torch

import algorithm.nn_models as m


class ModelRep(m.ModelBaseRep):
    def _build_model(self):
        self.rnn = m.GRU(self.obs_shapes[0][0] + sum(self.d_action_sizes) + self.c_action_size, 8, 2)

    def forward(self,
                obs_list: list[torch.Tensor],
                pre_action: torch.Tensor,
                pre_seq_hidden_state: torch.Tensor | None,
                padding_mask: torch.Tensor | None = None):
        obs = obs_list[0]

        state, hn = self.rnn(torch.cat([obs, pre_action], dim=-1), pre_seq_hidden_state)

        return state, hn


class ModelOptionRep(m.ModelBaseRep):
    def forward(self,
                obs_list: list[torch.Tensor],
                pre_action: torch.Tensor,
                pre_seq_hidden_state: torch.Tensor | None,
                padding_mask: torch.Tensor | None = None):
        high_state, vec_obs = obs_list

        state = torch.concat([high_state, vec_obs], dim=-1)

        return state, self._get_empty_seq_hidden_state(state)


ModelQ = m.ModelQ
ModelPolicy = m.ModelPolicy

ModelTermination = m.ModelTermination

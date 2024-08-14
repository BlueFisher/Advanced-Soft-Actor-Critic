import torch

import algorithm.nn_models as m


class ModelRep(m.ModelBaseOptionSelectorRep):
    def forward(self,
                obs_list: list[torch.Tensor],
                pre_action: torch.Tensor,
                pre_seq_hidden_state: torch.Tensor | None,
                pre_termination_mask: torch.Tensor | None = None,
                padding_mask: torch.Tensor | None = None):

        if pre_action is not None:
            assert obs_list[0].shape[1] == pre_action.shape[1]

        state = torch.cat([o for o, os in zip(obs_list, self.obs_shapes) if len(os) == 1], dim=-1)

        return state, self._get_empty_seq_hidden_state(state)


ModelOptionRep = m.ModelSimpleRep


ModelQ = m.ModelQ
ModelPolicy = m.ModelPolicy
ModelTermination = m.ModelTermination

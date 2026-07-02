import torch

from torch import nn

import algorithm.nn_models as m

ModelRep = m.ModelSimpleRep


# class ModelRep(m.ModelBaseRep):
#     def _build_model(self, **kwargs):
#         self.batch_norm = nn.BatchNorm1d(self.obs_shapes[0][0])

#     def forward(self,
#                 obs_list: list[torch.Tensor],
#                 pre_action: torch.Tensor,
#                 pre_seq_hidden_state: torch.Tensor | None,
#                 padding_mask: torch.Tensor | None = None):
#         obs = obs_list[0]

#         obs = self.batch_norm(obs)

#         return obs, self._get_empty_seq_hidden_state(obs)


class ModelQ(m.ModelQ):
    def _build_model(self):
        super()._build_model(c_dense_n=64, c_dense_depth=2)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        super()._build_model(c_dense_n=64, c_dense_depth=2)

import torch
from torch import nn


class ModelBaseSimpleRep(nn.Module):
    def __init__(self, obs_shapes, **kwargs):
        super().__init__()
        self.obs_shapes = obs_shapes

        self._build_model(**kwargs)

    def _build_model(self, **kwargs):
        pass

    def forward(self, obs_list):
        raise Exception("ModelSimpleRep not implemented")

    def get_contrastive_encoder(self, obs_list):
        return None


class ModelSimpleRep(ModelBaseSimpleRep):
    def forward(self, obs_list):
        return torch.cat(obs_list, dim=-1)


class ModelBaseRNNRep(nn.Module):
    def __init__(self, obs_shapes, d_action_size, c_action_size, **kwargs):
        super().__init__()
        self.obs_shapes = obs_shapes
        self.d_action_size = d_action_size
        self.c_action_size = c_action_size

        self._build_model(**kwargs)

    def _build_model(self, **kwargs):
        pass

    def forward(self, obs_list, pre_action, rnn_state=None):
        raise Exception("ModelRNNRep not implemented")

    def get_contrastive_encoder(self, obs_list):
        return None

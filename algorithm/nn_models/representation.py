from typing import List, Tuple

import torch
from torch import nn

from algorithm.utils.image_visual import ImageVisual


class ModelBaseSimpleRep(nn.Module):
    def __init__(self, obs_shapes,
                 is_target: bool,
                 model_abs_dir=None, **kwargs):
        super().__init__()
        self.obs_shapes = obs_shapes
        self.is_target = is_target
        self.model_abs_dir = model_abs_dir

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
    def __init__(self, obs_shapes: List[Tuple], d_action_size: int, c_action_size: int,
                 is_target: bool,
                 model_abs_dir: str, **kwargs):
        super().__init__()
        self.obs_shapes = obs_shapes
        self.d_action_size = d_action_size
        self.c_action_size = c_action_size
        self.is_target = is_target
        self.model_abs_dir = model_abs_dir

        self._build_model(**kwargs)

    def _build_model(self, **kwargs):
        pass

    def forward(self, obs_list, pre_action, rnn_state=None):
        raise Exception("ModelRNNRep not implemented")

    def get_contrastive_encoder(self, obs_list):
        return None

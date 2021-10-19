from typing import List, Tuple

import torch
from torch import nn

from algorithm.nn_models.layers import LinearLayers
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

    def get_augmented_encoder(self, obs_list) -> torch.Tensor:
        raise Exception("get_augmented_encoder not implemented")


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

    def get_augmented_encoder(self, obs_list) -> torch.Tensor:
        raise Exception("get_augmented_encoder not implemented")


class ModelBaseRepProjection(nn.Module):
    def __init__(self, encoder_size):
        super().__init__()
        self.encoder_size = encoder_size

        self._build_model()

    def _build_model(self):
        pass

    def forward(self, encoder):
        raise Exception("ModelBaseRepProjection not implemented")


class ModelRepProjection(ModelBaseRepProjection):
    def _build_model(self, dense_n=None, dense_depth=1, projection_size=None):
        dense_n = self.encoder_size if dense_n is None else dense_n
        projection_size = dense_n - 2 if projection_size is None else projection_size

        self.dense = LinearLayers(self.encoder_size, dense_n, dense_depth, projection_size)

    def forward(self, encoder):
        return self.dense(encoder)


class ModelBaseRepPrediction(nn.Module):
    def __init__(self, encoder_size):
        super().__init__()
        self.encoder_size = encoder_size

        self._build_model()

    def _build_model(self):
        pass

    def forward(self, encoder):
        raise Exception("ModelBaseRepPrediction not implemented")


class ModelRepPrediction(ModelBaseRepPrediction):
    def _build_model(self, dense_n=None, dense_depth=1):
        dense_n = self.encoder_size if dense_n is None else dense_n

        self.dense = LinearLayers(self.encoder_size, dense_n, dense_depth, self.encoder_size)

    def forward(self, encoder):
        return self.dense(encoder)

from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from algorithm.nn_models.layers import LinearLayers


class ModelBaseSimpleRep(nn.Module):
    def __init__(self, obs_shapes,
                 is_target: bool, train_mode: bool,
                 model_abs_dir: Optional[Path] = None, **kwargs):
        super().__init__()
        self.obs_shapes = obs_shapes
        self.is_target = is_target
        self.train_mode = train_mode
        self.model_abs_dir = model_abs_dir

        self._build_model(**kwargs)

    def _build_model(self, **kwargs):
        pass

    def forward(self, obs_list: List[torch.Tensor]):
        raise Exception("ModelSimpleRep not implemented")

    def get_augmented_encoders(self,
                               obs_list: List[torch.Tensor]) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        raise Exception("get_augmented_encoders not implemented")

    def get_state_from_encoders(self,
                                obs_list: List[torch.Tensor],
                                encoders: Union[torch.Tensor, Tuple[torch.Tensor]]) -> torch.Tensor:
        raise Exception("get_state_from_encoders not implemented")


class ModelSimpleRep(ModelBaseSimpleRep):
    def forward(self, obs_list):
        return torch.cat([o for o, os in zip(obs_list, self.obs_shapes) if len(os) == 1], dim=-1)


class ModelBaseRNNRep(nn.Module):
    def __init__(self,
                 obs_shapes: List[Tuple],
                 d_action_sizes: List[int], c_action_size: int,
                 is_target: bool, train_mode: bool,
                 model_abs_dir: Optional[Path] = None, **kwargs):
        super().__init__()
        self.obs_shapes = obs_shapes
        self.d_action_sizes = d_action_sizes
        self.c_action_size = c_action_size
        self.train_mode = train_mode
        self.is_target = is_target
        self.model_abs_dir = model_abs_dir

        self._build_model(**kwargs)

    def _build_model(self, **kwargs):
        pass

    def forward(self,
                obs_list: List[torch.Tensor],
                pre_action: torch.Tensor,
                rnn_state: Optional[torch.Tensor] = None,
                padding_mask: Optional[torch.Tensor] = None):
        """
        Args:
            obs_list: list([batch, l, *obs_shapes_i], ...)
            pre_action: [batch, l, action_size]
            rnn_state: [batch, l, *seq_hidden_state_shape]
            padding_mask (torch.bool): [batch, l]
        """
        raise Exception("ModelRNNRep not implemented")

    def get_augmented_encoders(self, obs_list: List[torch.Tensor]) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        raise Exception("get_augmented_encoders not implemented")

    def get_state_from_encoders(self,
                                obs_list: List[torch.Tensor],
                                encoders: Union[torch.Tensor, Tuple[torch.Tensor]],
                                pre_action: torch.Tensor,
                                rnn_state: Optional[torch.Tensor] = None,
                                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise Exception("get_state_from_encoders not implemented")


class ModelBaseAttentionRep(nn.Module):
    def __init__(self,
                 obs_shapes: List[Tuple],
                 d_action_sizes: List[int], c_action_size: int,
                 is_target: bool, train_mode: bool,
                 model_abs_dir: Optional[Path] = None,
                 use_dilated_attn=False,  # For option critic
                 **kwargs):
        super().__init__()
        self.obs_shapes = obs_shapes
        self.d_action_sizes = d_action_sizes
        self.c_action_size = c_action_size
        self.train_mode = train_mode
        self.is_target = is_target
        self.model_abs_dir = model_abs_dir
        self.use_dilated_attn = use_dilated_attn

        self._build_model(**kwargs)

    def forward(self,
                index: torch.Tensor,
                obs_list: List[torch.Tensor],
                pre_action: Optional[torch.Tensor] = None,
                query_length=1,
                hidden_state: Optional[torch.Tensor] = None,
                is_prev_hidden_state=False,
                padding_mask: Optional[torch.Tensor] = None):
        raise Exception('ModelAttentionRep not implemented')

    def get_augmented_encoders(self, obs_list: List[torch.Tensor]) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        raise Exception("get_augmented_encoders not implemented")

    def get_state_from_encoders(self,
                                index: torch.Tensor,
                                obs_list: List[torch.Tensor],
                                encoders: Union[torch.Tensor, Tuple[torch.Tensor]],
                                pre_action: Optional[torch.Tensor] = None,
                                query_length=1,
                                hidden_state: Optional[torch.Tensor] = None,
                                is_prev_hidden_state=False,
                                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise Exception("get_state_from_encoders not implemented")


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

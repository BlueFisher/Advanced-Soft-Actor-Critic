from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch import nn

from algorithm.nn_models.layers import LinearLayers


class ModelBaseRep(nn.Module):
    def __init__(self,
                 obs_names: List[str],
                 obs_shapes: List[Tuple],
                 is_target: bool,
                 model_abs_dir: Optional[Path] = None, **kwargs):
        super().__init__()

        self.obs_names = obs_names
        self.obs_shapes = obs_shapes
        self.is_target = is_target
        self.model_abs_dir = model_abs_dir

        self._offline_action_index = -1
        try:
            self._offline_action_index = self.obs_names.index('_OFFLINE_ACTION')
        except:
            pass

    def _build_model(self, **kwargs):
        pass

    def forward(self, obs_list: List[torch.Tensor]):
        raise Exception("ModelRep not implemented")

    def get_offline_action(self, obs_list: List[torch.Tensor]) -> torch.Tensor:
        if self._offline_action_index != -1:
            return obs_list[self._offline_action_index]
        else:
            return None

    def get_augmented_encoders(self,
                               obs_list: List[torch.Tensor]) -> torch.Tensor | Tuple[torch.Tensor]:
        raise Exception("get_augmented_encoders not implemented")

    def get_state_from_encoders(self,
                                obs_list: List[torch.Tensor],
                                encoders: torch.Tensor | Tuple[torch.Tensor]) -> torch.Tensor:
        raise Exception("get_state_from_encoders not implemented")


class ModelBaseSimpleRep(ModelBaseRep):
    def __init__(self,
                 obs_names: List[str],
                 obs_shapes: List[Tuple],
                 is_target: bool,
                 model_abs_dir: Optional[Path] = None, **kwargs):
        super().__init__(obs_names, obs_shapes, is_target, model_abs_dir)

        self._build_model(**kwargs)


class ModelSimpleRep(ModelBaseSimpleRep):
    def forward(self, obs_list):
        return torch.cat([o for o, os in zip(obs_list, self.obs_shapes) if len(os) == 1], dim=-1)


class ModelBaseRNNRep(ModelBaseRep):
    def __init__(self,
                 obs_names: List[str],
                 obs_shapes: List[Tuple],
                 d_action_sizes: List[int], c_action_size: int,
                 is_target: bool,
                 model_abs_dir: Optional[Path] = None,
                 use_dilation=False,  # For option critic
                 **kwargs):
        super().__init__(obs_names, obs_shapes, is_target, model_abs_dir)
        self.d_action_sizes = d_action_sizes
        self.c_action_size = c_action_size
        self.use_dilation = use_dilation

        self._build_model(**kwargs)

    def forward(self,
                obs_list: List[torch.Tensor],
                pre_action: Optional[torch.Tensor] = None,
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

    def get_state_from_encoders(self,
                                obs_list: List[torch.Tensor],
                                encoders: torch.Tensor | Tuple[torch.Tensor],
                                pre_action: torch.Tensor,
                                rnn_state: Optional[torch.Tensor] = None,
                                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise Exception("get_state_from_encoders not implemented")


class ModelBaseAttentionRep(ModelBaseRep):
    def __init__(self,
                 obs_names: List[str],
                 obs_shapes: List[Tuple],
                 d_action_sizes: List[int], c_action_size: int,
                 is_target: bool,
                 model_abs_dir: Optional[Path] = None,
                 use_dilation=False,  # For option critic
                 **kwargs):
        super().__init__(obs_names, obs_shapes, is_target, model_abs_dir)
        self.d_action_sizes = d_action_sizes
        self.c_action_size = c_action_size
        self.use_dilation = use_dilation

        self._build_model(**kwargs)

    def forward(self,
                index: torch.Tensor,
                obs_list: List[torch.Tensor],
                pre_action: Optional[torch.Tensor] = None,
                seq_q_len=1,
                hidden_state: Optional[torch.Tensor] = None,
                is_prev_hidden_state=False,
                query_only_attend_to_rest_key=False,
                padding_mask: Optional[torch.Tensor] = None):
        raise Exception('ModelAttentionRep not implemented')

    def get_state_from_encoders(self,
                                index: torch.Tensor,
                                obs_list: List[torch.Tensor],
                                encoders: torch.Tensor | Tuple[torch.Tensor],
                                pre_action: Optional[torch.Tensor] = None,
                                seq_q_len=1,
                                hidden_state: Optional[torch.Tensor] = None,
                                is_prev_hidden_state=False,
                                query_only_attend_to_rest_key=False,
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

from pathlib import Path

import torch
from torch import nn

from algorithm.nn_models.layers import LinearLayers


class ModelBaseRep(nn.Module):
    def __init__(self,
                 obs_names: list[str],
                 obs_shapes: list[tuple],
                 d_action_sizes: list[int], c_action_size: int,
                 is_target: bool,
                 model_abs_dir: Path | None = None,
                 **kwargs):
        super().__init__()

        self.obs_names = obs_names
        self.obs_shapes = obs_shapes
        self.d_action_sizes = d_action_sizes
        self.c_action_size = c_action_size
        self.is_target = is_target
        self.model_abs_dir = model_abs_dir

        self._build_model(**kwargs)

    def _build_model(self, **kwargs):
        pass

    def forward(self,
                obs_list: list[torch.Tensor],
                pre_action: torch.Tensor,
                pre_seq_hidden_state: torch.Tensor | None,
                padding_mask: torch.Tensor | None = None):
        """
        Args:
            obs_list: list([batch, l, *obs_shapes_i], ...)
            pre_action: [batch, l, action_size]
            pre_seq_hidden_state: [batch, l, *seq_hidden_state_shape]
            padding_mask (torch.bool): [batch, l]

        Returns:
            state: [batch, l, state_size]
            seq_hidden_state: [batch, l, *seq_hidden_state_shape] if seq_encoder is None
                              [batch, *seq_hidden_state_shape] if seq_encoder is RNN
        """

        raise Exception("ModelRep not implemented")

    def __call__(self,
                 obs_list: list[torch.Tensor],
                 pre_action: torch.Tensor,
                 pre_seq_hidden_state: torch.Tensor | None,
                 padding_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        return super().__call__(obs_list, pre_action, pre_seq_hidden_state, padding_mask)

    def _get_empty_seq_hidden_state(self, state: torch.Tensor):
        return torch.zeros((*state.shape[:-1], 0), dtype=state.dtype, device=state.device)

    def get_augmented_encoders(self,
                               obs_list: list[torch.Tensor]) -> torch.Tensor | tuple[torch.Tensor]:
        raise Exception("get_augmented_encoders not implemented")

    def get_state_from_encoders(self,
                                encoders: torch.Tensor | tuple[torch.Tensor],
                                obs_list: list[torch.Tensor],
                                pre_action: torch.Tensor,
                                pre_seq_hidden_state: torch.Tensor | None,
                                padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        raise Exception("get_state_from_encoders not implemented")


class ModelSimpleRep(ModelBaseRep):
    def forward(self,
                obs_list: list[torch.Tensor],
                pre_action: torch.Tensor,
                pre_seq_hidden_state: torch.Tensor | None,
                padding_mask: torch.Tensor | None = None):

        state = torch.cat([o for o, os in zip(obs_list, self.obs_shapes) if len(os) == 1], dim=-1)

        return state, self._get_empty_seq_hidden_state(state)


class ModelBaseAttentionRep(ModelBaseRep):
    def forward(self,
                seq_q_len: int,
                index: torch.Tensor,
                obs_list: list[torch.Tensor],
                pre_action: torch.Tensor,
                pre_seq_hidden_state: torch.Tensor | None,
                is_prev_hidden_state=False,
                query_only_attend_to_rest_key=False,
                padding_mask: torch.Tensor | None = None):
        """
        Args:
            seq_q_len: int
            index: [batch, l]
            obs_list: list([batch, l, *obs_shapes_i], ...)
            pre_action: [batch, l, action_size]
            pre_seq_hidden_state: [batch, ?, *seq_hidden_state_shape]
            is_prev_hidden_state: bool
            query_only_attend_to_rest_key: bool
        """

        raise Exception('ModelAttentionRep not implemented')

    def __call__(self,
                 seq_q_len: int,
                 index: torch.Tensor,
                 obs_list: list[torch.Tensor],
                 pre_action: torch.Tensor,
                 pre_seq_hidden_state: torch.Tensor | None,
                 is_prev_hidden_state=False,
                 query_only_attend_to_rest_key=False,
                 padding_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        return nn.Module.__call__(self,
                                  seq_q_len,
                                  index,
                                  obs_list,
                                  pre_action,
                                  pre_seq_hidden_state,
                                  is_prev_hidden_state,
                                  query_only_attend_to_rest_key,
                                  padding_mask)

    def get_state_from_encoders(self,
                                seq_q_len: int,
                                encoders: torch.Tensor | tuple[torch.Tensor],
                                index: torch.Tensor,
                                obs_list: list[torch.Tensor],
                                pre_action: torch.Tensor,
                                pre_seq_hidden_state: torch.Tensor | None,
                                is_prev_hidden_state=False,
                                query_only_attend_to_rest_key=False,
                                padding_mask: torch.Tensor | None = None):

        raise Exception("get_state_from_encoders not implemented")


#################### ! OPTION SELECTOR ####################


class ModelBaseOptionSelectorRep(ModelBaseRep):
    def __init__(self,
                 obs_names: list[str],
                 obs_shapes: list[tuple],
                 d_action_sizes: list[int], c_action_size: int,
                 is_target: bool,
                 use_dilation,  # For HRL
                 model_abs_dir: Path | None = None,
                 **kwargs):

        self.use_dilation = use_dilation

        super().__init__(obs_names,
                         obs_shapes,
                         d_action_sizes,
                         c_action_size,
                         is_target,
                         model_abs_dir,
                         **kwargs)

    def forward(self,
                obs_list: list[torch.Tensor],
                pre_action: torch.Tensor,
                pre_seq_hidden_state: torch.Tensor | None,
                pre_termination_mask: torch.Tensor | None = None,
                padding_mask: torch.Tensor | None = None):
        """
        Args:
            obs_list: list([batch, l, *obs_shapes_i], ...)
            pre_action: [batch, l, action_size]
            pre_seq_hidden_state: [batch, l, *seq_hidden_state_shape]
            pre_termination_mask (torch.bool): [batch, ]
            padding_mask (torch.bool): [batch, l]
        """

        raise Exception("ModelOptionSelectorRep not implemented")

    def __call__(self,
                 obs_list: list[torch.Tensor],
                 pre_action: torch.Tensor,
                 pre_seq_hidden_state: torch.Tensor | None,
                 pre_termination_mask: torch.Tensor | None = None,
                 padding_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        return nn.Module.__call__(self,
                                  obs_list,
                                  pre_action,
                                  pre_seq_hidden_state,
                                  pre_termination_mask,
                                  padding_mask)


class ModelBaseOptionSelectorAttentionRep(ModelBaseOptionSelectorRep, ModelBaseAttentionRep):
    def forward(self,
                seq_q_len: int,
                index: torch.Tensor,
                obs_list: list[torch.Tensor],
                pre_action: torch.Tensor,
                pre_seq_hidden_state: torch.Tensor | None,
                pre_termination_mask: torch.Tensor | None = None,
                is_prev_hidden_state=False,
                query_only_attend_to_rest_key=False,
                padding_mask: torch.Tensor | None = None):
        raise Exception('ModelOptionSelectorAttentionRep not implemented')

    def __call__(self,
                 seq_q_len: int,
                 index: torch.Tensor,
                 obs_list: list[torch.Tensor],
                 pre_action: torch.Tensor,
                 pre_seq_hidden_state: torch.Tensor | None,
                 pre_termination_mask: torch.Tensor | None = None,
                 is_prev_hidden_state=False,
                 query_only_attend_to_rest_key=False,
                 padding_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        return nn.Module.__call__(self,
                                  seq_q_len,
                                  index,
                                  obs_list,
                                  pre_action,
                                  pre_seq_hidden_state,
                                  pre_termination_mask,
                                  is_prev_hidden_state,
                                  query_only_attend_to_rest_key,
                                  padding_mask)


class ModelVOverOptions(nn.Module):
    def __init__(self,
                 state_size: int,
                 num_options: int,
                 is_target: bool):
        super().__init__()
        self.state_size = state_size
        self.num_options = num_options
        self.is_target = is_target

        self._build_model()

    def _build_model(self, dense_n=64, dense_depth=2):
        self.dense = LinearLayers(self.state_size, dense_n, dense_depth,
                                  self.num_options)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.dense(state)

    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        return super().__call__(state)


class ModelBaseRepProjection(nn.Module):
    def __init__(self, encoder_size):
        super().__init__()
        self.encoder_size = encoder_size

        self._build_model()

    def _build_model(self):
        pass

    def forward(self, encoder: torch.Tensor):
        raise Exception("ModelBaseRepProjection not implemented")

    def __call__(self, encoder: torch.Tensor) -> torch.Tensor:
        return super().__call__(encoder)


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

    def forward(self, encoder: torch.Tensor):
        raise Exception("ModelBaseRepPrediction not implemented")

    def __call__(self, encoder: torch.Tensor) -> torch.Tensor:
        return super().__call__(encoder)


class ModelRepPrediction(ModelBaseRepPrediction):
    def _build_model(self, dense_n=None, dense_depth=1):
        dense_n = self.encoder_size if dense_n is None else dense_n

        self.dense = LinearLayers(self.encoder_size, dense_n, dense_depth, self.encoder_size)

    def forward(self, encoder):
        return self.dense(encoder)

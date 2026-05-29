from pathlib import Path

import torch
from torch import nn

from .layers import LinearLayers


class ModelBaseV(nn.Module):
    def __init__(self,
                 state_size: int,
                 is_target: bool,
                 model_abs_dir: Path | None = None):
        super().__init__()
        self.state_size = state_size
        self.is_target = is_target
        self.model_abs_dir = model_abs_dir

        self._build_model()

    def _build_model(self):
        pass

    def forward(self, state: torch.Tensor, obs_list: list[torch.Tensor]):
        raise Exception('ModelV not implemented')

    def __call__(self, state: torch.Tensor, obs_list: list[torch.Tensor]) -> torch.Tensor:
        return super().__call__(state, obs_list)


class ModelV(ModelBaseV):
    def _build_model(self, dense_n=64, dense_depth=2,
                     dropout=0.):
        self.dense = LinearLayers(self.state_size, dense_n, dense_depth, 1, dropout=dropout)

    def forward(self, state, obs_list):
        return self.dense(state)

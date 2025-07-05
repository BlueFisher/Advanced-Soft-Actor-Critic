from pathlib import Path

import torch
from torch import nn

from .layers import LinearLayers


class ModelBaseQ(nn.Module):
    def __init__(self,
                 state_size: int,
                 d_action_sizes: list[int], c_action_size: int,
                 is_target: bool,
                 model_abs_dir: Path | None = None):
        super().__init__()
        self.state_size = state_size
        self.d_action_sizes = d_action_sizes
        self.c_action_size = c_action_size
        self.is_target = is_target
        self.model_abs_dir = model_abs_dir

        self._build_model()

    def _build_model(self):
        pass

    def forward(self, state: torch.Tensor, action: torch.Tensor, obs_list: list[torch.Tensor]):
        raise Exception("ModelQ not implemented")

    def __call__(self, state: torch.Tensor, action: torch.Tensor, obs_list: list[torch.Tensor]) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return super().__call__(state, action, obs_list)


class ModelQ(ModelBaseQ):
    def _build_model(self, dense_n=64, dense_depth=0,
                     d_dense_n=64, d_dense_depth=3,
                     c_state_n=64, c_state_depth=0,
                     c_action_n=64, c_action_depth=0,
                     c_dense_n=64, c_dense_depth=3,
                     dropout=0.):
        """
                         state                  c_action
                           │                        │
                        ┌──▼──┐                     │
              ┌─────────┤dense├────┐                │
              │         └─────┘    │                │
              │                    │                │
              │             ┌──────▼──────┐ ┌───────▼──────┐
              │             │c_state_dense│ │c_action_dense│
              │             └──────┬──────┘ └───────┬──────┘
              │                    │                │
        ┌─────▼──────┐        ┌──────▼────────────────▼──────┐
        │d_dense_list│        │           c_dense            │
        └─────┬──────┘        └──────────────┬───────────────┘
              │                            │
              ▼                            ▼
        d_action_sizes                     1
        """
        self.dense = LinearLayers(self.state_size, dense_n, dense_depth, dropout=dropout)

        if self.d_action_sizes:
            self.d_dense_list = nn.ModuleList(
                [LinearLayers(self.dense.output_size, d_dense_n, d_dense_depth, d_action_size, dropout=dropout)
                 for d_action_size in self.d_action_sizes]
            )

        if self.c_action_size:
            self.c_state_dense = LinearLayers(self.dense.output_size, c_state_n, c_state_depth, dropout=dropout)
            self.c_action_dense = LinearLayers(self.c_action_size, c_action_n, c_action_depth, dropout=dropout)

            self.c_dense = LinearLayers(self.c_state_dense.output_size + self.c_action_dense.output_size,
                                        c_dense_n, c_dense_depth, 1, dropout=dropout)

    def forward(self, state, c_action, obs_list):
        state = self.dense(state)

        if self.d_action_sizes:
            d_qs = [d_dense(state) for d_dense in self.d_dense_list]
            d_qs = torch.concat(d_qs, dim=-1)
        else:
            d_qs = None

        if self.c_action_size:
            c_state = self.c_state_dense(state)
            c_action = self.c_action_dense(c_action)

            c_q = self.c_dense(torch.cat([c_state, c_action], dim=-1))
        else:
            c_q = None

        return d_qs, c_q

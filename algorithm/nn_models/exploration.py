from typing import List

import torch
from torch import nn

from .layers import LinearLayers


class ModelBaseRND(nn.Module):
    def __init__(self, state_size, d_action_size, c_action_size):
        super().__init__()
        self.state_size = state_size
        self.d_action_size = d_action_size
        self.c_action_size = c_action_size

        self._build_model()

    def _build_model(self):
        pass

    def cal_d_rnd(self, state) -> torch.Tensor:
        raise Exception("ModelBaseRND not implemented")

    def cal_c_rnd(self, state, c_action) -> torch.Tensor:
        raise Exception("ModelBaseRND not implemented")


class ModelRND(ModelBaseRND):
    def _build_model(self, dense_n=64, dense_depth=2, output_size=None):
        if self.d_action_size:
            self.d_dense_list = nn.ModuleList([
                LinearLayers(self.state_size,
                             dense_n, dense_depth, output_size)
                for _ in range(self.d_action_size)
            ])

        if self.c_action_size:
            self.c_dense = LinearLayers(self.state_size + self.c_action_size,
                                        dense_n, dense_depth, output_size)

    def cal_d_rnd(self, state):
        """
        Returns:
            d_rnd: [*batch, d_action_size, f]
        """
        d_rnd_list = [d(state).unsqueeze(-2) for d in self.d_dense_list]

        return torch.concat(d_rnd_list, dim=-2)

    def cal_c_rnd(self, state, c_action):
        """
        Returns:
            c_rnd: [*batch, f]
        """
        c_rnd = self.c_dense(torch.cat([state, c_action], dim=-1))

        return c_rnd


class ModelBaseForwardDynamic(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size

        self._build_model()

    def _build_model(self):
        pass

    def forward(self, state, action):
        raise Exception("ModelBaseForwardDynamic not implemented")


class ModelForwardDynamic(ModelBaseForwardDynamic):
    def _build_model(self, dense_n=64, dense_depth=2):
        self.dense = LinearLayers(self.state_size + self.action_size,
                                  dense_n, dense_depth, self.state_size)

    def forward(self, state, action):
        return self.dense(torch.cat([state, action], dim=-1))


class ModelBaseInverseDynamic(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size

        self._build_model()

    def _build_model(self):
        pass

    def forward(self, state_from, state_to):
        raise Exception("ModelBaseInverseDynamic not implemented")


class ModelInverseDynamic(ModelBaseInverseDynamic):
    def _build_model(self, dense_n=64, dense_depth=2):
        self.dense = LinearLayers(self.state_size + self.state_size,
                                  dense_n, dense_depth, self.action_size)

    def forward(self, state_from, state_to):
        return self.dense(torch.cat([state_from, state_to], dim=-1))

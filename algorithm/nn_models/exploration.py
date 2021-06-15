import torch
from torch import nn

from .layers import LinearLayers


class ModelBaseRND(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size

        self._build_model()

    def _build_model(self):
        pass

    def forward(self, state, action):
        raise Exception("ModelBaseRND not implemented")


class ModelRND(ModelBaseRND):
    def _build_model(self, dense_n=64, dense_depth=2, output_size=None):
        self.dense = LinearLayers(self.state_size + self.action_size,
                                  dense_n, dense_depth, output_size)

    def forward(self, state, action):
        return self.dense(torch.cat([state, action], dim=-1))


class ModelBaseForward(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size

        self._build_model()

    def _build_model(self):
        pass

    def forward(self, state, action):
        raise Exception("ModelBaseForward not implemented")


class ModelForward(ModelBaseForward):
    def _build_model(self, dense_n=64, dense_depth=2):
        self.dense = LinearLayers(self.state_size + self.action_size,
                                  dense_n, dense_depth, self.state_size)

    def forward(self, state, action):
        return self.dense(torch.cat([state, action], dim=-1))

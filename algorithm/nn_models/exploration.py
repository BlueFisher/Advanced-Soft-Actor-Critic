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

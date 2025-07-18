import torch
from torch import nn

from .layers import LinearLayers


class ModelRND(nn.Module):
    def __init__(self,
                 state_size: int,
                 d_action_summed_size: int,
                 c_action_size: int):
        super().__init__()
        self.state_size = state_size
        self.d_action_summed_size = d_action_summed_size
        self.c_action_size = c_action_size

        self._build_model()

    def _build_model(self, dense_n=64, dense_depth=2, output_size=None):
        self.s_dense = LinearLayers(self.state_size,
                                    dense_n, dense_depth, output_size)

        if self.d_action_summed_size:
            self.d_dense_list = nn.ModuleList([
                LinearLayers(self.state_size,
                             dense_n, dense_depth, output_size)
                for _ in range(self.d_action_summed_size)
            ])

        if self.c_action_size:
            self.c_dense = LinearLayers(self.state_size + self.c_action_size,
                                        dense_n, dense_depth, output_size)

    def cal_s_rnd(self, state) -> torch.Tensor:
        """
        Returns:
            s_rnd: [*batch, f]
        """
        s_rnd = self.s_dense(state)

        return s_rnd

    def cal_d_rnd(self, state) -> torch.Tensor:
        """
        Returns:
            d_rnd: [*batch, d_action_summed_size, f]
        """
        d_rnd_list = [d(state).unsqueeze(-2) for d in self.d_dense_list]

        return torch.concat(d_rnd_list, dim=-2)

    def cal_c_rnd(self, state, c_action) -> torch.Tensor:
        """
        Returns:
            c_rnd: [*batch, f]
        """
        c_rnd = self.c_dense(torch.cat([state, c_action], dim=-1))

        return c_rnd


class ModelOptionSelectorRND(nn.Module):
    def __init__(self, state_size, num_options: int):
        super().__init__()
        self.state_size = state_size
        self.num_options = num_options

        self._build_model()

    def _build_model(self, dense_n=64, dense_depth=2, output_size=None):
        self.dense_list = nn.ModuleList([
            LinearLayers(self.state_size,
                         dense_n, dense_depth, output_size)
            for _ in range(self.num_options)
        ])

    def cal_rnd(self, state) -> torch.Tensor:
        """
        Returns:
            d_rnd: [*batch, num_options, f]
        """
        rnd_list = [d(state).unsqueeze(-2) for d in self.dense_list]

        return torch.concat(rnd_list, dim=-2)


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

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super().__call__(x)


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

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super().__call__(x)


class ModelInverseDynamic(ModelBaseInverseDynamic):
    def _build_model(self, dense_n=64, dense_depth=2):
        self.dense = LinearLayers(self.state_size + self.state_size,
                                  dense_n, dense_depth, self.action_size)

    def forward(self, state_from, state_to):
        return self.dense(torch.cat([state_from, state_to], dim=-1))

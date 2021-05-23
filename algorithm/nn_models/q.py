import torch
from torch import nn

from .layers import dense_layers


class ModelBaseQ(nn.Module):
    def __init__(self, state_size, d_action_size, c_action_size):
        super().__init__()
        self.state_size = state_size
        self.d_action_size = d_action_size
        self.c_action_size = c_action_size

        self._build_model()

    def _build_model(self):
        pass

    def forward(self, state, action):
        raise Exception("ModelQ not implemented")


class ModelQ(ModelBaseQ):
    def _build_model(self, dense_n=64, dense_depth=0,
                     d_dense_n=64, d_dense_depth=3,
                     c_state_n=64, c_state_depth=0,
                     c_action_n=64, c_action_depth=0,
                     c_dense_n=64, c_dense_depth=3):

        self.dense, _output_dim = dense_layers(self.state_size, dense_n, dense_depth)

        if self.d_action_size:
            self.d_dense, _ = dense_layers(_output_dim, d_dense_n, d_dense_depth, self.d_action_size)

        if self.c_action_size:
            self.c_state_dense, _c_state_output_dim = dense_layers(_output_dim, c_state_n, c_state_depth)
            self.c_action_dense, _c_action_ouput_dim = dense_layers(self.c_action_size, c_action_n, c_action_depth)

            self.c_dense, _ = dense_layers(_c_state_output_dim + _c_action_ouput_dim,
                                           c_dense_n, c_dense_depth, 1)

    def forward(self, state, c_action):
        state = self.dense(state)

        if self.d_action_size:
            d_q = self.d_dense(state)
        else:
            d_q = None

        if self.c_action_size:
            c_state = self.c_state_dense(state)
            c_action = self.c_action_dense(c_action)

            c_q = self.c_dense(torch.cat([c_state, c_action], dim=-1))
        else:
            c_q = None

        return d_q, c_q

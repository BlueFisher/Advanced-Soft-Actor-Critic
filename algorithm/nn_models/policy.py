import torch
from torch import nn

from .layers import LinearLayers


class ModelBasePolicy(nn.Module):
    def __init__(self, state_size, d_action_size, c_action_size):
        super().__init__()
        self.state_size = state_size
        self.d_action_size = d_action_size
        self.c_action_size = c_action_size

        self._build_model()

    def _build_model(self):
        pass

    def forward(self, state):
        raise Exception("ModelPolicy not implemented")


class ModelPolicy(ModelBasePolicy):
    def _build_model(self, dense_n=64, dense_depth=0,
                     d_dense_n=64, d_dense_depth=3,
                     c_dense_n=64, c_dense_depth=3,
                     mean_n=64, mean_depth=0,
                     logstd_n=64, logstd_depth=0):

        self.dense = LinearLayers(self.state_size, dense_n, dense_depth)

        if self.d_action_size:
            self.d_dense = LinearLayers(self.dense.output_size, d_dense_n, d_dense_depth, self.d_action_size)

        if self.c_action_size:
            self.c_dense = LinearLayers(self.dense.output_size, c_dense_n, c_dense_depth)

            self.mean_dense = LinearLayers(self.c_dense.output_size, mean_n, mean_depth, self.c_action_size)
            self.logstd_dense = LinearLayers(self.c_dense.output_size, logstd_n, logstd_depth, self.c_action_size)

    def forward(self, state):
        state = self.dense(state)

        if self.d_action_size:
            logits = self.d_dense(state)
            d_policy = torch.distributions.OneHotCategorical(logits=logits)
        else:
            d_policy = None

        if self.c_action_size:
            l = self.c_dense(state)
            mean = self.mean_dense(l)
            logstd = self.logstd_dense(l)
            c_policy = torch.distributions.Normal(torch.tanh(mean), torch.clamp(torch.exp(logstd), 0.1, 1.0))
        else:
            c_policy = None

        return d_policy, c_policy

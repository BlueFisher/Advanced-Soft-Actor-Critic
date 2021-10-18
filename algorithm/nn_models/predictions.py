import torch
from torch import nn

from .layers import LinearLayers


class ModelBaseTransition(nn.Module):
    def __init__(self, state_size, d_action_size, c_action_size, use_extra_data):
        super().__init__()
        self.state_size = state_size
        self.d_action_size = d_action_size
        self.c_action_size = c_action_size
        self.use_extra_data = use_extra_data

        self.action_size = d_action_size + c_action_size

        self._build_model()

    def _build_model(self):
        pass

    def forward(self, obs_list, state, action):
        # (s_t, a_t) -> p(\approx{s}_{t+1})
        # OR
        # ((s_t, extra_obs_t), a_t) -> p(\approx{s}_{t+1})
        # Should return a Gaussian distribution
        raise Exception("ModelBaseTransition not implemented")

    def extra_obs(self, obs_list):
        raise Exception("ModelBaseTransition.extra_obs not implemented")


class ModelTransition(ModelBaseTransition):
    def _build_model(self, dense_n=64, dense_depth=0, extra_size=0):
        input_size = self.state_size + self.action_size
        if self.use_extra_data:
            if extra_size == 0:
                raise Exception("use_extra_data is True but extra_size is zero")
            input_size += extra_size
        self.dense = LinearLayers(input_size,
                                  dense_n, dense_depth,
                                  self.state_size * 2)

    def forward(self, obs_list, state, action):
        if self.use_extra_data:
            state = torch.cat([state, self.extra_obs(obs_list)], dim=-1)

        next_state = self.dense(torch.cat([state, action], dim=-1))
        mean, logstd = torch.chunk(next_state, 2, dim=-1)
        return torch.distributions.Normal(mean, torch.clamp(torch.exp(logstd), 0.1, 1.0))


class ModelBaseReward(nn.Module):
    def __init__(self, state_size):
        super().__init__()
        self.state_size = state_size

        self._build_model()

    def _build_model(self):
        pass

    def forward(self, state):
        # s_t -> \approx{r}_t
        raise Exception("ModelBaseReward not implemented")


class ModelReward(ModelBaseReward):
    def _build_model(self, dense_n=64, dense_depth=0):
        self.dense = LinearLayers(self.state_size, dense_n, dense_depth, 1)

    def forward(self, state):
        return self.dense(state)


class ModelBaseObservation(nn.Module):
    def __init__(self, state_size, obs_shapes, use_extra_data):
        super().__init__()
        self.state_size = state_size
        self.obs_shapes = obs_shapes
        self.use_extra_data = use_extra_data

        self._build_model()

    def _build_model(self):
        pass

    def forward(self, state):
        # s_t -> \approx{o}_t
        # Could return Tensor or [Tensor, Tensor, ...]
        raise Exception("ModelBaseObservation not implemented")

    def get_loss(self, state, obs_list) -> torch.Tensor:
        # loss(s_t -> \approx{o}_t, o_t)
        raise Exception("ModelBaseObservation.get_loss not implemented")

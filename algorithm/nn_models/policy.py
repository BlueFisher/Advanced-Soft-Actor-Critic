from pathlib import Path

import torch
from torch import nn
from torch.distributions.utils import _standard_normal

from .layers import LinearLayers


class NormalWithPadding(torch.distributions.Normal):
    def __init__(self, loc, scale, padding_mask, validate_args=None):
        super().__init__(loc, scale, validate_args)

        self.padding_mask = padding_mask

    def sample(self, sample_shape=torch.Size()):
        v = super().sample(sample_shape)

        v[..., self.padding_mask] = 0.

        return v

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)

        loc = self.loc * ~self.padding_mask
        scale = self.scale * ~self.padding_mask

        return loc + eps * scale

    def log_prob(self, value):
        _log_prob = super().log_prob(value)

        _log_prob[self.padding_mask] = torch.inf

        return _log_prob

    def entropy(self):
        _entropy = super().entropy()

        _entropy[self.padding_mask] = torch.inf

        return _entropy


class JointOneHotCategorical(torch.distributions.Distribution):
    def __init__(self, dists: list[torch.distributions.OneHotCategorical]):
        self._dists = dists
        self.logits_size_list = [dist.logits.shape[-1] for dist in dists]

    @property
    def probs(self) -> torch.Tensor:
        return torch.concat([dist.probs for dist in self._dists], dim=-1)

    @property
    def logits(self) -> torch.Tensor:
        return torch.concat([dist.logits for dist in self._dists], dim=-1)

    @property
    def dists(self) -> torch.distributions.OneHotCategorical:
        return self._dists

    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:
        sampled_list = [dist.sample(sample_shape) for dist in self._dists]

        return torch.concat(sampled_list, dim=-1)

    def sample_deter(self) -> torch.Tensor:
        logits_list = self.logits.split(self.logits_size_list)
        sampled_list = [torch.functional.one_hot(logits.argmax(dim=-1),
                                                 logits_size)
                        for logits, logits_size in zip(logits_list, self.logits_size_list)]

        return torch.concat(sampled_list, dim=-1)

    def log_prob(self, value) -> torch.Tensor:
        values = value.split(self.logits_size_list, dim=-1)
        log_prob_list = [dist.log_prob(value) for dist, value in zip(self._dists, values)]
        return torch.stack(log_prob_list, dim=-1)

    def entropy(self) -> torch.Tensor:
        entropy_list = [dist.entropy() for dist in self._dists]
        return torch.stack(entropy_list, dim=-1)


class ModelBasePolicy(nn.Module):
    def __init__(self,
                 state_size: int,
                 d_action_sizes: list[int], c_action_size: int,
                 model_abs_dir: Path | None = None, **kwargs):
        super().__init__()
        self.state_size = state_size
        self.d_action_sizes = d_action_sizes
        self.c_action_size = c_action_size
        self.model_abs_dir = model_abs_dir

        self._build_model(**kwargs)

    def _build_model(self, **kwargs):
        pass

    def forward(self, state, obs_list) -> tuple[torch.distributions.OneHotCategorical, torch.distributions.Normal]:
        raise Exception("ModelPolicy not implemented")


class ModelPolicy(ModelBasePolicy):
    def _build_model(self, dense_n=64, dense_depth=0,
                     d_dense_n=64, d_dense_depth=3,
                     c_dense_n=64, c_dense_depth=3,
                     mean_n=64, mean_depth=0,
                     logstd_n=64, logstd_depth=0,
                     dropout=0.):
        """
                      state
                        │
                     ┌──▼──┐
               ┌─────┤dense├────────┐
               │     └─────┘        │
               │                ┌───▼───┐
               │            ┌───┤c_dense├────┐
               │            │   └───────┘    │
               │            │                │
           ┌───▼───┐   ┌────▼─────┐   ┌──────▼─────┐
           │d_dense│   │mean_dense│   │logstd_dense│
           └───┬───┘   └────┬─────┘   └──────┬─────┘
               │            │                │
               ▼            │                │
        OneHotCategorical   └──► Normal ◄────┘
        """
        self.dense = LinearLayers(self.state_size, dense_n, dense_depth, dropout=dropout)

        if self.d_action_sizes:
            self.d_dense_list = nn.ModuleList(
                [LinearLayers(self.dense.output_size, d_dense_n, d_dense_depth, d_action_size, dropout=dropout)
                 for d_action_size in self.d_action_sizes]
            )

        if self.c_action_size:
            self.c_dense = LinearLayers(self.dense.output_size, c_dense_n, c_dense_depth, dropout=dropout)

            self.mean_dense = LinearLayers(self.c_dense.output_size, mean_n, mean_depth, self.c_action_size, dropout=dropout)
            self.logstd_dense = LinearLayers(self.c_dense.output_size, logstd_n, logstd_depth, self.c_action_size, dropout=dropout)

    def forward(self, state, obs_list):
        state = self.dense(state)

        if self.d_action_sizes:
            logits_list = [d_dense(state) for d_dense in self.d_dense_list]
            d_policy_list = [torch.distributions.OneHotCategorical(logits=logits) for logits in logits_list]
            d_policy = JointOneHotCategorical(d_policy_list)
        else:
            d_policy = None

        if self.c_action_size:
            l = self.c_dense(state)
            mean = self.mean_dense(l)
            logstd = self.logstd_dense(l)
            c_policy = torch.distributions.Normal(torch.tanh(mean / 5.) * 5., torch.exp(torch.clamp(logstd, -20, 0.5)))
        else:
            c_policy = None

        return d_policy, c_policy


class ModelTermination(nn.Module):
    def __init__(self, state_size):
        super().__init__()
        self.state_size = state_size

        self._build_model()

    def _build_model(self, dense_n=64, dense_depth=2,
                     dropout=0.):
        self.dense = LinearLayers(self.state_size, dense_n, dense_depth, output_size=1, dropout=dropout)

    def forward(self, state, obs_list):
        return torch.sigmoid(self.dense(state))

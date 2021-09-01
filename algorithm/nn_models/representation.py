import torch
from torch import nn


class ModelBaseSimpleRep(nn.Module):
    def __init__(self, obs_shapes):
        super().__init__()
        self.obs_shapes = obs_shapes

        self._build_model()

    def _build_model(self):
        pass

    def get_output_shape(self, device):
        output = self([torch.empty(1, *obs_shape, device=device) for obs_shape in self.obs_shapes])

        assert len(output.shape) == 2

        return output.shape[-1]

    def forward(self, obs_list):
        raise Exception("ModelSimpleRep not implemented")


class ModelSimpleRep(ModelBaseSimpleRep):
    def forward(self, obs_list):
        return torch.cat(obs_list, dim=-1)


class ModelBaseRNNRep(nn.Module):
    def __init__(self, obs_shapes, d_action_size, c_action_size):
        super().__init__()
        self.obs_shapes = obs_shapes
        self.d_action_size = d_action_size
        self.c_action_size = c_action_size

        self._build_model()

    def _build_model(self):
        pass

    def get_output_shape(self, device):
        obs_list = [torch.empty(1, 1, *obs_shape, device=device) for obs_shape in self.obs_shapes]
        pre_action = torch.empty(1, 1, self.d_action_size + self.c_action_size, device=device)
        output, next_rnn_state = self(obs_list, pre_action)

        return output.shape[-1], next_rnn_state.shape[1:]

    def forward(self, obs_list, pre_action, rnn_state=None):
        raise Exception("ModelRNNRep not implemented")

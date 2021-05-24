import torch
from torch import nn


def dense_layers(input_size, dense_n=64, dense_depth=0, output_size=None):
    true_ouput_dim = input_size
    dense = []
    for i in range(dense_depth):
        dense.append(nn.Linear(input_size if i == 0 else dense_n, dense_n))
        dense.append(nn.ReLU())
        true_ouput_dim = dense_n
    if output_size:
        dense.append(nn.Linear(true_ouput_dim, output_size))
        true_ouput_dim = output_size
    return nn.Sequential(*dense), true_ouput_dim


class GRU(nn.GRU):
    def forward(self, x: torch.Tensor, h0: torch.Tensor = None):
        output, hn = super().forward(x.transpose(0, 1).contiguous(),
                                     h0.transpose(0, 1).contiguous() if h0 is not None else None)

        return output.transpose(0, 1), hn.transpose(0, 1)

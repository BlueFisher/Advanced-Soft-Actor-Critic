import torch
from torch import nn


class LinearLayers(nn.Module):
    def __init__(self, input_size, dense_n=64, dense_depth=0, output_size=None):
        """
                 ┌────────┐
             ┌───► dense_n│
             │   └───┬────┘
        dense_depth  │
             │   ┌───▼────┐
             └───┤  relu  ├─────┐
                 └───┬────┘     │
                     │          │
               ┌─────▼──────┐   │
               │output_size │   │
               └─────┬──────┘   │
                     │          │
                     ▼          ▼

        """
        super().__init__()

        self.input_size = input_size
        self.output_size = input_size
        dense = []
        for i in range(dense_depth):
            linear = nn.Linear(input_size if i == 0 else dense_n, dense_n)
            nn.init.xavier_uniform_(linear.weight.data)
            torch.zero_(linear.bias.data)
            dense.append(linear)
            dense.append(nn.ReLU())
            self.output_size = dense_n
        if output_size:
            dense.append(nn.Linear(self.output_size, output_size))
            self.output_size = output_size

        self.dense = nn.Sequential(*dense)

    def forward(self, x):
        assert x.shape[-1] == self.input_size

        return self.dense(x)

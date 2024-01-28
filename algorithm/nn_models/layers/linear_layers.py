from typing import Optional

import torch
from torch import nn

acts = {}


def hook(layer, input, output):
    output = output.view(-1, output.shape[-1])
    new_a = output.shape[0]
    new_avg = output.mean(0)

    if layer not in acts:
        acts[layer] = (new_a, new_avg)
    else:
        a, avg = acts[layer]
        n = a + new_a
        new_avg = a / n * avg + (a + 1) / n * new_avg
        acts[layer] = (n, new_avg)

    print(id(layer), new_avg)
    # print(id(layer), (new_avg==0).sum().detach().cpu().numpy(), new_avg.shape[0])


class LinearLayers(nn.Module):
    def __init__(self, input_size, dense_n=64, dense_depth=0, output_size=None,
                 activation: Optional[nn.Module] = None,
                 dropout: float = 0.):
        """
                 ┌────────┐
             ┌───► dense_n│
             │   └───┬────┘
        dense_depth  │
             │   ┌───▼────┐
             └───┤  act   ├─────┐
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

        if activation is None:
            activation = nn.ReLU

        dense = []
        for i in range(dense_depth):
            linear = nn.Linear(input_size if i == 0 else dense_n, dense_n)
            nn.init.kaiming_uniform_(linear.weight.data)
            torch.zero_(linear.bias.data)
            dense.append(linear)

            act = activation()
            # act.register_forward_hook(hook)
            dense.append(act)
            dense.append(nn.Dropout(dropout))
            self.output_size = dense_n
        if output_size:
            linear = nn.Linear(self.output_size, output_size)
            nn.init.kaiming_uniform_(linear.weight.data)
            torch.zero_(linear.bias.data)
            dense.append(linear)
            dense.append(nn.Dropout(dropout))
            self.output_size = output_size

        self.dense = nn.Sequential(*dense)

    def forward(self, x):
        assert x.shape[-1] == self.input_size

        return self.dense(x)

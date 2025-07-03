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


class ResBlock(nn.Module):
    def __init__(self, input_size: int, output_size: int | None = None,
                 activation: nn.Module | None = None,
                 residual: bool = True):
        super().__init__()

        if output_size is None:
            output_size = input_size

        if residual and input_size != output_size:
            residual = False
        self.residual = residual

        if activation is None:
            activation = nn.GELU

        self.linear = nn.Linear(input_size, output_size)
        nn.init.kaiming_uniform_(self.linear.weight.data)
        torch.zero_(self.linear.bias.data)

        self.act = activation()

    def forward(self, x):
        assert x.shape[-1] == self.linear.in_features

        residual = x
        x = self.linear(x)
        x = self.act(x)

        if self.residual:
            return x + residual
        else:
            return x


class LinearLayers(nn.Module):
    def __init__(self, input_size, dense_n=64, dense_depth=0, output_size=None,
                 activation: nn.Module | None = None,
                 residual: bool = True,
                 dropout: float = 0.):
        """
                    ┌─────────┐      
                ┌───► dense_n ┼──┐   
                │   └────┬────┘  │   
                │        │       │   
        dense_depth ┌────▼────┐  │   
                │   │   act   │  │   
                │   └────┬────┘  │   
                │     ┌──▼──┐    │   
                └─────┤  +  ◄────┘   
                      └──┬──┘        
                         ├─────────┐ 
                   ┌─────▼──────┐  │ 
                   │output_size │  │ 
                   └─────┬──────┘  │ 
                         │         │ 
                         ▼         ▼ 
        """
        super().__init__()

        self.input_size = input_size
        self.output_size = input_size

        dense = []
        for i in range(dense_depth):
            res = ResBlock(input_size if i == 0 else dense_n, dense_n,
                           activation=activation,
                           residual=residual)

            dense.append(res)
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

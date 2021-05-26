from typing import Tuple, Union

import torch
from torch import nn


class LinearLayers(nn.Module):
    def __init__(self, input_size, dense_n=64, dense_depth=0, output_size=None):
        super().__init__()

        self.output_size = input_size
        dense = []
        for i in range(dense_depth):
            dense.append(nn.Linear(input_size if i == 0 else dense_n, dense_n))
            dense.append(nn.ReLU())
            self.output_size = dense_n
        if output_size:
            dense.append(nn.Linear(self.output_size, output_size))
            self.output_size = output_size

        self.dense = nn.Sequential(*dense)

    def forward(self, x):
        return self.dense(x)


def conv_output_shape(
    h_w: Tuple[int, int],
    kernel_size: Union[int, Tuple[int, int]] = 1,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> Tuple[int, int]:
    """
    Calculates the output shape (height and width) of the output of a convolution layer.
    kernel_size, stride, padding and dilation correspond to the inputs of the
    torch.nn.Conv2d layer (https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
    :param h_w: The height and width of the input.
    :param kernel_size: The size of the kernel of the convolution (can be an int or a
    tuple [width, height])
    :param stride: The stride of the convolution
    :param padding: The padding of the convolution
    :param dilation: The dilation of the convolution
    """
    from math import floor

    if not isinstance(kernel_size, tuple):
        kernel_size = (int(kernel_size), int(kernel_size))
    h = floor(
        ((h_w[0] + (2 * padding) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1
    )
    w = floor(
        ((h_w[1] + (2 * padding) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1
    )
    return h, w


def pool_out_shape(h_w: Tuple[int, int], kernel_size: int) -> Tuple[int, int]:
    """
    Calculates the output shape (height and width) of the output of a max pooling layer.
    kernel_size corresponds to the inputs of the
    torch.nn.MaxPool2d layer (https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html)
    :param kernel_size: The size of the kernel of the convolution
    """
    height = (h_w[0] - kernel_size) // 2 + 1
    width = (h_w[1] - kernel_size) // 2 + 1
    return height, width


def small_visual(height, width, channels):
    conv_1_hw = conv_output_shape((height, width), 3, 1)
    conv_2_hw = conv_output_shape(conv_1_hw, 3, 1)

    return nn.Sequential(
        nn.Conv2d(channels, 35, [3, 3], [1, 1]),
        nn.LeakyReLU(),
        nn.Conv2d(35, 144, [3, 3], [1, 1]),
        nn.LeakyReLU(),
    ), conv_2_hw, 144


def simple_visual(height, width, channels):
    conv_1_hw = conv_output_shape((height, width), 8, 4)
    conv_2_hw = conv_output_shape(conv_1_hw, 4, 2)

    return nn.Sequential(
        nn.Conv2d(channels, 16, [8, 8], [4, 4]),
        nn.LeakyReLU(),
        nn.Conv2d(16, 32, [4, 4], [2, 2]),
        nn.LeakyReLU(),
    ), conv_2_hw, 32


def nature_visual(height, width, channels):
    conv_1_hw = conv_output_shape((height, width), 8, 4)
    conv_2_hw = conv_output_shape(conv_1_hw, 4, 2)
    conv_3_hw = conv_output_shape(conv_2_hw, 3, 1)

    return nn.Sequential(
        nn.Conv2d(channels, 32, [8, 8], [4, 4]),
        nn.LeakyReLU(),
        nn.Conv2d(32, 64, [4, 4], [2, 2]),
        nn.LeakyReLU(),
        nn.Conv2d(64, 64, [3, 3], [1, 1]),
        nn.LeakyReLU(),
    ), conv_3_hw, 64


class ConvLayers(nn.Module):
    def __init__(self, height, width, channels,
                 conv,
                 dense_n=64,
                 dense_depth=0,
                 output_size=None):
        super().__init__()

        if isinstance(conv, str):
            if conv == 'small':
                self.conv_layers, (h, w), out_c = small_visual(height, width, channels)
            elif conv == 'simple':
                self.conv_layers, (h, w), out_c = simple_visual(height, width, channels)
            elif conv == 'nature':
                self.conv_layers, (h, w), out_c = nature_visual(height, width, channels)
        elif isinstance(conv, tuple):
            self.conv_layers, (h, w), out_c = conv
        else:
            raise RuntimeError('Argument conv should a Tuple[nn.Module, Tuple[int, int], int]')

        self.final_flat = h * w * out_c

        self.dense = LinearLayers(
            self.final_flat,
            dense_n,
            dense_depth,
            output_size
        )
        self.output_size = self.dense.output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 4:
            x = x.permute([0, 3, 1, 2])
            hidden = self.conv_layers(x)
            hidden = hidden.reshape(-1, self.final_flat)
            return self.dense(hidden)
        elif len(x.shape) == 5:
            batch = x.shape[:-3]
            x = x.reshape(-1, *x.shape[-3:])
            x = x.permute([0, 3, 1, 2])
            hidden = self.conv_layers(x)
            hidden = hidden.reshape(*batch, self.final_flat)
            return self.dense(hidden)
        else:
            raise Exception('The dimension of input should be larger than 4')


class GRU(nn.GRU):
    def forward(self, x: torch.Tensor, h0: torch.Tensor = None):
        output, hn = super().forward(x.transpose(0, 1).contiguous(),
                                     h0.transpose(0, 1).contiguous() if h0 is not None else None)

        return output.transpose(0, 1), hn.transpose(0, 1)

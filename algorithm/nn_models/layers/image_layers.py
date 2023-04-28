from typing import Optional, Tuple, Union

import torch
from torch import nn

from .linear_layers import LinearLayers


def conv1d_output_size(
        l: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1) -> int:

    from math import floor

    l_out = floor(
        ((l + (2 * padding) - (dilation * (kernel_size - 1)) - 1) / stride) + 1
    )

    return l_out


def conv2d_output_shape(
    h_w: Tuple[int, int],
    kernel_size: Union[int, Tuple[int, int]] = 1,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1
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


def convtranspose_output_shape(
    h_w: Tuple[int, int],
    kernel_size: Union[int, Tuple[int, int]] = 1,
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
    dilation: int = 1
) -> Tuple[int, int]:
    """
    https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html#torch.nn.ConvTranspose2d
    """

    if not isinstance(kernel_size, tuple):
        kernel_size = (int(kernel_size), int(kernel_size))

    h = (h_w[0] - 1) * stride - 2 * padding + dilation * (kernel_size[0] - 1) + output_padding + 1
    w = (h_w[1] - 1) * stride - 2 * padding + dilation * (kernel_size[1] - 1) + output_padding + 1

    return h, w


def default_conv1d(l, channels) -> Tuple[nn.Module, int, int]:
    conv_1_l = conv1d_output_size(l, 8, 4)
    conv_2_l = conv1d_output_size(conv_1_l, 4, 2)

    return nn.Sequential(
        nn.Conv1d(channels, 16, 8, 4),
        nn.LeakyReLU(),
        nn.Conv1d(16, 32, 4, 2),
        nn.LeakyReLU(),
    ), conv_2_l, 32


class Conv1dLayers(nn.Module):
    def __init__(self, in_l: int, in_channels: int,
                 conv: Union[str, Tuple[nn.Module, int, int]],
                 out_dense_n: int = 64, out_dense_depth: int = 0, output_size: int = None):
        super().__init__()

        if isinstance(conv, str):
            if conv == 'default':
                self.conv_layers, l, out_c = default_conv1d(in_l, in_channels)
            else:
                raise RuntimeError(f'No pre-defined {conv} convolutional layer')
        elif isinstance(conv, tuple):
            self.conv_layers, l, out_c = conv
        else:
            raise RuntimeError('Argument conv should a Tuple[nn.Module, Tuple[int, int], int]')

        self.conv_output_size = l * out_c

        self.dense = LinearLayers(
            self.conv_output_size,
            out_dense_n,
            out_dense_depth,
            output_size
        )
        self.output_size = self.dense.output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) >= 3:
            batch = x.shape[:-2]
            x = x.reshape(-1, *x.shape[-2:])
            x = x.permute([0, 2, 1])
            hidden = self.conv_layers(x)
            hidden = hidden.reshape(*batch, self.conv_output_size)
            return self.dense(hidden)
        else:
            raise Exception('The dimension of input should be greater than or equal to 3')


def small_visual(height, width, channels) -> Tuple[nn.Module, int, int]:
    conv_1_hw = conv2d_output_shape((height, width), 3, 1)
    conv_2_hw = conv2d_output_shape(conv_1_hw, 3, 1)

    return nn.Sequential(
        nn.Conv2d(channels, 35, [3, 3], [1, 1]),
        nn.LeakyReLU(),
        nn.Conv2d(35, 144, [3, 3], [1, 1]),
        nn.LeakyReLU(),
    ), conv_2_hw, 144


def simple_visual(height, width, channels) -> Tuple[nn.Module, int, int]:
    conv_1_hw = conv2d_output_shape((height, width), 8, 4)
    conv_2_hw = conv2d_output_shape(conv_1_hw, 4, 2)

    return nn.Sequential(
        nn.Conv2d(channels, 16, [8, 8], [4, 4]),
        nn.LeakyReLU(),
        nn.Conv2d(16, 32, [4, 4], [2, 2]),
        nn.LeakyReLU(),
    ), conv_2_hw, 32


def nature_visual(height, width, channels) -> Tuple[nn.Module, int, int]:
    conv_1_hw = conv2d_output_shape((height, width), 8, 4)
    conv_2_hw = conv2d_output_shape(conv_1_hw, 4, 2)
    conv_3_hw = conv2d_output_shape(conv_2_hw, 3, 1)

    return nn.Sequential(
        nn.Conv2d(channels, 32, [8, 8], [4, 4]),
        nn.LeakyReLU(),
        nn.Conv2d(32, 64, [4, 4], [2, 2]),
        nn.LeakyReLU(),
        nn.Conv2d(64, 64, [3, 3], [1, 1]),
        nn.LeakyReLU(),
    ), conv_3_hw, 64


class ConvLayers(nn.Module):
    def __init__(self, in_height: int, in_width: int, in_channels: int,
                 conv: Union[str, Tuple[nn.Module, Tuple[int, int], int]],
                 out_dense_n: int = 64, out_dense_depth: int = 0, output_size: int = None):
        super().__init__()

        if isinstance(conv, str):
            if conv == 'small':
                self.conv_layers, (h, w), out_c = small_visual(in_height, in_width, in_channels)
            elif conv == 'simple':
                self.conv_layers, (h, w), out_c = simple_visual(in_height, in_width, in_channels)
            elif conv == 'nature':
                self.conv_layers, (h, w), out_c = nature_visual(in_height, in_width, in_channels)
            else:
                raise RuntimeError(f'No pre-defined {conv} convolutional layer')
        elif isinstance(conv, tuple):
            self.conv_layers, (h, w), out_c = conv
        else:
            raise RuntimeError('Argument conv should a Tuple[nn.Module, Tuple[int, int], int]')

        self.conv_output_size = h * w * out_c

        self.dense = LinearLayers(
            self.conv_output_size,
            out_dense_n,
            out_dense_depth,
            output_size
        )
        self.output_size = self.dense.output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) >= 4:
            batch = x.shape[:-3]
            x = x.reshape(-1, *x.shape[-3:])
            x = x.permute([0, 3, 1, 2])
            hidden = self.conv_layers(x)
            hidden = hidden.reshape(*batch, self.conv_output_size)
            return self.dense(hidden)
        else:
            raise Exception('The dimension of input should be greater than or equal to 4')


class ConvTransposeLayers(nn.Module):
    def __init__(self, input_size: int, in_dense_n: int, in_dense_depth: int,
                 height: int, width: int, channels: int,
                 conv_transpose: nn.Module):
        super().__init__()

        self._height = height
        self._width = width
        self._channels = channels

        self.dense = LinearLayers(input_size, in_dense_n, in_dense_depth,
                                  height * width * channels)

        self.conv_transpose = conv_transpose

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense(x)

        if len(x.shape) >= 2:
            batch = x.shape[:-1]
            x = x.reshape(-1, self._channels, self._height, self._width)
            vis = self.conv_transpose(x)
            vis = vis.permute([0, 2, 3, 1])
            return vis.reshape(*batch, *vis.shape[1:])
        else:
            raise Exception('The dimension of input should be greater than or equal to 2')


class Transform(nn.Module):
    def __init__(self, transform):
        super().__init__()

        self.transform = transform

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) >= 4:
            batch = x.shape[:-3]
            x = x.reshape(-1, *x.shape[-3:])
            x = x.permute([0, 3, 1, 2])
            x = self.transform(x).permute([0, 2, 3, 1])
            return x.reshape(*batch, *x.shape[1:])
        else:
            raise Exception('The dimension of input should be greater than or equal to 4')

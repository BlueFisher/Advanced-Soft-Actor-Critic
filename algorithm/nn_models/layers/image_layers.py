import math
from functools import partial
from typing import Callable

import torch
from torch import nn
from torchvision.models.vision_transformer import Encoder

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
    h_w: tuple[int, int],
    kernel_size: int | tuple[int, int] = 1,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1
) -> tuple[int, int]:
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


def pool_out_shape(h_w: tuple[int, int], kernel_size: int) -> tuple[int, int]:
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
    h_w: tuple[int, int],
    kernel_size: int | tuple[int, int] = 1,
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
    dilation: int = 1
) -> tuple[int, int]:
    """
    https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html#torch.nn.ConvTranspose2d
    """

    if not isinstance(kernel_size, tuple):
        kernel_size = (int(kernel_size), int(kernel_size))

    h = (h_w[0] - 1) * stride - 2 * padding + dilation * (kernel_size[0] - 1) + output_padding + 1
    w = (h_w[1] - 1) * stride - 2 * padding + dilation * (kernel_size[1] - 1) + output_padding + 1

    return h, w


def default_conv1d(l, channels) -> tuple[nn.Module, int, int]:
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
                 conv: str | tuple[nn.Module, int, int],
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
            raise RuntimeError('Argument conv should a tuple[nn.Module, tuple[int, int], int]')

        self.conv_output_size = l * out_c

        self.dense = LinearLayers(
            self.conv_output_size,
            out_dense_n,
            out_dense_depth,
            output_size
        )
        self.output_size = self.dense.output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() >= 3:
            batch = x.shape[:-2]
            x = x.reshape(-1, *x.shape[-2:])
            x = x.permute([0, 2, 1])
            hidden = self.conv_layers(x)
            hidden = hidden.reshape(*batch, self.conv_output_size)
            return self.dense(hidden)
        else:
            raise Exception('The dimension of input should be greater than or equal to 3')


def small_visual(height, width, channels) -> tuple[nn.Module, int, int]:
    conv_1_hw = conv2d_output_shape((height, width), 3, 1)
    conv_2_hw = conv2d_output_shape(conv_1_hw, 3, 1)

    return nn.Sequential(
        nn.Conv2d(channels, 35, [3, 3], [1, 1]),
        nn.LeakyReLU(),
        nn.Conv2d(35, 144, [3, 3], [1, 1]),
        nn.LeakyReLU(),
    ), conv_2_hw, 144


def simple_visual(height, width, channels) -> tuple[nn.Module, int, int]:
    conv_1_hw = conv2d_output_shape((height, width), 8, 4)
    conv_2_hw = conv2d_output_shape(conv_1_hw, 4, 2)

    return nn.Sequential(
        nn.Conv2d(channels, 16, [8, 8], [4, 4]),
        nn.LeakyReLU(),
        nn.Conv2d(16, 32, [4, 4], [2, 2]),
        nn.LeakyReLU(),
    ), conv_2_hw, 32


def nature_visual(height, width, channels) -> tuple[nn.Module, int, int]:
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
                 conv: str | tuple[nn.Module, tuple[int, int], int],
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
            raise RuntimeError('Argument conv should a tuple[nn.Module, tuple[int, int], int]')

        self.conv_output_size = h * w * out_c

        self.dense = LinearLayers(
            self.conv_output_size,
            out_dense_n,
            out_dense_depth,
            output_size
        )
        self.output_size = self.dense.output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() >= 4:
            batch = x.shape[:-3]
            x = x.reshape(-1, *x.shape[-3:])
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

        if x.dim() >= 2:
            batch = x.shape[:-1]
            x = x.reshape(-1, self._channels, self._height, self._width)
            vis = self.conv_transpose(x)
            return vis.reshape(*batch, *vis.shape[1:])
        else:
            raise Exception('The dimension of input should be greater than or equal to 2')


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int,
        in_channels: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.norm_layer = norm_layer

        self.conv_proj = nn.Conv2d(
            in_channels=in_channels, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
        )

        seq_length = (image_size // patch_size) ** 2

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        self.seq_length = seq_length

        # Init the patchify stem
        fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
        nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
        if self.conv_proj.bias is not None:
            nn.init.zeros_(self.conv_proj.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor):
        if x.dim() >= 4:
            batch = x.shape[:-3]

            x = x.reshape(-1, *x.shape[-3:])

            # Reshape and permute the input tensor
            x = self._process_input(x)
            n = x.shape[0]

            # Expand the class token to the full batch
            batch_class_token = self.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)

            x = self.encoder(x)

            # Classifier "token" as used by standard language architectures
            x = x[:, 0]

            x = x.reshape(*batch, self.hidden_dim)

            return x
        else:
            raise Exception('The dimension of input should be greater than or equal to 4')


class Transform(nn.Module):
    def __init__(self, transform: Callable[[torch.Tensor], torch.Tensor] | None = None):
        super().__init__()

        self.transform = transform

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.transform is None:
            return x

        if x.dim() >= 4:
            batch = x.shape[:-3]
            x = x.reshape(-1, *x.shape[-3:])
            x = self.transform(x)
            return x.reshape(*batch, *x.shape[1:])
        else:
            raise Exception('The dimension of input should be greater than or equal to 4')

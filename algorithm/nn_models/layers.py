from typing import Optional, Tuple, Union

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


def conv1d_output_size(
        l: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1):

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
):
    """
    https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html#torch.nn.ConvTranspose2d
    """

    if not isinstance(kernel_size, tuple):
        kernel_size = (int(kernel_size), int(kernel_size))

    h = (h_w[0] - 1) * stride - 2 * padding + dilation * (kernel_size[0] - 1) + output_padding + 1
    w = (h_w[1] - 1) * stride - 2 * padding + dilation * (kernel_size[1] - 1) + output_padding + 1

    return h, w


def default_conv1d(l, channels):
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


def small_visual(height, width, channels):
    conv_1_hw = conv2d_output_shape((height, width), 3, 1)
    conv_2_hw = conv2d_output_shape(conv_1_hw, 3, 1)

    return nn.Sequential(
        nn.Conv2d(channels, 35, [3, 3], [1, 1]),
        nn.LeakyReLU(),
        nn.Conv2d(35, 144, [3, 3], [1, 1]),
        nn.LeakyReLU(),
    ), conv_2_hw, 144


def simple_visual(height, width, channels):
    conv_1_hw = conv2d_output_shape((height, width), 8, 4)
    conv_2_hw = conv2d_output_shape(conv_1_hw, 4, 2)

    return nn.Sequential(
        nn.Conv2d(channels, 16, [8, 8], [4, 4]),
        nn.LeakyReLU(),
        nn.Conv2d(16, 32, [4, 4], [2, 2]),
        nn.LeakyReLU(),
    ), conv_2_hw, 32


def nature_visual(height, width, channels):
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


class GRU(nn.GRU):
    def forward(self, x: torch.Tensor, h0: torch.Tensor = None):
        if h0 is not None:
            h0 = h0.transpose(0, 1).contiguous()

        output, hn = super().forward(x.transpose(0, 1).contiguous(), h0)

        return output.transpose(0, 1), hn.transpose(0, 1)


class LSTM(nn.LSTM):
    def forward(self, x: torch.Tensor, hc_0: torch.Tensor = None):
        if hc_0 is not None:
            hc_0 = hc_0.transpose(0, 1)
            h0, c0 = torch.chunk(hc_0, 2, dim=-1)
            h0 = h0.contiguous()
            c0 = c0.contiguous()
            hc_0 = (h0, c0)

        output, (hn, cn) = super().forward(x.transpose(0, 1).contiguous(), hc_0)

        return output.transpose(0, 1), torch.cat([hn, cn], dim=-1).transpose(0, 1)


class MultiheadAttention(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads,
                 dropout=0,
                 bias=True,
                 add_bias_kv=False,
                 add_zero_attn=False,
                 kdim=None,
                 vdim=None,
                 device=None,
                 dtype=None) -> None:
        super().__init__(embed_dim, num_heads,
                         dropout=dropout,
                         bias=bias,
                         add_bias_kv=add_bias_kv,
                         add_zero_attn=add_zero_attn,
                         kdim=kdim,
                         vdim=vdim,
                         batch_first=True,
                         device=device,
                         dtype=dtype)


class EpisodeMultiheadAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.attn = MultiheadAttention(embed_dim, num_heads)
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.dense = LinearLayers(embed_dim, embed_dim, 1, embed_dim)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)

    def get_attn_mask(self,
                      key_length: int,
                      query_length: int,
                      key_padding_mask=None,
                      device='cpu'):
        """
        Args:
            key_length: int
            query_length: int
            key_padding_mask: [Batch, key_length]
        """
        triu = torch.triu(torch.ones(key_length, key_length, dtype=bool, device=device), diagonal=1)
        attn_mask = triu[-query_length:]  # [query_length, key_length]

        if key_padding_mask is not None:
            batch_size = key_padding_mask.shape[0]

            attn_mask = attn_mask.repeat(batch_size * self.num_heads, 1, 1)  # [Batch * num_heads, query_length, key_length]
            key_padding_mask = key_padding_mask.repeat(self.num_heads, 1)  # [Batch * num_heads, key_length]
            key_padding_mask = key_padding_mask.unsqueeze(1)  # [Batch * num_heads, 1, key_length]
            attn_mask = torch.logical_or(attn_mask, key_padding_mask)  # [Batch * num_heads, query_length, key_length]
            eye = torch.eye(key_length, dtype=bool, device=device)
            eye = ~eye[-query_length:]  # [query_length, key_length]
            eye = eye.repeat(batch_size * self.num_heads, 1, 1)  # [Batch * num_heads, query_length, key_length]
            attn_mask = torch.logical_and(attn_mask, eye)

        return attn_mask

    def forward(self,
                key: torch.Tensor,
                query_length: int = 1,
                key_padding_mask: Optional[torch.Tensor] = None):
        """
        Args:
            key: [Batch, key_length, embed_dim]
            query_length: int
            key_padding_mask: [Batch, key_padding_mask_length], key_padding_mask_length could be shorter than key_length
        """
        key_length = key.shape[1]

        if key_padding_mask is not None:
            key_padding_mask_length = key_padding_mask.shape[1]
            assert key_padding_mask_length <= key_length

            key_padding_mask = torch.concat([
                key_padding_mask[:, :1].repeat(1, key_length - key_padding_mask_length),
                key_padding_mask
            ], dim=1)

        attn_mask = self.get_attn_mask(key_length,
                                       query_length,
                                       key_padding_mask=key_padding_mask,
                                       device=key.device)

        query = key[:, -query_length:]
        output, attn_weights = self.attn(query, key, key,
                                         attn_mask=attn_mask)
        output += query
        output = _t = self.layer_norm_1(output)
        output = self.dense(output)
        output += _t
        output = self.layer_norm_2(output)

        return output, attn_weights


class EpisodeMultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int,
                 num_layers: int = 2):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self._attn_list = nn.ModuleList(
            [EpisodeMultiheadAttentionBlock(embed_dim, num_heads) for _ in range(num_layers)]
        )

    def forward(self,
                key: torch.Tensor,
                query_length: int = 1,
                hidden_state: Optional[torch.Tensor] = None,
                is_prev_hidden_state: bool = False,
                key_padding_mask: Optional[torch.Tensor] = None):
        """
        Args:
            [Batch, key_length, embed_dim],
            query_length: int
            hidden_state: [Batch, hidden_state_length, embed_dim]
            is_prev_hidden_state: bool
            key_padding_mask: [Batch, key_length]
        Returns:
            encoded_query: [Batch, query_length, embed_dim]
            next_hidden_state: [Batch, query_length, embed_dim * num_layers]
            attn_weights_list: List[[Batch, query_length, key_length_i], ...]
        """
        key_length = key.shape[1]
        assert query_length <= key_length

        next_hidden_state_list = []
        attn_weights_list = []

        if hidden_state is None:
            _k = key
            for attn in self._attn_list[:-1]:
                output, attn_weight = attn(_k, key_length,
                                           key_padding_mask=key_padding_mask)
                _k = output
                _q = _k[:, -query_length:]
                next_hidden_state_list.append(_q)
                attn_weights_list.append(attn_weight[:, -query_length:])

            output, attn_weight = self._attn_list[-1](_k, query_length,
                                                      key_padding_mask=key_padding_mask)
            attn_weights_list.append(attn_weight)
            _q = output

        elif not is_prev_hidden_state:
            output, attn_weight = self._attn_list[0](key, query_length,
                                                     key_padding_mask=key_padding_mask)
            attn_weights_list.append(attn_weight)

            hidden_state_list = hidden_state.chunk(self.num_layers - 1, dim=-1)

            for i, attn in enumerate(self._attn_list[1:]):
                next_hidden_state_list.append(output)

                _k = torch.concat([hidden_state_list[i], output], dim=1)

                output, attn_weight = attn(_k, query_length,
                                           key_padding_mask=key_padding_mask)
                attn_weights_list.append(attn_weight)

            _q = output

        elif is_prev_hidden_state:
            output, attn_weight = self._attn_list[0](key, key_length,
                                                     key_padding_mask=key_padding_mask)
            next_hidden_state_list.append(output[:, -query_length:])
            attn_weights_list.append(attn_weight[:, -query_length:])

            hidden_state_list = hidden_state.chunk(self.num_layers - 1, dim=-1)

            for i, attn in enumerate(self._attn_list[1:-1]):
                _k = output[:, -key_length:]
                _k = torch.concat([hidden_state_list[i], _k], dim=1)

                output, attn_weight = attn(_k, key_length,
                                           key_padding_mask=key_padding_mask)
                next_hidden_state_list.append(output[:, -query_length:])
                attn_weights_list.append(attn_weight[:, -query_length:])

            _k = output[:, -key_length:]
            _k = torch.concat([hidden_state_list[-1], _k], dim=1)

            output, attn_weight = self._attn_list[-1](_k, query_length,
                                                      key_padding_mask=key_padding_mask)
            attn_weights_list.append(attn_weight)

            _q = output

        return _q, torch.concat(next_hidden_state_list, dim=-1), attn_weights_list


if __name__ == '__main__':
    import time

    batch_size = 16
    embed_dim = 4
    query_length = 2
    key_length = 4

    num_layers = 4

    m = EpisodeMultiheadAttention(embed_dim, 2, num_layers).to('cuda')

    key_padding_mask = torch.randint(0, 2, (batch_size, key_length), dtype=torch.bool).to('cuda')

    key = torch.rand((batch_size, key_length, embed_dim)).to('cuda')
    hidden_state = torch.rand((batch_size, 5, embed_dim * (num_layers - 1))).to('cuda')

    t = time.time()

    o, hn, attn_weights = m(key, query_length, None, False, key_padding_mask=key_padding_mask)
    print(o.shape, hn.shape, attn_weights.shape)
    o, hn, attn_weights = m(key, query_length, hidden_state, False, key_padding_mask=key_padding_mask)
    print(o.shape, hn.shape, attn_weights.shape)
    o, hn, attn_weights = m(key, query_length, hidden_state, True, key_padding_mask=key_padding_mask)
    print(o.shape, hn.shape, attn_weights.shape)

    print(time.time() - t)

    o.mean().backward()
    print([p.grad is not None for p in m.parameters()])

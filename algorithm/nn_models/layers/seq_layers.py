import math
from typing import Optional, Tuple

import torch
from torch import nn


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


# The extension of vanilla MultiheadAttention
# Treat the last two dimensions as seq_len and embed_dim, all previous dimensions as batch
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

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                need_weights: bool = True,
                attn_mask: Optional[torch.Tensor] = None,
                average_attn_weights: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        if len(query.shape) >= 3:
            batch = query.shape[:-2]
            query = query.reshape(-1, *query.shape[-2:])
            key = key.reshape(-1, *key.shape[-2:])
            value = value.reshape(-1, *value.shape[-2:])
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.reshape(-1, key_padding_mask.shape[-1])

            attn_output, attn_output_weights = super().forward(query,
                                                               key,
                                                               value,
                                                               key_padding_mask,
                                                               need_weights,
                                                               attn_mask,
                                                               average_attn_weights)

            attn_output = attn_output.reshape(*batch, *attn_output.shape[1:])
            attn_output_weights = attn_output_weights.reshape(*batch, *attn_output_weights.shape[1:])

            return attn_output, attn_output_weights
        else:
            raise Exception('The dimension of input should be greater than or equal to 3')


class EpisodeMultiheadAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int,
                 use_residual: bool = True,
                 use_layer_norm: bool = True):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm

        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(self.embed_dim)

        self.attn = MultiheadAttention(embed_dim, num_heads)

    def get_attn_mask(self,
                      key_length: int,
                      query_length: int,
                      query_only_attend_to_reset_key=False,
                      key_padding_mask: Optional[torch.Tensor] = None,
                      device='cpu') -> torch.Tensor:
        """
        Args:
            key_length: int
            query_length: int
            query_only_attend_to_reset_key: bool
                Whether the query only need to attend to the key-query and itself (useful in OC)
            key_padding_mask: [batch, key_length]
                A `True` value indicates that the corresponding key value will be ignored 
                for the purpose of attention.
        """
        triu = torch.triu(torch.ones(key_length, key_length, dtype=bool, device=device), diagonal=1)
        attn_mask = triu[-query_length:]  # [query_length, key_length]

        if query_only_attend_to_reset_key:
            _attn_mask = torch.concat([
                torch.zeros((query_length, key_length - query_length), dtype=bool, device=device),  # [query_length, key_length - query_length]
                ~torch.eye(query_length, dtype=bool, device=device)  # [query_length, query_length]
            ], dim=1)
            attn_mask = torch.logical_or(attn_mask, _attn_mask)  # [query_length, key_length]

        if key_padding_mask is not None:
            batch_size = key_padding_mask.shape[0]

            attn_mask = attn_mask.repeat(batch_size * self.num_heads, 1, 1)  # [Batch * num_heads, query_length, key_length]
            key_padding_mask = key_padding_mask.repeat(self.num_heads, 1)  # [Batch * num_heads, key_length]
            key_padding_mask = key_padding_mask.unsqueeze(1)  # [Batch * num_heads, 1, key_length]
            attn_mask = torch.logical_or(attn_mask, key_padding_mask)  # [Batch * num_heads, query_length, key_length]

            # Preventing NAN, each element should attend to it self.
            eye = torch.eye(key_length, dtype=bool, device=device)
            eye = ~eye[-query_length:]  # [query_length, key_length]
            eye = eye.repeat(batch_size * self.num_heads, 1, 1)  # [Batch * num_heads, query_length, key_length]
            attn_mask = torch.logical_and(attn_mask, eye)

        return attn_mask

    def forward(self,
                key: torch.Tensor,
                query_length: int,
                query_only_attend_to_reset_key=False,
                key_padding_mask: Optional[torch.Tensor] = None):
        """
        Args:
            key: [batch, key_length, embed_dim]
            query_length: int
            query_only_attend_to_reset_key: bool, Whether the query only need to attend to the key-query and itself
            key_padding_mask: [batch, key_padding_mask_length]
                A `True` value indicates that the corresponding key value will be ignored 
                for the purpose of attention.
                `key_padding_mask_length` could be shorter than key_length.
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
                                       query_only_attend_to_reset_key=query_only_attend_to_reset_key,
                                       key_padding_mask=key_padding_mask,
                                       device=key.device)

        query = key[:, -query_length:]
        output, attn_weights = self.attn(query, key, key,
                                         attn_mask=attn_mask)

        if self.use_layer_norm:
            output = self.layer_norm(output)

        if self.use_residual:
            output = output + query

        return output, attn_weights


class EpisodeMultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int,
                 num_layers: int = 2,
                 use_residual: bool = True,
                 use_layer_norm: bool = False):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self._attn_list = nn.ModuleList(
            [EpisodeMultiheadAttentionBlock(embed_dim, num_heads,
                                            use_residual=use_residual,
                                            use_layer_norm=use_layer_norm) for _ in range(num_layers)]
        )

    def forward(self,
                key: torch.Tensor,
                query_length: int = 1,
                hidden_state: Optional[torch.Tensor] = None,
                is_prev_hidden_state: bool = False,

                query_only_attend_to_reset_key: bool = False,
                key_padding_mask: Optional[torch.Tensor] = None):
        """
        Args:
            key: [batch, key_length, embed_dim],
            query_length: int
            hidden_state: [batch, hidden_state_length, embed_dim]
            is_prev_hidden_state: bool

            query_only_attend_to_reset_key: bool
            key_padding_mask: [batch, key_length]

        Returns:
            encoded_query: [batch, query_length, embed_dim]
            next_hidden_state: [batch, query_length, embed_dim * num_layers]
            attn_weights_list: List[[batch, query_length, key_length_i], ...]
        """
        key_length = key.shape[1]
        assert query_length <= key_length

        next_hidden_state_list = []
        attn_weights_list = []

        if hidden_state is None:
            _k = key
            for attn in self._attn_list[:-1]:
                output, attn_weight = attn(_k, key_length,
                                           query_only_attend_to_reset_key=query_only_attend_to_reset_key,
                                           key_padding_mask=key_padding_mask)
                _k = output
                _q = _k[:, -query_length:]
                next_hidden_state_list.append(_q)
                attn_weights_list.append(attn_weight[:, -query_length:])

            output, attn_weight = self._attn_list[-1](_k, query_length,
                                                      query_only_attend_to_reset_key=query_only_attend_to_reset_key,
                                                      key_padding_mask=key_padding_mask)
            attn_weights_list.append(attn_weight)
            _q = output

        elif not is_prev_hidden_state:
            output, attn_weight = self._attn_list[0](key, query_length,
                                                     query_only_attend_to_reset_key=query_only_attend_to_reset_key,
                                                     key_padding_mask=key_padding_mask)
            attn_weights_list.append(attn_weight)

            if self.num_layers > 1:
                hidden_state_list = hidden_state.chunk(self.num_layers - 1, dim=-1)

            for i, attn in enumerate(self._attn_list[1:]):
                next_hidden_state_list.append(output)

                _k = torch.concat([hidden_state_list[i], output], dim=1)

                output, attn_weight = attn(_k, query_length,
                                           query_only_attend_to_reset_key=query_only_attend_to_reset_key,
                                           key_padding_mask=key_padding_mask)
                attn_weights_list.append(attn_weight)

            _q = output

        elif is_prev_hidden_state:
            output, attn_weight = self._attn_list[0](key, key_length,
                                                     query_only_attend_to_reset_key=query_only_attend_to_reset_key,
                                                     key_padding_mask=key_padding_mask)
            next_hidden_state_list.append(output[:, -query_length:])
            attn_weights_list.append(attn_weight[:, -query_length:])

            if self.num_layers > 1:
                hidden_state_list = hidden_state.chunk(self.num_layers - 1, dim=-1)
            else:
                output = output[:, -query_length:]

            for i, attn in enumerate(self._attn_list[1:-1]):
                _k = output[:, -key_length:]
                _k = torch.concat([hidden_state_list[i], _k], dim=1)

                output, attn_weight = attn(_k, key_length,
                                           query_only_attend_to_reset_key=query_only_attend_to_reset_key,
                                           key_padding_mask=key_padding_mask)
                next_hidden_state_list.append(output[:, -query_length:])
                attn_weights_list.append(attn_weight[:, -query_length:])

            if self.num_layers > 1:
                _k = output[:, -key_length:]
                _k = torch.concat([hidden_state_list[-1], _k], dim=1)

                output, attn_weight = self._attn_list[-1](_k, query_length,
                                                          query_only_attend_to_reset_key=query_only_attend_to_reset_key,
                                                          key_padding_mask=key_padding_mask)
                attn_weights_list.append(attn_weight)

            _q = output

        if self.num_layers > 1:
            return _q, torch.concat(next_hidden_state_list, dim=-1), attn_weights_list
        else:
            return _q, torch.zeros(key.shape[0], query_length, 1), attn_weights_list


class AbsolutePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        self.register_buffer('pe', pe)

    def forward(self, indexes):
        with torch.no_grad():
            return self.pe[indexes.type(torch.int64)]

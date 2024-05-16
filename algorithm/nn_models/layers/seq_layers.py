import math
from enum import Enum
from typing import List, Optional

import torch
from torch import nn

if __name__ == '__main__':
    from linear_layers import LinearLayers
else:
    from .linear_layers import LinearLayers


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


class POSITIONAL_ENCODING(Enum):
    ABSOLUTE = 1
    ABSOLUTE_CAT = 2
    ROPE = 3
    ROPE2 = 4


class GATE(Enum):
    RESIDUAL = 1
    OUTPUT = 2
    RECURRENT = 3
    CAT = 4


class MultiheadAttention(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 1,
                 pe: Optional[POSITIONAL_ENCODING] = None,
                 qkv_dense_depth: int = 0,
                 out_dense_depth: int = 1,
                 dropout: float = 0.) -> None:

        super().__init__()

        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.pe = pe

        _ori_embed_dim = embed_dim
        if pe == POSITIONAL_ENCODING.ABSOLUTE:
            self.abpe = AbsolutePositionalEncoding(embed_dim)
        elif pe == POSITIONAL_ENCODING.ABSOLUTE_CAT:
            self.abpe = AbsolutePositionalEncoding(embed_dim)
            _ori_embed_dim = embed_dim * 2
        elif pe == POSITIONAL_ENCODING.ROPE:
            self.rope = RotaryPositionalEncoding(embed_dim)
        elif pe == POSITIONAL_ENCODING.ROPE2:
            self.rope = RotaryPositionalEncoding2(embed_dim)

        self.dropout = dropout

        self.q_proj = LinearLayers(_ori_embed_dim, dense_n=embed_dim, dense_depth=qkv_dense_depth, output_size=embed_dim, dropout=dropout)
        self.k_proj = LinearLayers(_ori_embed_dim, dense_n=embed_dim, dense_depth=qkv_dense_depth, output_size=embed_dim, dropout=dropout)
        self.v_proj = LinearLayers(_ori_embed_dim, dense_n=embed_dim, dense_depth=qkv_dense_depth, output_size=embed_dim, dropout=dropout)

        self.out_proj = LinearLayers(embed_dim, dense_n=embed_dim, dense_depth=out_dense_depth, dropout=dropout)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                query_index: Optional[torch.Tensor] = None,
                key_index: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None):
        """
        Args:
            query: [batch, seq_q_len, embed_dim]
            key: [batch, seq_k_len, embed_dim]
            value: [batch, seq_k_len, embed_dim]
            query_index: [batch, seq_q_len]
            key_index: [batch, seq_k_len]
            key_padding_mask: [batch, seq_k_len]
                A `True` value indicates that the corresponding key value will be ignored 
                for the purpose of attention.
            attn_mask: [batch, seq_q_len, seq_k_len] OR [seq_q_len, seq_k_len]

        Returns:
            attn_output: [batch, seq_q_len, embed_dim]
            attn_output_weights: [batch, seq_q_len, seq_k_len]
        """

        batch = query.shape[:-2]
        query = query.reshape(-1, *query.shape[-2:])
        key = key.reshape(-1, *key.shape[-2:])
        value = value.reshape(-1, *value.shape[-2:])
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.reshape(-1, key_padding_mask.shape[-1])
        if attn_mask is not None and len(attn_mask.shape) >= 3:
            attn_mask = attn_mask.reshape(-1, *attn_mask.shape[-2:])

        if self.pe is not None and query_index is None:
            query_index = torch.arange(query.shape[1], device=query.device)
        if self.pe is not None and key_index is None:
            key_index = torch.arange(key.shape[1], device=key.device)

        if self.pe == POSITIONAL_ENCODING.ABSOLUTE:
            pe = self.abpe(query_index)
            query = pe + query
            pe = self.abpe(key_index)
            key = pe + key
            value = pe + value
        elif self.pe == POSITIONAL_ENCODING.ABSOLUTE_CAT:
            pe = self.abpe(query_index)
            query = torch.concat([query, pe], dim=-1)
            pe = self.abpe(key_index)
            key = torch.concat([key, pe], dim=-1)
            value = torch.concat([value, pe], dim=-1)

        q = self.q_proj(query)  # [bsz, seq_q_len, embed_dim]
        k = self.k_proj(key)  # [bsz, seq_k_len, embed_dim]
        v = self.v_proj(value)  # [bsz, seq_k_len, embed_dim]

        if self.pe in (POSITIONAL_ENCODING.ROPE, POSITIONAL_ENCODING.ROPE2):
            q, k = self.rope(query_index, key_index, q, k)

        if self.num_heads > 1:
            q = torch.cat(q.chunk(self.num_heads, dim=-1), dim=0)  # [bsz * num_heads, seq_q_len, head_dim]
            k = torch.cat(k.chunk(self.num_heads, dim=-1), dim=0)  # [bsz * num_heads, seq_k_len, head_dim]
            v = torch.cat(v.chunk(self.num_heads, dim=-1), dim=0)  # [bsz * num_heads, seq_k_len, head_dim]

        q_scaled = q / math.sqrt(self.head_dim)  # [bsz * num_heads, seq_q_len, head_dim]

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1)  # [bsz, 1, seq_k_len]
            if attn_mask is None:
                attn_mask = key_padding_mask.repeat_interleave(q.shape[1], dim=1)  # [bsz, seq_q_len, seq_k_len]
            else:
                attn_mask = torch.logical_or(attn_mask, key_padding_mask)  # [bsz, seq_q_len, seq_k_len]

        if attn_mask is not None:
            # [bsz, seq_q_len, seq_k_len] OR [seq_q_len, seq_k_len]
            attn_mask = torch.zeros_like(attn_mask, dtype=query.dtype,
                                         device=query.device).masked_fill_(attn_mask, float("-inf"))

            if len(attn_mask.shape) == 3:
                # [bsz, seq_q_len, seq_k_len] -> [bsz * num_heads, seq_q_len, seq_k_len]
                attn_mask = attn_mask.repeat(self.num_heads, 1, 1)

            # Prevent NAN
            nan_attn_mask = attn_mask.all(dim=-1)  # [bsz * num_heads, seq_q_len]
            attn_mask[nan_attn_mask] = torch.zeros_like(attn_mask[-1, -1])

            # [bsz * num_heads, seq_q_len, seq_k_len]
            attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
        else:
            # [bsz * num_heads, seq_q_len, seq_k_len]
            attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))

        attn_output_weights = torch.softmax(attn_output_weights, dim=-1)  # [bsz * num_heads, seq_q_len, seq_k_len]

        dropout = self.dropout
        if not self.training:
            dropout = 0.

        if dropout > 0.:
            attn_output_weights = nn.functional.dropout(attn_output_weights, p=dropout)

        attn_output = torch.bmm(attn_output_weights, v)  # [bsz * num_heads, seq_q_len, head_dim]

        if self.num_heads > 1:
            attn_output_weights = torch.stack(attn_output_weights.chunk(self.num_heads, dim=0), dim=1)  # [bsz, num_heads, seq_q_len, seq_k_len]
            attn_output_weights = attn_output_weights.mean(1)  # [bsz, seq_q_len, seq_k_len]

            attn_output = torch.cat(attn_output.chunk(self.num_heads, dim=0), dim=-1)  # [bsz, seq_q_len, embed_dim]

        attn_output = self.out_proj(attn_output)  # [bsz, seq_q_len, embed_dim]

        if attn_mask is not None:  # Set NAN zero
            nan_attn_mask = nan_attn_mask[:query.shape[0]]  # [bsz, seq_q_len]
            attn_output = attn_output * ~nan_attn_mask.unsqueeze(-1)

        attn_output = attn_output.reshape(*batch, *attn_output.shape[1:])
        attn_output_weights = attn_output_weights.reshape(*batch, *attn_output_weights.shape[1:])

        return attn_output, attn_output_weights


class GatedResidualLayer(nn.Module):
    def forward(self, x, y):
        return x + y


class GatedOutputLayer(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()

        self.embed_dim = embed_dim
        self.dense = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.kaiming_uniform_(self.dense.weight.data)

    def forward(self, x, y):
        return x + torch.sigmoid(self.dense(x) * y)


class GatedRecurrentLayer(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()

        self.embed_dim = embed_dim

        self.dense_x_r = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dense_y_r = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dense_x_z = nn.Linear(embed_dim, embed_dim, bias=True)
        self.dense_y_z = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dense_x_g = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dense_y_g = nn.Linear(embed_dim, embed_dim, bias=False)

        nn.init.kaiming_uniform_(self.dense_x_r.weight.data)
        nn.init.kaiming_uniform_(self.dense_y_r.weight.data)
        nn.init.kaiming_uniform_(self.dense_x_z.weight.data)
        nn.init.kaiming_uniform_(self.dense_y_z.weight.data)
        nn.init.kaiming_uniform_(self.dense_x_g.weight.data)
        nn.init.kaiming_uniform_(self.dense_y_g.weight.data)

    def forward(self, x, y):
        _r = torch.sigmoid(self.dense_x_r(x) + self.dense_y_r(y))
        _z = torch.sigmoid(self.dense_x_z(x) + self.dense_y_z(y))
        _h = torch.tanh(self.dense_x_g(_r * x) + self.dense_y_g(y))
        return (1 - _z) * x + _z * _h


class GatedCatLayer(nn.Module):
    def forward(self, x, y):
        return torch.concat([x, y], dim=-1)


class EpisodeMultiheadAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int,

                 pe: Optional[POSITIONAL_ENCODING] = None,
                 qkv_dense_depth: int = 0,
                 out_dense_depth: int = 1,
                 dropout: float = 0.,
                 gate: Optional[GATE] = None,
                 use_layer_norm: bool = False):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.gate = gate
        self.use_layer_norm = use_layer_norm

        self.output_dim = embed_dim

        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(self.embed_dim)

        self.attn = MultiheadAttention(embed_dim=embed_dim,
                                       num_heads=num_heads,
                                       pe=pe,
                                       qkv_dense_depth=qkv_dense_depth,
                                       out_dense_depth=out_dense_depth,
                                       dropout=dropout)

        if gate == GATE.RESIDUAL:
            self.gatedlayer = GatedResidualLayer()
        elif gate == GATE.OUTPUT:
            self.gatedlayer = GatedOutputLayer(embed_dim)
        elif gate == GATE.RECURRENT:
            self.gatedlayer = GatedRecurrentLayer(embed_dim)
        elif gate == GATE.CAT:
            self.gatedlayer = GatedCatLayer()
            self.output_dim = self.output_dim * 2

    def get_attn_mask(self,
                      seq_k_len: int,
                      seq_q_len_only_attend_to_rest_key: Optional[int] = None,
                      key_index: Optional[torch.Tensor] = None,
                      key_padding_mask: Optional[torch.Tensor] = None,
                      device='cpu') -> torch.Tensor:
        """
        Args:
            seq_k_len: int
            seq_q_len_only_attend_to_rest_key: None | int
                Whether the query only need to attend to the key-query and itself (useful in OC)
            key_index (torch.int32): None | [batch, seq_k_len]
                Needed only when seq_q_len_only_attend_to_rest_key is not None
            key_padding_mask: [batch, seq_k_len]
                A `True` value indicates that the corresponding key value will be ignored 
                for the purpose of attention.

        Returns:                     
            [seq_k_len, seq_k_len] OR [batch, seq_k_len, seq_k_len]
        """

        attn_mask = torch.triu(torch.ones(seq_k_len, seq_k_len, dtype=bool, device=device),
                               diagonal=1)
        # [seq_k_len, seq_k_len]
        # Each element only attends to previous element and itself

        if seq_q_len_only_attend_to_rest_key is not None:
            seq_q_len = seq_q_len_only_attend_to_rest_key

            attn_mask = torch.ones(seq_k_len, seq_k_len, dtype=bool, device=device)

            attn_mask[:seq_k_len - seq_q_len, :seq_k_len - seq_q_len] = torch.eye(seq_k_len - seq_q_len, seq_k_len - seq_q_len, dtype=bool, device=device)
            # [seq_k_len, seq_k_len - seq_q_len]

            # Query only attends on itself
            _attn_mask = ~torch.eye(seq_q_len, dtype=bool, device=device)  # [seq_q_len, seq_q_len]
            attn_mask[-seq_q_len:, -seq_q_len:] = torch.logical_or(
                attn_mask[-seq_q_len:, -seq_q_len:],
                _attn_mask
            )  # [seq_k_len, seq_k_len]

            if key_index is not None:
                batch = key_index.shape[0]

                attn_mask = attn_mask.repeat(batch, 1, 1)  # [batch, seq_k_len, seq_k_len]

                _query_index = key_index[:, -seq_q_len:]  # [batch, seq_q_len]
                _rest_key_index = key_index[:, :seq_k_len - seq_q_len]  # [batch, seq_k_len - seq_q_len]

                _query_index = _query_index.unsqueeze(-1).repeat_interleave(seq_k_len - seq_q_len, dim=-1)  # [batch, seq_q_len, seq_k_len - seq_q_len]
                _rest_key_index = _rest_key_index.unsqueeze(1).repeat_interleave(seq_q_len, dim=-2)  # [batch, seq_q_len, seq_k_len - seq_q_len]

                _attn_mask = ~(_query_index >= _rest_key_index)  # [batch, seq_q_len, seq_k_len - seq_q_len]

                attn_mask[:, -seq_q_len:, :seq_k_len - seq_q_len] = _attn_mask  # [batch, seq_k_len, seq_k_len]

                # # Preventing NAN, each element should attend to the first element.
                # attn_mask[:, -seq_q_len:, 0] = False

        if key_padding_mask is not None:
            batch = key_padding_mask.shape[0]

            if len(attn_mask.shape) < 3:
                attn_mask = attn_mask.repeat(batch, 1, 1)  # [batch, seq_k_len, seq_k_len]

            key_padding_mask = key_padding_mask.unsqueeze(1)  # [batch, 1, seq_k_len]
            attn_mask = torch.logical_or(attn_mask, key_padding_mask)  # [batch, seq_k_len, seq_k_len]

            # # Preventing NAN, each element should attend to the first element.
            # attn_mask[:, :, 0] = False

        return attn_mask

    def forward(self,
                key: torch.Tensor,
                seq_q_len: int,
                cut_query: bool = True,
                query_only_attend_to_rest_key: bool = False,
                key_index: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None):
        """
        Args:
            key: [batch, seq_k_len, embed_dim]
            seq_q_len: int
            cut_query: Whether to cut the key into query, or only act on `attn_mask`
            query_only_attend_to_rest_key: bool, Whether the query only need to attend to the key-query and itself
            key_index (torch.int32): None | [batch, key_index_length]
                Needed only when seq_q_len_only_attend_to_rest_key is not None
                `key_index_length` could be shorter than seq_k_len
            key_padding_mask: [batch, key_padding_mask_length]
                A `True` value indicates that the corresponding key value will be ignored 
                for the purpose of attention.
                `key_padding_mask_length` could be shorter than seq_k_len.

        Returns:
            output: [batch, seq_q_len, embed_dim]  if cut_query
                    [batch, seq_k_len, embed_dim]    if not cut_query
            attn_weights: [batch, seq_q_len, seq_k_len]  if cut_query
                          [batch, seq_k_len, seq_k_len]    if not cut_query
        """
        seq_k_len = key.shape[1]

        ori_key = key
        if cut_query:
            ori_query = ori_key[:, -seq_q_len:]
        else:
            ori_query = ori_key

        if self.use_layer_norm:
            key = self.layer_norm(key)

        if key_index is not None:
            key_index_length = key_index.shape[1]
            assert key_index_length <= seq_k_len

            key_index = torch.concat([
                -torch.ones((key_index.shape[0], seq_k_len - key_index_length),
                            dtype=key_index.dtype,
                            device=key_index.device),
                key_index
            ], dim=1)
        query_index = key_index

        if key_padding_mask is not None:
            key_padding_mask_length = key_padding_mask.shape[1]
            assert key_padding_mask_length <= seq_k_len

            key_padding_mask = torch.concat([
                key_padding_mask[:, :1].repeat(1, seq_k_len - key_padding_mask_length),
                key_padding_mask
            ], dim=1)

        attn_mask = self.get_attn_mask(seq_k_len,
                                       seq_q_len_only_attend_to_rest_key=seq_q_len if query_only_attend_to_rest_key else None,
                                       key_index=key_index,
                                       key_padding_mask=key_padding_mask,
                                       device=key.device)

        if cut_query:
            query = key[:, -seq_q_len:]
            if query_index is not None:
                query_index = query_index[:, -seq_q_len:]
            if len(attn_mask.shape) == 2:
                attn_mask = attn_mask[-seq_q_len:]
            else:
                attn_mask = attn_mask[:, -seq_q_len:]
        else:
            query = key

        output, attn_weights = self.attn(query, key, key,
                                         query_index=query_index,
                                         key_index=key_index,
                                         attn_mask=attn_mask)

        if self.gate is not None:
            output = self.gatedlayer(ori_query, output)

        if key_padding_mask is not None:
            output = output * (~key_padding_mask[:, -output.shape[1]:]).to(output.dtype).unsqueeze(-1)

        return output, attn_weights


class EpisodeMultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int,
                 num_layers: int = 2,
                 num_heads: int | List[int] = 1,
                 pe: Optional[POSITIONAL_ENCODING] | List[Optional[POSITIONAL_ENCODING]] = False,
                 qkv_dense_depth: int | List[int] = 0,
                 out_dense_depth: int | List[int] = 1,
                 dropout: float | List[float] = 0.,
                 gate: Optional[GATE] | List[Optional[GATE]] = None,
                 use_layer_norm: bool | List[bool] = False):
        super().__init__()

        self.num_layers = num_layers

        if not isinstance(num_heads, list):
            num_heads = [num_heads] * num_layers
        assert len(num_heads) == num_layers

        if not isinstance(pe, list):
            pe = [pe] * num_layers
        assert len(pe) == num_layers

        if not isinstance(qkv_dense_depth, list):
            qkv_dense_depth = [qkv_dense_depth] * num_layers
        assert len(qkv_dense_depth) == num_layers

        if not isinstance(out_dense_depth, list):
            out_dense_depth = [out_dense_depth] * num_layers
        assert len(out_dense_depth) == num_layers

        if not isinstance(dropout, list):
            dropout = [dropout] * num_layers
        assert len(dropout) == num_layers

        if not isinstance(gate, list):
            gate = [gate] * num_layers
        assert len(gate) == num_layers

        if not isinstance(use_layer_norm, list):
            use_layer_norm = [use_layer_norm] * num_layers
        assert len(use_layer_norm) == num_layers

        self._attn_list = nn.ModuleList()
        _embed_dim = embed_dim
        for i in range(num_layers):
            attn = EpisodeMultiheadAttentionBlock(_embed_dim, num_heads[i],
                                                  pe=pe[i],
                                                  qkv_dense_depth=qkv_dense_depth[i],
                                                  out_dense_depth=out_dense_depth[i],
                                                  dropout=dropout[i],
                                                  gate=gate[i],
                                                  use_layer_norm=use_layer_norm[i])
            self._attn_list.append(attn)
            _embed_dim = attn.output_dim

        self._output_dim_list = [attn.output_dim for attn in self._attn_list]
        self.output_dim = _embed_dim
        self.output_hidden_state_dim = sum(self._output_dim_list[:-1]) if self.num_layers > 1 else 1

    def forward(self,
                key: torch.Tensor,
                seq_q_len: int = 1,
                cut_query: bool = True,
                hidden_state: Optional[torch.Tensor] = None,
                is_prev_hidden_state: bool = False,

                query_only_attend_to_rest_key: bool = False,
                key_index: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None):
        """
        Args:
            key: [batch, seq_k_len, embed_dim]
            seq_q_len: int
            cut_query: bool
            hidden_state: [batch, hidden_state_length, embed_dim]
            is_prev_hidden_state: bool

            query_only_attend_to_rest_key: bool
            key_index: [batch, seq_k_len]
            key_padding_mask: [batch, seq_k_len]

        Returns:
            encoded_query: [batch, seq_q_len, output_dim]  if cut_query
                           [batch, seq_k_len, output_dim]  if not cut_query
            next_hidden_state: [batch, seq_q_len, sum(output_dim_list[:-1])]
            attn_weights_list: List[[batch, seq_k_len_i, seq_k_len_i], ...]
        """
        seq_k_len = key.shape[1]
        assert seq_q_len <= seq_k_len

        next_hidden_state_list = []
        attn_weights_list = []

        if hidden_state is None:
            _k = key
            for i, attn in enumerate(self._attn_list[:-1]):
                output, attn_weight = attn(_k, seq_q_len,
                                           cut_query=False,
                                           query_only_attend_to_rest_key=query_only_attend_to_rest_key,
                                           key_index=key_index,
                                           key_padding_mask=key_padding_mask)
                _k = output
                _q = _k[:, -seq_q_len:]
                next_hidden_state_list.append(_q)
                attn_weights_list.append(attn_weight)

            output, attn_weight = self._attn_list[-1](_k, seq_q_len,
                                                      cut_query=cut_query,
                                                      query_only_attend_to_rest_key=query_only_attend_to_rest_key,
                                                      key_index=key_index,
                                                      key_padding_mask=key_padding_mask)
            attn_weights_list.append(attn_weight)
            _q = output

        elif not is_prev_hidden_state:
            output, attn_weight = self._attn_list[0](key, seq_q_len,
                                                     cut_query=False if self.num_layers > 1 else cut_query,
                                                     query_only_attend_to_rest_key=query_only_attend_to_rest_key,
                                                     key_index=key_index,
                                                     key_padding_mask=key_padding_mask)
            attn_weights_list.append(attn_weight)

            if self.num_layers > 1:
                hidden_state_list = hidden_state.split(self._output_dim_list[:-1], dim=-1)

            for i, attn in enumerate(self._attn_list[1:]):
                next_hidden_state_list.append(output[:, -seq_q_len:])

                _k = torch.concat([hidden_state_list[i], output], dim=1)

                output, attn_weight = attn(_k, seq_q_len,
                                           cut_query=False if i != self.num_layers - 2 else cut_query,
                                           query_only_attend_to_rest_key=query_only_attend_to_rest_key,
                                           key_index=key_index,
                                           key_padding_mask=key_padding_mask)
                attn_weights_list.append(attn_weight)

            _q = output

        elif is_prev_hidden_state:
            output, attn_weight = self._attn_list[0](key, seq_q_len,
                                                     cut_query=False,
                                                     query_only_attend_to_rest_key=query_only_attend_to_rest_key,
                                                     key_index=key_index,
                                                     key_padding_mask=key_padding_mask)
            next_hidden_state_list.append(output[:, -seq_q_len:])
            attn_weights_list.append(attn_weight)

            if self.num_layers > 1:
                hidden_state_list = hidden_state.split(self._output_dim_list[:-1], dim=-1)
            else:
                output = output[:, -seq_q_len:] if cut_query else output

            for i, attn in enumerate(self._attn_list[1:-1]):
                _k = output[:, -seq_k_len:]
                _k = torch.concat([hidden_state_list[i], _k], dim=1)

                output, attn_weight = attn(_k, seq_q_len,
                                           cut_query=False,
                                           query_only_attend_to_rest_key=query_only_attend_to_rest_key,
                                           key_index=key_index,
                                           key_padding_mask=key_padding_mask)
                next_hidden_state_list.append(output[:, -seq_q_len:])
                attn_weights_list.append(attn_weight)

            if self.num_layers > 1:
                _k = output[:, -seq_k_len:]
                _k = torch.concat([hidden_state_list[-1], _k], dim=1)

                output, attn_weight = self._attn_list[-1](_k, seq_q_len,
                                                          cut_query=cut_query,
                                                          query_only_attend_to_rest_key=query_only_attend_to_rest_key,
                                                          key_index=key_index,
                                                          key_padding_mask=key_padding_mask)
                attn_weights_list.append(attn_weight)

            _q = output

        if self.num_layers > 1:
            return _q, torch.concat(next_hidden_state_list, dim=-1), attn_weights_list
        else:
            return _q, torch.zeros(key.shape[0], seq_q_len, 1, device=key.device), attn_weights_list


class AbsolutePositionalEncoding(nn.Module):
    def __init__(self,
                 d_model: int,
                 max_seq_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        self.register_buffer('pe', pe)

    @torch.no_grad()
    def forward(self, indexes):
        return self.pe[indexes.type(torch.int64)]


# https://zhuanlan.zhihu.com/p/647109286
class RotaryPositionalEncoding(nn.Module):
    def __init__(self,
                 d_model: int,
                 max_seq_len: int = 5000,
                 theta: float = 10000.0):
        super().__init__()

        # 计算词向量元素两两分组之后，每组元素对应的旋转角度\theta_i
        freqs = 1.0 / (theta ** (torch.arange(0, d_model, 2)[: (d_model // 2)] / d_model))

        # 生成 token 序列索引 t = [0, 1,..., max_seq_len-1]
        t = torch.arange(max_seq_len)
        # freqs.shape = [max_seq_len, d_model // 2]
        freqs = torch.outer(t, freqs)  # 计算m * \theta

        # 计算结果是个复数向量
        # 假设 freqs = [x, y]
        # 则 freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

        self.register_buffer('freqs_cis', freqs_cis)

    def forward(self,
                xq_indexes: torch.Tensor,
                xk_indexes: torch.Tensor,
                xq: torch.Tensor,
                xk: torch.Tensor):
        xq_freqs_cis = self.freqs_cis[xq_indexes.type(torch.int64)]
        xk_freqs_cis = self.freqs_cis[xk_indexes.type(torch.int64)]

        xq_ = xq.reshape(*xq.shape[:-1], -1, 2)
        xk_ = xk.reshape(*xk.shape[:-1], -1, 2)

        # 转为复数域
        xq_ = torch.view_as_complex(xq_)
        xk_ = torch.view_as_complex(xk_)

        # 应用旋转操作，然后将结果转回实数域
        # xq_out.shape = [batch, seq_len, d_model]
        xq_out = torch.view_as_real(xq_ * xq_freqs_cis).flatten(2)
        xk_out = torch.view_as_real(xk_ * xk_freqs_cis).flatten(2)
        return xq_out.type_as(xq), xk_out.type_as(xk)


class RotaryPositionalEncoding2(nn.Module):
    def __init__(self,
                 d_model: int,
                 max_seq_len: int = 5000,
                 base: int = 10_000):
        """
        * `d` is the number of features $d$
        * `base` is the constant used for calculating $\Theta$
        """
        super().__init__()

        self.d_model = d_model

        # $\Theta = {\theta_i = 10000^{-\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        theta = 1. / (base ** (torch.arange(0, self.d_model, 2).float() / self.d_model))

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(max_seq_len).float()

        # Calculate the product of position index and $\theta_i$
        idx_theta = torch.einsum('n,d->nd', seq_idx, theta)

        # Concatenate so that for row $m$ we have
        # $[m \theta_0, m \theta_1, ..., m \theta_{\frac{d}{2}}, m \theta_0, m \theta_1, ..., m \theta_{\frac{d}{2}}]$
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

        self.register_buffer('cos_cached', idx_theta2.cos())
        self.register_buffer('sin_cached', idx_theta2.sin())

    def _neg_half(self, x: torch.Tensor):
        # $\frac{d}{2}$
        d_2 = self.d_model // 2

        # Calculate $[-x^{(\frac{d}{2} + 1)}, -x^{(\frac{d}{2} + 2)}, ..., -x^{(d)}, x^{(1)}, x^{(2)}, ..., x^{(\frac{d}{2})}]$
        return torch.cat([-x[:, :, d_2:], x[:, :, :d_2]], dim=-1)

    def forward(self,
                xq_indexes: torch.Tensor,
                xk_indexes: torch.Tensor,
                xq: torch.Tensor,
                xk: torch.Tensor):
        """
        * `x` is the Tensor at the head of a key or a query with shape `[seq_len, batch_size, n_heads, d]`
        """

        xq_rope, xq_pass = xq[..., :self.d_model], xq[..., self.d_model:]
        xk_rope, xk_pass = xk[..., :self.d_model], xk[..., self.d_model:]

        neg_half_xq = self._neg_half(xq_rope)
        neg_half_xk = self._neg_half(xk_rope)

        xq_rope = (xq_rope * self.cos_cached[xq_indexes]) + (neg_half_xq * self.sin_cached[xq_indexes])
        xk_rope = (xk_rope * self.cos_cached[xk_indexes]) + (neg_half_xk * self.sin_cached[xk_indexes])

        return torch.cat((xq_rope, xq_pass), dim=-1), torch.cat((xk_rope, xk_pass), dim=-1)


if __name__ == '__main__':
    attn = MultiheadAttention(4, num_heads=2, pe=POSITIONAL_ENCODING.ROPE2)
    x = torch.rand(2, 3, 4)
    y, w = attn(x, x, x, key_padding_mask=torch.tensor([[True, True, True], [False, True, True]]))
    print(y)
    print(w)
    # y.mean().backward()
    # for p in attn.parameters():
    #     print(p.grad)

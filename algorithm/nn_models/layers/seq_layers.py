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


class MultiheadAttention(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 num_head: int,
                 q_dim: Optional[int] = None,
                 k_dim: Optional[int] = None,
                 v_dim: Optional[int] = None,
                 use_rope: bool = False) -> None:

        super().__init__()

        if hidden_dim % 2 != 0:
            hidden_dim = 2 * (hidden_dim // 2)

        self.hidden_dim = hidden_dim
        self.q_dim = q_dim if q_dim is not None else hidden_dim
        self.k_dim = k_dim if k_dim is not None else hidden_dim
        self.v_dim = v_dim if v_dim is not None else hidden_dim
        self.use_rope = use_rope

        self.rope = RotaryPositionalEncoding(hidden_dim)

        self.q_proj = nn.Linear(self.q_dim, hidden_dim)
        self.k_proj = nn.Linear(self.k_dim, hidden_dim)
        self.v_proj = nn.Linear(self.v_dim, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, self.q_dim)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                query_index: Optional[torch.Tensor] = None,
                key_index: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None):
        """
        Args:
            query: [batch, seq_q_len, q_dim]
            key: [batch, seq_k_len, k_dim]
            value: [batch, seq_k_len, v_dim]
            query_index: [batch, seq_q_len]
            key_index: [batch, seq_k_len]
            attn_mask: [batch, seq_q_len, seq_v_len] OR [seq_q_len, seq_v_len]

        Returns:
            attn_output: [batch, seq_q_len, q_dim]
            attn_output_weights: [batch, seq_q_len, seq_k_len]
        """

        batch = query.shape[:-2]
        query = query.reshape(-1, *query.shape[-2:])
        key = key.reshape(-1, *key.shape[-2:])
        value = value.reshape(-1, *value.shape[-2:])

        q = self.q_proj(query)  # [batch, seq_q_len, hidden_dim]
        k = self.k_proj(key)  # [batch, seq_k_len, hidden_dim]
        v = self.v_proj(value)  # [batch, seq_k_len, hidden_dim]

        if self.use_rope:
            q, k = self.rope(query_index, key_index, q, k)

        q_scaled = q / math.sqrt(self.hidden_dim)  # [batch, seq_q_len, hidden_dim]

        if attn_mask is not None:
            # [batch, seq_q_len, seq_v_len] OR [seq_q_len, seq_v_len]
            attn_mask = torch.zeros_like(attn_mask, dtype=query.dtype,
                                         device=query.device).masked_fill_(attn_mask, float("-inf"))

            # [batch, seq_q_len, seq_v_len]
            attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
        else:
            # [batch, seq_q_len, seq_v_len]
            attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))

        attn_output_weights = torch.softmax(attn_output_weights, dim=-1)  # [batch, seq_q_len, seq_v_len]

        attn_output = torch.bmm(attn_output_weights, v)  # [batch, seq_q_len, hidden_dim]

        attn_output = self.out_proj(attn_output)  # [batch, seq_q_len, q_dim]

        attn_output = attn_output.reshape(*batch, *attn_output.shape[1:])
        attn_output_weights = attn_output_weights.reshape(*batch, *attn_output_weights.shape[1:])

        return attn_output, attn_output_weights


class EpisodeMultiheadAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int,

                 use_pe: bool = True,
                 use_residual: bool = True,
                 use_gated: bool = True,
                 use_layer_norm: bool = False):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.use_residual = use_residual
        self.use_gated = use_gated
        self.use_layer_norm = use_layer_norm

        if self.use_gated:
            self.dense_x_r = nn.Linear(embed_dim, embed_dim)
            self.dense_y_r = nn.Linear(embed_dim, embed_dim)

            self.dense_x_z = nn.Linear(embed_dim, embed_dim)
            self.dense_y_z = nn.Linear(embed_dim, embed_dim)

            self.dense_x_g = nn.Linear(embed_dim, embed_dim)
            self.dense_y_g = nn.Linear(embed_dim, embed_dim)

        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(self.embed_dim)

        self.attn = MultiheadAttention(hidden_dim=embed_dim,
                                       num_head=num_heads,
                                       q_dim=embed_dim,
                                       k_dim=embed_dim,
                                       v_dim=embed_dim,
                                       use_rope=use_pe)

    def get_attn_mask(self,
                      key_length: int,
                      query_length_only_attend_to_rest_key: Optional[int] = None,
                      key_index: Optional[torch.Tensor] = None,
                      key_padding_mask: Optional[torch.Tensor] = None,
                      device='cpu') -> torch.Tensor:
        """
        Args:
            key_length: int
            query_length_only_attend_to_rest_key: None | int
                Whether the query only need to attend to the key-query and itself (useful in OC)
            key_index (torch.int32): None | [batch, key_length]
                Needed only when query_length_only_attend_to_rest_key is not None
            key_padding_mask: [batch, key_length]
                A `True` value indicates that the corresponding key value will be ignored 
                for the purpose of attention.

        Returns:                     
            [key_length, key_length] OR [batch * num_heads, key_length, key_length]
        """

        attn_mask = torch.triu(torch.ones(key_length, key_length, dtype=bool, device=device),
                               diagonal=1)
        # [key_length, key_length]
        # Each element only attends to previous element and itself

        if query_length_only_attend_to_rest_key is not None:
            query_length = query_length_only_attend_to_rest_key

            _attn_mask = ~torch.eye(query_length, dtype=bool, device=device)  # [query_length, query_length]
            attn_mask[-query_length:, -query_length:] = torch.logical_or(
                attn_mask[-query_length:, -query_length:],
                _attn_mask
            )  # [key_length, key_length]

            if key_index is not None:
                batch = key_index.shape[0]

                attn_mask = attn_mask.repeat(batch, 1, 1)  # [batch, key_length, key_length]

                _query_index = key_index[:, -query_length:]  # [batch, query_length]
                _rest_key_index = key_index[:, :key_length - query_length]  # [batch, key_length - query_length]

                _query_index = _query_index.unsqueeze(-1).repeat_interleave(key_length - query_length, dim=-1)  # [batch, query_length, key_length - query_length]
                _rest_key_index = _rest_key_index.unsqueeze(1).repeat_interleave(query_length, dim=-2)  # [batch, query_length, key_length - query_length]

                _attn_mask = ~(_query_index > _rest_key_index)  # [batch, query_length, key_length - query_length]

                attn_mask[:, -query_length:, :key_length - query_length] = torch.logical_or(
                    attn_mask[:, -query_length:, :key_length - query_length],
                    _attn_mask
                )  # [batch, key_length, key_length]

        if key_padding_mask is not None:
            batch = key_padding_mask.shape[0]

            if len(attn_mask.shape) < 3:
                attn_mask = attn_mask.repeat(batch, 1, 1)  # [batch, key_length, key_length]

            key_padding_mask = key_padding_mask.unsqueeze(1)  # [batch, 1, key_length]
            attn_mask = torch.logical_or(attn_mask, key_padding_mask)  # [batch, key_length, key_length]

            # Preventing NAN, each element should attend to it self.
            eye = ~torch.eye(key_length, dtype=bool, device=device)  # [key_length, key_length]
            eye = eye.repeat(batch, 1, 1)  # [batch, key_length, key_length]
            attn_mask = torch.logical_and(attn_mask, eye)

        if len(attn_mask.shape) == 3:
            attn_mask = attn_mask.repeat_interleave(self.num_heads, dim=0)  # [batch * num_heads, key_length, key_length]

        return attn_mask

    def forward(self,
                key: torch.Tensor,
                query_length: int,
                cut_query: bool = True,
                query_only_attend_to_rest_key: bool = False,
                key_index: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None):
        """
        Args:
            key: [batch, key_length, embed_dim]
            query_length: int
            cut_query: Whether to cut the key into query, or only act on `attn_mask`
            query_only_attend_to_rest_key: bool, Whether the query only need to attend to the key-query and itself
            key_index (torch.int32): None | [batch, key_index_length]
                Needed only when query_length_only_attend_to_rest_key is not None
                `key_index_length` could be shorter than key_length
            key_padding_mask: [batch, key_padding_mask_length]
                A `True` value indicates that the corresponding key value will be ignored 
                for the purpose of attention.
                `key_padding_mask_length` could be shorter than key_length.

        Returns:
            output: [batch, query_length, embed_dim]  if cut_query
                    [batch, key_length, embed_dim]    if not cut_query
            attn_weights: [batch, query_length, key_length]  if cut_query
                          [batch, key_length, key_length]    if not cut_query
        """
        key_length = key.shape[1]

        ori_key = key
        if cut_query:
            ori_query = ori_key[:, -query_length:]
        else:
            ori_query = ori_key

        if self.use_layer_norm:
            key = self.layer_norm(key)

        if key_index is not None:
            key_index_length = key_index.shape[1]
            assert key_index_length <= key_length

            key_index = torch.concat([
                -torch.ones((key_index.shape[0], key_length - key_index_length),
                            dtype=key_index.dtype,
                            device=key_index.device),
                key_index
            ], dim=1)
        query_index = key_index

        if key_padding_mask is not None:
            key_padding_mask_length = key_padding_mask.shape[1]
            assert key_padding_mask_length <= key_length

            key_padding_mask = torch.concat([
                key_padding_mask[:, :1].repeat(1, key_length - key_padding_mask_length),
                key_padding_mask
            ], dim=1)

        attn_mask = self.get_attn_mask(key_length,
                                       query_length_only_attend_to_rest_key=query_length if query_only_attend_to_rest_key else None,
                                       key_index=key_index,
                                       key_padding_mask=key_padding_mask,
                                       device=key.device)

        if cut_query:
            query = key[:, -query_length:]
            if query_index is not None:
                query_index = query_index[:, -query_length:]
            if len(attn_mask.shape) == 2:
                attn_mask = attn_mask[-query_length:]
            else:
                attn_mask = attn_mask[:, -query_length:]
        else:
            query = key

        output, attn_weights = self.attn(query, key, key,
                                         query_index=query_index,
                                         key_index=key_index,
                                         attn_mask=attn_mask)

        if key_padding_mask is not None:
            output = output * (~key_padding_mask[:, -output.shape[1]:]).to(output.dtype).unsqueeze(-1)

        output = torch.relu(output)

        if self.use_residual:
            output = output + ori_query

        if self.use_gated:
            _r = torch.relu(self.dense_x_r(ori_query) + self.dense_y_r(output))
            _z = torch.relu(self.dense_x_z(ori_query) + self.dense_y_z(output))
            _h = torch.tanh(self.dense_x_g(_r * ori_query) + self.dense_y_g(output))
            output = (1 - _z) * ori_query + _z * _h

        return output, attn_weights


class EpisodeMultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int,
                 num_layers: int = 2,
                 use_residual: bool = True,
                 use_gated: bool = True,
                 use_layer_norm: bool = False):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self._attn_list = nn.ModuleList(
            [EpisodeMultiheadAttentionBlock(embed_dim, num_heads,
                                            use_pe=True,
                                            use_residual=use_residual,
                                            use_gated=use_gated,
                                            use_layer_norm=use_layer_norm) for i in range(num_layers)]
        )

    def forward(self,
                key: torch.Tensor,
                query_length: int = 1,
                hidden_state: Optional[torch.Tensor] = None,
                is_prev_hidden_state: bool = False,

                query_only_attend_to_rest_key: bool = False,
                key_index: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None):
        """
        Args:
            key: [batch, key_length, embed_dim]
            query_length: int
            hidden_state: [batch, hidden_state_length, embed_dim]
            is_prev_hidden_state: bool

            query_only_attend_to_rest_key: bool
            key_index: [batch, key_length]
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
            for i, attn in enumerate(self._attn_list[:-1]):
                output, attn_weight = attn(_k, query_length,
                                           cut_query=False,
                                           query_only_attend_to_rest_key=query_only_attend_to_rest_key,
                                           key_index=key_index,
                                           key_padding_mask=key_padding_mask)
                _k = output
                _q = _k[:, -query_length:]
                next_hidden_state_list.append(_q)
                attn_weights_list.append(attn_weight[:, -query_length:])

            output, attn_weight = self._attn_list[-1](_k, query_length,
                                                      cut_query=True,
                                                      query_only_attend_to_rest_key=query_only_attend_to_rest_key,
                                                      key_index=key_index,
                                                      key_padding_mask=key_padding_mask)
            attn_weights_list.append(attn_weight)
            _q = output

        elif not is_prev_hidden_state:
            output, attn_weight = self._attn_list[0](key, query_length,
                                                     query_only_attend_to_rest_key=query_only_attend_to_rest_key,
                                                     key_index=key_index,
                                                     key_padding_mask=key_padding_mask)
            attn_weights_list.append(attn_weight)

            if self.num_layers > 1:
                hidden_state_list = hidden_state.chunk(self.num_layers - 1, dim=-1)

            for i, attn in enumerate(self._attn_list[1:]):
                next_hidden_state_list.append(output)

                _k = torch.concat([hidden_state_list[i], output], dim=1)

                output, attn_weight = attn(_k, query_length,
                                           query_only_attend_to_rest_key=query_only_attend_to_rest_key,
                                           key_index=key_index,
                                           key_padding_mask=key_padding_mask)
                attn_weights_list.append(attn_weight)

            _q = output

        elif is_prev_hidden_state:
            output, attn_weight = self._attn_list[0](key, query_length,
                                                     cut_query=False,
                                                     query_only_attend_to_rest_key=query_only_attend_to_rest_key,
                                                     key_index=key_index,
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

                output, attn_weight = attn(_k, query_length,
                                           cut_query=False,
                                           query_only_attend_to_rest_key=query_only_attend_to_rest_key,
                                           key_index=key_index,
                                           key_padding_mask=key_padding_mask)
                next_hidden_state_list.append(output[:, -query_length:])
                attn_weights_list.append(attn_weight[:, -query_length:])

            if self.num_layers > 1:
                _k = output[:, -key_length:]
                _k = torch.concat([hidden_state_list[-1], _k], dim=1)

                output, attn_weight = self._attn_list[-1](_k, query_length,
                                                          cut_query=True,
                                                          query_only_attend_to_rest_key=query_only_attend_to_rest_key,
                                                          key_index=key_index,
                                                          key_padding_mask=key_padding_mask)
                attn_weights_list.append(attn_weight)

            _q = output

        if self.num_layers > 1:
            return _q, torch.concat(next_hidden_state_list, dim=-1), attn_weights_list
        else:
            return _q, torch.zeros(key.shape[0], query_length, 1), attn_weights_list


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

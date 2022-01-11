import torch
from torch import nn
import math
import algorithm.nn_models as m

EXTRA_SIZE = 2


class ModelRep(m.ModelBaseAttentionRep):
    def _build_model(self):
        assert self.obs_shapes[0] == (6, )

        self.attn = m.EpisodeMultiheadAttention(6 - EXTRA_SIZE + 2, 2, 3)
        self.pos = PositionalEncoding(6 - EXTRA_SIZE + 2)

    def forward(self, index, obs_list, pre_action,
                query_length=1,
                hidden_state=None,
                is_prev_hidden_state=False,
                padding_mask=None):
        obs = obs_list[0][..., :-EXTRA_SIZE]

        # i = self.pos(torch.concat([obs, pre_action], dim=-1))
        i = torch.concat([obs, pre_action], dim=-1)
        output, hn, attn_weights_list = self.attn(i,
                                                  query_length,
                                                  hidden_state,
                                                  is_prev_hidden_state,
                                                  padding_mask)

        # print(index[0])
        # for w in attn_weights_list:
        #     print(w[0])
        return output, hn


class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(c_dense_n=64, c_dense_depth=2)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        return super()._build_model(c_dense_n=64, c_dense_depth=2)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

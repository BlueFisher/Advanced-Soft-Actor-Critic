import torch
from torch import nn

import algorithm.nn_models as m
from algorithm.nn_models.layers.seq_layers import GATE, POSITIONAL_ENCODING


class ModelRep(m.ModelBaseAttentionRep):
    def _build_model(self):
        assert self.obs_shapes[0] == (3 + 1, )
        assert self.obs_shapes[1] == (3, 4)

        embed_size = 3 + 1 + 3 * 4  # 16

        self.rnn = m.GRU(embed_size, embed_size, num_layers=1)

        self.attn = m.EpisodeMultiheadAttention(embed_size, num_layers=1,
                                                num_heads=8,
                                                pe=POSITIONAL_ENCODING.ROPE,
                                                qkv_dense_depth=1,
                                                out_dense_depth=1,
                                                dropout=0.,
                                                gate=GATE.RESIDUAL,
                                                use_layer_norm=False)

        self.rnn1 = m.GRU(embed_size, embed_size, num_layers=1)

        self.attn1 = m.EpisodeMultiheadAttention(embed_size, num_layers=1,
                                                 num_heads=8,
                                                 pe=POSITIONAL_ENCODING.ROPE,
                                                 qkv_dense_depth=1,
                                                 out_dense_depth=1,
                                                 dropout=0.,
                                                 gate=GATE.RESIDUAL,
                                                 use_layer_norm=False)

    def forward(self, index, obs_list, pre_action=None,
                seq_q_len=1,
                hidden_state=None,
                is_prev_hidden_state=False,
                query_only_attend_to_rest_key=False,
                padding_mask=None):
        if self._offline_action_index != -1:
            option_index, vec_obs, _ = obs_list
        else:
            option_index, vec_obs = obs_list

        batch = index.shape[0]

        vec_obs = vec_obs.reshape(*vec_obs.shape[:-2], 3 * 4)
        vec_obs = torch.concat([option_index, vec_obs], dim=-1)

        if hidden_state is not None:
            assert hidden_state.shape[1] == 1
            rnn_hidden_state = hidden_state[..., :16]
            attn_hidden_state = hidden_state[..., 16:16 + 1]
            rnn_1_hidden_state = hidden_state[..., 16 + 1:16 + 1 + 16]
            attn1_hidden_state = hidden_state[..., 16 + 1 + 16:]

            rnn_hidden_state = rnn_hidden_state.reshape(batch,
                                                        self.rnn.num_layers,
                                                        self.rnn.hidden_size)
            rnn1_hidden_state = rnn_1_hidden_state.reshape(batch,
                                                           self.rnn1.num_layers,
                                                           self.rnn1.hidden_size)
        else:
            rnn_hidden_state = attn_hidden_state = rnn1_hidden_state = attn1_hidden_state = None

        key_vec_obs = vec_obs[:, :-seq_q_len]
        query_vec_obs = vec_obs[:, -seq_q_len:]

        rnn_outputs = []
        rnn_hidden_states = []

        for i in range(key_vec_obs.shape[1]):
            rnn_output, rnn_hidden_state = self.rnn(key_vec_obs[:, i:i + 1], rnn_hidden_state)
            rnn_outputs.append(rnn_output)

        for i in range(query_vec_obs.shape[1]):
            rnn_output, _rnn_hidden_state = self.rnn(query_vec_obs[:, i:i + 1], rnn_hidden_state)
            rnn_outputs.append(rnn_output)
            rnn_hidden_states.append(_rnn_hidden_state)

        rnn_output = torch.concat(rnn_outputs, dim=1)
        rnn_output = rnn_output + vec_obs

        output, hn, attn_weights_list = self.attn(rnn_output,
                                                  seq_q_len=seq_q_len,
                                                  cut_query=False,
                                                  hidden_state=attn_hidden_state,
                                                  is_prev_hidden_state=is_prev_hidden_state,

                                                  query_only_attend_to_rest_key=query_only_attend_to_rest_key,
                                                  key_index=index,
                                                  key_padding_mask=padding_mask)

        key_output = output[:, :-seq_q_len]
        query_output = output[:, -seq_q_len:]
        rnn1_outputs = []
        rnn1_hidden_states = []
        for i in range(key_output.shape[1]):
            rnn1_output, rnn1_hidden_state = self.rnn1(key_output[:, i:i + 1], rnn1_hidden_state)
            rnn1_outputs.append(rnn1_output)

        for i in range(query_output.shape[1]):
            rnn1_output, _rnn1_hidden_state = self.rnn1(query_output[:, i:i + 1], rnn1_hidden_state)
            rnn1_outputs.append(rnn1_output)
            rnn1_hidden_states.append(_rnn1_hidden_state)

        rnn1_output = torch.concat(rnn1_outputs, dim=1)
        rnn1_output = rnn1_output + output

        output, hn1, attn1_weights_list = self.attn1(rnn1_output,
                                                     seq_q_len=seq_q_len,
                                                     cut_query=True,
                                                     hidden_state=attn1_hidden_state,
                                                     is_prev_hidden_state=is_prev_hidden_state,

                                                     query_only_attend_to_rest_key=query_only_attend_to_rest_key,
                                                     key_index=index,
                                                     key_padding_mask=padding_mask)

        new_rnn_hidden_state = torch.concat(rnn_hidden_states, dim=1)
        new_rnn_hidden_state = new_rnn_hidden_state.reshape(batch, seq_q_len, self.rnn.hidden_size * self.rnn.num_layers)

        new_rnn1_hidden_state = torch.concat(rnn1_hidden_states, dim=1)
        new_rnn1_hidden_state = new_rnn1_hidden_state.reshape(batch, seq_q_len, self.rnn1.hidden_size * self.rnn1.num_layers)

        return output, torch.concat([new_rnn_hidden_state, hn, new_rnn1_hidden_state, hn1], dim=-1), attn_weights_list + attn1_weights_list


class ModelOptionRep(m.ModelBaseSimpleRep):
    def _build_model(self):
        assert self.obs_shapes[1] == (3, 4)

        embed_size = 3 * 4

        # self.mlp = m.LinearLayers(embed_size, dense_n=embed_size, dense_depth=2, dropout=0.)

    def forward(self, obs_list):
        if self._offline_action_index != -1:
            high_state, vec_obs, _ = obs_list
        else:
            high_state, vec_obs = obs_list

        # vec_obs = vec_obs.reshape(*vec_obs.shape[:-2], 3 * 4)

        # output = torch.concat([high_state, self.mlp(vec_obs)], dim=-1)

        return high_state


class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(d_dense_n=128, d_dense_depth=2, dropout=0.)


class ModelTermination(m.ModelTermination):
    def _build_model(self):
        return super()._build_model(dense_n=128, dense_depth=2, dropout=0.)

    def forward(self, state, obs_list):
        high_state, vec_obs = obs_list

        vec_obs = vec_obs.reshape(*vec_obs.shape[:-2], 3 * 4)

        t = vec_obs.any(-1, keepdim=True)
        t = t.to(state.dtype)
        return t


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        return super()._build_model(d_dense_n=32, d_dense_depth=1)


class ModelRND(m.ModelRND):
    def _build_model(self):
        return super()._build_model(dense_n=32, dense_depth=2, output_size=32)

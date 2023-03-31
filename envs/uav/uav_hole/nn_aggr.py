import torch

import algorithm.nn_models as m


class ModelRep(m.ModelBaseRNNRep):
    def _build_model(self):
        assert self.obs_shapes[0] == (3, 2, 9)  # AgentsBufferSensor
        assert self.obs_shapes[1] == (3, 84, 84, 3)
        assert self.obs_shapes[2] == (3, 9)

        self.conv = m.ConvLayers(84, 84, 3, 'simple',
                                 out_dense_n=64, out_dense_depth=2)
        self.conv_attn = m.MultiheadAttention(self.conv.output_size, 8)
        self.vec_attn = m.MultiheadAttention(9, 1)

        print(self.conv.output_size)

        self.rnn = m.GRU(3 * (9 + self.conv.output_size) + self.c_action_size, 128, 1)

    def forward(self, obs_list, pre_action, rnn_state=None, padding_mask=None):
        feature_agents, vis_obs, vec_obs = obs_list

        aggr_mask = ~vec_obs.any(dim=-1)
        aggr_mask[..., 0] = False

        vis_obs = self.conv(vis_obs)
        vis_obs, _ = self.conv_attn(vis_obs, vis_obs, vis_obs, key_padding_mask=aggr_mask)
        vec_obs, _ = self.vec_attn(vec_obs, vec_obs, vec_obs, key_padding_mask=aggr_mask)

        features = torch.concat([vec_obs, vis_obs], dim=-1)

        if padding_mask is not None:
            features[padding_mask] = 0.

        features = features.reshape(*features.shape[:-2], -1)

        state, hn = self.rnn(torch.concat([features, pre_action], dim=-1), rnn_state)

        return state, hn


class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(c_dense_n=128, c_dense_depth=3)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        return super()._build_model(c_dense_n=128, c_dense_depth=3)


class ModelRND(m.ModelRND):
    def _build_model(self):
        return super()._build_model(dense_n=128, dense_depth=3, output_size=128)

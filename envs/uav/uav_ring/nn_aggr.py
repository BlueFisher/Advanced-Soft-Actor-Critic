import torch

import algorithm.nn_models as m

N_OTHER_AGENTS = 1


class ModelRep(m.ModelSimpleRep):
    def _build_model(self):
        assert self.obs_shapes[0][1:] == (N_OTHER_AGENTS, 9)  # AgentsBufferSensor
        assert self.obs_shapes[1][1:] == (84, 84, 3)
        assert self.obs_shapes[2][1:] == (9, )

        self.group_size = self.obs_shapes[0][0]

        self.conv = m.ConvLayers(84, 84, 3, 'simple',
                                 out_dense_n=64, out_dense_depth=2)
        
        # self.conv_attn = m.MultiheadAttention(self.conv.output_size, 8)
        # self.vec_attn = m.MultiheadAttention(9, 1)


    def forward(self, obs_list):
        feature_agents, vis_obs, vec_obs = obs_list

        # aggr_mask = ~vec_obs.any(dim=-1)
        # aggr_mask[..., 0] = False

        vis_obs = self.conv(vis_obs)
        # vis_obs, _ = self.conv_attn(vis_obs, vis_obs, vis_obs, key_padding_mask=aggr_mask)
        # vec_obs, _ = self.vec_attn(vec_obs, vec_obs, vec_obs, key_padding_mask=aggr_mask)

        features = torch.concat([vec_obs, vis_obs], dim=-1)

        # if padding_mask is not None:
        #     features[padding_mask] = 0.

        features = features.reshape(*features.shape[:-2], -1)

        return features


class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(d_dense_n=128, d_dense_depth=2,
                                    c_dense_n=128, c_dense_depth=2)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        return super()._build_model(d_dense_n=128, d_dense_depth=2,
                                    c_dense_n=128, c_dense_depth=2)


class ModelRND(m.ModelRND):
    def _build_model(self):
        return super()._build_model(dense_n=128, dense_depth=3, output_size=128)

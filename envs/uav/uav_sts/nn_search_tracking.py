import torch

import algorithm.nn_models as m

OBS_NAMES = ['_Padding',
             'AgentsBufferSensor', 'BoundingBoxSensor', 'CameraSensor',
             'RayPerceptionSensor1', 'RayPerceptionSensor2', 'RayPerceptionSensor3',
             'VectorSensor_size9']
OBS_SHAPES = [(1,), (8, 10), (8, 6), (84, 84, 1), (22,), (22,), (22,), (10,)]


class ModelRep(m.ModelBaseRNNRep):
    def _build_model(self):
        for u_s, s in zip(self.obs_shapes, OBS_SHAPES):
            assert u_s == s, f'{u_s} {s}'

        self.attn_uavs = m.MultiheadAttention(10, 1)

        self.attn_bbox = m.MultiheadAttention(6, 1)

        self.conv = m.ConvLayers(84, 84, 1, 'simple',
                                 out_dense_n=64, out_dense_depth=2)

        self.ray_dense = m.LinearLayers(input_size=22 * 3,
                                        dense_n=64,
                                        dense_depth=1)

        self.rnn = m.GRU(10 + 6 + 64 + 64 + 10 + self.c_action_size, 128, 1)

    def forward(self, obs_list, pre_action, rnn_state=None, padding_mask=None):
        _, feat_uavs, feat_bbox, vis_obs, ray_obs_1, ray_obs_2, ray_obs_3, vec_obs = obs_list

        feat_uavs_mask = ~feat_uavs.any(dim=-1)
        attned_uavs, _ = self.attn_uavs(vec_obs.unsqueeze(-2), feat_uavs, feat_uavs,
                                        key_padding_mask=feat_uavs_mask)
        attned_uavs = attned_uavs.squeeze(-2)

        feat_bbox_mask = ~feat_bbox.any(dim=-1)
        attned_bbox, _ = self.attn_bbox(feat_bbox, feat_bbox, feat_bbox,
                                        key_padding_mask=feat_bbox_mask)
        attned_bbox = attned_bbox.sum(-2)  # [batch, seq_len, f]
        count = (~feat_bbox_mask).sum(-1, keepdim=True)  # [batch, seq_len, f]
        count = torch.maximum(torch.ones_like(count), count)
        attned_bbox = attned_bbox / count  # [batch, seq_len, f]

        # if padding_mask is not None:
        #     attned_uavs[padding_mask] = 0.
        #     attned_bbox[padding_mask] = 0.

        vis_obs = self.conv(vis_obs)

        ray = self.ray_dense(torch.concat([ray_obs_1, ray_obs_2, ray_obs_3], dim=-1))

        state, hn = self.rnn(torch.concat([attned_uavs,
                                           attned_bbox,
                                           vis_obs,
                                           ray,
                                           vec_obs,
                                           pre_action], dim=-1), rnn_state)

        if padding_mask is not None:
            state = state * (~padding_mask).to(state.dtype).unsqueeze(-1)

        return state, hn


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
        return super()._build_model(dense_n=128, dense_depth=2, output_size=128)

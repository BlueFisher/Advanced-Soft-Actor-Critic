import torch
from torch import nn

import algorithm.nn_models as m

RAY_SIZE = 61


class ModelRep(m.ModelBaseRep):
    def _build_model(self):
        assert self.obs_shapes[0] == (15, 10)  # AgentsBufferSensor
        assert self.obs_shapes[1] == (3, 6)  # EnemiesBufferSensor
        assert self.obs_shapes[2] == (122, )  # RayPerceptionSensor
        assert self.obs_shapes[3] == (10, )  # VectorSensor_size8

        self.attn_usvs = m.MultiheadAttention(10, 1)
        self.attn_enemies = m.MultiheadAttention(6, 1)

        self.ray_conv = m.Conv1dLayers(RAY_SIZE, 2, 'default',
                                       out_dense_n=64, out_dense_depth=2)

        self.dense = m.LinearLayers(10 + 6 + self.ray_conv.output_size, output_size=64)

    def forward(self, obs_list, pre_action, rnn_state=None, padding_mask=None):
        feat_usvs, feat_enemeis, ray, vec_obs = obs_list

        ray = ray.view(*ray.shape[:-1], RAY_SIZE, 2)

        feat_usv_class_token = torch.zeros_like(feat_usvs[..., :1, :])
        feat_usvs = torch.cat([feat_usv_class_token, feat_usvs], dim=-2)
        feat_uavs_mask = ~feat_usvs.any(dim=-1)
        feat_uavs_mask[..., 0] = False
        attned_usvs, _ = self.attn_usvs(feat_usvs, feat_usvs, feat_usvs,
                                        key_padding_mask=feat_uavs_mask)
        attned_usvs = attned_usvs[..., 0, :]

        feat_enemy_class_token = torch.zeros_like(feat_enemeis[..., :1, :])
        feat_enemeis = torch.cat([feat_enemy_class_token, feat_enemeis], dim=-2)
        feat_enemeis_mask = ~feat_enemeis.any(dim=-1)
        feat_enemeis_mask[..., 0] = False
        attned_enemies, _ = self.attn_enemies(feat_enemeis, feat_enemeis, feat_enemeis,
                                              key_padding_mask=feat_enemeis_mask)
        attned_enemies = attned_enemies[..., 0, :]

        ray_encoder = self.ray_conv(ray)

        if rnn_state is not None:
            rnn_state = rnn_state[:, 0]

        x = self.dense(torch.concat([attned_usvs,
                                     attned_enemies,
                                     ray_encoder], dim=-1))
        state = torch.cat([x, vec_obs], dim=-1)

        return state, self._get_empty_seq_hidden_state(state)


class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(c_dense_n=64, c_dense_depth=2)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        return super()._build_model(c_dense_n=64, c_dense_depth=2)

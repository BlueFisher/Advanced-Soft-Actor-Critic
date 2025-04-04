import torch
from torch import nn

import algorithm.nn_models as m


class ModelRep(m.ModelBaseRep):
    def _build_model(self):
        assert self.obs_shapes[0] == (10, 9)  # AgentsBufferSensor
        assert self.obs_shapes[1] == (3, 84, 84)  # CameraSensor
        assert self.obs_shapes[2] == (3, 6)  # EnemiesBufferSensor
        assert self.obs_shapes[3] == (122, )  # RayPerceptionSensor
        assert self.obs_shapes[4] == (8, )  # VectorSensor_size6

        self.attn_usvs = m.MultiheadAttention(9, 1)
        self.feat_usv_class_token = nn.Parameter(torch.zeros(1, 1, 9))
        self.attn_enemies = m.MultiheadAttention(6, 1)
        self.feat_enemy_class_token = nn.Parameter(torch.zeros(1, 1, 6))

        self.conv = m.ConvLayers(84, 84, 3, 'simple',
                                 out_dense_n=64, out_dense_depth=2)

        self.ray_dense = m.LinearLayers(122, 8, 1)

        self.rnn = m.GRU(9 + 6 + self.conv.output_size + 8 + self.c_action_size, 64, 1)

    def forward(self, obs_list, pre_action, rnn_state=None, padding_mask=None):
        feat_usvs, vis_obs, feat_enemeis, ray_obs, vec_obs = obs_list

        feat_usv_class_token = self.feat_usv_class_token.expand(*feat_usvs.shape[:-2], -1, -1)
        feat_usvs = torch.cat([feat_usv_class_token, feat_usvs], dim=-2)
        feat_uavs_mask = ~feat_usvs.any(dim=-1)
        feat_uavs_mask[..., 0] = False
        attned_usvs, _ = self.attn_usvs(feat_usvs, feat_usvs, feat_usvs,
                                        key_padding_mask=feat_uavs_mask)
        attned_usvs = attned_usvs[..., 0, :]

        vis_encoder = self.conv(vis_obs)

        feat_enemy_class_token = self.feat_enemy_class_token.expand(*feat_enemeis.shape[:-2], -1, -1)
        feat_enemeis = torch.cat([feat_enemy_class_token, feat_enemeis], dim=-2)
        feat_enemeis_mask = ~feat_enemeis.any(dim=-1)
        feat_enemeis_mask[..., 0] = False
        attned_enemies, _ = self.attn_enemies(feat_enemeis, feat_enemeis, feat_enemeis,
                                              key_padding_mask=feat_enemeis_mask)
        attned_enemies = attned_enemies[..., 0, :]

        ray = self.ray_dense(ray_obs)

        if rnn_state is not None:
            rnn_state = rnn_state[:, 0]

        state, hn = self.rnn(torch.concat([attned_usvs,
                                           attned_enemies,
                                           vis_encoder,
                                           ray,
                                           pre_action], dim=-1), rnn_state)
        state = torch.cat([state, vec_obs], dim=-1)

        return state, hn


class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(c_dense_n=64, c_dense_depth=2)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        return super()._build_model(c_dense_n=64, c_dense_depth=2)

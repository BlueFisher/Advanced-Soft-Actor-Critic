import torch
import algorithm.nn_models as m


class ModelRep(m.ModelBaseRep):
    def _build_model(self):
        assert self.obs_shapes[0] == (10, 9)  # AgentsBufferSensor
        assert self.obs_shapes[1] == (3, 6)  # EnemiesBufferSensor
        assert self.obs_shapes[2] == (122, )  # RayPerceptionSensor
        assert self.obs_shapes[3] == (5, 2)  # SearchTargetsBufferSensor
        assert self.obs_shapes[4] == (6, )  # VectorSensor_size6

        self.attn_usvs = m.MultiheadAttention(9, 1)
        self.attn_enemies = m.MultiheadAttention(6, 1)
        self.attn_targets = m.MultiheadAttention(2, 1)

        self.ray_dense = m.LinearLayers(122, 8, 1)

        self.rnn = m.GRU(9 + 6 + 2 + 8 + self.c_action_size, 64, 1)

    def forward(self, obs_list, pre_action, rnn_state=None, padding_mask=None):
        feat_usvs, feat_enemeis, ray_obs, feat_targets, vec_obs = obs_list

        feat_uavs_mask = ~feat_usvs.any(dim=-1)
        feat_uavs_mask[..., 0] = False
        attned_usvs, _ = self.attn_usvs(feat_usvs, feat_usvs, feat_usvs,
                                        key_padding_mask=feat_uavs_mask)
        attned_usvs = attned_usvs.mean(-2)

        feat_enemeis_mask = ~feat_enemeis.any(dim=-1)
        feat_enemeis_mask[..., 0] = False
        attned_enemies, _ = self.attn_enemies(feat_enemeis, feat_enemeis, feat_enemeis,
                                              key_padding_mask=feat_enemeis_mask)
        attned_enemies = attned_enemies.mean(-2)

        feat_targets_mask = ~feat_targets.any(dim=-1)
        feat_targets_mask[..., 0] = False
        attned_targets, _ = self.attn_targets(feat_targets, feat_targets, feat_targets,
                                              key_padding_mask=feat_targets_mask)
        attned_targets = attned_targets.mean(-2)

        ray = self.ray_dense(ray_obs)

        if rnn_state is not None:
            rnn_state = rnn_state[:, 0]

        state, hn = self.rnn(torch.concat([attned_usvs,
                                           attned_enemies,
                                           attned_targets,
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

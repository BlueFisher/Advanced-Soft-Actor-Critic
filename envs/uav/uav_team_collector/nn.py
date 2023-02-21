import torch

import algorithm.nn_models as m


class ModelRep(m.ModelBaseRNNRep):
    def _build_model(self):
        assert self.obs_shapes[0] == (2, 10)  # AgentsBufferSensor
        assert self.obs_shapes[1] == (3, 4)  # TargetsBufferSensor
        assert self.obs_shapes[2] == (10, )

        self.agents_attn = m.MultiheadAttention(10, 1)
        self.targets_attn = m.MultiheadAttention(4, 1)
        self.agents_attn_dense = m.LinearLayers(10, output_size=16)
        self.targets_attn_dense = m.LinearLayers(4, output_size=16)
 
        self.rnn = m.GRU(16 + 16 + self.obs_shapes[2][0] + self.c_action_size, 64, 1)

    def _handle_bbox(self, bbox, attn):
        bbox_mask = ~bbox.any(dim=-1)
        bbox_mask[..., 0] = False
        bbox, _ = attn(bbox, bbox, bbox, key_padding_mask=bbox_mask)
        return bbox.mean(-2)

    def forward(self, obs_list, pre_action, rnn_state=None):
        feature_agents, feature_targets, vec_obs = obs_list

        feature_agents = self._handle_bbox(feature_agents, self.agents_attn)
        feature_targets = self._handle_bbox(feature_targets, self.targets_attn)

        feature_agents = self.agents_attn_dense(feature_agents)
        feature_targets = self.targets_attn_dense(feature_targets)

        state, hn = self.rnn(torch.concat([feature_agents,
                                           feature_targets,
                                           vec_obs,
                                           pre_action], dim=-1), rnn_state)

        return state, hn


class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(c_dense_n=128, c_dense_depth=2)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        return super()._build_model(c_dense_n=128, c_dense_depth=2)


class ModelRND(m.ModelRND):
    def _build_model(self):
        return super()._build_model(dense_n=128, dense_depth=2, output_size=128)

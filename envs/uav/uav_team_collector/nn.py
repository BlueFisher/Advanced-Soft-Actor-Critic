import torch

import algorithm.nn_models as m


class ModelRep(m.ModelBaseRNNRep):
    def _build_model(self):
        assert self.obs_shapes[0] == (2, 9)  # AgentsBufferSensor
        assert self.obs_shapes[1] == (3, 4)  # TargetsBufferSensor
        assert self.obs_shapes[2] == (9, )

        self.bbox_agents_attn = m.MultiheadAttention(9, 1)
        self.bbox_targets_attn = m.MultiheadAttention(4, 1)

        self.bbox_agents_dense = m.LinearLayers(9, 8)
        self.bbox_targets_dense = m.LinearLayers(4, 8)

        self.rnn = m.GRU(self.bbox_agents_dense.output_size + self.bbox_targets_dense.output_size + 9 + self.c_action_size, 64, 1)

    def _handle_bbox(self, bbox, attn):
        bbox_mask = ~bbox.any(dim=-1)
        bbox_mask[..., 0] = False
        bbox, _ = attn(bbox, bbox, bbox, key_padding_mask=bbox_mask)
        return bbox.mean(-2)

    def forward(self, obs_list, pre_action, rnn_state=None):
        bbox_agents, bbox_targets, vec_obs = obs_list

        bbox_agents = self._handle_bbox(bbox_agents, self.bbox_agents_attn)
        bbox_targets = self._handle_bbox(bbox_targets, self.bbox_targets_attn)
        bbox_agents = self.bbox_agents_dense(bbox_agents)
        bbox_targets = self.bbox_targets_dense(bbox_targets)

        state, hn = self.rnn(torch.concat([bbox_agents, bbox_targets, vec_obs, pre_action], dim=-1), rnn_state)

        return state, hn


class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(c_dense_n=64, c_dense_depth=2)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        return super()._build_model(c_dense_n=64, c_dense_depth=2)

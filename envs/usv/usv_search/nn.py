import torch

import algorithm.nn_models as m


class ModelRep(m.ModelBaseRep):
    def _build_model(self, need_speed):
        assert self.obs_shapes[0] == (6, 6)  # BoundingBoxSensor
        assert self.obs_shapes[1] == (3, 84, 84)  # CameraSensor
        assert self.obs_shapes[2] == (3, 84, 84)  # SegmentationSensor
        assert self.obs_shapes[3] == (7,)

        self.need_speed = need_speed

        self.bbox_attn = m.MultiheadAttention(6, 1)

        self.conv = m.ConvLayers(84, 84, 3 * 2, 'simple',
                                 out_dense_n=64, out_dense_depth=1,
                                 output_size=16)

        self.dense = m.LinearLayers(6 + self.conv.output_size + 7,
                                    dense_n=64, dense_depth=1)

        self.rnn = m.GRU(self.dense.output_size + self.c_action_size, 64, 1)

    def _handle_bbox(self, bbox):
        bbox_mask = ~bbox.any(dim=-1)
        bbox_mask[..., 0] = False
        bbox, _ = self.bbox_attn(bbox, bbox, bbox, key_padding_mask=bbox_mask)
        bbox = bbox.mean(-2)

        return bbox

    def forward(self, obs_list, pre_action, rnn_state=None, padding_mask=None):
        bbox, vis_camera, vis_segmentation, vec = obs_list

        bbox = self._handle_bbox(bbox)

        vis = self.conv(torch.cat([vis_camera, vis_segmentation], dim=-3))

        state = self.dense(torch.cat([bbox, vis, vec], dim=-1))

        if rnn_state is not None:
            rnn_state = rnn_state[:, 0]

        state, hn = self.rnn(torch.cat([state, pre_action], dim=-1), rnn_state)

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

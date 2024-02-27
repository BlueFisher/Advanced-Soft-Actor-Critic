import torch

import algorithm.nn_models as m


GROUP_SIZE = 4
OBS_NAMES = ['_Padding',
             'AgentsBufferSensor', 'BoundingBoxSensor', 'CameraSensor',
             'RayPerceptionSensor1', 'RayPerceptionSensor2', 'RayPerceptionSensor3',
             'VectorSensor_size9']
OBS_SHAPES = [(GROUP_SIZE, 1),
              (GROUP_SIZE, 4, 10), (GROUP_SIZE, 5, 6), (GROUP_SIZE, 84, 84, 1),
              (GROUP_SIZE, 22), (GROUP_SIZE, 22), (GROUP_SIZE, 22),
              (GROUP_SIZE, 10)]


class ModelRep(m.ModelBaseRNNRep):
    def _build_model(self):
        for u_s, s in zip(self.obs_shapes, OBS_SHAPES):
            assert u_s == s

        self.attn_bbox = m.MultiheadAttention(6, 1)

        self.conv = m.ConvLayers(84, 84, 1, 'simple',
                                 out_dense_n=64, out_dense_depth=2)

        self.ray_dense = m.LinearLayers(input_size=22 * 3,
                                        dense_n=64,
                                        dense_depth=1)

        concat_size = 6 + 64 + 64 + 10 + self.c_action_size // GROUP_SIZE
        self.fuse_dense = m.LinearLayers(concat_size, dense_n=128, dense_depth=1)

        self.attn_feat = m.MultiheadAttention(128, 8)

        self.rnn = m.GRU(128, 128, 1)

    def forward(self, obs_list, pre_action, rnn_state=None, padding_mask=None):
        group_padding, _, feat_bbox, vis_obs, ray_obs_1, ray_obs_2, ray_obs_3, vec_obs = obs_list
        group_padding = group_padding == 1.
        group_padding = torch.logical_or(group_padding, ~vec_obs.any(-1, keepdim=True))

        batch, seq_len = vec_obs.shape[:2]

        feat_bbox_mask = ~feat_bbox.any(dim=-1)
        attned_bbox, _ = self.attn_bbox(feat_bbox, feat_bbox, feat_bbox,
                                        key_padding_mask=feat_bbox_mask)
        attned_bbox = attned_bbox.mean(-2)

        vis_obs = self.conv(vis_obs)

        ray = self.ray_dense(torch.concat([ray_obs_1, ray_obs_2, ray_obs_3], dim=-1))

        # [batch, seq_len, group * 4] -> [batch, seq_len, group, 4]
        pre_action = pre_action.reshape(*pre_action.shape[:2], GROUP_SIZE, -1)

        # [batch, seq_len, group, f]
        x = torch.concat([attned_bbox,
                          vis_obs,
                          ray,
                          vec_obs,
                          pre_action], dim=-1)
        x = x * ~group_padding

        # [batch, seq_len, group, f]
        x = self.fuse_dense(x)

        # [batch, seq_len, group, f]
        x_, _ = self.attn_feat(x, x, x, key_padding_mask=group_padding.squeeze(-1))
        x = x + x_

        # [batch, seq_len, group, f] -> [batch, group, seq_len, f]
        x = x.transpose(1, 2)
        # [batch, group, seq_len, f] -> [batch * group, seq_len, f]
        x = x.reshape(-1, *x.shape[2:])

        if rnn_state is not None:
            # [batch, 1, group * f] -> [batch, 1, group, f]
            rnn_state = rnn_state.reshape(*rnn_state.shape[:2], GROUP_SIZE, -1)
            # [batch, 1, group, f] -> [batch, group, 1, f]
            rnn_state = rnn_state.transpose(1, 2)
            # [batch, group, 1, f] -> [batch * group, 1, f]
            rnn_state = rnn_state.reshape(-1, *rnn_state.shape[2:])

        state, hn = self.rnn(x, rnn_state)

        # [batch * group, seq_len, f] -> [batch, group, seq_len, f]
        state = state.reshape(batch, GROUP_SIZE, seq_len, -1)
        # [batch, group, seq_len, f] -> [batch, seq_len, group, f]
        state = state.transpose(1, 2)
        # [batch, seq_len, group * f]
        state = state.reshape(batch, seq_len, -1)

        # [batch * group, 1, f] -> [batch, group, 1, f]
        hn = hn.reshape(batch, GROUP_SIZE, 1, -1)
        # [batch, group, 1, f] -> [batch, 1, group, f]
        hn = hn.transpose(1, 2)
        # [batch, 1, group * f]
        hn = hn.reshape(batch, 1, -1)

        if padding_mask is not None:
            state = state * (~padding_mask).to(state.dtype).unsqueeze(-1)

        return state, hn


class ModelQ(m.ModelQ):
    def _build_model(self):
        embed_size = (self.state_size + self.c_action_size) // GROUP_SIZE
        self.attn = m.MultiheadAttention(embed_size, 1)

        self.c_dense = m.LinearLayers(self.state_size + self.c_action_size, 128, 2, output_size=1)

    def forward(self, state, c_action, obs_list):
        group_padding, *_, vec_obs = obs_list
        # [..., group, 1]
        group_padding = group_padding == 1.
        group_padding = torch.logical_or(group_padding, ~vec_obs.any(-1, keepdim=True))

        # [..., group * f] -> [..., group, f]
        state = state.reshape(*state.shape[:-1], GROUP_SIZE, -1)
        c_action = c_action.reshape(*c_action.shape[:-1], GROUP_SIZE, -1)

        x = torch.concat([state, c_action], dim=-1)
        x = x * group_padding
        x = x.reshape(*x.shape[:-2], -1)
        return None, self.c_dense(x)

        # attned_x, _ = self.attn(x, x, x,
        #                         key_padding_mask=group_padding.squeeze(-1))
        # attned_x = attned_x * group_padding
        # attned_x = attned_x.reshape(*attned_x.shape[:-2], -1)

        # return None, self.c_dense(attned_x)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        self.c_dense = m.LinearLayers(self.state_size // GROUP_SIZE, 128, 2)
        self.mean_dense = m.LinearLayers(self.c_dense.output_size, output_size=self.c_action_size // GROUP_SIZE)
        self.logstd_dense = m.LinearLayers(self.c_dense.output_size, output_size=self.c_action_size // GROUP_SIZE)

    def forward(self, state, obs_list):
        group_padding, *_, vec_obs = obs_list
        # [..., group, 1]
        group_padding = group_padding == 1.
        group_padding = torch.logical_or(group_padding, ~vec_obs.any(-1, keepdim=True))

        # [..., group * f] -> [..., group, f]
        state = state.reshape(*state.shape[:-1], GROUP_SIZE, -1)
        l = self.c_dense(state)
        mean = self.mean_dense(l)
        logstd = self.logstd_dense(l)
        mean = mean * ~group_padding
        logstd = logstd * ~group_padding
        group_padding = group_padding.repeat_interleave(mean.shape[-1], -1)

        # [..., group, f] -> [..., group * f]
        mean = mean.reshape(*mean.shape[:-2], -1)
        logstd = logstd.reshape(*logstd.shape[:-2], -1)
        group_padding = group_padding.reshape(*group_padding.shape[:-2], -1)

        c_policy = m.NormalWithPadding(torch.tanh(mean / 5.) * 5.,
                                       torch.exp(torch.clamp(logstd, -20, 0.5)),
                                       group_padding)

        return None, c_policy


class ModelRND(m.ModelRND):
    def _build_model(self):
        return super()._build_model(dense_n=128, dense_depth=2, output_size=128)

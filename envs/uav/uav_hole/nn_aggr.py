import torch

import algorithm.nn_models as m


class ModelRep(m.ModelBaseRNNRep):
    def _build_model(self):
        assert self.obs_shapes[0] == (3, 2, 9)  # AgentsBufferSensor
        assert self.obs_shapes[1] == (3, 84, 84, 3)
        assert self.obs_shapes[2] == (3, 9)

        self.conv = m.ConvLayers(84, 84, 3, 'simple',
                                 out_dense_n=64, out_dense_depth=2)
        self.attn = m.MultiheadAttention(self.conv.output_size, 8)

        self.rnn = m.GRU(3 * 9 + self.conv.output_size + self.c_action_size, 128, 1)

    def forward(self, obs_list, pre_action, rnn_state=None, padding_mask=None):
        feature_agents, vis_obs, vec_obs = obs_list

        vis_obs = self.conv(vis_obs)
        vis_obs, _ = self.attn(vis_obs, vis_obs, vis_obs)
        vis_obs = vis_obs.mean(-2)

        vec_obs = vec_obs.reshape(*vec_obs.shape[:-2], -1)

        state, hn = self.rnn(torch.concat([vec_obs, vis_obs, pre_action], dim=-1), rnn_state)

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

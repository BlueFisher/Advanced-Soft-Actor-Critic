import torch

import algorithm.nn_models as m

ModelVOverOption = m.ModelVOverOption
ModelTermination = m.ModelTermination


class ModelRep(m.ModelBaseRNNRep):
    def _build_model(self):
        assert self.obs_shapes[0] == (35,)
        assert self.obs_shapes[1] == (5,)

        self.ray_mlp = m.LinearLayers(35, output_size=16)

        if self.use_dilation:
            self.rnn = m.GRU(16, 64, 1)
        else:
            self.rnn = m.GRU(16 + self.c_action_size, 64, 1)

    def forward(self, obs_list, pre_action, rnn_state=None, padding_mask=None):
        ray_obs, vec_obs = obs_list

        ray_obs = self.ray_mlp(ray_obs)

        if self.use_dilation:
            output, hn = self.rnn(ray_obs, rnn_state)
        else:
            output, hn = self.rnn(torch.cat([ray_obs, pre_action], dim=-1), rnn_state)

        state = torch.cat([vec_obs, output], dim=-1)

        return state, hn


class ModelOptionRep(m.ModelBaseRNNRep):
    def _build_model(self):
        assert self.obs_shapes[0] == (64 + 5,)
        assert self.obs_shapes[1] == (35,)
        assert self.obs_shapes[2] == (5,)

        self.ray_mlp = m.LinearLayers(35, output_size=16)

        self.rnn = m.GRU(16 + self.c_action_size, 64, 1)

    def forward(self, obs_list, pre_action, rnn_state=None, padding_mask=None):
        high_state, ray_obs, vec_obs = obs_list

        ray_obs = self.ray_mlp(ray_obs)

        output, hn = self.rnn(torch.cat([ray_obs, pre_action], dim=-1), rnn_state)

        state = torch.cat([high_state, vec_obs, output], dim=-1)

        return state, hn


class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(c_state_n=64, c_state_depth=1,
                                    c_action_n=64, c_action_depth=1,
                                    c_dense_n=64, c_dense_depth=1)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        return super()._build_model(c_dense_n=64, c_dense_depth=1,
                                    mean_n=64, mean_depth=1,
                                    logstd_n=64, logstd_depth=1)


class ModelRND(m.ModelRND):
    def _build_model(self):
        return super()._build_model(dense_n=128, dense_depth=2, output_size=128)

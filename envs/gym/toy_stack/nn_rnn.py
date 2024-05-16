import torch

import algorithm.nn_models as m

ModelVOverOption = m.ModelVOverOption

MAP_WIDTH = 3
TARGET_TYPE_NUM = 3


class ModelRep(m.ModelBaseRNNRep):
    def _build_model(self, pe: str | None, gate: str | None):
        assert self.obs_shapes[0] == (MAP_WIDTH, TARGET_TYPE_NUM + 1)

        self.rnn = m.GRU(MAP_WIDTH * (TARGET_TYPE_NUM + 1), 64, 1)

    def forward(self, obs_list, pre_action, rnn_state=None, padding_mask=None):
        vec_obs = obs_list[0]
        vec_obs = vec_obs.reshape(*vec_obs.shape[:-2], MAP_WIDTH * (TARGET_TYPE_NUM + 1))

        output, hn = self.rnn(vec_obs, rnn_state)

        return output, hn


class ModelOptionRep(m.ModelBaseSimpleRep):
    def _build_model(self):
        assert self.obs_shapes[1] == (MAP_WIDTH, (TARGET_TYPE_NUM + 1))

        embed_size = MAP_WIDTH * (TARGET_TYPE_NUM + 1)

        # self.mlp = m.LinearLayers(embed_size, dense_n=embed_size, dense_depth=2)

    def forward(self, obs_list):
        if self._offline_action_index != -1:
            high_state, vec_obs, _ = obs_list
        else:
            high_state, vec_obs = obs_list

        # vec_obs = vec_obs.reshape(*vec_obs.shape[:-2], MAP_WIDTH * (TARGET_TYPE_NUM + 1))

        # output = torch.concat([high_state, self.mlp(vec_obs)], dim=-1)

        return high_state


class ModelTermination(m.ModelTermination):
    def _build_model(self):
        return super()._build_model(dense_n=128, dense_depth=2)

    def forward(self, state, obs_list):
        high_state, vec_obs = obs_list

        vec_obs = vec_obs.reshape(*vec_obs.shape[:-2], MAP_WIDTH * (TARGET_TYPE_NUM + 1))

        t = vec_obs.any(-1, keepdim=True)
        t = t.to(state.dtype)
        return t


class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(d_dense_n=128, d_dense_depth=2)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        return super()._build_model(d_dense_n=128, d_dense_depth=1)


class ModelRND(m.ModelRND):
    def _build_model(self):
        return super()._build_model(dense_n=32, dense_depth=2)

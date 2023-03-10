import torch

import algorithm.nn_models as m


class ModelRep(m.ModelBaseSimpleRep):
    def _build_model(self):
        assert self.obs_shapes[0] == (2, 3)  # AgentsBufferSensor
        assert self.obs_shapes[1] == (3, 3)  # TargetsBufferSensor
        assert self.obs_shapes[2] == (9, )

        self.dense = m.LinearLayers(6 + 9, dense_n=16, dense_depth=2)

    def forward(self, obs_list):
        feature_agents, feature_targets, vec_obs = obs_list

        feature_agents = feature_agents.reshape(*feature_agents.shape[:-2], -1)
        feature_targets = feature_targets.reshape(*feature_targets.shape[:-2], -1)

        features = self.dense(torch.concat([feature_agents, feature_targets], dim=-1))

        return torch.concat([vec_obs, features], dim=-1)


class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(c_dense_n=32, c_dense_depth=2)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        return super()._build_model(c_dense_n=32, c_dense_depth=2)


class ModelRND(m.ModelRND):
    def _build_model(self):
        return super()._build_model(dense_n=32, dense_depth=2, output_size=32)

import algorithm.nn_models as m


class ModelRep(m.ModelBaseSimpleRep):
    def _bulid_model(self):
        assert self.obs_shapes[2] == (11,)

    def forward(self, obs_list):
        ray_1, ray_2, vec = obs_list

        return vec


class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(c_dense_n=128, c_dense_depth=2)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        return super()._build_model(c_dense_n=128, c_dense_depth=2)


class ModelRND(m.ModelRND):
    def _build_model(self):
        return super()._build_model(dense_n=128, dense_depth=1, output_size=128)

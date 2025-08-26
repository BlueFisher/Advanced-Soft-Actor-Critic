import algorithm.nn_models as m

ModelRep = m.ModelSimpleRep

EXTRA_SIZE = 3


# class ModelRep(m.ModelBaseRep):
#     def forward(self, obs_list):
#         obs = obs_list[0][..., :-EXTRA_SIZE]
#         return obs


class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(c_dense_n=64, c_dense_depth=2)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        return super()._build_model(c_dense_n=64, c_dense_depth=2)

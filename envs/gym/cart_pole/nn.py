import algorithm.nn_models as m

ModelRep = m.ModelSimpleRep


class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(d_dense_n=20, d_dense_depth=1)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        return super()._build_model(d_dense_n=20, d_dense_depth=1)

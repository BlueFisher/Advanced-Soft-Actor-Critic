import algorithm.nn_models as m

ModelRep = m.ModelSimpleRep


class ModelQ(m.ModelQ):
    def _build_model(self):
        super()._build_model(c_dense_n=64, c_dense_depth=2)


class ModelPolicy(m.ModelPolicy):
    def _build_model(self):
        super()._build_model(c_dense_n=64, c_dense_depth=2)


class ModelForward(m.ModelForward):
    def _build_model(self):
        return super()._build_model(dense_n=self.state_size + self.action_size, dense_depth=1)


class ModelRND(m.ModelRND):
    def _build_model(self):
        return super()._build_model(dense_n=32, dense_depth=1, output_size=32)

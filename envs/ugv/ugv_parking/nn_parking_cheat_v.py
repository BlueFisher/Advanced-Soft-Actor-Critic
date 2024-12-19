from .nn_parking import *


class ModelVOverOptions(m.ModelVOverOptions):
    def _build_model(self):
        pass

    def forward(self, state):
        state = state[..., :NUM_OPTIONS]
        state = state * 2. - 1.
        return state

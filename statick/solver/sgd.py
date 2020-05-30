from tick.solver import SGD as TS
from statick.solver.solver import Solver as SOLVER

class SGD(SOLVER, TS):

    def __init__(self, **kwargs):
        TS.__init__(self, **kwargs)
        SOLVER.__init__(self, **kwargs)
        object.__setattr__(self, "_s_name", "sgd")

    def set_model(self, model):
        return SOLVER.set_model(self, SUPER=TS, model=model)

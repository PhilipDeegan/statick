from tick.solver import SVRG as TS
from statick.solver.solver import Solver as SOLVER

class SVRG(SOLVER, TS):

    def __init__(self, **kwargs):
        TS.__init__(self, **kwargs)
        SOLVER.__init__(self, **kwargs)
        object.__setattr__(self, "_s_name", "svrg")

    def set_model(self, model):
        return SOLVER.set_model(self, SUPER=TS, model=model)

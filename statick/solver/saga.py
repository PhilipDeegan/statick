from tick.solver import SAGA as TS
from statick.solver.solver import Solver as SOLVER

class SAGA(SOLVER, TS):

    def __init__(self, **kwargs):
        TS.__init__(self, **kwargs)
        SOLVER.__init__(self, **kwargs)
        object.__setattr__(self, "_s_name", "saga")
        if self.n_threads > 1:
            object.__setattr__(self, "_s_name", "asaga")

    def set_model(self, model):
        return SOLVER.set_super_model(self, SUPER=TS, model=model)

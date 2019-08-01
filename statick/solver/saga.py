import statick.solver
from tick.solver import SAGA as TickSAGA


class SAGA(TickSAGA):

    def __init__(self, **kwargs):
        _, _, _, kvs = inspect.getargvalues(inspect.currentframe())
        constructor_map = kvs.copy()
        args = inspect.getfullargspec(TickSAGA.__init__)[0]
        for k, v in kvs.items():
            if k not in args:
                del constructor_map[k]
        self.dao = statick.Solver.log_reg_fit_sd(train_set_a, train_set_b)
        return TickSAGA(**constructor_map)

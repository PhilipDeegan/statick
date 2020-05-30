import scipy, numpy as np
from tick.solver.base import SolverFirstOrderSto as SFOS
from tick.prox.base.prox import Prox as TPROX
from tick.base_model import ModelGeneralizedLinear as TMGL

class DummySolver(SFOS):
    def _set_cpp_solver(self, dtype_or_object_with_dtype): return None
    def set_epoch_size(self, v): pass
    def set_rand_max(self, v): pass
    def set_model(self, model): pass

class Solver(DummySolver):
    @staticmethod
    def CFUNC_RESOLVER(model, s = ""):
        X = model.features
        C = "s" if isinstance(X, scipy.sparse.csr.csr_matrix) else "d"
        T = "d" if X.dtype == np.dtype('float64') else "s"
        return model._MANGLING + s + C + T

    def __init__(self, **kwargs):
        object.__setattr__(self, "log_every_n_epochs", 10)
        if "log_every_n_epochs" in kwargs:
            object.__setattr__(self, "log_every_n_epochs", kwargs["log_every_n_epochs"])
        object.__setattr__(self, "_solver", DummySolver())
        object.__setattr__(self, "_dao", None)

    def set_model(self, SUPER, model: TMGL):
        import statick.solver.bin.statick_solver as statick_solver
        SUPER.set_model(self, model)
        func = self._s_name + "_" + Solver.CFUNC_RESOLVER(model, "_dao_")
        if self.n_threads > 1:
            object.__setattr__(self, "_dao", getattr(statick_solver, func)(model._dao, self.max_iter, self.epoch_size, self.n_threads))
            self._dao.history.tol.val = self.tol
        else:
            object.__setattr__(self, "_dao", getattr(statick_solver, func)(model._dao))
        if hasattr(self._dao, 'history'):
            self._dao.history.log_every_n_epochs = self.log_every_n_epochs
        if hasattr(self._dao, 'step'):
            self._dao.step = self.step
        return self

    def set_prox(self, prox: TPROX):
        if self.model is None:
            raise ValueError("Set model, then Prox")
        object.__setattr__(self, "_prox", prox._set_dao(self.model.features.dtype))
        return self

    def solve(self):
        import statick.solver.bin.statick_solver as statick_solver
        f = "solve_" + self._s_name + "_" + Solver.CFUNC_RESOLVER(self.model, "_" + self._prox._MANGLING + "_")
        max_iter = self.max_iter
        if self.n_threads > 1:
            max_iter = 1
        for i in range(max_iter):
            getattr(statick_solver, f)(self._dao, self.model._dao, self._prox._dao)
        if hasattr(self._dao, 'history'):
            object.__setattr__(self, "objectives", self._dao.history.objectives)
            object.__setattr__(self, "time_history", self._dao.history.time_history)

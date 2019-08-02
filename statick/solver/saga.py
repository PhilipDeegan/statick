import scipy, numpy as np
import statick.solver
from tick.solver import SAGA as TS
from tick.base_model import ModelGeneralizedLinear as TMGL

def MODEL_CFUNC_RESOLVER(model, s = ""):
    X = model.features
    C = "s" if isinstance(X, scipy.sparse.csr.csr_matrix) else "d"
    T = "d" if X.dtype == np.dtype('float64') else "s"
    return model._MANGLING + s + C + T + "_ptr"

class SAGA(TS):

    def __init__(self, **kwargs):
        TS.__init__(self, **kwargs)
        object.__setattr__(self, "_solver", self)
        object.__setattr__(self, "_dao", None)

    def set_model(self, model: TMGL):
        print("type(model)", type(model))
        if model is None: return
        if model._dao is None: raise ValueError("model._dao is None")
        TS.set_model(self, model)
        func = "SAGA_" + MODEL_CFUNC_RESOLVER(model, "_DAO_")
        print("func", func)
        print("type(model._dao)", type(model._dao))
        object.__setattr__(self, "_dao", getattr(statick.solver, func)(model._dao))
        return self._dao

    def solve(self):
        if self._dao is None:
            raise ValueError("solver._dao is None")
        if self.model._dao is None:
            raise ValueError("solver.model._dao is None")
        getattr(statick.solver, "saga_solve_" + str.lower(MODEL_CFUNC_RESOLVER(self.model, "_")))(self._dao, self.model._dao)

    def _set_cpp_solver(self, dtype_or_object_with_dtype):
        return None

    def set_epoch_size(self, v):
        pass

    def set_rand_max(self, v):
        pass

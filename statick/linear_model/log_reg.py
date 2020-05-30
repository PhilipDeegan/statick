import scipy, numpy as np
from tick.linear_model import ModelLogReg as TMLR

class ModelLogReg(TMLR):

    @staticmethod
    def CFUNC_RESOLVER(model, s = ""):
        X = model.features
        C = "s" if isinstance(X, scipy.sparse.csr.csr_matrix) else "d"
        T = "d" if X.dtype == np.dtype('float64') else "s"
        return model._MANGLING + s + C + T

    def __init__(self):
        TMLR.__init__(self)
        self._model = None
        object.__setattr__(self, "_MANGLING", "log_reg")
        object.__setattr__(self, "_dao", None)

    def fit(self, X, y):
        import statick.linear_model.bin.statick_linear_model as statick_linear_model
        TMLR.fit(self, X, y)
        func = ModelLogReg.CFUNC_RESOLVER(self, "_dao_")
        object.__setattr__(self, "_dao", getattr(statick_linear_model, func)(X, y))
        return self

    def _print(self):
        if self._dao:
            self._dao.print()

    def _build_cpp_model(self, dtype_or_object_with_dtype):
        return None

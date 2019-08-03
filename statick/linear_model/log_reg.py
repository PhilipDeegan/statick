import scipy, numpy as np
import statick.linear_model
from tick.linear_model import ModelLogReg as TMLR

class ModelLogReg(TMLR):

    def __init__(self):
        TMLR.__init__(self)
        self._model = None
        object.__setattr__(self, "_MANGLING", "LOG_REG")
        object.__setattr__(self, "_dao", None)

    def fit(self, X, y):
        TMLR.fit(self, X, y)
        object.__setattr__(self, "_dao", statick.linear_model.LOGREG_DAO_sd(X, y))
        # print("type(self._dao)", type(self._dao))
        return self._dao

    def _print(self):
        if self._dao:
            self._dao.print()

    def _build_cpp_model(self, dtype_or_object_with_dtype):
        return None

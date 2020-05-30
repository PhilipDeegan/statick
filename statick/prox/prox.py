import numpy as np
from tick.prox.base.prox import Prox as TP

class DummyProx(TP):
    def _call(self):
        return self
    def value(self):
        return self

class Prox(DummyProx):

    @staticmethod
    def CFUNC_RESOLVER(prox, dtype, s = ""):
        T = "d" if dtype == np.dtype('float64') else "s"
        return prox._MANGLING + s + T

    def __init__(self, **kwargs):
        object.__setattr__(self, "_prox", DummyProx())
        object.__setattr__(self, "_dao", None)

    def _set_dao(self, dtype):
        raise ValueError("Override this")

    def _get_dao(self, dtype):
        import statick.prox.bin.statick_prox as statick_prox
        return getattr(statick_prox, Prox.CFUNC_RESOLVER(self, dtype, "_"))

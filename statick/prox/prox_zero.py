
from tick.prox import ProxZero as TPZERO
from statick.prox.prox import Prox as PROX

class ProxZero(PROX, TPZERO):

    def __init__(self, **kwargs):
        PROX.__init__(self, **kwargs)
        TPZERO.__init__(self, **kwargs)
        object.__setattr__(self, "_MANGLING", "zero")

    def _set_dao(self, dtype):
        object.__setattr__(self, "_dao", self._get_dao(dtype)())
        return self

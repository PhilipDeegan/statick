
from tick.prox import ProxL2Sq as TPL2SQ
from .prox import Prox as PROX

class ProxL2Sq(PROX, TPL2SQ):

    def __init__(self, **kwargs):
        PROX.__init__(self, **kwargs)
        TPL2SQ.__init__(self, **kwargs)
        object.__setattr__(self, "_MANGLING", "l2sq")

    def _set_dao(self, dtype):
        object.__setattr__(self, "_dao", self._get_dao(dtype)(self.strength))
        return self


import statick.prox
from tick.prox import ProxL2Sq as TPL2SQ

class ProxL2Sq(TPL2SQ):

    def __init__(self, **kwargs):
        TPL2SQ.__init__(self, **kwargs)
        object.__setattr__(self, "_dao", statick.prox.PROX_L2SQ_d(self.strength))
        object.__setattr__(self, "_MANGLING", "l2sq")


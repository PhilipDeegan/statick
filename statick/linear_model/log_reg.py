import statick.linear_model
from tick.linear_model import LogisticRegression as TickLogReg

class LogisticRegression(TickLogReg):

    def __init__(self):
        TickLogReg.__init__(self)

    def fit(self, train_set_a, train_set_b):
        object.__setattr__(self, "_dao",
            statick.linear_model.log_reg_fit_sd(train_set_a, train_set_b))
        return self._dao

    def _print(self):
        if self._dao:
            self._dao.print()

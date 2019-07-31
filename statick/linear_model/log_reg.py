import statick.linear_model
from tick.linear_model import LogisticRegression as TickLogReg

class LogisticRegression(TickLogReg):

    def __init__(self):
        TickLogReg.__init__(self)

    # def fit(self, train_set_a, train_set_b):
    #     statick.linear_model.log_reg_fit_sd(train_set_a, train_set_b)

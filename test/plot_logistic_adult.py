import scipy, numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from tick.dataset import fetch_tick_dataset

train_set = fetch_tick_dataset('binary/adult/adult.trn.bz2')
test_set = fetch_tick_dataset('binary/adult/adult.tst.bz2')
X, y = train_set[0], train_set[1]

from statick.linear_model import ModelLogReg
from statick.prox import ProxL2Sq
from statick.solver import SAGA

solver = SAGA(step=1e-3, max_iter=100, verbose=False, tol=0) \
          .set_model(ModelLogReg().fit(X, y))                \
          .set_prox (ProxL2Sq(strength=((1. / X.shape[0]) + 1e-10)))
for i in range(111):
    solver.solve()

# predictions = learner.predict_proba(test_set[0])
# fpr, tpr, _ = roc_curve(test_set[1], predictions[:, 1])

# plt.figure(figsize=(6, 5))
# plt.plot(fpr, tpr, lw=2)
# plt.title("ROC curve on adult dataset (area = {:.2f})".format(auc(fpr, tpr)))
# plt.ylabel("True Positive Rate")
# plt.xlabel("False Positive Rate")
# plt.show()

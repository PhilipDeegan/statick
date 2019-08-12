import math
from statick.linear_model import ModelLogReg
from statick.prox import ProxL2Sq
from statick.solver import SAGA

import statick
X = statick.load_double_sparse2d("url.features.cereal")
y = statick.load_double_array("url.labels.cereal")

N_THREADS=8
solver = SAGA(step=1e-3, max_iter=200, verbose=False, tol=1e-5, n_threads=N_THREADS) \
          .set_model(ModelLogReg().fit(X, y))                                        \
          .set_prox (ProxL2Sq(strength=((1. / X.shape[0]) + 1e-10)))

solver.solve()

log_every_n_epochs = solver.log_every_n_epochs
objs = solver.objectives
history = solver.time_history

min_objective = min(objs)
print("min_objective", min_objective)
for i in range(len(objs)):
    log_dist = 0 if objs[i] == min_objective else math.log10(objs[i] - min_objective);
    print(str(N_THREADS) + " " + str(i * log_every_n_epochs) + " " + str(history[i]) + " " + "1e" + str(log_dist));

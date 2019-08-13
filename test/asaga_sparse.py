import math
from statick.linear_model import ModelLogReg
from statick.prox import ProxL2Sq
from statick.solver import SAGA
import statick

N_THREADS=8
MAX_ITER=200
STEP=0.00257480411965
TOL=1e-5

X = statick.load_double_sparse2d("url.features.cereal")
y = statick.load_double_array("url.labels.cereal")

solver = SAGA(step=STEP, max_iter=MAX_ITER, verbose=False, tol=TOL, n_threads=N_THREADS) \
          .set_model(ModelLogReg().fit(X, y))                                        \
          .set_prox (ProxL2Sq(strength=((1. / X.shape[0]) + 1e-10)))
solver.solve()
objs = solver.objectives
history = solver.time_history
min_objective = min(objs)
for i in range(len(objs)):
    log_dist = 0 if objs[i] == min_objective else math.log10(objs[i] - min_objective);
    print(str(N_THREADS) + " " + str(i * solver.log_every_n_epochs) + " " + \
    	  str(history[i]) + " " + "1e" + str(log_dist));

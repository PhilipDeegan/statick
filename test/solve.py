#!/usr/bin/python3
# expect tick first on PYTHONPATH

from tick.array.build.array import tick_double_sparse2d_from_file, tick_double_array_from_file
from tick.prox import ProxL2Sq; from tick.solver import SAGA; from tick.linear_model import ModelLogReg

X = tick_double_sparse2d_from_file("url.features.cereal")
n_samples = X.shape[0]; n_features = X.shape[1]
y = tick_double_array_from_file   ("url.labels.cereal")

model = ModelLogReg(fit_intercept=False).fit(X, y)
prox = ProxL2Sq((1. / n_samples) + 1e-10, range=(0, n_features))
asaga = SAGA(step=0.00257480411965, max_iter=200, tol=1e-10, verbose=False,
            n_threads=8, record_every=10)
asaga.set_model(model).set_prox(prox)
asaga.solve()
asaga.print_history()

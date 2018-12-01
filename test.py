#!/usr/bin/python3
import example
import numpy as np
from scipy.sparse import csr_matrix, random
from scipy.stats import rv_continuous
class CustomDistribution(rv_continuous):
    def _rvs(self, *args, **kwargs):
        return self._random_state.randn(*self._size)
X = CustomDistribution(seed=2906)
Y = X()  # get a frozen version of the distribution
d1 = random(4, 4, format='csr', density=0.25, random_state=2906, data_rvs=Y.rvs)

d1 = csr_matrix(
    (np.array([1., 2, 3, 4, 5]), np.array([2, 4, 6, 8, 10]),
     np.array([0, 5])), shape=(1, 12))

print(d1)
example.take_sparse2d(d1)
print(d1)

v = example.make_vector()
print(v)

vt = example.make_tuple_vector()
print(vt)
print(example.take_tuple_vector(vt))

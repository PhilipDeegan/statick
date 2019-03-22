#!/usr/bin/python3
import example
import numpy as np
from scipy.sparse import csr_matrix, random
from scipy.stats import rv_continuous
# class CustomDistribution(rv_continuous):
#     def _rvs(self, *args, **kwargs):
#         return self._random_state.randn(*self._size)
# X = CustomDistribution(seed=2906)
# Y = X()  # get a frozen version of the distribution
# d1 = random(4, 4, format='csr', density=0.25, random_state=2906, data_rvs=Y.rvs)

# d1 = csr_matrix(
#     (np.array([1., 2, 3, 4, 5]), np.array([2, 4, 6, 8, 10]),
#      np.array([0, 5])), shape=(1, 12))

# print(d1)
# example.take_sparse2d(d1)
# print(d1)

# v = example.make_vector()
# print(v)

# vt = example.make_tuple_vector()
# print(vt)
# print(example.take_tuple_vector(vt))


import pandas as pd
import pickle
import json
import faulthandler
faulthandler.enable()

def read_parameters() -> dict:
    with open("parameters.json", "r") as parameters_file:
        parameters_json = "".join(parameters_file.readlines())
        return json.loads(parameters_json)


def unpickle(filepath):
    with open(filepath, "rb") as file:
        return pickle.load(file)


def write_json(obj, filepath):
    with open(filepath, "w") as file:
        json.dump(obj, file)


if __name__ == "__main__":
    print("Read inputs")
    features = unpickle("features")
    labels = unpickle("labels")
    mapping = unpickle("mapping")
    n_age_groups = unpickle("age_groups")

    print("Read parameters")
    n_mols = len(mapping) - n_age_groups

    mols_lags = 45
    features_wo_age = [f[:, :n_mols] for f in features]

    l_f = []
    for f in features:
      print(type(f))
      l_f.append(f.toarray())
      # break
    l_l = []
    # for l in labels:
    #   print(type(l))
    #   l_l.append(l)
    #   # break
    print(type(l_f))
    print(type(labels))
    example.solve_svrg_sccs(l_f, labels)

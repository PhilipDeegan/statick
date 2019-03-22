from tick.survival import ConvSCCS
import pandas as pd
import pickle
import json
import numpy as np


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
    censoring = unpickle("censoring")
    mapping = unpickle("mapping")
    n_age_groups = unpickle("age_groups")

    print("Read parameters")
    n_mols = len(mapping) - n_age_groups

    # parameters = read_parameters()
    # mols_lags = parameters["lag"]
    mols_lags = 45

    n_lags = np.repeat(mols_lags, n_mols)
    penalized_features = np.arange(n_mols)

    features_wo_age = [f[:, :n_mols] for f in features]

    X = features_wo_age
    y = [l.ravel() for l in labels]
    c = censoring.ravel()

    model = ConvSCCS(
        n_lags,
        penalized_features,
        max_iter=500,
        verbose=True,
        record_every=10,
        tol=1e-5,
    )

    print(
        "Number of Samples {}; Number of Features {}; Number of buckets {}".format(
            len(X), X[0].shape[1], X[0].shape[0]
        )
    )

    coeffs, cv_track = model.fit_kfold_cv(
        X, y, c, C_tv_range=(1, 8), C_group_l1_range=(1, 8), n_cv_iter=60, n_folds=3
    )

    print("Saving results")
    pd.DataFrame(coeffs).to_csv("coeffs.csv")
    print("Saving CV tracker")
    write_json(cv_track.todict(), "cv_track.json")

    best_parameters = cv_track.find_best_params()

    model_custom = ConvSCCS(
        n_lags,
        penalized_features,
        max_iter=500,
        C_tv=best_parameters["C_tv"],
        C_group_l1=best_parameters["C_group_l1"],
        verbose=False,
        tol=1e-5,
    )
    print("Launching Bootstrap")
    coeffs_custom, ci = model_custom.fit(
        X, y, c, confidence_intervals=True, n_samples_bootstrap=100
    )
    pd.DataFrame(coeffs_custom).to_csv("coefficients_custom.csv")
    pd.DataFrame(ci).to_csv("ci.csv")
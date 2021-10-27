import numpy as np
from numpy.random import default_rng

from src.data.ate.data_class import PVTrainDataSet, PVTestDataSet
from sklearn.preprocessing import StandardScaler

A_scaler = None

def standardise(X):
    scaler = StandardScaler()
    if X.ndim == 1:
        X_scaled = scaler.fit_transform(X.reshape(-1,1)).squeeze()
        return X_scaled, scaler
    else:
        X_scaled = scaler.fit_transform(X).squeeze()
        return X_scaled, scaler


def generate_train_kpv_experiment(n_sample: int, seed=42, **kwargs):
    global A_scaler
    rng = default_rng(seed=seed)
    U2 = rng.uniform(-1, 2, n_sample)
    U1 = rng.uniform(0, 1, n_sample)
    U1 = U1 - (np.logical_and(U2 > 0, U2 < 1))
    W1 = U1 + rng.uniform(-1, 1, n_sample)
    W2 = U2 + rng.normal(0, 1, n_sample) * 3
    Z1 = U1 + rng.normal(0, 1, n_sample) * 3
    Z2 = U2 + rng.uniform(-1, 1, n_sample)
    A = U2 + rng.normal(0, 1, n_sample) * 0.05
    Y = U2 * np.cos(2*(A+0.3*U1+0.2))
    A_scaler = standardise(A)[1]
    return PVTrainDataSet(outcome=standardise(Y)[0][:, np.newaxis],
                          treatment=standardise(A)[0][:, np.newaxis],
                          treatment_proxy=standardise(np.c_[Z1, Z2])[0],
                          outcome_proxy=standardise(np.c_[W1, W2])[0],
                          backdoor=None)

def get_structure(A: float):
    n_sample = 100
    rng = default_rng(seed=42)
    U2 = rng.uniform(-1, 2, n_sample)
    U1 = rng.uniform(0, 1, n_sample)
    U1 = U1 - (np.logical_and(U2 > 0, U2 < 1))
    Y = U2 * np.cos(2 * (A + 0.3 * U1 + 0.2))
    return np.mean(Y)


def generate_test_kpv_experiment():
    global A_scaler
    test_a = np.linspace(-2.0, 2.0, 20)
    do_a = A_scaler.inverse_transform(test_a[:, np.newaxis])[:, 0]
    structure = np.array([get_structure(a) for a in do_a])
    return PVTestDataSet(structural=standardise(structure)[0][:, np.newaxis],
                         treatment=test_a[:, np.newaxis])

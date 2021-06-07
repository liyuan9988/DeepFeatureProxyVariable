from filelock import FileLock
from numpy.random import default_rng
import numpy as np

import pathlib

from src.data.data_class import PVTrainDataSet, PVTestDataSet



def generate_train_cevae_experiment(n_sample: int, seed=42, **kwargs):
    rng = default_rng(seed)
    z = rng.binomial(1, 0.5, size=(n_sample, 1))
    x = rng.normal(z, 5*z + 3*(1-z))
    t = rng.binomial(1, p=0.75*z + 0.25*(1-z))
    y_prob = 1 / (1 + np.exp(-3*(z+2*(2*t-1))))
    y = rng.binomial(1, p=y_prob)
    return PVTrainDataSet(treatment=t,
                          treatment_proxy=x,
                          outcome_proxy=x,
                          outcome=y,
                          backdoor=None)

def generate_test_cevae_experiment():
    y1 = (1 / (1 + np.exp(-6)) + 1 / (1 + np.exp(-9))) / 2
    y0 = (1 / (1 + np.exp(6)) + 1 / (1 + np.exp(3))) / 2
    return PVTestDataSet(structural=np.array([[y0], [y1]]),
                         treatment=np.array([[0],[1]]))

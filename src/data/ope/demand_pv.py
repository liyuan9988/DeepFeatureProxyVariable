import numpy as np
from numpy.random import default_rng

from .data_class import OPETrainDataSet, OPETestDataSet

from src.data.demand_pv import cal_outcome, generatate_demand_core


def generate_train_demand_pv_att(n_sample_additional: int, seed=42, **kwargs):
    rng = default_rng(seed=10000 + seed)
    demand, cost1, cost2, price, views, outcome = generatate_demand_core(n_sample_additional, rng)
    return OPETrainDataSet(outcome_proxy=views[:, np.newaxis],
                           covariate=price[:, np.newaxis])


def generate_test_demand_pv_att():
    n_sample = 1000
    rng = default_rng(seed=100042)
    demand, cost1, cost2, price, views, outcome = generatate_demand_core(n_sample, rng)
    new_treatment = np.maximum(price * 0.7, 10.0)
    new_outcome = cal_outcome(price=new_treatment,
                              views=views,
                              demand=demand)

    return OPETestDataSet(treatment=new_treatment[:, np.newaxis],
                          covariate=price[:, np.newaxis],
                          structural=new_outcome[:, np.newaxis])


def generate_train_demand_pv_policy(n_sample_additional: int, seed=42, **kwargs):
    rng = default_rng(seed=10000 + seed)
    demand, cost1, cost2, price, views, outcome = generatate_demand_core(n_sample_additional, rng)
    return OPETrainDataSet(outcome_proxy=views[:, np.newaxis],
                           covariate=np.c_[cost1, cost2])


def generate_test_demand_pv_policy():
    n_sample = 1000
    rng = default_rng(seed=100042)
    demand, cost1, cost2, price, views, outcome = generatate_demand_core(n_sample, rng)
    new_treatment = 23 + cost1 * cost2
    new_outcome = cal_outcome(price=new_treatment,
                              views=views,
                              demand=demand)

    return OPETestDataSet(treatment=new_treatment[:, np.newaxis],
                          covariate=np.c_[cost1, cost2],
                          structural=new_outcome[:, np.newaxis])

import numpy as np
import operator
import jax.numpy as jnp
from typing import Dict, Any
from pathlib import Path

from src.utils.jax_utils import mat_mul, cal_loocv_emb
from src.utils.kernel_func import AbsKernel, ColumnWiseGaussianKernel
from src.data.ope.data_class import OPETrainDataSet, OPETestDataSet
from src.data.ope import generate_train_data_ope, generate_test_data_ope
from src.models.kernelPV.model import KernelPVModel


def get_kernel_func(data_name: str) -> AbsKernel:
    return ColumnWiseGaussianKernel()


class KernelPVOPEModel:
    weight: np.ndarray
    train_covariate: np.ndarray

    def __init__(self, base_model: KernelPVModel, lam3_max: float, lam3_min: float,
                 n_lam3_search: int, scale: float = 1.0, **kwargs):
        self.base_model = base_model
        self.lam3_max = lam3_max
        self.lam3_min = lam3_min
        self.n_lam3_search = n_lam3_search
        self.scale = scale

    def fit(self, additional_data: OPETrainDataSet, data_name: str):
        self.covariate_kernel_func = get_kernel_func(data_name)
        self.covariate_kernel_func.fit(additional_data.covariate, scale=self.scale)
        kernel_mat = self.covariate_kernel_func.cal_kernel_mat(additional_data.covariate,
                                                               additional_data.covariate)
        kernel_WW = self.base_model.outcome_proxy_kernel_func.cal_kernel_mat(additional_data.outcome_proxy,
                                                                             additional_data.outcome_proxy)
        lam3_candidate_list = np.logspace(np.log10(self.lam3_min), np.log10(self.lam3_max), self.n_lam3_search)
        grid_search = dict(
            [(lam3_candi, cal_loocv_emb(kernel_WW, kernel_mat, lam3_candi)) for lam3_candi in lam3_candidate_list])
        self.lam3, loo = min(grid_search.items(), key=operator.itemgetter(1))
        n_additional_data = additional_data.outcome_proxy.shape[0]
        kernel_mat += self.lam3 * n_additional_data * np.eye(n_additional_data)
        target_w_kernel = self.base_model.outcome_proxy_kernel_func.cal_kernel_mat(self.base_model.train_outcome_proxy,
                                                                                   additional_data.outcome_proxy)
        self.weight = np.linalg.solve(kernel_mat, target_w_kernel.T)
        self.train_covariate = additional_data.covariate

    def predict(self, treatment: np.ndarray, covariate: np.ndarray):
        test_kernel = self.covariate_kernel_func.cal_kernel_mat(self.train_covariate,
                                                                covariate)

        w_weight_pred = self.weight.T @ test_kernel
        test_treatment_kernel = self.base_model.treatment_kernel_func.cal_kernel_mat(self.base_model.train_treatment,
                                                                                     treatment)
        return jnp.asarray(jnp.diag(mat_mul(mat_mul(w_weight_pred.T, self.base_model.alpha), test_treatment_kernel)))

    def evaluate(self, test_data: OPETestDataSet):
        pred = self.predict(treatment=test_data.treatment, covariate=test_data.covariate)
        return np.mean((pred - test_data.structural) ** 2)


def kpv_ope_experiments_simple(data_config: Dict[str, Any], model_param: Dict[str, Any],
                               one_mdl_dump_dir: Path,
                               random_seed: int = 42, verbose: int = 0):

    org_data, additional_data = generate_train_data_ope(data_config, random_seed)
    test_data = generate_test_data_ope(data_config)
    base_model = KernelPVModel(**model_param["base_param"])
    if data_config["name"].startswith("demand"):
        data_name = "demand"
    else:
        raise ValueError
    base_model.fit(org_data, data_name)
    value_pred = np.mean(base_model.predict_bridge(additional_data.new_treatment,
                                                   additional_data.outcome_proxy))
    return np.abs(value_pred - np.mean(test_data.structural))

def kpv_ope_experiments(data_config: Dict[str, Any], model_param: Dict[str, Any],
                        one_mdl_dump_dir: Path,
                        random_seed: int = 42, verbose: int = 0):
    org_data, additional_data = generate_train_data_ope(data_config, random_seed)
    test_data = generate_test_data_ope(data_config)
    base_model = KernelPVModel(**model_param["base_param"])
    if data_config["name"].startswith("demand"):
        data_name = "demand"
    else:
        raise ValueError
    base_model.fit(org_data, data_name)
    ope_model = KernelPVOPEModel(base_model=base_model, **model_param)
    ope_model.fit(additional_data, data_config["name"])
    pred = ope_model.predict(test_data.treatment, test_data.covariate)
    np.savetxt(one_mdl_dump_dir.joinpath(f"{random_seed}.pred.txt"), pred)
    if test_data.structural is not None:
        l2_loss = np.mean((pred - test_data.structural) ** 2)
        np.savetxt(one_mdl_dump_dir.joinpath(f"{random_seed}.l2loss.txt"), np.array([l2_loss]))
        return np.abs(np.mean(pred) - np.mean(test_data.structural))
    else:
        return 0.0

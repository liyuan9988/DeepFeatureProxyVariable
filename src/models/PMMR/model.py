import numpy as np
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

from src.utils.kernel_func import ColumnWiseGaussianKernel, AbsKernel, BinaryKernel, GaussianKernel
from src.data.ate import generate_train_data_ate, generate_test_data_ate, get_preprocessor_ate
from src.data.ate.data_class import PVTrainDataSet, PVTestDataSet


def get_kernel_func(data_name: str) -> Tuple[AbsKernel, AbsKernel, AbsKernel, AbsKernel]:
    if data_name in ("dsprite_org", "dsprite"):
        return BinaryKernel(), GaussianKernel(), GaussianKernel(), GaussianKernel()
    elif data_name == "deaner":
        return GaussianKernel(), GaussianKernel(), GaussianKernel(), GaussianKernel()
    else:
        return ColumnWiseGaussianKernel(), ColumnWiseGaussianKernel(), ColumnWiseGaussianKernel(), ColumnWiseGaussianKernel()



class PMMRModel:
    treatment_kernel_func: AbsKernel
    treatment_proxy_kernel_func: AbsKernel
    outcome_proxy_kernel_func: AbsKernel
    backdoor_kernel_func: AbsKernel

    alpha: np.ndarray
    x_mean_vec: Optional[np.ndarray]
    w_mean_vec: np.ndarray
    train_treatment: np.ndarray
    train_outcome_proxy: np.ndarray

    def __init__(self, lam1, lam2=0.0001, scale=1.0, **kwargs):
        self.lam1 = lam1
        self.lam2 = lam2
        self.scale = scale
        self.x_mean_vec = None

    def fit(self, train_data: PVTrainDataSet, data_name: str):
        kernels = get_kernel_func(data_name)
        self.treatment_kernel_func = kernels[0]
        self.treatment_proxy_kernel_func = kernels[1]
        self.outcome_proxy_kernel_func = kernels[2]
        self.backdoor_kernel_func = kernels[3]
        n_train = train_data.treatment.shape[0]


        # Set scales to be median
        self.treatment_proxy_kernel_func.fit(train_data.treatment_proxy, scale=self.scale)
        self.treatment_kernel_func.fit(train_data.treatment, scale=self.scale)
        self.outcome_proxy_kernel_func.fit(train_data.outcome_proxy, scale=self.scale)

        if train_data.backdoor is not None:
            self.backdoor_kernel_func.fit(train_data.backdoor, scale=self.scale)

        treatment_mat = self.treatment_kernel_func.cal_kernel_mat(train_data.treatment,
                                                                  train_data.treatment)
        treatment_proxy_mat = self.treatment_proxy_kernel_func.cal_kernel_mat(train_data.treatment_proxy,
                                                                              train_data.treatment_proxy)
        outcome_proxy_mat = self.outcome_proxy_kernel_func.cal_kernel_mat(train_data.outcome_proxy,
                                                                          train_data.outcome_proxy)
        backdoor_mat = np.ones((n_train, n_train))
        if train_data.backdoor is not None:
            backdoor_mat = self.backdoor_kernel_func.cal_kernel_mat(train_data.backdoor,
                                                                    train_data.backdoor)
            self.x_mean_vec = np.mean(backdoor_mat, axis=0)[:, np.newaxis]
        W = treatment_mat * treatment_proxy_mat * backdoor_mat
        L = treatment_mat * outcome_proxy_mat * backdoor_mat
        self.alpha = np.linalg.solve(L @ W @ L + self.lam1 * n_train * L + self.lam2 * n_train * np.eye(n_train),
                                     L @ W @ train_data.outcome)
        self.w_mean_vec = np.mean(outcome_proxy_mat, axis=0)[:, np.newaxis]
        self.train_treatment = train_data.treatment
        self.train_outcome_proxy = train_data.outcome_proxy

    def predict(self, treatment: np.ndarray) -> np.ndarray:
        test_kernel = self.treatment_kernel_func.cal_kernel_mat(self.train_treatment, treatment)
        test_kernel *= self.w_mean_vec
        if self.x_mean_vec is not None:
            test_kernel = test_kernel * self.x_mean_vec

        pred = self.alpha.T @ test_kernel
        return pred.T

    def predict_bridge(self, treatment: np.ndarray, output_proxy: np.ndarray) -> np.ndarray:
        test_kernel = self.treatment_kernel_func.cal_kernel_mat(self.train_treatment, treatment)
        test_kernel *= self.outcome_proxy_kernel_func.cal_kernel_mat(self.train_outcome_proxy, output_proxy)

        pred = self.alpha.T @ test_kernel
        return pred.T

    def evaluate(self, test_data: PVTestDataSet):
        pred = self.predict(treatment=test_data.treatment)
        return np.mean((pred - test_data.structural) ** 2)


def pmmr_experiments(data_config: Dict[str, Any], model_param: Dict[str, Any],
                     one_mdl_dump_dir: Path,
                     random_seed: int = 42, verbose: int = 0):
    train_data_org = generate_train_data_ate(data_config=data_config, rand_seed=random_seed)
    test_data_org = generate_test_data_ate(data_config=data_config)

    preprocessor = get_preprocessor_ate(data_config.get("preprocess", "Identity"))
    train_data = preprocessor.preprocess_for_train(train_data_org)
    test_data = preprocessor.preprocess_for_test_input(test_data_org)

    model = PMMRModel(**model_param)
    model.fit(train_data, data_config["name"])
    pred = model.predict(test_data.treatment)
    pred = preprocessor.postprocess_for_prediction(pred)
    if test_data.structural is not None:
        if data_config["name"] in ("kpv", "deaner"):
            return np.mean(np.abs(pred - test_data.structural))
        return np.mean((pred - test_data.structural) ** 2)
    else:
        return 0.0

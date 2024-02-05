import numpy as np
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import jax.numpy as jnp
import jax.scipy.linalg as jsla
import operator

from sklearn.preprocessing import StandardScaler

from src.utils.kernel_func import ColumnWiseGaussianKernel, AbsKernel, BinaryKernel, GaussianKernel
from src.data.ate import generate_train_data_ate, generate_test_data_ate, get_preprocessor_ate
from src.data.ate.data_class import PVTrainDataSet, PVTestDataSet, split_train_data
from src.utils.jax_utils import Hadamard_prod, mat_mul, mat_trans, modif_kron, cal_loocv_emb, cal_loocv_alpha, \
    stage2_weights


def get_kernel_func(data_name: str) -> Tuple[AbsKernel, AbsKernel, AbsKernel, AbsKernel]:
    if data_name in ("dsprite", "dsprite_ver2"):
        return BinaryKernel(), GaussianKernel(), GaussianKernel(), GaussianKernel()
    else:
        return ColumnWiseGaussianKernel(), ColumnWiseGaussianKernel(), ColumnWiseGaussianKernel(), ColumnWiseGaussianKernel()


class KernelPVModel:
    treatment_kernel_func: AbsKernel
    treatment_proxy_kernel_func: AbsKernel
    outcome_proxy_kernel_func: AbsKernel
    backdoor_kernel_func: AbsKernel

    alpha: np.ndarray
    x_mean_vec: Optional[np.ndarray]
    w_mean_vec: np.ndarray
    train_treatment: np.ndarray
    train_outcome_proxy: np.ndarray

    def __init__(self, split_ratio: float, lam1=None, lam2=None, lam1_max=None, lam1_min=None,
                 n_lam1_search=None, lam2_max=None, lam2_min=None, n_lam2_search=None,
                 scale=1.0, **kwargs):
        self.lam1: Optional[float] = lam1
        self.lam2: Optional[float] = lam2
        self.lam1_max: Optional[float] = lam1_max
        self.lam1_min: Optional[float] = lam1_min
        self.n_lam1_search: Optional[int] = n_lam1_search
        self.lam2_max: Optional[float] = lam2_max
        self.lam2_min: Optional[float] = lam2_min
        self.n_lam2_search: Optional[int] = n_lam2_search
        self.scale: float = scale
        self.split_ratio: float = split_ratio
        self.x_mean_vec = None


    def cal_kernel_mat_ZAX(self, data1: PVTrainDataSet, data2: PVTrainDataSet):
        kernel_mat = self.treatment_kernel_func.cal_kernel_mat(data1.treatment, data2.treatment)
        kernel_mat *= self.treatment_proxy_kernel_func.cal_kernel_mat(data1.treatment_proxy,
                                                                      data2.treatment_proxy)
        if data1.backdoor is not None:
            kernel_mat *= self.backdoor_kernel_func.cal_kernel_mat(data1.backdoor,
                                                                   data2.backdoor)
        return kernel_mat

    def tune_lam1(self, kernel_1st, K_W1W1):
        lam1_candidate_list = np.logspace(np.log10(self.lam1_min), np.log10(self.lam1_max), self.n_lam1_search)
        grid_search = dict(
            [(lam1_candi, cal_loocv_emb(kernel_1st, K_W1W1, lam1_candi)) for lam1_candi in lam1_candidate_list])
        self.lam1, loo = min(grid_search.items(), key=operator.itemgetter(1))

    def tune_lam2(self, Gamma_w, kw1_gamma, kernel_mat_2nd, Sigma, train_data_2nd):
        n_train_2nd = train_data_2nd.treatment.shape[0]
        mk_gamma_I = mat_trans(modif_kron(Gamma_w, np.eye(n_train_2nd)))
        D_t = modif_kron(kw1_gamma, kernel_mat_2nd)
        lam2_candidate_list = np.logspace(np.log10(self.lam2_min), np.log10(self.lam2_max), self.n_lam2_search)
        grid_search = dict(
            [(lam2_candi, cal_loocv_alpha(D_t, Sigma, mk_gamma_I, train_data_2nd.treatment, lam2_candi)) for lam2_candi
             in
             lam2_candidate_list])
        self.lam2, loo = min(grid_search.items(), key=operator.itemgetter(1))
        self.alpha = mat_mul(mk_gamma_I, jsla.solve(Sigma + n_train_2nd * self.lam2 * np.eye(n_train_2nd),
                                                    train_data_2nd.outcome))

    def fit(self, train_data: PVTrainDataSet, data_name: str):
        train_data_1st, train_data_2nd = split_train_data(train_data, self.split_ratio)
        kernels = get_kernel_func(data_name)
        self.treatment_kernel_func = kernels[0]
        self.treatment_proxy_kernel_func = kernels[1]
        self.outcome_proxy_kernel_func = kernels[2]
        self.backdoor_kernel_func = kernels[3]
        n_train_1st = train_data_1st.treatment.shape[0]
        n_train_2nd = train_data_2nd.treatment.shape[0]


        # Set scales to be median
        self.treatment_proxy_kernel_func.fit(train_data_1st.treatment_proxy, scale=self.scale)
        self.treatment_kernel_func.fit(train_data_1st.treatment, scale=self.scale)
        self.outcome_proxy_kernel_func.fit(train_data_1st.outcome_proxy, scale=self.scale)
        if train_data_1st.backdoor is not None:
            self.backdoor_kernel_func.fit(train_data_1st.backdoor, scale=self.scale)

        kernel_1st = self.cal_kernel_mat_ZAX(train_data_1st, train_data_1st)
        K_W1W1 = self.outcome_proxy_kernel_func.cal_kernel_mat(train_data_1st.outcome_proxy,
                                                               train_data_1st.outcome_proxy)

        if self.lam1 is None:
            self.tune_lam1(kernel_1st, K_W1W1)
        kernel_1st += self.lam1 * n_train_1st * np.eye(n_train_1st)

        kernel_1st_2nd = self.cal_kernel_mat_ZAX(train_data_1st, train_data_2nd)
        Gamma_w = jsla.solve(kernel_1st, kernel_1st_2nd)

        kw1_gamma = mat_mul(K_W1W1, Gamma_w)
        g_kw1_g = mat_mul(mat_trans(Gamma_w), kw1_gamma)

        kernel_mat_2nd = self.treatment_kernel_func.cal_kernel_mat(train_data_2nd.treatment,
                                                                   train_data_2nd.treatment)
        if train_data_2nd.backdoor is not None:
            K_X2X2 = self.backdoor_kernel_func.cal_kernel_mat(train_data_2nd.backdoor,
                                                              train_data_2nd.backdoor)
            kernel_mat_2nd = Hadamard_prod(kernel_mat_2nd, K_X2X2)
            self.x_mean_vec = np.mean(K_X2X2, axis=0)[:, np.newaxis]

        Sigma = g_kw1_g * kernel_mat_2nd

        if self.lam2 is None:
            self.tune_lam2(Gamma_w, kw1_gamma, kernel_mat_2nd, Sigma, train_data_2nd)
        else:
            self.alpha = stage2_weights(Gamma_w, jsla.solve(Sigma + n_train_2nd * self.lam2 * np.eye(n_train_2nd),
                                                            train_data_2nd.outcome))
        self.alpha = self.alpha.reshape(n_train_1st, n_train_2nd)
        self.w_mean_vec = np.mean(K_W1W1, axis=0)[:, np.newaxis]
        self.train_treatment = train_data_2nd.treatment
        self.train_outcome_proxy = train_data_1st.outcome_proxy

    def predict(self, treatment: np.ndarray) -> np.ndarray:

        test_kernel = self.treatment_kernel_func.cal_kernel_mat(self.train_treatment, treatment)
        if self.x_mean_vec is not None:
            test_kernel = test_kernel * self.x_mean_vec
        pred = jnp.asarray(mat_mul(mat_mul(self.w_mean_vec.T, self.alpha), test_kernel)).T
        return pred

    def predict_bridge(self, treatment: np.ndarray, output_proxy: np.ndarray):
        test_kernel = self.treatment_kernel_func.cal_kernel_mat(self.train_treatment, treatment)
        proxy_kernel = self.outcome_proxy_kernel_func.cal_kernel_mat(output_proxy, self.train_outcome_proxy)
        n_test = treatment.shape[0]
        pred = [jnp.asarray(mat_mul(mat_mul(proxy_kernel[[i], :], self.alpha), test_kernel[:, [i]])) for i in range(n_test)]
        pred = np.array(pred)
        if pred.ndim == 1:
            pred = pred[:, np.newaxis]
        return pred

    def evaluate(self, test_data: PVTestDataSet):
        pred = self.predict(treatment=test_data.treatment)
        return np.mean((pred - test_data.structural) ** 2)


def kpv_experiments(data_config: Dict[str, Any], model_param: Dict[str, Any],
                    one_mdl_dump_dir: Path,
                    random_seed: int = 42, verbose: int = 0):
    train_data_org = generate_train_data_ate(data_config=data_config, rand_seed=random_seed)
    test_data_org = generate_test_data_ate(data_config=data_config)

    preprocessor = get_preprocessor_ate(data_config.get("preprocess", "Identity"))
    train_data = preprocessor.preprocess_for_train(train_data_org)
    test_data = preprocessor.preprocess_for_test_input(test_data_org)

    model = KernelPVModel(**model_param)
    model.fit(train_data, data_config["name"])
    pred = model.predict(test_data.treatment)
    pred = preprocessor.postprocess_for_prediction(pred)
    np.savetxt(one_mdl_dump_dir.joinpath(f"{random_seed}.pred.txt"), pred)
    if test_data.structural is not None:
        if data_config["name"] in ("kpv", "deaner"):
            return np.mean(np.abs(pred - test_data.structural))
        return np.mean((pred - test_data.structural) ** 2)
    else:
        return 0.0

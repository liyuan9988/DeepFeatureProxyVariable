from typing import List
import numpy as np
from scipy.spatial.distance import cdist


class AbsKernel:
    def fit(self, data: np.ndarray, **kwargs) -> None:
        raise NotImplementedError

    def cal_kernel_mat(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class LinearDotKernel(AbsKernel):
    def __init__(self):
        super(LinearDotKernel, self).__init__()

    def fit(self, data: np.ndarray, **kwargs) -> None:
        pass

    def cal_kernel_mat(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        return data1.dot(data2.T)

class BinaryKernel(AbsKernel):

    def __init__(self):
        super(BinaryKernel, self).__init__()

    def fit(self, data: np.ndarray, **kwargs):
        pass

    def cal_kernel_mat(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        res = data1.dot(data2.T)
        res += (1 - data1).dot(1 - data2.T)
        return res



class WarfarinBackdoorKernel(AbsKernel):
    sigma: np.float32

    def __init__(self):
        super(WarfarinBackdoorKernel, self).__init__()

    def fit(self, data: np.ndarray, **kwargs) -> None:
        dense_data = data[:, :2]
        dist = cdist(dense_data, dense_data, 'sqeuclidean')
        self.sigma = np.median(dist)

    def cal_kernel_mat(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        dense_data1 = data1[:, :2]
        cat_data1 = data1[:, [2, 7, 9, 13, 14]]
        dense_data2 = data2[:, :2]
        cat_data2 = data2[:, [2, 7, 9, 13, 14]]
        dists = cdist(dense_data1, dense_data2, 'sqeuclidean')
        dists = dists / self.sigma
        inner_dot = cat_data1.dot(cat_data2.T)
        inner_dot = inner_dot + (1 - cat_data1).dot(1 - cat_data2.T)
        # inner_dot = inner_dot / inner_dot[0, 0] - 1.0
        # print(inner_dot)
        return np.exp(-dists) * inner_dot


class FourthOrderGaussianKernel(AbsKernel):
    bandwidth: np.float32

    def fit(self, data: np.ndarray, **kwargs) -> None:
        dists = cdist(data, data, 'sqeuclidean')
        self.bandwidth = np.sqrt(np.median(dists))

    def cal_kernel_mat(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        diff_data = data1[:, np.newaxis, :] - data2[np.newaxis, :, :]
        u = diff_data / self.bandwidth
        kernel_tensor = np.exp(- u ** 2 / 2.0) * (3 - u ** 2) / 2.0 / np.sqrt(6.28)
        return np.product(kernel_tensor, axis=2)


class SixthOrderGaussianKernel(AbsKernel):
    bandwidth: np.float32

    def fit(self, data: np.ndarray, **kwargs) -> None:
        dists = cdist(data, data, 'sqeuclidean')
        self.bandwidth = np.sqrt(np.median(dists))

    def cal_kernel_mat(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        diff_data = data1[:, np.newaxis, :] - data2[np.newaxis, :, :]
        u = diff_data / self.bandwidth
        kernel_tensor = np.exp(- u ** 2 / 2.0) * (15 - 10 * u ** 2 + u ** 4) / 8.0 / np.sqrt(6.28)
        return np.product(kernel_tensor, axis=2)


class FourthOrderEpanechnikovKernel(AbsKernel):
    bandwidth: np.float32

    def fit(self, data: np.ndarray, **kwargs) -> None:
        n_data = data.shape[0]
        assert data.shape[1] == 1
        self.bandwidth = 3.03 * np.std(data) / (n_data ** 0.12)

    def cal_kernel_mat(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        assert data1.shape[1] == 1
        assert data2.shape[1] == 1
        dists = cdist(data1, data2, 'sqeuclidean') / (self.bandwidth ** 2)
        mat = (1.0 - dists) * (3 / 4) / self.bandwidth
        mat = np.maximum(mat, 0.0)
        mat = mat * (1.0 - 7 * dists / 3) * 15 / 8
        return mat


class EpanechnikovKernel(AbsKernel):
    bandwidth: np.float32

    def fit(self, data: np.ndarray, **kwargs) -> None:
        n_data = data.shape[0]
        assert data.shape[1] == 1
        self.bandwidth = 2.34 * np.std(data) / (n_data ** 0.25)

    def cal_kernel_mat(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        assert data1.shape[1] == 1
        assert data2.shape[1] == 1
        dists = cdist(data1, data2, 'sqeuclidean') / (self.bandwidth ** 2)
        mat = (1.0 - dists) * (3 / 4) / self.bandwidth
        mat = np.maximum(mat, 0.0)
        return mat


class GaussianKernel(AbsKernel):
    sigma: np.float32

    def __init__(self):
        super(GaussianKernel, self).__init__()

    def fit(self, data: np.ndarray, scale=1.0, **kwargs) -> None:
        dists = cdist(data, data, 'euclidean')
        self.sigma = 2 * (np.median(dists) ** 2) * scale

    def cal_kernel_mat(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        dists = cdist(data1, data2, 'sqeuclidean')
        dists = dists / self.sigma
        return np.exp(-dists)


class ColumnWiseGaussianKernel(AbsKernel):
    kernel_list: List[GaussianKernel]

    def __init__(self):
        super(ColumnWiseGaussianKernel, self).__init__()
        self.kernel_list = []

    def fit(self, data: np.ndarray, scale=1.0, **kwargs) -> None:
        for i in range(data.shape[1]):
            self.kernel_list.append(GaussianKernel())
            self.kernel_list[-1].fit(data[:, [i]], scale=scale)

    def cal_kernel_mat(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        assert self.kernel_list is not None
        res = []
        for i in range(len(self.kernel_list)):
            res.append(self.kernel_list[i].cal_kernel_mat(data1[:, [i]], data2[:, [i]])[:,:,np.newaxis])
        res_mat = np.concatenate(res, axis=2)
        return np.product(res_mat, axis=2)

from typing import NamedTuple, Optional
import numpy as np
import torch
from sklearn.model_selection import train_test_split


class PVTrainDataSet(NamedTuple):
    treatment: np.ndarray
    treatment_proxy: np.ndarray
    outcome_proxy: np.ndarray
    outcome: np.ndarray
    backdoor: Optional[np.ndarray]


class PVTestDataSet(NamedTuple):
    treatment: np.ndarray
    structural: Optional[np.ndarray]


class PVTrainDataSetTorch(NamedTuple):
    treatment: torch.Tensor
    treatment_proxy: torch.Tensor
    outcome_proxy: torch.Tensor
    outcome: torch.Tensor
    backdoor: Optional[torch.Tensor]

    @classmethod
    def from_numpy(cls, train_data: PVTrainDataSet):
        backdoor = None
        if train_data.backdoor is not None:
            backdoor = torch.tensor(train_data.backdoor, dtype=torch.float32)
        return PVTrainDataSetTorch(treatment=torch.tensor(train_data.treatment, dtype=torch.float32),
                                   treatment_proxy=torch.tensor(train_data.treatment_proxy, dtype=torch.float32),
                                   outcome_proxy=torch.tensor(train_data.outcome_proxy, dtype=torch.float32),
                                   backdoor=backdoor,
                                   outcome=torch.tensor(train_data.outcome, dtype=torch.float32))

    def to_gpu(self):
        backdoor = None
        if self.backdoor is not None:
            backdoor = self.backdoor.cuda()
        return PVTrainDataSetTorch(treatment=self.treatment.cuda(),
                                   treatment_proxy=self.treatment_proxy.cuda(),
                                   outcome_proxy=self.outcome_proxy.cuda(),
                                   backdoor=backdoor,
                                   outcome=self.outcome.cuda())


class PVTestDataSetTorch(NamedTuple):
    treatment: torch.Tensor
    structural: Optional[torch.Tensor]

    @classmethod
    def from_numpy(cls, test_data: PVTestDataSet):
        structural = None
        if test_data.structural is not None:
            structural = torch.tensor(test_data.structural, dtype=torch.float32)
        return PVTestDataSetTorch(treatment=torch.tensor(test_data.treatment, dtype=torch.float32),
                                  structural=structural)

    def to_gpu(self):
        structural = None
        if self.structural is not None:
            structural = self.structural.cuda()
        return PVTestDataSetTorch(treatment=self.treatment.cuda(),
                                  structural=structural)


def split_train_data(train_data: PVTrainDataSet, split_ratio=0.5):
    if split_ratio < 0.0:
        return train_data, train_data

    n_data = train_data[0].shape[0]
    idx_train_1st, idx_train_2nd = train_test_split(np.arange(n_data), train_size=split_ratio)

    def get_data(data, idx):
        return data[idx] if data is not None else None

    train_1st_data = PVTrainDataSet(*[get_data(data, idx_train_1st) for data in train_data])
    train_2nd_data = PVTrainDataSet(*[get_data(data, idx_train_2nd) for data in train_data])
    return train_1st_data, train_2nd_data

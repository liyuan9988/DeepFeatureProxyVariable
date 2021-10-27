from typing import NamedTuple, Optional
import numpy as np
import torch


class OPETrainDataSet(NamedTuple):
    outcome_proxy: np.ndarray
    covariate: np.ndarray  # can be treatment or treatment_proxy
    new_treatment: np.ndarray

class OPETestDataSet(NamedTuple):
    treatment: np.ndarray
    covariate: np.ndarray  # can be old treatment or treatment_proxy
    structural: np.ndarray


class OPETrainDataSetTorch(NamedTuple):
    outcome_proxy: torch.Tensor
    covariate: torch.Tensor  # can be treatment or treatment_proxy

    @classmethod
    def from_numpy(cls, train_data: OPETrainDataSet):
        return OPETrainDataSetTorch(outcome_proxy=torch.tensor(train_data.outcome_proxy, dtype=torch.float32),
                                    covariate=torch.tensor(train_data.covariate, dtype=torch.float32))

    def to_gpu(self):
        return OPETrainDataSetTorch(outcome_proxy=self.outcome_proxy.cuda(),
                                    covariate=self.covariate.cuda())


class OPETestDataSetTorch(NamedTuple):
    treatment: torch.Tensor
    covariate: torch.Tensor  # can be old treatment or treatment_proxy
    structural: torch.Tensor

    @classmethod
    def from_numpy(cls, test_data: OPETestDataSet):
        return OPETestDataSetTorch(treatment=torch.tensor(test_data.treatment, dtype=torch.float32),
                                   covariate=torch.tensor(test_data.covariate, dtype=torch.float32),
                                   structural=torch.tensor(test_data.structural, dtype=torch.float32))

    def to_gpu(self):
        return OPETestDataSetTorch(treatment=self.treatment.cuda(),
                                   covariate=self.treatment.cuda(),
                                   structural=self.treatment.cuda())

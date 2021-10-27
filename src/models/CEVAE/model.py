from typing import Optional
import torch
from torch import nn
import numpy as np
import logging

from src.data.ate.data_class import PVTestDataSet, PVTestDataSetTorch

logger = logging.getLogger()


class CEVAEModel:
    z_samples: Optional[torch.Tensor]

    def __init__(self, distribution: nn.Module):
        self.distribution = distribution

    def fit(self, proxy, treatment, outcome, n_samples: int = 10):
        with torch.no_grad():
            z_dist = self.distribution.q_z_xty(proxy, treatment, outcome)
        self.z_samples = torch.cat([z_dist.sample() for i in range(n_samples)],
                                   dim=0)


    def predict_t(self, treatment: torch.Tensor):
        res = []
        with torch.no_grad():
            n_z_samples = self.z_samples.size()[0]
            for i in range(treatment.size()[0]):
                treatment_one = treatment[i, :].expand(n_z_samples, -1)
                pred_all = self.distribution.p_y_zt(self.z_samples,
                                                    treatment_one).mean
                res.append(torch.mean(pred_all))
        return torch.tensor(res).unsqueeze(1)

    def predict(self, treatment: np.ndarray):
        treatment_t = torch.tensor(treatment, dtype=torch.float32)
        return self.predict_t(treatment_t).data.numpy()

    def evaluate_t(self, test_data: PVTestDataSetTorch):
        target = test_data.structural
        with torch.no_grad():
            pred = self.predict_t(test_data.treatment)
        return torch.mean((target - pred) ** 2)

    def evaluate(self, test_data: PVTestDataSet):
        return self.evaluate_t(PVTestDataSetTorch.from_numpy(test_data)).data.item()

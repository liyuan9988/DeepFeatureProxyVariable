import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm


def build_extractor_ope(data_name: str) -> nn.Module:
    if data_name == "demand_att":
        return nn.Sequential(nn.Linear(1, 32),
                             nn.ReLU(),
                             nn.Linear(32, 16),
                             nn.ReLU(), nn.Linear(16, 8), nn.ReLU())
    elif data_name == "demand_policy":
        return nn.Sequential(nn.Linear(2, 32),
                             nn.ReLU(),
                             nn.Linear(32, 16),
                             nn.ReLU(), nn.Linear(16, 8), nn.ReLU())
    else:
        raise ValueError(f"data name {data_name} is not valid")

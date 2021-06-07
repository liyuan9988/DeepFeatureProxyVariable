from typing import Optional, Tuple
import torch
from torch import nn
from torch.nn.utils import spectral_norm
from torch.nn import functional as F


def build_net_for_dsprite() -> Tuple[
    nn.Module, nn.Module, nn.Module, nn.Module, Optional[nn.Module], Optional[nn.Module]]:
    treatment_1st_net = nn.Sequential(spectral_norm(nn.Linear(64 * 64, 1024)),
                                      nn.ReLU(),
                                      spectral_norm(nn.Linear(1024, 512)),
                                      nn.ReLU(),
                                      nn.BatchNorm1d(512),
                                      spectral_norm(nn.Linear(512, 128)),
                                      nn.ReLU(),
                                      spectral_norm(nn.Linear(128, 32)),
                                      nn.ReLU())

    treatment_2nd_net = nn.Sequential(spectral_norm(nn.Linear(64 * 64, 1024)),
                                      nn.ReLU(),
                                      spectral_norm(nn.Linear(1024, 512)),
                                      nn.ReLU(),
                                      nn.BatchNorm1d(512),
                                      spectral_norm(nn.Linear(512, 128)),
                                      nn.ReLU(),
                                      spectral_norm(nn.Linear(128, 32)),
                                      nn.ReLU())

    outcome_proxy_net = nn.Sequential(spectral_norm(nn.Linear(64 * 64, 1024)),
                                      nn.ReLU(),
                                      spectral_norm(nn.Linear(1024, 512)),
                                      nn.ReLU(),
                                      nn.BatchNorm1d(512),
                                      spectral_norm(nn.Linear(512, 128)),
                                      nn.ReLU(),
                                      spectral_norm(nn.Linear(128, 32)),
                                      nn.ReLU())

    treatment_proxy_net = nn.Sequential(nn.Linear(3, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, 64),
                                        nn.ReLU(),
                                        nn.Linear(64, 32),
                                        nn.ReLU())

    return treatment_1st_net, treatment_2nd_net, treatment_proxy_net, outcome_proxy_net, None, None

from typing import Tuple, Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm


from .nn_structure_for_demand import DemandDistribution
from .nn_structure_for_dsprite import DspriteDistribution
from .nn_structure_for_cevae import CEVAEDistribution
import logging

logger = logging.getLogger()



def build_extractor(data_name: str, hidden_dim, n_sample):
    if data_name.startswith("demand"):
        logger.info("build for demand")
        return DemandDistribution(n_hidden_dim=hidden_dim, n_learning_sample=n_sample)
    elif data_name == "dsprite":
        logger.info("build for dsprite")
        return DspriteDistribution(n_hidden_dim=hidden_dim, n_learning_sample=n_sample)
    elif data_name == "cevae":
        return CEVAEDistribution(n_hidden_dim=hidden_dim, n_learning_sample=n_sample)
    else:
        raise ValueError(f"data name {data_name} is not valid")

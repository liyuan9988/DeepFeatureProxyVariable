from typing import Tuple, Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

from .nn_structure_for_kpv_experiment import build_net_for_kpv
from .nn_structure_for_demand import build_net_for_demand
from .nn_structure_for_dsprite import build_net_for_dsprite
from .nn_structure_for_deaner_experiment import build_net_for_deaner
import logging

logger = logging.getLogger()


def build_extractor(data_name: str) -> Tuple[
    nn.Module, nn.Module, nn.Module, nn.Module, Optional[nn.Module], Optional[nn.Module]]:
    if data_name == "kpv":
        logger.info("build for kpv")
        return build_net_for_kpv()
    elif data_name.startswith("demand"):
        logger.info("build for demand")
        return build_net_for_demand()
    elif data_name == "dsprite":
        logger.info("build for dsprite")
        return build_net_for_dsprite()
    elif data_name == "deaner":
        logger.info("build for deaner")
        return build_net_for_deaner()
    else:
        raise ValueError(f"data name {data_name} is not valid")

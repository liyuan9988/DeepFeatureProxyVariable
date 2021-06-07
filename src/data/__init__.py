from typing import Dict, Any, Optional

from .kpv_experiment import generate_train_kpv_experiment, generate_test_kpv_experiment
from .demand_pv import generate_test_demand_pv, generate_train_demand_pv
from .dsprite import generate_train_dsprite, generate_test_dsprite
from .data_class import PVTestDataSet, PVTrainDataSet
from .cevae_experiment import generate_train_cevae_experiment, generate_test_cevae_experiment


def generate_train_data(data_config: Dict[str, Any], rand_seed: int) -> PVTrainDataSet:
    data_name = data_config["name"]
    if data_name == "kpv":
        return generate_train_kpv_experiment(seed=rand_seed, **data_config)
    elif data_name == "demand":
        return generate_train_demand_pv(seed=rand_seed, **data_config)
    elif data_name == "dsprite":
        return generate_train_dsprite(rand_seed=rand_seed, **data_config)
    elif data_name == "cevae":
        return generate_train_cevae_experiment(rand_seed=rand_seed, **data_config)
    else:
        raise ValueError(f"data name {data_name} is not valid")


def generate_test_data(data_config: Dict[str, Any]) -> Optional[PVTestDataSet]:
    data_name = data_config["name"]
    if data_name == "kpv":
        return generate_test_kpv_experiment()
    elif data_name == "demand":
        return generate_test_demand_pv()
    elif data_name == "dsprite":
        return generate_test_dsprite()
    elif data_name == "cevae":
        return generate_test_cevae_experiment()
    else:
        raise ValueError(f"data name {data_name} is not valid")

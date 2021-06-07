from typing import Dict, Any, Optional, Tuple

from .demand_pv import generate_test_demand_pv_att, generate_train_demand_pv_att, generate_test_demand_pv_policy, \
    generate_train_demand_pv_policy
from ..demand_pv import generate_train_demand_pv, generate_test_demand_pv
from ..data_class import PVTestDataSet, PVTrainDataSet
from .data_class import OPETrainDataSet, OPETestDataSet


def generate_train_data(data_config: Dict[str, Any], rand_seed: int) -> Tuple[PVTrainDataSet, OPETrainDataSet]:
    data_name = data_config["name"]
    if data_name == "demand_att":
        org_data = generate_train_demand_pv(seed=rand_seed, **data_config)
        additional_data = generate_train_demand_pv_att(seed=rand_seed, **data_config)
        return org_data, additional_data
    elif data_name == "demand_policy":
        org_data = generate_train_demand_pv(seed=rand_seed, **data_config)
        additional_data = generate_train_demand_pv_policy(seed=rand_seed, **data_config)
        return org_data, additional_data
    else:
        raise ValueError(f"data name {data_name} is not valid")


def generate_test_data(data_config: Dict[str, Any]) -> OPETestDataSet:
    data_name = data_config["name"]
    if data_name == "demand_att":
        return generate_test_demand_pv_att()
    elif data_name == "demand_policy":
        return generate_test_demand_pv_policy()
    else:
        raise ValueError(f"data name {data_name} is not valid")

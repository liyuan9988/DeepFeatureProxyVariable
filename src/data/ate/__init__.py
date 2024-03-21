from typing import Dict, Any, Optional, Tuple

from sklearn.preprocessing import StandardScaler

from src.data.ate.preprocess import get_preprocessor_ate
from src.data.ate.kpv_experiment_sim import generate_train_kpv_experiment, generate_test_kpv_experiment
from src.data.ate.deaner_experiment import generate_train_deaner_experiment, generate_test_deaner_experiment
from src.data.ate.demand_pv import generate_test_demand_pv, generate_train_demand_pv
from src.data.ate.dsprite import generate_train_dsprite, generate_test_dsprite
from src.data.ate.dsprite_ver2 import generate_train_dsprite_ver2, generate_test_dsprite_ver2
from src.data.ate.data_class import PVTestDataSet, PVTrainDataSet
from src.data.ate.cevae_experiment import generate_train_cevae_experiment, generate_test_cevae_experiment


def generate_train_data_ate(data_config: Dict[str, Any], rand_seed: int) -> PVTrainDataSet:
    data_name = data_config["name"]
    if data_name == "kpv":
        return generate_train_kpv_experiment(seed=rand_seed, **data_config)
    elif data_name == "demand":
        return generate_train_demand_pv(seed=rand_seed, **data_config)
    elif data_name == "dsprite_org":
        return generate_train_dsprite(rand_seed=rand_seed, **data_config)
    elif data_name == "dsprite":
        return generate_train_dsprite_ver2(rand_seed=rand_seed, **data_config)
    elif data_name == "cevae":
        return generate_train_cevae_experiment(rand_seed=rand_seed, **data_config)
    elif data_name == "deaner":
        return generate_train_deaner_experiment(seed=rand_seed, **data_config)
    else:
        raise ValueError(f"data name {data_name} is not valid")


def generate_test_data_ate(data_config: Dict[str, Any]) -> Optional[PVTestDataSet]:
    data_name = data_config["name"]
    if data_name == "kpv":
        return generate_test_kpv_experiment()
    elif data_name == "demand":
        return generate_test_demand_pv()
    elif data_name == "dsprite_org":
        return generate_test_dsprite()
    elif data_name == "dsprite":
        return generate_test_dsprite_ver2()
    elif data_name == "cevae":
        return generate_test_cevae_experiment()
    elif data_name == "deaner":
        return generate_test_deaner_experiment(id=data_config["id"])
    else:
        raise ValueError(f"data name {data_name} is not valid")


def standardise(data: PVTrainDataSet) -> Tuple[PVTrainDataSet, Dict[str, StandardScaler]]:
    treatment_proxy_scaler = StandardScaler()
    treatment_proxy_s = treatment_proxy_scaler.fit_transform(data.treatment_proxy)

    treatment_scaler = StandardScaler()
    treatment_s = treatment_scaler.fit_transform(data.treatment)

    outcome_scaler = StandardScaler()
    outcome_s = outcome_scaler.fit_transform(data.outcome)

    outcome_proxy_scaler = StandardScaler()
    outcome_proxy_s = outcome_proxy_scaler.fit_transform(data.outcome_proxy)

    backdoor_s = None
    backdoor_scaler = None
    if data.backdoor is not None:
        backdoor_scaler = StandardScaler()
        backdoor_s = backdoor_scaler.fit_transform(data.backdoor)

    train_data = PVTrainDataSet(treatment=treatment_s,
                                treatment_proxy=treatment_proxy_s,
                                outcome_proxy=outcome_proxy_s,
                                outcome=outcome_s,
                                backdoor=backdoor_s)

    scalers = dict(treatment_proxy_scaler=treatment_proxy_scaler,
                   treatment_scaler=treatment_scaler,
                   outcome_proxy_scaler=outcome_proxy_scaler,
                   outcome_scaler=outcome_scaler,
                   backdoor_scaler=backdoor_scaler)

    return train_data, scalers

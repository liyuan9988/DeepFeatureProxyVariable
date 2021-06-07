from filelock import FileLock
import numpy as np

import pathlib

from src.data.data_class import PVTrainDataSet, PVTestDataSet

DATA_PATH = pathlib.Path(__file__).resolve().parent.parent.parent.joinpath("data/")
DATA_SEED = 1009


def generate_train_kpv_experiment(n_sample: int, use_x: bool, seed=42, **kwargs):
    with FileLock(DATA_PATH.joinpath("./data.lock")):
        data = np.load(DATA_PATH.joinpath('KPV_experiments/main_seed%s_std.npz' % DATA_SEED))
    rng = np.random.default_rng(seed)
    idx = rng.choice(data["train_y"].shape[0], n_sample)
    if use_x:
        backdoor = data["train_x"][idx, :]
    else:
        backdoor = None

    return PVTrainDataSet(outcome=data["train_y"][idx, np.newaxis],
                          treatment=data["train_a"][idx, np.newaxis],
                          treatment_proxy=data["train_z"][idx, :],
                          outcome_proxy=data["train_w"][idx, :],
                          backdoor=backdoor)


def generate_test_kpv_experiment():
    with FileLock(DATA_PATH.joinpath("./data.lock")):
        test_data = np.load(DATA_PATH.joinpath('KPV_experiments/do_A_seed%s_std.npz' % DATA_SEED))

    return PVTestDataSet(structural=test_data['gt_EY_do_A'][:, np.newaxis],
                         treatment=test_data['do_A'][:, np.newaxis])

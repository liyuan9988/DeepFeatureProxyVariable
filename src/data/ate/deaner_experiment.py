from filelock import FileLock
import numpy as np

import pathlib

from src.data.ate.data_class import PVTrainDataSet, PVTestDataSet


DATA_PATH = pathlib.Path(__file__).resolve().parent.parent.parent.parent.joinpath("data/")
DATA_SEED = 1009


def generate_train_deaner_experiment(id: str, seed=42, **kwargs):
    with FileLock(DATA_PATH.joinpath("./data.lock")):
        seed = (seed % 10) * 100
        if seed == 0:
            seed = 1000

        data = np.load(DATA_PATH.joinpath(f'sim_1d_no_x/main_edu_{id}_80_seed{seed}.npz'))

    return PVTrainDataSet(outcome=data["train_y"],
                          treatment=data["train_a"],
                          treatment_proxy=data["train_z"],
                          outcome_proxy=data["train_w"],
                          backdoor=None)


def generate_test_deaner_experiment(id: str, **kwargs):
    with FileLock(DATA_PATH.joinpath("./data.lock")):
        test_data = np.load(DATA_PATH.joinpath(f'sim_1d_no_x/do_A_edu_{id}_80_seed100.npz'))

    return PVTestDataSet(structural=test_data['gt_EY_do_A'][:, np.newaxis],
                         treatment=test_data['do_A'])

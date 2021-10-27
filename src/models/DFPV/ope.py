from __future__ import annotations
from typing import Dict, Any, Optional
import torch
import logging
from pathlib import Path

import numpy as np

from src.models.DFPV.nn_structure.ope_nn_structure import build_extractor_ope
from src.models.DFPV.trainer import DFPVTrainer
from src.models.DFPV.model import DFPVModel
from src.data.ope import generate_train_data_ope, generate_test_data_ope
from src.data.ope.data_class import OPETrainDataSetTorch, OPETestDataSetTorch, OPETrainDataSet, OPETestDataSet

from src.utils.pytorch_linear_reg_utils import fit_linear, linear_reg_pred, add_const_col, linear_reg_loss

logger = logging.getLogger()


class DFPVOPEModel(object):
    weight: torch.Tensor

    def __init__(self, base_model: DFPVModel, data_configs: Dict[str, Any], train_params: Dict[str, Any],
                 gpu_flg: bool = False, dump_folder: Optional[Path] = None):
        self.data_config = data_configs
        self.gpu_flg = gpu_flg and torch.cuda.is_available()
        if self.gpu_flg:
            logger.info("gpu mode")
        # configure training params
        self.lam3: float = train_params["lam3"]
        self.n_epoch: int = train_params["n_epoch"]
        self.add_intercept = True
        self.weight_decay = train_params["weight_decay"]

        # build networks
        self.covariate_network = build_extractor_ope(data_configs["name"])
        if self.gpu_flg:
            self.covariate_network.to("cuda:0")

        self.opt = torch.optim.Adam(self.covariate_network.parameters(),
                                    weight_decay=self.weight_decay)
        self.base_model = base_model

    def fit(self, train_data: OPETrainDataSet, verbose: int = 0):
        train_data_t = OPETrainDataSetTorch.from_numpy(train_data)
        if self.gpu_flg:
            train_data_t = train_data_t.to_gpu()

        self.covariate_network.train(True)
        self.base_model.outcome_proxy_net.train(False)
        with torch.no_grad():
            outcome_proxy_feature = self.base_model.outcome_proxy_net(train_data_t.outcome_proxy)

        for t in range(self.n_epoch):
            self.opt.zero_grad()
            covariate_feature = self.covariate_network(train_data_t.covariate)
            aug_covariate_feaure = self.aug_covariate_feature(covariate_feature)
            loss = linear_reg_loss(outcome_proxy_feature, aug_covariate_feaure, self.lam3)
            loss.backward()
            if verbose >= 2:
                logger.info(f"stage3 learning: {loss.item()}")
            self.opt.step()

        self.covariate_network.train(False)
        covariate_feature = self.covariate_network(train_data_t.covariate)
        aug_covariate_feaure = self.aug_covariate_feature(covariate_feature)
        self.weight = fit_linear(outcome_proxy_feature, aug_covariate_feaure, self.lam3)

    def aug_covariate_feature(self, covariate_feature):
        if self.add_intercept:
            covariate_feature = add_const_col(covariate_feature)
        return covariate_feature

    def predict_t(self, treatment: torch.Tensor, covariate: torch.Tensor):
        treatment_feature = self.base_model.treatment_2nd_net(treatment)
        n_data = treatment_feature.shape[0]

        covariate_feature = self.covariate_network(covariate)
        aug_covariate_feaure = self.aug_covariate_feature(covariate_feature)
        outcome_proxy_mat = linear_reg_pred(aug_covariate_feaure, self.weight)
        mean_backdoor_feature_mat = None

        feature = DFPVModel.augment_stage2_feature(outcome_proxy_mat,
                                                   treatment_feature,
                                                   mean_backdoor_feature_mat,
                                                   self.base_model.add_stage2_intercept)
        return linear_reg_pred(feature, self.base_model.stage2_weight)

    def predict(self, treatment: np.ndarray, covariate: np.ndarray):
        treatment_t = torch.tensor(treatment, dtype=torch.float32)
        covariate_t = torch.tensor(covariate, dtype=torch.float32)
        return self.predict_t(treatment_t, covariate_t).data.numpy()

    def evaluate_t(self, test_data: OPETestDataSetTorch):
        target = test_data.structural
        with torch.no_grad():
            pred = self.predict_t(test_data.treatment, test_data.covariate)
        return torch.mean((target - pred) ** 2)

    def evaluate(self, test_data: OPETestDataSet):
        return self.evaluate_t(OPETestDataSetTorch.from_numpy(test_data)).data.item()


def dfpv_ope_experiments_simple(data_config: Dict[str, Any], model_param: Dict[str, Any],
                                one_mdl_dump_dir: Path, random_seed: int = 42, verbose: int = 0):
    dump_dir = one_mdl_dump_dir.joinpath(f"{random_seed}")
    torch.manual_seed(random_seed)
    org_data, additional_data = generate_train_data_ope(data_config=data_config, rand_seed=random_seed)
    test_data = generate_test_data_ope(data_config=data_config)
    trainer = DFPVTrainer(data_config, model_param["base_param"], False, dump_dir)
    base_mdl = trainer.train(org_data, verbose)
    value_pred = np.mean(base_mdl.predict_bridge(additional_data.new_treatment, additional_data.outcome_proxy))
    return np.abs(np.mean(value_pred) - np.mean(test_data.structural))


def dfpv_ope_experiments(data_config: Dict[str, Any], model_param: Dict[str, Any],
                         one_mdl_dump_dir: Path, random_seed: int = 42, verbose: int = 0):
    dump_dir = one_mdl_dump_dir.joinpath(f"{random_seed}")
    torch.manual_seed(random_seed)
    org_data, additional_data = generate_train_data_ope(data_config=data_config, rand_seed=random_seed)
    test_data = generate_test_data_ope(data_config=data_config)

    trainer = DFPVTrainer(data_config, model_param["base_param"], False, dump_dir)
    base_mdl = trainer.train(org_data, verbose)

    ope_model = DFPVOPEModel(base_mdl, data_config, model_param, False, dump_dir)
    ope_model.fit(additional_data, verbose)
    test_data_t = OPETestDataSetTorch.from_numpy(test_data)
    if trainer.gpu_flg:
        torch.cuda.empty_cache()
        test_data_t = test_data_t.to_gpu()

    pred: np.ndarray = ope_model.predict_t(test_data_t.treatment, test_data_t.covariate).data.cpu().numpy()
    oos_loss = 0.0
    if test_data.structural is not None:
        oos_loss: float = ope_model.evaluate_t(test_data_t).data.item()
    np.savetxt(one_mdl_dump_dir.joinpath(f"{random_seed}.pred.txt"), pred)
    np.savetxt(one_mdl_dump_dir.joinpath(f"{random_seed}.l2loss.txt"), np.array([oos_loss]))

    return np.abs(np.mean(pred) - np.mean(test_data.structural))

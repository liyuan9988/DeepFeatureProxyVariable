from __future__ import annotations
from typing import Dict, Any, Optional
import torch
from torch import nn
import logging
from pathlib import Path

import numpy as np

from src.models.DFPV.nn_structure import build_extractor
from src.models.DFPV.model import DFPVModel
from src.data.ate import generate_train_data_ate, generate_test_data_ate, get_preprocessor_ate
from src.data.ate.data_class import PVTrainDataSetTorch, PVTestDataSetTorch, split_train_data, PVTrainDataSet, \
    PVTestDataSet
from src.utils.pytorch_linear_reg_utils import linear_reg_loss

logger = logging.getLogger()


class DFPVTrainer(object):

    def __init__(self, data_configs: Dict[str, Any], train_params: Dict[str, Any],
                 gpu_flg: bool = False, dump_folder: Optional[Path] = None):
        self.data_config = data_configs
        self.gpu_flg = gpu_flg and torch.cuda.is_available()
        if self.gpu_flg:
            logger.info("gpu mode")
        # configure training params
        self.lam1: float = train_params["lam1"]
        self.lam2: float = train_params["lam2"]
        self.stage1_iter: int = train_params["stage1_iter"]
        self.stage2_iter: int = train_params["stage2_iter"]
        self.n_epoch: int = train_params["n_epoch"]
        self.split_ratio: float = train_params["split_ratio"]
        self.add_stage1_intercept = True
        self.add_stage2_intercept = True
        self.treatment_weight_decay = train_params["treatment_weight_decay"]
        self.treatment_proxy_weight_decay = train_params["treatment_proxy_weight_decay"]
        self.outcome_proxy_weight_decay = train_params["outcome_proxy_weight_decay"]
        self.backdoor_weight_decay = train_params["backdoor_weight_decay"]

        # build networks
        networks = build_extractor(data_configs["name"])
        self.treatment_1st_net: nn.Module = networks[0]
        self.treatment_2nd_net: nn.Module = networks[1]
        self.treatment_proxy_net: nn.Module = networks[2]
        self.outcome_proxy_net: nn.Module = networks[3]
        self.backdoor_1st_net: Optional[nn.Module] = networks[4]
        self.backdoor_2nd_net: Optional[nn.Module] = networks[5]
        if self.gpu_flg:
            self.treatment_1st_net.to("cuda:0")
            self.treatment_2nd_net.to("cuda:0")
            self.treatment_proxy_net.to("cuda:0")
            self.outcome_proxy_net.to("cuda:0")
            if self.backdoor_1st_net is not None:
                self.backdoor_1st_net.to("cuda:0")
                self.backdoor_2nd_net.to("cuda:0")

        self.treatment_1st_opt = torch.optim.Adam(self.treatment_1st_net.parameters(),
                                                  weight_decay=self.treatment_weight_decay)
        self.treatment_2nd_opt = torch.optim.Adam(self.treatment_2nd_net.parameters(),
                                                  weight_decay=self.treatment_weight_decay)

        self.treatment_proxy_opt = torch.optim.Adam(self.treatment_proxy_net.parameters(),
                                                    weight_decay=self.treatment_proxy_weight_decay)
        self.outcome_proxy_opt = torch.optim.Adam(self.outcome_proxy_net.parameters(),
                                                  weight_decay=self.outcome_proxy_weight_decay)

        if self.backdoor_1st_net:
            self.backdoor_1st_opt = torch.optim.Adam(self.backdoor_1st_net.parameters(),
                                                     weight_decay=self.backdoor_weight_decay)
            self.backdoor_2nd_opt = torch.optim.Adam(self.backdoor_2nd_net.parameters(),
                                                     weight_decay=self.backdoor_weight_decay)

    def train(self, train_data: PVTrainDataSet, verbose: int = 0) -> DFPVModel:
        train_1st, train_2nd = split_train_data(train_data, self.split_ratio)
        train_1st_t = PVTrainDataSetTorch.from_numpy(train_1st)
        train_2nd_t = PVTrainDataSetTorch.from_numpy(train_2nd)
        if self.gpu_flg:
            train_1st_t = train_1st_t.to_gpu()
            train_2nd_t = train_2nd_t.to_gpu()

        for t in range(self.n_epoch):
            self.stage1_update(train_1st_t, verbose)
            self.stage2_update(train_1st_t, train_2nd_t, verbose)
            if verbose >= 1:
                logger.info(f"Epoch {t} ended")

        mdl = DFPVModel(self.treatment_1st_net, self.treatment_2nd_net, self.treatment_proxy_net,
                        self.outcome_proxy_net,
                        self.backdoor_1st_net, self.backdoor_2nd_net, self.add_stage1_intercept,
                        self.add_stage2_intercept)

        mdl.fit_t(train_1st_t, train_2nd_t, self.lam1, self.lam2)
        return mdl

    def stage1_update(self, train_1st_t: PVTrainDataSetTorch, verbose: int):
        self.treatment_1st_net.train(True)
        self.treatment_2nd_net.train(False)
        self.treatment_proxy_net.train(True)
        self.outcome_proxy_net.train(False)
        if self.backdoor_1st_net:
            self.backdoor_1st_net.train(True)
            self.backdoor_2nd_net.train(False)

        with torch.no_grad():
            outcome_proxy_feature = self.outcome_proxy_net(train_1st_t.outcome_proxy)

        for i in range(self.stage1_iter):
            self.treatment_1st_opt.zero_grad()
            self.treatment_proxy_opt.zero_grad()
            if self.backdoor_1st_net:
                self.backdoor_1st_opt.zero_grad()

            treatment_1st_feature_1st = self.treatment_1st_net(train_1st_t.treatment)
            treatment_proxy_feature_1st = self.treatment_proxy_net(train_1st_t.treatment_proxy)
            backdoor_1st_feature_1st = None
            if self.backdoor_1st_net:
                backdoor_1st_feature_1st = self.backdoor_1st_net(train_1st_t.backdoor)

            feature = DFPVModel.augment_stage1_feature(treatment_feature=treatment_1st_feature_1st,
                                                       treatment_proxy_feature=treatment_proxy_feature_1st,
                                                       backdoor_feature=backdoor_1st_feature_1st,
                                                       add_stage1_intercept=self.add_stage1_intercept)
            loss = linear_reg_loss(outcome_proxy_feature, feature, self.lam1)
            loss.backward()
            if verbose >= 2:
                logger.info(f"stage1 learning: {loss.item()}")

            self.treatment_1st_opt.step()
            self.treatment_proxy_opt.step()
            if self.backdoor_1st_net:
                self.backdoor_1st_opt.step()

    def stage2_update(self, train_1st_data_t, train_2nd_data_t, verbose):
        self.treatment_1st_net.train(False)
        self.treatment_2nd_net.train(True)
        self.treatment_proxy_net.train(False)
        self.outcome_proxy_net.train(True)
        if self.backdoor_1st_net:
            self.backdoor_1st_net.train(False)
            self.backdoor_2nd_net.train(True)

        with torch.no_grad():
            treatment_1st_feature_1st = self.treatment_1st_net(train_1st_data_t.treatment)
            treatment_1st_feature_2nd = self.treatment_1st_net(train_2nd_data_t.treatment)
            treatment_proxy_feature_1st = self.treatment_proxy_net(train_1st_data_t.treatment_proxy)
            treatment_proxy_feature_2nd = self.treatment_proxy_net(train_2nd_data_t.treatment_proxy)

            backdoor_1st_feature_1st = None
            backdoor_1st_feature_2nd = None
            if self.backdoor_1st_net is not None:
                backdoor_1st_feature_1st = self.backdoor_1st_net(train_1st_data_t.backdoor)
                backdoor_1st_feature_2nd = self.backdoor_1st_net(train_2nd_data_t.backdoor)

        for i in range(self.stage2_iter):
            self.treatment_2nd_opt.zero_grad()
            self.outcome_proxy_opt.zero_grad()
            if self.backdoor_2nd_net:
                self.backdoor_2nd_opt.zero_grad()

            outcome_proxy_feature_1st = self.outcome_proxy_net(train_1st_data_t.outcome_proxy)
            treatment_2nd_feature_2nd = self.treatment_2nd_net(train_2nd_data_t.treatment)
            backdoor_2nd_feature_2nd = None
            if self.backdoor_2nd_net:
                backdoor_2nd_feature_2nd = self.backdoor_2nd_net(train_2nd_data_t.backdoor)

            res = DFPVModel.fit_2sls(treatment_1st_feature_1st,
                                     treatment_1st_feature_2nd,
                                     treatment_2nd_feature_2nd,
                                     treatment_proxy_feature_1st,
                                     treatment_proxy_feature_2nd,
                                     outcome_proxy_feature_1st,
                                     None,  # Not passing outcome_proxy_feature_2nd for training
                                     backdoor_1st_feature_1st,
                                     backdoor_1st_feature_2nd,
                                     backdoor_2nd_feature_2nd,
                                     train_2nd_data_t.outcome,
                                     self.lam1, self.lam2,
                                     self.add_stage1_intercept,
                                     self.add_stage2_intercept)

            loss = res["stage2_loss"]
            loss.backward()
            if verbose >= 2:
                logger.info(f"stage2 learning: {loss.item()}")

            self.treatment_2nd_opt.step()
            self.outcome_proxy_opt.step()
            if self.backdoor_2nd_net:
                self.backdoor_2nd_opt.step()


def dfpv_experiments(data_config: Dict[str, Any], model_param: Dict[str, Any],
                     one_mdl_dump_dir: Path, random_seed: int = 42, verbose: int = 0):
    dump_dir = one_mdl_dump_dir.joinpath(f"{random_seed}")

    train_data_org = generate_train_data_ate(data_config=data_config, rand_seed=random_seed)
    test_data_org = generate_test_data_ate(data_config=data_config)

    preprocessor = get_preprocessor_ate(data_config.get("preprocess", "Identity"))
    train_data = preprocessor.preprocess_for_train(train_data_org)
    test_data = preprocessor.preprocess_for_test_input(test_data_org)


    torch.manual_seed(random_seed)
    trainer = DFPVTrainer(data_config, model_param, False, dump_dir)
    mdl = trainer.train(train_data, verbose)

    test_data_t = PVTestDataSetTorch.from_numpy(test_data)
    if trainer.gpu_flg:
        torch.cuda.empty_cache()
        test_data_t = test_data_t.to_gpu()

    pred: np.ndarray = mdl.predict_t(test_data_t.treatment).data.cpu().numpy()
    pred = preprocessor.postprocess_for_prediction(pred)
    oos_loss = 0.0
    if test_data.structural is not None:
        oos_loss: float = np.mean((pred - test_data_org.structural) ** 2)
        if data_config["name"] in ["kpv", "deaner"]:
            oos_loss = np.mean(np.abs(pred - test_data_org.structural))
    np.savetxt(one_mdl_dump_dir.joinpath(f"{random_seed}.pred.txt"), pred)
    return oos_loss

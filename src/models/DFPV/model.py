from typing import Optional
import torch
from torch import nn
import numpy as np
import logging

from src.utils.pytorch_linear_reg_utils import fit_linear, linear_reg_pred, outer_prod, add_const_col
from src.data.ate.data_class import PVTrainDataSet, PVTestDataSet, PVTrainDataSetTorch, PVTestDataSetTorch

logger = logging.getLogger()


class DFPVModel:
    stage1_weight: torch.Tensor
    stage2_weight: torch.Tensor
    mean_backdoor_feature: torch.Tensor
    mean_outcome_proxy_feature: torch.Tensor

    def __init__(self,
                 treatment_1st_net: nn.Module,
                 treatment_2nd_net: nn.Module,
                 treatment_proxy_net: nn.Module,
                 outcome_proxy_net: nn.Module,
                 backdoor_1st_net: Optional[nn.Module],
                 backdoor_2nd_net: Optional[nn.Module],
                 add_stage1_intercept: bool,
                 add_stage2_intercept: bool
                 ):
        self.treatment_1st_net = treatment_1st_net
        self.treatment_2nd_net = treatment_2nd_net
        self.treatment_proxy_net = treatment_proxy_net
        self.outcome_proxy_net = outcome_proxy_net
        self.backdoor_1st_net = backdoor_1st_net
        self.backdoor_2nd_net = backdoor_2nd_net
        self.add_stage1_intercept = add_stage1_intercept
        self.add_stage2_intercept = add_stage2_intercept

    @staticmethod
    def augment_stage1_feature(treatment_feature: torch.Tensor,
                               treatment_proxy_feature: torch.Tensor,
                               backdoor_feature: Optional[torch.Tensor],
                               add_stage1_intercept: bool):

        if add_stage1_intercept:
            treatment_feature = add_const_col(treatment_feature)
            treatment_proxy_feature = add_const_col(treatment_proxy_feature)
            if backdoor_feature is not None:
                backdoor_feature = add_const_col(backdoor_feature)

        feature = outer_prod(treatment_feature, treatment_proxy_feature)
        feature = torch.flatten(feature, start_dim=1)
        if backdoor_feature is not None:
            feature = outer_prod(feature, backdoor_feature)
            feature = torch.flatten(feature, start_dim=1)

        return feature

    @staticmethod
    def augment_stage2_feature(predicted_outcome_proxy_feature: torch.Tensor,
                               treatment_feature: torch.Tensor,
                               backdoor_feature: Optional[torch.Tensor],
                               add_stage2_intercept: bool):

        if add_stage2_intercept:
            predicted_outcome_proxy_feature = add_const_col(predicted_outcome_proxy_feature)
            treatment_feature = add_const_col(treatment_feature)
            if backdoor_feature is not None:
                backdoor_feature = add_const_col(backdoor_feature)

        feature = outer_prod(treatment_feature, predicted_outcome_proxy_feature)
        feature = torch.flatten(feature, start_dim=1)
        if backdoor_feature is not None:
            feature = outer_prod(feature, backdoor_feature)
            feature = torch.flatten(feature, start_dim=1)

        return feature

    @staticmethod
    def fit_2sls(treatment_1st_feature_1st: torch.Tensor,
                 treatment_1st_feature_2nd: torch.Tensor,
                 treatment_2nd_feature_2nd: torch.Tensor,
                 treatment_proxy_feature_1st: torch.Tensor,
                 treatment_proxy_feature_2nd: torch.Tensor,
                 outcome_proxy_feature_1st: torch.Tensor,
                 outcome_proxy_feature_2nd: Optional[torch.Tensor],
                 backdoor_1st_feature_1st: Optional[torch.Tensor],
                 backdoor_1st_feature_2nd: Optional[torch.Tensor],
                 backdoor_2nd_feature_2nd: Optional[torch.Tensor],
                 outcome_2nd_t: torch.Tensor,
                 lam1: float, lam2: float,
                 add_stage1_intercept: bool,
                 add_stage2_intercept: bool,
                 ):

        # stage1
        feature = DFPVModel.augment_stage1_feature(treatment_1st_feature_1st,
                                                   treatment_proxy_feature_1st,
                                                   backdoor_1st_feature_1st, add_stage1_intercept)

        stage1_weight = fit_linear(outcome_proxy_feature_1st, feature, lam1)

        # predicting for stage 2
        feature = DFPVModel.augment_stage1_feature(treatment_1st_feature_2nd,
                                                   treatment_proxy_feature_2nd,
                                                   backdoor_1st_feature_2nd,
                                                   add_stage1_intercept)

        predicted_outcome_proxy_feature = linear_reg_pred(feature, stage1_weight)

        # stage2
        feature = DFPVModel.augment_stage2_feature(predicted_outcome_proxy_feature,
                                                   treatment_2nd_feature_2nd,
                                                   backdoor_2nd_feature_2nd,
                                                   add_stage2_intercept)

        stage2_weight = fit_linear(outcome_2nd_t, feature, lam2)
        pred = linear_reg_pred(feature, stage2_weight)
        stage2_loss = torch.norm((outcome_2nd_t - pred)) ** 2 + lam2 * torch.norm(stage2_weight) ** 2

        mean_outcome_proxy_feature = None
        if outcome_proxy_feature_2nd is not None:
            mean_outcome_proxy_feature=torch.mean(outcome_proxy_feature_2nd, dim=0, keepdim=True)

        mean_backdoor_feature = None
        if backdoor_2nd_feature_2nd is not None:
            mean_backdoor_feature = torch.mean(backdoor_2nd_feature_2nd, dim=0, keepdim=True)

        return dict(stage1_weight=stage1_weight,
                    predicted_outcome_proxy_feature=predicted_outcome_proxy_feature,
                    mean_outcome_proxy_feature=mean_outcome_proxy_feature,
                    mean_backdoor_feature=mean_backdoor_feature,
                    stage2_weight=stage2_weight,
                    stage2_loss=stage2_loss)

    def fit_t(self,
              train_1st_data_t: PVTrainDataSetTorch,
              train_2nd_data_t: PVTrainDataSetTorch,
              lam1: float, lam2: float):
        with torch.no_grad():
            treatment_1st_feature_1st = self.treatment_1st_net(train_1st_data_t.treatment)
            treatment_1st_feature_2nd = self.treatment_1st_net(train_2nd_data_t.treatment)
            treatment_2nd_feature_2nd = self.treatment_2nd_net(train_2nd_data_t.treatment)

            treatment_proxy_feature_1st = self.treatment_proxy_net(train_1st_data_t.treatment_proxy)
            treatment_proxy_feature_2nd = self.treatment_proxy_net(train_2nd_data_t.treatment_proxy)

            outcome_proxy_feature_1st = self.outcome_proxy_net(train_1st_data_t.outcome_proxy)
            outcome_proxy_feature_2nd = self.outcome_proxy_net(train_2nd_data_t.outcome_proxy)
            backdoor_1st_feature_1st = None
            backdoor_1st_feature_2nd = None
            backdoor_2nd_feature_2nd = None

            outcome_2nd_t = train_2nd_data_t.outcome
            if self.backdoor_1st_net is not None:
                backdoor_1st_feature_1st = self.backdoor_1st_net(train_1st_data_t.backdoor)
                backdoor_1st_feature_2nd = self.backdoor_1st_net(train_2nd_data_t.backdoor)
                backdoor_2nd_feature_2nd = self.backdoor_2nd_net(train_2nd_data_t.backdoor)

        res = self.fit_2sls(treatment_1st_feature_1st,
                            treatment_1st_feature_2nd,
                            treatment_2nd_feature_2nd,
                            treatment_proxy_feature_1st,
                            treatment_proxy_feature_2nd,
                            outcome_proxy_feature_1st,
                            outcome_proxy_feature_2nd,
                            backdoor_1st_feature_1st,
                            backdoor_1st_feature_2nd,
                            backdoor_2nd_feature_2nd,
                            outcome_2nd_t,
                            lam1, lam2,
                            self.add_stage1_intercept,
                            self.add_stage2_intercept
                            )

        self.stage1_weight = res["stage1_weight"]
        self.stage2_weight = res["stage2_weight"]
        self.mean_outcome_proxy_feature = res["mean_outcome_proxy_feature"]
        self.mean_backdoor_feature = res["mean_backdoor_feature"]

    def fit(self, train_1st_data: PVTrainDataSet, train_2nd_data: PVTrainDataSet, lam1: float, lam2: float):
        train_1st_data_t = PVTrainDataSetTorch.from_numpy(train_1st_data)
        train_2nd_data_t = PVTrainDataSetTorch.from_numpy(train_2nd_data)
        self.fit_t(train_1st_data_t, train_2nd_data_t, lam1, lam2)

    def predict_t(self, treatment: torch.Tensor):
        treatment_feature = self.treatment_2nd_net(treatment)
        n_data = treatment_feature.shape[0]
        mean_outcome_proxy_mat = self.mean_outcome_proxy_feature.expand(n_data, -1)
        mean_backdoor_feature_mat = None
        if self.mean_backdoor_feature is not None:
            mean_backdoor_feature_mat = self.mean_backdoor_feature.expand(n_data, -1)

        feature = DFPVModel.augment_stage2_feature(mean_outcome_proxy_mat,
                                                   treatment_feature,
                                                   mean_backdoor_feature_mat,
                                                   self.add_stage2_intercept)
        return linear_reg_pred(feature, self.stage2_weight)

    def predict(self, treatment: np.ndarray):
        treatment_t = torch.tensor(treatment, dtype=torch.float32)
        return self.predict_t(treatment_t).data.numpy()

    def predict_bridge(self, treatment: np.ndarray, output_proxy: np.ndarray):
        treatment_t = torch.tensor(treatment, dtype=torch.float32)
        output_proxy_t = torch.tensor(output_proxy, dtype=torch.float32)
        return self.predict_bridge_t(treatment_t, output_proxy_t).data.numpy()

    def predict_bridge_t(self, treatment: torch.Tensor, output_proxy: torch.Tensor):
        treatment_feature = self.treatment_2nd_net(treatment)
        output_proxy_feature = self.outcome_proxy_net(output_proxy)
        feature = DFPVModel.augment_stage2_feature(output_proxy_feature,
                                                   treatment_feature,
                                                   None,
                                                   self.add_stage2_intercept)
        return linear_reg_pred(feature, self.stage2_weight)


    def evaluate_t(self, test_data: PVTestDataSetTorch):
        target = test_data.structural
        with torch.no_grad():
            pred = self.predict_t(test_data.treatment)
        return torch.mean((target - pred) ** 2)

    def evaluate(self, test_data: PVTestDataSet):
        return self.evaluate_t(PVTestDataSetTorch.from_numpy(test_data)).data.item()

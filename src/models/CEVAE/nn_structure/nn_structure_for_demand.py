from typing import Optional, Tuple
import torch
from torch import nn
from torch.distributions import Normal

from ..nn_structure.abstract import AbstractDistribution


class DemandDistribution(AbstractDistribution):
    pxz_mean_net: nn.Module
    pxz_sigma_net: nn.Module
    ptz_mean_net: nn.Module
    ptz_sigma_net: nn.Module
    pytz_mean_net: nn.Module
    qzxty_mean_net: nn.Module
    qzxty_sigma_net: nn.Module

    def __init__(self, n_hidden_dim: int = 20, n_learning_sample: int = 10):
        super(DemandDistribution, self).__init__(n_hidden_dim, n_learning_sample)

    def build_nets(self):
        self.build_p_x_z()
        self.build_q_z_xty()
        self.build_p_t_z()
        self.build_p_y_tz()
        self.build_q_z_xty()

    @staticmethod
    def p_z(z):
        return Normal(loc=torch.zeros(z.size()),
                      scale=torch.ones(z.size()))

    def p_x_z(self, z):
        return Normal(loc=self.pxz_mean_net(z),
                      scale=self.ptz_sigma_net(z))

    def p_t_z(self, z):
        return Normal(loc=self.ptz_mean_net(z),
                      scale=self.ptz_sigma_net(z))

    def p_y_zt(self, z, t):
        tz = torch.cat([t, z], dim=1)
        p_y_tz = Normal(loc=self.pytz_mean_net(tz), scale=1)
        return p_y_tz

    def q_z_xty(self, x, t, y):
        xty = torch.cat([x, t, y], dim=1)
        try:
            q_z_xty = Normal(loc=self.qzxty_mean_net(xty),
                             scale=torch.clamp(self.qzxty_sigma_net(xty), min=0.01))
        except:
            print(self.qzxty_mean_net(xty))
            print(self.qzxty_sigma_net(xty))
            raise ValueError
        return q_z_xty

    def build_p_x_z(self):
        feature = nn.Sequential(nn.Linear(self.n_hidden_dim, 64),
                                nn.ReLU(),
                                nn.Linear(64, 32),
                                nn.ReLU(),
                                nn.Linear(32, 16),
                                nn.ReLU())
        self.pxz_mean_net = nn.Sequential(feature, nn.Linear(16, 3))
        self.pxz_sigma_net = nn.Sequential(feature, nn.Linear(16, 3), nn.Softplus())

    def build_p_t_z(self):
        feature = nn.Sequential(nn.Linear(self.n_hidden_dim, 64),
                                nn.ReLU(),
                                nn.Linear(64, 32),
                                nn.ReLU(),
                                nn.Linear(32, 16),
                                nn.ReLU())
        self.ptz_mean_net = nn.Sequential(feature, nn.Linear(16, 1))
        self.ptz_sigma_net = nn.Sequential(feature, nn.Linear(16, 1), nn.Softplus())

    def build_p_y_tz(self):
        feature = nn.Sequential(nn.Linear(self.n_hidden_dim + 1, 64),
                                nn.ReLU(),
                                nn.Linear(64, 32),
                                nn.ReLU(),
                                nn.Linear(32, 16),
                                nn.ReLU())
        self.pytz_mean_net = nn.Sequential(feature, nn.Linear(16, 1))
        ## the sigma for p(y|t,z) is set to one

    def build_q_z_xty(self):
        feature = nn.Sequential(nn.Linear(5, 128),
                                nn.ReLU(),
                                nn.Linear(128, 64),
                                nn.ReLU(),
                                nn.Linear(64, 32),
                                nn.ReLU())
        self.qzxty_mean_net = nn.Sequential(feature, nn.Linear(32, self.n_hidden_dim))
        self.qzxty_sigma_net = nn.Sequential(feature, nn.Linear(32, self.n_hidden_dim), nn.Softplus())

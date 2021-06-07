from typing import Optional, Tuple
import torch
from torch import nn
from torch.distributions import Normal, Bernoulli

from ..nn_structure.abstract import AbstractDistribution


class CEVAEDistribution(AbstractDistribution):
    pxz_mean_net: nn.Module
    pxz_sigma_net: nn.Module
    ptz_logit_net: nn.Module
    pytz_logit_net: nn.Module
    qzxty_logit_net: nn.Module

    def __init__(self, n_hidden_dim: int = 20, n_learning_sample: int = 10):
        super(CEVAEDistribution, self).__init__(n_hidden_dim, n_learning_sample)

    def build_nets(self):
        self.build_p_x_z()
        self.build_q_z_xty()
        self.build_p_t_z()
        self.build_p_y_tz()
        self.build_q_z_xty()

    @staticmethod
    def p_z(z):
        return Bernoulli(probs=0.5)

    def p_x_z(self, z):
        return Normal(loc=self.pxz_mean_net(z),
                      scale=self.pxz_sigma_net(z))

    def p_t_z(self, z):
        return Bernoulli(logits=self.ptz_logit_net(z))

    def p_y_zt(self, z, t):
        logits = self.pytz_logit_net(z)
        logits = t * logits[:, [0]] + (1 - t) * logits[:, [1]]
        p_y_zt = Bernoulli(logits=logits)
        return p_y_zt

    def q_z_xty(self, x, t, y):
        xy = torch.cat([x, y], dim=1)
        logits = self.qzxty_logit_net(xy)
        logits = t * logits[:, :self.n_hidden_dim] + (1 - t) * logits[:, self.n_hidden_dim:]
        return Bernoulli(logits=logits)

    def build_p_x_z(self):
        feature = nn.Sequential(nn.Linear(self.n_hidden_dim, 5),
                                nn.ReLU())

        self.pxz_mean_net = nn.Sequential(feature, nn.Linear(5, 1))
        self.pxz_sigma_net = nn.Sequential(feature, nn.Linear(5, 1), nn.Softplus())

    def build_p_t_z(self):
        self.ptz_logit_net = nn.Sequential(nn.Linear(self.n_hidden_dim, 5),
                                           nn.ReLU(),
                                           nn.Linear(5, 1))

    def build_p_y_tz(self):
        self.pytz_logit_net = nn.Sequential(nn.Linear(self.n_hidden_dim, 5),
                                            nn.ReLU(),
                                            nn.Linear(5, 2))

    def build_q_z_xty(self):
        feature = nn.Sequential(nn.Linear(3, 16),
                                nn.ReLU())
        self.qzxty_logit_net = nn.Sequential(feature,
                                             nn.Linear(16, self.n_hidden_dim * 2))

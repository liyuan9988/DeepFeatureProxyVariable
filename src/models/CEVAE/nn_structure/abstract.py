from torch import nn
import torch


class AbstractDistribution(nn.Module):

    def __init__(self, n_hidden_dim: int = 20, n_learning_sample: int = 10):
        super(AbstractDistribution, self).__init__()
        self.n_hidden_dim = n_hidden_dim
        self.n_learning_sample = n_learning_sample
        self.build_nets()


    def forward(self, proxy, treatment, outcome):
        q_z_xty = self.q_z_xty(proxy, treatment, outcome)
        loss = 0.0

        for i in range(self.n_learning_sample):
            posterior_sample = q_z_xty.sample()
            p_x_z = self.p_x_z(posterior_sample)
            loss += torch.sum(p_x_z.log_prob(proxy))

            p_t_z = self.p_t_z(posterior_sample)
            loss += torch.sum(p_t_z.log_prob(treatment))

            p_y_zt = self.p_y_zt(posterior_sample, treatment)
            loss += torch.sum(p_y_zt.log_prob(outcome))

            p_z = self.p_z(posterior_sample)
            loss += torch.sum(p_z.log_prob(posterior_sample))

            loss -= torch.sum(q_z_xty.log_prob(posterior_sample))

        loss /= self.n_learning_sample
        return -loss

    @staticmethod
    def p_z(z):
        raise NotImplementedError

    def p_x_z(self, z):
        raise NotImplementedError

    def p_t_z(self, z):
        raise NotImplementedError

    def p_y_zt(self, z, t):
        raise NotImplementedError

    def q_z_xty(self, x, t, y):
        raise NotImplementedError

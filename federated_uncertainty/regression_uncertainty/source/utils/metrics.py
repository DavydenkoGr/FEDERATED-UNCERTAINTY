import torch
import torch.nn as nn
from torch.distributions.normal import Normal


class MSE(nn.Module):
    def forward(self, x, y):
        return torch.mean((x - y) ** 2)


def mse(means, y_test):
    return (means - y_test) ** 2

def nll(means, variances, y_test):
    return 0.5 * torch.log(2 * torch.pi * variances) + (means - y_test) ** 2 / (2 * variances)

def crps(means, variances, y_test):
    stds = torch.sqrt(variances)
    norm = Normal(0, 1)
    z = (y_test - means) / stds
    z_cdf = norm.cdf(z)
    z_pdf = torch.exp(norm.log_prob(z))
    return stds * (z * (2 * z_cdf - 1) + 2 * z_pdf - 1 / torch.sqrt(torch.tensor(torch.pi)))
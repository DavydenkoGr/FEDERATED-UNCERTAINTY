import torch
from torch.nn.functional import softplus


def variance_link(x: torch.Tensor):
    assert len(x.shape) >= 2, "Input tensor must have shape (..., 2)"
    assert x.shape[-1] == 2, "Input tensor must have shape (..., 2)"

    mean = x[..., 0]
    var = softplus(x[..., 1]) + 1e-8  # for stability

    return mean, var


def natural_link(x: torch.Tensor):
    eta_1, eta_2 = variance_link(x)

    return eta_1, -eta_2


def gauss_to_natural(mean: torch.Tensor, var: torch.Tensor):
    eta_1 = mean / var
    eta_2 = -0.5 / var

    return eta_1, eta_2


def natural_to_gauss(eta_1: torch.Tensor, eta_2: torch.Tensor):
    mean = eta_1 / (-2 * eta_2)
    var = -0.5 / eta_2

    return mean, var

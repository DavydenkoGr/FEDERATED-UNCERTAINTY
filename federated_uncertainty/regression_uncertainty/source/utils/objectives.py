import math
import torch
import torch.nn as nn

C = -0.5 * torch.log(2 * torch.tensor(math.pi))


class GaussianNLL(nn.Module):
    def forward(self, mean, var, y):
        return torch.mean(0.5 * torch.log(var) + (y - mean) ** 2 / (2 * var)) + C


class NaturalGaussianNLL(nn.Module):
    def forward(self, eta_1, eta_2, y):
        assert (eta_2 < 0).all(), "negative inverse variance (eta_2) must be negative"
        quadratic = eta_1 * y + eta_2 * y**2
        return (
            -torch.mean(
                quadratic + eta_1**2 / (4 * eta_2) + 0.5 * torch.log(-2 * eta_2)
            )
            + C
        )


def compute_anti_regularization(model) -> torch.Tensor:
    """
    Computes the anti-regularization term: (1/d) * sum_k^d log(w_k^2)
    where d is the total number of learnable parameters.
    """
    log_w_sq_sum = 0.0
    d = 0
    for param in model.parameters():
        if param.requires_grad:
            log_w_sq_sum += torch.sum(torch.log(param**2 + 1e-12))
            d += param.numel()
    if d == 0:
        return 0.0
    return log_w_sq_sum / d

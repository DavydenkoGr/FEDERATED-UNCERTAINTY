import torch


def compute_anti_regularization(model):
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

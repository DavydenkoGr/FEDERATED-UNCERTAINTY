import torch


def _calc_surrogate_variance(
    means: torch.Tensor, variances: torch.Tensor
) -> torch.Tensor:
    """
    Calculate the variance of a single Gaussian distribution given its mean and variance.

    Args:
        means (torch.Tensor): Mean of the Gaussian distribution.
        vars (torch.Tensor): Variance of the Gaussian distribution.

    Returns:
        torch.Tensor: Variance of the single Gaussian distribution.
    """
    return (variances + means**2).mean(dim=-1) - means.mean(dim=-1) ** 2


def _calculate_A_function(means: torch.Tensor, stds: torch.Tensor) -> torch.Tensor:
    """
    Calculate the A function for a set of predictions.

    Args:
        means (torch.Tensor): Predicted means for N predictors.
        vars (torch.Tensor): Predicted standard deviations for N predictors.

    Returns:
        torch.Tensor: A function values.
    """

    def _std_gaussian_cdf(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))

    def _std_gaussian_pdf(x: torch.Tensor) -> torch.Tensor:
        return (1 / torch.sqrt(torch.tensor(2.0) * torch.pi)) * torch.exp(-0.5 * x**2)

    return 2 * stds * _std_gaussian_pdf(means / stds) + means * (
        2 * _std_gaussian_cdf(means / stds) - 1
    )


def calculate_uncertainties_crps(
    means: torch.Tensor, variances: torch.Tensor,
    means_exc: torch.Tensor | None = None, variances_exc: torch.Tensor | None = None,
    eps: float = 1e-9
) -> dict[str, torch.Tensor]:
    """
    Calculate the Continuous Ranked Probability Score (CRPS) for a set of predictions.

    Args:
        means [..., n_preds]: Predicted means for N predictors.
        vars [..., n_preds]: Predicted standard deviations for N predictors.

    Returns:
        dict:
    """

    N = means.shape[-1]
    if variances.shape[-1] != N:
        raise ValueError(
            f"Expected vars to have the same last dimension as means, got {variances.shape[-1]} vs {N}"
        )
    
    if means_exc is None or variances_exc is None:
        means_exc = means
        variances_exc = variances

    sqrtpi = torch.sqrt(torch.tensor(torch.pi))

    bayes_1 = torch.sqrt(variances).mean(dim=-1) / sqrtpi

    means_diffs = means.unsqueeze(-1) - means.unsqueeze(-2)
    stds_diffs = torch.sqrt(variances.unsqueeze(-1) + variances.unsqueeze(-2))
    A_pairwise = _calculate_A_function(means_diffs, stds_diffs)
    bayes_2 = 0.5 * A_pairwise.mean(dim=(-1, -2))

    means_diffs = means.unsqueeze(-1) - means_exc.unsqueeze(-2)
    stds_diffs = torch.sqrt(variances.unsqueeze(-1) + variances_exc.unsqueeze(-2))
    A_pairwise = _calculate_A_function(means_diffs, stds_diffs)
    #pairwise_stds = torch.sqrt(variances.unsqueeze(-1)) + torch.sqrt(variances_exc.unsqueeze(-2))
    excess_1_1 = A_pairwise.mean(dim=(-1, -2)) - 2 / sqrtpi * torch.sqrt(variances).mean(dim=-1)

    excess_2_1 = 0.5 * excess_1_1


    # 3a calculations
    surrogate_vars = _calc_surrogate_variance(means, variances)

    bayes_3a = torch.sqrt(surrogate_vars / torch.pi)

    means_surrogate_diffs = means.mean(dim=-1, keepdim=True) - means_exc
    stds_surrogate_diffs = torch.sqrt(surrogate_vars.unsqueeze(-1) + variances_exc)
    excess_3a_1 = (
        _calculate_A_function(means_surrogate_diffs, stds_surrogate_diffs).mean(dim=-1)
        - bayes_3a
        - torch.sqrt(variances_exc).mean(dim=-1) / sqrtpi
    )

    means_diffs = means_exc.unsqueeze(-1) - means_exc.unsqueeze(-2)
    stds_diffs = torch.sqrt(variances_exc.unsqueeze(-1) + variances_exc.unsqueeze(-2))
    A_pairwise = _calculate_A_function(means_diffs, stds_diffs)
    means_surrogate_diffs = means.mean(dim=-1, keepdim=True) - means_exc
    excess_3a_2 = (
        _calculate_A_function(means_surrogate_diffs, stds_surrogate_diffs).mean(dim=-1)
        - bayes_3a
        - 0.5 * A_pairwise.mean(dim=(-1, -2))
    )

    #### 3b calculations
    surrogate_vars = variances.mean(dim=-1)

    bayes_3b = torch.sqrt(surrogate_vars / torch.pi)

    means_surrogate_diffs = means.mean(dim=-1, keepdim=True) - means_exc
    stds_surrogate_diffs = torch.sqrt(surrogate_vars.unsqueeze(-1) + variances_exc)
    excess_3b_1 = (
        _calculate_A_function(means_surrogate_diffs, stds_surrogate_diffs).mean(dim=-1)
        - bayes_3b
        - torch.sqrt(variances_exc).mean(dim=-1) / sqrtpi
    )

    means_diffs = means_exc.unsqueeze(-1) - means_exc.unsqueeze(-2)
    stds_diffs = torch.sqrt(variances_exc.unsqueeze(-1) + variances_exc.unsqueeze(-2))
    A_pairwise = _calculate_A_function(means_diffs, stds_diffs)
    means_surrogate_diffs = means.mean(dim=-1, keepdim=True) - means_exc
    excess_3b_2 = (
        _calculate_A_function(means_surrogate_diffs, stds_surrogate_diffs).mean(dim=-1)
        - bayes_3b
        - 0.5 * A_pairwise.mean(dim=(-1, -2))
    )

    return {
        "total_1_1": bayes_1 + excess_1_1,
        "total_2_1": bayes_2 + excess_2_1,
        "total_3a_1": bayes_3a + excess_3a_1,
        "total_3b_1": bayes_3b + excess_3b_1,
        "total_3a_2": bayes_3a + excess_3a_2,
        "total_3b_2": bayes_3b + excess_3b_2,
        "bayes_1": bayes_1,
        "bayes_2": bayes_2,
        "bayes_3a": bayes_3a,
        "bayes_3b": bayes_3b,
        "excess_1_1": excess_1_1,
        "excess_2_1": excess_2_1,
        "excess_3a_1": excess_3a_1,
        "excess_3b_1": excess_3b_1,
        "excess_3a_2": excess_3a_2,
        "excess_3b_2": excess_3b_2,
    }


def calculate_uncertainties_log(
    means: torch.Tensor, variances: torch.Tensor, 
    means_exc: torch.Tensor | None = None, variances_exc: torch.Tensor | None = None,
    eps: float = 1e-9
) -> dict[str, torch.Tensor]:
    """
    Calculate the Mean Squared Error (MSE) for a set of predictions.

    Args:
        means [..., n_preds]: Predicted means for N predictors.
        vars [..., n_preds]: Predicted standard deviations for N predictors.

    Returns:
        dict:
    """
    if variances.shape[-1] != means.shape[-1]:
        raise ValueError(
            f"Expected vars to have the same last dimension as means, got {variances.shape[-1]} vs {means.shape[-1]}"
        )

    if means_exc is None or variances_exc is None:
        means_exc = means
        variances_exc = variances

    pi2e = 2.0 * torch.pi * torch.exp(torch.tensor(1.0))

    # Bayes risks
    bayes_1 = 0.5 * torch.log(variances * pi2e).mean(dim=-1)
    bayes_2 = torch.tensor([float("nan")])  # no closed form solution
    bayes_3a = 0.5 * torch.log(_calc_surrogate_variance(means, variances) * pi2e)
    bayes_3b = 0.5 * torch.log(variances.mean(dim=-1) * pi2e)

    # Excess risks
    excess_1_1 = 0.5 * ((variances_exc.unsqueeze(-2) + (means.unsqueeze(-1) - means_exc.unsqueeze(-2)) ** 2) \
        / variances.unsqueeze(-1) - 1).mean(dim=(-1, -2))
    excess_2_1 = excess_1_1 #torch.tensor([float("nan")])  # no closed form solution
    excess_3a_1 = 0.5 * (
        torch.log(_calc_surrogate_variance(means, variances))
        - torch.log(variances_exc).mean(dim=-1)
    )
    excess_3b_1 = 0.5 * (
        torch.log(variances.mean(dim=-1)) - torch.log(variances_exc).mean(dim=-1) \
        + ((means.mean(dim=-1, keepdim=True) - means_exc) ** 2).mean(dim=-1) / variances.mean(dim=-1)
    )
    excess_3a_2 = torch.tensor([float("nan")])
    excess_3b_2 = torch.tensor([float("nan")])

    return {
        "total_1_1": bayes_1 + excess_1_1,
        "total_2_1": bayes_1 + excess_1_1, # due to linearity
        "total_3a_1": bayes_3a + excess_3a_1,
        "total_3b_1": bayes_3b + excess_3b_1,
        "total_3a_2": torch.tensor([float("nan")]),
        "total_3b_2": torch.tensor([float("nan")]),
        "bayes_1": bayes_1,
        "bayes_2": bayes_2,
        "bayes_3a": bayes_3a,
        "bayes_3b": bayes_3b,
        "excess_1_1": excess_1_1,
        "excess_2_1": excess_2_1,
        "excess_3a_1": excess_3a_1,
        "excess_3b_1": excess_3b_1,
        "excess_3a_2": excess_3a_2,
        "excess_3b_2": excess_3b_2,
    }


def calculate_uncertainties_mse(
    means: torch.Tensor, variances: torch.Tensor, 
    means_exc: torch.Tensor | None = None, variances_exc: torch.Tensor | None = None,
    eps: float = 1e-9
) -> dict[str, torch.Tensor]:
    """
    Calculate the Mean Squared Error (MSE) for a set of predictions.

    Args:
        means [..., n_preds]: Predicted means for N predictors.
        vars [..., n_preds]: Predicted standard deviations for N predictors.

    Returns:
        dict:
    """
    if variances.shape[-1] != means.shape[-1]:
        raise ValueError(
            f"Expected vars to have the same last dimension as means, got {variances.shape[-1]} vs {means.shape[-1]}"
        )

    if means_exc is None or variances_exc is None:
        means_exc = means
        variances_exc = variances

    bayes_1 = variances.mean(dim=-1)
    bayes_2 = _calc_surrogate_variance(means, variances)
    bayes_3a = bayes_2
    bayes_3b = bayes_1

    excess_1_1 = ((means.unsqueeze(-1) - means_exc.unsqueeze(-2)) ** 2).mean(dim=(-1, -2))
    excess_2_1 = ((means.mean(dim=-1, keepdim=True) - means_exc) ** 2).mean(dim=-1)
    excess_3a_1 = excess_2_1
    excess_3b_1 = excess_2_1
    excess_3a_2 = (means.mean(dim=-1) - means_exc.mean(dim=-1)) ** 2
    excess_3b_2 = excess_3a_2

    return {
        "total_1_1": bayes_1 + excess_1_1,
        "total_2_1": bayes_2 + excess_2_1,
        "total_3a_1": bayes_3a + excess_3a_1,
        "total_3b_1": bayes_3b + excess_3b_1,
        "total_3a_2": bayes_3a + excess_3a_2,
        "total_3b_2": bayes_3b + excess_3b_2,
        "bayes_1": bayes_1,
        "bayes_2": bayes_2,
        "bayes_3a": bayes_3a,
        "bayes_3b": bayes_3b,
        "excess_1_1": excess_1_1,
        "excess_2_1": excess_2_1,
        "excess_3a_1": excess_3a_1,
        "excess_3b_1": excess_3b_1,
        "excess_3a_2": excess_3a_2,
        "excess_3b_2": excess_3b_2,
    }

def _gaussian_pdf(x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """
    Calculate the probability density function of a Gaussian distribution.
    
    Args:
        x (torch.Tensor): Points at which to evaluate the PDF.
        mean (torch.Tensor): Mean of the Gaussian distribution.
        var (torch.Tensor): Variance of the Gaussian distribution.
    Returns:
        torch.Tensor: PDF values at the specified points.
    """
    return (1 / torch.sqrt(2 * torch.pi * var)) * torch.exp(-0.5 * ((x - mean) ** 2) / var)

def _quadratic_score(
    means: torch.Tensor, variances: torch.Tensor, 
    means_exc: torch.Tensor, variances_exc: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Calculate the quadratic score for a set of predictions.

    Args:
        means (torch.Tensor): Predicted means for N predictors.
        variances (torch.Tensor): Predicted variances for N predictors.
        means_exc (torch.Tensor, optional): Means of the excess predictions.
        variances_exc (torch.Tensor, optional): Variances of the excess predictions.

    Returns:
        torch.Tensor: Quadratic score values.
    """

    first_term = - 2 * _gaussian_pdf(means, means_exc, variances + variances_exc)
    second_term = 1 / (torch.sqrt(variances) + eps) / 2 / torch.sqrt(torch.tensor(torch.pi))

    return first_term + second_term

def calculate_uncertainties_quadratic(
    means: torch.Tensor, variances: torch.Tensor, 
    means_exc: torch.Tensor | None = None, variances_exc: torch.Tensor | None = None,
    eps: float = 1e-9
) -> dict[str, torch.Tensor]:
    """
    Calculate the Mean Squared Error (MSE) for a set of predictions.

    Args:
        means [..., n_preds]: Predicted means for N predictors.
        vars [..., n_preds]: Predicted standard deviations for N predictors.

    Returns:
        dict:
    """

    if variances.shape[-1] != means.shape[-1]:
        raise ValueError(
            f"Expected vars to have the same last dimension as means, got {variances.shape[-1]} vs {means.shape[-1]}"
        )
    M = means.shape[-1]
    
    if means_exc is None or variances_exc is None:
        means_exc = means
        variances_exc = variances
    
    surrogate_vars = _calc_surrogate_variance(means, variances)
    twosqrtpi = 2.0 * torch.sqrt(torch.tensor(torch.pi))

    bayes_1 = - (1 / (torch.sqrt(variances) + eps)).mean(dim=-1) / twosqrtpi
    #print(f"{bayes_1=}")
    score_matrix = _quadratic_score(means.unsqueeze(-1), variances.unsqueeze(-1),
                                    means_exc.unsqueeze(-2), variances_exc.unsqueeze(-2), eps)
    # zero out the diagonal
    score_matrix = score_matrix.masked_fill(torch.eye(means.shape[-1]).bool(), 0.0)
    bayes_2 = (0.5 / M + 0.5 ) * bayes_1 + 0.5 * score_matrix.mean(dim=(-1, -2))
    bayes_3a = - 1 / (torch.sqrt(surrogate_vars) + eps) / twosqrtpi
    bayes_3b = - 1 / (torch.sqrt(variances.mean(dim=-1)) + eps) / twosqrtpi

    # Excess risks
    bayes_1_exc = - (1 / (torch.sqrt(variances_exc) + eps)).mean(dim=-1) / twosqrtpi
    bayes_2_exc = (0.5 / M + 0.5 ) * bayes_1_exc + 0.5 * score_matrix.mean(dim=(-1, -2))
    #print(f"{bayes_1_exc=}")
    excess_1_1 = (- bayes_1 - bayes_1_exc) \
        - 2 * _gaussian_pdf(means.unsqueeze(-1), means_exc.unsqueeze(-2), 
                            variances.unsqueeze(-1) + variances_exc.unsqueeze(-2)).mean(dim=(-1, -2))

    excess_2_1 = - bayes_2 - bayes_1_exc \
        - 2 * _gaussian_pdf(means.unsqueeze(-1), means_exc.unsqueeze(-2), 
                            variances.unsqueeze(-1) + variances_exc.unsqueeze(-2)).mean(dim=(-1, -2))
    
    excess_3a_1 = - bayes_3a - bayes_1_exc \
        - 2 * _gaussian_pdf(means.mean(dim=(-1), keepdim=True), means_exc, 
                            surrogate_vars.unsqueeze(-1) + variances_exc).mean(dim=-1)

    excess_3b_1 = - bayes_3b - bayes_1_exc \
        - 2 * _gaussian_pdf(means.mean(dim=(-1), keepdim=True), means_exc, 
                            variances.mean(dim=-1, keepdim=True) + variances_exc).mean(dim=-1)

    excess_3a_2 = - bayes_3a - bayes_2_exc \
        - 2 * _gaussian_pdf(means.mean(dim=(-1), keepdim=True), means_exc, 
                            surrogate_vars.unsqueeze(-1) + variances_exc).mean(dim=-1)
    
    excess_3b_2 = - bayes_3b - bayes_2_exc \
        - 2 * _gaussian_pdf(means.mean(dim=(-1), keepdim=True), means_exc, 
                            variances.mean(dim=-1, keepdim=True) + variances_exc).mean(dim=-1)

    return {
        "total_1_1": bayes_1 + excess_1_1,
        "total_2_1": bayes_2 + excess_2_1,
        "total_3a_1": bayes_3a + excess_3a_1,
        "total_3b_1": bayes_3b + excess_3b_1,
        "total_3a_2": bayes_3a + excess_3a_2,
        "total_3b_2": bayes_3b + excess_3b_2,
        "bayes_1": bayes_1,
        "bayes_2": bayes_2,
        "bayes_3a": bayes_3a,
        "bayes_3b": bayes_3b,
        "excess_1_1": excess_1_1,
        "excess_2_1": excess_2_1,
        "excess_3a_1": excess_3a_1,
        "excess_3b_1": excess_3b_1,
        "excess_3a_2": excess_3a_2,
        "excess_3b_2": excess_3b_2,
    }

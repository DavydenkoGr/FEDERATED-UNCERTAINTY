from typing import Optional
import numpy as np
from .constants import ApproximationType, GName, RiskType
from .getters import (
    get_probability_approximation,
    get_specific_risk,
)
from .utils import safe_softmax


def energy(logits: np.ndarray, T: float) -> np.ndarray:
    return -T * logsumexp(logits / T, axis=-1)


def get_energy_inner(logits: np.ndarray, T: float) -> np.ndarray:
    return np.squeeze(energy(np.mean(logits, keepdims=True, axis=0), T=T))


def get_energy_outer(logits: np.ndarray, T: float) -> np.ndarray:
    return np.squeeze(np.mean(energy(logits, T=T), axis=0, keepdims=True))


def get_risk_approximation(
    g_name: GName,
    risk_type: RiskType,
    logits: np.ndarray,
    gt_approx: ApproximationType,
    T: float = 1.0,
    probabilities: Optional[np.ndarray] = None,
    pred_approx: Optional[ApproximationType] = None,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    if probabilities is None:
        probabilities = safe_softmax(logits)

    if weights is not None:
        weights = weights / np.sum(weights)

    risk = get_specific_risk(g_name=g_name, risk_type=risk_type)
    prob_gt = get_probability_approximation(
        g_name=g_name, approximation=gt_approx, logits=logits, T=T, weights=None
    )
    
    if risk_type.value == RiskType.BAYES_RISK.value:
        risk_vals = risk(prob_gt=prob_gt)
        if weights is not None:
            result = np.sum(risk_vals * weights[:, None, None], axis=0)
        else:
            result = np.mean(risk_vals, axis=0)
            
    else:
        prob_pred = get_probability_approximation(
            g_name=g_name, approximation=pred_approx, logits=logits, T=T, weights=weights
        )
        risk_vals = risk(prob_gt=prob_gt, prob_pred=prob_pred)
        
        if weights is not None:
            result = np.sum(risk_vals * weights[:, None, None], axis=0)
        else:
            result = np.mean(risk_vals, axis=0)
        result = np.mean(result, axis=0)

    return np.squeeze(result)


def check_scalar_product(
    g_name: GName,
    logits: np.ndarray,
    T: float = 1.0,
    probabilities: Optional[np.ndarray] = None,
) -> np.ndarray:
    if probabilities is None:
        probabilities = safe_softmax(logits)
    _, g_grad_func = get_g_functions(g_name=g_name)
    bma_probs = posterior_predictive(logits, T=T)
    central_probs = get_central_prediction(g_name=g_name)(logits=logits, T=T)

    if g_name.value == GName.ZERO_ONE_SCORE.value:
        probabilities = probabilities[None]
        central_probs = central_probs[None]
    grad_pred = g_grad_func(probabilities)
    grad_central = g_grad_func(central_probs)
    if g_name.value == GName.ZERO_ONE_SCORE.value:
        probabilities = probabilities[0]
        central_probs = central_probs[0]

    vec_1 = np.mean(grad_pred, axis=0, keepdims=True) - grad_central
    vec_2 = central_probs - bma_probs
    res = np.sum(vec_1 * vec_2, axis=-1)
    return res

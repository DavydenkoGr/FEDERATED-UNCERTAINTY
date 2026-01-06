import numpy as np
from .utils import posterior_predictive, safe_softmax

def log_score_central_prediction(logits: np.ndarray, T: float = 1.0, weights: np.ndarray = None) -> np.ndarray:
    if weights is not None:
        w = weights[:, None, None] / np.sum(weights)
        mean_logit = np.sum(logits * w, axis=0, keepdims=True) / T
    else:
        mean_logit = np.mean(logits, axis=0, keepdims=True) / T
        
    central_pred = safe_softmax(mean_logit)
    return central_pred

def brier_score_central_prediction(logits: np.ndarray, T: float = 1.0, weights: np.ndarray = None) -> np.ndarray:
    return posterior_predictive(logits, T, weights)

def zero_one_central_prediction(logits: np.ndarray, T: float = 1.0, weights: np.ndarray = None) -> np.ndarray:
    is_max = (logits == logits.max(axis=-1, keepdims=True)).astype(float)
    
    if weights is not None:
        w = weights[:, None, None] / np.sum(weights)
        tilde_p = np.sum(is_max * w, axis=0, keepdims=True)
    else:
        tilde_p = np.mean(is_max, axis=0, keepdims=True)
        
    central_pred = (tilde_p != 0.0) * tilde_p
    central_pred = central_pred / (np.sum(central_pred, axis=-1, keepdims=True) + 1e-12)
    
    return central_pred

def spherical_score_central_prediction(logits: np.ndarray, T: float = 1.0, weights: np.ndarray = None):
    probs = safe_softmax(logits / T)
    K = logits.shape[-1]

    norms = np.linalg.norm(probs, axis=-1, keepdims=True, ord=2)
    normalized_probs = probs / norms

    if weights is not None:
        w = weights[:, None, None] / np.sum(weights)
        x = np.sum(normalized_probs * w, axis=0, keepdims=True)
    else:
        x = np.mean(normalized_probs, axis=0, keepdims=True)

    x0 = np.ones(K).reshape(1, 1, K) / K
    x0_norm = np.linalg.norm(x0, ord=2, keepdims=True, axis=-1)

    y_orthogonal = x - np.sum(x * x0, axis=-1, keepdims=True) * (x0 / x0_norm**2)
    y_orthogonal_norm = np.linalg.norm(y_orthogonal, ord=2, keepdims=True, axis=-1)
    safe_denom = np.sqrt(np.maximum(1 - y_orthogonal_norm**2, 1e-12))
    
    central_pred = x0 + (y_orthogonal / safe_denom) * x0_norm
    return central_pred

import numpy as np
from typing import Optional


_DEF_ATOL = 1e-4


class ShapeError(ValueError):
    pass


def check_probs(
    probs: np.ndarray, *, name: str = "probs", atol: float = _DEF_ATOL
) -> None:
    if probs.ndim != 2:
        raise ShapeError(
            f"{name} must have shape (n_samples, n_classes), got {probs.shape}."
        )
    if np.any(probs < -atol) or np.any(probs > 1 + atol):
        raise ValueError(f"{name} values must lie in [0,1].")
    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-4):
        raise ValueError(
            f"Each row of {name} must sum to 1 (±1e-4). Got min sum value: {row_sums.min()}"
        )


def check_labels(
    y_true: np.ndarray, *, name: str = "y_true", n_classes: Optional[int] = None
) -> None:
    if y_true.ndim != 1:
        raise ShapeError(f"{name} must have shape (n_samples,), got {y_true.shape}.")
    if n_classes is not None:
        if (y_true < 0).any() or (y_true >= n_classes).any():
            raise ValueError(
                f"{name} must be integer class indices in [0, {n_classes - 1}]."
            )

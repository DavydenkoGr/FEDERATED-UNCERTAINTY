from typing import Optional
import numpy as np
from .constants import UncertaintyType
from .general_metrics.mahalanobis import MahalanobisDistance
from .risk_metrics.create_specific_risks import get_risk_approximation


class UncertaintyWrapper:
    """
    General wrapper for uncertainty estimation, providing a scikit-learn-like interface
    with fit and predict methods. Handles both trainable and non-trainable uncertainty measures.
    """

    def __init__(self, uncertainty_type: UncertaintyType, **kwargs):
        """
        Args:
            uncertainty_type: Type of uncertainty measure (e.g., MAHALANOBIS, RISK)
            **kwargs: Additional arguments required for specific uncertainty measures
        """
        self.uncertainty_type = uncertainty_type
        self.kwargs = kwargs
        self.model = None
        self.is_fitted = False

    def fit(self, train_logits: Optional[np.ndarray] = None):
        """
        Fit the uncertainty estimator if required.

        Args:
            train_logits: Training logits, required for trainable uncertainty measures
        Returns:
            self
        """
        if self.uncertainty_type == UncertaintyType.MAHALANOBIS:
            if train_logits is None:
                raise ValueError(
                    "train_logits must be provided for Mahalanobis uncertainty."
                )
            self.model = MahalanobisDistance().fit(train_logits)
            self.is_fitted = True
        elif self.uncertainty_type == UncertaintyType.RISK:
            # Risk-based uncertainty does not require fitting
            self.is_fitted = True
        else:
            # For future uncertainty types, add logic here
            raise ValueError(
                f"Unknown or unsupported uncertainty type: {self.uncertainty_type}"
            )
        return self

    def predict(self, logits: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute uncertainty scores for the given logits.

        Args:
            logits: np.ndarray, shape depends on the uncertainty measure
            **kwargs: Additional arguments for specific uncertainty measures
        Returns:
            np.ndarray: Uncertainty scores
        """
        if not self.is_fitted:
            raise RuntimeError(
                "UncertaintyWrapper must be fitted before calling predict."
            )

        if self.uncertainty_type == UncertaintyType.MAHALANOBIS:
            if self.model is None:
                raise RuntimeError("Mahalanobis model is not fitted.")
            return self.model.predict(logits)
        elif self.uncertainty_type == UncertaintyType.RISK:
            # Merge self.kwargs and runtime kwargs, runtime kwargs take precedence
            params = {**self.kwargs, **kwargs}
            required_keys = ["g_name", "risk_type", "gt_approx", "pred_approx", "T"]
            for key in required_keys:
                if key not in params:
                    raise ValueError(
                        f"Missing required parameter '{key}' for risk-based uncertainty."
                    )
            return get_risk_approximation(
                g_name=params["g_name"],
                risk_type=params["risk_type"],
                gt_approx=params["gt_approx"],
                pred_approx=params["pred_approx"],
                logits=logits,
                probabilities=params.get("probabilities", None),
                T=params["T"],
            )
        else:
            raise ValueError(
                f"Unknown or unsupported uncertainty type: {self.uncertainty_type}"
            )

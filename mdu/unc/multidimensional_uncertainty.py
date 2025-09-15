from typing import List, Dict, Any, Optional, Sequence
import numpy as np
from mdu.unc.risk_metrics.create_specific_risks import get_risk_approximation
from mdu.unc.risk_metrics.constants import RiskType
from mdu.unc.general_metrics.mahalanobis import MahalanobisDistance
from mdu.unc.general_metrics.gmm import GMM
from mdu.unc.constants import UncertaintyType


class UncertaintyEstimator:
    """
    General wrapper for uncertainty estimation, providing a scikit-learn-like interface
    with fit and predict methods. Handles both trainable and non-trainable uncertainty measures.
    """

    def __init__(
        self,
        uncertainty_type: UncertaintyType,
        print_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            uncertainty_type: Type of uncertainty measure (e.g., MAHALANOBIS, RISK)
            print_name: Name of the uncertainty measure for visualization
            **kwargs: Additional arguments required for specific uncertainty measures
        """
        self.uncertainty_type = uncertainty_type
        self.print_name = print_name
        self.kwargs = kwargs
        self.model = None
        self.is_fitted = False
        self.name = self._make_name()

    def _make_name(self):
        if self.uncertainty_type == UncertaintyType.MAHALANOBIS:
            return "MahalanobisDistance"
        elif self.uncertainty_type == UncertaintyType.RISK:
            g_name = self.kwargs.get("g_name", None)
            risk_type = self.kwargs.get("risk_type", None)
            gt_approx = self.kwargs.get("gt_approx", None)
            pred_approx = self.kwargs.get("pred_approx", None)
            # pred_approx can be None for BayesRisk
            g_name_str = (
                str(getattr(g_name, "name", g_name)) if g_name is not None else "None"
            )
            risk_type_str = (
                str(getattr(risk_type, "name", risk_type))
                if risk_type is not None
                else "None"
            )
            gt_approx_str = (
                str(getattr(gt_approx, "name", gt_approx))
                if gt_approx is not None
                else "None"
            )
            pred_approx_str = (
                str(getattr(pred_approx, "name", pred_approx))
                if pred_approx is not None
                else "None"
            )
            if pred_approx_str == "None":
                return f"{g_name_str}_{risk_type_str}_{gt_approx_str}"
            else:
                return f"{g_name_str}_{risk_type_str}_{gt_approx_str}_{pred_approx_str}"
        else:
            return str(self.uncertainty_type)

    def fit(
        self,
        train_logits: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
    ):
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
            self.model = MahalanobisDistance().fit(X=train_logits, y=y_train)
            self.is_fitted = True
        elif self.uncertainty_type == UncertaintyType.GMM:
            if train_logits is None:
                raise ValueError("train_logits must be provided for GMM uncertainty.")
            self.model = GMM().fit(X=train_logits, y=y_train)
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
                "UncertaintyEstimator must be fitted before calling predict."
            )

        if (
            self.uncertainty_type == UncertaintyType.MAHALANOBIS
            or self.uncertainty_type == UncertaintyType.GMM
        ):
            if self.model is None:
                raise RuntimeError("Mahalanobis model is not fitted.")
            return self.model.predict(logits)
        elif self.uncertainty_type == UncertaintyType.RISK:
            # Merge self.kwargs and runtime kwargs, runtime kwargs take precedence
            params = {**self.kwargs, **kwargs}
            required_keys = ["g_name", "risk_type", "gt_approx", "pred_approx", "T"]
            for key in required_keys:
                if key not in params:
                    # Allow pred_approx to be missing if risk_type is BAYES_RISK
                    if key == "pred_approx" and (
                        params.get("risk_type") == RiskType.BAYES_RISK
                        or getattr(params.get("risk_type"), "value", None)
                        == RiskType.BAYES_RISK
                    ):
                        continue
                    raise ValueError(
                        f"Missing required parameter '{key}' for risk-based uncertainty."
                    )
            # Update name in case runtime kwargs change the configuration
            self.name = self._make_name() if kwargs else self.name
            return get_risk_approximation(
                g_name=params["g_name"],
                risk_type=params["risk_type"],
                gt_approx=params["gt_approx"],
                pred_approx=params.get("pred_approx", None),
                logits=logits,
                probabilities=params.get("probabilities", None),
                T=params["T"],
            )
        else:
            raise ValueError(
                f"Unknown or unsupported uncertainty type: {self.uncertainty_type}"
            )


def compute_all_uncertainties(
    estimators: Sequence[UncertaintyEstimator], logits: np.ndarray
) -> list[np.ndarray]:
    """Return [est.predict(logits) for each estimator]."""
    return [est.predict(logits) for est in estimators]


def get_uncertainty_estimators(
    uncertainty_configs: List[Dict[str, Any]],
) -> List[UncertaintyEstimator]:
    uncertainty_estimators = []
    for config in uncertainty_configs:
        estimator = UncertaintyEstimator(
            config["type"],
            print_name=config.get("print_name", None),
            **config["kwargs"],
        )
        uncertainty_estimators.append(estimator)
    return uncertainty_estimators


def fit_uncertainty_estimators(
    uncertainty_estimators: List[UncertaintyEstimator],
    logits_train: np.ndarray,
    y_train: np.ndarray,
) -> List[UncertaintyEstimator]:
    for estimator in uncertainty_estimators:
        estimator.fit(logits_train, y_train)
    return uncertainty_estimators


def pretty_compute_all_uncertainties(
    uncertainty_estimators: List[UncertaintyEstimator],
    logits_test: np.ndarray,
) -> list[tuple[str, np.ndarray]]:
    calibration_uncertainties = compute_all_uncertainties(
        uncertainty_estimators, logits_test
    )
    uncertainty_tuples = [
        (estimator.print_name, scores)
        for estimator, scores in zip(uncertainty_estimators, calibration_uncertainties)
    ]
    return uncertainty_tuples


def fit_and_apply_uncertainty_estimators(
    uncertainty_configs: List[Dict[str, Any]],
    X_calib_logits: np.ndarray,
    y_calib: np.ndarray,
    X_test_logits: np.ndarray,
):
    uncertainty_estimators = get_uncertainty_estimators(
        uncertainty_configs=uncertainty_configs,
    )
    fitted_uncertainty_estimators = fit_uncertainty_estimators(
        uncertainty_estimators=uncertainty_estimators,
        logits_train=X_calib_logits,
        y_train=y_calib,
    )
    pretty_uncertainty_scores_calib = pretty_compute_all_uncertainties(
        uncertainty_estimators=fitted_uncertainty_estimators,
        logits_test=X_test_logits,
    )
    return pretty_uncertainty_scores_calib, fitted_uncertainty_estimators

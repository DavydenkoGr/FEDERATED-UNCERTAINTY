from typing import List, Dict, Any, Optional
import numpy as np
from ..otcp.functions import OTCPOrdering
from .constants import UncertaintyType
from .risk_metrics.create_specific_risks import get_risk_approximation
from .risk_metrics.constants import RiskType
from .general_metrics.mahalanobis import MahalanobisDistance


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


class MultiDimensionalUncertainty:
    """
    Ensemble uncertainty estimator that combines multiple uncertainty measures using Optimal Transport.

    This class follows the scikit-learn interface pattern with fit and predict methods.
    It manages multiple UncertaintyEstimator instances and uses Optimal Transport to learn
    a mapping from the combined uncertainty space to a reference distribution.
    """

    def __init__(
        self, uncertainty_configs: List[Dict[str, Any]], positive: bool = True
    ):
        """
        Initialize the ensemble with a list of uncertainty measure configurations.

        Args:
            uncertainty_configs: List of dictionaries, each containing configuration
                               for one UncertaintyEstimator (type and kwargs)
            positive: Whether to use positive reference distribution for Optimal Transport

        Example:
            configs = [
                {"type": UncertaintyType.MAHALANOBIS, "kwargs": {}},
                {"type": UncertaintyType.RISK, "kwargs": {...}}
            ]
            ensemble = UncertaintyEnsemble(configs)
        """
        self.uncertainty_configs = uncertainty_configs
        self.positive = positive

        # Initialize individual uncertainty estimators
        self.uncertainty_estimators = []
        for config in uncertainty_configs:
            estimator = UncertaintyEstimator(
                config["type"],
                print_name=config.get("print_name", None),
                **config["kwargs"],
            )
            self.uncertainty_estimators.append(estimator)

        # Optimal Transport scorer for combining uncertainty measures
        self.ot_scorer = OTCPOrdering(positive=positive)

        # Track fitting state
        self.is_fitted = False

        # Store estimator names for debugging/interpretation
        self.estimator_names = [est.name for est in self.uncertainty_estimators]
        self.estimator_print_names = [
            est.print_name for est in self.uncertainty_estimators
        ]

    def fit(
        self, logits_train: np.ndarray, y_train: np.ndarray, logits_calib: np.ndarray
    ):
        """
        Fit the uncertainty ensemble.

        This method:
        1. Fits each individual uncertainty estimator using logits_train
        2. Computes uncertainty measures on logits_calib for each estimator
        3. Stacks the results and fits the Optimal Transport mapping

        Args:
            logits_train: Training data for fitting uncertainty estimators (e.g., logits)
            logits_calib: Calibration data for fitting the Optimal Transport mapping

        Returns:
            self
        """
        # Step 1: Fit each uncertainty estimator
        for estimator in self.uncertainty_estimators:
            estimator.fit(logits_train, y_train)

        # Step 2: Compute uncertainty measures on calibration data
        calibration_uncertainties = self._compute_all_uncertainties(logits_calib)

        # Step 3: Stack uncertainties into a matrix (n_samples, n_measures)
        uncertainty_matrix = np.column_stack(calibration_uncertainties)

        # Step 4: Fit Optimal Transport on the combined uncertainty measures
        self.ot_scorer.fit(uncertainty_matrix)

        self.is_fitted = True
        return self

    def predict(
        self, logits_test: np.ndarray
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Predict uncertainty scores using the fitted ensemble.

        This method:
        1. Computes uncertainty measures for each estimator on logits_test
        2. Applies the fitted Optimal Transport mapping to get final scores

        Args:
            logits_test: Test data for computing uncertainty scores

        Returns:
            tuple: (ordering_indices, uncertainty_scores)
                - ordering_indices: Indices that sort test data by uncertainty (ascending)
                - uncertainty_scores: Dict mapping uncertainty measure names to their scores,
                                    including 'ot_scores' for the OT-combined scores
        """
        if not self.is_fitted:
            raise RuntimeError(
                "UncertaintyEnsemble must be fitted before calling predict."
            )

        # Step 1: Compute uncertainty measures for each estimator
        test_uncertainties = self._compute_all_uncertainties(logits_test)

        # Step 2: Stack uncertainties into a matrix (n_samples, n_measures)
        uncertainty_matrix = np.column_stack(test_uncertainties)

        # Step 3: Apply Optimal Transport mapping
        grid_l2_norms, ordering_indices = self.ot_scorer.predict(uncertainty_matrix)

        # Step 4: Create dictionary mapping uncertainty measure names to their scores
        uncertainty_scores = {}
        for name, print_name, scores in zip(
            self.estimator_names, self.estimator_print_names, test_uncertainties
        ):
            if print_name is None:
                uncertainty_scores[name] = scores
            else:
                uncertainty_scores[print_name] = scores

        # Add OT scores to the dictionary
        uncertainty_scores["ot_scores"] = grid_l2_norms

        return ordering_indices, uncertainty_scores

    @property
    def n_uncertainty_measures(self) -> int:
        """Number of uncertainty measures in the ensemble."""
        return len(self.uncertainty_estimators)

    def __repr__(self) -> str:
        estimator_info = ", ".join(self.estimator_names)
        return f"UncertaintyEnsemble(n_measures={self.n_uncertainty_measures}, measures=[{estimator_info}])"

    def _compute_all_uncertainties(self, logits: np.ndarray) -> List[np.ndarray]:
        """
        Compute uncertainty measures for all estimators on the given data.

        Args:
            logits: Input data for uncertainty computation

        Returns:
            List of uncertainty scores from each estimator
        """

        uncertainties = []
        for estimator in self.uncertainty_estimators:
            uncertainty_scores = estimator.predict(logits)
            uncertainties.append(uncertainty_scores)

        return uncertainties

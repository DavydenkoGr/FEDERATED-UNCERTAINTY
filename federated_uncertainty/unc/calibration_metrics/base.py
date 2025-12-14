from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np
from federated_uncertainty.unc.calibration_metrics.validation import check_probs, check_labels


class MetricBase(ABC):
    """A base class for metrics."""

    def __init__(self):
        self.requires_labels = False
        self.requires_true_proba = False

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the metric."""
        raise NotImplementedError

    def __call__(
        self,
        *,
        probs: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        true_proba: Optional[np.ndarray] = None,
    ) -> float:
        self._validate_inputs(
            probs=probs,
            y_true=y_true,
            true_proba=true_proba,
        )
        return self._compute(
            probs=probs,
            y_true=y_true,
            true_proba=true_proba,
        )

    @abstractmethod
    def _compute(self, **kwargs: Any) -> float:
        """The actual metric computation."""
        raise NotImplementedError

    def _validate_inputs(
        self,
        *,
        probs: np.ndarray,
        y_true: Optional[np.ndarray],
        true_proba: Optional[np.ndarray],
    ) -> None:
        check_probs(probs, name="probs")
        _, k = probs.shape
        if self.requires_labels:
            if y_true is None:
                raise ValueError(f"{self.name} requires `y_true`.")
            check_labels(y_true, n_classes=k)
        if self.requires_true_proba:
            if true_proba is None:
                raise ValueError(f"{self.name} requires `true_proba`.")
            check_probs(true_proba, name="true_proba")
            if true_proba.shape != probs.shape:
                raise ValueError("true_proba and probs must have the same shape.")


class LabelBasedMetricBase(MetricBase, ABC):
    """Base class for metrics that require ground truth labels."""

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def _compute(
        self,
        *,
        probs: np.ndarray,
        y_true: Optional[np.ndarray],
        true_proba: Optional[np.ndarray],
    ) -> float:
        raise NotImplementedError


class TrueProbMetricBase(MetricBase, ABC):
    """Base class for metrics that require true probabilities."""

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def _compute(
        self,
        *,
        probs: np.ndarray,
        y_true: Optional[np.ndarray],
        true_proba: Optional[np.ndarray],
    ) -> float:
        raise NotImplementedError

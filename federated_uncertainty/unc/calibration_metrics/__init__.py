"""
This module provides a set of metrics for evaluating the performance of a model.
"""

from typing import Any

from federated_uncertainty.unc.calibration_metrics.base import (
    LabelBasedMetricBase,
    MetricBase,
    TrueProbMetricBase,
)
from federated_uncertainty.unc.calibration_metrics.calibration_errors import (
    ClasswiseExpectedCalibrationError,
    ExpectedCalibrationError,
    MaximumCalibrationError,
)


def get_metric(name: str, **kwargs: Any) -> MetricBase:
    """
    Factory function to get a metric instance by name.
    """
    name = name.lower().strip()
    if name == "ece":
        return ExpectedCalibrationError(**kwargs)
    elif name == "mce":
        return MaximumCalibrationError(**kwargs)
    elif name == "cw-ece":
        return ClasswiseExpectedCalibrationError(**kwargs)
    else:
        raise ValueError(f"Unknown metric: {name}")


__all__ = [
    "MetricBase",
    "LabelBasedMetricBase",
    "TrueProbMetricBase",
    "ExpectedCalibrationError",
    "MaximumCalibrationError",
    "ClasswiseExpectedCalibrationError",
    "get_metric",
]

from collections import defaultdict
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class MahalanobisDistance(BaseEstimator, TransformerMixin):
    """
    Scikit-learn style estimator for Mahalanobis uncertainty.
    Fits mean and covariance on in-distribution logits, and computes
    Mahalanobis distance for new logits.
    """

    def __init__(self, regularization: float = 1e-6):
        self.regularization = regularization
        self.mean_ = defaultdict(lambda: None)
        self.inv_cov_ = defaultdict(lambda: None)
        self.n_models_ = None

    def fit(self, X: np.ndarray, y=None):
        """
        Fit the mean and covariance matrix on in-distribution logits.

        Args:
            X: np.ndarray of shape (n_samples, n_classes)
            y: Ignored
        Returns:
            self
        """
        self.n_models_ = X.shape[0]
        for i in range(self.n_models_):
            self.mean_[i] = np.mean(X[i], axis=0)
            cov = np.cov(X[i], rowvar=False)
            cov += np.eye(cov.shape[0]) * self.regularization
            self.inv_cov_[i] = np.linalg.inv(cov)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Compute Mahalanobis distance for each sample in X.

        Args:
            X: np.ndarray of shape (n_samples, n_classes)
        Returns:
            distances: np.ndarray of shape (n_samples,)
        """
        if self.mean_[0] is None or self.inv_cov_[0] is None:
            raise RuntimeError(
                "MahalanobisUncertainty must be fitted before calling predict."
            )
        distances = []
        for i in range(self.n_models_):
            diff = X[i] - self.mean_[i]
            left = np.dot(diff, self.inv_cov_[i])
            distances.append(np.sqrt(np.sum(left * diff, axis=1)))
        return np.mean(distances, axis=0)

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
        self.mean_ = None
        self.inv_cov_ = None

    def fit(self, X: np.ndarray, y=None):
        """
        Fit the mean and covariance matrix on in-distribution logits.

        Args:
            X: np.ndarray of shape (n_samples, n_classes)
            y: Ignored
        Returns:
            self
        """
        self.mean_ = np.mean(X, axis=0)
        cov = np.cov(X, rowvar=False)
        cov += np.eye(cov.shape[0]) * self.regularization
        self.inv_cov_ = np.linalg.inv(cov)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Compute Mahalanobis distance for each sample in X.

        Args:
            X: np.ndarray of shape (n_samples, n_classes)
        Returns:
            distances: np.ndarray of shape (n_samples,)
        """
        if self.mean_ is None or self.inv_cov_ is None:
            raise RuntimeError(
                "MahalanobisUncertainty must be fitted before calling predict."
            )
        diff = X - self.mean_
        left = np.dot(diff, self.inv_cov_)
        distances = np.sqrt(np.sum(left * diff, axis=1))
        return distances

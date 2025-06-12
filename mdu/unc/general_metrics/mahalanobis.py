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
        # Per-model, per-class parameters
        self.class_mean_ = defaultdict(lambda: {})          # { model_idx: {class_label: mean_vector} }
        self.class_inv_cov_ = defaultdict(lambda: {})       # { model_idx: {class_label: inv_cov_matrix} }
        self.n_models_ = None


    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit class-conditional means and covariances for each model.

        Args:
            X: np.ndarray of shape (n_models, n_samples, n_features)
            y: np.ndarray of shape (n_samples,)
        Returns:
            self
        """
        # Number of ensemble members
        self.n_models_ = X.shape[0]

        # Identify all classes once
        classes = np.unique(y)

        for i in range(self.n_models_):
            # For each class, compute mean and covariance on model i’s logits
            for c in classes:
                # Select logits belonging to class c
                X_ic = X[i][y == c]
                # Empirical mean
                mu_ic = np.mean(X_ic, axis=0)
                # Empirical covariance with regularization
                cov_ic = np.cov(X_ic, rowvar=False)
                cov_ic += np.eye(cov_ic.shape[0]) * self.regularization
                # Store parameters
                self.class_mean_[i][c] = mu_ic
                self.class_inv_cov_[i][c] = np.linalg.inv(cov_ic)

        return self


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Compute class-conditional Mahalanobis distance scores.

        Args:
            X: np.ndarray of shape (n_models, n_samples, n_features)
        Returns:
            distances: np.ndarray of shape (n_samples,)
        """
        if self.n_models_ is None:
            raise RuntimeError("Fit must be called before predict.")

        # Collect per-model minimal distances
        per_model_dists = []

        for i in range(self.n_models_):
            # For each sample, compute distance to each class and take min
            dists_i = []
            for c, mu_ic in self.class_mean_[i].items():
                inv_cov_ic = self.class_inv_cov_[i][c]
                diff = X[i] - mu_ic  # shape: (n_samples, n_features)
                # Mahalanobis distance per sample for class c
                m_dist_c = np.sqrt(np.sum(diff @ inv_cov_ic * diff, axis=1))
                dists_i.append(m_dist_c)
            # Stack distances for all classes: shape (n_classes, n_samples)
            dists_i = np.stack(dists_i, axis=0)
            # Minimum across classes for each sample: shape (n_samples,)
            per_model_dists.append(np.min(dists_i, axis=0))

        # Average distances across models: final shape (n_samples,)
        return np.mean(np.stack(per_model_dists, axis=0), axis=0)

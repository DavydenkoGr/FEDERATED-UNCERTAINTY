from collections import defaultdict
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class MahalanobisDistance(BaseEstimator, TransformerMixin):
    """
    Scikit-learn style estimator for class-conditional Mahalanobis confidence.
    Fits per-class means and a tied covariance per ensemble member,
    then averages the negative squared Mahalanobis distances.
    """

    def __init__(self, regularization: float = 1e-6):
        self.regularization = regularization
        # { model_idx: { class_label: mean_vector } }
        self.class_mean_ = defaultdict(dict)
        # { model_idx: inverse_covariance_matrix }
        self.inv_cov_ = {}
        self.n_models_ = None
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Estimate class-conditional means and tied covariance for each model.

        Parameters
        ----------
        X : array, shape (n_models, n_samples, n_features)
            Logits or embeddings from each ensemble member.
        y : array, shape (n_samples,)
            True class labels for in-distribution data.

        Returns
        -------
        self
        """
        self.n_models_ = X.shape[0]
        self.classes_ = np.unique(y)
        N_total = X.shape[1]

        for i in range(self.n_models_):
            Xi = X[i]  # shape: (n_samples, n_features)
            # 1) Compute per-class means μ̂_{i,c}
            for c in self.classes_:
                X_ic = Xi[y == c]
                self.class_mean_[i][c] = np.mean(X_ic, axis=0)

            # 2) Compute tied covariance Σ̂_i across all classes
            #    by stacking (X_ic - μ̂_{i,c}) for every class
            diffs = np.vstack(
                [Xi[y == c] - self.class_mean_[i][c] for c in self.classes_]
            )  # shape: (N_total, n_features)

            cov = (diffs.T @ diffs) / N_total
            cov += np.eye(cov.shape[0]) * self.regularization

            # 3) Store inverse covariance
            self.inv_cov_[i] = np.linalg.inv(cov)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the averaged Mahalanobis confidence score.

        Parameters
        ----------
        X : array, shape (n_models, n_samples, n_features)
            Test logits or embeddings.

        Returns
        -------
        scores : array, shape (n_samples,)
            Final Mahalanobis-based confidence scores.
        """
        if self.n_models_ is None:
            raise RuntimeError("Must call fit before predict.")

        n_models, n_samples, _ = X.shape
        model_scores = np.zeros((n_models, n_samples))

        for i in range(n_models):
            Xi = X[i]
            inv_cov = self.inv_cov_[i]

            # Compute negative squared Mahalanobis for each class
            per_class_scores = []
            for c in self.classes_:
                mu_ic = self.class_mean_[i][c]
                diff = Xi - mu_ic  # shape: (n_samples, n_features)
                sq_maha = np.sum((diff @ inv_cov) * diff, axis=1)
                per_class_scores.append(sq_maha)

            # shape: (n_classes, n_samples)
            per_class_scores = np.stack(per_class_scores, axis=0)
            # take the highest (least distance) per sample
            model_scores[i] = -np.max(-per_class_scores, axis=0)

        # Average across ensemble members
        return np.mean(model_scores, axis=0)

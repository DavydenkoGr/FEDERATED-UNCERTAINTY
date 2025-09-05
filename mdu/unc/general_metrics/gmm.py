from collections import defaultdict
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import multivariate_normal
from scipy.special import logsumexp


class GMM(BaseEstimator, TransformerMixin):
    """
    Scikit-learn style estimator for GMM.
    Fits a GMM per ensemble member, then averages the log-likelihoods.
    """

    def __init__(self, regularization: float = 1e-6):
        self.regularization = regularization
        # { model_idx: { class_label: mean_vector } }
        self.class_mean_ = defaultdict(dict)
        # { model_idx: { class_label: std_vector } }
        self.std_ = defaultdict(dict)
        self.class_weights_ = defaultdict(dict)
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
        self.classes_, class_counts_ = np.unique(y, return_counts=True)
        self.class_weights_ = {
            c: count / len(y) for c, count in zip(self.classes_, class_counts_)
        }

        for i in range(self.n_models_):
            Xi = X[i]
            for c in self.classes_:
                X_ic = Xi[y == c]
                self.class_mean_[i][c] = np.mean(X_ic, axis=0)
                self.std_[i][c] = np.std(X_ic, axis=0)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the log-likelihood of the GMM for each sample.

        Parameters
        ----------
        X : array, shape (n_models, n_samples, n_features)
            Test logits or embeddings.

        Returns
        -------
        scores : array, shape (n_samples,)
            Log-likelihood of the GMM for each sample.
        """
        if self.n_models_ is None:
            raise RuntimeError("Must call fit before predict.")

        n_models, n_samples, _ = X.shape
        model_scores = np.zeros((n_models, n_samples))

        for i in range(n_models):
            Xi = X[i]

            current_scores = []

            for c in self.classes_:
                log_pdf = multivariate_normal(
                    mean=self.class_mean_[i][c], cov=self.std_[i][c]
                ).logpdf(Xi)
                current_scores.append(log_pdf + np.log(self.class_weights_[c]))
            current_scores_array = np.vstack(current_scores)
            model_scores[i] = logsumexp(current_scores_array, axis=0)

        return -(logsumexp(model_scores, axis=0) - np.log(n_models))

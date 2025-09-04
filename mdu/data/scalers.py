import numpy as np

class MahalanobisWhiteningScaler:
    """
    Mahalanobis whitening (a.k.a. covariance whitening).
    Learns mean μ and (shrunk) covariance Σ, then transforms X -> (Σ^{-1/2})(X-μ).
    
    Parameters
    ----------
    shrinkage : float in [0, 1], default=0.1
        Ledoit–Wolf-style convex shrinkage toward identity:
        Σ_shrunk = (1 - a) * Σ + a * (tr(Σ)/d) * I
        (diag mode shrinks per-feature variances toward their mean).
    diagonal : bool, default=False
        If True, uses diagonal Σ only (variance scaling + decorrelation off).
    eps : float, default=1e-8
        Floor added to eigenvalues/variances for numerical stability.

    Attributes (set by fit)
    -----------------------
    mean_ : (d,) array
    cov_ : (d, d) array  (or diagonal as (d,) when diagonal=True)
    inv_sqrt_ : (d, d) array  (or (d,) when diagonal=True)  # Σ^{-1/2}
    metric_ : (d, d) array  # Σ^{-1} (or diagonal as (d,))
    """
    def __init__(self, shrinkage: float = 0.1, diagonal: bool = False, eps: float = 1e-8):
        self.shrinkage = float(shrinkage)
        self.diagonal = bool(diagonal)
        self.eps = float(eps)

        self.mean_ = None
        self.cov_ = None
        self.inv_sqrt_ = None
        self.metric_ = None  # Σ^{-1}

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n, d = X.shape

        # Mean
        self.mean_ = X.mean(axis=0)

        # Center
        Xc = X - self.mean_

        if self.diagonal:
            # Diagonal covariance (per-feature variances)
            var = Xc.var(axis=0, ddof=1) if n > 1 else np.zeros(d)
            # Shrink toward mean variance
            vbar = var.mean() if d > 0 else 0.0
            var_shrunk = (1.0 - self.shrinkage) * var + self.shrinkage * vbar
            var_shrunk = np.maximum(var_shrunk, self.eps)

            self.cov_ = var_shrunk  # store diagonal as 1-D
            self.inv_sqrt_ = 1.0 / np.sqrt(var_shrunk)
            self.metric_ = 1.0 / var_shrunk  # Σ^{-1} diagonal

        else:
            # Full covariance
            if n > 1:
                cov = (Xc.T @ Xc) / (n - 1)
            else:
                cov = np.zeros((d, d))

            # Shrink toward identity scaled by average variance
            tr_over_d = np.trace(cov) / max(d, 1)
            I = np.eye(d)
            cov_shrunk = (1.0 - self.shrinkage) * cov + self.shrinkage * tr_over_d * I

            # Symmetrize for safety
            cov_shrunk = 0.5 * (cov_shrunk + cov_shrunk.T)

            # Eigen-decomposition (covariance is PSD)
            w, Q = np.linalg.eigh(cov_shrunk)
            w = np.maximum(w, self.eps)

            inv_sqrt = (Q * (1.0 / np.sqrt(w))) @ Q.T
            inv_cov = (Q * (1.0 / w)) @ Q.T

            self.cov_ = cov_shrunk
            self.inv_sqrt_ = inv_sqrt
            self.metric_ = inv_cov

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        single = False
        if X.ndim == 1:
            X = X.reshape(1, -1)
            single = True

        Xc = X - self.mean_

        if self.diagonal:
            Xw = Xc * self.inv_sqrt_
        else:
            Xw = Xc @ self.inv_sqrt_.T  # (n,d) @ (d,d)

        return Xw[0] if single else Xw

    # Convenience: get the quadratic metric M = Σ^{-1} for use in OT costs
    def get_metric_matrix(self):
        """
        Returns Σ^{-1}. If diagonal=True, returns a 1-D vector of diagonal entries.
        """
        return self.metric_



class GlobalMinMaxScaler:
    """Simple global min-max scaler that uses global min/max across all features"""
    
    def __init__(self):
        self.global_min_ = None
        self.global_max_ = None
        
    def fit(self, X):
        """Fit the scaler using global min/max from the data"""
        self.global_min_ = np.min(X)
        self.global_max_ = np.max(X)
        return self
        
    def transform(self, X):
        """Transform data using global min/max"""
        if self.global_max_ > self.global_min_:
            return (X - self.global_min_) / (self.global_max_ - self.global_min_)
        else:
            # If all values are the same, return zeros
            return np.zeros_like(X)

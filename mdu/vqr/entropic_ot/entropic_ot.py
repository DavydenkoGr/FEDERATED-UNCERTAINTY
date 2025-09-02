from mdu.vqr.prototype import BaseMultidimensionalOrdering
from typing import Tuple, Optional, Dict, Any
from scipy.special import betainc as _betainc
import numpy as np
import torch
import ot

def _sinkhorn_potentials_pot(
    a: np.ndarray, b: np.ndarray, C: np.ndarray,
    eps: float, max_iters: int, tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    # Log-domain stabilized Sinkhorn. Returns plan and a log dict.
    G, log = ot.bregman.sinkhorn_log(
        a, b, C, reg=eps, numItermax=max_iters, stopThr=tol, log=True, verbose=False
    )
    # POT may provide either (alpha,beta) (dual potentials) or (u,v) (scalings).
    if "alpha" in log and "beta" in log:
        f = log["alpha"]                   # shape (n,)
        g = log["beta"]                    # shape (m,)
    else:
        # fall back to u,v -> potentials: f = eps * log u, g = eps * log v
        u = np.maximum(log["u"], 1e-300)
        v = np.maximum(log["v"], 1e-300)
        f = eps * np.log(u)
        g = eps * np.log(v)
    return f, g


class EntropicOTOrdering(BaseMultidimensionalOrdering):
    """
    Entropic OT (balanced, ε>0) from empirical source μ_X to a discrete target
    sampled from one of:
      - "ball": Uniform on unit d-ball (center-outward MK ranks; scalar rank = ||û|| ∈ [0,1])
      - "exp" : Product of independent Exponential(λ_j) on R_+^d
      - "beta": Product of independent Beta(α_j, β_j) on [0,1]^d

    Fit solves Sinkhorn in the log-domain and stores the column potential g_
    and target cloud Y_. Predict maps new points out-of-sample by barycentric
    projection using the stored g_ (no re-optimization) and returns both the
    mapped points and a scalar rank in [0,1] (definition depends on target).

    Parameters
    ----------
    target : {"ball","exp","beta"}
        Reference measure.
    target_params : dict
        Parameters for the target:
          - "exp": {"rates": array_like of shape (d,) or scalar}
          - "beta": {"alpha": array_like (d,) or scalar, "beta": array_like (d,) or scalar}
          - "ball": {}  (ignored)
    eps : float
        Entropic regularization ε (>0). Smaller = crisper, larger = smoother.
    n_targets : Optional[int]
        Number of target samples; default = n_samples at fit time.
    standardize : bool
        If True, z-score features in fit and predict (improves stability).
    max_iters : int
        Max Sinkhorn iterations.
    tol : float
        Convergence tolerance on the row potential.
    random_state : Optional[int]
        RNG seed for target sampling.

    Attributes after fit
    --------------------
    mean_, scale_ : arrays of shape (d,)
        Standardization stats (or zeros/ones if standardize=False).
    dim_ : int
        Data dimension.
    Y_ : (m, d)
        Target support points used in OT.
    g_ : (m,)
        Learned column potential for the Sinkhorn solution at fit time.
    """

    def __init__(
        self,
        target: str = "ball",
        target_params: Optional[Dict[str, Any]] = None,
        fit_mse_params: bool = False,
        eps: float = 0.25,
        n_targets: Optional[int] = None,
        standardize: bool = True,
        max_iters: int = 2000,
        tol: float = 1e-9,
        random_state: Optional[int] = None,
    ):
        self.target = target
        self.params = target_params or {}
        self.fit_mse_params = fit_mse_params
        self.eps = float(eps)
        self.n_targets = n_targets
        self.standardize = bool(standardize)
        self.max_iters = int(max_iters)
        self.tol = float(tol)
        self.rng = np.random.default_rng(random_state)

        # learned state
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.dim_: Optional[int] = None
        self.Y_: Optional[np.ndarray] = None
        self.g_: Optional[np.ndarray] = None

    def fit(
        self, train_loader: torch.utils.data.DataLoader, train_params: dict
    ) -> "EntropicOTOrdering":
        scores_cal = train_loader.dataset.cpu().numpy()
        X = np.asarray(scores_cal, dtype=float)
        n, d = X.shape
        self.dim_ = d

        # standardize
        if self.standardize:
            self.mean_ = X.mean(axis=0)
            self.scale_ = X.std(axis=0)
            self.scale_[self.scale_ < 1e-12] = 1.0
            Xz = (X - self.mean_) / self.scale_
        else:
            self.mean_ = np.zeros(d)
            self.scale_ = np.ones(d)
            Xz = X

        if self.fit_mse_params:
            self.params = self._fit_target_params(Xz, self.target)

        # sample target cloud
        m = int(self.n_targets) if self.n_targets is not None else n
        self.Y_ = self._sample_target(self.target, m, d, self.params)

        # Sinkhorn (log domain) to get dual potentials
        a = np.full(n, 1.0 / n)
        b = np.full(m, 1.0 / m)
        C = self._cdist_sqeuclidean(Xz, self.Y_)
        # f, g = self._sinkhorn_log_balanced(a, b, C, self.eps, self.max_iters, self.tol)
        f, g = _sinkhorn_potentials_pot(a, b, C, self.eps, self.max_iters, self.tol)
        self.g_ = g
        return self

    def predict(
        self,
        scores_test,
    ) -> (
        np.ndarray
        | Tuple[np.ndarray, np.ndarray]
        | Tuple[np.ndarray, np.ndarray, np.ndarray]
    ):
        """
        Out-of-sample mapping and scalar ranks for new points.

        Returns
        -------
        rank : (k,) array in [0,1]
            Scalar "rank" (definition depends on target).
        If return_map:
            Uhat : (k, d) barycentric images on the target space.
        If return_plan_rows:
            W : (k, m) soft row couplings (each row sums to 1).
        """
        self._check_is_fitted()

        X_new = np.asarray(scores_test, dtype=float)
        Xz = (X_new - self.mean_) / self.scale_

        C_new = self._cdist_sqeuclidean(Xz, self.Y_)  # (k, m)
        L = (self.g_[None, :] - C_new) / max(self.eps, 1e-12)  # log-weights
        L -= L.max(axis=1, keepdims=True)  # stabilize
        W = np.exp(L)
        W /= W.sum(axis=1, keepdims=True)

        Uhat = W @ self.Y_  # barycentric image
        # rank = self._scalar_rank_from_target(self.target, Uhat, self.params)
        rank = np.linalg.norm(Uhat, axis=1, ord=2)

        return rank, C_new

    def predict_ranks(self, scores_test):
        scores_test = np.asarray(scores_test, dtype=float)
        return self.predict(scores_test)[0]

    def fit_predict(self, X: np.ndarray, **predict_kwargs):
        self.fit(X)
        return self.predict(X, **predict_kwargs)

    # -------------------- internals --------------------

    @staticmethod
    def _cdist_sqeuclidean(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        A2 = np.sum(A**2, axis=1, keepdims=True)
        B2 = np.sum(B**2, axis=1, keepdims=True).T
        return A2 + B2 - 2.0 * (A @ B.T)

    @staticmethod
    def _logsumexp(M: np.ndarray, axis: int) -> np.ndarray:
        m = np.max(M, axis=axis, keepdims=True)
        out = np.log(np.sum(np.exp(M - m), axis=axis, keepdims=True)) + m
        return out.squeeze(axis=axis)

    def _sinkhorn_log_balanced(
        self,
        a: np.ndarray,
        b: np.ndarray,
        C: np.ndarray,
        eps: float,
        max_iters: int,
        tol: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n, m = C.shape
        f = np.zeros(n)
        g = np.zeros(m)
        log_a = np.log(a + 1e-300)
        log_b = np.log(b + 1e-300)
        inv_eps = 1.0 / max(eps, 1e-12)

        for _ in range(max_iters):
            f_prev = f.copy()
            M = (f[:, None] + g[None, :] - C) * inv_eps
            f += eps * (log_a - self._logsumexp(M, axis=1))
            M = (f[:, None] + g[None, :] - C) * inv_eps
            g += eps * (log_b - self._logsumexp(M, axis=0))
            if np.max(np.abs(f - f_prev)) < tol:
                break
        return f, g

    def _fit_target_params(self, data: np.ndarray, target: str) -> Dict[str, Any]:
        target = target.lower()
        n_samples, n_features = data.shape
        
        if target == "ball":
            return {}
            
        elif target == "exp":
            rates = np.zeros(n_features)
            for j in range(n_features):
                mean_j = np.mean(data[:, j])
                rates[j] = 1.0 / max(mean_j, 1e-12)  # Avoid division by zero
            return {"rates": rates}
            
        elif target == "beta":
            # Fit beta distribution parameters coordinate-wise using method of moments
            alpha = np.zeros(n_features)
            beta = np.zeros(n_features)
            
            for j in range(n_features):
                # Check if data is in [0, 1] range for beta distribution
                if np.any(data[:, j] < 0) or np.any(data[:, j] > 1):
                    raise ValueError(
                        f"Beta distribution requires data in [0, 1] range, "
                        f"but column {j} has values in [{np.min(data[:, j]):.6f}, {np.max(data[:, j]):.6f}]"
                    )
                col_data = np.clip(data[:, j], 1e-12, 1 - 1e-12)
                
                # Method of moments estimators
                sample_mean = np.mean(col_data)
                sample_var = np.var(col_data)
                
                # Avoid numerical issues
                sample_mean = np.clip(sample_mean, 1e-6, 1 - 1e-6)
                sample_var = min(sample_var, sample_mean * (1 - sample_mean) * 0.99)
                
                # Method of moments formulas
                nu = sample_mean * (1 - sample_mean) / sample_var - 1
                alpha[j] = sample_mean * nu
                beta[j] = (1 - sample_mean) * nu
                
                # Ensure positive parameters
                alpha[j] = max(alpha[j], 0.1)
                beta[j] = max(beta[j], 0.1)
                
            return {"alpha": alpha, "beta": beta}
            
        else:
            raise ValueError(
                f"Unknown target '{target}' (use 'ball', 'exp', or 'beta')."
            )



    def _sample_target(
        self, target: str, m: int, d: int, params: Dict[str, Any]
    ) -> np.ndarray:
        target = target.lower()
        if target == "ball":
            # Uniform on unit d-ball
            Z = self.rng.normal(size=(m, d))
            Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
            radii = self.rng.random(m) ** (1.0 / d)
            return Z * radii[:, None]

        elif target == "exp":
            # Product of independent exponentials with rates λ_j > 0
            lam = params.get("rates", 2.3) # 1 default
            lam = np.asarray(lam, dtype=float)
            if lam.size == 1:
                lam = np.full(d, lam.item())
            assert lam.shape == (d,) and np.all(lam > 0), (
                "rates must be shape (d,) and >0"
            )
            U = self.rng.random((m, d))
            # Inverse CDF: y_j = -ln(1-U)/λ_j
            return -np.log(1.0 - U) / lam[None, :]

        elif target == "beta":
            # Product of independent Beta(α_j, β_j) on [0,1]
            alpha = params.get("alpha", 1.0)
            beta = params.get("beta", 1.0)
            alpha = np.asarray(alpha, dtype=float)
            beta = np.asarray(beta, dtype=float)
            if alpha.size == 1:
                alpha = np.full(d, alpha.item())
            if beta.size == 1:
                beta = np.full(d, beta.item())
            assert alpha.shape == (d,) and beta.shape == (d,), (
                "alpha,beta must be shape (d,)"
            )
            assert np.all(alpha > 0) and np.all(beta > 0), "alpha,beta must be >0"
            Y = np.empty((m, d))
            for j in range(d):
                Y[:, j] = self.rng.beta(alpha[j], beta[j], size=m)
            return Y

        else:
            raise ValueError(
                f"Unknown target '{target}' (use 'ball', 'exp', or 'beta')."
            )

    # ---- scalar rank for coloring/ordering ----

    def _scalar_rank_from_target(
        self, target: str, Uhat: np.ndarray, params: Dict[str, Any]
    ) -> np.ndarray:
        target = target.lower()
        if target == "ball":
            # classic MK rank radius
            return np.linalg.norm(Uhat, axis=1)

        elif target == "exp":
            # product of marginal exponential CDFs: Π_j (1 - exp(-λ_j y_j))
            lam = params.get("rates", 1.0)
            lam = np.asarray(lam, dtype=float)
            if lam.size == 1:
                lam = np.full(Uhat.shape[1], lam.item())
            lam = lam[None, :]  # (1, d)
            F = 1.0 - np.exp(-np.clip(Uhat, 0.0, None) * lam)
            # return np.prod(F, axis=1)
            return np.linalg.norm(F, axis=1, ord=2)

        elif target == "beta":
            # product of marginal Beta CDFs: Π_j I_{y_j}(α_j, β_j)
            alpha = params.get("alpha", 1.0)
            beta = params.get("beta", 1.0)
            alpha = np.asarray(alpha, dtype=float)
            beta = np.asarray(beta, dtype=float)
            if alpha.size == 1:
                alpha = np.full(Uhat.shape[1], alpha.item())
            if beta.size == 1:
                beta = np.full(Uhat.shape[1], beta.item())
            # betainc(a,b,x) is the regularized incomplete beta (CDF)
            F = np.empty_like(Uhat)
            # clip to [0,1] for safety
            Y = np.clip(Uhat, 0.0, 1.0)
            for j in range(Uhat.shape[1]):
                F[:, j] = _betainc(alpha[j], beta[j], Y[:, j])
            # return np.prod(F, axis=1)
            return np.linalg.norm(F, axis=1, ord=2)

        else:
            raise ValueError(f"Unknown target '{target}'.")

    # ---- checks ----

    def _check_is_fitted(self):
        if self.Y_ is None or self.g_ is None or self.dim_ is None:
            raise RuntimeError("Call fit(X) before predict().")

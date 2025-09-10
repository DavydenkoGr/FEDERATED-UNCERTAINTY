from typing import Tuple, Optional, Dict, Any
from scipy.special import betainc
import numpy as np

from mdu.unc.constants import OTTarget, SamplingMethod, ScalingType
from mdu.unc.ot_utils import (
    sample_uniform_random,
    sample_uniform_sobol,
    sample_uniform_grid,
    transform_to_ball,
    sinkhorn_potentials_pot,
    transform_to_beta,
    generate_unit_hypercube_grid_nodes
)
from mdu.data.scalers import GlobalMinMaxScaler
from sklearn.preprocessing import MinMaxScaler


class EntropicOTOrdering:
    def __init__(
        self,
        target: OTTarget,
        sampling_method: SamplingMethod,
        scaling_type: ScalingType,
        grid_size: int,
        target_params: Dict[str, Any],
        eps: float,
        n_targets_multiplier: int,
        max_iters: int,
        random_state: int,
        tol: float = 1e-6,
    ):
        self.target = target.value
        self.sampling_method = sampling_method.value
        self.scaling_type = scaling_type.value
        self.grid_size = grid_size

        self.params = target_params
        self.eps = eps
        self.n_targets_multiplier = n_targets_multiplier
        self.max_iters = max_iters
        self.tol = tol
        self.rng = np.random.default_rng(random_state)

        self.dim_: Optional[int] = None
        self.Y_: Optional[np.ndarray] = None
        self.g_: Optional[np.ndarray] = None
        self.scaler: Optional[MinMaxScaler | GlobalMinMaxScaler] = None

    def fit(self, scores_cal: np.ndarray) -> "EntropicOTOrdering":
        Xz = np.asarray(scores_cal, dtype=np.float64)
        self._fit_scaler(Xz)
        Xz_transformed = self._transform_and_add_grid(Xz)
        n, d = Xz.shape
        self.dim_ = d

        # sample target cloud
        m = n * self.n_targets_multiplier
        self.Y_ = self._sample_target(self.target, m, d, self.params)

        # Sinkhorn (log domain) to get dual potentials
        a = np.full(n, 1.0 / n)
        b = np.full(m, 1.0 / m)
        C = self._cdist_sqeuclidean(Xz_transformed, self.Y_)
        _, g = sinkhorn_potentials_pot(a, b, C, self.eps, self.max_iters, self.tol)
        self.g_ = g
        return self

    def predict(
        self,
        scores_test: np.ndarray,
    ) -> (
        np.ndarray
        | Tuple[np.ndarray, np.ndarray]
        | Tuple[np.ndarray, np.ndarray, np.ndarray]
    ):
        self._check_is_fitted()

        X_new = np.asarray(scores_test, dtype=np.float64)
        Xz = self.scaler.transform(X_new)

        C_new = self._cdist_sqeuclidean(Xz, self.Y_)  # (k, m)
        L = (self.g_[None, :] - C_new) / max(self.eps, 1e-12)  # log-weights
        L -= L.max(axis=1, keepdims=True)  # stabilize
        W = np.exp(L)
        W /= W.sum(axis=1, keepdims=True)

        Uhat = W @ self.Y_  # barycentric image
        rank = np.linalg.norm(Uhat, axis=1, ord=2)

        return rank, C_new

    # -------------------- internals --------------------
    def _fit_scaler(self, scores_cal: np.ndarray) -> None:
        if self.scaling_type == ScalingType.GLOBAL.value:
            self.scaler = GlobalMinMaxScaler()
        elif self.scaling_type == ScalingType.FEATURE_WISE.value:
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling type: {self.scaling_type}")

        self.scaler.fit(scores_cal)

    def _transform_and_add_grid(self, scores_cal: np.ndarray) -> np.ndarray:
        scores_cal_transformed = self.scaler.transform(scores_cal)
        n_measures = scores_cal_transformed.shape[1]
        unit_grid_nodes = generate_unit_hypercube_grid_nodes(n_measures)

        if self.scaling_type == ScalingType.GLOBAL.value:
            metric_wise_scaled_maximums = self.scaler.local_max_ / self.scaler.global_max_
        else:
            metric_wise_scaled_maximums = 1
        maximal_elements_grid = (
            np.vstack(unit_grid_nodes)
            * self.grid_size
            * metric_wise_scaled_maximums
        )
        stacked_transformed_scores = np.vstack(
            [scores_cal_transformed, maximal_elements_grid]
        )
        return stacked_transformed_scores

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

    def _sample_target(
        self, target: str, m: int, d: int, params: Dict[str, Any]
    ) -> np.ndarray:
        target = target.lower()

        # Generate uniform samples using the specified sampling method
        if self.sampling_method == SamplingMethod.RANDOM.value:
            U = sample_uniform_random(self.rng, m, d)
        elif self.sampling_method == SamplingMethod.SOBOL.value:
            U = sample_uniform_sobol(self.rng, m, d)
        elif self.sampling_method == SamplingMethod.GRID.value:
            U = sample_uniform_grid(self.rng, m, d)
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling_method}")

        # Transform uniform samples to target distribution
        if target == OTTarget.BALL.value:
            return transform_to_ball(self.sampling_method, U, d)

        elif target == OTTarget.EXP.value:
            # Product of independent exponentials with rates λ_j > 0
            lam = params.get("rates", 2.3)
            lam = np.asarray(lam, dtype=np.float64)
            if lam.size == 1:
                lam = np.full(d, lam.item())
            assert lam.shape == (d,) and np.all(lam > 0), (
                "rates must be shape (d,) and >0"
            )
            # Inverse CDF: y_j = -ln(1-U)/λ_j
            return -np.log(1.0 - U) / lam[None, :]

        elif target == OTTarget.BETA.value:
            # Product of independent Beta(α_j, β_j) on [0,1]
            alpha = params.get("alpha", 1.0)
            beta = params.get("beta", 1.0)
            alpha = np.asarray(alpha, dtype=np.float64)
            beta = np.asarray(beta, dtype=np.float64)
            if alpha.size == 1:
                alpha = np.full(d, alpha.item())
            if beta.size == 1:
                beta = np.full(d, beta.item())
            assert alpha.shape == (d,) and beta.shape == (d,), (
                "alpha,beta must be shape (d,)"
            )
            assert np.all(alpha > 0) and np.all(beta > 0), "alpha,beta must be >0"
            return transform_to_beta(U, alpha, beta)

        else:
            raise ValueError(
                f"Unknown target '{target}' (use 'ball', 'exp', or 'beta')."
            )

    def _check_is_fitted(self):
        if self.scaler is None or self.Y_ is None or self.g_ is None or self.dim_ is None:
            raise RuntimeError("Call fit(X) before predict().")

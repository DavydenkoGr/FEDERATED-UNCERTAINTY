import numpy as np
from scipy.stats import qmc
import ot

from federated_uncertainty.unc.constants import SamplingMethod


def sinkhorn_potentials_pot(
    a: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    eps: float,
    max_iters: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    # Log-domain stabilized Sinkhorn. Returns plan and a log dict.
    G, log = ot.bregman.sinkhorn_log(
        a, b, C, reg=eps, numItermax=max_iters, stopThr=tol, log=True, verbose=False
    )
    # POT may provide either (alpha,beta) (dual potentials) or (u,v) (scalings).
    if "alpha" in log and "beta" in log:
        f = log["alpha"]  # shape (n,)
        g = log["beta"]  # shape (m,)
    else:
        # fall back to u,v -> potentials: f = eps * log u, g = eps * log v
        u = np.maximum(log["u"], 1e-300)
        v = np.maximum(log["v"], 1e-300)
        f = eps * np.log(u)
        g = eps * np.log(v)
    return f, g


def sample_uniform_random(rng: np.random.Generator, m: int, d: int) -> np.ndarray:
    """Generate uniform random samples in [0, 1]^d."""
    return rng.random((m, d))


def sample_uniform_sobol(rng: np.random.Generator, m: int, d: int) -> np.ndarray:
    """Generate low-discrepancy Sobol sequence samples in [0, 1]^d."""
    # Create Sobol sampler
    sampler = qmc.Sobol(d, scramble=True, seed=rng.integers(0, 2**31))

    # Generate samples
    if m <= 1:
        return sampler.random(m)

    # For Sobol sequences, it's better to use powers of 2
    # But we'll generate exactly m samples as requested
    samples = sampler.random(m)
    return samples


def sample_uniform_grid(rng: np.random.Generator, m: int, d: int) -> np.ndarray:
    """Generate grid-based samples in [0, 1]^d."""
    if d == 1:
        # 1D case: simple linear grid
        return np.linspace(0.01, 0.99, m).reshape(-1, 1)
    elif d == 2:
        # 2D case: create rectangular grid
        n_per_dim = int(np.ceil(m ** (1.0 / d)))
        x = np.linspace(0.01, 0.99, n_per_dim)
        y = np.linspace(0.01, 0.99, n_per_dim)
        xx, yy = np.meshgrid(x, y)
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])

        # If we have more points than needed, randomly subsample
        if len(grid_points) > m:
            indices = rng.choice(len(grid_points), size=m, replace=False)
            return grid_points[indices]
        elif len(grid_points) < m:
            # If we need more points, add some random ones
            additional_needed = m - len(grid_points)
            additional_points = rng.random((additional_needed, d)) * 0.98 + 0.01
            return np.vstack([grid_points, additional_points])
        else:
            return grid_points
    else:
        # Higher dimensions: use Latin Hypercube Sampling
        sampler = qmc.LatinHypercube(d, scramble=True, seed=rng.integers(0, 2**31))
        samples = sampler.random(m)
        # Scale to avoid boundary issues
        return samples * 0.98 + 0.01


def transform_to_ball(sampling_method: str, U: np.ndarray, d: int) -> np.ndarray:
    """Transform uniform samples to unit ball using structured approach."""
    m = U.shape[0]

    if sampling_method == SamplingMethod.GRID.value and d <= 3:
        # For grid sampling in low dimensions, create more structured ball sampling
        if d == 1:
            # 1D: just use the uniform samples as radii
            radii = U[:, 0] ** (1.0 / d)
            return np.column_stack([radii * (2 * (np.arange(m) % 2) - 1)])
        elif d == 2:
            # 2D: use polar coordinates with structured angles
            radii = U[:, 0] ** (1.0 / d)
            angles = U[:, 1] * 2 * np.pi
            return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        elif d == 3:
            # 3D: use spherical coordinates
            radii = U[:, 0] ** (1.0 / d)
            theta = U[:, 1] * 2 * np.pi  # azimuthal angle
            phi = np.arccos(1 - 2 * U[:, 2])  # polar angle
            x = radii * np.sin(phi) * np.cos(theta)
            y = radii * np.sin(phi) * np.sin(theta)
            z = radii * np.cos(phi)
            return np.column_stack([x, y, z])

    # Default approach: use Box-Muller-like transformation
    # Generate points on unit sphere, then scale by radius
    if d == 1:
        # 1D case
        radii = U[:, 0] ** (1.0 / d)
        directions = 2 * (U[:, 0] > 0.5) - 1  # random ±1
        return radii[:, None] * directions[:, None]
    else:
        # Use the first d columns of U for direction, last for radius
        if U.shape[1] < d + 1:
            # If we don't have enough dimensions, repeat the process
            U_extended = np.tile(U, (1, (d + 1) // U.shape[1] + 1))[:, : d + 1]
        else:
            U_extended = U[:, : d + 1]

        # Transform uniform to normal for direction
        # Use Box-Muller for pairs, handle odd dimensions
        Z = np.zeros((m, d))
        for i in range(0, d, 2):
            if i + 1 < d:
                # Box-Muller for pairs
                u1, u2 = U_extended[:, i], U_extended[:, i + 1]
                r = np.sqrt(-2 * np.log(np.maximum(u1, 1e-10)))
                theta = 2 * np.pi * u2
                Z[:, i] = r * np.cos(theta)
                Z[:, i + 1] = r * np.sin(theta)
            else:
                # Handle odd dimension
                u = U_extended[:, i]
                Z[:, i] = np.sqrt(-2 * np.log(np.maximum(u, 1e-10))) * np.cos(
                    2 * np.pi * U_extended[:, (i + 1) % d]
                )

        # Normalize to unit sphere
        Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12

        # Scale by radius
        radii = U_extended[:, -1] ** (1.0 / d)
        return Z * radii[:, None]


def transform_to_beta(U: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Transform uniform samples to Beta distribution using inverse CDF."""
    from scipy.special import betaincinv

    m, d = U.shape
    Y = np.empty((m, d))
    for j in range(d):
        Y[:, j] = betaincinv(alpha[j], beta[j], U[:, j])
    return Y


def generate_unit_hypercube_grid_nodes(n_measures: int) -> list[np.ndarray]:
    grid_nodes = []
    for i in range(1, 2**n_measures):
        binary_vector = [(i >> j) & 1 for j in range(n_measures)]
        grid_nodes.append(np.array(binary_vector, dtype=np.float64))
    return grid_nodes

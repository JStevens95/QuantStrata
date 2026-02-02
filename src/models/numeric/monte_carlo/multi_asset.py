"""
Multi-Asset Monte Carlo Simulation.

This module provides correlated asset simulation using Geometric Brownian Motion
for pricing multi-asset derivatives (baskets, spreads, rainbow options).

Key Components
--------------
- CorrelationMatrix: Validated correlation structure with Cholesky decomposition
- MultiAssetGBM: Multi-dimensional GBM simulation

Mathematical Framework
----------------------
For n correlated assets under the risk-neutral measure:

    dS_i / S_i = (r - q_i) dt + σ_i dW_i^Q

Where the Brownian motions satisfy:
    dW_i · dW_j = ρ_ij dt

Simulation uses Cholesky decomposition: if Σ = LL^T, then
    Z_correlated = L · Z_independent

References
----------
- Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering.
- Hull, J.C. (2018). Options, Futures, and Other Derivatives.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional
from src.models.numeric.monte_carlo.rng import NormalRng


# =============================================================================
# Correlation Matrix
# =============================================================================

@dataclass(slots=True)
class CorrelationMatrix:
    """
    Validated correlation matrix with Cholesky decomposition.

    Parameters
    ----------
    matrix : np.ndarray
        Correlation matrix, shape (n, n). Must be symmetric, positive
        semi-definite, with ones on diagonal.

    Attributes
    ----------
    n_assets : int
        Number of assets.
    cholesky : np.ndarray
        Lower triangular Cholesky factor L where Σ = LL^T.

    Raises
    ------
    ValueError
        If matrix is not valid (not symmetric, not PSD, diagonal not 1).
    """

    matrix: np.ndarray
    _cholesky: np.ndarray = None

    def __post_init__(self):
        self._validate()
        self._cholesky = np.linalg.cholesky(self.matrix)

    def _validate(self):
        """Validate correlation matrix properties."""
        if self.matrix.ndim != 2:
            raise ValueError("Correlation matrix must be 2-dimensional.")

        n, m = self.matrix.shape
        if n != m:
            raise ValueError("Correlation matrix must be square.")

        if n < 2:
            raise ValueError("Correlation matrix must have at least 2 assets.")

        if not np.allclose(self.matrix, self.matrix.T, atol=1e-10):
            raise ValueError("Correlation matrix must be symmetric.")

        if not np.allclose(np.diag(self.matrix), 1.0, atol=1e-10):
            raise ValueError("Correlation matrix diagonal must be 1.")

        if np.any(self.matrix < -1 - 1e-10) or np.any(self.matrix > 1 + 1e-10):
            raise ValueError("Correlation values must be in [-1, 1].")

        eigenvalues = np.linalg.eigvalsh(self.matrix)
        if np.any(eigenvalues < -1e-10):
            raise ValueError("Correlation matrix must be positive semi-definite.")

    @property
    def n_assets(self) -> int:
        """Number of assets."""
        return self.matrix.shape[0]

    @property
    def cholesky(self) -> np.ndarray:
        """Cholesky factor L where Σ = LL^T."""
        return self._cholesky

    @classmethod
    def from_flat(cls, rho: float, n: int) -> "CorrelationMatrix":
        """
        Create flat correlation matrix where all off-diagonal elements are ρ.

        Parameters
        ----------
        rho : float
            Pairwise correlation, must be in [-(1/(n-1)), 1] for PSD.
        n : int
            Number of assets.

        Returns
        -------
        CorrelationMatrix
            Flat correlation structure.
        """
        if n < 2:
            raise ValueError("n must be at least 2.")

        min_rho = -1 / (n - 1)
        if rho < min_rho - 1e-10:
            raise ValueError(f"For {n} assets, rho must be >= {min_rho:.4f} for PSD matrix.")

        matrix = np.full((n, n), rho)
        np.fill_diagonal(matrix, 1.0)
        return cls(matrix)

    @classmethod
    def from_pairs(cls, correlations: dict[tuple[int, int], float], n: int) -> "CorrelationMatrix":
        """
        Create correlation matrix from pairwise correlations.

        Parameters
        ----------
        correlations : dict
            Dictionary mapping (i, j) pairs to correlation values.
            Missing pairs default to 0.
        n : int
            Number of assets.

        Returns
        -------
        CorrelationMatrix
            Correlation structure from pairs.
        """
        matrix = np.eye(n)
        for (i, j), rho in correlations.items():
            if not (0 <= i < n and 0 <= j < n):
                raise ValueError(f"Invalid index pair ({i}, {j}) for {n} assets.")
            matrix[i, j] = rho
            matrix[j, i] = rho
        return cls(matrix)


# =============================================================================
# Multi-Asset Simulation Result
# =============================================================================

@dataclass(frozen=True, slots=True)
class MultiAssetSimulation:
    """
    Container for multi-asset simulation results.

    Attributes
    ----------
    spots : np.ndarray
        Simulated spot paths, shape (n_paths, n_steps + 1, n_assets).
        spots[i, t, k] is path i, time t, asset k.
    correlation : CorrelationMatrix
        Correlation structure used.
    dt : float
        Time step size.
    """

    spots: np.ndarray
    correlation: CorrelationMatrix
    dt: float

    @property
    def n_paths(self) -> int:
        """Number of simulated paths."""
        return self.spots.shape[0]

    @property
    def n_steps(self) -> int:
        """Number of time steps."""
        return self.spots.shape[1] - 1

    @property
    def n_assets(self) -> int:
        """Number of assets."""
        return self.spots.shape[2]

    @property
    def terminal_spots(self) -> np.ndarray:
        """Terminal spot prices, shape (n_paths, n_assets)."""
        return self.spots[:, -1, :]


# =============================================================================
# Multi-Asset GBM
# =============================================================================

@dataclass(frozen=True, slots=True)
class MultiAssetGBM:
    """
    Multi-dimensional Geometric Brownian Motion simulator.

    Parameters
    ----------
    spots : np.ndarray
        Initial spot prices, shape (n_assets,).
    r : float
        Risk-free rate.
    dividends : np.ndarray
        Dividend yields, shape (n_assets,).
    volatilities : np.ndarray
        Volatilities, shape (n_assets,).
    correlation : CorrelationMatrix
        Correlation structure.
    """

    spots: np.ndarray
    r: float
    dividends: np.ndarray
    volatilities: np.ndarray
    correlation: CorrelationMatrix

    def __post_init__(self):
        n = len(self.spots)
        if len(self.dividends) != n:
            raise ValueError("dividends must have same length as spots.")
        if len(self.volatilities) != n:
            raise ValueError("volatilities must have same length as spots.")
        if self.correlation.n_assets != n:
            raise ValueError("correlation must have same dimension as spots.")
        if np.any(self.spots <= 0):
            raise ValueError("spots must be positive.")
        if np.any(self.volatilities < 0):
            raise ValueError("volatilities must be non-negative.")

    @property
    def n_assets(self) -> int:
        """Number of assets."""
        return len(self.spots)

    def simulate(
        self,
        maturity: float,
        n_paths: int,
        n_steps: int,
        seed: Optional[int] = None,
        antithetic: bool = True,
    ) -> MultiAssetSimulation:
        """
        Simulate correlated GBM paths.

        Parameters
        ----------
        maturity : float
            Time horizon in years.
        n_paths : int
            Number of paths to simulate.
        n_steps : int
            Number of time steps.
        seed : int, optional
            Random seed.
        antithetic : bool
            Use antithetic variates.

        Returns
        -------
        MultiAssetSimulation
            Simulation results.
        """
        dt = maturity / n_steps
        n_assets = self.n_assets

        rng = NormalRng(seed=seed)

        if antithetic:
            half_paths = (n_paths + 1) // 2
            Z_half = rng.standard_normals(half_paths * n_steps, n_assets)
            Z_half = Z_half.reshape(half_paths, n_steps, n_assets)
            Z = np.concatenate([Z_half, -Z_half], axis=0)[:n_paths]
        else:
            Z = rng.standard_normals(n_paths * n_steps, n_assets)
            Z = Z.reshape(n_paths, n_steps, n_assets)

        L = self.correlation.cholesky
        Z_corr = Z @ L.T

        paths = np.zeros((n_paths, n_steps + 1, n_assets))
        paths[:, 0, :] = self.spots

        drifts = (self.r - self.dividends - 0.5 * self.volatilities ** 2) * dt
        diffusions = self.volatilities * np.sqrt(dt)

        for t in range(n_steps):
            log_increments = drifts + diffusions * Z_corr[:, t, :]
            paths[:, t + 1, :] = paths[:, t, :] * np.exp(log_increments)

        return MultiAssetSimulation(
            spots=paths,
            correlation=self.correlation,
            dt=dt,
        )

    def simulate_terminal(
        self,
        maturity: float,
        n_paths: int,
        seed: Optional[int] = None,
        antithetic: bool = True,
    ) -> np.ndarray:
        """
        Simulate terminal spots only (more efficient for European options).

        Parameters
        ----------
        maturity : float
            Time to maturity.
        n_paths : int
            Number of paths.
        seed : int, optional
            Random seed.
        antithetic : bool
            Use antithetic variates.

        Returns
        -------
        np.ndarray
            Terminal spots, shape (n_paths, n_assets).
        """
        n_assets = self.n_assets
        rng = NormalRng(seed=seed)

        if antithetic:
            half_paths = (n_paths + 1) // 2
            Z_half = rng.standard_normals(half_paths, n_assets)
            Z = np.concatenate([Z_half, -Z_half], axis=0)[:n_paths]
        else:
            Z = rng.standard_normals(n_paths, n_assets)

        L = self.correlation.cholesky
        Z_corr = Z @ L.T

        drifts = (self.r - self.dividends - 0.5 * self.volatilities ** 2) * maturity
        diffusions = self.volatilities * np.sqrt(maturity)

        terminals = self.spots * np.exp(drifts + diffusions * Z_corr)
        return terminals

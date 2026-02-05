"""
Factor model adapter for time series generation.

Mathematical Model
------------------
The factor follows OU dynamics:
    dF_t = κ(θ - F_t) dt + σ dW_t

The factor is then transformed via loadings to drive curve/surface shifts:
    ΔR(τ) = λ(τ) × F_t

where λ(τ) is the loading at tenor τ.

Applications
------------
- Yield curve factors: Level (parallel), Slope (twist), Curvature (butterfly)
- Vol surface factors: ATM level, Skew, Smile

PCA-Based Factor Models
-----------------------
For a yield curve with observed rates R(τ₁), ..., R(τₙ):
1. Compute covariance matrix of rate changes ΔR
2. Perform PCA to get eigenvectors (loadings) and eigenvalues
3. First 3 PCs typically explain >95% of variance
4. PC1 ≈ Level (parallel shift), PC2 ≈ Slope (steepening), PC3 ≈ Curvature (butterfly)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict

from src.marketdata.scenarios.timeseries.config import FactorDynamicsSpec


@dataclass(frozen=True, slots=True)
class FactorAdapter:
    """
    Adapter for factor model dynamics.

    The factor itself follows OU dynamics. The loadings dictionary
    specifies how the factor maps to different tenors/points.

    Parameters
    ----------
    spec : FactorDynamicsSpec
        Factor dynamics specification with mean, kappa, vol, and loadings.

    Examples
    --------
    >>> from src.marketdata.scenarios.timeseries.config import FactorDynamicsSpec
    >>>
    >>> # Level factor (parallel shift)
    >>> level_spec = FactorDynamicsSpec(
    ...     mean=0.0,
    ...     kappa=0.1,
    ...     vol=0.005,
    ...     loadings={"1Y": 1.0, "2Y": 1.0, "5Y": 1.0, "10Y": 1.0, "30Y": 1.0},
    ... )
    >>> level_adapter = FactorAdapter(spec=level_spec)
    >>>
    >>> # Slope factor (2s10s steepener)
    >>> slope_spec = FactorDynamicsSpec(
    ...     mean=0.0,
    ...     kappa=0.2,
    ...     vol=0.002,
    ...     loadings={"1Y": -0.5, "2Y": -0.3, "5Y": 0.0, "10Y": 0.3, "30Y": 0.5},
    ... )
    >>> slope_adapter = FactorAdapter(spec=slope_spec)
    >>>
    >>> # Simulate factor paths
    >>> shocks = np.random.standard_normal((252, 1000))
    >>> factor_paths = level_adapter.simulate(
    ...     initial_value=0.0,
    ...     n_time=252,
    ...     n_scenarios=1000,
    ...     shocks=shocks,
    ...     dt=1/252,
    ... )
    >>>
    >>> # Get rate shift at 5Y for time=100, scenario=0
    >>> rate_shift_5y = level_adapter.apply_loading("5Y", factor_paths[100, 0])
    """

    spec: FactorDynamicsSpec

    @property
    def requires_variance_paths(self) -> bool:
        """Factor model does not produce variance paths."""
        return False

    @property
    def loadings(self) -> Dict[str, float]:
        """Factor loadings dictionary."""
        return self.spec.loadings

    def simulate(
        self,
        initial_value: float,
        n_time: int,
        n_scenarios: int,
        shocks: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Simulate factor paths using exact OU discretization.

        Parameters
        ----------
        initial_value : float
            Starting factor value F_0 (typically 0 for shock factors).
        n_time : int
            Number of time steps.
        n_scenarios : int
            Number of scenarios.
        shocks : np.ndarray
            Standard normal shocks, shape (n_time, n_scenarios).
        dt : float
            Time step in years.

        Returns
        -------
        np.ndarray
            Simulated factor paths, shape (n_time + 1, n_scenarios).
        """
        # Validate shocks shape
        if shocks.shape != (n_time, n_scenarios):
            raise ValueError(
                f"shocks shape {shocks.shape} doesn't match "
                f"(n_time={n_time}, n_scenarios={n_scenarios})"
            )

        # Extract parameters (same as OU)
        theta = float(self.spec.mean)
        kappa = float(self.spec.kappa)
        sigma = float(self.spec.vol)

        # Pre-compute exact OU transition coefficients
        exp_neg_kappa_dt = np.exp(-kappa * dt)
        variance_coeff = (sigma * sigma) * (1.0 - np.exp(-2.0 * kappa * dt)) / (2.0 * kappa)
        sigma_eff = np.sqrt(variance_coeff)

        # Allocate paths array
        paths = np.empty((n_time + 1, n_scenarios), dtype=np.float64)
        paths[0, :] = initial_value

        # Simulate using exact OU formula
        for t in range(n_time):
            mean_next = theta + (paths[t, :] - theta) * exp_neg_kappa_dt
            paths[t + 1, :] = mean_next + sigma_eff * shocks[t, :]

        return paths

    def simulate_with_variance(
        self,
        initial_value: float,
        n_time: int,
        n_scenarios: int,
        shocks: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate factor paths (variance is N/A, returned as NaN).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (paths, variance_paths filled with NaN).
        """
        paths = self.simulate(initial_value, n_time, n_scenarios, shocks, dt)
        variance = np.full_like(paths, np.nan)
        return paths, variance

    def apply_loading(self, tenor: str, factor_value: float) -> float:
        """
        Apply factor loading to get rate/vol shift at a specific tenor.

        Parameters
        ----------
        tenor : str
            Tenor key (e.g., "5Y").
        factor_value : float
            Current factor value.

        Returns
        -------
        float
            Shift at the specified tenor: λ(tenor) × factor_value.

        Raises
        ------
        KeyError
            If tenor is not in loadings dictionary.
        """
        if tenor not in self.loadings:
            raise KeyError(f"Tenor '{tenor}' not in factor loadings: {list(self.loadings.keys())}")
        return self.loadings[tenor] * factor_value

    def apply_loadings_all(self, factor_value: float) -> Dict[str, float]:
        """
        Apply factor loadings to get all rate/vol shifts.

        Parameters
        ----------
        factor_value : float
            Current factor value.

        Returns
        -------
        Dict[str, float]
            Shifts at all tenors: {tenor: λ(tenor) × factor_value}.
        """
        return {tenor: loading * factor_value for tenor, loading in self.loadings.items()}

    def apply_loadings_array(
        self,
        factor_paths: np.ndarray,
        tenors: list[str],
    ) -> np.ndarray:
        """
        Apply factor loadings to get rate/vol shifts for multiple tenors.

        Parameters
        ----------
        factor_paths : np.ndarray
            Factor paths, shape (n_time + 1, n_scenarios).
        tenors : list[str]
            List of tenors to compute shifts for.

        Returns
        -------
        np.ndarray
            Shifts, shape (n_time + 1, n_scenarios, n_tenors).
        """
        n_time_plus_1, n_scenarios = factor_paths.shape
        n_tenors = len(tenors)

        shifts = np.empty((n_time_plus_1, n_scenarios, n_tenors), dtype=np.float64)

        for i, tenor in enumerate(tenors):
            if tenor not in self.loadings:
                raise KeyError(f"Tenor '{tenor}' not in factor loadings.")
            shifts[:, :, i] = self.loadings[tenor] * factor_paths

        return shifts

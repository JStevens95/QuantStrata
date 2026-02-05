"""
Ornstein-Uhlenbeck adapter for time series generation.

Mathematical Model
------------------
dX_t = κ(θ - X_t) dt + σ dW_t

Exact discretization:
    X_{t+dt} = θ + (X_t - θ)e^(-κ dt) + σ√((1 - e^(-2κ dt))/(2κ)) Z

where Z ~ N(0, 1) is a standard normal shock.

Properties
----------
- Mean: E[X_t] = θ + (X_0 - θ)e^(-κt)
- Variance: Var(X_t) = σ²(1 - e^(-2κt))/(2κ)
- Stationary variance: σ²/(2κ)
- Half-life: ln(2)/κ

Applications
------------
- Interest rate levels (Vasicek model for short rate)
- Credit spreads
- Volatility factors
- Basis/spread trading
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from src.marketdata.scenarios.timeseries.config import OUDynamicsSpec


@dataclass(frozen=True, slots=True)
class OUAdapter:
    """
    Adapter for Ornstein-Uhlenbeck (mean-reverting) dynamics.

    Parameters
    ----------
    spec : OUDynamicsSpec
        OU dynamics specification with mean, kappa, and vol.

    Examples
    --------
    >>> from src.marketdata.scenarios.timeseries.config import OUDynamicsSpec
    >>>
    >>> # Rate level with 5% mean, 50% annual mean reversion
    >>> spec = OUDynamicsSpec(mean=0.05, kappa=0.5, vol=0.005)
    >>> adapter = OUAdapter(spec=spec)
    >>>
    >>> shocks = np.random.standard_normal((252, 1000))
    >>> paths = adapter.simulate(
    ...     initial_value=0.05,
    ...     n_time=252,
    ...     n_scenarios=1000,
    ...     shocks=shocks,
    ...     dt=1/252,
    ... )
    """

    spec: OUDynamicsSpec

    @property
    def requires_variance_paths(self) -> bool:
        """OU does not produce variance paths."""
        return False

    def simulate(
        self,
        initial_value: float,
        n_time: int,
        n_scenarios: int,
        shocks: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Simulate OU paths using exact discretization.

        Parameters
        ----------
        initial_value : float
            Starting value X_0.
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
            Simulated paths, shape (n_time + 1, n_scenarios).

        Notes
        -----
        Uses exact OU discretization:
            X_{t+dt} = θ + (X_t - θ) × e^(-κ dt) + σ_eff × Z

        where σ_eff = σ × √((1 - e^(-2κ dt))/(2κ))
        """
        # Validate shocks shape
        if shocks.shape != (n_time, n_scenarios):
            raise ValueError(
                f"shocks shape {shocks.shape} doesn't match "
                f"(n_time={n_time}, n_scenarios={n_scenarios})"
            )

        # Extract parameters
        theta = float(self.spec.mean)
        kappa = float(self.spec.kappa)
        sigma = float(self.spec.vol)

        # Pre-compute exact OU transition coefficients
        exp_neg_kappa_dt = np.exp(-kappa * dt)
        # Variance of increment: σ²(1 - e^(-2κdt))/(2κ)
        variance_coeff = (sigma * sigma) * (1.0 - np.exp(-2.0 * kappa * dt)) / (2.0 * kappa)
        sigma_eff = np.sqrt(variance_coeff)

        # Allocate paths array
        paths = np.empty((n_time + 1, n_scenarios), dtype=np.float64)
        paths[0, :] = initial_value

        # Simulate using exact OU formula
        for t in range(n_time):
            # X_{t+1} = θ + (X_t - θ) × e^(-κdt) + σ_eff × Z
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
        Simulate OU paths (variance is N/A, returned as NaN).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (paths, variance_paths filled with NaN).
        """
        paths = self.simulate(initial_value, n_time, n_scenarios, shocks, dt)
        variance = np.full_like(paths, np.nan)
        return paths, variance

    def expected_value(self, initial_value: float, t: float) -> float:
        """
        Expected value E[X_t] at time t.

        E[X_t] = θ + (X_0 - θ) × e^(-κt)
        """
        theta = self.spec.mean
        kappa = self.spec.kappa
        return theta + (initial_value - theta) * np.exp(-kappa * t)

    def variance_at(self, t: float) -> float:
        """
        Variance Var(X_t) at time t.

        Var(X_t) = σ²(1 - e^(-2κt))/(2κ)
        """
        kappa = self.spec.kappa
        sigma = self.spec.vol
        return (sigma * sigma) * (1.0 - np.exp(-2.0 * kappa * t)) / (2.0 * kappa)

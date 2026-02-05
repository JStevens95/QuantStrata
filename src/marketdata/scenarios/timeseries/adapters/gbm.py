"""
Geometric Brownian Motion adapter for time series generation.

Mathematical Model
------------------
dS_t = μ S_t dt + σ S_t dW_t

Exact discretization:
    S_{t+dt} = S_t × exp((μ - σ²/2)dt + σ√dt Z)

where Z ~ N(0, 1) is a standard normal shock.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from src.marketdata.scenarios.timeseries.config import GBMDynamicsSpec


@dataclass(frozen=True, slots=True)
class GBMAdapter:
    """
    Adapter for Geometric Brownian Motion dynamics.

    Parameters
    ----------
    spec : GBMDynamicsSpec
        GBM dynamics specification with drift and vol.

    Examples
    --------
    >>> from src.marketdata.scenarios.timeseries.config import GBMDynamicsSpec
    >>>
    >>> spec = GBMDynamicsSpec(drift=0.0, vol=0.08)
    >>> adapter = GBMAdapter(spec=spec)
    >>>
    >>> # Generate 252 daily steps for 1000 scenarios
    >>> shocks = np.random.standard_normal((252, 1000))
    >>> paths = adapter.simulate(
    ...     initial_value=1.08,
    ...     n_time=252,
    ...     n_scenarios=1000,
    ...     shocks=shocks,
    ...     dt=1/252,
    ... )
    >>> paths.shape
    (253, 1000)
    """

    spec: GBMDynamicsSpec

    @property
    def requires_variance_paths(self) -> bool:
        """GBM does not produce variance paths."""
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
        Simulate GBM paths using exact discretization.

        Parameters
        ----------
        initial_value : float
            Starting spot S_0.
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
            Simulated spot paths, shape (n_time + 1, n_scenarios).
        """
        # Validate shocks shape
        if shocks.shape != (n_time, n_scenarios):
            raise ValueError(
                f"shocks shape {shocks.shape} doesn't match "
                f"(n_time={n_time}, n_scenarios={n_scenarios})"
            )

        # Extract parameters
        mu = float(self.spec.drift)
        sigma = float(self.spec.vol)
        sqrt_dt = np.sqrt(dt)

        # Pre-compute the drift-adjusted increment
        # log(S_{t+dt}/S_t) = (μ - σ²/2)dt + σ√dt Z
        drift_adj = (mu - 0.5 * sigma * sigma) * dt
        diffusion = sigma * sqrt_dt

        # Allocate paths array
        paths = np.empty((n_time + 1, n_scenarios), dtype=np.float64)
        paths[0, :] = initial_value

        # Simulate using exact GBM formula
        for t in range(n_time):
            log_return = drift_adj + diffusion * shocks[t, :]
            paths[t + 1, :] = paths[t, :] * np.exp(log_return)

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
        Simulate GBM paths (variance is constant, returned as NaN).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (spot_paths, variance_paths filled with NaN).
        """
        paths = self.simulate(initial_value, n_time, n_scenarios, shocks, dt)
        variance = np.full_like(paths, np.nan)
        return paths, variance

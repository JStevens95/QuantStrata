"""
Heston stochastic volatility adapter for time series generation.

Mathematical Model
------------------
dS_t = μ S_t dt + √V_t S_t dW_t^S
dV_t = κ(θ - V_t) dt + ξ √V_t dW_t^V
Corr(dW_t^S, dW_t^V) = ρ

This adapter handles the internal spot-vol correlation (ρ) separately from
the cross-factor correlation handled by TimeseriesGenerator.

The adapter receives a single shock stream (for spot) and generates its own
correlated variance shocks internally.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Literal

from src.marketdata.scenarios.timeseries.config import HestonDynamicsSpec


HestonScheme = Literal["euler", "full_truncation", "reflection"]


@dataclass(frozen=True, slots=True)
class HestonAdapter:
    """
    Adapter for Heston stochastic volatility dynamics.

    Parameters
    ----------
    spec : HestonDynamicsSpec
        Heston dynamics specification.
    scheme : HestonScheme
        Discretization scheme for variance process.
        - "euler": Simple Euler (may go negative).
        - "full_truncation": max(V, 0) truncation (recommended).
        - "reflection": |V| reflection.
    rng_seed_offset : int
        Offset added to scenario index for variance RNG.
        Ensures variance shocks are consistent but different from spot shocks.

    Notes
    -----
    The adapter uses the input shocks for the spot process (dW^S) and
    generates internal shocks for variance (dW^V) using:
        dW^V = ρ dW^S + √(1-ρ²) dZ

    where dZ is an independent standard normal.

    Examples
    --------
    >>> from src.marketdata.scenarios.timeseries.config import HestonDynamicsSpec
    >>>
    >>> spec = HestonDynamicsSpec(
    ...     drift=0.03,
    ...     kappa=2.0,
    ...     theta=0.04,
    ...     xi=0.3,
    ...     v0=0.04,
    ...     rho_internal=-0.7,
    ... )
    >>> adapter = HestonAdapter(spec=spec, scheme="full_truncation")
    >>>
    >>> shocks = np.random.standard_normal((252, 1000))
    >>> spot_paths, var_paths = adapter.simulate_with_variance(
    ...     initial_value=100.0,
    ...     n_time=252,
    ...     n_scenarios=1000,
    ...     shocks=shocks,
    ...     dt=1/252,
    ... )
    """

    spec: HestonDynamicsSpec
    scheme: HestonScheme = "full_truncation"
    rng_seed_offset: int = 123456789

    @property
    def requires_variance_paths(self) -> bool:
        """Heston produces variance paths."""
        return True

    def simulate(
        self,
        initial_value: float,
        n_time: int,
        n_scenarios: int,
        shocks: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Simulate Heston spot paths (variance paths discarded).

        Parameters
        ----------
        initial_value : float
            Starting spot S_0.
        n_time : int
            Number of time steps.
        n_scenarios : int
            Number of scenarios.
        shocks : np.ndarray
            Standard normal shocks for spot, shape (n_time, n_scenarios).
        dt : float
            Time step in years.

        Returns
        -------
        np.ndarray
            Simulated spot paths, shape (n_time + 1, n_scenarios).
        """
        spot_paths, _ = self.simulate_with_variance(
            initial_value, n_time, n_scenarios, shocks, dt
        )
        return spot_paths

    def simulate_with_variance(
        self,
        initial_value: float,
        n_time: int,
        n_scenarios: int,
        shocks: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate Heston spot and variance paths.

        Parameters
        ----------
        initial_value : float
            Starting spot S_0.
        n_time : int
            Number of time steps.
        n_scenarios : int
            Number of scenarios.
        shocks : np.ndarray
            Standard normal shocks for spot (dW^S), shape (n_time, n_scenarios).
        dt : float
            Time step in years.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (spot_paths, variance_paths), each shape (n_time + 1, n_scenarios).
        """
        # Validate shocks shape
        if shocks.shape != (n_time, n_scenarios):
            raise ValueError(
                f"shocks shape {shocks.shape} doesn't match "
                f"(n_time={n_time}, n_scenarios={n_scenarios})"
            )

        # Extract parameters
        mu = float(self.spec.drift)
        kappa = float(self.spec.kappa)
        theta = float(self.spec.theta)
        xi = float(self.spec.xi)
        v0 = float(self.spec.v0)
        rho = float(self.spec.rho_internal)

        sqrt_dt = np.sqrt(dt)
        sqrt_1_rho2 = np.sqrt(1.0 - rho * rho)

        # Generate independent shocks for variance process
        rng = np.random.default_rng(seed=self.rng_seed_offset)
        z_independent = rng.standard_normal((n_time, n_scenarios))

        # Compute correlated variance shocks: dW^V = ρ dW^S + √(1-ρ²) dZ
        dW_S = sqrt_dt * shocks
        dW_V = sqrt_dt * (rho * shocks + sqrt_1_rho2 * z_independent)

        # Allocate paths
        S = np.empty((n_time + 1, n_scenarios), dtype=np.float64)
        V = np.empty((n_time + 1, n_scenarios), dtype=np.float64)
        S[0, :] = initial_value
        V[0, :] = v0

        # Simulate step by step
        for t in range(n_time):
            V_curr = V[t, :]
            S_curr = S[t, :]

            # Variance step (using selected scheme)
            V_next = self._variance_step(V_curr, kappa, theta, xi, dt, dW_V[t, :])

            # Spot step (log-Euler for positivity)
            sqrt_V = np.sqrt(np.maximum(V_curr, 0.0))
            log_increment = (mu - 0.5 * V_curr) * dt + sqrt_V * dW_S[t, :]
            S_next = S_curr * np.exp(log_increment)

            V[t + 1, :] = V_next
            S[t + 1, :] = S_next

        return S, V

    def _variance_step(
        self,
        V: np.ndarray,
        kappa: float,
        theta: float,
        xi: float,
        dt: float,
        dW: np.ndarray,
    ) -> np.ndarray:
        """Apply variance discretization scheme."""
        if self.scheme == "euler":
            return self._euler_step(V, kappa, theta, xi, dt, dW)
        elif self.scheme == "full_truncation":
            return self._full_truncation_step(V, kappa, theta, xi, dt, dW)
        elif self.scheme == "reflection":
            return self._reflection_step(V, kappa, theta, xi, dt, dW)
        else:
            raise ValueError(f"Unknown scheme: {self.scheme}")

    @staticmethod
    def _euler_step(
        V: np.ndarray,
        kappa: float,
        theta: float,
        xi: float,
        dt: float,
        dW: np.ndarray,
    ) -> np.ndarray:
        """Euler step (may go negative)."""
        sqrt_V = np.sqrt(np.maximum(V, 0.0))
        return V + kappa * (theta - V) * dt + xi * sqrt_V * dW

    @staticmethod
    def _full_truncation_step(
        V: np.ndarray,
        kappa: float,
        theta: float,
        xi: float,
        dt: float,
        dW: np.ndarray,
    ) -> np.ndarray:
        """Full truncation scheme: max(V, 0) in diffusion and result."""
        V_pos = np.maximum(V, 0.0)
        sqrt_V = np.sqrt(V_pos)
        V_next = V + kappa * (theta - V_pos) * dt + xi * sqrt_V * dW
        return np.maximum(V_next, 0.0)

    @staticmethod
    def _reflection_step(
        V: np.ndarray,
        kappa: float,
        theta: float,
        xi: float,
        dt: float,
        dW: np.ndarray,
    ) -> np.ndarray:
        """Reflection scheme: |V| if V < 0."""
        V_pos = np.maximum(V, 0.0)
        sqrt_V = np.sqrt(V_pos)
        V_next = V + kappa * (theta - V_pos) * dt + xi * sqrt_V * dW
        return np.abs(V_next)

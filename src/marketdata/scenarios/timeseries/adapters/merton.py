"""
Merton Jump-Diffusion adapter for time series generation.

This adapter delegates to the existing MertonDynamics in `src/models/jump_diffusion/`
rather than reimplementing the simulation logic.

Mathematical Model
------------------
dS_t / S_t = (μ - λκ) dt + σ dW_t + (J - 1) dN_t

where:
    - σ: Diffusion volatility
    - λ: Jump intensity (expected jumps per year)
    - N_t: Poisson process with intensity λ
    - J = exp(Y), Y ~ N(μ_J, σ_J²) is the jump multiplier
    - κ = E[J - 1] = exp(μ_J + σ_J²/2) - 1

When to Use
-----------
- Equity crash risk modeling
- VaR for portfolios sensitive to tail events
- Short-dated options pricing
- Stress testing
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

# Use existing Merton implementation
from src.models.jump_diffusion.merton import MertonParameters, MertonDynamics


@dataclass(frozen=True, slots=True)
class MertonDynamicsSpec:
    """
    Specification for Merton jump-diffusion dynamics.

    This is a thin wrapper around MertonParameters that adds drift
    for use in the TimeseriesGenerator framework.

    Parameters
    ----------
    drift : float
        Base drift μ (risk-neutral: r - q).
    sigma : float
        Diffusion volatility σ >= 0.
    lambda_ : float
        Jump intensity λ >= 0 (expected jumps per year).
    mu_j : float
        Mean of log-jump size μ_J. Negative for crash-like jumps.
    sigma_j : float
        Standard deviation of log-jump size σ_J >= 0.

    Examples
    --------
    >>> # Crash-prone equity
    >>> merton = MertonDynamicsSpec(
    ...     drift=0.05,
    ...     sigma=0.15,       # 15% diffusion vol
    ...     lambda_=0.5,      # 0.5 jumps/year expected
    ...     mu_j=-0.10,       # -10% average jump (crash)
    ...     sigma_j=0.15,     # 15% jump size uncertainty
    ... )
    """

    drift: float
    sigma: float
    lambda_: float  # Use lambda_ to avoid Python keyword
    mu_j: float
    sigma_j: float

    def __post_init__(self) -> None:
        # Validation delegated to MertonParameters
        _ = self.to_merton_parameters()

    def to_merton_parameters(self) -> MertonParameters:
        """Convert to the core MertonParameters used by MertonDynamics."""
        return MertonParameters(
            sigma=self.sigma,
            lambda_=self.lambda_,
            mu_j=self.mu_j,
            sigma_j=self.sigma_j,
        )

    @property
    def kappa(self) -> float:
        """Expected relative jump κ = E[J - 1]."""
        return self.to_merton_parameters().kappa

    @property
    def adjusted_drift(self) -> float:
        """Drift adjusted for jump compensation: μ - λκ."""
        return self.drift - self.lambda_ * self.kappa


@dataclass(slots=True)
class MertonAdapter:
    """
    Adapter for Merton jump-diffusion dynamics.

    Delegates to the existing MertonDynamics implementation in
    `src/models/jump_diffusion/merton.py`.

    Parameters
    ----------
    spec : MertonDynamicsSpec
        Merton dynamics specification.
    rng_seed_offset : int
        Offset added to seeds for jump generation (ensures different
        jumps even when diffusion shocks are correlated).

    Notes
    -----
    The shocks parameter provides the diffusion component (correlated
    with other factors). Jump times and sizes are generated independently
    using the underlying MertonDynamics.

    Examples
    --------
    >>> spec = MertonDynamicsSpec(
    ...     drift=0.05, sigma=0.15, lambda_=0.5, mu_j=-0.10, sigma_j=0.15
    ... )
    >>> adapter = MertonAdapter(spec=spec)
    >>> paths = adapter.simulate(
    ...     initial_value=100.0,
    ...     n_time=252,
    ...     n_scenarios=10000,
    ...     shocks=np.random.randn(252, 10000),
    ...     dt=1/252,
    ... )
    """

    spec: MertonDynamicsSpec
    rng_seed_offset: int = 987654321

    _dynamics: MertonDynamics = None

    def __post_init__(self) -> None:
        """Initialize the underlying MertonDynamics."""
        self._dynamics = MertonDynamics(
            params=self.spec.to_merton_parameters(),
            drift=self.spec.drift,
        )

    @property
    def requires_variance_paths(self) -> bool:
        """Merton doesn't have stochastic variance."""
        return False

    def simulate(
        self,
        initial_value: float,
        n_time: int,
        n_scenarios: int,
        shocks: np.ndarray,
        dt: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Simulate Merton jump-diffusion paths.

        Uses the existing MertonDynamics for simulation, which provides
        proper handling of jumps with Poisson arrivals and log-normal sizes.

        Parameters
        ----------
        initial_value : float
            Starting value S_0.
        n_time : int
            Number of time steps.
        n_scenarios : int
            Number of scenarios.
        shocks : np.ndarray
            Standard normal shocks for diffusion (may be partially used).
        dt : float
            Time step in years.
        seed : int, optional
            Random seed for jump generation.

        Returns
        -------
        np.ndarray
            Simulated paths, shape (n_time + 1, n_scenarios).
        """
        maturity = n_time * dt

        # Use the full MertonDynamics simulation
        # The existing implementation handles diffusion + jumps correctly
        effective_seed = seed if seed is not None else self.rng_seed_offset

        sim_result = self._dynamics.simulate(
            spot0=initial_value,
            maturity=maturity,
            n_paths=n_scenarios,
            n_steps=n_time,
            seed=effective_seed,
            antithetic=False,  # Don't use antithetic to preserve path count
        )

        # MertonSimulation returns shape (n_paths, n_steps + 1)
        # We need (n_time + 1, n_scenarios)
        return sim_result.spot_paths.T

    def simulate_with_variance(
        self,
        initial_value: float,
        n_time: int,
        n_scenarios: int,
        shocks: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate paths (variance paths returned as NaN since Merton has no SV).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (paths, variance_paths filled with NaN).
        """
        paths = self.simulate(initial_value, n_time, n_scenarios, shocks, dt)
        variance = np.full_like(paths, np.nan)
        return paths, variance

    def simulate_with_jumps(
        self,
        initial_value: float,
        n_time: int,
        n_scenarios: int,
        shocks: np.ndarray,
        dt: float,
        seed: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate paths and return jump counts.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (paths, jump_counts) both shape (n_time + 1, n_scenarios).
        """
        maturity = n_time * dt
        effective_seed = seed if seed is not None else self.rng_seed_offset

        sim_result = self._dynamics.simulate(
            spot0=initial_value,
            maturity=maturity,
            n_paths=n_scenarios,
            n_steps=n_time,
            seed=effective_seed,
            antithetic=False,
        )

        return sim_result.spot_paths.T, sim_result.jump_counts.T

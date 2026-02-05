"""
Merton Jump-Diffusion adapter for time series generation.

Mathematical Model
------------------
dS_t / S_t = (μ - λκ) dt + σ dW_t + (J - 1) dN_t

where:
    - σ: Diffusion volatility
    - λ: Jump intensity (expected jumps per year)
    - N_t: Poisson process with intensity λ
    - J = exp(Y), Y ~ N(μ_J, σ_J²) is the jump multiplier
    - κ = E[J - 1] = exp(μ_J + σ_J²/2) - 1

Properties
----------
- Fat tails: Jumps generate heavier tails than pure GBM
- Implied vol smile: Creates steep short-term smiles
- Jump clustering: Can model market crashes

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

from src.marketdata.scenarios.timeseries.config import GBMDynamicsSpec


@dataclass(frozen=True, slots=True)
class MertonDynamicsSpec:
    """
    Specification for Merton jump-diffusion dynamics.

    Mathematical Model
    ------------------
    dS_t / S_t = (μ - λκ) dt + σ dW_t + (J - 1) dN_t

    Parameters
    ----------
    drift : float
        Base drift μ (before jump adjustment).
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
        if not np.isfinite(self.drift):
            raise ValueError("MertonDynamicsSpec.drift must be finite.")
        if self.sigma < 0.0:
            raise ValueError("MertonDynamicsSpec.sigma must be >= 0.")
        if self.lambda_ < 0.0:
            raise ValueError("MertonDynamicsSpec.lambda_ must be >= 0.")
        if self.sigma_j < 0.0:
            raise ValueError("MertonDynamicsSpec.sigma_j must be >= 0.")

    @property
    def expected_jump(self) -> float:
        """Expected jump multiplier E[J] = exp(μ_J + σ_J²/2)."""
        return np.exp(self.mu_j + 0.5 * self.sigma_j ** 2)

    @property
    def kappa(self) -> float:
        """Expected relative jump κ = E[J - 1] = E[J] - 1."""
        return self.expected_jump - 1.0

    @property
    def compensated_drift(self) -> float:
        """Risk-neutral drift: μ - λκ."""
        return self.drift - self.lambda_ * self.kappa

    @property
    def total_variance(self) -> float:
        """
        Total instantaneous variance (diffusion + jump contribution).
        
        Var = σ² + λ(μ_J² + σ_J²)
        """
        jump_var = self.lambda_ * (self.mu_j ** 2 + self.sigma_j ** 2)
        return self.sigma ** 2 + jump_var


@dataclass(frozen=True, slots=True)
class MertonAdapter:
    """
    Adapter for Merton jump-diffusion dynamics.

    The adapter transforms Gaussian shocks into jump-diffusion paths.
    It internally generates Poisson jumps and log-normal jump sizes.

    Parameters
    ----------
    spec : MertonDynamicsSpec
        Merton dynamics specification.
    rng_seed_offset : int
        Offset for internal RNG (for jump generation).

    Examples
    --------
    >>> spec = MertonDynamicsSpec(
    ...     drift=0.05, sigma=0.15, lambda_=0.5, mu_j=-0.10, sigma_j=0.15
    ... )
    >>> adapter = MertonAdapter(spec=spec)
    >>>
    >>> shocks = np.random.standard_normal((252, 1000))
    >>> paths = adapter.simulate(
    ...     initial_value=100.0,
    ...     n_time=252,
    ...     n_scenarios=1000,
    ...     shocks=shocks,
    ...     dt=1/252,
    ... )
    """

    spec: MertonDynamicsSpec
    rng_seed_offset: int = 987654321

    @property
    def requires_variance_paths(self) -> bool:
        """Merton does not produce variance paths."""
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
        Simulate Merton jump-diffusion paths.

        Parameters
        ----------
        initial_value : float
            Starting spot S_0.
        n_time : int
            Number of time steps.
        n_scenarios : int
            Number of scenarios.
        shocks : np.ndarray
            Standard normal shocks for diffusion, shape (n_time, n_scenarios).
        dt : float
            Time step in years.

        Returns
        -------
        np.ndarray
            Simulated paths, shape (n_time + 1, n_scenarios).
        """
        # Validate shocks shape
        if shocks.shape != (n_time, n_scenarios):
            raise ValueError(
                f"shocks shape {shocks.shape} doesn't match "
                f"(n_time={n_time}, n_scenarios={n_scenarios})"
            )

        # Extract parameters
        mu = float(self.spec.drift)
        sigma = float(self.spec.sigma)
        lambda_ = float(self.spec.lambda_)
        mu_j = float(self.spec.mu_j)
        sigma_j = float(self.spec.sigma_j)
        kappa = float(self.spec.kappa)

        sqrt_dt = np.sqrt(dt)

        # Initialize RNG for jumps
        rng = np.random.default_rng(seed=self.rng_seed_offset)

        # Allocate paths
        paths = np.empty((n_time + 1, n_scenarios), dtype=np.float64)
        paths[0, :] = initial_value

        # Compensated drift
        drift_comp = (mu - lambda_ * kappa - 0.5 * sigma * sigma) * dt

        for t in range(n_time):
            # Diffusion component
            diffusion = sigma * sqrt_dt * shocks[t, :]

            # Jump component
            # Number of jumps in this time step (Poisson)
            n_jumps = rng.poisson(lam=lambda_ * dt, size=n_scenarios)

            # Total jump size for each path
            jump_returns = np.zeros(n_scenarios)
            for s in range(n_scenarios):
                if n_jumps[s] > 0:
                    # Sum of log-normal jumps
                    jump_sizes = rng.normal(mu_j, sigma_j, size=n_jumps[s])
                    jump_returns[s] = np.sum(jump_sizes)

            # Log return = drift + diffusion + jumps
            log_return = drift_comp + diffusion + jump_returns

            # Update paths
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
        Simulate Merton paths (variance is N/A, returned as NaN).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (paths, variance_paths filled with NaN).
        """
        paths = self.simulate(initial_value, n_time, n_scenarios, shocks, dt)
        variance = np.full_like(paths, np.nan)
        return paths, variance

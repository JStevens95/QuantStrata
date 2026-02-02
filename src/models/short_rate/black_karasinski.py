"""
Black-Karasinski One-Factor Short Rate Model Implementation.

This module provides the Black-Karasinski model for pricing interest rate derivatives
where the LOG of the short rate follows a mean-reverting Gaussian process.

Mathematical Framework
----------------------
The Black-Karasinski model specifies the short rate dynamics:

    d(ln r(t)) = [θ(t) - a·ln r(t)] dt + σ dW(t)

Or equivalently in terms of x(t) = ln r(t):

    dx(t) = [θ(t) - a·x(t)] dt + σ dW(t)

where:
    - r(t): instantaneous short rate (always positive: r(t) = exp(x(t)))
    - x(t) = ln r(t): log of the short rate
    - θ(t): time-dependent drift (fitted to initial term structure)
    - a: mean reversion speed (a > 0)
    - σ: volatility of log-rate (σ > 0)

Key Properties
--------------
1. **Log-Normal Rates**: r(t) is always positive (unlike Hull-White)
2. **Mean Reversion in Log-Space**: ln r(t) reverts to θ/a
3. **No Closed-Form Bond Prices**: Requires numerical methods (MC, FDE)
4. **Higher Volatility at High Rates**: Vol of r is proportional to r
5. **Industry Use**: Popular for positive-rate environments

Comparison with Hull-White
--------------------------
| Aspect                  | Hull-White           | Black-Karasinski      |
|-------------------------|----------------------|------------------------|
| Rate distribution       | Gaussian             | Log-normal             |
| Negative rates possible | Yes                  | No                     |
| Bond price formula      | Closed-form          | Numerical              |
| Volatility structure    | Constant (additive)  | Proportional to rate   |
| Mean reversion          | In rate              | In log-rate            |

Simulation
----------
The log-rate x = ln(r) follows an Ornstein-Uhlenbeck process, so we use:
- **Exact**: Uses exact OU transition for x(t), then r(t) = exp(x(t))
- **Euler**: Euler-Maruyama for x(t), then r(t) = exp(x(t))

References
----------
- Black, F. & Karasinski, P. (1991). "Bond and Option Pricing when Short Rates
  are Lognormal." Financial Analysts Journal.
- Brigo, D. & Mercurio, F. (2006). "Interest Rate Models - Theory and Practice."
- Hull, J. (2018). "Options, Futures, and Other Derivatives." Chapter 31.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional

from src.models.numeric.monte_carlo.rng import NormalRng


# =============================================================================
# Type definitions
# =============================================================================

# Discretization schemes for Black-Karasinski log-rate process.
BlackKarasinskiScheme = Literal["exact", "euler"]


# =============================================================================
# Black-Karasinski Parameters
# =============================================================================

@dataclass(frozen=True, slots=True)
class BlackKarasinskiParameters:
    """
    Parameters for the Black-Karasinski one-factor short rate model.

    The Black-Karasinski model specifies:
        d(ln r(t)) = [θ(t) - a·ln r(t)] dt + σ dW(t)

    For practical use with constant θ:
        d(ln r(t)) = a·(θ - ln r(t)) dt + σ dW(t)

    Parameters
    ----------
    a : float
        Mean reversion speed (a > 0). Higher values = faster reversion.
        Typical values: 0.01 to 0.5.
    sigma : float
        Volatility of log-rate (σ > 0).
        Typical values: 0.1 to 0.3 (10-30% vol of log-rate).
        Note: This is vol of ln(r), not vol of r itself.
    r0 : float
        Initial short rate r(0). Must be positive.
        Typical values: 0.01 to 0.10 (1-10%).
    theta : float, optional
        Long-term mean reversion level for ln(r). If None, defaults to ln(r0).
        The long-term rate level is exp(theta).

    Attributes
    ----------
    x0 : float
        Initial log-rate: x(0) = ln(r0).
    half_life : float
        Time for log-rate to revert halfway to mean: ln(2)/a.
    long_term_vol : float
        Asymptotic volatility of ln(r): σ/√(2a).
    long_term_rate : float
        Long-term rate level: exp(theta).

    Examples
    --------
    >>> from src.models.short_rate.black_karasinski import BlackKarasinskiParameters
    >>> params = BlackKarasinskiParameters(
    ...     a=0.1,       # Mean reversion speed
    ...     sigma=0.15,  # 15% vol of log-rate
    ...     r0=0.03,     # 3% initial short rate
    ...     theta=-3.0,  # ln(0.05) ≈ -3.0 → 5% long-term rate
    ... )
    >>> params.x0  # ln(0.03)
    -3.506...
    >>> params.long_term_rate
    0.049787...
    """

    a: float              # Mean reversion speed
    sigma: float          # Volatility of log-rate
    r0: float             # Initial short rate (positive)
    theta: float = None   # type: ignore  # Long-term mean for ln(r)

    def __post_init__(self) -> None:
        """Validate parameters and set defaults."""
        if self.a <= 0.0:
            raise ValueError(f"Mean reversion speed a must be > 0; got {self.a}.")
        if self.sigma <= 0.0:
            raise ValueError(f"Volatility sigma must be > 0; got {self.sigma}.")
        if self.r0 <= 0.0:
            raise ValueError(f"Initial rate r0 must be > 0; got {self.r0}.")

        # Set default theta = ln(r0) if not provided.
        if self.theta is None:
            object.__setattr__(self, "theta", float(np.log(self.r0)))

    @property
    def x0(self) -> float:
        """Initial log-rate: x(0) = ln(r0)."""
        return float(np.log(self.r0))

    @property
    def half_life(self) -> float:
        """Time for log-rate to revert halfway to mean: ln(2)/a."""
        return float(np.log(2.0) / self.a)

    @property
    def long_term_vol(self) -> float:
        """Asymptotic standard deviation of ln(r): σ/√(2a)."""
        return float(self.sigma / np.sqrt(2.0 * self.a))

    @property
    def long_term_rate(self) -> float:
        """Long-term rate level: exp(θ)."""
        return float(np.exp(self.theta))

    def expected_log_rate(self, t: float) -> float:
        """
        Expected log-rate E[ln r(t)] starting from x(0) = ln(r0).

        E[x(t)] = θ + (x₀ - θ)·exp(-a·t)

        Parameters
        ----------
        t : float
            Time horizon.

        Returns
        -------
        float
            Expected log-rate at time t.
        """
        return float(self.theta + (self.x0 - self.theta) * np.exp(-self.a * t))

    def variance_log_rate(self, t: float) -> float:
        """
        Variance of log-rate Var[ln r(t)].

        Var[x(t)] = (σ²/(2a))·(1 - exp(-2a·t))

        Parameters
        ----------
        t : float
            Time horizon.

        Returns
        -------
        float
            Variance of log-rate at time t.
        """
        return float(
            (self.sigma ** 2 / (2.0 * self.a)) * (1.0 - np.exp(-2.0 * self.a * t))
        )

    def std_log_rate(self, t: float) -> float:
        """Standard deviation of log-rate at time t."""
        return float(np.sqrt(self.variance_log_rate(t)))


# =============================================================================
# Black-Karasinski Simulation Result
# =============================================================================

@dataclass(frozen=True, slots=True)
class BlackKarasinskiSimulation:
    """
    Container for Black-Karasinski simulation results.

    Attributes
    ----------
    rate_paths : np.ndarray
        Simulated short rate paths r(t), shape (n_paths, n_steps + 1).
    log_rate_paths : np.ndarray
        Simulated log-rate paths x(t) = ln(r(t)), shape (n_paths, n_steps + 1).
    times : np.ndarray
        Time grid, shape (n_steps + 1,).
    params : BlackKarasinskiParameters
        Model parameters used.
    n_paths : int
        Number of simulated paths.
    n_steps : int
        Number of time steps.
    scheme : BlackKarasinskiScheme
        Discretization scheme used.
    seed : int or None
        Random seed used.
    discount_factors : np.ndarray, optional
        Simulated discount factors exp(-∫r(s)ds), shape (n_paths,).
    """

    rate_paths: np.ndarray
    log_rate_paths: np.ndarray
    times: np.ndarray
    params: BlackKarasinskiParameters
    n_paths: int
    n_steps: int
    scheme: BlackKarasinskiScheme
    seed: Optional[int]
    discount_factors: Optional[np.ndarray] = None

    @property
    def terminal_rates(self) -> np.ndarray:
        """Terminal short rate values r(T)."""
        return self.rate_paths[:, -1]

    @property
    def terminal_log_rates(self) -> np.ndarray:
        """Terminal log-rate values ln(r(T))."""
        return self.log_rate_paths[:, -1]

    @property
    def maturity(self) -> float:
        """Time to maturity T."""
        return float(self.times[-1])

    @property
    def mean_terminal_rate(self) -> float:
        """Mean of terminal rates across paths."""
        return float(np.mean(self.terminal_rates))

    @property
    def std_terminal_rate(self) -> float:
        """Standard deviation of terminal rates across paths."""
        return float(np.std(self.terminal_rates))

    @property
    def mean_terminal_log_rate(self) -> float:
        """Mean of terminal log-rates across paths."""
        return float(np.mean(self.terminal_log_rates))

    @property
    def std_terminal_log_rate(self) -> float:
        """Standard deviation of terminal log-rates across paths."""
        return float(np.std(self.terminal_log_rates))


# =============================================================================
# Black-Karasinski Dynamics Simulator
# =============================================================================

@dataclass(frozen=True, slots=True)
class BlackKarasinskiDynamics:
    """
    Simulator for Black-Karasinski one-factor short rate dynamics.

    Simulates the log-rate process under the risk-neutral measure:
        dx(t) = [θ - a·x(t)] dt + σ dW(t)

    where x(t) = ln r(t), then computes r(t) = exp(x(t)).

    Parameters
    ----------
    params : BlackKarasinskiParameters
        Black-Karasinski model parameters (a, σ, r₀, θ).

    Examples
    --------
    >>> from src.models.short_rate.black_karasinski import (
    ...     BlackKarasinskiDynamics, BlackKarasinskiParameters
    ... )
    >>> params = BlackKarasinskiParameters(a=0.1, sigma=0.15, r0=0.03)
    >>> dynamics = BlackKarasinskiDynamics(params=params)
    >>> sim = dynamics.simulate(maturity=1.0, n_paths=10000, n_steps=252)
    >>> sim.mean_terminal_rate  # Close to r0 for short maturity
    0.03...
    """

    params: BlackKarasinskiParameters

    def simulate(
        self,
        maturity: float,
        n_paths: int,
        n_steps: int,
        scheme: BlackKarasinskiScheme = "exact",
        seed: Optional[int] = None,
        antithetic: bool = True,
        compute_discount_factors: bool = True,
    ) -> BlackKarasinskiSimulation:
        """
        Simulate Black-Karasinski short rate paths.

        Parameters
        ----------
        maturity : float
            Time to maturity T > 0.
        n_paths : int
            Number of paths to simulate.
        n_steps : int
            Number of time steps.
        scheme : BlackKarasinskiScheme
            Discretization scheme:
            - "exact": Exact OU transition for log-rate (recommended)
            - "euler": Euler-Maruyama for log-rate
        seed : int, optional
            Random seed for reproducibility.
        antithetic : bool
            Use antithetic variates for variance reduction.
        compute_discount_factors : bool
            If True, compute path-wise discount factors using trapezoidal rule.

        Returns
        -------
        BlackKarasinskiSimulation
            Container with simulated paths and metadata.
        """
        # Validate inputs.
        if maturity <= 0.0:
            raise ValueError("maturity must be > 0.")
        if n_paths <= 0:
            raise ValueError("n_paths must be > 0.")
        if n_steps <= 0:
            raise ValueError("n_steps must be > 0.")

        # Use reproducible RNG from base models.
        rng = NormalRng(seed=seed)

        # Time discretization.
        dt = maturity / n_steps
        sqrt_dt = np.sqrt(dt)
        times = np.linspace(0.0, maturity, n_steps + 1)

        # Generate standard normal increments using NormalRng.
        Z = rng.standard_normals(n_paths, n_steps, antithetic=antithetic)
        n_actual = Z.shape[0]

        # Initialize log-rate paths.
        x = np.zeros((n_actual, n_steps + 1))
        x[:, 0] = self.params.x0  # x(0) = ln(r0)

        # Extract parameters.
        a = self.params.a
        sigma = self.params.sigma
        theta = self.params.theta

        # Simulate log-rate paths.
        if scheme == "exact":
            # Exact OU transition for x: x(t+dt) | x(t) ~ N(μ, σ²)
            # μ = θ + (x(t) - θ)·exp(-a·dt)
            # σ² = (σ²/(2a))·(1 - exp(-2a·dt))
            exp_adt = np.exp(-a * dt)
            var_dt = (sigma ** 2 / (2.0 * a)) * (1.0 - np.exp(-2.0 * a * dt))
            std_dt = np.sqrt(var_dt)

            for i in range(n_steps):
                x[:, i + 1] = theta + (x[:, i] - theta) * exp_adt + std_dt * Z[:, i]

        elif scheme == "euler":
            # Euler-Maruyama: x(t+dt) = x(t) + a(θ - x(t))dt + σ·√dt·Z
            for i in range(n_steps):
                x[:, i + 1] = x[:, i] + a * (theta - x[:, i]) * dt + sigma * sqrt_dt * Z[:, i]

        else:
            raise ValueError(f"Unknown scheme: {scheme}. Use 'exact' or 'euler'.")

        # Convert log-rates to rates: r(t) = exp(x(t))
        r = np.exp(x)

        # Compute discount factors if requested.
        discount_factors = None
        if compute_discount_factors:
            # Trapezoidal rule for ∫r(s)ds.
            integral = dt * (0.5 * r[:, 0] + np.sum(r[:, 1:-1], axis=1) + 0.5 * r[:, -1])
            discount_factors = np.exp(-integral)

        return BlackKarasinskiSimulation(
            rate_paths=r,
            log_rate_paths=x,
            times=times,
            params=self.params,
            n_paths=n_actual,
            n_steps=n_steps,
            scheme=scheme,
            seed=seed,
            discount_factors=discount_factors,
        )


# =============================================================================
# Black-Karasinski Numerical Bond Pricing (MC-based)
# =============================================================================

def bk_zc_bond_price_mc(
    T: float,
    params: BlackKarasinskiParameters,
    n_paths: int = 100_000,
    n_steps: int = 252,
    seed: Optional[int] = None,
) -> float:
    """
    Compute zero-coupon bond price using Monte Carlo under Black-Karasinski.

    P(0, T) = E[exp(-∫₀ᵀ r(s) ds)]

    Parameters
    ----------
    T : float
        Bond maturity.
    params : BlackKarasinskiParameters
        Model parameters.
    n_paths : int
        Number of MC paths.
    n_steps : int
        Number of time steps.
    seed : int, optional
        Random seed.

    Returns
    -------
    float
        MC estimate of ZC bond price.
    """
    if T <= 0.0:
        return 1.0

    dynamics = BlackKarasinskiDynamics(params=params)
    sim = dynamics.simulate(
        maturity=T,
        n_paths=n_paths,
        n_steps=n_steps,
        scheme="exact",
        seed=seed,
        antithetic=True,
        compute_discount_factors=True,
    )

    return float(np.mean(sim.discount_factors))


def bk_zc_bond_option_price_mc(
    K: float,
    T_option: float,
    T_bond: float,
    params: BlackKarasinskiParameters,
    is_call: bool = True,
    notional: float = 100.0,
    n_paths: int = 100_000,
    n_steps: int = 252,
    seed: Optional[int] = None,
) -> float:
    """
    Price a European option on a zero-coupon bond using Monte Carlo.

    Call payoff: max(P(T_option, T_bond) - K, 0) × discount
    Put payoff:  max(K - P(T_option, T_bond), 0) × discount

    Parameters
    ----------
    K : float
        Strike price (as fraction of face value, e.g., 0.95 for 95%).
    T_option : float
        Option expiry time.
    T_bond : float
        Underlying bond maturity (T_bond > T_option).
    params : BlackKarasinskiParameters
        Model parameters.
    is_call : bool
        True for call, False for put.
    notional : float
        Notional amount.
    n_paths : int
        Number of MC paths.
    n_steps : int
        Number of time steps for option period.
    seed : int, optional
        Random seed.

    Returns
    -------
    float
        MC estimate of option price.
    """
    if T_option <= 0.0:
        # Expired option - intrinsic value.
        P_bond = bk_zc_bond_price_mc(T_bond, params, n_paths, n_steps, seed)
        if is_call:
            return notional * max(P_bond - K, 0.0)
        return notional * max(K - P_bond, 0.0)

    if T_bond <= T_option:
        raise ValueError(f"T_bond ({T_bond}) must be > T_option ({T_option}).")

    # Simulate to option expiry.
    dynamics = BlackKarasinskiDynamics(params=params)
    sim = dynamics.simulate(
        maturity=T_option,
        n_paths=n_paths,
        n_steps=n_steps,
        scheme="exact",
        seed=seed,
        antithetic=True,
        compute_discount_factors=True,
    )

    # At option expiry, compute bond price P(T_option, T_bond) for each path.
    # Under BK, we need to simulate further or use approximation.
    # Simple approximation: P(T_opt, T_bond) ≈ exp(-r(T_opt) × (T_bond - T_opt))
    # This is a rough first-order approximation (exact for flat forward curve).
    r_T = sim.terminal_rates
    tau = T_bond - T_option
    P_bond_at_T = np.exp(-r_T * tau)

    # Compute option payoffs.
    if is_call:
        payoffs = np.maximum(P_bond_at_T - K, 0.0)
    else:
        payoffs = np.maximum(K - P_bond_at_T, 0.0)

    # Discount payoffs to t=0.
    discounted_payoffs = sim.discount_factors * payoffs * notional

    return float(np.mean(discounted_payoffs))


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Parameters
    "BlackKarasinskiParameters",
    # Simulation
    "BlackKarasinskiScheme",
    "BlackKarasinskiSimulation",
    "BlackKarasinskiDynamics",
    # MC Pricing
    "bk_zc_bond_price_mc",
    "bk_zc_bond_option_price_mc",
]

"""
Variance Gamma Process Implementation.

This module provides the Variance Gamma (VG) model for pricing derivatives
using a time-changed Brownian motion with Gamma subordinator.

Mathematical Framework
----------------------
The Variance Gamma process X_t is defined as:

    X_t = θ G_t + σ W_{G_t}

where:
    - G_t ~ Gamma(t/ν, 1/ν): Gamma process with mean t, variance νt
    - W_t: Standard Brownian motion
    - θ: Drift parameter (controls skewness)
    - σ: Volatility parameter
    - ν: Variance rate of Gamma time (controls kurtosis)

For the asset price under the risk-neutral measure:

    S_t = S_0 exp((r - q + ω)t + X_t)

where the martingale correction is:
    ω = (1/ν) ln(1 - θν - σ²ν/2)

Key Properties
--------------
1. **Pure jump process**: No diffusion component
2. **Finite variation**: Integrable paths
3. **Fat tails**: Kurtosis controlled by ν
4. **Skewness**: Sign of θ determines skew direction
5. **Semi-closed form**: European options via FFT

Characteristic Function
-----------------------
The characteristic function of X_t is:

    φ(u) = E[exp(iuX_t)] = (1 - iuθν + σ²ν u²/2)^(-t/ν)

This enables FFT-based pricing.

References
----------
- Madan, D.B., Carr, P., & Chang, E.C. (1998). "The Variance Gamma Process 
  and Option Pricing." European Finance Review.
- Carr, P. & Madan, D.B. (1999). "Option Valuation Using the Fast Fourier 
  Transform." Journal of Computational Finance.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Optional

from src.models.numeric.monte_carlo.rng import NormalRng


# =============================================================================
# Variance Gamma Parameters
# =============================================================================

@dataclass(frozen=True, slots=True)
class VarianceGammaParameters:
    """
    Parameters for the Variance Gamma process.

    The VG process is X_t = θ G_t + σ W_{G_t} where G_t is Gamma(t/ν, 1/ν).

    Parameters
    ----------
    theta : float
        Drift parameter θ (controls skewness).
        - θ < 0: Negative skew (equity-like)
        - θ > 0: Positive skew
        - θ = 0: Symmetric distribution
    sigma : float
        Volatility parameter σ > 0.
    nu : float
        Variance rate of Gamma time ν > 0.
        - Higher ν: Fatter tails
        - Lower ν: More like Black-Scholes

    Attributes
    ----------
    omega : float
        Martingale drift correction.
    kurtosis_excess : float
        Excess kurtosis of the VG distribution.
    skewness : float
        Skewness of the VG distribution.

    Examples
    --------
    >>> params = VarianceGammaParameters(
    ...     theta=-0.1,   # Negative skew
    ...     sigma=0.2,    # 20% vol
    ...     nu=0.2,       # Moderate fat tails
    ... )
    >>> params.omega  # Martingale correction
    -0.024...
    """

    theta: float  # Drift parameter θ
    sigma: float  # Volatility parameter σ
    nu: float     # Variance rate of Gamma time ν

    def __post_init__(self) -> None:
        """Validate VG parameters."""
        # Validate theta (drift)
        if not np.isfinite(self.theta):
            raise ValueError("theta must be finite.")

        # Validate sigma (volatility)
        if not np.isfinite(self.sigma):
            raise ValueError("sigma must be finite.")
        if self.sigma <= 0.0:
            raise ValueError("sigma must be > 0.")

        # Validate nu (variance rate)
        if not np.isfinite(self.nu):
            raise ValueError("nu must be finite.")
        if self.nu <= 0.0:
            raise ValueError("nu must be > 0.")

        # Check that omega is well-defined (argument to log must be positive)
        arg = 1.0 - self.theta * self.nu - 0.5 * self.sigma**2 * self.nu
        if arg <= 0.0:
            raise ValueError(
                f"Invalid parameters: 1 - θν - σ²ν/2 = {arg:.4f} must be > 0. "
                f"Reduce theta or nu."
            )

    @property
    def omega(self) -> float:
        """
        Martingale drift correction ω.

        ω = (1/ν) ln(1 - θν - σ²ν/2)

        Ensures E[S_T] = S_0 exp((r-q)T).
        """
        arg = 1.0 - self.theta * self.nu - 0.5 * self.sigma**2 * self.nu
        return math.log(arg) / self.nu

    @property
    def variance_rate(self) -> float:
        """
        Variance rate of X_t: Var[X_1] = σ² + θ²ν.
        """
        return self.sigma**2 + self.theta**2 * self.nu

    @property
    def equivalent_bs_vol(self) -> float:
        """
        Approximate equivalent Black-Scholes volatility.
        """
        return math.sqrt(self.variance_rate)

    @property
    def skewness(self) -> float:
        """
        Skewness of X_1.

        Skew = θ(3σ²ν + 2θ²ν²) / (σ² + θ²ν)^(3/2)
        """
        var = self.variance_rate
        return self.theta * (3.0 * self.sigma**2 * self.nu + 2.0 * self.theta**2 * self.nu**2) / var**1.5

    @property
    def excess_kurtosis(self) -> float:
        """
        Excess kurtosis of X_1.

        Excess Kurt = 3(σ⁴ν + 2θ⁴ν³ + 4σ²θ²ν²) / (σ² + θ²ν)²
        """
        s2 = self.sigma**2
        t2 = self.theta**2
        var = s2 + t2 * self.nu
        numerator = 3.0 * (s2**2 * self.nu + 2.0 * t2**2 * self.nu**3 + 4.0 * s2 * t2 * self.nu**2)
        return numerator / var**2


# =============================================================================
# Variance Gamma Simulation Output
# =============================================================================

@dataclass(frozen=True, slots=True)
class VarianceGammaSimulation:
    """
    Output container for Variance Gamma path simulation.

    Attributes
    ----------
    spot_paths : np.ndarray
        Simulated spot paths, shape (n_paths, n_steps + 1).
    gamma_times : np.ndarray
        Cumulative Gamma time for each path, shape (n_paths, n_steps + 1).
    times : np.ndarray
        Calendar time grid, shape (n_steps + 1,).
    params : VarianceGammaParameters
        VG parameters used in simulation.
    drift : float
        Drift rate used.
    n_paths : int
        Number of simulated paths.
    n_steps : int
        Number of time steps.
    seed : int or None
        Random seed used.
    """

    spot_paths: np.ndarray
    gamma_times: np.ndarray
    times: np.ndarray
    params: VarianceGammaParameters
    drift: float
    n_paths: int
    n_steps: int
    seed: Optional[int]

    @property
    def terminal_spots(self) -> np.ndarray:
        """Terminal spot values S_T."""
        return self.spot_paths[:, -1]

    @property
    def total_gamma_time(self) -> np.ndarray:
        """Total Gamma time G_T for each path."""
        return self.gamma_times[:, -1]

    @property
    def maturity(self) -> float:
        """Calendar time to maturity T."""
        return float(self.times[-1])

    @property
    def average_gamma_time(self) -> float:
        """Average Gamma time across paths (should ≈ T)."""
        return float(np.mean(self.total_gamma_time))


# =============================================================================
# Variance Gamma Dynamics Simulator
# =============================================================================

@dataclass(frozen=True, slots=True)
class VarianceGammaDynamics:
    """
    Simulator for Variance Gamma dynamics.

    Simulates the process under the risk-neutral measure:
        S_t = S_0 exp((μ + ω)t + X_t)

    where X_t = θ G_t + σ W_{G_t} and μ = r - q.

    Parameters
    ----------
    params : VarianceGammaParameters
        VG model parameters (θ, σ, ν).
    drift : float
        Drift coefficient μ = r - q (risk-neutral drift).

    Examples
    --------
    >>> from src.models.levy.variance_gamma import (
    ...     VarianceGammaDynamics, VarianceGammaParameters
    ... )
    >>> params = VarianceGammaParameters(theta=-0.1, sigma=0.2, nu=0.2)
    >>> dynamics = VarianceGammaDynamics(params=params, drift=0.03)
    >>> sim = dynamics.simulate(
    ...     spot0=100.0, maturity=1.0, n_paths=10000, n_steps=252
    ... )
    >>> sim.terminal_spots.mean()  # Around 100 * exp(0.03)
    """

    params: VarianceGammaParameters
    drift: float  # μ = r - q

    def simulate(
        self,
        spot0: float,
        maturity: float,
        n_paths: int,
        n_steps: int,
        seed: Optional[int] = None,
        antithetic: bool = True,
    ) -> VarianceGammaSimulation:
        """
        Simulate Variance Gamma paths via subordination.

        The VG process is simulated by:
        1. Simulate Gamma time increments ΔG_i
        2. Simulate Brownian increments ΔW_i scaled by √(ΔG_i)
        3. Compute X increment: ΔX_i = θ ΔG_i + σ √(ΔG_i) Z_i

        Parameters
        ----------
        spot0 : float
            Initial spot price S_0 > 0.
        maturity : float
            Calendar time to maturity T > 0.
        n_paths : int
            Number of paths to simulate.
        n_steps : int
            Number of calendar time steps.
        seed : int, optional
            Random seed for reproducibility.
        antithetic : bool
            Use antithetic variates for variance reduction.

        Returns
        -------
        VarianceGammaSimulation
            Container with simulated paths and metadata.
        """
        # Validate inputs
        if spot0 <= 0.0:
            raise ValueError("spot0 must be > 0.")
        if maturity <= 0.0:
            raise ValueError("maturity must be > 0.")
        if n_paths <= 0:
            raise ValueError("n_paths must be > 0.")
        if n_steps <= 0:
            raise ValueError("n_steps must be > 0.")

        # Initialize RNG
        rng = NormalRng(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        # Time discretization
        dt = maturity / n_steps
        times = np.linspace(0.0, maturity, n_steps + 1)

        # Handle antithetic variates (only for Brownian, not Gamma)
        if antithetic:
            n_base = (n_paths + 1) // 2
            n_actual = 2 * n_base
        else:
            n_base = n_paths
            n_actual = n_paths

        # Extract parameters
        theta = self.params.theta
        sigma = self.params.sigma
        nu = self.params.nu
        omega = self.params.omega

        # Total drift including martingale correction
        total_drift = self.drift + omega

        # Gamma increment parameters for time step dt
        # G(dt) ~ Gamma(dt/ν, 1/ν) has shape = dt/ν, scale = ν
        shape = dt / nu
        scale = nu

        # Initialize path arrays
        S = np.zeros((n_actual, n_steps + 1))
        G = np.zeros((n_actual, n_steps + 1))  # Cumulative Gamma time
        S[:, 0] = spot0
        G[:, 0] = 0.0

        # Log-spot for stability
        log_S = np.full(n_actual, np.log(spot0))

        # Generate normal increments
        Z = rng.standard_normals(n_base * n_steps, 1).reshape(n_base, n_steps)

        if antithetic:
            Z_full = np.vstack([Z, -Z])
        else:
            Z_full = Z

        # Simulate paths step by step
        for i in range(n_steps):
            # Gamma time increments (same for antithetic pairs to preserve variance reduction)
            dG = np.random.gamma(shape, scale, n_actual)

            # VG increment: ΔX = θ ΔG + σ √(ΔG) Z
            sqrt_dG = np.sqrt(dG)
            dX = theta * dG + sigma * sqrt_dG * Z_full[:, i]

            # Update log-spot
            log_S = log_S + total_drift * dt + dX

            # Store paths
            S[:, i + 1] = np.exp(log_S)
            G[:, i + 1] = G[:, i] + dG

        return VarianceGammaSimulation(
            spot_paths=S,
            gamma_times=G,
            times=times,
            params=self.params,
            drift=self.drift,
            n_paths=n_actual,
            n_steps=n_steps,
            seed=seed,
        )

    def simulate_terminal(
        self,
        spot0: float,
        maturity: float,
        n_paths: int,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Exact simulation of terminal spot S_T (single time step).

        For European option pricing, we can simulate S_T directly:
            S_T = S_0 exp((μ + ω)T + θ G_T + σ √(G_T) Z)

        where G_T ~ Gamma(T/ν, ν) and Z ~ N(0, 1) independent.

        Parameters
        ----------
        spot0 : float
            Initial spot price.
        maturity : float
            Calendar time to maturity T.
        n_paths : int
            Number of samples.
        seed : int, optional
            Random seed.

        Returns
        -------
        np.ndarray
            Terminal spot values S_T, shape (n_paths,).
        """
        if spot0 <= 0.0:
            raise ValueError("spot0 must be > 0.")
        if maturity <= 0.0:
            raise ValueError("maturity must be > 0.")

        if seed is not None:
            np.random.seed(seed)

        # Extract parameters
        theta = self.params.theta
        sigma = self.params.sigma
        nu = self.params.nu
        omega = self.params.omega

        total_drift = self.drift + omega

        # Gamma time: G_T ~ Gamma(T/ν, ν)
        shape = maturity / nu
        scale = nu
        G_T = np.random.gamma(shape, scale, n_paths)

        # Brownian component
        Z = np.random.standard_normal(n_paths)

        # VG value: X_T = θ G_T + σ √(G_T) Z
        X_T = theta * G_T + sigma * np.sqrt(G_T) * Z

        # Terminal spot
        S_T = spot0 * np.exp(total_drift * maturity + X_T)

        return S_T


# =============================================================================
# Variance Gamma European Option Pricing (FFT Method)
# =============================================================================

def vg_characteristic_function(
    u: np.ndarray,
    T: float,
    params: VarianceGammaParameters,
) -> np.ndarray:
    """
    Characteristic function of the VG process X_T.

    φ(u) = (1 - iuθν + σ²ν u²/2)^(-T/ν)

    Parameters
    ----------
    u : np.ndarray
        Argument values (can be complex).
    T : float
        Time to maturity.
    params : VarianceGammaParameters
        VG parameters.

    Returns
    -------
    np.ndarray
        Characteristic function values.
    """
    theta = params.theta
    sigma = params.sigma
    nu = params.nu

    # φ(u) = (1 - iuθν + σ²ν u²/2)^(-T/ν)
    base = 1.0 - 1j * u * theta * nu + 0.5 * sigma**2 * nu * u**2
    return base ** (-T / nu)


def vg_european_call(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    params: VarianceGammaParameters,
    n_paths: int = 100000,
    seed: Optional[int] = None,
) -> float:
    """
    Price European call via Monte Carlo simulation.

    For Variance Gamma, MC is more robust than FFT for typical use cases.

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to maturity.
    r : float
        Risk-free rate.
    q : float
        Dividend/foreign rate.
    params : VarianceGammaParameters
        VG parameters.
    n_paths : int
        Number of MC paths.
    seed : int, optional
        Random seed.

    Returns
    -------
    float
        European call option price.
    """
    return vg_mc_call(
        spot0=S, strike=K, maturity=T, r=r, q=q,
        params=params, n_paths=n_paths, seed=seed
    )


def vg_european_put(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    params: VarianceGammaParameters,
    n_paths: int = 100000,
    seed: Optional[int] = None,
) -> float:
    """
    Price European put via Monte Carlo simulation.

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to maturity.
    r : float
        Risk-free rate.
    q : float
        Dividend/foreign rate.
    params : VarianceGammaParameters
        VG parameters.
    n_paths : int
        Number of MC paths.
    seed : int, optional
        Random seed.

    Returns
    -------
    float
        European put option price.
    """
    return vg_mc_put(
        spot0=S, strike=K, maturity=T, r=r, q=q,
        params=params, n_paths=n_paths, seed=seed
    )


def vg_mc_call(
    spot0: float,
    strike: float,
    maturity: float,
    r: float,
    q: float,
    params: VarianceGammaParameters,
    n_paths: int = 100000,
    seed: Optional[int] = None,
) -> float:
    """
    Price European call via VG Monte Carlo.

    Parameters
    ----------
    spot0 : float
        Initial spot price.
    strike : float
        Strike price.
    maturity : float
        Time to maturity.
    r : float
        Risk-free rate.
    q : float
        Dividend/foreign rate.
    params : VarianceGammaParameters
        VG parameters.
    n_paths : int
        Number of MC paths.
    seed : int, optional
        Random seed.

    Returns
    -------
    float
        Call option price.
    """
    dynamics = VarianceGammaDynamics(params=params, drift=r - q)
    S_T = dynamics.simulate_terminal(
        spot0=spot0, maturity=maturity, n_paths=n_paths, seed=seed
    )

    payoffs = np.maximum(S_T - strike, 0.0)
    return float(np.exp(-r * maturity) * np.mean(payoffs))


def vg_mc_put(
    spot0: float,
    strike: float,
    maturity: float,
    r: float,
    q: float,
    params: VarianceGammaParameters,
    n_paths: int = 100000,
    seed: Optional[int] = None,
) -> float:
    """
    Price European put via VG Monte Carlo.

    See `vg_mc_call` for parameter descriptions.
    """
    dynamics = VarianceGammaDynamics(params=params, drift=r - q)
    S_T = dynamics.simulate_terminal(
        spot0=spot0, maturity=maturity, n_paths=n_paths, seed=seed
    )

    payoffs = np.maximum(strike - S_T, 0.0)
    return float(np.exp(-r * maturity) * np.mean(payoffs))

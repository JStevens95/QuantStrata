"""
SABR Stochastic Volatility Model - Dynamics and Simulation.

This module provides the SABR model dynamics for Monte Carlo simulation,
complementing the analytical Hagan approximation in the calibration module.

SABR Model Dynamics
-------------------
Under the forward measure, the SABR model specifies:

    dF_t = σ_t F_t^β dW_t^F
    dσ_t = ν σ_t dW_t^σ
    dW_t^F dW_t^σ = ρ dt

where:
    - F_t: Forward price (or rate)
    - σ_t: Stochastic volatility
    - β: CEV exponent (0 ≤ β ≤ 1)
    - ν: Vol-of-vol
    - ρ: Correlation between forward and vol processes

Key Properties
--------------
1. **Martingale**: F_t is a martingale (no drift under forward measure)
2. **Absorbing at zero**: F_t = 0 is absorbing when β < 1
3. **Vol smile**: Negative ρ creates downside skew
4. **Term structure**: ν controls how fast smile flattens with maturity

Special Cases
-------------
- β = 0: Normal SABR (Bachelier-like, allows negative forwards)
- β = 0.5: CIR-like dynamics
- β = 1: Log-normal SABR (simplest, good for FX)

Discretization
--------------
We use the log-Euler scheme for stability:
    ln F_{t+dt} = ln F_t + σ_t F_t^{β-1} dW^F - 0.5 σ_t² F_t^{2β-2} dt  (β < 1)
    ln F_{t+dt} = ln F_t + σ_t dW^F - 0.5 σ_t² dt  (β = 1)
    ln σ_{t+dt} = ln σ_t + ν dW^σ - 0.5 ν² dt

References
----------
- Hagan, P.S. et al. (2002). "Managing Smile Risk." Wilmott Magazine.
- Islah, O. (2009). "Solving SABR in exact form and unifying it with LIBOR 
  market model." Available at SSRN.
- Andersen, L. & Piterbarg, V. (2010). Interest Rate Modeling.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional

from src.models.numeric.monte_carlo.rng import NormalRng


# Re-export the calibration parameters for convenience
from src.calibration.volatility_surface.sabr import (
    SabrParameters,
    SabrConfig,
    sabr_implied_vol,
    sabr_implied_vol_vec,
    calibrate_sabr_to_smile,
)


# =============================================================================
# Type Definitions
# =============================================================================

SabrScheme = Literal["euler", "log_euler", "absorbing"]


# =============================================================================
# SABR Simulation Output
# =============================================================================

@dataclass(frozen=True, slots=True)
class SabrSimulation:
    """
    Output container for SABR path simulation.

    Attributes
    ----------
    forward_paths : np.ndarray
        Simulated forward paths, shape (n_paths, n_steps + 1).
    vol_paths : np.ndarray
        Simulated volatility paths, shape (n_paths, n_steps + 1).
    times : np.ndarray
        Time grid, shape (n_steps + 1,).
    params : SabrParameters
        SABR parameters used in simulation.
    n_paths : int
        Number of simulated paths.
    n_steps : int
        Number of time steps.
    scheme : SabrScheme
        Discretization scheme used.
    seed : int or None
        Random seed used.
    """

    forward_paths: np.ndarray
    vol_paths: np.ndarray
    times: np.ndarray
    params: SabrParameters
    n_paths: int
    n_steps: int
    scheme: SabrScheme
    seed: Optional[int]

    @property
    def terminal_forwards(self) -> np.ndarray:
        """Terminal forward values F_T."""
        return self.forward_paths[:, -1]

    @property
    def terminal_vols(self) -> np.ndarray:
        """Terminal volatility values σ_T."""
        return self.vol_paths[:, -1]

    @property
    def maturity(self) -> float:
        """Time to maturity T."""
        return float(self.times[-1])

    @property
    def absorbed_paths(self) -> int:
        """Number of paths absorbed at F=0 (for β < 1)."""
        return int(np.sum(self.terminal_forwards <= 0))

    @property
    def absorption_fraction(self) -> float:
        """Fraction of paths absorbed at F=0."""
        return self.absorbed_paths / self.n_paths


# =============================================================================
# SABR Dynamics Simulator
# =============================================================================

@dataclass(frozen=True, slots=True)
class SabrDynamics:
    """
    Simulator for SABR stochastic volatility dynamics.

    Simulates the joint process (F_t, σ_t) under the forward measure:
        dF_t = σ_t F_t^β dW_t^F
        dσ_t = ν σ_t dW_t^σ
        Corr(dW^F, dW^σ) = ρ

    Parameters
    ----------
    params : SabrParameters
        SABR model parameters (α, β, ρ, ν).

    Examples
    --------
    >>> from src.models.stochastic_volatility.sabr import SabrDynamics
    >>> from src.calibration.volatility_surface.sabr import SabrParameters
    >>> params = SabrParameters(alpha=0.3, beta=1.0, rho=-0.5, nu=0.4)
    >>> dynamics = SabrDynamics(params=params)
    >>> sim = dynamics.simulate(
    ...     forward0=100.0, maturity=1.0, n_paths=10000, n_steps=252
    ... )
    >>> sim.terminal_forwards.mean()  # Should be around 100 (martingale)
    """

    params: SabrParameters

    def simulate(
        self,
        forward0: float,
        maturity: float,
        n_paths: int,
        n_steps: int,
        scheme: SabrScheme = "log_euler",
        seed: Optional[int] = None,
        antithetic: bool = True,
    ) -> SabrSimulation:
        """
        Simulate SABR paths.

        Parameters
        ----------
        forward0 : float
            Initial forward price F_0.
        maturity : float
            Time to maturity T.
        n_paths : int
            Number of paths to simulate.
        n_steps : int
            Number of time steps.
        scheme : SabrScheme
            Discretization scheme:
            - "euler": Simple Euler (can give negative forwards for β < 1)
            - "log_euler": Log-Euler (preserves positivity)
            - "absorbing": Euler with absorption at F=0
        seed : int, optional
            Random seed for reproducibility.
        antithetic : bool
            Use antithetic variates for variance reduction.

        Returns
        -------
        SabrSimulation
            Container with simulated paths and metadata.
        """
        # Validate inputs
        if forward0 <= 0.0:
            raise ValueError("forward0 must be > 0.")
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
        sqrt_dt = np.sqrt(dt)
        times = np.linspace(0.0, maturity, n_steps + 1)

        # Handle antithetic variates
        if antithetic:
            n_base = (n_paths + 1) // 2
            n_actual = 2 * n_base
        else:
            n_base = n_paths
            n_actual = n_paths

        # Extract parameters
        alpha = self.params.alpha
        beta = self.params.beta
        rho = self.params.rho
        nu = self.params.nu

        # Generate correlated Brownian increments
        Z1 = rng.standard_normals(n_base * n_steps, 1).reshape(n_base, n_steps)
        Z2 = rng.standard_normals(n_base * n_steps, 1).reshape(n_base, n_steps)

        # Correlate: dW^σ = ρ dW^F + √(1-ρ²) dZ_independent
        sqrt_1_rho2 = np.sqrt(1.0 - rho**2)

        dW_F_base = sqrt_dt * Z1
        dW_sigma_base = sqrt_dt * (rho * Z1 + sqrt_1_rho2 * Z2)

        if antithetic:
            dW_F = np.vstack([dW_F_base, -dW_F_base])
            dW_sigma = np.vstack([dW_sigma_base, -dW_sigma_base])
        else:
            dW_F = dW_F_base
            dW_sigma = dW_sigma_base

        # Initialize path arrays
        F = np.zeros((n_actual, n_steps + 1))
        sigma = np.zeros((n_actual, n_steps + 1))
        F[:, 0] = forward0
        sigma[:, 0] = alpha  # Initial vol = alpha

        # Simulate paths step by step
        for i in range(n_steps):
            F_curr = F[:, i]
            sigma_curr = sigma[:, i]

            # Volatility step (log-Euler for positivity)
            # dσ = ν σ dW^σ → d(ln σ) = -0.5 ν² dt + ν dW^σ
            ln_sigma_next = np.log(sigma_curr) - 0.5 * nu**2 * dt + nu * dW_sigma[:, i]
            sigma_next = np.exp(ln_sigma_next)

            # Forward step
            if scheme == "euler":
                F_next = self._euler_step(F_curr, sigma_curr, beta, dt, dW_F[:, i])
            elif scheme == "log_euler":
                F_next = self._log_euler_step(F_curr, sigma_curr, beta, dt, dW_F[:, i])
            elif scheme == "absorbing":
                F_next = self._absorbing_step(F_curr, sigma_curr, beta, dt, dW_F[:, i])
            else:
                raise ValueError(f"Unknown scheme: {scheme}")

            F[:, i + 1] = F_next
            sigma[:, i + 1] = sigma_next

        return SabrSimulation(
            forward_paths=F,
            vol_paths=sigma,
            times=times,
            params=self.params,
            n_paths=n_actual,
            n_steps=n_steps,
            scheme=scheme,
            seed=seed,
        )

    def _euler_step(
        self,
        F: np.ndarray,
        sigma: np.ndarray,
        beta: float,
        dt: float,
        dW: np.ndarray,
    ) -> np.ndarray:
        """
        Euler step for forward: dF = σ F^β dW.

        Can produce negative values for β < 1.
        """
        F_pos = np.maximum(F, 1e-10)  # Protect against numerical issues
        F_beta = F_pos ** beta
        return F + sigma * F_beta * dW

    def _log_euler_step(
        self,
        F: np.ndarray,
        sigma: np.ndarray,
        beta: float,
        dt: float,
        dW: np.ndarray,
    ) -> np.ndarray:
        """
        Log-Euler step for forward.

        For β = 1: d(ln F) = -0.5 σ² dt + σ dW
        For β < 1: d(ln F) = -0.5 σ² F^{2β-2} dt + σ F^{β-1} dW
        """
        F_pos = np.maximum(F, 1e-10)

        if abs(beta - 1.0) < 1e-10:
            # Log-normal case: dF/F = σ dW
            log_increment = -0.5 * sigma**2 * dt + sigma * dW
        else:
            # General CEV case
            F_beta_m1 = F_pos ** (beta - 1.0)
            F_2beta_m2 = F_pos ** (2.0 * beta - 2.0)
            log_increment = -0.5 * sigma**2 * F_2beta_m2 * dt + sigma * F_beta_m1 * dW

        return F * np.exp(log_increment)

    def _absorbing_step(
        self,
        F: np.ndarray,
        sigma: np.ndarray,
        beta: float,
        dt: float,
        dW: np.ndarray,
    ) -> np.ndarray:
        """
        Euler step with absorption at F = 0.

        If F < 0 after step, absorb to 0.
        """
        F_pos = np.maximum(F, 0.0)
        F_beta = np.where(F_pos > 0, F_pos ** beta, 0.0)
        F_next = F + sigma * F_beta * dW
        return np.maximum(F_next, 0.0)

    def simulate_terminal(
        self,
        forward0: float,
        maturity: float,
        n_paths: int,
        n_steps: int = 100,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Efficient simulation of terminal forward F_T only.

        Parameters
        ----------
        forward0 : float
            Initial forward price.
        maturity : float
            Time to maturity T.
        n_paths : int
            Number of samples.
        n_steps : int
            Number of time steps (default 100).
        seed : int, optional
            Random seed.

        Returns
        -------
        np.ndarray
            Terminal forward values F_T, shape (n_paths,).
        """
        sim = self.simulate(
            forward0=forward0,
            maturity=maturity,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=seed,
            antithetic=True,
        )
        return sim.terminal_forwards


# =============================================================================
# SABR European Option Pricing (Monte Carlo)
# =============================================================================

def sabr_mc_call(
    forward0: float,
    strike: float,
    maturity: float,
    discount_factor: float,
    params: SabrParameters,
    n_paths: int = 100000,
    n_steps: int = 100,
    seed: Optional[int] = None,
) -> float:
    """
    Price European call via SABR Monte Carlo.

    Parameters
    ----------
    forward0 : float
        Initial forward price.
    strike : float
        Strike price.
    maturity : float
        Time to maturity.
    discount_factor : float
        Discount factor to payment date.
    params : SabrParameters
        SABR parameters.
    n_paths : int
        Number of MC paths.
    n_steps : int
        Number of time steps.
    seed : int, optional
        Random seed.

    Returns
    -------
    float
        Call option price.
    """
    dynamics = SabrDynamics(params=params)
    F_T = dynamics.simulate_terminal(
        forward0=forward0, maturity=maturity, n_paths=n_paths, n_steps=n_steps, seed=seed
    )

    payoffs = np.maximum(F_T - strike, 0.0)
    return float(discount_factor * np.mean(payoffs))


def sabr_mc_put(
    forward0: float,
    strike: float,
    maturity: float,
    discount_factor: float,
    params: SabrParameters,
    n_paths: int = 100000,
    n_steps: int = 100,
    seed: Optional[int] = None,
) -> float:
    """
    Price European put via SABR Monte Carlo.

    Parameters
    ----------
    forward0 : float
        Initial forward price.
    strike : float
        Strike price.
    maturity : float
        Time to maturity.
    discount_factor : float
        Discount factor to payment date.
    params : SabrParameters
        SABR parameters.
    n_paths : int
        Number of MC paths.
    n_steps : int
        Number of time steps.
    seed : int, optional
        Random seed.

    Returns
    -------
    float
        Put option price.
    """
    dynamics = SabrDynamics(params=params)
    F_T = dynamics.simulate_terminal(
        forward0=forward0, maturity=maturity, n_paths=n_paths, n_steps=n_steps, seed=seed
    )

    payoffs = np.maximum(strike - F_T, 0.0)
    return float(discount_factor * np.mean(payoffs))

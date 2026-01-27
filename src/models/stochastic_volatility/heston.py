"""
Heston Stochastic Volatility Model Implementation.

This module provides the Heston model for pricing derivatives where volatility
follows a mean-reverting square-root (CIR) process correlated with the spot.

Mathematical Framework
----------------------
The Heston model specifies two coupled SDEs under the risk-neutral measure:

    dS_t = (r - q) S_t dt + √V_t S_t dW_t^S
    dV_t = κ(θ - V_t) dt + ξ √V_t dW_t^V

where:
    Corr(dW_t^S, dW_t^V) = ρ

Parameters
----------
- κ (kappa): Mean reversion speed (typical: 1-5)
- θ (theta): Long-term variance (typical: 0.01-0.10)
- ξ (xi/vol_of_vol): Volatility of variance (typical: 0.1-1.0)
- V_0: Initial variance (typical: θ or current ATM implied vol squared)
- ρ (rho): Spot-variance correlation (typical: -0.9 to -0.3 for equities)

Key Properties
--------------
1. **Mean Reversion**: Variance reverts to θ at rate κ.
2. **Feller Condition**: If 2κθ > ξ², variance stays positive a.s.
3. **Vol Smile**: Negative ρ generates downward-sloping implied vol smile.
4. **Vol Term Structure**: κ controls how fast smile flattens with maturity.

Discretization Schemes
----------------------
- Euler: Simple but can give negative variance.
- Full Truncation: max(V, 0) truncation.
- Reflection: |V| if V < 0.
- Quadratic-Exponential (QE): Advanced scheme, better accuracy.

References
----------
- Heston, S. (1993). "A Closed-Form Solution for Options with Stochastic
  Volatility." Review of Financial Studies.
- Andersen, L. (2008). "Efficient Simulation of the Heston Stochastic
  Volatility Model."
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional


# =============================================================================
# Type definitions
# =============================================================================

# Discretization schemes for Heston variance process.
HestonScheme = Literal["euler", "full_truncation", "reflection", "qe"]


# =============================================================================
# Heston Parameters
# =============================================================================

@dataclass(frozen=True, slots=True)
class HestonParameters:
    """
    Parameters for the Heston stochastic volatility model.

    The Heston model specifies:
        dS_t = (r - q) S_t dt + √V_t S_t dW_t^S
        dV_t = κ(θ - V_t) dt + ξ √V_t dW_t^V
        Corr(dW_t^S, dW_t^V) = ρ

    Parameters
    ----------
    kappa : float
        Mean reversion speed κ > 0.
    theta : float
        Long-term variance θ > 0.
    xi : float
        Volatility of variance ξ > 0 (also called vol-of-vol).
    v0 : float
        Initial variance V_0 > 0.
    rho : float
        Spot-variance correlation ρ ∈ (-1, 1).

    Attributes
    ----------
    feller_ratio : float
        2κθ/ξ². If > 1, Feller condition satisfied.
    long_term_vol : float
        √θ, the long-term volatility level.

    Examples
    --------
    >>> from src.models.stochastic_vol.heston import HestonParameters
    >>> params = HestonParameters(
    ...     kappa=2.0,    # Mean reversion speed.
    ...     theta=0.04,   # Long-term variance (20% vol).
    ...     xi=0.3,       # Vol of vol.
    ...     v0=0.04,      # Initial variance.
    ...     rho=-0.7,     # Negative correlation (equity-like).
    ... )
    >>> params.feller_satisfied
    True
    >>> params.long_term_vol
    0.2
    """

    kappa: float  # Mean reversion speed κ.
    theta: float  # Long-term variance θ.
    xi: float     # Vol of variance ξ.
    v0: float     # Initial variance V_0.
    rho: float    # Spot-variance correlation ρ.

    def __post_init__(self) -> None:
        """Validate Heston parameters."""
        # Validate kappa (mean reversion speed).
        if not np.isfinite(self.kappa):
            raise ValueError("kappa must be finite.")
        if self.kappa <= 0.0:
            raise ValueError("kappa must be > 0.")

        # Validate theta (long-term variance).
        if not np.isfinite(self.theta):
            raise ValueError("theta must be finite.")
        if self.theta <= 0.0:
            raise ValueError("theta must be > 0.")

        # Validate xi (vol of vol).
        if not np.isfinite(self.xi):
            raise ValueError("xi must be finite.")
        if self.xi <= 0.0:
            raise ValueError("xi must be > 0.")

        # Validate v0 (initial variance).
        if not np.isfinite(self.v0):
            raise ValueError("v0 must be finite.")
        if self.v0 <= 0.0:
            raise ValueError("v0 must be > 0.")

        # Validate rho (correlation).
        if not np.isfinite(self.rho):
            raise ValueError("rho must be finite.")
        if self.rho <= -1.0 or self.rho >= 1.0:
            raise ValueError("rho must be in (-1, 1).")

    @property
    def feller_ratio(self) -> float:
        """
        Feller ratio: 2κθ/ξ².

        If > 1, the Feller condition is satisfied and variance stays positive.
        """
        return 2.0 * self.kappa * self.theta / (self.xi ** 2)

    @property
    def feller_satisfied(self) -> bool:
        """Check if Feller condition 2κθ > ξ² is satisfied."""
        return self.feller_ratio > 1.0

    @property
    def long_term_vol(self) -> float:
        """Long-term volatility level √θ."""
        return np.sqrt(self.theta)

    @property
    def initial_vol(self) -> float:
        """Initial volatility level √V_0."""
        return np.sqrt(self.v0)

    def expected_variance(self, t: float) -> float:
        """
        Expected variance E[V_t] under risk-neutral measure.

        E[V_t] = θ + (V_0 - θ) × e^(-κt)
        """
        return self.theta + (self.v0 - self.theta) * np.exp(-self.kappa * t)


# =============================================================================
# Heston Simulation Output
# =============================================================================

@dataclass(frozen=True, slots=True)
class HestonSimulation:
    """
    Output container for Heston path simulation.

    Attributes
    ----------
    spot_paths : np.ndarray
        Simulated spot paths, shape (n_paths, n_steps + 1).
    variance_paths : np.ndarray
        Simulated variance paths, shape (n_paths, n_steps + 1).
    times : np.ndarray
        Time grid, shape (n_steps + 1,).
    params : HestonParameters
        Heston parameters used in simulation.
    n_paths : int
        Number of simulated paths.
    n_steps : int
        Number of time steps.
    scheme : HestonScheme
        Discretization scheme used.
    seed : int or None
        Random seed used.
    """

    spot_paths: np.ndarray
    variance_paths: np.ndarray
    times: np.ndarray
    params: HestonParameters
    n_paths: int
    n_steps: int
    scheme: HestonScheme
    seed: Optional[int]

    @property
    def terminal_spots(self) -> np.ndarray:
        """Terminal spot values S_T."""
        return self.spot_paths[:, -1]

    @property
    def terminal_variances(self) -> np.ndarray:
        """Terminal variance values V_T."""
        return self.variance_paths[:, -1]

    @property
    def maturity(self) -> float:
        """Time to maturity T."""
        return float(self.times[-1])


# =============================================================================
# Heston Dynamics Simulator
# =============================================================================

@dataclass(frozen=True, slots=True)
class HestonDynamics:
    """
    Simulator for Heston stochastic volatility dynamics.

    Simulates the joint process (S_t, V_t) under the risk-neutral measure:
        dS_t = (r - q) S_t dt + √V_t S_t dW_t^S
        dV_t = κ(θ - V_t) dt + ξ √V_t dW_t^V

    Parameters
    ----------
    params : HestonParameters
        Heston model parameters (κ, θ, ξ, V_0, ρ).
    drift : float
        Drift coefficient μ = r - q (risk-neutral drift).

    Examples
    --------
    >>> from src.models.stochastic_vol.heston import HestonDynamics, HestonParameters
    >>> params = HestonParameters(kappa=2.0, theta=0.04, xi=0.3, v0=0.04, rho=-0.7)
    >>> dynamics = HestonDynamics(params=params, drift=0.03)
    >>> sim = dynamics.simulate(
    ...     spot0=100.0, maturity=1.0, n_paths=10000, n_steps=252
    ... )
    >>> sim.terminal_spots.mean()  # Should be around 100 * exp(0.03)
    """

    params: HestonParameters
    drift: float  # μ = r - q.

    def simulate(
        self,
        spot0: float,
        maturity: float,
        n_paths: int,
        n_steps: int,
        scheme: HestonScheme = "full_truncation",
        seed: Optional[int] = None,
        antithetic: bool = True,
    ) -> HestonSimulation:
        """
        Simulate Heston paths.

        Parameters
        ----------
        spot0 : float
            Initial spot price S_0.
        maturity : float
            Time to maturity T.
        n_paths : int
            Number of paths to simulate.
        n_steps : int
            Number of time steps.
        scheme : HestonScheme
            Discretization scheme for variance process.
            - "euler": Simple Euler (may go negative).
            - "full_truncation": max(V, 0) truncation.
            - "reflection": |V| reflection.
            - "qe": Quadratic-exponential (advanced).
        seed : int, optional
            Random seed for reproducibility.
        antithetic : bool
            Use antithetic variates for variance reduction.

        Returns
        -------
        HestonSimulation
            Container with simulated paths and metadata.
        """
        # Validate inputs.
        if spot0 <= 0.0:
            raise ValueError("spot0 must be > 0.")
        if maturity <= 0.0:
            raise ValueError("maturity must be > 0.")
        if n_paths <= 0:
            raise ValueError("n_paths must be > 0.")
        if n_steps <= 0:
            raise ValueError("n_steps must be > 0.")

        # Set random seed.
        if seed is not None:
            np.random.seed(seed)

        # Time discretization.
        dt = maturity / n_steps
        sqrt_dt = np.sqrt(dt)
        times = np.linspace(0.0, maturity, n_steps + 1)

        # Handle antithetic variates.
        if antithetic:
            n_base = (n_paths + 1) // 2
            n_actual = 2 * n_base
        else:
            n_base = n_paths
            n_actual = n_paths

        # Generate correlated Brownian increments.
        # dW^S and dW^V with correlation ρ.
        Z1 = np.random.standard_normal((n_base, n_steps))  # For spot.
        Z2 = np.random.standard_normal((n_base, n_steps))  # Independent.

        # Correlate: dW^V = ρ dW^S + √(1-ρ²) dZ_independent.
        rho = self.params.rho
        sqrt_1_rho2 = np.sqrt(1.0 - rho**2)

        dW_S_base = sqrt_dt * Z1
        dW_V_base = sqrt_dt * (rho * Z1 + sqrt_1_rho2 * Z2)

        if antithetic:
            dW_S = np.vstack([dW_S_base, -dW_S_base])
            dW_V = np.vstack([dW_V_base, -dW_V_base])
        else:
            dW_S = dW_S_base
            dW_V = dW_V_base

        # Initialize path arrays.
        S = np.zeros((n_actual, n_steps + 1))
        V = np.zeros((n_actual, n_steps + 1))
        S[:, 0] = spot0
        V[:, 0] = self.params.v0

        # Extract parameters.
        kappa = self.params.kappa
        theta = self.params.theta
        xi = self.params.xi
        mu = self.drift

        # Simulate paths step by step.
        for i in range(n_steps):
            V_curr = V[:, i]
            S_curr = S[:, i]

            # Apply variance scheme.
            if scheme == "euler":
                V_next = self._euler_step_variance(V_curr, kappa, theta, xi, dt, dW_V[:, i])
            elif scheme == "full_truncation":
                V_next = self._full_truncation_step(V_curr, kappa, theta, xi, dt, dW_V[:, i])
            elif scheme == "reflection":
                V_next = self._reflection_step(V_curr, kappa, theta, xi, dt, dW_V[:, i])
            elif scheme == "qe":
                V_next = self._qe_step(V_curr, kappa, theta, xi, dt)
            else:
                raise ValueError(f"Unknown scheme: {scheme}")

            # Spot step (log-Euler for positivity).
            sqrt_V = np.sqrt(np.maximum(V_curr, 0.0))
            log_increment = (mu - 0.5 * V_curr) * dt + sqrt_V * dW_S[:, i]
            S_next = S_curr * np.exp(log_increment)

            V[:, i + 1] = V_next
            S[:, i + 1] = S_next

        return HestonSimulation(
            spot_paths=S,
            variance_paths=V,
            times=times,
            params=self.params,
            n_paths=n_actual,
            n_steps=n_steps,
            scheme=scheme,
            seed=seed,
        )

    def _euler_step_variance(
        self,
        V: np.ndarray,
        kappa: float,
        theta: float,
        xi: float,
        dt: float,
        dW: np.ndarray,
    ) -> np.ndarray:
        """
        Euler step for variance (may go negative).

        dV = κ(θ - V) dt + ξ√V dW
        """
        sqrt_V = np.sqrt(np.maximum(V, 0.0))
        return V + kappa * (theta - V) * dt + xi * sqrt_V * dW

    def _full_truncation_step(
        self,
        V: np.ndarray,
        kappa: float,
        theta: float,
        xi: float,
        dt: float,
        dW: np.ndarray,
    ) -> np.ndarray:
        """
        Full truncation scheme: use max(V, 0) in diffusion.

        This is the most common simple fix for negative variance.
        """
        V_pos = np.maximum(V, 0.0)
        sqrt_V = np.sqrt(V_pos)
        V_next = V + kappa * (theta - V_pos) * dt + xi * sqrt_V * dW
        return np.maximum(V_next, 0.0)

    def _reflection_step(
        self,
        V: np.ndarray,
        kappa: float,
        theta: float,
        xi: float,
        dt: float,
        dW: np.ndarray,
    ) -> np.ndarray:
        """
        Reflection scheme: reflect negative variance to positive.

        V_next = |V + drift + diffusion| if V + drift + diffusion < 0.
        """
        V_pos = np.maximum(V, 0.0)
        sqrt_V = np.sqrt(V_pos)
        V_next = V + kappa * (theta - V_pos) * dt + xi * sqrt_V * dW
        return np.abs(V_next)

    def _qe_step(
        self,
        V: np.ndarray,
        kappa: float,
        theta: float,
        xi: float,
        dt: float,
    ) -> np.ndarray:
        """
        Quadratic-Exponential (QE) scheme (Andersen, 2008).

        Advanced scheme that exactly matches first two moments.
        Simplified implementation here.
        """
        # Moment matching parameters.
        c1 = np.exp(-kappa * dt)
        c2 = (xi**2 * c1 / kappa) * (1 - c1)

        # Mean and variance of V_{t+dt} | V_t.
        m = theta + (V - theta) * c1
        s2 = V * c2 + theta * (xi**2 / (2 * kappa)) * (1 - c1)**2

        # Use different approximations based on ψ = s²/m².
        psi = s2 / (m**2 + 1e-10)
        psi_crit = 1.5

        # Allocate output.
        V_next = np.zeros_like(V)

        # Quadratic scheme for psi <= psi_crit.
        mask_quad = psi <= psi_crit
        if np.any(mask_quad):
            inv_psi = 1.0 / (psi[mask_quad] + 1e-10)
            b2 = 2 * inv_psi - 1 + np.sqrt(2 * inv_psi) * np.sqrt(2 * inv_psi - 1)
            b = np.sqrt(np.maximum(b2, 0.0))
            a = m[mask_quad] / (1 + b2)
            Z = np.random.standard_normal(np.sum(mask_quad))
            V_next[mask_quad] = a * (b + Z)**2

        # Exponential scheme for psi > psi_crit.
        mask_exp = ~mask_quad
        if np.any(mask_exp):
            p = (psi[mask_exp] - 1) / (psi[mask_exp] + 1)
            beta = (1 - p) / (m[mask_exp] + 1e-10)
            U = np.random.uniform(0, 1, np.sum(mask_exp))
            V_next[mask_exp] = np.where(
                U <= p,
                0.0,
                np.log((1 - p) / (1 - U + 1e-10)) / beta,
            )

        return np.maximum(V_next, 0.0)

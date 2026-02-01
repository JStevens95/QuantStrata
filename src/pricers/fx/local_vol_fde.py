"""
Local Volatility Finite Difference Pricer for FX Options.

This module provides finite-difference (PDE) pricing of European options
under the local volatility model where σ = σ(S, t).

Mathematical Framework
----------------------
Under the local volatility model, the spot price follows:

    dS_t = (r_d - r_f) S_t dt + σ(S_t, t) S_t dW_t

The option value V(t, S) satisfies the PDE:

    ∂V/∂t + (r_d - r_f)S ∂V/∂S + ½σ(S,t)² S² ∂²V/∂S² - r_d V = 0

with terminal condition V(T, S) = payoff(S).

Key Difference from Constant Vol
--------------------------------
- The diffusion coefficient ½σ²S² now depends on both S and t.
- At each time step, we must evaluate σ(S, t) at all grid points.
- The tridiagonal system coefficients become time and space dependent.

Implementation Notes
--------------------
1. We solve backward from T to 0.
2. At each time step, we construct the operator using σ(S, t_n).
3. Crank-Nicolson (theta=0.5) for second-order accuracy.
4. Log-space transformation recommended for stability.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Protocol, Union

from src.instruments.fx.options.vanilla import FxVanillaEuropeanOption
from src.marketdata.core.market import Market
from src.marketdata.surfaces.local_vol_surface import LocalVolSurface, FlatLocalVolSurface
from src.models.payoffs.types import OptionType


# =============================================================================
# Type definitions
# =============================================================================

# Protocol for local vol surface callable.
class LocalVolCallable(Protocol):
    """Protocol for local volatility function σ(S, t)."""

    def __call__(self, spot: float, time: float) -> float:
        """Return local volatility at (spot, time)."""
        ...


# Union of supported local vol types.
LocalVolInput = Union[LocalVolSurface, FlatLocalVolSurface, LocalVolCallable, float]


# =============================================================================
# Local Vol FD Pricer
# =============================================================================

@dataclass(frozen=True, slots=True)
class FxLocalVolEuropeanOptionFdPricer:
    """
    Finite-difference pricer for European FX options under local volatility.

    Solves the PDE:
        ∂V/∂t + (r-q)S ∂V/∂S + ½σ(S,t)² S² ∂²V/∂S² - rV = 0

    Parameters
    ----------
    local_vol_surface : LocalVolInput
        Local volatility surface σ(S, t). Can be:
        - LocalVolSurface or FlatLocalVolSurface object
        - Callable (S, t) -> σ
        - float (constant vol)
    n_space : int
        Number of spatial grid points. Default: 401.
    n_time_steps : int
        Number of time steps. Default: 200.
    n_std : float
        Number of standard deviations for grid bounds. Default: 6.0.
    theta : float
        Time-stepping scheme (0.5=Crank-Nicolson, 1.0=implicit). Default: 0.5.
    use_log_space : bool
        If True, solve in log-space (recommended). Default: True.

    Examples
    --------
    >>> from src.pricers.fx.local_vol_fde import FxLocalVolEuropeanOptionFdPricer
    >>> from src.marketdata.surfaces.local_vol_surface import FlatLocalVolSurface
    >>> # Create a flat local vol surface (equivalent to BSM).
    >>> local_vol = FlatLocalVolSurface(sigma=0.20)
    >>> pricer = FxLocalVolFdPricer(local_vol_surface=local_vol)
    >>> # Price call option.
    >>> price = pricer.price_european(
    ...     spot=100.0, strike=100.0, maturity=1.0,
    ...     domestic_rate=0.05, foreign_rate=0.02,
    ...     option_type="call"
    ... )
    """

    local_vol_surface: LocalVolInput
    n_space: int = 401
    n_time_steps: int = 200
    n_std: float = 6.0
    theta: float = 0.5
    use_log_space: bool = True

    def __post_init__(self) -> None:
        """Validate pricer configuration."""
        if self.n_space < 10:
            raise ValueError("n_space must be >= 10.")
        if self.n_time_steps < 1:
            raise ValueError("n_time_steps must be >= 1.")
        if self.n_std <= 0.0:
            raise ValueError("n_std must be > 0.")
        if not (0.0 <= self.theta <= 1.0):
            raise ValueError("theta must be in [0, 1].")

    def _get_local_vol(self, spot: float, time: float) -> float:
        """Get local volatility at (spot, time)."""
        lv = self.local_vol_surface
        if isinstance(lv, (int, float)):
            return float(lv)
        elif isinstance(lv, (LocalVolSurface, FlatLocalVolSurface)):
            return lv.local_vol(spot, time)
        elif callable(lv):
            return lv(spot, time)
        else:
            raise TypeError(f"Unsupported local_vol_surface type: {type(lv)}")

    def price_european(
        self,
        spot: float,
        strike: float,
        maturity: float,
        domestic_rate: float,
        foreign_rate: float,
        option_type: OptionType,
    ) -> float:
        """
        Price a European vanilla option under local volatility.

        Parameters
        ----------
        spot : float
            Current spot price S_0.
        strike : float
            Option strike K.
        maturity : float
            Time to maturity T.
        domestic_rate : float
            Domestic risk-free rate r_d (continuous).
        foreign_rate : float
            Foreign rate r_f (continuous).
        option_type : OptionType
            "call" or "put".

        Returns
        -------
        float
            Option price.
        """
        # Validate inputs.
        if spot <= 0.0:
            raise ValueError("spot must be > 0.")
        if strike <= 0.0:
            raise ValueError("strike must be > 0.")
        if maturity < 0.0:
            raise ValueError("maturity must be >= 0.")

        # Handle T=0 case (intrinsic value).
        if maturity == 0.0:
            if option_type == "call":
                return max(spot - strike, 0.0)
            else:
                return max(strike - spot, 0.0)

        # Get representative volatility for grid sizing.
        sigma_ref = self._get_local_vol(spot, maturity / 2)

        # Build spatial grid.
        if self.use_log_space:
            return self._price_log_space(
                spot, strike, maturity, domestic_rate, foreign_rate, option_type, sigma_ref
            )
        else:
            return self._price_spot_space(
                spot, strike, maturity, domestic_rate, foreign_rate, option_type, sigma_ref
            )

    def _price_log_space(
        self,
        spot: float,
        strike: float,
        maturity: float,
        r_d: float,
        r_f: float,
        option_type: OptionType,
        sigma_ref: float,
    ) -> float:
        """Price in log-space coordinates."""
        T = maturity
        S0 = spot
        K = strike

        # Log-space coordinates.
        x0 = np.log(S0)

        # Grid bounds in log-space.
        drift = (r_d - r_f - 0.5 * sigma_ref**2)
        vol_sqrt_T = sigma_ref * np.sqrt(T)
        x_min = x0 + drift * T - self.n_std * vol_sqrt_T
        x_max = x0 + drift * T + self.n_std * vol_sqrt_T

        # Spatial grid.
        x = np.linspace(x_min, x_max, self.n_space)
        dx = x[1] - x[0]

        # Time grid.
        dt = T / self.n_time_steps

        # Terminal condition: payoff in spot-space.
        S_grid = np.exp(x)
        if option_type == "call":
            V = np.maximum(S_grid - K, 0.0)
        else:
            V = np.maximum(K - S_grid, 0.0)

        # Backward time-stepping.
        for n in range(self.n_time_steps):
            # Current time (going backward).
            t_curr = T - n * dt
            t_next = T - (n + 1) * dt

            # Get local vol at each grid point for current time.
            sigma = np.array([self._get_local_vol(S, t_curr) for S in S_grid])

            # PDE coefficients in log-space.
            # ∂V/∂t + (r-q-½σ²)∂V/∂x + ½σ² ∂²V/∂x² - rV = 0
            # Let μ = r_d - r_f - ½σ², D = ½σ²
            mu = (r_d - r_f - 0.5 * sigma**2)
            D = 0.5 * sigma**2

            # Tridiagonal coefficients for Crank-Nicolson.
            # At interior points, the scheme is:
            # (V_n - V_{n+1})/dt = θ L[V_n] + (1-θ) L[V_{n+1}]
            # where L[V] = D ∂²V/∂x² + μ ∂V/∂x - r V

            # Coefficients: a_i V_{i-1} + b_i V_i + c_i V_{i+1} = d_i
            alpha = D / dx**2 - mu / (2 * dx)  # coefficient for V_{i-1}
            beta = -2 * D / dx**2 - r_d         # coefficient for V_i
            gamma = D / dx**2 + mu / (2 * dx)  # coefficient for V_{i+1}

            # LHS: (I - θ dt L)
            a_lhs = -self.theta * dt * alpha[1:-1]
            b_lhs = 1 - self.theta * dt * beta[1:-1]
            c_lhs = -self.theta * dt * gamma[1:-1]

            # RHS: (I + (1-θ) dt L) V
            theta_bar = 1 - self.theta
            a_rhs = theta_bar * dt * alpha[1:-1]
            b_rhs = 1 + theta_bar * dt * beta[1:-1]
            c_rhs = theta_bar * dt * gamma[1:-1]

            # Build RHS vector.
            rhs = np.zeros(self.n_space - 2)
            rhs[:] = a_rhs * V[:-2] + b_rhs * V[1:-1] + c_rhs * V[2:]

            # Boundary conditions (Dirichlet).
            # At x_min (far out-of-the-money): V → 0 for call, V → K*e^(-r*t) - S*e^(-q*t) for put.
            # At x_max (deep in-the-money): V → S - K*e^(-r*t) for call, V → 0 for put.
            if option_type == "call":
                V_left = 0.0
                V_right = S_grid[-1] - K * np.exp(-r_d * t_next)
            else:
                V_left = K * np.exp(-r_d * t_next) - S_grid[0]
                V_right = 0.0

            # Adjust RHS for boundary.
            rhs[0] += self.theta * dt * alpha[1] * V_left + theta_bar * dt * alpha[1] * V[0]
            rhs[-1] += self.theta * dt * gamma[-2] * V_right + theta_bar * dt * gamma[-2] * V[-1]

            # Solve tridiagonal system.
            V_interior = _solve_tridiag(a_lhs, b_lhs, c_lhs, rhs)

            # Update V.
            V_new = np.zeros(self.n_space)
            V_new[0] = V_left
            V_new[-1] = V_right
            V_new[1:-1] = V_interior
            V = V_new

        # Interpolate to get price at S0.
        return float(np.interp(x0, x, V))

    def _price_spot_space(
        self,
        spot: float,
        strike: float,
        maturity: float,
        r_d: float,
        r_f: float,
        option_type: OptionType,
        sigma_ref: float,
    ) -> float:
        """Price in spot-space coordinates (less stable)."""
        T = maturity
        S0 = spot
        K = strike

        # Grid bounds.
        S_min = max(S0 * np.exp(-self.n_std * sigma_ref * np.sqrt(T)), 1e-6)
        S_max = S0 * np.exp(self.n_std * sigma_ref * np.sqrt(T))

        # Spatial grid.
        S = np.linspace(S_min, S_max, self.n_space)
        dS = S[1] - S[0]

        # Time grid.
        dt = T / self.n_time_steps

        # Terminal condition.
        if option_type == "call":
            V = np.maximum(S - K, 0.0)
        else:
            V = np.maximum(K - S, 0.0)

        # Backward time-stepping.
        for n in range(self.n_time_steps):
            t_curr = T - n * dt
            t_next = T - (n + 1) * dt

            # Get local vol at each grid point.
            sigma = np.array([self._get_local_vol(s, t_curr) for s in S])

            # PDE coefficients in spot-space.
            # ∂V/∂t + (r-q)S ∂V/∂S + ½σ²S² ∂²V/∂S² - rV = 0
            mu = r_d - r_f
            D = 0.5 * sigma**2 * S**2

            # Central differences.
            alpha = D / dS**2 - mu * S / (2 * dS)
            beta = -2 * D / dS**2 - r_d
            gamma = D / dS**2 + mu * S / (2 * dS)

            # Crank-Nicolson coefficients.
            a_lhs = -self.theta * dt * alpha[1:-1]
            b_lhs = 1 - self.theta * dt * beta[1:-1]
            c_lhs = -self.theta * dt * gamma[1:-1]

            theta_bar = 1 - self.theta
            a_rhs = theta_bar * dt * alpha[1:-1]
            b_rhs = 1 + theta_bar * dt * beta[1:-1]
            c_rhs = theta_bar * dt * gamma[1:-1]

            rhs = a_rhs * V[:-2] + b_rhs * V[1:-1] + c_rhs * V[2:]

            # Boundary conditions.
            if option_type == "call":
                V_left = 0.0
                V_right = S[-1] - K * np.exp(-r_d * t_next)
            else:
                V_left = K * np.exp(-r_d * t_next) - S[0]
                V_right = 0.0

            rhs[0] += self.theta * dt * alpha[1] * V_left
            rhs[-1] += self.theta * dt * gamma[-2] * V_right

            V_interior = _solve_tridiag(a_lhs, b_lhs, c_lhs, rhs)

            V_new = np.zeros(self.n_space)
            V_new[0] = V_left
            V_new[-1] = V_right
            V_new[1:-1] = V_interior
            V = V_new

        return float(np.interp(S0, S, V))


# =============================================================================
# Helper: Tridiagonal solver
# =============================================================================

def _solve_tridiag(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> np.ndarray:
    """
    Solve tridiagonal system Ax = d using Thomas algorithm.

    The matrix A has:
    - a[i] on sub-diagonal (i = 1, ..., n-1)
    - b[i] on diagonal (i = 0, ..., n-1)
    - c[i] on super-diagonal (i = 0, ..., n-2)

    Parameters
    ----------
    a : np.ndarray
        Sub-diagonal, length n.
    b : np.ndarray
        Diagonal, length n.
    c : np.ndarray
        Super-diagonal, length n.
    d : np.ndarray
        RHS vector, length n.

    Returns
    -------
    np.ndarray
        Solution vector x, length n.
    """
    n = len(d)
    if n == 0:
        return np.array([])

    # Copy to avoid modifying inputs.
    c_prime = np.zeros(n)
    d_prime = np.zeros(n)

    # Forward sweep.
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] - a[i] * c_prime[i - 1]
        if i < n - 1:
            c_prime[i] = c[i] / denom
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / denom

    # Back substitution.
    x = np.zeros(n)
    x[-1] = d_prime[-1]

    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]

    return x

"""
Dupire Local Volatility Calibration.

This module implements Dupire's formula for extracting local volatility from
a surface of implied volatilities (or call prices).

Mathematical Framework
----------------------
Dupire (1994) showed that the unique local volatility consistent with
European call prices C(K, T) is given by:

    σ_LV²(K, T) = [∂C/∂T + (r-q)K ∂C/∂K + qC] / [½K² ∂²C/∂K²]

This can be rewritten in terms of implied volatility σ_BS(K, T):

    σ_LV²(K, T) = [∂w/∂T] / [1 - (y/w)(∂w/∂y) + ¼(-¼ - 1/w + y²/w²)(∂w/∂y)² + ½(∂²w/∂y²)]

where:
    - w = σ_BS² × T (total variance)
    - y = ln(K/F) (log-moneyness)
    - F = forward price

Key Implementation Details
--------------------------
1. Derivatives are computed numerically using finite differences.
2. Care is taken near ATM and short expiries where derivatives can be unstable.
3. Output is clamped to avoid negative local variances (arbitrage).
4. Grid is typically denser near ATM for accuracy.

References
----------
- Dupire, B. (1994). "Pricing with a Smile." Risk, 7(1), 18-20.
- Gatheral, J. (2006). "The Volatility Surface: A Practitioner's Guide."
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Protocol
from scipy.stats import norm

from src.marketdata.surfaces.local_vol_surface import LocalVolSurface


# =============================================================================
# Protocol for implied vol surface input
# =============================================================================

class ImpliedVolSurface(Protocol):
    """Protocol for implied volatility surfaces used as input to Dupire calibration."""

    def implied_vol(self, expiry: float, strike: float) -> float:
        """Return implied volatility at (expiry, strike)."""
        ...


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True, slots=True)
class DupireConfig:
    """
    Configuration for Dupire local volatility calibration.

    Parameters
    ----------
    dT : float
        Time step for ∂/∂T finite difference. Default 1/252 (1 day).
    dK_pct : float
        Strike step as percentage of strike for ∂/∂K. Default 0.01 (1%).
    min_local_vol : float
        Minimum allowed local vol (clamp negative values). Default 0.01.
    max_local_vol : float
        Maximum allowed local vol (clamp extreme values). Default 2.0.
    min_expiry : float
        Minimum expiry for calibration (avoid T=0 singularity). Default 1/365.
    use_forward_space : bool
        If True, work in forward moneyness space. Default True.
    """

    dT: float = 1.0 / 252.0           # 1 trading day.
    dK_pct: float = 0.01              # 1% of strike.
    min_local_vol: float = 0.01       # 1% floor.
    max_local_vol: float = 2.0        # 200% cap.
    min_expiry: float = 1.0 / 365.0   # Minimum 1 day.
    use_forward_space: bool = True    # Use forward moneyness.

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.dT <= 0.0:
            raise ValueError("dT must be > 0.")
        if self.dK_pct <= 0.0:
            raise ValueError("dK_pct must be > 0.")
        if self.min_local_vol <= 0.0:
            raise ValueError("min_local_vol must be > 0.")
        if self.max_local_vol <= self.min_local_vol:
            raise ValueError("max_local_vol must be > min_local_vol.")
        if self.min_expiry <= 0.0:
            raise ValueError("min_expiry must be > 0.")


# =============================================================================
# Black-Scholes helpers
# =============================================================================

def _bs_call_price(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
) -> float:
    """Black-Scholes call price."""
    if T <= 0.0:
        return max(S - K, 0.0)
    if sigma <= 0.0:
        # Zero vol: deterministic payoff.
        F = S * np.exp((r - q) * T)
        return max(F - K, 0.0) * np.exp(-r * T)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def _forward_price(S: float, r: float, q: float, T: float) -> float:
    """Forward price F = S * exp((r-q)*T)."""
    return S * np.exp((r - q) * T)


# =============================================================================
# Dupire Calibrator
# =============================================================================

@dataclass(frozen=True, slots=True)
class DupireCalibrator:
    """
    Calibrator for extracting local volatility from implied volatility surfaces.

    This implements Dupire's formula using numerical differentiation of call
    prices (computed from implied vols via Black-Scholes).

    Parameters
    ----------
    config : DupireConfig
        Configuration for the calibration (finite difference steps, bounds, etc.).

    Examples
    --------
    >>> from src.calibration.local_vol.dupire import DupireCalibrator, DupireConfig
    >>> from src.marketdata.surfaces.vol_surface import FlatVolSurface
    >>> # Create a flat implied vol surface (should give constant local vol).
    >>> implied_surface = FlatVolSurface(sigma=0.20)
    >>> calibrator = DupireCalibrator(config=DupireConfig())
    >>> # Calibrate local vol at specific points.
    >>> local_vol = calibrator.local_vol_at_point(
    ...     implied_surface=implied_surface,
    ...     spot=100.0, strike=100.0, expiry=1.0,
    ...     r=0.05, q=0.02
    ... )
    >>> abs(local_vol - 0.20) < 0.01  # Should be close to 0.20.
    True
    """

    config: DupireConfig = DupireConfig()

    def local_vol_at_point(
        self,
        implied_surface: ImpliedVolSurface,
        spot: float,
        strike: float,
        expiry: float,
        r: float,
        q: float,
    ) -> float:
        """
        Compute local volatility at a single (strike, expiry) point.

        Uses Dupire's formula with numerical differentiation of call prices.

        Parameters
        ----------
        implied_surface : ImpliedVolSurface
            Surface providing implied_vol(expiry, strike).
        spot : float
            Current spot price S_0.
        strike : float
            Strike K at which to compute local vol.
        expiry : float
            Time to expiry T.
        r : float
            Risk-free rate (continuous).
        q : float
            Dividend/foreign rate (continuous).

        Returns
        -------
        float
            Local volatility σ_LV(K, T), clamped to [min_local_vol, max_local_vol].
        """
        S = float(spot)
        K = float(strike)
        T = float(expiry)

        # Handle edge cases.
        if T < self.config.min_expiry:
            # At very short expiry, return ATM implied vol.
            return float(np.clip(
                implied_surface.implied_vol(T, K),
                self.config.min_local_vol,
                self.config.max_local_vol,
            ))

        # Compute call price and its derivatives using finite differences.
        dT = self.config.dT
        dK = K * self.config.dK_pct

        # Get implied vols at grid points.
        def call_price(T_q: float, K_q: float) -> float:
            """Compute call price from implied vol."""
            if T_q <= 0.0:
                return max(S - K_q, 0.0)
            sigma = implied_surface.implied_vol(T_q, K_q)
            return _bs_call_price(S, K_q, T_q, r, q, sigma)

        # Central price.
        C = call_price(T, K)

        # Partial derivative ∂C/∂T (forward difference for stability).
        C_Tplus = call_price(T + dT, K)
        dC_dT = (C_Tplus - C) / dT

        # Partial derivative ∂C/∂K (central difference).
        C_Kplus = call_price(T, K + dK)
        C_Kminus = call_price(T, K - dK)
        dC_dK = (C_Kplus - C_Kminus) / (2 * dK)

        # Second derivative ∂²C/∂K² (central difference).
        d2C_dK2 = (C_Kplus - 2 * C + C_Kminus) / (dK**2)

        # Dupire's formula:
        # σ_LV²(K, T) = [∂C/∂T + (r-q)K ∂C/∂K + qC] / [½K² ∂²C/∂K²]
        numerator = dC_dT + (r - q) * K * dC_dK + q * C
        denominator = 0.5 * K**2 * d2C_dK2

        # Handle edge cases.
        if denominator <= 0.0:
            # Arbitrage or numerical instability: fall back to implied vol.
            sigma_imp = implied_surface.implied_vol(T, K)
            return float(np.clip(
                sigma_imp,
                self.config.min_local_vol,
                self.config.max_local_vol,
            ))

        local_var = numerator / denominator

        if local_var <= 0.0:
            # Negative variance indicates arbitrage: use implied vol.
            sigma_imp = implied_surface.implied_vol(T, K)
            return float(np.clip(
                sigma_imp,
                self.config.min_local_vol,
                self.config.max_local_vol,
            ))

        local_vol = np.sqrt(local_var)

        # Clamp to bounds.
        return float(np.clip(
            local_vol,
            self.config.min_local_vol,
            self.config.max_local_vol,
        ))

    def calibrate_grid(
        self,
        implied_surface: ImpliedVolSurface,
        spot: float,
        r: float,
        q: float,
        times: np.ndarray,
        spots: np.ndarray,
    ) -> LocalVolSurface:
        """
        Calibrate a full LocalVolSurface on a grid of (time, spot) points.

        Parameters
        ----------
        implied_surface : ImpliedVolSurface
            Surface providing implied_vol(expiry, strike).
        spot : float
            Current spot price S_0 (used for forward calculation).
        r : float
            Risk-free rate (continuous).
        q : float
            Dividend/foreign rate (continuous).
        times : np.ndarray
            Time grid for local vol surface (year fractions).
        spots : np.ndarray
            Spot grid for local vol surface.

        Returns
        -------
        LocalVolSurface
            Calibrated local volatility surface.

        Notes
        -----
        The grid (times, spots) defines where σ_LV(S, t) is computed.
        In Dupire's formula, we compute σ_LV at (K=S_grid, T=times).
        This is because local vol σ(S, t) and Dupire's σ(K, T) coincide
        when evaluated at the same point.
        """
        times = np.asarray(times, dtype=float).reshape(-1)
        spots = np.asarray(spots, dtype=float).reshape(-1)

        n_times = len(times)
        n_spots = len(spots)

        # Allocate output grid.
        local_vols = np.zeros((n_times, n_spots), dtype=float)

        # Calibrate at each grid point.
        for i, t in enumerate(times):
            for j, s in enumerate(spots):
                # At spot grid point s, strike K = s (local vol at spot level).
                local_vols[i, j] = self.local_vol_at_point(
                    implied_surface=implied_surface,
                    spot=spot,
                    strike=s,  # K = spot grid point.
                    expiry=t,
                    r=r,
                    q=q,
                )

        return LocalVolSurface(
            times=times,
            spots=spots,
            local_vols=local_vols,
        )


# =============================================================================
# Convenience function
# =============================================================================

def calibrate_local_vol_from_implied(
    implied_surface: ImpliedVolSurface,
    spot: float,
    r: float,
    q: float,
    times: Optional[np.ndarray] = None,
    spots: Optional[np.ndarray] = None,
    config: Optional[DupireConfig] = None,
) -> LocalVolSurface:
    """
    Convenience function to calibrate local vol surface from implied vol surface.

    Parameters
    ----------
    implied_surface : ImpliedVolSurface
        Input implied volatility surface.
    spot : float
        Current spot price.
    r : float
        Risk-free rate.
    q : float
        Dividend/foreign rate.
    times : np.ndarray, optional
        Time grid. Default: [0.01, 0.1, 0.25, 0.5, 1.0, 2.0].
    spots : np.ndarray, optional
        Spot grid. Default: spot * [0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3].
    config : DupireConfig, optional
        Calibration configuration. Default: DupireConfig().

    Returns
    -------
    LocalVolSurface
        Calibrated local volatility surface.

    Examples
    --------
    >>> from src.calibration.local_vol import calibrate_local_vol_from_implied
    >>> from src.marketdata.surfaces.vol_surface import FlatVolSurface
    >>> implied = FlatVolSurface(sigma=0.20)
    >>> local_vol_surface = calibrate_local_vol_from_implied(
    ...     implied_surface=implied,
    ...     spot=100.0,
    ...     r=0.05,
    ...     q=0.02,
    ... )
    >>> local_vol_surface.local_vol(spot=100.0, time=0.5)  # Close to 0.20.
    """
    # Default grids.
    if times is None:
        times = np.array([0.01, 0.1, 0.25, 0.5, 1.0, 2.0])
    if spots is None:
        spots = spot * np.array([0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3])
    if config is None:
        config = DupireConfig()

    calibrator = DupireCalibrator(config=config)
    return calibrator.calibrate_grid(
        implied_surface=implied_surface,
        spot=spot,
        r=r,
        q=q,
        times=times,
        spots=spots,
    )

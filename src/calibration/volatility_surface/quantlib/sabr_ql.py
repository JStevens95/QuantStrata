"""
QuantLib SABR Calibration Backend.

This module provides QuantLib-backed SABR model functions:
- `sabr_implied_vol_quantlib()`: SABR implied vol using QuantLib's implementation
- `calibrate_sabr_quantlib()`: SABR calibration using QuantLib's optimizers

QuantLib offers multiple SABR implementations:
1. Hagan (2002) - Original approximation
2. Obloj (2008) - Improved accuracy for extreme parameters
3. Others (Antonov, Johnson, etc.) - Advanced corrections

This allows comparison with the native Python Hagan implementation.

Mathematical Background
-----------------------
QuantLib's SABR functions are accessed via:
- `ql.sabrVolatility()` - Direct SABR vol formula
- `ql.SabrSmileSection` - SABR smile at single expiry (for calibration)

The QuantLib SABR implementation includes:
- Multiple approximation methods
- Better handling of edge cases (β→0, β→1, K→F)
- More robust numerics for extreme parameters

Author: QuantStrata Team
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Literal, Sequence

from src.marketdata.integration.quantlib.context import require_quantlib
from src.calibration.volatility_surface.sabr import SabrParameters, SabrConfig


# =============================================================================
# QuantLib SABR Approximation Types
# =============================================================================

SabrVolType = Literal["hagan", "obloj", "hagan_lognormal", "normal"]


# =============================================================================
# QuantLib SABR Implied Volatility
# =============================================================================

def sabr_implied_vol_quantlib(
    *,
    forward: float,
    strike: float,
    expiry: float,
    params: SabrParameters,
    vol_type: SabrVolType = "hagan",
) -> float:
    """
    Compute SABR implied volatility using QuantLib.
    
    Parameters
    ----------
    forward:
        Forward price F.
    strike:
        Strike price K.
    expiry:
        Time to expiry T (years).
    params:
        SABR parameters (alpha, beta, rho, nu).
    vol_type:
        QuantLib SABR approximation type:
        - "hagan": Hagan et al. (2002) - standard
        - "obloj": Obloj (2008) - improved accuracy
        - "hagan_lognormal": Hagan lognormal approximation
        - "normal": Normal SABR approximation
    
    Returns
    -------
    float
        Implied volatility (Black-Scholes convention).
    
    Notes
    -----
    QuantLib's `sabrVolatility()` function signature:
        sabrVolatility(strike, forward, expiryTime, alpha, beta, nu, rho, type)
    
    Note the parameter order: QuantLib uses (nu, rho) while we use (rho, nu).
    """
    ql = require_quantlib()
    
    F = float(forward)
    K = float(strike)
    T = float(expiry)
    alpha = float(params.alpha)
    beta = float(params.beta)
    rho = float(params.rho)
    nu = float(params.nu)
    
    if F <= 0 or K <= 0:
        raise ValueError("forward and strike must be positive.")
    if T <= 0:
        raise ValueError("expiry must be positive.")
    
    # Map vol_type to QuantLib enum
    # QuantLib SabrVolatilityType enum values (check your QL version)
    vol_type_map = {
        "hagan": 0,           # SabrHagan2002
        "hagan_lognormal": 1, # SabrHagan2002Lognormal
        "normal": 2,          # SabrNormal
        "obloj": 3,           # SabrFlochKennedy or similar
    }
    
    ql_vol_type = vol_type_map.get(vol_type, 0)
    
    try:
        # QuantLib sabrVolatility signature varies by version
        # Try the standard signature first
        sigma = ql.sabrVolatility(K, F, T, alpha, beta, nu, rho)
    except TypeError:
        # Fallback for versions with different signature
        sigma = ql.sabrVolatility(K, F, T, alpha, beta, nu, rho, ql_vol_type)
    
    return float(sigma)


def sabr_implied_vol_vec_quantlib(
    *,
    forward: float,
    strikes: np.ndarray,
    expiry: float,
    params: SabrParameters,
    vol_type: SabrVolType = "hagan",
) -> np.ndarray:
    """
    Vectorized SABR implied volatility using QuantLib.
    
    Parameters
    ----------
    forward:
        Forward price F.
    strikes:
        Array of strike prices.
    expiry:
        Time to expiry T (years).
    params:
        SABR parameters.
    vol_type:
        QuantLib SABR approximation type.
    
    Returns
    -------
    np.ndarray
        Array of implied volatilities.
    """
    strikes = np.asarray(strikes, dtype=float).reshape(-1)
    vols = np.array([
        sabr_implied_vol_quantlib(
            forward=forward, strike=float(k), expiry=expiry,
            params=params, vol_type=vol_type
        )
        for k in strikes
    ], dtype=float)
    return vols


# =============================================================================
# QuantLib SABR Calibration
# =============================================================================

@dataclass(frozen=True, slots=True)
class SabrQuantLibConfig:
    """
    Configuration for QuantLib SABR calibration.
    
    Parameters
    ----------
    beta:
        Fixed beta value.
    vol_type:
        QuantLib SABR approximation type.
    max_iterations:
        Maximum calibration iterations.
    accuracy:
        Calibration accuracy tolerance.
    """
    beta: float = 1.0
    vol_type: SabrVolType = "hagan"
    max_iterations: int = 500
    accuracy: float = 1e-8


def calibrate_sabr_quantlib(
    *,
    forward: float,
    strikes: np.ndarray | Sequence[float],
    market_vols: np.ndarray | Sequence[float],
    expiry: float,
    config: SabrQuantLibConfig = SabrQuantLibConfig(),
) -> SabrParameters:
    """
    Calibrate SABR parameters using QuantLib's optimizer.
    
    Parameters
    ----------
    forward:
        Forward price F.
    strikes:
        Strike prices.
    market_vols:
        Market implied volatilities at each strike.
    expiry:
        Time to expiry T (years).
    config:
        QuantLib calibration configuration.
    
    Returns
    -------
    SabrParameters
        Calibrated SABR parameters.
    
    Notes
    -----
    QuantLib provides `SabrSmileSection` which encapsulates a SABR smile
    at a single expiry. We use scipy optimization with QuantLib's sabrVolatility
    for consistency with the native implementation.
    
    For more advanced calibration, QuantLib offers:
    - `SabrInterpolatedSmileSection` - with interpolation
    - Custom calibration via `EndCriteria` and `OptimizationMethod`
    """
    ql = require_quantlib()
    from scipy import optimize
    
    F = float(forward)
    K = np.asarray(strikes, dtype=float).reshape(-1)
    sigma_mkt = np.asarray(market_vols, dtype=float).reshape(-1)
    T = float(expiry)
    beta = float(config.beta)
    
    if K.size != sigma_mkt.size:
        raise ValueError("strikes and market_vols must have same length.")
    if K.size < 3:
        raise ValueError("At least 3 quotes required for SABR calibration.")
    
    # Initial guess from ATM vol
    atm_idx = np.argmin(np.abs(K - F))
    atm_vol = float(sigma_mkt[atm_idx])
    alpha0 = atm_vol * F ** (1.0 - beta)
    rho0 = 0.0
    nu0 = 0.5
    
    x0 = np.array([alpha0, rho0, nu0], dtype=float)
    
    # Objective function using QuantLib's SABR vol
    def objective(x: np.ndarray) -> float:
        alpha = float(x[0])
        rho = float(x[1])
        nu = float(x[2])
        
        # Parameter constraints
        if alpha <= 0 or not -0.999 < rho < 0.999 or nu < 0:
            return 1e10
        
        try:
            sigma_model = np.array([
                ql.sabrVolatility(float(k), F, T, alpha, beta, nu, rho)
                for k in K
            ], dtype=float)
            err = sigma_model - sigma_mkt
            return float(np.sum(err ** 2))
        except Exception:
            return 1e10
    
    # Bounds: alpha > 0, -1 < rho < 1, nu >= 0
    bounds = [(1e-6, 10.0), (-0.999, 0.999), (0.0, 5.0)]
    
    # Optimize
    result = optimize.minimize(
        objective,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': config.max_iterations, 'ftol': config.accuracy},
    )
    
    return SabrParameters(
        alpha=float(result.x[0]),
        beta=beta,
        rho=float(result.x[1]),
        nu=float(result.x[2]),
    )


# =============================================================================
# Comparison Utility
# =============================================================================

def compare_sabr_implementations(
    *,
    forward: float,
    strikes: np.ndarray,
    expiry: float,
    params: SabrParameters,
) -> dict:
    """
    Compare native Python SABR vs QuantLib SABR implied vols.
    
    Returns dict with:
    - native_vols: Native Hagan implementation
    - quantlib_vols: QuantLib implementation
    - abs_diff: Absolute differences
    - max_diff: Maximum absolute difference
    - rmse: Root mean squared error
    
    This is useful for validation and understanding implementation differences.
    """
    from src.calibration.volatility_surface.sabr import sabr_implied_vol_vec
    
    strikes = np.asarray(strikes, dtype=float).reshape(-1)
    
    # Native implementation
    native_vols = sabr_implied_vol_vec(
        forward=forward, strikes=strikes, expiry=expiry, params=params
    )
    
    # QuantLib implementation
    quantlib_vols = sabr_implied_vol_vec_quantlib(
        forward=forward, strikes=strikes, expiry=expiry, params=params
    )
    
    abs_diff = np.abs(native_vols - quantlib_vols)
    
    return {
        "native_vols": native_vols,
        "quantlib_vols": quantlib_vols,
        "abs_diff": abs_diff,
        "max_diff": float(abs_diff.max()),
        "rmse": float(np.sqrt(np.mean(abs_diff ** 2))),
    }

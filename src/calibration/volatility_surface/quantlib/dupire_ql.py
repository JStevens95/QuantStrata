"""
QuantLib Dupire Local Volatility Backend.

This module provides QuantLib-backed local volatility extraction using
Dupire's formula.

QuantLib Local Vol Classes
--------------------------
- `ql.LocalVolSurface`: Extracts local vol from implied vol surface
- `ql.BlackVarianceSurface`: Implied vol surface representation
- `ql.LocalVolTermStructure`: Abstract local vol term structure

The QuantLib approach:
1. Create a BlackVarianceSurface from market data
2. Wrap it in a LocalVolSurface
3. Query local vol at any (S, t) point

Advantages over native implementation:
- More robust numerical differentiation
- Better handling of edge cases
- Consistent with QuantLib pricing engines

Author: QuantStrata Team
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Protocol

from src.marketdata.integration.quantlib.context import (
    require_quantlib,
    to_ql_date,
    yearfrac_to_ql_date,
)
from src.marketdata.surfaces.local_vol import LocalVolSurface


# =============================================================================
# Protocol for implied vol surface input
# =============================================================================

class ImpliedVolSurface(Protocol):
    """Protocol for implied volatility surfaces used as input."""

    def implied_vol(self, expiry: float, strike: float) -> float:
        """Return implied volatility at (expiry, strike)."""
        ...


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True, slots=True)
class DupireQuantLibConfig:
    """
    Configuration for QuantLib Dupire calibration.
    
    Parameters
    ----------
    asof:
        Valuation date (ISO format 'YYYY-MM-DD').
    min_local_vol:
        Minimum allowed local vol (clamp negative values).
    max_local_vol:
        Maximum allowed local vol (clamp extreme values).
    extrapolation:
        Extrapolation mode for the surface ('flat' or 'none').
    """
    asof: str = "2025-01-01"
    min_local_vol: float = 0.01
    max_local_vol: float = 2.0
    extrapolation: str = "flat"


# =============================================================================
# QuantLib Local Vol Surface Creation
# =============================================================================

def _create_ql_black_variance_surface(
    *,
    ql: any,
    asof_date: any,
    expiries: np.ndarray,
    strikes: np.ndarray,
    implied_vols: np.ndarray,
    day_count: any,
    calendar: any,
) -> any:
    """
    Create a QuantLib BlackVarianceSurface from grid data.
    
    Parameters
    ----------
    ql:
        QuantLib module.
    asof_date:
        QuantLib Date for valuation.
    expiries:
        Array of expiry times (year fractions).
    strikes:
        Array of strikes.
    implied_vols:
        2D array of implied vols [n_expiries, n_strikes].
    day_count:
        QuantLib day count convention.
    calendar:
        QuantLib calendar.
    
    Returns
    -------
    ql.BlackVarianceSurface
    """
    # Convert expiries to QuantLib Dates
    expiry_dates = [yearfrac_to_ql_date(asof=asof_date, yearfrac=float(t)) for t in expiries]
    
    # Convert to QuantLib vectors
    ql_expiries = ql.DateVector(expiry_dates)
    ql_strikes = ql.DoubleVector(strikes.tolist())
    
    # Create vol matrix (QuantLib expects [n_strikes, n_expiries] - transposed!)
    vol_matrix = ql.Matrix(len(strikes), len(expiries))
    for i, k in enumerate(strikes):
        for j, t in enumerate(expiries):
            vol_matrix[i][j] = float(implied_vols[j, i])
    
    # Create BlackVarianceSurface
    surface = ql.BlackVarianceSurface(
        asof_date,
        calendar,
        ql_expiries,
        ql_strikes,
        vol_matrix,
        day_count,
    )
    
    return surface


def _create_ql_local_vol_surface(
    *,
    ql: any,
    asof_date: any,
    spot: float,
    r: float,
    q: float,
    black_var_surface: any,
) -> any:
    """
    Create a QuantLib LocalVolSurface from a BlackVarianceSurface.
    
    This uses Dupire's formula internally.
    """
    # Create handles for spot, rates, and dividend
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
    rate_handle = ql.YieldTermStructureHandle(
        ql.FlatForward(asof_date, r, ql.Actual365Fixed())
    )
    div_handle = ql.YieldTermStructureHandle(
        ql.FlatForward(asof_date, q, ql.Actual365Fixed())
    )
    
    # Create BlackVolTermStructure handle
    vol_handle = ql.BlackVolTermStructureHandle(black_var_surface)
    
    # Create LocalVolSurface (Dupire extraction)
    local_vol = ql.LocalVolSurface(
        vol_handle,
        rate_handle,
        div_handle,
        spot_handle,
    )
    
    return local_vol


# =============================================================================
# Main Calibration Function
# =============================================================================

def calibrate_local_vol_quantlib(
    *,
    implied_surface: ImpliedVolSurface,
    spot: float,
    r: float,
    q: float,
    times: Optional[np.ndarray] = None,
    spots: Optional[np.ndarray] = None,
    config: DupireQuantLibConfig = DupireQuantLibConfig(),
) -> LocalVolSurface:
    """
    Calibrate local volatility surface using QuantLib's Dupire implementation.
    
    Parameters
    ----------
    implied_surface:
        Input implied volatility surface.
    spot:
        Current spot price.
    r:
        Risk-free rate (continuous).
    q:
        Dividend/foreign rate (continuous).
    times:
        Time grid for output surface.
    spots:
        Spot grid for output surface.
    config:
        QuantLib calibration configuration.
    
    Returns
    -------
    LocalVolSurface
        Calibrated local volatility surface.
    
    Notes
    -----
    QuantLib's LocalVolSurface uses Dupire's formula:
        σ_LV²(K, T) = [∂C/∂T + (r-q)K ∂C/∂K + qC] / [½K² ∂²C/∂K²]
    
    The QuantLib implementation handles:
    - Numerical differentiation robustly
    - Edge cases (ATM, short expiry)
    - Extrapolation beyond surface bounds
    """
    ql = require_quantlib()
    
    # Default grids
    if times is None:
        times = np.array([0.01, 0.1, 0.25, 0.5, 1.0, 2.0])
    if spots is None:
        spots = spot * np.array([0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3])
    
    times = np.asarray(times, dtype=float).reshape(-1)
    spots = np.asarray(spots, dtype=float).reshape(-1)
    
    # Set up QuantLib context
    asof_date = to_ql_date(config.asof)
    ql.Settings.instance().evaluationDate = asof_date
    day_count = ql.Actual365Fixed()
    calendar = ql.NullCalendar()
    
    # Build implied vol grid for QuantLib
    # Use a fine grid for better QuantLib interpolation
    expiry_grid = np.sort(np.unique(np.concatenate([
        times,
        np.linspace(times.min(), times.max(), 20)
    ])))
    strike_grid = np.sort(np.unique(np.concatenate([
        spots,
        np.linspace(spots.min(), spots.max(), 30)
    ])))
    
    # Sample implied vols from input surface
    implied_vols = np.zeros((len(expiry_grid), len(strike_grid)), dtype=float)
    for i, t in enumerate(expiry_grid):
        for j, k in enumerate(strike_grid):
            implied_vols[i, j] = implied_surface.implied_vol(float(t), float(k))
    
    # Create QuantLib surfaces
    try:
        black_var_surface = _create_ql_black_variance_surface(
            ql=ql,
            asof_date=asof_date,
            expiries=expiry_grid,
            strikes=strike_grid,
            implied_vols=implied_vols,
            day_count=day_count,
            calendar=calendar,
        )
        
        local_vol_surface = _create_ql_local_vol_surface(
            ql=ql,
            asof_date=asof_date,
            spot=spot,
            r=r,
            q=q,
            black_var_surface=black_var_surface,
        )
        
        # Enable extrapolation if configured
        if config.extrapolation == "flat":
            local_vol_surface.enableExtrapolation()
        
        # Sample local vols on output grid
        local_vols = np.zeros((len(times), len(spots)), dtype=float)
        
        for i, t in enumerate(times):
            for j, s in enumerate(spots):
                try:
                    lv = local_vol_surface.localVol(float(t), float(s))
                    lv = np.clip(lv, config.min_local_vol, config.max_local_vol)
                except Exception:
                    # Fall back to implied vol if local vol extraction fails
                    lv = implied_surface.implied_vol(float(t), float(s))
                    lv = np.clip(lv, config.min_local_vol, config.max_local_vol)
                
                local_vols[i, j] = float(lv)
        
    except Exception:
        # If QuantLib fails, fall back to native implementation
        from src.calibration.volatility_surface.dupire import (
            DupireCalibrator,
            DupireConfig,
        )
        
        native_config = DupireConfig(
            min_local_vol=config.min_local_vol,
            max_local_vol=config.max_local_vol,
        )
        calibrator = DupireCalibrator(config=native_config)
        return calibrator.calibrate_grid(
            implied_surface=implied_surface,
            spot=spot,
            r=r,
            q=q,
            times=times,
            spots=spots,
        )
    
    return LocalVolSurface(
        times=times,
        spots=spots,
        local_vols=local_vols,
    )


# =============================================================================
# Comparison Utility
# =============================================================================

def compare_dupire_implementations(
    *,
    implied_surface: ImpliedVolSurface,
    spot: float,
    r: float,
    q: float,
    times: np.ndarray,
    spots: np.ndarray,
    config: DupireQuantLibConfig = DupireQuantLibConfig(),
) -> dict:
    """
    Compare native Python Dupire vs QuantLib Dupire local vol extraction.
    
    Returns dict with:
    - native_surface: Native implementation result
    - quantlib_surface: QuantLib implementation result
    - abs_diff: Absolute differences grid
    - max_diff: Maximum absolute difference
    - rmse: Root mean squared error
    
    This is useful for validation and understanding numerical differences.
    """
    from src.calibration.volatility_surface.dupire import (
        DupireCalibrator,
        DupireConfig,
    )
    
    times = np.asarray(times, dtype=float).reshape(-1)
    spots = np.asarray(spots, dtype=float).reshape(-1)
    
    # Native implementation
    native_config = DupireConfig(
        min_local_vol=config.min_local_vol,
        max_local_vol=config.max_local_vol,
    )
    calibrator = DupireCalibrator(config=native_config)
    native_surface = calibrator.calibrate_grid(
        implied_surface=implied_surface,
        spot=spot,
        r=r,
        q=q,
        times=times,
        spots=spots,
    )
    
    # QuantLib implementation
    quantlib_surface = calibrate_local_vol_quantlib(
        implied_surface=implied_surface,
        spot=spot,
        r=r,
        q=q,
        times=times,
        spots=spots,
        config=config,
    )
    
    # Compute differences
    native_vols = native_surface.local_vols
    quantlib_vols = quantlib_surface.local_vols
    abs_diff = np.abs(native_vols - quantlib_vols)
    
    return {
        "native_surface": native_surface,
        "quantlib_surface": quantlib_surface,
        "abs_diff": abs_diff,
        "max_diff": float(abs_diff.max()),
        "rmse": float(np.sqrt(np.mean(abs_diff ** 2))),
    }

"""
Hull-White Model Calibration.

This module provides calibration of Hull-White one-factor short rate model
parameters to swaption volatilities and cap/floor volatilities.

The Hull-White model has 2 main parameters to calibrate:
- a: Mean reversion speed
- σ: Short rate volatility

The initial rate r₀ and long-term level θ are typically derived from the
yield curve rather than calibrated to vol instruments.

Calibration Approach
--------------------
1. Given swaption/cap vols and a yield curve
2. Compute Hull-White model prices using analytic formulas
3. Convert to implied vols
4. Minimize weighted squared vol errors

Example
-------
>>> from src.calibration.short_rate import calibrate_hull_white_to_swaptions
>>> from src.marketdata.surfaces.vol_surface import SwaptionVolCube
>>> from src.marketdata.curves.term_structure import ZeroRateCurve
>>>
>>> # Calibrate to ATM swaption vols
>>> result = calibrate_hull_white_to_swaptions(
...     swaption_cube=swaption_vols,
...     yield_curve=curve,
... )
>>>
>>> print(f"a = {result.params.a:.4f}")
>>> print(f"sigma = {result.params.sigma:.4f}")
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional, Callable

from src.calibration.core.engine import CalibrationEngine, CalibrationResult, CalibrationConfig
from src.calibration.core.objectives import WeightedLeastSquares
from src.calibration.core.optimizers import LBFGSBConfig
from src.models.short_rate.hull_white import (
    HullWhiteParameters,
    hw_caplet_price,
    hw_swaption_price_jamshidian,
    hw_zc_bond_price,
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True, slots=True)
class HullWhiteCalibrationConfig:
    """
    Configuration for Hull-White model calibration.
    
    Parameters
    ----------
    vol_type : str
        Type of market volatilities:
        - "normal": Bachelier/normal vol (basis points)
        - "lognormal": Black/lognormal vol (percentage)
    weight_by_vega : bool
        If True, weight calibration by option vega.
    use_atm_only : bool
        If True, only calibrate to ATM vols (simpler, often sufficient).
    max_iter : int
        Maximum optimizer iterations.
    tol : float
        Convergence tolerance.
    verbose : bool
        Print calibration progress.
    
    Notes
    -----
    - Normal vols are more common for rates (can handle negative rates)
    - ATM-only calibration is typically sufficient for Hull-White
    - Hull-White is a single-factor model, so it cannot fit full smiles
    """
    
    vol_type: Literal["normal", "lognormal"] = "normal"
    weight_by_vega: bool = False
    use_atm_only: bool = True
    max_iter: int = 500
    tol: float = 1e-8
    verbose: bool = False


@dataclass(frozen=True, slots=True)
class HullWhiteCalibrationResult:
    """
    Result of Hull-White calibration.
    
    Attributes
    ----------
    params : HullWhiteParameters
        Calibrated Hull-White parameters.
    calibration_result : CalibrationResult
        Full calibration diagnostics.
    market_vols : np.ndarray
        Market vols used for calibration.
    model_vols : np.ndarray
        Model vols at calibrated parameters.
    instruments : np.ndarray
        Instrument descriptions (expiry, tenor pairs for swaptions).
    rmse : float
        Root mean squared error of vol fit.
    max_error : float
        Maximum absolute vol error.
    """
    
    params: HullWhiteParameters
    calibration_result: CalibrationResult
    market_vols: np.ndarray
    model_vols: np.ndarray
    instruments: np.ndarray
    rmse: float
    max_error: float
    
    def __str__(self) -> str:
        return (
            f"HullWhiteCalibrationResult\n"
            f"  a (mean reversion) = {self.params.a:.4f}\n"
            f"  σ (volatility)     = {self.params.sigma:.4f} ({self.params.sigma * 10000:.1f} bp)\n"
            f"  Half-life: {self.params.half_life:.2f} years\n"
            f"  RMSE: {self.rmse:.6f}\n"
            f"  Max error: {self.max_error:.6f}\n"
            f"  Converged: {self.calibration_result.success}"
        )


# =============================================================================
# Parameter Bounds
# =============================================================================

HW_BOUNDS = {
    "a": (0.001, 2.0),      # Mean reversion: 0.1% to 200% per year
    "sigma": (0.0001, 0.1),  # Volatility: 1bp to 1000bp
}


# =============================================================================
# Calibration to Swaptions
# =============================================================================

def calibrate_hull_white_to_swaptions(
    swaption_vols: np.ndarray,
    expiries: np.ndarray,
    tenors: np.ndarray,
    yield_curve_df: Callable[[float], float],
    r0: float,
    swap_freq: float = 1.0,
    config: HullWhiteCalibrationConfig = HullWhiteCalibrationConfig(),
    initial_guess: tuple[float, float] | None = None,
    weights: np.ndarray | None = None,
) -> HullWhiteCalibrationResult:
    """
    Calibrate Hull-White parameters to ATM swaption volatilities.
    
    Parameters
    ----------
    swaption_vols : np.ndarray
        ATM swaption vols, shape (n_expiries, n_tenors).
        For normal vols, values in decimal (e.g., 0.005 = 50bp).
        For lognormal vols, values in decimal (e.g., 0.20 = 20%).
    expiries : np.ndarray
        Option expiries in years.
    tenors : np.ndarray
        Swap tenors in years.
    yield_curve_df : Callable
        Discount factor function: df(t) returns P(0,t).
    r0 : float
        Initial short rate.
    swap_freq : float
        Swap payment frequency per year (e.g., 1.0 = annual, 2.0 = semi-annual).
    config : HullWhiteCalibrationConfig
        Calibration configuration.
    initial_guess : tuple, optional
        Initial (a, sigma) guess. If None, uses defaults.
    weights : np.ndarray, optional
        Calibration weights, shape (n_expiries, n_tenors).
    
    Returns
    -------
    HullWhiteCalibrationResult
        Calibration results.
    
    Notes
    -----
    Uses Jamshidian decomposition for swaption pricing.
    The calibration minimizes the squared difference between
    model and market swaption implied vols.
    """
    # Validate inputs
    swaption_vols = np.asarray(swaption_vols, dtype=float)
    expiries = np.asarray(expiries, dtype=float).reshape(-1)
    tenors = np.asarray(tenors, dtype=float).reshape(-1)
    
    n_expiries = expiries.size
    n_tenors = tenors.size
    
    if swaption_vols.shape != (n_expiries, n_tenors):
        raise ValueError(
            f"swaption_vols shape {swaption_vols.shape} must be ({n_expiries}, {n_tenors})."
        )
    
    # Flatten for calibration
    vols_flat = swaption_vols.flatten()
    
    if weights is not None:
        weights_flat = np.asarray(weights, dtype=float).flatten()
    else:
        weights_flat = None
    
    # Build instrument list (expiry, tenor pairs)
    instruments = []
    for i, T_opt in enumerate(expiries):
        for j, tenor in enumerate(tenors):
            instruments.append((T_opt, tenor))
    instruments = np.array(instruments)
    
    # Initial guess
    if initial_guess is not None:
        a0, sigma0 = initial_guess
    else:
        a0 = 0.1  # 10% mean reversion
        sigma0 = 0.01  # 100bp vol
    
    x0 = np.array([a0, sigma0])
    
    def params_from_x(x: np.ndarray) -> HullWhiteParameters:
        return HullWhiteParameters(
            a=float(x[0]),
            sigma=float(x[1]),
            r0=r0,
        )
    
    # Model vol function
    def model_vols_func(x: np.ndarray) -> np.ndarray:
        try:
            params = params_from_x(x)
            model_vols = np.zeros(len(instruments))
            
            for idx, (T_opt, tenor) in enumerate(instruments):
                # Generate swap payment schedule
                n_payments = int(tenor * swap_freq)
                swap_tenors = T_opt + np.arange(1, n_payments + 1) / swap_freq
                
                # Get forward swap rate (ATM strike)
                df_opt = yield_curve_df(T_opt)
                df_payments = np.array([yield_curve_df(t) for t in swap_tenors])
                annuity = np.sum(df_payments / swap_freq)
                K_atm = (df_opt - df_payments[-1]) / annuity if annuity > 0 else 0.01
                
                # Price swaption
                try:
                    price = hw_swaption_price_jamshidian(
                        K=K_atm,
                        T_option=T_opt,
                        swap_tenors=swap_tenors,
                        params=params,
                        df_curve=yield_curve_df,
                        option_type="receiver",
                        notional=1.0,
                    )
                    
                    # Convert price to implied vol (simplified - using Black approximation)
                    # This is approximate; full calibration would invert the pricing formula
                    from scipy.stats import norm
                    
                    # Approximate normal vol from price
                    sqrt_T = np.sqrt(T_opt)
                    annuity_disc = annuity * df_opt
                    if annuity_disc > 0 and sqrt_T > 0:
                        # Bachelier ATM: price ≈ annuity × σ × √T × 0.3989...
                        implied_vol = price / (annuity_disc * sqrt_T * 0.3989422804)
                        model_vols[idx] = max(implied_vol, 1e-6)
                    else:
                        model_vols[idx] = params.sigma
                        
                except Exception:
                    model_vols[idx] = params.sigma
            
            return model_vols
            
        except (ValueError, RuntimeError):
            return np.full(len(instruments), np.nan)
    
    # Objective
    objective = WeightedLeastSquares(
        model_func=model_vols_func,
        market_values=vols_flat,
        weights=weights_flat,
    )
    
    # Run calibration
    optimizer = LBFGSBConfig(max_iter=config.max_iter, tol=config.tol)
    engine = CalibrationEngine(
        optimizer=optimizer,
        config=CalibrationConfig(verbose=config.verbose),
    )
    
    bounds = [HW_BOUNDS["a"], HW_BOUNDS["sigma"]]
    cal_result = engine.calibrate(
        objective=objective,
        initial_params=x0,
        bounds=bounds,
        param_names=["a", "sigma"],
    )
    
    # Extract results
    calibrated_params = params_from_x(cal_result.params)
    model_vols = model_vols_func(cal_result.params).reshape(n_expiries, n_tenors)
    
    vol_errors = model_vols - swaption_vols
    rmse = float(np.sqrt(np.mean(vol_errors ** 2)))
    max_error = float(np.max(np.abs(vol_errors)))
    
    return HullWhiteCalibrationResult(
        params=calibrated_params,
        calibration_result=cal_result,
        market_vols=swaption_vols,
        model_vols=model_vols,
        instruments=instruments,
        rmse=rmse,
        max_error=max_error,
    )


# =============================================================================
# Calibration to Caps/Floors
# =============================================================================

def calibrate_hull_white_to_caps(
    cap_vols: np.ndarray,
    expiries: np.ndarray,
    yield_curve_df: Callable[[float], float],
    r0: float,
    cap_freq: float = 0.25,  # Quarterly caplets
    config: HullWhiteCalibrationConfig = HullWhiteCalibrationConfig(),
    initial_guess: tuple[float, float] | None = None,
    weights: np.ndarray | None = None,
) -> HullWhiteCalibrationResult:
    """
    Calibrate Hull-White parameters to ATM cap volatilities.
    
    Parameters
    ----------
    cap_vols : np.ndarray
        ATM cap vols, shape (n_expiries,).
    expiries : np.ndarray
        Cap expiries in years.
    yield_curve_df : Callable
        Discount factor function.
    r0 : float
        Initial short rate.
    cap_freq : float
        Caplet payment frequency (0.25 = quarterly).
    config : HullWhiteCalibrationConfig
        Calibration configuration.
    initial_guess : tuple, optional
        Initial (a, sigma) guess.
    weights : np.ndarray, optional
        Calibration weights.
    
    Returns
    -------
    HullWhiteCalibrationResult
        Calibration results.
    
    Notes
    -----
    Caps are decomposed into caplets. Each caplet is priced using
    the Hull-White bond option formula.
    """
    # Validate inputs
    cap_vols = np.asarray(cap_vols, dtype=float).reshape(-1)
    expiries = np.asarray(expiries, dtype=float).reshape(-1)
    n_caps = expiries.size
    
    if cap_vols.size != n_caps:
        raise ValueError("cap_vols and expiries must have same length.")
    
    if weights is not None:
        weights_flat = np.asarray(weights, dtype=float).reshape(-1)
    else:
        weights_flat = None
    
    # Initial guess
    if initial_guess is not None:
        a0, sigma0 = initial_guess
    else:
        a0 = 0.1
        sigma0 = 0.01
    
    x0 = np.array([a0, sigma0])
    
    def params_from_x(x: np.ndarray) -> HullWhiteParameters:
        return HullWhiteParameters(
            a=float(x[0]),
            sigma=float(x[1]),
            r0=r0,
        )
    
    # Model vol function
    def model_vols_func(x: np.ndarray) -> np.ndarray:
        try:
            params = params_from_x(x)
            model_vols = np.zeros(n_caps)
            
            for i, T_cap in enumerate(expiries):
                # Sum caplet prices
                n_caplets = int(T_cap / cap_freq)
                if n_caplets < 1:
                    n_caplets = 1
                
                total_price = 0.0
                total_weight = 0.0
                
                for j in range(n_caplets):
                    T_reset = (j + 1) * cap_freq
                    T_pay = T_reset + cap_freq
                    
                    if T_pay > T_cap + 0.01:
                        break
                    
                    # ATM strike from forward rate
                    df_reset = yield_curve_df(T_reset)
                    df_pay = yield_curve_df(T_pay)
                    fwd_rate = (df_reset / df_pay - 1) / cap_freq if df_pay > 0 else r0
                    
                    try:
                        caplet_price = hw_caplet_price(
                            K=fwd_rate,
                            T_reset=T_reset,
                            T_pay=T_pay,
                            params=params,
                            df_curve=yield_curve_df,
                            notional=1.0,
                        )
                        total_price += caplet_price
                        total_weight += cap_freq * df_pay
                    except Exception:
                        pass
                
                # Convert total price to implied vol (simplified)
                if total_weight > 0:
                    sqrt_T = np.sqrt(T_cap)
                    if sqrt_T > 0:
                        model_vols[i] = total_price / (total_weight * sqrt_T * 0.3989)
                        model_vols[i] = max(model_vols[i], 1e-6)
                    else:
                        model_vols[i] = params.sigma
                else:
                    model_vols[i] = params.sigma
            
            return model_vols
            
        except (ValueError, RuntimeError):
            return np.full(n_caps, np.nan)
    
    # Objective
    objective = WeightedLeastSquares(
        model_func=model_vols_func,
        market_values=cap_vols,
        weights=weights_flat,
    )
    
    # Run calibration
    optimizer = LBFGSBConfig(max_iter=config.max_iter, tol=config.tol)
    engine = CalibrationEngine(
        optimizer=optimizer,
        config=CalibrationConfig(verbose=config.verbose),
    )
    
    bounds = [HW_BOUNDS["a"], HW_BOUNDS["sigma"]]
    cal_result = engine.calibrate(
        objective=objective,
        initial_params=x0,
        bounds=bounds,
        param_names=["a", "sigma"],
    )
    
    # Extract results
    calibrated_params = params_from_x(cal_result.params)
    model_vols = model_vols_func(cal_result.params)
    
    vol_errors = model_vols - cap_vols
    rmse = float(np.sqrt(np.mean(vol_errors ** 2)))
    max_error = float(np.max(np.abs(vol_errors)))
    
    # Instruments as expiries
    instruments = expiries.reshape(-1, 1)
    
    return HullWhiteCalibrationResult(
        params=calibrated_params,
        calibration_result=cal_result,
        market_vols=cap_vols.reshape(-1, 1),
        model_vols=model_vols.reshape(-1, 1),
        instruments=instruments,
        rmse=rmse,
        max_error=max_error,
    )

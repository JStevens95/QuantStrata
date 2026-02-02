"""
Heston Model Calibration.

This module provides calibration of Heston stochastic volatility model parameters
to implied volatility surfaces using the generic CalibrationEngine.

The Heston model has 5 parameters:
- κ (kappa): Mean reversion speed
- θ (theta): Long-term variance
- ξ (xi): Vol-of-vol
- V₀ (v0): Initial variance
- ρ (rho): Spot-variance correlation

Calibration Approach
--------------------
1. Compute Heston implied vols using characteristic function (fast)
2. Compare to market implied vols
3. Minimize weighted sum of squared vol errors
4. Optionally enforce Feller condition via penalty

Example
-------
>>> from src.calibration.stochastic_volatility import calibrate_heston_to_surface
>>> from src.marketdata.surfaces.vol_surface import GridVolSurface
>>>
>>> # Market vol surface
>>> surface = GridVolSurface(expiries, strikes, market_vols)
>>>
>>> # Calibrate
>>> result = calibrate_heston_to_surface(
...     surface=surface,
...     spot=100.0,
...     r=0.05,
...     q=0.02,
... )
>>>
>>> print(f"Calibrated params: {result.params}")
>>> print(f"RMSE: {result.rmse:.4f}")
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Sequence

from src.calibration.core.engine import CalibrationEngine, CalibrationResult, CalibrationConfig
from src.calibration.core.objectives import WeightedLeastSquares, PenalizedObjective
from src.calibration.core.optimizers import (
    LBFGSBConfig,
    DifferentialEvolutionConfig,
    create_global_then_local_optimizer,
)
from src.models.stochastic_volatility.heston import (
    HestonParameters,
    heston_implied_vol,
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True, slots=True)
class HestonCalibrationConfig:
    """
    Configuration for Heston model calibration.
    
    Parameters
    ----------
    fix_v0_to_atm : bool
        If True, set V₀ = σ²_ATM (reduces to 4-parameter calibration).
        This is common practice as V₀ should match current ATM vol.
    enforce_feller : bool
        If True, add penalty for Feller condition violation (2κθ > ξ²).
    feller_penalty_weight : float
        Weight for Feller condition penalty.
    use_global_optimizer : bool
        If True, use Differential Evolution for global search before
        local refinement. Recommended for difficult surfaces.
    max_iter : int
        Maximum optimizer iterations.
    tol : float
        Convergence tolerance.
    verbose : bool
        Print calibration progress.
    
    Notes
    -----
    - For equity vol surfaces, negative ρ is typical (-0.9 to -0.3)
    - Feller condition ensures variance stays positive
    - Global optimizer helps avoid local minima
    """
    
    fix_v0_to_atm: bool = True
    enforce_feller: bool = True
    feller_penalty_weight: float = 1000.0
    use_global_optimizer: bool = True
    max_iter: int = 500
    tol: float = 1e-8
    verbose: bool = False


@dataclass(frozen=True, slots=True)
class HestonCalibrationResult:
    """
    Result of Heston calibration.
    
    Attributes
    ----------
    params : HestonParameters
        Calibrated Heston parameters.
    calibration_result : CalibrationResult
        Full calibration diagnostics.
    market_vols : np.ndarray
        Market implied vols used for calibration.
    model_vols : np.ndarray
        Model implied vols at calibrated parameters.
    strikes : np.ndarray
        Strikes used.
    expiries : np.ndarray
        Expiries used.
    rmse : float
        Root mean squared error of vol fit.
    max_error : float
        Maximum absolute vol error.
    feller_satisfied : bool
        Whether Feller condition is satisfied.
    """
    
    params: HestonParameters
    calibration_result: CalibrationResult
    market_vols: np.ndarray
    model_vols: np.ndarray
    strikes: np.ndarray
    expiries: np.ndarray
    rmse: float
    max_error: float
    feller_satisfied: bool
    
    def __str__(self) -> str:
        return (
            f"HestonCalibrationResult\n"
            f"  κ (kappa) = {self.params.kappa:.4f}\n"
            f"  θ (theta) = {self.params.theta:.6f} (long-term vol = {self.params.long_term_vol:.2%})\n"
            f"  ξ (xi)    = {self.params.xi:.4f}\n"
            f"  V₀ (v0)   = {self.params.v0:.6f} (initial vol = {self.params.initial_vol:.2%})\n"
            f"  ρ (rho)   = {self.params.rho:.4f}\n"
            f"  Feller satisfied: {self.feller_satisfied}\n"
            f"  RMSE: {self.rmse:.4f} ({self.rmse * 100:.2f}%)\n"
            f"  Max error: {self.max_error:.4f} ({self.max_error * 100:.2f}%)\n"
            f"  Converged: {self.calibration_result.success}"
        )


# =============================================================================
# Parameter Bounds
# =============================================================================

# Default parameter bounds for Heston calibration
HESTON_BOUNDS = {
    "kappa": (0.01, 10.0),    # Mean reversion: 1-1000% per year
    "theta": (0.0001, 0.25),  # Long-term variance: 1%-50% vol
    "xi": (0.01, 2.0),        # Vol-of-vol: 1%-200%
    "v0": (0.0001, 0.25),     # Initial variance: 1%-50% vol
    "rho": (-0.99, 0.99),     # Correlation: (-1, 1)
}


def _get_bounds(fix_v0: bool) -> list[tuple[float, float]]:
    """Get parameter bounds, optionally excluding v0."""
    if fix_v0:
        return [
            HESTON_BOUNDS["kappa"],
            HESTON_BOUNDS["theta"],
            HESTON_BOUNDS["xi"],
            HESTON_BOUNDS["rho"],
        ]
    else:
        return [
            HESTON_BOUNDS["kappa"],
            HESTON_BOUNDS["theta"],
            HESTON_BOUNDS["xi"],
            HESTON_BOUNDS["v0"],
            HESTON_BOUNDS["rho"],
        ]


# =============================================================================
# Calibration Functions
# =============================================================================

def calibrate_heston_to_vols(
    market_vols: np.ndarray,
    strikes: np.ndarray,
    expiries: np.ndarray,
    spot: float,
    r: float,
    q: float,
    config: HestonCalibrationConfig = HestonCalibrationConfig(),
    initial_guess: HestonParameters | None = None,
    weights: np.ndarray | None = None,
) -> HestonCalibrationResult:
    """
    Calibrate Heston parameters to market implied volatilities.
    
    Parameters
    ----------
    market_vols : np.ndarray
        Market implied vols, shape (n_expiries, n_strikes).
    strikes : np.ndarray
        Strike prices, shape (n_strikes,).
    expiries : np.ndarray
        Expiries in years, shape (n_expiries,).
    spot : float
        Current spot price.
    r : float
        Risk-free rate.
    q : float
        Dividend yield.
    config : HestonCalibrationConfig
        Calibration configuration.
    initial_guess : HestonParameters, optional
        Initial parameter guess. If None, uses defaults.
    weights : np.ndarray, optional
        Calibration weights, shape (n_expiries, n_strikes).
        Higher weight = more important to fit.
    
    Returns
    -------
    HestonCalibrationResult
        Calibration results including parameters and diagnostics.
    """
    # Validate inputs
    market_vols = np.asarray(market_vols, dtype=float)
    strikes = np.asarray(strikes, dtype=float).reshape(-1)
    expiries = np.asarray(expiries, dtype=float).reshape(-1)
    
    n_expiries = expiries.size
    n_strikes = strikes.size
    
    if market_vols.shape != (n_expiries, n_strikes):
        raise ValueError(
            f"market_vols shape {market_vols.shape} must be ({n_expiries}, {n_strikes})."
        )
    
    if spot <= 0:
        raise ValueError("spot must be positive.")
    
    # Flatten for calibration
    market_vols_flat = market_vols.flatten()
    
    if weights is not None:
        weights = np.asarray(weights, dtype=float).flatten()
    
    # Get ATM vol for initial v0
    atm_idx = np.argmin(np.abs(strikes - spot))
    atm_vol = float(market_vols[0, atm_idx])  # Use first expiry ATM
    atm_var = atm_vol ** 2
    
    # Initial guess
    if initial_guess is not None:
        kappa0 = initial_guess.kappa
        theta0 = initial_guess.theta
        xi0 = initial_guess.xi
        v0_0 = initial_guess.v0
        rho0 = initial_guess.rho
    else:
        # Default initial guess
        kappa0 = 2.0
        theta0 = atm_var  # Long-term = ATM
        xi0 = 0.5
        v0_0 = atm_var
        rho0 = -0.5  # Typical equity correlation
    
    # Set up optimization
    fix_v0 = config.fix_v0_to_atm
    
    if fix_v0:
        v0_fixed = atm_var
        x0 = np.array([kappa0, theta0, xi0, rho0])
        param_names = ["kappa", "theta", "xi", "rho"]
        
        def params_from_x(x: np.ndarray) -> HestonParameters:
            return HestonParameters(
                kappa=float(x[0]),
                theta=float(x[1]),
                xi=float(x[2]),
                v0=v0_fixed,
                rho=float(x[3]),
            )
    else:
        x0 = np.array([kappa0, theta0, xi0, v0_0, rho0])
        param_names = ["kappa", "theta", "xi", "v0", "rho"]
        
        def params_from_x(x: np.ndarray) -> HestonParameters:
            return HestonParameters(
                kappa=float(x[0]),
                theta=float(x[1]),
                xi=float(x[2]),
                v0=float(x[3]),
                rho=float(x[4]),
            )
    
    # Model function
    def model_vols_func(x: np.ndarray) -> np.ndarray:
        try:
            params = params_from_x(x)
            vols = np.zeros((n_expiries, n_strikes), dtype=float)
            for i, tau in enumerate(expiries):
                for j, K in enumerate(strikes):
                    vols[i, j] = heston_implied_vol(
                        params, spot, K, r, q, tau, option_type="call"
                    )
            return vols.flatten()
        except (ValueError, RuntimeError):
            return np.full(n_expiries * n_strikes, np.nan)
    
    # Base objective
    base_objective = WeightedLeastSquares(
        model_func=model_vols_func,
        market_values=market_vols_flat,
        weights=weights,
    )
    
    # Add Feller penalty if requested
    if config.enforce_feller:
        def feller_penalty(x: np.ndarray) -> float:
            kappa, theta, xi = x[0], x[1], x[2]
            violation = xi**2 - 2 * kappa * theta
            return max(0, violation) ** 2
        
        objective = PenalizedObjective(
            base_objective=base_objective,
            penalty_func=feller_penalty,
            penalty_weight=config.feller_penalty_weight,
        )
    else:
        objective = base_objective
    
    # Select optimizer
    if config.use_global_optimizer:
        optimizer = create_global_then_local_optimizer(
            global_iters=config.max_iter // 2,
        )
    else:
        optimizer = LBFGSBConfig(max_iter=config.max_iter, tol=config.tol)
    
    # Run calibration
    engine = CalibrationEngine(
        optimizer=optimizer,
        config=CalibrationConfig(verbose=config.verbose),
    )
    
    bounds = _get_bounds(fix_v0)
    cal_result = engine.calibrate(
        objective=objective,
        initial_params=x0,
        bounds=bounds,
        param_names=param_names,
    )
    
    # Extract calibrated parameters
    calibrated_params = params_from_x(cal_result.params)
    
    # Compute model vols at calibrated params
    model_vols = model_vols_func(cal_result.params).reshape(n_expiries, n_strikes)
    
    # Compute error metrics
    vol_errors = model_vols - market_vols
    rmse = float(np.sqrt(np.mean(vol_errors ** 2)))
    max_error = float(np.max(np.abs(vol_errors)))
    
    return HestonCalibrationResult(
        params=calibrated_params,
        calibration_result=cal_result,
        market_vols=market_vols,
        model_vols=model_vols,
        strikes=strikes,
        expiries=expiries,
        rmse=rmse,
        max_error=max_error,
        feller_satisfied=calibrated_params.feller_satisfied,
    )


def calibrate_heston_to_surface(
    surface,  # GridVolSurface
    spot: float,
    r: float,
    q: float,
    config: HestonCalibrationConfig = HestonCalibrationConfig(),
    initial_guess: HestonParameters | None = None,
    use_subset: bool = True,
    max_strikes: int = 15,
    max_expiries: int = 8,
) -> HestonCalibrationResult:
    """
    Calibrate Heston parameters to a GridVolSurface.
    
    Parameters
    ----------
    surface : GridVolSurface
        Market implied vol surface.
    spot : float
        Current spot price.
    r : float
        Risk-free rate.
    q : float
        Dividend yield.
    config : HestonCalibrationConfig
        Calibration configuration.
    initial_guess : HestonParameters, optional
        Initial parameter guess.
    use_subset : bool
        If True, subsample the surface for faster calibration.
    max_strikes : int
        Maximum number of strikes to use (if use_subset=True).
    max_expiries : int
        Maximum number of expiries to use (if use_subset=True).
    
    Returns
    -------
    HestonCalibrationResult
        Calibration results.
    
    Notes
    -----
    For large surfaces, subsampling speeds up calibration significantly
    while maintaining accuracy.
    """
    # Extract surface data
    expiries = np.asarray(surface.expiries, dtype=float)
    strikes = np.asarray(surface.strikes, dtype=float)
    
    # Subsample if requested
    if use_subset:
        # Select expiries evenly spaced
        if expiries.size > max_expiries:
            exp_idx = np.linspace(0, expiries.size - 1, max_expiries, dtype=int)
            expiries = expiries[exp_idx]
        else:
            exp_idx = np.arange(expiries.size)
        
        # Select strikes centered around ATM
        if strikes.size > max_strikes:
            atm_idx = np.argmin(np.abs(strikes - spot))
            half = max_strikes // 2
            start = max(0, atm_idx - half)
            end = min(strikes.size, start + max_strikes)
            start = max(0, end - max_strikes)
            strike_idx = np.arange(start, end)
            strikes = strikes[strike_idx]
        else:
            strike_idx = np.arange(strikes.size)
        
        # Extract subset vols
        market_vols = surface.implied_vols[np.ix_(exp_idx, strike_idx)]
    else:
        market_vols = np.asarray(surface.implied_vols, dtype=float)
    
    return calibrate_heston_to_vols(
        market_vols=market_vols,
        strikes=strikes,
        expiries=expiries,
        spot=spot,
        r=r,
        q=q,
        config=config,
        initial_guess=initial_guess,
    )

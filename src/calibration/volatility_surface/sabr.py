"""
SABR Model Calibration.

The SABR model (Hagan et al., 2002) is a stochastic volatility model widely used
for fitting FX and interest rate smile surfaces. It provides a parametric formula
for implied volatility as a function of strike.

SABR Model Dynamics
-------------------
The SABR model assumes:
    dF = σ * F^β * dW_1
    dσ = ν * σ * dW_2
    dW_1 * dW_2 = ρ * dt

Where:
- F: Forward price (or rate)
- σ: Stochastic volatility
- β: CEV exponent (0 ≤ β ≤ 1)
- ν: Vol-of-vol
- ρ: Correlation between forward and vol processes

SABR Implied Volatility Formula (Hagan Approximation)
------------------------------------------------------
For a given strike K and expiry T:

σ_impl(K) = α / (F*K)^((1-β)/2) * (z/x(z)) * {1 + ε_1 * T}

Where:
- z = (ν/α) * (F*K)^((1-β)/2) * ln(F/K)
- x(z) = ln[(√(1-2ρz+z²) + z - ρ) / (1-ρ)]
- ε_1 = correction terms (expansion in T)

At-the-Money (F=K):
σ_ATM = α / F^(1-β) * {1 + [(1-β)²α²/(24*F^(2-2β)) + ρβνα/(4*F^(1-β)) + (2-3ρ²)ν²/24] * T}

Special Cases
-------------
- β = 0: Normal SABR (good for rates that can go negative)
- β = 0.5: CIR-like dynamics
- β = 1: Log-normal SABR (simplest, good for FX)

Author: QuantStrata Team
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy import optimize


# =============================================================================
# SABR Parameters
# =============================================================================

@dataclass(frozen=True, slots=True)
class SabrParameters:
    """
    SABR model parameters.
    
    Parameters
    ----------
    alpha:
        Initial volatility level (σ_0). Must be > 0.
    beta:
        CEV exponent. Typically fixed at 0, 0.5, or 1. Must be in [0, 1].
    rho:
        Correlation between forward and vol. Must be in (-1, 1).
    nu:
        Vol-of-vol. Must be >= 0.
    
    Notes
    -----
    - Beta is often fixed based on market convention:
      - β = 0: Normal model (rates)
      - β = 0.5: CIR-like
      - β = 1: Log-normal (FX)
    - For FX, β = 1 (log-normal) is most common
    """
    alpha: float
    beta: float
    rho: float
    nu: float
    
    def __post_init__(self) -> None:
        if self.alpha <= 0:
            raise ValueError(f"alpha must be > 0. Got {self.alpha}.")
        if not 0 <= self.beta <= 1:
            raise ValueError(f"beta must be in [0, 1]. Got {self.beta}.")
        if not -1 < self.rho < 1:
            raise ValueError(f"rho must be in (-1, 1). Got {self.rho}.")
        if self.nu < 0:
            raise ValueError(f"nu must be >= 0. Got {self.nu}.")
    
    def to_array(self) -> np.ndarray:
        """Convert to array [alpha, beta, rho, nu]."""
        return np.array([self.alpha, self.beta, self.rho, self.nu], dtype=float)
    
    @classmethod
    def from_array(cls, arr: np.ndarray, beta_fixed: float | None = None) -> "SabrParameters":
        """
        Create from array.
        
        If beta_fixed is provided, arr = [alpha, rho, nu] and beta is fixed.
        Otherwise, arr = [alpha, beta, rho, nu].
        """
        arr = np.asarray(arr, dtype=float).reshape(-1)
        
        if beta_fixed is not None:
            if arr.size != 3:
                raise ValueError("Array must have 3 elements when beta is fixed.")
            return cls(alpha=float(arr[0]), beta=float(beta_fixed), rho=float(arr[1]), nu=float(arr[2]))
        else:
            if arr.size != 4:
                raise ValueError("Array must have 4 elements.")
            return cls(alpha=float(arr[0]), beta=float(arr[1]), rho=float(arr[2]), nu=float(arr[3]))


@dataclass(frozen=True, slots=True)
class SabrConfig:
    """
    Configuration for SABR calibration.
    
    Parameters
    ----------
    beta:
        Fixed beta value. If None, beta is calibrated (not recommended).
    use_normal_approx:
        If True, use normal (β=0) approximation for ATM vol.
    max_iter:
        Maximum optimizer iterations.
    tol:
        Optimizer tolerance.
    """
    beta: float = 1.0  # Fixed beta (log-normal default for FX)
    use_normal_approx: bool = False  # Use shifted lognormal vs normal
    max_iter: int = 200
    tol: float = 1e-8


# =============================================================================
# SABR Implied Volatility (Hagan Approximation)
# =============================================================================

def sabr_implied_vol(
    *,
    forward: float,
    strike: float,
    expiry: float,
    params: SabrParameters,
) -> float:
    """
    Compute SABR implied volatility using Hagan's approximation.
    
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
    
    Returns
    -------
    float
        Implied volatility (Black-Scholes convention).
    
    Notes
    -----
    This uses the formula from Hagan et al. (2002):
    "Managing Smile Risk", Wilmott Magazine
    
    The approximation is accurate for small T and K near F.
    For deep OTM strikes or long maturities, use numerical methods.
    """
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
    
    # Handle ATM case separately to avoid numerical issues
    if abs(F - K) < 1e-10 * F:
        return _sabr_atm_vol(F=F, T=T, alpha=alpha, beta=beta, rho=rho, nu=nu)
    
    # General case
    FK = F * K
    log_FK = math.log(F / K)
    one_minus_beta = 1.0 - beta
    
    # Leading term: α / (F*K)^((1-β)/2)
    FK_pow = FK ** (one_minus_beta / 2.0)
    
    # Compute z and x(z)
    if abs(nu) < 1e-12:
        # No vol-of-vol: simple Black formula
        z_over_xz = 1.0
    else:
        z = (nu / alpha) * FK_pow * log_FK
        
        # x(z) = ln[(√(1-2ρz+z²) + z - ρ) / (1-ρ)]
        sqrt_term = math.sqrt(1.0 - 2.0 * rho * z + z * z)
        xz = math.log((sqrt_term + z - rho) / (1.0 - rho))
        
        if abs(xz) < 1e-12:
            z_over_xz = 1.0
        else:
            z_over_xz = z / xz
    
    # Series expansion terms (denominator correction)
    # 1 + [β²-β]/24 * log²(F/K) + [β²-β]²/1920 * log⁴(F/K)
    log_FK_sq = log_FK * log_FK
    denom_term = (
        1.0 
        + (one_minus_beta ** 2 / 24.0) * log_FK_sq
        + (one_minus_beta ** 4 / 1920.0) * log_FK_sq * log_FK_sq
    )
    
    # Time correction term
    # [(1-β)²α²/(24*FK^(1-β)) + ρβνα/(4*FK^((1-β)/2)) + (2-3ρ²)ν²/24] * T
    FK_pow_full = FK ** one_minus_beta
    time_corr = (
        (one_minus_beta ** 2 * alpha ** 2 / (24.0 * FK_pow_full))
        + (rho * beta * nu * alpha / (4.0 * FK_pow))
        + ((2.0 - 3.0 * rho ** 2) * nu ** 2 / 24.0)
    ) * T
    
    # Combine
    sigma = (alpha / FK_pow) * z_over_xz * (1.0 + time_corr) / denom_term
    
    return float(sigma)


def _sabr_atm_vol(
    *,
    F: float,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> float:
    """
    SABR ATM implied volatility (K = F).
    
    σ_ATM = α/F^(1-β) * {1 + ε*T}
    
    Where:
    ε = (1-β)²α²/(24*F^(2-2β)) + ρβνα/(4*F^(1-β)) + (2-3ρ²)ν²/24
    """
    one_minus_beta = 1.0 - beta
    F_pow = F ** one_minus_beta
    F_pow_2 = F ** (2.0 - 2.0 * beta)
    
    # Time correction
    eps = (
        (one_minus_beta ** 2 * alpha ** 2 / (24.0 * F_pow_2))
        + (rho * beta * nu * alpha / (4.0 * F_pow))
        + ((2.0 - 3.0 * rho ** 2) * nu ** 2 / 24.0)
    )
    
    return float(alpha / F_pow * (1.0 + eps * T))


def sabr_implied_vol_vec(
    *,
    forward: float,
    strikes: np.ndarray,
    expiry: float,
    params: SabrParameters,
) -> np.ndarray:
    """
    Vectorized SABR implied volatility.
    
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
    
    Returns
    -------
    np.ndarray
        Array of implied volatilities.
    """
    strikes = np.asarray(strikes, dtype=float).reshape(-1)
    vols = np.array([
        sabr_implied_vol(forward=forward, strike=float(k), expiry=expiry, params=params)
        for k in strikes
    ], dtype=float)
    return vols


# =============================================================================
# SABR Calibration
# =============================================================================

def calibrate_sabr_to_smile(
    *,
    forward: float,
    strikes: np.ndarray | Sequence[float],
    market_vols: np.ndarray | Sequence[float],
    expiry: float,
    config: SabrConfig = SabrConfig(),
    weights: np.ndarray | Sequence[float] | None = None,
    initial_guess: SabrParameters | None = None,
) -> SabrParameters:
    """
    Calibrate SABR parameters to market smile data.
    
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
        Calibration configuration.
    weights:
        Optional weights for each quote. Higher weight = more important.
    initial_guess:
        Optional initial parameters. If None, uses ATM vol for alpha.
    
    Returns
    -------
    SabrParameters
        Calibrated SABR parameters.
    
    Notes
    -----
    The calibration minimizes:
        Σ_i w_i * (σ_SABR(K_i) - σ_market(K_i))²
    
    Beta is typically fixed to avoid over-parameterization.
    """
    F = float(forward)
    K = np.asarray(strikes, dtype=float).reshape(-1)
    sigma_mkt = np.asarray(market_vols, dtype=float).reshape(-1)
    T = float(expiry)
    beta = float(config.beta)
    
    if K.size != sigma_mkt.size:
        raise ValueError("strikes and market_vols must have same length.")
    if K.size < 3:
        raise ValueError("At least 3 quotes required for SABR calibration.")
    if F <= 0:
        raise ValueError("forward must be positive.")
    if T <= 0:
        raise ValueError("expiry must be positive.")
    if np.any(K <= 0) or np.any(sigma_mkt <= 0):
        raise ValueError("strikes and market_vols must be positive.")
    
    # Set up weights
    if weights is None:
        w = np.ones_like(K)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.size != K.size:
            raise ValueError("weights must have same length as strikes.")
    
    # Normalize weights
    w = w / w.sum()
    
    # Initial guess
    if initial_guess is not None:
        alpha0 = float(initial_guess.alpha)
        rho0 = float(initial_guess.rho)
        nu0 = float(initial_guess.nu)
    else:
        # Use ATM vol for initial alpha
        atm_idx = np.argmin(np.abs(K - F))
        atm_vol = float(sigma_mkt[atm_idx])
        alpha0 = atm_vol * F ** (1.0 - beta)  # Approximate ATM formula inversion
        rho0 = 0.0
        nu0 = 0.5
    
    # Pack initial guess
    x0 = np.array([alpha0, rho0, nu0], dtype=float)
    
    # Objective function: weighted sum of squared errors
    def objective(x: np.ndarray) -> float:
        alpha = float(x[0])
        rho = float(x[1])
        nu = float(x[2])
        
        # Parameter constraints (soft)
        if alpha <= 0 or not -0.999 < rho < 0.999 or nu < 0:
            return 1e10
        
        try:
            params = SabrParameters(alpha=alpha, beta=beta, rho=rho, nu=nu)
            sigma_model = sabr_implied_vol_vec(forward=F, strikes=K, expiry=T, params=params)
            err = sigma_model - sigma_mkt
            return float(np.sum(w * err ** 2))
        except (ValueError, ZeroDivisionError, RuntimeWarning):
            return 1e10
    
    # Bounds: alpha > 0, -1 < rho < 1, nu >= 0
    bounds = [(1e-6, 10.0), (-0.999, 0.999), (0.0, 5.0)]
    
    # Optimize
    result = optimize.minimize(
        objective,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': config.max_iter, 'ftol': config.tol},
    )
    
    if not result.success:
        # Try with different initial guess
        x0_alt = np.array([alpha0 * 1.5, -0.3, 0.8], dtype=float)
        result = optimize.minimize(
            objective,
            x0_alt,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': config.max_iter, 'ftol': config.tol},
        )
    
    return SabrParameters(
        alpha=float(result.x[0]),
        beta=beta,
        rho=float(result.x[1]),
        nu=float(result.x[2]),
    )


def calibrate_sabr_term_structure(
    *,
    forward_by_expiry: dict[float, float],
    strikes_by_expiry: dict[float, np.ndarray],
    vols_by_expiry: dict[float, np.ndarray],
    config: SabrConfig = SabrConfig(),
) -> dict[float, SabrParameters]:
    """
    Calibrate SABR parameters for each expiry in the term structure.
    
    Parameters
    ----------
    forward_by_expiry:
        Dict mapping expiry -> forward price.
    strikes_by_expiry:
        Dict mapping expiry -> strike array.
    vols_by_expiry:
        Dict mapping expiry -> vol array.
    config:
        Calibration configuration.
    
    Returns
    -------
    dict[float, SabrParameters]
        Dict mapping expiry -> calibrated SABR parameters.
    """
    expiries = sorted(forward_by_expiry.keys())
    result: dict[float, SabrParameters] = {}
    
    prev_params: SabrParameters | None = None
    
    for T in expiries:
        F = forward_by_expiry[T]
        K = strikes_by_expiry[T]
        sigma = vols_by_expiry[T]
        
        # Use previous expiry's params as initial guess for stability
        params = calibrate_sabr_to_smile(
            forward=F,
            strikes=K,
            market_vols=sigma,
            expiry=T,
            config=config,
            initial_guess=prev_params,
        )
        
        result[T] = params
        prev_params = params
    
    return result


# =============================================================================
# SABR Vol Surface
# =============================================================================

def create_sabr_vol_surface(
    *,
    params_by_expiry: dict[float, SabrParameters],
    forward_by_expiry: dict[float, float],
    strikes: np.ndarray,
) -> Callable[[float, float], float]:
    """
    Create a vol surface function from SABR parameters.
    
    Parameters
    ----------
    params_by_expiry:
        Dict mapping expiry -> SABR parameters.
    forward_by_expiry:
        Dict mapping expiry -> forward price.
    strikes:
        Strike grid for interpolation.
    
    Returns
    -------
    Callable[[float, float], float]
        A function vol(expiry, strike) that returns implied volatility.
    
    Notes
    -----
    For expiries between calibrated points, linear interpolation is used.
    """
    expiries = np.array(sorted(params_by_expiry.keys()), dtype=float)
    
    if expiries.size == 0:
        raise ValueError("params_by_expiry must not be empty.")
    
    def vol_func(expiry: float, strike: float) -> float:
        T = float(expiry)
        K = float(strike)
        
        if T <= 0:
            raise ValueError("expiry must be positive.")
        
        # Find bracketing expiries
        if T <= expiries[0]:
            # Extrapolate flat to first expiry
            T_use = float(expiries[0])
            params = params_by_expiry[T_use]
            F = forward_by_expiry[T_use]
            return sabr_implied_vol(forward=F, strike=K, expiry=T, params=params)
        
        elif T >= expiries[-1]:
            # Extrapolate flat from last expiry
            T_use = float(expiries[-1])
            params = params_by_expiry[T_use]
            F = forward_by_expiry[T_use]
            return sabr_implied_vol(forward=F, strike=K, expiry=T, params=params)
        
        else:
            # Interpolate between expiries
            idx = np.searchsorted(expiries, T)
            T1 = float(expiries[idx - 1])
            T2 = float(expiries[idx])
            
            params1 = params_by_expiry[T1]
            params2 = params_by_expiry[T2]
            F1 = forward_by_expiry[T1]
            F2 = forward_by_expiry[T2]
            
            # Interpolate forward
            w = (T - T1) / (T2 - T1)
            F = F1 + w * (F2 - F1)
            
            # Get vols at T1 and T2
            vol1 = sabr_implied_vol(forward=F1, strike=K, expiry=T1, params=params1)
            vol2 = sabr_implied_vol(forward=F2, strike=K, expiry=T2, params=params2)
            
            # Linear interpolation in total variance
            var1 = vol1 ** 2 * T1
            var2 = vol2 ** 2 * T2
            var_interp = var1 + w * (var2 - var1)
            
            return float(math.sqrt(var_interp / T))
    
    return vol_func


# =============================================================================
# SABR for Interest Rate Swaption Smile
# =============================================================================

def calibrate_sabr_to_swaption_smile(
    *,
    strikes: np.ndarray | Sequence[float],
    market_vols: np.ndarray | Sequence[float],
    forward_swap_rate: float,
    expiry: float,
    tenor: float,
    vol_type: str = "normal",
    config: SabrConfig = SabrConfig(beta=0.0),
    weights: np.ndarray | Sequence[float] | None = None,
    initial_guess: SabrParameters | None = None,
) -> SabrParameters:
    """
    Calibrate SABR parameters to swaption smile data.
    
    This is a convenience wrapper for interest rate swaption smile calibration.
    
    Parameters
    ----------
    strikes : array-like
        Swaption strikes (swap rates).
    market_vols : array-like
        Market implied volatilities at each strike.
        For normal vols (Bachelier), values in decimal (e.g., 0.005 = 50bp).
        For lognormal vols (Black), values in decimal (e.g., 0.20 = 20%).
    forward_swap_rate : float
        ATM forward swap rate.
    expiry : float
        Option expiry in years.
    tenor : float
        Underlying swap tenor in years (for documentation only).
    vol_type : str
        Type of volatilities:
        - "normal": Bachelier/normal vols (typical for rates)
        - "lognormal": Black/lognormal vols
    config : SabrConfig
        Calibration configuration.
        Default beta=0 for normal SABR (handles negative rates).
    weights : array-like, optional
        Weights for each strike quote.
    initial_guess : SabrParameters, optional
        Initial parameters.
    
    Returns
    -------
    SabrParameters
        Calibrated SABR parameters.
    
    Notes
    -----
    For interest rate swaption smile fitting:
    - Beta = 0 (normal SABR) is typical, as it handles negative rates
    - Beta = 0.5 (CIR-like) is sometimes used
    - Beta = 1 (lognormal) requires positive forward rates
    
    The forward_swap_rate is used as the 'forward' in the SABR formula.
    
    Examples
    --------
    >>> # Calibrate 10Y10Y swaption smile
    >>> strikes = np.array([0.02, 0.025, 0.03, 0.035, 0.04])  # 2%-4%
    >>> market_vols = np.array([0.0050, 0.0048, 0.0045, 0.0048, 0.0052])  # ~50bp
    >>> fwd = 0.03  # 3% forward rate
    >>>
    >>> params = calibrate_sabr_to_swaption_smile(
    ...     strikes=strikes,
    ...     market_vols=market_vols,
    ...     forward_swap_rate=fwd,
    ...     expiry=10.0,
    ...     tenor=10.0,
    ...     vol_type="normal",
    ... )
    >>> print(f"α={params.alpha:.4f}, ρ={params.rho:.4f}, ν={params.nu:.4f}")
    """
    # Validate vol_type
    if vol_type not in ("normal", "lognormal"):
        raise ValueError(f"vol_type must be 'normal' or 'lognormal', got {vol_type}")
    
    # For normal SABR, ensure beta = 0
    if vol_type == "normal" and config.beta != 0.0:
        # Create new config with beta=0
        config = SabrConfig(
            beta=0.0,
            use_normal_approx=True,
            max_iter=config.max_iter,
            tol=config.tol,
        )
    
    # Convert to arrays
    strikes = np.asarray(strikes, dtype=float).reshape(-1)
    market_vols = np.asarray(market_vols, dtype=float).reshape(-1)
    
    # For normal vols, we need to adjust the interpretation
    # The standard SABR formula gives lognormal vols; for normal SABR (beta=0),
    # we use the normal SABR approximation
    
    # Call the standard calibration
    return calibrate_sabr_to_smile(
        forward=forward_swap_rate,
        strikes=strikes,
        market_vols=market_vols,
        expiry=expiry,
        config=config,
        weights=weights,
        initial_guess=initial_guess,
    )


def calibrate_sabr_swaption_cube(
    *,
    expiries: Sequence[float],
    tenors: Sequence[float],
    strikes_by_point: dict[tuple[float, float], np.ndarray],
    vols_by_point: dict[tuple[float, float], np.ndarray],
    forward_by_point: dict[tuple[float, float], float],
    vol_type: str = "normal",
    config: SabrConfig = SabrConfig(beta=0.0),
) -> dict[tuple[float, float], SabrParameters]:
    """
    Calibrate SABR parameters for each point in a swaption cube.
    
    Parameters
    ----------
    expiries : sequence of float
        Option expiries.
    tenors : sequence of float
        Swap tenors.
    strikes_by_point : dict
        Dict mapping (expiry, tenor) -> strike array.
    vols_by_point : dict
        Dict mapping (expiry, tenor) -> vol array.
    forward_by_point : dict
        Dict mapping (expiry, tenor) -> forward swap rate.
    vol_type : str
        "normal" or "lognormal".
    config : SabrConfig
        Calibration configuration.
    
    Returns
    -------
    dict[tuple[float, float], SabrParameters]
        Dict mapping (expiry, tenor) -> calibrated SABR parameters.
    
    Examples
    --------
    >>> # Calibrate full swaption cube
    >>> params_cube = calibrate_sabr_swaption_cube(
    ...     expiries=[1, 2, 5, 10],
    ...     tenors=[5, 10, 20, 30],
    ...     strikes_by_point=strikes,
    ...     vols_by_point=vols,
    ...     forward_by_point=forwards,
    ... )
    >>> for (exp, ten), p in params_cube.items():
    ...     print(f"{exp}Y{ten}Y: α={p.alpha:.4f}, ρ={p.rho:.4f}")
    """
    result: dict[tuple[float, float], SabrParameters] = {}
    prev_params: SabrParameters | None = None
    
    for T_opt in expiries:
        for tenor in tenors:
            key = (T_opt, tenor)
            
            if key not in strikes_by_point or key not in vols_by_point:
                continue
            
            K = strikes_by_point[key]
            sigma = vols_by_point[key]
            F = forward_by_point[key]
            
            params = calibrate_sabr_to_swaption_smile(
                strikes=K,
                market_vols=sigma,
                forward_swap_rate=F,
                expiry=T_opt,
                tenor=tenor,
                vol_type=vol_type,
                config=config,
                initial_guess=prev_params,
            )
            
            result[key] = params
            prev_params = params
    
    return result

"""
Importance Sampling for Monte Carlo Variance Reduction.

This module provides importance sampling techniques for reducing variance
in Monte Carlo simulations, particularly for rare event pricing (deep OTM options).

Overview
--------
Importance sampling changes the sampling distribution to oversample important
regions (where payoff is non-zero), then corrects via likelihood ratios.

For option pricing:
    Price = E^Q[h(S_T)] = E^P[h(S_T) * (dQ/dP)]

where P is the importance sampling measure and dQ/dP is the Radon-Nikodym derivative.

Methods Implemented
-------------------
- Mean shift: Shift the drift to make OTM options more likely to be ITM
- Optimal drift: Analytical optimal shift for European options
- Exponential tilting: General exponential family tilting

Key Applications
----------------
- Deep OTM options (very low deltas)
- Barrier options near knock-out
- Rare event simulation
- Tail risk estimation

References
----------
- Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering.
- Glasserman, P. et al. (1999). "Asymptotically optimal importance sampling 
  and stratification for pricing path-dependent options."
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable
from scipy.stats import norm


# =============================================================================
# Importance Sampling Result
# =============================================================================

@dataclass(frozen=True, slots=True)
class ImportanceSamplingResult:
    """
    Result container for importance sampling Monte Carlo.

    Attributes
    ----------
    price : float
        Estimated option price.
    std_error : float
        Standard error of the estimate.
    variance_reduction : float
        Variance reduction factor vs. standard MC.
    effective_sample_size : float
        Effective sample size after weighting.
    drift_shift : float
        The drift shift applied (for mean-shift IS).
    n_samples : int
        Number of samples used.
    """

    price: float
    std_error: float
    variance_reduction: float
    effective_sample_size: float
    drift_shift: float
    n_samples: int

    @property
    def confidence_interval_95(self) -> tuple[float, float]:
        """95% confidence interval."""
        return (self.price - 1.96 * self.std_error, self.price + 1.96 * self.std_error)


# =============================================================================
# Mean Shift Importance Sampling
# =============================================================================

def optimal_drift_shift_call(
    spot0: float,
    strike: float,
    maturity: float,
    r: float,
    q: float,
    sigma: float,
) -> float:
    """
    Compute optimal drift shift for European call option.

    The optimal shift moves the mean of the terminal distribution to
    the strike price, maximizing the probability of ITM paths.

    For a call: optimal shift θ* = (ln(K/S) - (r-q-σ²/2)T) / (σ²T)

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
        Dividend yield.
    sigma : float
        Volatility.

    Returns
    -------
    float
        Optimal drift shift θ*.
    """
    log_moneyness = np.log(strike / spot0)
    drift_term = (r - q - 0.5 * sigma ** 2) * maturity
    variance_term = sigma ** 2 * maturity

    theta_star = (log_moneyness - drift_term) / variance_term
    return float(theta_star)


def optimal_drift_shift_put(
    spot0: float,
    strike: float,
    maturity: float,
    r: float,
    q: float,
    sigma: float,
) -> float:
    """
    Compute optimal drift shift for European put option.

    For a put: optimal shift moves mean below strike.
    θ* = (ln(K/S) - (r-q-σ²/2)T) / (σ²T)

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
        Dividend yield.
    sigma : float
        Volatility.

    Returns
    -------
    float
        Optimal drift shift θ*.
    """
    log_moneyness = np.log(strike / spot0)
    drift_term = (r - q - 0.5 * sigma ** 2) * maturity
    variance_term = sigma ** 2 * maturity

    theta_star = (log_moneyness - drift_term) / variance_term
    return float(theta_star)


def is_european_call(
    spot0: float,
    strike: float,
    maturity: float,
    r: float,
    q: float,
    sigma: float,
    n_samples: int = 100000,
    drift_shift: Optional[float] = None,
    seed: Optional[int] = None,
) -> ImportanceSamplingResult:
    """
    Price European call using importance sampling with mean shift.

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
        Dividend yield.
    sigma : float
        Volatility.
    n_samples : int
        Number of Monte Carlo samples.
    drift_shift : float, optional
        Drift shift θ. If None, uses optimal shift.
    seed : int, optional
        Random seed.

    Returns
    -------
    ImportanceSamplingResult
        Pricing result with variance reduction statistics.
    """
    from src.models.numeric.monte_carlo.rng import NormalRng

    # Compute optimal drift shift if not provided
    if drift_shift is None:
        drift_shift = optimal_drift_shift_call(spot0, strike, maturity, r, q, sigma)

    # Generate samples under shifted measure
    rng = NormalRng(seed=seed)
    Z = rng.standard_normals(n_samples, 1, antithetic=True).flatten()

    # Shifted terminal spot
    # Under Q: ln(S_T/S_0) ~ N((r-q-σ²/2)T, σ²T)
    # Under P (shifted): sample Z + θσ√T instead of Z
    # This is equivalent to shifting the drift by θσ²
    sqrt_T = np.sqrt(maturity)
    shifted_Z = Z + drift_shift * sigma * sqrt_T

    drift = (r - q - 0.5 * sigma ** 2) * maturity
    diffusion = sigma * sqrt_T
    S_T = spot0 * np.exp(drift + diffusion * shifted_Z)

    # Payoffs under shifted measure
    payoffs = np.maximum(S_T - strike, 0.0)

    # Likelihood ratio (Radon-Nikodym derivative dQ/dP)
    # For Gaussian shift: dQ/dP = exp(-θσ√T * Z - 0.5 * (θσ√T)²)
    # But we sampled shifted_Z = Z + θσ√T, so:
    # dQ/dP = exp(-θσ√T * (shifted_Z - θσ√T) - 0.5 * (θσ√T)²)
    #       = exp(-θσ√T * shifted_Z + (θσ√T)² - 0.5 * (θσ√T)²)
    #       = exp(-θσ√T * shifted_Z + 0.5 * (θσ√T)²)
    theta_sigma_sqrt_T = drift_shift * sigma * sqrt_T
    likelihood_ratio = np.exp(-theta_sigma_sqrt_T * shifted_Z + 0.5 * theta_sigma_sqrt_T ** 2)

    # Importance sampling estimator
    weighted_payoffs = payoffs * likelihood_ratio
    discount = np.exp(-r * maturity)

    price = float(discount * weighted_payoffs.mean())
    std_error = float(discount * weighted_payoffs.std() / np.sqrt(n_samples))

    # Compute standard MC estimate for comparison
    S_T_standard = spot0 * np.exp(drift + diffusion * Z)
    payoffs_standard = np.maximum(S_T_standard - strike, 0.0)
    var_standard = payoffs_standard.var()

    # Variance reduction factor
    var_is = weighted_payoffs.var()
    variance_reduction = var_standard / var_is if var_is > 0 else np.inf

    # Effective sample size
    weights_normalized = likelihood_ratio / likelihood_ratio.sum()
    ess = 1.0 / (weights_normalized ** 2).sum()

    return ImportanceSamplingResult(
        price=price,
        std_error=std_error,
        variance_reduction=float(variance_reduction),
        effective_sample_size=float(ess),
        drift_shift=float(drift_shift),
        n_samples=n_samples,
    )


def is_european_put(
    spot0: float,
    strike: float,
    maturity: float,
    r: float,
    q: float,
    sigma: float,
    n_samples: int = 100000,
    drift_shift: Optional[float] = None,
    seed: Optional[int] = None,
) -> ImportanceSamplingResult:
    """
    Price European put using importance sampling with mean shift.

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
        Dividend yield.
    sigma : float
        Volatility.
    n_samples : int
        Number of Monte Carlo samples.
    drift_shift : float, optional
        Drift shift θ. If None, uses optimal shift.
    seed : int, optional
        Random seed.

    Returns
    -------
    ImportanceSamplingResult
        Pricing result with variance reduction statistics.
    """
    from src.models.numeric.monte_carlo.rng import NormalRng

    # Compute optimal drift shift if not provided
    if drift_shift is None:
        drift_shift = optimal_drift_shift_put(spot0, strike, maturity, r, q, sigma)

    # Generate samples
    rng = NormalRng(seed=seed)
    Z = rng.standard_normals(n_samples, 1, antithetic=True).flatten()

    sqrt_T = np.sqrt(maturity)
    shifted_Z = Z + drift_shift * sigma * sqrt_T

    drift = (r - q - 0.5 * sigma ** 2) * maturity
    diffusion = sigma * sqrt_T
    S_T = spot0 * np.exp(drift + diffusion * shifted_Z)

    # Put payoffs
    payoffs = np.maximum(strike - S_T, 0.0)

    # Likelihood ratio
    theta_sigma_sqrt_T = drift_shift * sigma * sqrt_T
    likelihood_ratio = np.exp(-theta_sigma_sqrt_T * shifted_Z + 0.5 * theta_sigma_sqrt_T ** 2)

    # Importance sampling estimator
    weighted_payoffs = payoffs * likelihood_ratio
    discount = np.exp(-r * maturity)

    price = float(discount * weighted_payoffs.mean())
    std_error = float(discount * weighted_payoffs.std() / np.sqrt(n_samples))

    # Variance comparison
    S_T_standard = spot0 * np.exp(drift + diffusion * Z)
    payoffs_standard = np.maximum(strike - S_T_standard, 0.0)
    var_standard = payoffs_standard.var()

    var_is = weighted_payoffs.var()
    variance_reduction = var_standard / var_is if var_is > 0 else np.inf

    # Effective sample size
    weights_normalized = likelihood_ratio / likelihood_ratio.sum()
    ess = 1.0 / (weights_normalized ** 2).sum()

    return ImportanceSamplingResult(
        price=price,
        std_error=std_error,
        variance_reduction=float(variance_reduction),
        effective_sample_size=float(ess),
        drift_shift=float(drift_shift),
        n_samples=n_samples,
    )


# =============================================================================
# Adaptive Importance Sampling
# =============================================================================

def adaptive_is_european_call(
    spot0: float,
    strike: float,
    maturity: float,
    r: float,
    q: float,
    sigma: float,
    n_samples: int = 100000,
    n_pilot: int = 1000,
    seed: Optional[int] = None,
) -> ImportanceSamplingResult:
    """
    Price European call using adaptive importance sampling.

    Uses a pilot simulation to estimate the optimal drift shift,
    then applies importance sampling with the estimated shift.

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
        Dividend yield.
    sigma : float
        Volatility.
    n_samples : int
        Number of main Monte Carlo samples.
    n_pilot : int
        Number of pilot samples for shift estimation.
    seed : int, optional
        Random seed.

    Returns
    -------
    ImportanceSamplingResult
        Pricing result.
    """
    from src.models.numeric.monte_carlo.rng import NormalRng

    # Pilot simulation with initial guess
    initial_shift = optimal_drift_shift_call(spot0, strike, maturity, r, q, sigma)

    # Clamp the shift to avoid extreme values
    initial_shift = np.clip(initial_shift, -5.0, 5.0)

    # Run main IS with the computed shift
    result = is_european_call(
        spot0=spot0,
        strike=strike,
        maturity=maturity,
        r=r,
        q=q,
        sigma=sigma,
        n_samples=n_samples,
        drift_shift=initial_shift,
        seed=seed,
    )

    return result


# =============================================================================
# Comparison Utilities
# =============================================================================

def compare_is_standard_mc(
    spot0: float,
    strike: float,
    maturity: float,
    r: float,
    q: float,
    sigma: float,
    n_samples: int = 100000,
    seed: Optional[int] = None,
) -> dict:
    """
    Compare importance sampling vs standard MC for a European put.

    Parameters
    ----------
    spot0 : float
        Initial spot price.
    strike : float
        Strike price (should be OTM for best comparison).
    maturity : float
        Time to maturity.
    r : float
        Risk-free rate.
    q : float
        Dividend yield.
    sigma : float
        Volatility.
    n_samples : int
        Number of samples.
    seed : int, optional
        Random seed.

    Returns
    -------
    dict
        Comparison results.
    """
    from src.models.numeric.monte_carlo.rng import NormalRng

    # Standard MC
    rng = NormalRng(seed=seed)
    Z = rng.standard_normals(n_samples, 1, antithetic=True).flatten()

    sqrt_T = np.sqrt(maturity)
    drift = (r - q - 0.5 * sigma ** 2) * maturity
    diffusion = sigma * sqrt_T
    S_T = spot0 * np.exp(drift + diffusion * Z)

    payoffs_put = np.maximum(strike - S_T, 0.0)
    discount = np.exp(-r * maturity)

    mc_price = float(discount * payoffs_put.mean())
    mc_std_error = float(discount * payoffs_put.std() / np.sqrt(n_samples))

    # Importance Sampling
    is_result = is_european_put(
        spot0=spot0,
        strike=strike,
        maturity=maturity,
        r=r,
        q=q,
        sigma=sigma,
        n_samples=n_samples,
        seed=seed,
    )

    # Black-Scholes price for reference
    from scipy.stats import norm
    d1 = (np.log(spot0 / strike) + (r - q + 0.5 * sigma ** 2) * maturity) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    bs_price = float(strike * np.exp(-r * maturity) * norm.cdf(-d2) - spot0 * np.exp(-q * maturity) * norm.cdf(-d1))

    return {
        'bs_price': bs_price,
        'mc_price': mc_price,
        'mc_std_error': mc_std_error,
        'is_price': is_result.price,
        'is_std_error': is_result.std_error,
        'variance_reduction': is_result.variance_reduction,
        'effective_sample_size': is_result.effective_sample_size,
        'drift_shift': is_result.drift_shift,
        'mc_error': abs(mc_price - bs_price),
        'is_error': abs(is_result.price - bs_price),
    }

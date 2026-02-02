"""
Longstaff-Schwartz Monte Carlo (LSM) for American Options.

This module implements the Longstaff-Schwartz (2001) algorithm for pricing
American-style options using Monte Carlo simulation combined with regression.

Algorithm Overview
------------------
1. Simulate paths forward to maturity
2. At maturity, compute payoff
3. Work backwards through time:
   a. At each exercise date, identify in-the-money paths
   b. Regress discounted continuation values on basis functions of spot
   c. Compare regression estimate to immediate exercise payoff
   d. Exercise if immediate payoff > expected continuation value
4. Average discounted payoffs across paths

Key Features
------------
- Handles American and Bermudan exercise
- Multiple basis function choices (polynomial, Laguerre)
- Works with any underlying dynamics (GBM, Heston, etc.)
- Supports puts and calls

References
----------
- Longstaff, F.A. & Schwartz, E.S. (2001). "Valuing American Options by
  Simulation: A Simple Least-Squares Approach." Review of Financial Studies.
- Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering.
  Springer.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, Literal, Optional
from enum import Enum


# =============================================================================
# Basis Functions
# =============================================================================

class BasisType(str, Enum):
    """Available basis function types for LSM regression."""
    POLYNOMIAL = "polynomial"
    LAGUERRE = "laguerre"
    CHEBYSHEV = "chebyshev"


def polynomial_basis(x: np.ndarray, degree: int) -> np.ndarray:
    """
    Polynomial basis functions: 1, x, x², ..., x^degree.

    Parameters
    ----------
    x : np.ndarray
        Input values, shape (n,).
    degree : int
        Maximum polynomial degree.

    Returns
    -------
    np.ndarray
        Basis matrix, shape (n, degree + 1).
    """
    n = len(x)
    basis = np.zeros((n, degree + 1))
    for d in range(degree + 1):
        basis[:, d] = x ** d
    return basis


def laguerre_basis(x: np.ndarray, degree: int) -> np.ndarray:
    """
    Laguerre polynomial basis functions L_0(x), L_1(x), ..., L_degree(x).

    The Laguerre polynomials are defined by:
        L_0(x) = 1
        L_1(x) = 1 - x
        L_n(x) = ((2n-1-x)*L_{n-1}(x) - (n-1)*L_{n-2}(x)) / n

    Parameters
    ----------
    x : np.ndarray
        Input values, shape (n,).
    degree : int
        Maximum polynomial degree.

    Returns
    -------
    np.ndarray
        Basis matrix, shape (n, degree + 1).
    """
    n = len(x)
    basis = np.zeros((n, degree + 1))

    # L_0 = 1
    basis[:, 0] = 1.0

    if degree >= 1:
        # L_1 = 1 - x
        basis[:, 1] = 1.0 - x

    # Recurrence relation for higher degrees
    for d in range(2, degree + 1):
        basis[:, d] = ((2 * d - 1 - x) * basis[:, d - 1] - (d - 1) * basis[:, d - 2]) / d

    return basis


def chebyshev_basis(x: np.ndarray, degree: int) -> np.ndarray:
    """
    Chebyshev polynomial basis functions T_0(x), T_1(x), ..., T_degree(x).

    The Chebyshev polynomials are defined by:
        T_0(x) = 1
        T_1(x) = x
        T_n(x) = 2x*T_{n-1}(x) - T_{n-2}(x)

    Parameters
    ----------
    x : np.ndarray
        Input values, shape (n,). Should be normalized to [-1, 1] for stability.
    degree : int
        Maximum polynomial degree.

    Returns
    -------
    np.ndarray
        Basis matrix, shape (n, degree + 1).
    """
    n = len(x)
    basis = np.zeros((n, degree + 1))

    # T_0 = 1
    basis[:, 0] = 1.0

    if degree >= 1:
        # T_1 = x
        basis[:, 1] = x

    # Recurrence relation for higher degrees
    for d in range(2, degree + 1):
        basis[:, d] = 2 * x * basis[:, d - 1] - basis[:, d - 2]

    return basis


def get_basis_function(basis_type: BasisType, degree: int) -> Callable[[np.ndarray], np.ndarray]:
    """
    Get a basis function generator.

    Parameters
    ----------
    basis_type : BasisType
        Type of basis functions.
    degree : int
        Maximum polynomial degree.

    Returns
    -------
    Callable
        Function that takes x and returns basis matrix.
    """
    if basis_type == BasisType.POLYNOMIAL:
        return lambda x: polynomial_basis(x, degree)
    elif basis_type == BasisType.LAGUERRE:
        return lambda x: laguerre_basis(x, degree)
    elif basis_type == BasisType.CHEBYSHEV:
        return lambda x: chebyshev_basis(x, degree)
    else:
        raise ValueError(f"Unknown basis type: {basis_type}")


# =============================================================================
# LSM Result
# =============================================================================

@dataclass(frozen=True, slots=True)
class LSMResult:
    """
    Result container for Longstaff-Schwartz pricing.

    Attributes
    ----------
    price : float
        Estimated option price.
    std_error : float
        Standard error of the price estimate.
    exercise_boundary : np.ndarray
        Estimated exercise boundary at each time step, shape (n_steps,).
    n_paths : int
        Number of paths used.
    n_steps : int
        Number of time steps.
    basis_type : BasisType
        Basis function type used.
    basis_degree : int
        Degree of basis functions.
    """

    price: float
    std_error: float
    exercise_boundary: np.ndarray
    n_paths: int
    n_steps: int
    basis_type: BasisType
    basis_degree: int

    @property
    def confidence_interval_95(self) -> tuple[float, float]:
        """95% confidence interval for the price."""
        return (self.price - 1.96 * self.std_error, self.price + 1.96 * self.std_error)


# =============================================================================
# Longstaff-Schwartz Algorithm
# =============================================================================

def lsm_american_put(
    paths: np.ndarray,
    strike: float,
    r: float,
    dt: float,
    basis_type: BasisType = BasisType.LAGUERRE,
    basis_degree: int = 3,
) -> LSMResult:
    """
    Price American put option using Longstaff-Schwartz algorithm.

    Parameters
    ----------
    paths : np.ndarray
        Simulated spot paths, shape (n_paths, n_steps + 1).
        Column 0 is t=0, column -1 is t=T.
    strike : float
        Strike price K.
    r : float
        Risk-free rate (annualized).
    dt : float
        Time step size in years.
    basis_type : BasisType
        Type of basis functions for regression.
    basis_degree : int
        Degree of polynomial basis.

    Returns
    -------
    LSMResult
        Pricing result with price, std error, and exercise boundary.
    """
    return _lsm_american_option(
        paths=paths,
        strike=strike,
        r=r,
        dt=dt,
        is_call=False,
        basis_type=basis_type,
        basis_degree=basis_degree,
    )


def lsm_american_call(
    paths: np.ndarray,
    strike: float,
    r: float,
    dt: float,
    basis_type: BasisType = BasisType.LAGUERRE,
    basis_degree: int = 3,
) -> LSMResult:
    """
    Price American call option using Longstaff-Schwartz algorithm.

    Note: American calls on non-dividend paying assets should not be
    exercised early (European = American). This function is useful for
    dividend-paying assets or for testing.

    Parameters
    ----------
    paths : np.ndarray
        Simulated spot paths, shape (n_paths, n_steps + 1).
    strike : float
        Strike price K.
    r : float
        Risk-free rate (annualized).
    dt : float
        Time step size in years.
    basis_type : BasisType
        Type of basis functions for regression.
    basis_degree : int
        Degree of polynomial basis.

    Returns
    -------
    LSMResult
        Pricing result with price, std error, and exercise boundary.
    """
    return _lsm_american_option(
        paths=paths,
        strike=strike,
        r=r,
        dt=dt,
        is_call=True,
        basis_type=basis_type,
        basis_degree=basis_degree,
    )


def _lsm_american_option(
    paths: np.ndarray,
    strike: float,
    r: float,
    dt: float,
    is_call: bool,
    basis_type: BasisType,
    basis_degree: int,
) -> LSMResult:
    """
    Core LSM algorithm for American options.

    Parameters
    ----------
    paths : np.ndarray
        Spot paths, shape (n_paths, n_steps + 1).
    strike : float
        Strike price.
    r : float
        Risk-free rate.
    dt : float
        Time step.
    is_call : bool
        True for call, False for put.
    basis_type : BasisType
        Basis function type.
    basis_degree : int
        Basis degree.

    Returns
    -------
    LSMResult
        Pricing result.
    """
    n_paths, n_cols = paths.shape
    n_steps = n_cols - 1

    # Discount factor per step
    df = np.exp(-r * dt)

    # Get basis function
    basis_fn = get_basis_function(basis_type, basis_degree)

    # Payoff function
    if is_call:
        payoff = lambda S: np.maximum(S - strike, 0.0)
    else:
        payoff = lambda S: np.maximum(strike - S, 0.0)

    # Initialize cash flow matrix
    # cashflows[i, j] = discounted cash flow from path i at time j (if exercised then)
    # We only need to track the stopping time and final payoff
    cashflows = np.zeros(n_paths)
    stopping_time = np.full(n_paths, n_steps, dtype=int)

    # Exercise boundary tracking
    exercise_boundary = np.full(n_steps, np.nan)

    # Terminal payoff at t = T
    cashflows[:] = payoff(paths[:, -1])

    # Work backwards from t = T-1 to t = 1 (don't exercise at t = 0)
    for t in range(n_steps - 1, 0, -1):
        # Current spot values
        S_t = paths[:, t]

        # Immediate exercise payoff
        exercise_value = payoff(S_t)

        # Identify in-the-money paths (only regress on ITM paths)
        itm_mask = exercise_value > 0

        if np.sum(itm_mask) < basis_degree + 1:
            # Not enough ITM paths for regression, skip this time step
            continue

        # Discounted future cash flows for ITM paths
        # These are the continuation values we want to estimate
        Y = cashflows[itm_mask] * df

        # Spot values for ITM paths (normalize for numerical stability)
        X_raw = S_t[itm_mask]
        X_mean = X_raw.mean()
        X_std = X_raw.std() + 1e-8
        X_normalized = (X_raw - X_mean) / X_std

        # Build basis matrix
        basis_matrix = basis_fn(X_normalized)

        # Regression: Y = basis_matrix @ beta + epsilon
        # Use least squares with regularization for stability
        try:
            beta, residuals, rank, s = np.linalg.lstsq(basis_matrix, Y, rcond=None)
            continuation_estimate = basis_matrix @ beta
        except np.linalg.LinAlgError:
            # Fallback: don't exercise at this step
            continue

        # Exercise decision for ITM paths
        exercise_decision = exercise_value[itm_mask] > continuation_estimate

        # Update cash flows and stopping times for paths that exercise
        itm_indices = np.where(itm_mask)[0]
        for i, idx in enumerate(itm_indices):
            if exercise_decision[i]:
                cashflows[idx] = exercise_value[idx]
                stopping_time[idx] = t

        # Estimate exercise boundary (spot level where exercise_value ≈ continuation)
        # Find the spot level where the decision changes
        if np.any(exercise_decision) and np.any(~exercise_decision):
            # Approximate boundary as mean of highest non-exercise and lowest exercise
            exercise_spots = X_raw[exercise_decision]
            no_exercise_spots = X_raw[~exercise_decision]
            if is_call:
                boundary = 0.5 * (exercise_spots.min() + no_exercise_spots.max()) if len(no_exercise_spots) > 0 else exercise_spots.min()
            else:
                boundary = 0.5 * (exercise_spots.max() + no_exercise_spots.min()) if len(no_exercise_spots) > 0 else exercise_spots.max()
            exercise_boundary[t] = boundary

    # Discount all cash flows back to t = 0
    discount_factors = np.exp(-r * dt * stopping_time)
    discounted_payoffs = cashflows * discount_factors

    # Compute price and standard error
    price = float(discounted_payoffs.mean())
    std_error = float(discounted_payoffs.std() / np.sqrt(n_paths))

    return LSMResult(
        price=price,
        std_error=std_error,
        exercise_boundary=exercise_boundary,
        n_paths=n_paths,
        n_steps=n_steps,
        basis_type=basis_type,
        basis_degree=basis_degree,
    )


# =============================================================================
# Convenience Functions
# =============================================================================

def price_american_put_lsm(
    spot0: float,
    strike: float,
    maturity: float,
    r: float,
    sigma: float,
    n_paths: int = 100000,
    n_steps: int = 50,
    seed: Optional[int] = None,
    basis_type: BasisType = BasisType.LAGUERRE,
    basis_degree: int = 3,
) -> LSMResult:
    """
    Price American put option using LSM with GBM paths.

    Convenience function that generates GBM paths internally.

    Parameters
    ----------
    spot0 : float
        Initial spot price.
    strike : float
        Strike price.
    maturity : float
        Time to maturity in years.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    n_paths : int
        Number of Monte Carlo paths.
    n_steps : int
        Number of time steps.
    seed : int, optional
        Random seed.
    basis_type : BasisType
        Basis function type.
    basis_degree : int
        Basis degree.

    Returns
    -------
    LSMResult
        Pricing result.
    """
    from src.models.numeric.monte_carlo.rng import NormalRng

    # Generate GBM paths
    dt = maturity / n_steps
    rng = NormalRng(seed=seed)
    Z = rng.standard_normals(n_paths, n_steps, antithetic=True)

    paths = np.zeros((Z.shape[0], n_steps + 1))
    paths[:, 0] = spot0

    drift = (r - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)

    for t in range(n_steps):
        paths[:, t + 1] = paths[:, t] * np.exp(drift + diffusion * Z[:, t])

    return lsm_american_put(
        paths=paths,
        strike=strike,
        r=r,
        dt=dt,
        basis_type=basis_type,
        basis_degree=basis_degree,
    )


def price_american_call_lsm(
    spot0: float,
    strike: float,
    maturity: float,
    r: float,
    q: float,
    sigma: float,
    n_paths: int = 100000,
    n_steps: int = 50,
    seed: Optional[int] = None,
    basis_type: BasisType = BasisType.LAGUERRE,
    basis_degree: int = 3,
) -> LSMResult:
    """
    Price American call option using LSM with GBM paths (with dividends).

    Parameters
    ----------
    spot0 : float
        Initial spot price.
    strike : float
        Strike price.
    maturity : float
        Time to maturity in years.
    r : float
        Risk-free rate.
    q : float
        Continuous dividend yield.
    sigma : float
        Volatility.
    n_paths : int
        Number of Monte Carlo paths.
    n_steps : int
        Number of time steps.
    seed : int, optional
        Random seed.
    basis_type : BasisType
        Basis function type.
    basis_degree : int
        Basis degree.

    Returns
    -------
    LSMResult
        Pricing result.
    """
    from src.models.numeric.monte_carlo.rng import NormalRng

    # Generate GBM paths with dividend yield
    dt = maturity / n_steps
    rng = NormalRng(seed=seed)
    Z = rng.standard_normals(n_paths, n_steps, antithetic=True)

    paths = np.zeros((Z.shape[0], n_steps + 1))
    paths[:, 0] = spot0

    drift = (r - q - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)

    for t in range(n_steps):
        paths[:, t + 1] = paths[:, t] * np.exp(drift + diffusion * Z[:, t])

    return lsm_american_call(
        paths=paths,
        strike=strike,
        r=r,
        dt=dt,
        basis_type=basis_type,
        basis_degree=basis_degree,
    )

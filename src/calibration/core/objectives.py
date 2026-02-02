"""
Calibration Objective Functions.

This module provides standard objective function types for model calibration:
- WeightedLeastSquares: Σ wᵢ(model - market)² (most common)
- MaxLikelihood: -Σ log(L) for probabilistic calibration
- PenalizedObjective: Adds soft constraints to any objective

Objective functions follow the ObjectiveFunction protocol, which requires
a __call__ method that maps parameter arrays to scalar objective values.

Example
-------
>>> from src.calibration.core.objectives import WeightedLeastSquares
>>>
>>> def model_prices(params):
...     # Return model prices for given parameters
...     return np.array([price1, price2, ...])
>>>
>>> objective = WeightedLeastSquares(
...     model_func=model_prices,
...     market_values=market_prices,
...     weights=vega_weights,  # Weight by vega for vol calibration
... )
>>>
>>> error = objective(params)  # Returns weighted SSE
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Sequence, runtime_checkable

import numpy as np


# =============================================================================
# Objective Function Protocol
# =============================================================================

@runtime_checkable
class ObjectiveFunction(Protocol):
    """
    Protocol for calibration objective functions.
    
    Any callable that takes a parameter array and returns a scalar
    objective value satisfies this protocol.
    
    Parameters
    ----------
    params : np.ndarray
        Parameter array to evaluate.
    
    Returns
    -------
    float
        Objective function value (to be minimized).
    """
    
    def __call__(self, params: np.ndarray) -> float:
        """Evaluate objective at given parameters."""
        ...


# =============================================================================
# Weighted Least Squares
# =============================================================================

@dataclass
class WeightedLeastSquares:
    """
    Weighted least squares objective function.
    
    Computes: Σᵢ wᵢ × (model(params)ᵢ - marketᵢ)²
    
    This is the most common objective for calibration where we want
    to minimize the squared difference between model and market values.
    
    Parameters
    ----------
    model_func : Callable[[np.ndarray], np.ndarray]
        Function that maps parameters to model values.
        Must return array of same length as market_values.
    market_values : np.ndarray
        Target market values to match.
    weights : np.ndarray, optional
        Per-observation weights. If None, uniform weights.
        Weights are normalized to sum to 1.
    use_relative_error : bool
        If True, compute relative error: ((model - market) / market)²
        Useful when market values span different scales.
    
    Examples
    --------
    >>> # Calibrate to implied volatilities
    >>> objective = WeightedLeastSquares(
    ...     model_func=lambda p: heston_implied_vols(p, strikes, expiries),
    ...     market_values=market_vols,
    ...     weights=vegas,  # Weight by vega
    ... )
    >>> error = objective(params)
    
    >>> # With relative errors (for prices spanning different scales)
    >>> objective = WeightedLeastSquares(
    ...     model_func=price_func,
    ...     market_values=market_prices,
    ...     use_relative_error=True,
    ... )
    """
    
    model_func: Callable[[np.ndarray], np.ndarray]
    market_values: np.ndarray
    weights: Optional[np.ndarray] = None
    use_relative_error: bool = False
    
    def __post_init__(self) -> None:
        """Validate and normalize inputs."""
        self.market_values = np.asarray(self.market_values, dtype=float).reshape(-1)
        n = self.market_values.size
        
        if self.weights is None:
            self._weights = np.ones(n, dtype=float) / n
        else:
            w = np.asarray(self.weights, dtype=float).reshape(-1)
            if w.size != n:
                raise ValueError(
                    f"weights length ({w.size}) must match market_values ({n})."
                )
            if np.any(w < 0):
                raise ValueError("weights must be non-negative.")
            w_sum = w.sum()
            if w_sum <= 0:
                raise ValueError("weights must sum to positive value.")
            self._weights = w / w_sum
        
        if self.use_relative_error and np.any(np.abs(self.market_values) < 1e-12):
            raise ValueError(
                "use_relative_error requires non-zero market values."
            )
    
    def __call__(self, params: np.ndarray) -> float:
        """Compute weighted sum of squared errors."""
        try:
            model_values = self.model_func(params)
            model_values = np.asarray(model_values, dtype=float).reshape(-1)
            
            if model_values.size != self.market_values.size:
                return 1e20  # Penalty for invalid output
            
            if self.use_relative_error:
                errors = (model_values - self.market_values) / self.market_values
            else:
                errors = model_values - self.market_values
            
            sse = float(np.sum(self._weights * errors ** 2))
            
            if not np.isfinite(sse):
                return 1e20
            
            return sse
            
        except Exception:
            return 1e20
    
    def residuals(self, params: np.ndarray) -> np.ndarray:
        """
        Compute weighted residuals (for Levenberg-Marquardt).
        
        Returns √wᵢ × (model - market)ᵢ
        """
        model_values = self.model_func(params)
        model_values = np.asarray(model_values, dtype=float).reshape(-1)
        
        if self.use_relative_error:
            errors = (model_values - self.market_values) / self.market_values
        else:
            errors = model_values - self.market_values
        
        return np.sqrt(self._weights) * errors
    
    @property
    def n_observations(self) -> int:
        """Number of market observations."""
        return self.market_values.size


# =============================================================================
# Maximum Likelihood
# =============================================================================

@dataclass
class MaxLikelihood:
    """
    Maximum likelihood objective function.
    
    Computes: -Σᵢ log(L(xᵢ | params))
    
    This is used for probabilistic calibration where we have a likelihood
    function rather than direct model values.
    
    Parameters
    ----------
    log_likelihood_func : Callable[[np.ndarray], float]
        Function that maps parameters to total log-likelihood.
        Should return log(L) (higher = better), objective is -log(L).
    regularization : float
        L2 regularization strength: adds λ × ||params||² to objective.
    
    Examples
    --------
    >>> # Calibrate using likelihood
    >>> def log_likelihood(params):
    ...     # Compute log-likelihood of observed data given params
    ...     return sum(norm.logpdf(x, loc=params[0], scale=params[1]))
    >>>
    >>> objective = MaxLikelihood(log_likelihood_func=log_likelihood)
    >>> mle_estimate = calibrate(objective, initial_params, bounds)
    """
    
    log_likelihood_func: Callable[[np.ndarray], float]
    regularization: float = 0.0
    
    def __call__(self, params: np.ndarray) -> float:
        """Compute negative log-likelihood (to minimize)."""
        try:
            log_lik = float(self.log_likelihood_func(params))
            
            if not np.isfinite(log_lik):
                return 1e20
            
            # Negative log-likelihood (we minimize)
            nll = -log_lik
            
            # Add L2 regularization
            if self.regularization > 0:
                nll += self.regularization * np.sum(params ** 2)
            
            return nll
            
        except Exception:
            return 1e20


# =============================================================================
# Penalized Objective (for constraints)
# =============================================================================

@dataclass
class PenalizedObjective:
    """
    Wrapper that adds soft constraint penalties to any objective.
    
    Useful for enforcing conditions like the Feller constraint in Heston:
        2κθ > ξ²  =>  penalty if ξ² - 2κθ > 0
    
    Parameters
    ----------
    base_objective : Callable[[np.ndarray], float]
        The underlying objective function.
    penalty_func : Callable[[np.ndarray], float]
        Function that returns penalty value (0 if constraint satisfied).
    penalty_weight : float
        Multiplier for penalty term.
    
    Examples
    --------
    >>> # Heston calibration with Feller constraint
    >>> base = WeightedLeastSquares(model_func, market_vols)
    >>>
    >>> def feller_penalty(params):
    ...     kappa, theta, xi, v0, rho = params
    ...     violation = xi**2 - 2 * kappa * theta
    ...     return max(0, violation)**2
    >>>
    >>> objective = PenalizedObjective(
    ...     base_objective=base,
    ...     penalty_func=feller_penalty,
    ...     penalty_weight=1000.0,
    ... )
    """
    
    base_objective: Callable[[np.ndarray], float]
    penalty_func: Callable[[np.ndarray], float]
    penalty_weight: float = 1000.0
    
    def __call__(self, params: np.ndarray) -> float:
        """Compute objective with penalty."""
        try:
            base_value = float(self.base_objective(params))
            penalty = float(self.penalty_func(params))
            
            total = base_value + self.penalty_weight * penalty
            
            if not np.isfinite(total):
                return 1e20
            
            return total
            
        except Exception:
            return 1e20


# =============================================================================
# Combined Objectives
# =============================================================================

@dataclass
class CombinedObjective:
    """
    Combine multiple objectives with weights.
    
    Computes: Σⱼ αⱼ × objectiveⱼ(params)
    
    Useful for multi-criteria calibration, e.g., matching both
    implied vols and option prices.
    
    Parameters
    ----------
    objectives : sequence of Callable
        List of objective functions.
    weights : sequence of float
        Weight for each objective.
    
    Examples
    --------
    >>> combined = CombinedObjective(
    ...     objectives=[vol_objective, price_objective],
    ...     weights=[1.0, 0.1],  # Prioritize vol fit
    ... )
    """
    
    objectives: Sequence[Callable[[np.ndarray], float]]
    weights: Sequence[float]
    
    def __post_init__(self) -> None:
        """Validate inputs."""
        if len(self.objectives) != len(self.weights):
            raise ValueError("objectives and weights must have same length.")
        if len(self.objectives) == 0:
            raise ValueError("At least one objective required.")
    
    def __call__(self, params: np.ndarray) -> float:
        """Compute weighted sum of objectives."""
        total = 0.0
        for obj, w in zip(self.objectives, self.weights):
            try:
                val = float(obj(params))
                if np.isfinite(val):
                    total += w * val
                else:
                    return 1e20
            except Exception:
                return 1e20
        return total


# =============================================================================
# Utility Functions
# =============================================================================

def create_vol_fitting_objective(
    model_vol_func: Callable[[np.ndarray], np.ndarray],
    market_vols: np.ndarray,
    vegas: Optional[np.ndarray] = None,
) -> WeightedLeastSquares:
    """
    Create objective for volatility surface fitting.
    
    Convenience function that creates a WeightedLeastSquares objective
    with vega-weighting (standard practice for vol calibration).
    
    Parameters
    ----------
    model_vol_func : Callable
        Function that computes model implied vols from parameters.
    market_vols : array-like
        Market implied volatilities.
    vegas : array-like, optional
        Option vegas for weighting. If None, uniform weights.
    
    Returns
    -------
    WeightedLeastSquares
        Configured objective function.
    """
    return WeightedLeastSquares(
        model_func=model_vol_func,
        market_values=market_vols,
        weights=vegas,
        use_relative_error=False,
    )


def create_price_fitting_objective(
    model_price_func: Callable[[np.ndarray], np.ndarray],
    market_prices: np.ndarray,
    use_relative: bool = True,
    vegas: Optional[np.ndarray] = None,
) -> WeightedLeastSquares:
    """
    Create objective for option price fitting.
    
    Parameters
    ----------
    model_price_func : Callable
        Function that computes model prices from parameters.
    market_prices : array-like
        Market option prices.
    use_relative : bool
        Use relative errors (recommended for prices spanning scales).
    vegas : array-like, optional
        Option vegas for weighting.
    
    Returns
    -------
    WeightedLeastSquares
        Configured objective function.
    """
    return WeightedLeastSquares(
        model_func=model_price_func,
        market_values=market_prices,
        weights=vegas,
        use_relative_error=use_relative,
    )

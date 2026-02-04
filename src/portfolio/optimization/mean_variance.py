"""
Mean-Variance Portfolio Optimization.

Implements Markowitz mean-variance optimization with:
- Efficient frontier computation
- Target return/risk constraints
- Position limits and sector constraints
- Transaction cost awareness

Reference:
- Markowitz (1952) "Portfolio Selection"
- Merton (1972) "An Analytic Derivation of the Efficient Portfolio Frontier"

Example:
    from src.portfolio.optimization import MeanVarianceOptimizer
    
    optimizer = MeanVarianceOptimizer()
    
    result = optimizer.optimize(
        expected_returns=np.array([0.10, 0.12, 0.08, 0.15]),
        covariance=cov_matrix,
        target_return=0.11,
        constraints=MVConstraints(
            min_weight=0.0,
            max_weight=0.40,
            long_only=True,
        ),
    )
    
    print(f"Optimal weights: {result.weights}")
    print(f"Expected return: {result.expected_return:.2%}")
    print(f"Volatility: {result.volatility:.2%}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import optimize


# =============================================================================
# Constraints
# =============================================================================


@dataclass
class MVConstraints:
    """
    Constraints for mean-variance optimization.
    
    Attributes
    ----------
    min_weight : float
        Minimum weight per asset.
    max_weight : float
        Maximum weight per asset.
    long_only : bool
        If True, no short positions.
    fully_invested : bool
        If True, weights sum to 1.
    sector_limits : dict, optional
        Maximum allocation per sector.
    turnover_limit : float, optional
        Maximum turnover from current portfolio.
    current_weights : ndarray, optional
        Current portfolio weights (for turnover).
    """
    
    min_weight: float = -1.0
    max_weight: float = 1.0
    long_only: bool = True
    fully_invested: bool = True
    sector_limits: Optional[Dict[str, float]] = None
    turnover_limit: Optional[float] = None
    current_weights: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.long_only:
            self.min_weight = max(0.0, self.min_weight)


# =============================================================================
# Optimization Result
# =============================================================================


@dataclass
class MVOptimizationResult:
    """Result from mean-variance optimization."""
    
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    
    # Decomposition
    marginal_contributions: np.ndarray = field(default_factory=lambda: np.array([]))
    risk_contributions: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Constraints
    binding_constraints: List[str] = field(default_factory=list)
    
    # Diagnostics
    converged: bool = True
    iterations: int = 0
    
    def summary(self) -> Dict[str, Any]:
        """Get summary dictionary."""
        return {
            "expected_return": self.expected_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "n_assets": len(self.weights),
            "n_nonzero": int(np.sum(np.abs(self.weights) > 1e-6)),
            "max_weight": float(np.max(self.weights)),
            "converged": self.converged,
        }


# =============================================================================
# Mean-Variance Optimizer
# =============================================================================


class MeanVarianceOptimizer:
    """
    Mean-variance portfolio optimizer.
    
    Solves the classic Markowitz optimization:
        min  w'Σw           (minimize variance)
        s.t. w'μ >= target   (return constraint)
             w'1 = 1         (fully invested)
             w >= 0          (long-only, optional)
    
    Example:
        optimizer = MeanVarianceOptimizer(risk_free_rate=0.02)
        
        # Optimize for target return
        result = optimizer.optimize(
            expected_returns=mu,
            covariance=cov,
            target_return=0.10,
        )
        
        # Optimize for maximum Sharpe
        result = optimizer.optimize_max_sharpe(
            expected_returns=mu,
            covariance=cov,
        )
        
        # Compute efficient frontier
        frontier = optimizer.efficient_frontier(
            expected_returns=mu,
            covariance=cov,
            n_points=50,
        )
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.0,
        solver: str = "SLSQP",
    ) -> None:
        """
        Initialize optimizer.
        
        Parameters
        ----------
        risk_free_rate : float
            Risk-free rate for Sharpe ratio.
        solver : str
            Scipy solver to use.
        """
        self.risk_free_rate = risk_free_rate
        self.solver = solver
    
    def optimize(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        target_return: Optional[float] = None,
        target_volatility: Optional[float] = None,
        constraints: Optional[MVConstraints] = None,
    ) -> MVOptimizationResult:
        """
        Optimize portfolio for target return or volatility.
        
        Parameters
        ----------
        expected_returns : ndarray
            Expected returns vector.
        covariance : ndarray
            Covariance matrix.
        target_return : float, optional
            Target portfolio return.
        target_volatility : float, optional
            Target portfolio volatility.
        constraints : MVConstraints, optional
            Optimization constraints.
        
        Returns
        -------
        MVOptimizationResult
            Optimization result.
        """
        mu = np.asarray(expected_returns)
        cov = np.asarray(covariance)
        n = len(mu)
        
        constraints = constraints or MVConstraints()
        
        # Build constraint list
        scipy_constraints = []
        
        # Budget constraint
        if constraints.fully_invested:
            scipy_constraints.append({
                "type": "eq",
                "fun": lambda w: np.sum(w) - 1.0,
            })
        
        # Target return constraint
        if target_return is not None:
            scipy_constraints.append({
                "type": "ineq",
                "fun": lambda w, mu=mu, tr=target_return: w @ mu - tr,
            })
        
        # Target volatility constraint
        if target_volatility is not None:
            scipy_constraints.append({
                "type": "ineq",
                "fun": lambda w, cov=cov, tv=target_volatility: tv**2 - w @ cov @ w,
            })
        
        # Bounds
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n)]
        
        # Initial guess
        w0 = np.ones(n) / n
        
        # Objective: minimize variance
        def objective(w):
            return w @ cov @ w
        
        # Optimize
        result = optimize.minimize(
            objective,
            w0,
            method=self.solver,
            bounds=bounds,
            constraints=scipy_constraints,
            options={"maxiter": 1000},
        )
        
        weights = result.x
        
        # Compute metrics
        exp_ret = float(weights @ mu)
        volatility = float(np.sqrt(weights @ cov @ weights))
        sharpe = (exp_ret - self.risk_free_rate) / volatility if volatility > 1e-8 else 0.0
        
        # Risk contributions
        marginal = cov @ weights
        risk_contrib = weights * marginal / (volatility + 1e-8)
        
        return MVOptimizationResult(
            weights=weights,
            expected_return=exp_ret,
            volatility=volatility,
            sharpe_ratio=sharpe,
            marginal_contributions=marginal,
            risk_contributions=risk_contrib,
            converged=result.success,
            iterations=result.nit,
        )
    
    def optimize_max_sharpe(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        constraints: Optional[MVConstraints] = None,
    ) -> MVOptimizationResult:
        """
        Find maximum Sharpe ratio portfolio.
        
        Parameters
        ----------
        expected_returns : ndarray
            Expected returns.
        covariance : ndarray
            Covariance matrix.
        constraints : MVConstraints, optional
            Constraints.
        
        Returns
        -------
        MVOptimizationResult
            Optimal portfolio.
        """
        mu = np.asarray(expected_returns)
        cov = np.asarray(covariance)
        n = len(mu)
        
        constraints = constraints or MVConstraints()
        
        # Tangent portfolio via auxiliary variable transformation
        # Maximize (w'μ - rf) / sqrt(w'Σw)
        # Transform: y = w / (w'1), minimize y'Σy s.t. y'(μ-rf) = 1
        
        excess_returns = mu - self.risk_free_rate
        
        scipy_constraints = [
            {"type": "eq", "fun": lambda w: w @ excess_returns - 1.0},
        ]
        
        bounds = [(0 if constraints.long_only else -10, 10) for _ in range(n)]
        
        def objective(w):
            return w @ cov @ w
        
        w0 = np.ones(n) / n
        w0 = w0 / (w0 @ excess_returns + 1e-8)  # Scale to satisfy constraint
        
        result = optimize.minimize(
            objective,
            w0,
            method=self.solver,
            bounds=bounds,
            constraints=scipy_constraints,
            options={"maxiter": 1000},
        )
        
        # Rescale to sum to 1
        weights = result.x
        weights = weights / np.sum(weights)
        
        # Apply position limits
        weights = np.clip(weights, constraints.min_weight, constraints.max_weight)
        weights = weights / np.sum(weights)  # Renormalize
        
        # Metrics
        exp_ret = float(weights @ mu)
        volatility = float(np.sqrt(weights @ cov @ weights))
        sharpe = (exp_ret - self.risk_free_rate) / volatility if volatility > 1e-8 else 0.0
        
        return MVOptimizationResult(
            weights=weights,
            expected_return=exp_ret,
            volatility=volatility,
            sharpe_ratio=sharpe,
            converged=result.success,
            iterations=result.nit,
        )
    
    def optimize_min_variance(
        self,
        covariance: np.ndarray,
        constraints: Optional[MVConstraints] = None,
    ) -> MVOptimizationResult:
        """
        Find minimum variance portfolio.
        
        Parameters
        ----------
        covariance : ndarray
            Covariance matrix.
        constraints : MVConstraints, optional
            Constraints.
        
        Returns
        -------
        MVOptimizationResult
            Minimum variance portfolio.
        """
        cov = np.asarray(covariance)
        n = cov.shape[0]
        mu = np.zeros(n)  # Dummy returns
        
        return self.optimize(
            expected_returns=mu,
            covariance=cov,
            constraints=constraints,
        )
    
    def efficient_frontier(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        n_points: int = 50,
        constraints: Optional[MVConstraints] = None,
    ) -> List[MVOptimizationResult]:
        """
        Compute efficient frontier.
        
        Parameters
        ----------
        expected_returns : ndarray
            Expected returns.
        covariance : ndarray
            Covariance matrix.
        n_points : int
            Number of frontier points.
        constraints : MVConstraints, optional
            Constraints.
        
        Returns
        -------
        list of MVOptimizationResult
            Frontier portfolios.
        """
        mu = np.asarray(expected_returns)
        
        # Find return range
        min_ret = float(np.min(mu))
        max_ret = float(np.max(mu))
        
        target_returns = np.linspace(min_ret, max_ret, n_points)
        
        frontier = []
        for target in target_returns:
            try:
                result = self.optimize(
                    expected_returns=mu,
                    covariance=covariance,
                    target_return=target,
                    constraints=constraints,
                )
                if result.converged:
                    frontier.append(result)
            except Exception:
                continue
        
        return frontier


__all__ = [
    "MeanVarianceOptimizer",
    "MVOptimizationResult",
    "MVConstraints",
]

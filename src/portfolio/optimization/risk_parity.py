"""
Risk Parity Portfolio Optimization.

Constructs portfolios where each asset contributes equally to total risk.

Key insight: Unlike mean-variance, risk parity doesn't require
return estimates, making it more robust to estimation error.

Reference:
- Maillard, Roncalli, Teiletche (2010) "On the Properties of 
  Equally-Weighted Risk Contribution Portfolios"
- Qian (2011) "Risk Parity Portfolios"

Example:
    from src.portfolio.optimization import RiskParityOptimizer
    
    optimizer = RiskParityOptimizer()
    
    result = optimizer.optimize(
        covariance=cov_matrix,
        risk_budgets=np.array([0.25, 0.25, 0.25, 0.25]),  # Equal risk
    )
    
    print(f"Weights: {result.weights}")
    print(f"Risk contributions: {result.risk_contributions}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import optimize


# =============================================================================
# Risk Parity Result
# =============================================================================


@dataclass
class RiskParityResult:
    """Result from risk parity optimization."""
    
    weights: np.ndarray
    volatility: float
    
    # Risk decomposition
    risk_contributions: np.ndarray
    marginal_risks: np.ndarray
    
    # Target budgets
    target_budgets: np.ndarray
    budget_deviation: float  # Sum of squared deviations
    
    # Diagnostics
    converged: bool = True
    iterations: int = 0
    
    def summary(self) -> Dict[str, Any]:
        """Get summary dictionary."""
        return {
            "volatility": self.volatility,
            "max_weight": float(np.max(self.weights)),
            "min_weight": float(np.min(self.weights)),
            "budget_deviation": self.budget_deviation,
            "risk_contribution_range": (
                float(np.min(self.risk_contributions)),
                float(np.max(self.risk_contributions)),
            ),
            "converged": self.converged,
        }


# =============================================================================
# Risk Parity Optimizer
# =============================================================================


class RiskParityOptimizer:
    """
    Risk parity portfolio optimizer.
    
    Finds weights such that each asset's contribution to total
    portfolio variance equals the target risk budget.
    
    Risk contribution of asset i:
        RC_i = w_i * (Σw)_i / σ_p
    
    Risk parity: RC_i = b_i for all i (budget constraint)
    
    Example:
        optimizer = RiskParityOptimizer()
        
        # Equal risk contribution
        result = optimizer.optimize(covariance=cov)
        
        # Custom risk budgets
        result = optimizer.optimize(
            covariance=cov,
            risk_budgets=np.array([0.4, 0.3, 0.2, 0.1]),
        )
        
        # With leverage constraint
        result = optimizer.optimize(
            covariance=cov,
            leverage=2.0,
        )
    """
    
    def __init__(
        self,
        max_iterations: int = 1000,
        tolerance: float = 1e-8,
    ) -> None:
        """
        Initialize optimizer.
        
        Parameters
        ----------
        max_iterations : int
            Maximum iterations.
        tolerance : float
            Convergence tolerance.
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def optimize(
        self,
        covariance: np.ndarray,
        risk_budgets: Optional[np.ndarray] = None,
        leverage: float = 1.0,
        long_only: bool = True,
    ) -> RiskParityResult:
        """
        Optimize for risk parity.
        
        Parameters
        ----------
        covariance : ndarray
            Covariance matrix.
        risk_budgets : ndarray, optional
            Target risk budgets (must sum to 1). Default is equal.
        leverage : float
            Leverage (weights sum to this value).
        long_only : bool
            If True, enforce non-negative weights.
        
        Returns
        -------
        RiskParityResult
            Optimization result.
        """
        cov = np.asarray(covariance)
        n = cov.shape[0]
        
        # Default to equal risk budgets
        if risk_budgets is None:
            risk_budgets = np.ones(n) / n
        else:
            risk_budgets = np.asarray(risk_budgets)
            risk_budgets = risk_budgets / np.sum(risk_budgets)  # Normalize
        
        # Objective: minimize sum of squared differences from target budgets
        def objective(w):
            w = np.abs(w) if long_only else w
            sigma_w = cov @ w
            sigma_p = np.sqrt(w @ sigma_w)
            
            if sigma_p < 1e-10:
                return 1e10
            
            # Risk contributions
            rc = w * sigma_w / sigma_p
            
            # Target contributions
            target_rc = risk_budgets * sigma_p
            
            # Sum of squared deviations
            return np.sum((rc - target_rc) ** 2)
        
        # Budget constraint
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(np.abs(w) if long_only else w) - leverage}
        ]
        
        # Bounds
        if long_only:
            bounds = [(1e-6, 1) for _ in range(n)]
        else:
            bounds = [(-1, 1) for _ in range(n)]
        
        # Initial guess (inverse volatility)
        vols = np.sqrt(np.diag(cov))
        w0 = 1.0 / vols
        w0 = w0 / np.sum(w0) * leverage
        
        # Optimize
        result = optimize.minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": self.max_iterations, "ftol": self.tolerance},
        )
        
        weights = np.abs(result.x) if long_only else result.x
        weights = weights / np.sum(np.abs(weights)) * leverage  # Ensure leverage
        
        # Compute final risk metrics
        sigma_w = cov @ weights
        volatility = float(np.sqrt(weights @ sigma_w))
        marginal_risks = sigma_w / volatility if volatility > 1e-10 else sigma_w
        risk_contributions = weights * marginal_risks
        
        # Normalize to percentages
        total_contrib = np.sum(risk_contributions)
        if total_contrib > 1e-10:
            risk_contributions_pct = risk_contributions / total_contrib
        else:
            risk_contributions_pct = np.ones(n) / n
        
        # Budget deviation
        budget_deviation = float(np.sum((risk_contributions_pct - risk_budgets) ** 2))
        
        return RiskParityResult(
            weights=weights,
            volatility=volatility,
            risk_contributions=risk_contributions_pct,
            marginal_risks=marginal_risks,
            target_budgets=risk_budgets,
            budget_deviation=budget_deviation,
            converged=result.success,
            iterations=result.nit,
        )
    
    def optimize_hierarchical(
        self,
        covariance: np.ndarray,
        clusters: List[List[int]],
        cluster_budgets: Optional[np.ndarray] = None,
    ) -> RiskParityResult:
        """
        Hierarchical risk parity.
        
        First allocates risk budget across clusters, then
        within each cluster using equal risk contribution.
        
        Parameters
        ----------
        covariance : ndarray
            Covariance matrix.
        clusters : list of list
            Asset indices per cluster.
        cluster_budgets : ndarray, optional
            Risk budget per cluster.
        
        Returns
        -------
        RiskParityResult
            Optimization result.
        """
        cov = np.asarray(covariance)
        n = cov.shape[0]
        n_clusters = len(clusters)
        
        if cluster_budgets is None:
            cluster_budgets = np.ones(n_clusters) / n_clusters
        
        weights = np.zeros(n)
        
        # First, compute weights within each cluster
        for c_idx, cluster in enumerate(clusters):
            cluster_cov = cov[np.ix_(cluster, cluster)]
            
            # Equal risk within cluster
            cluster_result = self.optimize(
                covariance=cluster_cov,
                leverage=1.0,
            )
            
            # Scale by cluster budget
            for i, asset_idx in enumerate(cluster):
                weights[asset_idx] = cluster_result.weights[i] * cluster_budgets[c_idx]
        
        # Normalize
        weights = weights / np.sum(weights)
        
        # Compute metrics
        sigma_w = cov @ weights
        volatility = float(np.sqrt(weights @ sigma_w))
        marginal_risks = sigma_w / volatility if volatility > 1e-10 else sigma_w
        risk_contributions = weights * marginal_risks
        
        total_contrib = np.sum(risk_contributions)
        risk_contributions_pct = risk_contributions / total_contrib if total_contrib > 1e-10 else np.ones(n) / n
        
        return RiskParityResult(
            weights=weights,
            volatility=volatility,
            risk_contributions=risk_contributions_pct,
            marginal_risks=marginal_risks,
            target_budgets=np.ones(n) / n,  # Not used in hierarchical
            budget_deviation=0.0,
            converged=True,
        )


__all__ = [
    "RiskParityOptimizer",
    "RiskParityResult",
]

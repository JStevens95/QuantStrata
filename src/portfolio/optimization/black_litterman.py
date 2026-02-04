"""
Black-Litterman Model for Portfolio Optimization.

Combines market equilibrium returns with investor views to produce
more stable, intuitive portfolio allocations.

Key insight: Instead of using raw expected returns (which are very
sensitive to estimation error), BL uses market-implied returns as
a starting point and adjusts based on investor views.

Reference:
- Black & Litterman (1992) "Global Portfolio Optimization"
- He & Litterman (1999) "The Intuition Behind Black-Litterman"

Example:
    from src.portfolio.optimization import BlackLittermanModel
    
    bl = BlackLittermanModel(
        market_caps=caps,
        covariance=cov,
        risk_aversion=2.5,
    )
    
    # Add views
    views = [
        ("AAPL", 0.05),           # AAPL will return 5%
        (["MSFT", "GOOGL"], 0.02),  # MSFT will outperform GOOGL by 2%
    ]
    
    posterior_mu = bl.posterior_mean(views=views, confidences=[0.5, 0.8])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# =============================================================================
# Black-Litterman Result
# =============================================================================


@dataclass
class BlackLittermanResult:
    """Result from Black-Litterman model."""
    
    # Returns
    prior_returns: np.ndarray  # Market equilibrium returns
    posterior_returns: np.ndarray  # After incorporating views
    
    # Covariance
    posterior_covariance: np.ndarray
    
    # View information
    n_views: int
    view_matrix: np.ndarray  # P matrix
    view_returns: np.ndarray  # Q vector
    view_confidences: np.ndarray
    
    # Optimal weights
    optimal_weights: Optional[np.ndarray] = None
    
    def summary(self) -> Dict[str, Any]:
        """Get summary dictionary."""
        return {
            "n_assets": len(self.prior_returns),
            "n_views": self.n_views,
            "prior_return_range": (
                float(np.min(self.prior_returns)),
                float(np.max(self.prior_returns)),
            ),
            "posterior_return_range": (
                float(np.min(self.posterior_returns)),
                float(np.max(self.posterior_returns)),
            ),
        }


# =============================================================================
# Black-Litterman Model
# =============================================================================


class BlackLittermanModel:
    """
    Black-Litterman asset allocation model.
    
    Steps:
    1. Compute equilibrium returns from market capitalization weights
    2. Specify investor views and confidences
    3. Combine to get posterior expected returns
    4. Optimize portfolio using posterior
    
    Example:
        # Initialize with market data
        bl = BlackLittermanModel(
            market_caps=np.array([3e12, 2e12, 1e12, 0.5e12]),
            covariance=cov_matrix,
            risk_aversion=2.5,
        )
        
        # Get equilibrium returns (no views)
        equilibrium = bl.equilibrium_returns()
        
        # Add absolute view: asset 0 will return 10%
        views = [(0, 0.10)]
        
        # Add relative view: asset 1 outperforms asset 2 by 3%
        views.append(([1, 2], 0.03))
        
        # Get posterior
        result = bl.posterior(views=views, confidences=[0.5, 0.3])
        
        # Optimize
        weights = bl.optimize(result)
    """
    
    def __init__(
        self,
        market_caps: np.ndarray,
        covariance: np.ndarray,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        risk_free_rate: float = 0.0,
    ) -> None:
        """
        Initialize Black-Litterman model.
        
        Parameters
        ----------
        market_caps : ndarray
            Market capitalizations for each asset.
        covariance : ndarray
            Covariance matrix.
        risk_aversion : float
            Market risk aversion coefficient (δ).
        tau : float
            Scaling factor for prior uncertainty.
        risk_free_rate : float
            Risk-free rate.
        """
        self.market_caps = np.asarray(market_caps)
        self.covariance = np.asarray(covariance)
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.risk_free_rate = risk_free_rate
        
        self.n_assets = len(market_caps)
        
        # Market weights
        self.market_weights = self.market_caps / np.sum(self.market_caps)
    
    def equilibrium_returns(self) -> np.ndarray:
        """
        Compute equilibrium expected returns.
        
        Uses reverse optimization:
            π = δ * Σ * w_mkt
        
        Returns
        -------
        ndarray
            Equilibrium returns (implied by market weights).
        """
        return self.risk_aversion * self.covariance @ self.market_weights
    
    def posterior(
        self,
        views: List[Tuple[Union[int, List[int]], float]],
        confidences: List[float],
    ) -> BlackLittermanResult:
        """
        Compute posterior distribution incorporating views.
        
        Parameters
        ----------
        views : list of tuples
            Each view is (asset_or_assets, return).
            - Absolute view: (asset_idx, expected_return)
            - Relative view: ([long_idx, short_idx], expected_outperformance)
        confidences : list of float
            Confidence in each view (0 to 1).
        
        Returns
        -------
        BlackLittermanResult
            Posterior distribution and optimal weights.
        """
        # Prior
        pi = self.equilibrium_returns()
        
        # Build view matrices
        P, Q = self._build_view_matrices(views)
        Omega = self._build_uncertainty_matrix(P, confidences)
        
        # Prior covariance of returns
        Sigma_prior = self.tau * self.covariance
        
        # Posterior mean (BL formula)
        # μ_post = π + τΣP'(PτΣP' + Ω)⁻¹(Q - Pπ)
        M = P @ Sigma_prior @ P.T + Omega
        M_inv = np.linalg.inv(M)
        
        adjustment = Sigma_prior @ P.T @ M_inv @ (Q - P @ pi)
        posterior_mu = pi + adjustment
        
        # Posterior covariance
        # Σ_post = Σ + τΣ - τΣP'(PτΣP' + Ω)⁻¹PτΣ
        posterior_cov = self.covariance + Sigma_prior - \
            Sigma_prior @ P.T @ M_inv @ P @ Sigma_prior
        
        # Optimal weights (mean-variance)
        optimal_weights = self._optimize_weights(posterior_mu, posterior_cov)
        
        return BlackLittermanResult(
            prior_returns=pi,
            posterior_returns=posterior_mu,
            posterior_covariance=posterior_cov,
            n_views=len(views),
            view_matrix=P,
            view_returns=Q,
            view_confidences=np.array(confidences),
            optimal_weights=optimal_weights,
        )
    
    def _build_view_matrices(
        self,
        views: List[Tuple[Union[int, List[int]], float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build P (pick) matrix and Q (view returns) vector."""
        n_views = len(views)
        
        P = np.zeros((n_views, self.n_assets))
        Q = np.zeros(n_views)
        
        for i, (assets, ret) in enumerate(views):
            Q[i] = ret
            
            if isinstance(assets, int):
                # Absolute view
                P[i, assets] = 1.0
            else:
                # Relative view: long first asset, short second
                P[i, assets[0]] = 1.0
                if len(assets) > 1:
                    P[i, assets[1]] = -1.0
        
        return P, Q
    
    def _build_uncertainty_matrix(
        self,
        P: np.ndarray,
        confidences: List[float],
    ) -> np.ndarray:
        """
        Build Omega (view uncertainty) matrix.
        
        Uses proportional-to-variance uncertainty:
            Ω_ii = (1/c_i - 1) * P_i * τΣ * P_i'
        
        Lower confidence = higher uncertainty = less impact on posterior.
        """
        n_views = len(confidences)
        Omega = np.zeros((n_views, n_views))
        
        Sigma_prior = self.tau * self.covariance
        
        for i in range(n_views):
            # Variance of the view portfolio
            view_var = P[i] @ Sigma_prior @ P[i].T
            
            # Scale by confidence
            # High confidence (1) -> small uncertainty
            # Low confidence (0) -> large uncertainty
            conf = np.clip(confidences[i], 0.01, 0.99)
            Omega[i, i] = (1.0 / conf - 1.0) * view_var
        
        return Omega
    
    def _optimize_weights(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
    ) -> np.ndarray:
        """Compute optimal weights via mean-variance."""
        # w* = (1/δ) * Σ⁻¹ * μ
        cov_inv = np.linalg.inv(covariance)
        weights = cov_inv @ expected_returns / self.risk_aversion
        
        # Normalize to sum to 1
        weights = weights / np.sum(weights)
        
        # Ensure long-only (optional)
        weights = np.maximum(weights, 0)
        weights = weights / np.sum(weights)
        
        return weights
    
    def sensitivity_analysis(
        self,
        views: List[Tuple[Union[int, List[int]], float]],
        base_confidences: List[float],
        confidence_range: Tuple[float, float] = (0.1, 0.9),
        n_steps: int = 10,
    ) -> Dict[str, np.ndarray]:
        """
        Analyze sensitivity of results to view confidences.
        
        Parameters
        ----------
        views : list
            Views specification.
        base_confidences : list
            Base confidence levels.
        confidence_range : tuple
            Range of confidences to test.
        n_steps : int
            Number of confidence levels to test.
        
        Returns
        -------
        dict
            Sensitivity results.
        """
        results = {
            "confidences": [],
            "posterior_returns": [],
            "weights": [],
        }
        
        conf_levels = np.linspace(confidence_range[0], confidence_range[1], n_steps)
        
        for conf in conf_levels:
            # Use same confidence for all views
            confidences = [conf] * len(views)
            
            result = self.posterior(views, confidences)
            
            results["confidences"].append(conf)
            results["posterior_returns"].append(result.posterior_returns)
            results["weights"].append(result.optimal_weights)
        
        results["confidences"] = np.array(results["confidences"])
        results["posterior_returns"] = np.array(results["posterior_returns"])
        results["weights"] = np.array(results["weights"])
        
        return results


__all__ = [
    "BlackLittermanModel",
    "BlackLittermanResult",
]

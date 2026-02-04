"""
Covariance Matrix Estimation.

Provides robust covariance estimation methods for portfolio optimization:
- Sample covariance
- Shrinkage estimators (Ledoit-Wolf)
- Exponentially weighted
- Factor models

Reference:
- Ledoit & Wolf (2004) "A well-conditioned estimator for large-dimensional covariance matrices"
- DeMiguel et al. (2009) "A Generalized Approach to Portfolio Optimization"

Example:
    from src.portfolio.optimization import CovarianceEstimator, ShrinkageEstimator
    
    # Simple sample covariance
    estimator = CovarianceEstimator()
    cov = estimator.estimate(returns)
    
    # Shrinkage to identity
    shrink_estimator = ShrinkageEstimator(shrinkage_target="identity")
    cov = shrink_estimator.estimate(returns)
    
    # Exponentially weighted
    cov = estimator.estimate_ewm(returns, halflife=60)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


# =============================================================================
# Covariance Estimator
# =============================================================================


class CovarianceEstimator:
    """
    Covariance matrix estimation.
    
    Provides multiple estimation methods:
    - Sample covariance
    - Exponentially weighted
    - Constant correlation
    - Factor model
    
    Example:
        estimator = CovarianceEstimator()
        
        # Sample covariance
        cov = estimator.estimate(returns)
        
        # EWM covariance
        cov = estimator.estimate_ewm(returns, halflife=60)
        
        # With minimum observations
        cov = estimator.estimate(returns, min_observations=100)
    """
    
    def __init__(
        self,
        annualization: int = 252,
    ) -> None:
        """
        Initialize estimator.
        
        Parameters
        ----------
        annualization : int
            Number of observations per year.
        """
        self.annualization = annualization
    
    def estimate(
        self,
        returns: np.ndarray,
        annualize: bool = True,
        min_observations: int = 30,
    ) -> np.ndarray:
        """
        Estimate sample covariance matrix.
        
        Parameters
        ----------
        returns : ndarray
            Returns of shape (n_observations, n_assets).
        annualize : bool
            Whether to annualize.
        min_observations : int
            Minimum observations required.
        
        Returns
        -------
        ndarray
            Covariance matrix.
        """
        returns = np.asarray(returns)
        
        if len(returns) < min_observations:
            raise ValueError(
                f"Need at least {min_observations} observations, got {len(returns)}"
            )
        
        cov = np.cov(returns, rowvar=False)
        
        if annualize:
            cov = cov * self.annualization
        
        return cov
    
    def estimate_ewm(
        self,
        returns: np.ndarray,
        halflife: int = 60,
        annualize: bool = True,
    ) -> np.ndarray:
        """
        Estimate exponentially weighted covariance.
        
        Recent observations have higher weight.
        
        Parameters
        ----------
        returns : ndarray
            Returns.
        halflife : int
            Halflife in observations.
        annualize : bool
            Whether to annualize.
        
        Returns
        -------
        ndarray
            EWM covariance matrix.
        """
        returns = np.asarray(returns)
        n_obs, n_assets = returns.shape
        
        # Compute weights
        alpha = 1 - np.exp(-np.log(2) / halflife)
        weights = np.array([(1 - alpha) ** i for i in range(n_obs - 1, -1, -1)])
        weights = weights / np.sum(weights)
        
        # Weighted mean
        mean = np.sum(weights[:, np.newaxis] * returns, axis=0)
        
        # Centered returns
        centered = returns - mean
        
        # Weighted covariance
        cov = np.zeros((n_assets, n_assets))
        for i in range(n_obs):
            cov += weights[i] * np.outer(centered[i], centered[i])
        
        if annualize:
            cov = cov * self.annualization
        
        return cov
    
    def estimate_constant_correlation(
        self,
        returns: np.ndarray,
        annualize: bool = True,
    ) -> np.ndarray:
        """
        Estimate using constant correlation model.
        
        Assumes all pairwise correlations are equal to the average.
        More stable for large portfolios.
        
        Parameters
        ----------
        returns : ndarray
            Returns.
        annualize : bool
            Whether to annualize.
        
        Returns
        -------
        ndarray
            Constant correlation covariance matrix.
        """
        returns = np.asarray(returns)
        
        # Sample covariance
        sample_cov = np.cov(returns, rowvar=False)
        
        # Standard deviations
        std = np.sqrt(np.diag(sample_cov))
        
        # Sample correlation
        corr = sample_cov / np.outer(std, std)
        np.fill_diagonal(corr, 1.0)
        
        # Average correlation (off-diagonal)
        n = len(std)
        mask = ~np.eye(n, dtype=bool)
        avg_corr = np.mean(corr[mask])
        
        # Constant correlation matrix
        const_corr = np.full((n, n), avg_corr)
        np.fill_diagonal(const_corr, 1.0)
        
        # Convert back to covariance
        cov = const_corr * np.outer(std, std)
        
        if annualize:
            cov = cov * self.annualization
        
        return cov


# =============================================================================
# Shrinkage Estimator
# =============================================================================


@dataclass
class ShrinkageResult:
    """Result from shrinkage estimation."""
    
    covariance: np.ndarray
    shrinkage_intensity: float
    target: str
    
    def summary(self) -> Dict[str, Any]:
        """Get summary dictionary."""
        return {
            "shrinkage_intensity": self.shrinkage_intensity,
            "target": self.target,
            "n_assets": self.covariance.shape[0],
        }


class ShrinkageEstimator:
    """
    Shrinkage covariance estimator (Ledoit-Wolf).
    
    Shrinks sample covariance toward a structured target:
        Σ_shrunk = α * Target + (1 - α) * Σ_sample
    
    Targets:
    - identity: Scaled identity matrix
    - diagonal: Diagonal of sample covariance
    - constant_correlation: Constant correlation structure
    
    The optimal shrinkage intensity α is chosen to minimize
    the Frobenius norm of estimation error.
    
    Example:
        estimator = ShrinkageEstimator(shrinkage_target="identity")
        
        result = estimator.estimate(returns)
        print(f"Shrinkage intensity: {result.shrinkage_intensity:.2%}")
        
        cov = result.covariance
    """
    
    def __init__(
        self,
        shrinkage_target: str = "identity",
        annualization: int = 252,
    ) -> None:
        """
        Initialize shrinkage estimator.
        
        Parameters
        ----------
        shrinkage_target : str
            Target type: "identity", "diagonal", "constant_correlation".
        annualization : int
            Number of observations per year.
        """
        self.shrinkage_target = shrinkage_target
        self.annualization = annualization
    
    def estimate(
        self,
        returns: np.ndarray,
        shrinkage_intensity: Optional[float] = None,
        annualize: bool = True,
    ) -> ShrinkageResult:
        """
        Estimate covariance with shrinkage.
        
        Parameters
        ----------
        returns : ndarray
            Returns of shape (n_observations, n_assets).
        shrinkage_intensity : float, optional
            Manual shrinkage intensity (0 to 1). If None, computed optimally.
        annualize : bool
            Whether to annualize.
        
        Returns
        -------
        ShrinkageResult
            Shrinkage estimation result.
        """
        returns = np.asarray(returns)
        n_obs, n_assets = returns.shape
        
        # Sample covariance
        sample_cov = np.cov(returns, rowvar=False)
        
        # Compute target
        target = self._compute_target(sample_cov)
        
        # Compute optimal shrinkage if not specified
        if shrinkage_intensity is None:
            shrinkage_intensity = self._optimal_shrinkage(
                returns, sample_cov, target
            )
        
        shrinkage_intensity = np.clip(shrinkage_intensity, 0.0, 1.0)
        
        # Shrink
        cov = shrinkage_intensity * target + (1 - shrinkage_intensity) * sample_cov
        
        if annualize:
            cov = cov * self.annualization
        
        return ShrinkageResult(
            covariance=cov,
            shrinkage_intensity=shrinkage_intensity,
            target=self.shrinkage_target,
        )
    
    def _compute_target(self, sample_cov: np.ndarray) -> np.ndarray:
        """Compute shrinkage target matrix."""
        n = sample_cov.shape[0]
        
        if self.shrinkage_target == "identity":
            # Scaled identity (average variance on diagonal)
            avg_var = np.trace(sample_cov) / n
            return avg_var * np.eye(n)
        
        elif self.shrinkage_target == "diagonal":
            # Diagonal matrix
            return np.diag(np.diag(sample_cov))
        
        elif self.shrinkage_target == "constant_correlation":
            # Constant correlation structure
            std = np.sqrt(np.diag(sample_cov))
            corr = sample_cov / np.outer(std, std)
            np.fill_diagonal(corr, 1.0)
            
            mask = ~np.eye(n, dtype=bool)
            avg_corr = np.mean(corr[mask])
            
            const_corr = np.full((n, n), avg_corr)
            np.fill_diagonal(const_corr, 1.0)
            
            return const_corr * np.outer(std, std)
        
        else:
            raise ValueError(f"Unknown shrinkage target: {self.shrinkage_target}")
    
    def _optimal_shrinkage(
        self,
        returns: np.ndarray,
        sample_cov: np.ndarray,
        target: np.ndarray,
    ) -> float:
        """
        Compute optimal shrinkage intensity (Ledoit-Wolf formula).
        
        Minimizes the expected Frobenius norm of estimation error.
        """
        n_obs, n_assets = returns.shape
        
        # Center returns
        mean = np.mean(returns, axis=0)
        centered = returns - mean
        
        # Compute components for optimal shrinkage
        # Following Ledoit-Wolf (2004)
        
        # Sample covariance elements as outer products
        X2 = np.zeros((n_obs, n_assets, n_assets))
        for t in range(n_obs):
            X2[t] = np.outer(centered[t], centered[t])
        
        # Mean of squared sample cov
        sample_cov2 = sample_cov ** 2
        
        # pi: sum of asymptotic variances of sample cov entries
        pi = 0.0
        for i in range(n_assets):
            for j in range(n_assets):
                pi += np.sum((X2[:, i, j] - sample_cov[i, j]) ** 2) / n_obs
        
        # rho: sum of asymptotic covariances between target and sample
        rho = 0.0
        for i in range(n_assets):
            for j in range(n_assets):
                rho += np.sum(
                    (X2[:, i, j] - sample_cov[i, j]) * (target[i, j] - sample_cov[i, j])
                ) / n_obs
        
        # gamma: squared Frobenius norm of (target - sample)
        gamma = np.sum((target - sample_cov) ** 2)
        
        # Optimal shrinkage
        if gamma < 1e-10:
            return 0.0
        
        kappa = (pi - rho) / gamma
        shrinkage = kappa / n_obs
        
        return float(np.clip(shrinkage, 0.0, 1.0))


__all__ = [
    "CovarianceEstimator",
    "ShrinkageEstimator",
    "ShrinkageResult",
]

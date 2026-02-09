"""
Unit tests for risk parity optimization module.

Tests RiskParityOptimizer and RiskParityResult.
"""

import numpy as np
import pytest

from src.portfolio.optimization.risk_parity import (
    RiskParityOptimizer,
    RiskParityResult,
)


class TestRiskParityResult:
    """Tests for RiskParityResult dataclass."""
    
    def test_result_creation(self) -> None:
        """Test result creation."""
        result = RiskParityResult(
            weights=np.array([0.3, 0.35, 0.35]),
            volatility=0.10,
            risk_contributions=np.array([0.33, 0.33, 0.34]),
            marginal_risks=np.array([0.1, 0.1, 0.1]),
            target_budgets=np.array([1 / 3, 1 / 3, 1 / 3]),
            budget_deviation=0.0,
        )
        
        assert len(result.weights) == 3
        assert abs(sum(result.weights) - 1.0) < 1e-6


class TestRiskParityOptimizer:
    """Tests for RiskParityOptimizer."""
    
    @pytest.fixture
    def simple_cov(self) -> np.ndarray:
        """Create simple covariance matrix."""
        return np.array([
            [0.04, 0.01, 0.02],
            [0.01, 0.03, 0.01],
            [0.02, 0.01, 0.05],
        ])
    
    def test_optimizer_creation(self) -> None:
        """Test optimizer creation."""
        optimizer = RiskParityOptimizer(max_iterations=1000)
        
        assert optimizer.max_iterations == 1000
    
    def test_optimize_equal_risk(self, simple_cov: np.ndarray) -> None:
        """Test equal risk contribution optimization."""
        optimizer = RiskParityOptimizer()
        
        result = optimizer.optimize(covariance=simple_cov)
        
        # Weights should sum to 1
        assert abs(sum(result.weights) - 1.0) < 1e-4
        
        # Risk contributions should be approximately equal
        rc = result.risk_contributions
        avg_rc = np.mean(rc)
        for r in rc:
            assert abs(r - avg_rc) < 0.05
    
    def test_optimize_with_risk_budgets(self, simple_cov: np.ndarray) -> None:
        """Test optimization with custom risk budgets."""
        optimizer = RiskParityOptimizer()
        
        # Asset 0 contributes 50%, assets 1 and 2 contribute 25% each
        risk_budgets = np.array([0.5, 0.25, 0.25])
        
        result = optimizer.optimize(
            covariance=simple_cov,
            risk_budgets=risk_budgets,
        )
        
        # First asset should contribute more risk
        assert result.risk_contributions[0] > result.risk_contributions[1]
    
    def test_optimize_with_leverage(self, simple_cov: np.ndarray) -> None:
        """Test optimization with leverage constraint."""
        optimizer = RiskParityOptimizer()
        
        result = optimizer.optimize(
            covariance=simple_cov,
            leverage=1.5,
        )
        
        # Weights should sum to 1.5
        assert abs(sum(result.weights) - 1.5) < 1e-4
    
    def test_long_only_constraint(self, simple_cov: np.ndarray) -> None:
        """Test long-only constraint."""
        optimizer = RiskParityOptimizer()
        
        result = optimizer.optimize(
            covariance=simple_cov,
            long_only=True,
        )
        
        # All weights should be non-negative
        assert all(w >= -1e-6 for w in result.weights)
    
    def test_two_asset_portfolio(self) -> None:
        """Test with two assets of different volatilities."""
        # Asset 1: 10% vol, Asset 2: 20% vol
        cov = np.array([
            [0.01, 0.0],
            [0.0, 0.04],
        ])
        
        optimizer = RiskParityOptimizer()
        result = optimizer.optimize(covariance=cov)
        
        # Lower vol asset should have higher weight
        assert result.weights[0] > result.weights[1]
        
        # Risk contributions should be equal
        assert abs(result.risk_contributions[0] - result.risk_contributions[1]) < 0.02
    
    def test_diagonal_covariance(self) -> None:
        """Test with diagonal covariance (uncorrelated assets)."""
        vols = np.array([0.10, 0.15, 0.20, 0.25])
        cov = np.diag(vols ** 2)
        
        optimizer = RiskParityOptimizer()
        result = optimizer.optimize(covariance=cov)
        
        # Risk contributions should be equal
        rc = result.risk_contributions
        for i in range(len(rc) - 1):
            assert abs(rc[i] - rc[i+1]) < 0.02
    
    def test_hierarchical_risk_parity(self, simple_cov: np.ndarray) -> None:
        """Test hierarchical risk parity (if implemented)."""
        optimizer = RiskParityOptimizer()
        
        # HRP should work if method exists
        if hasattr(optimizer, "hierarchical"):
            result = optimizer.hierarchical(covariance=simple_cov)
            assert abs(sum(result.weights) - 1.0) < 1e-4
        else:
            # Standard RP as fallback
            result = optimizer.optimize(covariance=simple_cov)
            assert len(result.weights) == 3
    
    def test_volatility_calculation(self, simple_cov: np.ndarray) -> None:
        """Test that portfolio volatility is calculated correctly."""
        optimizer = RiskParityOptimizer()
        result = optimizer.optimize(covariance=simple_cov)
        
        # Manually calculate volatility
        w = result.weights
        manual_vol = np.sqrt(w @ simple_cov @ w)
        
        assert abs(result.volatility - manual_vol) < 1e-6
    
    def test_convergence_with_many_assets(self) -> None:
        """Test convergence with larger portfolio."""
        np.random.seed(42)
        n_assets = 15
        
        # Generate positive definite covariance
        A = np.random.randn(n_assets, n_assets) * 0.1
        cov = A @ A.T + np.eye(n_assets) * 0.02
        
        optimizer = RiskParityOptimizer(max_iterations=2000)
        result = optimizer.optimize(covariance=cov)
        
        # Should converge
        assert abs(sum(result.weights) - 1.0) < 1e-3
        
        # Risk contributions should be approximately equal
        rc = result.risk_contributions
        avg_rc = np.mean(rc)
        for r in rc:
            assert abs(r - avg_rc) < 0.05

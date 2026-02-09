"""
Unit tests for mean-variance optimization module.

Tests MeanVarianceOptimizer, MVConstraints, and MVOptimizationResult.
"""

import numpy as np
import pytest

from src.portfolio.optimization.mean_variance import (
    MeanVarianceOptimizer,
    MVConstraints,
    MVOptimizationResult,
)


class TestMVConstraints:
    """Tests for MVConstraints dataclass."""
    
    def test_default_constraints(self) -> None:
        """Test default constraints."""
        constraints = MVConstraints()
        
        assert constraints.long_only is True
        assert constraints.max_weight == 1.0
        assert constraints.min_weight == 0.0  # set in __post_init__ when long_only
    
    def test_custom_constraints(self) -> None:
        """Test custom constraints."""
        constraints = MVConstraints(
            long_only=False,
            max_weight=0.3,
            min_weight=-0.1,
            sector_limits={"tech": 0.4},
        )
        
        assert constraints.long_only is False
        assert constraints.max_weight == 0.3


class TestMVOptimizationResult:
    """Tests for MVOptimizationResult dataclass."""
    
    def test_result_creation(self) -> None:
        """Test result creation."""
        result = MVOptimizationResult(
            weights=np.array([0.4, 0.3, 0.3]),
            expected_return=0.08,
            volatility=0.15,
            sharpe_ratio=0.53,
        )
        
        assert len(result.weights) == 3
        assert abs(sum(result.weights) - 1.0) < 1e-6
        assert result.expected_return == 0.08


class TestMeanVarianceOptimizer:
    """Tests for MeanVarianceOptimizer."""
    
    @pytest.fixture
    def simple_inputs(self) -> tuple:
        """Create simple optimization inputs."""
        expected_returns = np.array([0.10, 0.08, 0.12])
        covariance = np.array([
            [0.04, 0.01, 0.02],
            [0.01, 0.03, 0.01],
            [0.02, 0.01, 0.05],
        ])
        return expected_returns, covariance
    
    def test_optimizer_creation(self) -> None:
        """Test optimizer creation."""
        optimizer = MeanVarianceOptimizer(risk_free_rate=0.02)
        
        assert optimizer.risk_free_rate == 0.02
    
    def test_optimize_max_sharpe(self, simple_inputs: tuple) -> None:
        """Test max Sharpe ratio optimization."""
        returns, cov = simple_inputs
        optimizer = MeanVarianceOptimizer(risk_free_rate=0.02)
        
        result = optimizer.optimize(
            expected_returns=returns,
            covariance=cov,
        )
        
        # Weights should sum to 1
        assert abs(sum(result.weights) - 1.0) < 1e-6
        
        # Sharpe should be positive
        assert result.sharpe_ratio > 0
    
    def test_optimize_target_return(self, simple_inputs: tuple) -> None:
        """Test optimization with target return."""
        returns, cov = simple_inputs
        optimizer = MeanVarianceOptimizer()
        
        result = optimizer.optimize(
            expected_returns=returns,
            covariance=cov,
            target_return=0.09,
        )
        
        # Should achieve target return
        assert abs(result.expected_return - 0.09) < 0.01
    
    def test_optimize_target_volatility(self, simple_inputs: tuple) -> None:
        """Test optimization with target volatility."""
        returns, cov = simple_inputs
        optimizer = MeanVarianceOptimizer()
        
        result = optimizer.optimize(
            expected_returns=returns,
            covariance=cov,
            target_volatility=0.15,
        )
        
        # Should achieve target volatility
        assert abs(result.volatility - 0.15) < 0.02
    
    def test_optimize_min_variance(self, simple_inputs: tuple) -> None:
        """Test minimum variance portfolio (optimize with no target return/vol)."""
        returns, cov = simple_inputs
        optimizer = MeanVarianceOptimizer()
        
        # No target return or volatility -> minimizes variance subject to budget
        result = optimizer.optimize(
            expected_returns=returns,
            covariance=cov,
        )
        
        # Weights should sum to 1
        assert abs(sum(result.weights) - 1.0) < 1e-6
        
        # Should have minimum possible variance (positive)
        assert result.volatility > 0
    
    def test_long_only_constraint(self, simple_inputs: tuple) -> None:
        """Test long-only constraint."""
        returns, cov = simple_inputs
        optimizer = MeanVarianceOptimizer()
        
        constraints = MVConstraints(long_only=True)
        
        result = optimizer.optimize(
            expected_returns=returns,
            covariance=cov,
            constraints=constraints,
        )
        
        # All weights should be non-negative
        assert all(w >= -1e-6 for w in result.weights)
    
    def test_max_weight_constraint(self, simple_inputs: tuple) -> None:
        """Test max weight constraint."""
        returns, cov = simple_inputs
        optimizer = MeanVarianceOptimizer()
        
        constraints = MVConstraints(max_weight=0.5)
        
        result = optimizer.optimize(
            expected_returns=returns,
            covariance=cov,
            constraints=constraints,
        )
        
        # All weights should be <= 0.5
        assert all(w <= 0.5 + 1e-6 for w in result.weights)
    
    def test_efficient_frontier(self, simple_inputs: tuple) -> None:
        """Test efficient frontier computation."""
        returns, cov = simple_inputs
        optimizer = MeanVarianceOptimizer()
        
        frontier = optimizer.efficient_frontier(
            expected_returns=returns,
            covariance=cov,
            n_points=10,
        )
        
        # Should have multiple portfolios
        assert len(frontier) == 10
        
        # Returns should be increasing
        for i in range(len(frontier) - 1):
            assert frontier[i+1].expected_return >= frontier[i].expected_return - 1e-6
    
    def test_weights_sum_to_one(self, simple_inputs: tuple) -> None:
        """Test that weights always sum to 1."""
        returns, cov = simple_inputs
        optimizer = MeanVarianceOptimizer()
        
        for target_ret in [0.08, 0.09, 0.10, 0.11]:
            result = optimizer.optimize(
                expected_returns=returns,
                covariance=cov,
                target_return=target_ret,
            )
            
            assert abs(sum(result.weights) - 1.0) < 1e-4
    
    def test_with_two_assets(self) -> None:
        """Test optimization with just two assets."""
        returns = np.array([0.08, 0.12])
        cov = np.array([
            [0.04, 0.01],
            [0.01, 0.09],
        ])
        
        optimizer = MeanVarianceOptimizer()
        result = optimizer.optimize(
            expected_returns=returns,
            covariance=cov,
        )
        
        assert len(result.weights) == 2
        assert abs(sum(result.weights) - 1.0) < 1e-6
    
    def test_with_many_assets(self) -> None:
        """Test optimization with many assets."""
        np.random.seed(42)
        n_assets = 20
        
        returns = np.random.uniform(0.05, 0.15, n_assets)
        
        # Generate positive definite covariance
        A = np.random.randn(n_assets, n_assets) * 0.1
        cov = A @ A.T + np.eye(n_assets) * 0.01
        
        optimizer = MeanVarianceOptimizer()
        result = optimizer.optimize(
            expected_returns=returns,
            covariance=cov,
        )
        
        assert len(result.weights) == n_assets
        assert abs(sum(result.weights) - 1.0) < 1e-4

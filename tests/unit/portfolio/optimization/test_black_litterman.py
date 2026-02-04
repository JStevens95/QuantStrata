"""
Unit tests for Black-Litterman model module.

Tests BlackLittermanModel and BlackLittermanResult.
"""

import numpy as np
import pytest

from src.portfolio.optimization.black_litterman import (
    BlackLittermanModel,
    BlackLittermanResult,
)


class TestBlackLittermanResult:
    """Tests for BlackLittermanResult dataclass."""
    
    def test_result_creation(self) -> None:
        """Test result creation."""
        result = BlackLittermanResult(
            posterior_returns=np.array([0.08, 0.09, 0.10]),
            posterior_covariance=np.eye(3) * 0.04,
            optimal_weights=np.array([0.3, 0.35, 0.35]),
            equilibrium_returns=np.array([0.07, 0.08, 0.09]),
        )
        
        assert len(result.posterior_returns) == 3
        assert result.posterior_covariance.shape == (3, 3)


class TestBlackLittermanModel:
    """Tests for BlackLittermanModel."""
    
    @pytest.fixture
    def simple_inputs(self) -> tuple:
        """Create simple model inputs."""
        market_caps = np.array([100.0, 80.0, 50.0])  # Billions
        covariance = np.array([
            [0.04, 0.01, 0.02],
            [0.01, 0.03, 0.01],
            [0.02, 0.01, 0.05],
        ])
        return market_caps, covariance
    
    def test_model_creation(self, simple_inputs: tuple) -> None:
        """Test model creation."""
        market_caps, cov = simple_inputs
        
        model = BlackLittermanModel(
            market_caps=market_caps,
            covariance=cov,
            risk_aversion=2.5,
            tau=0.05,
        )
        
        assert model.risk_aversion == 2.5
        assert model.tau == 0.05
    
    def test_equilibrium_returns(self, simple_inputs: tuple) -> None:
        """Test equilibrium returns calculation."""
        market_caps, cov = simple_inputs
        
        model = BlackLittermanModel(
            market_caps=market_caps,
            covariance=cov,
        )
        
        eq_returns = model.equilibrium_returns
        
        # Should have one return per asset
        assert len(eq_returns) == 3
        
        # All returns should be positive (assuming positive risk premium)
        assert all(r >= 0 for r in eq_returns)
    
    def test_market_weights(self, simple_inputs: tuple) -> None:
        """Test market cap weights calculation."""
        market_caps, cov = simple_inputs
        
        model = BlackLittermanModel(
            market_caps=market_caps,
            covariance=cov,
        )
        
        weights = model.market_weights
        
        # Should sum to 1
        assert abs(sum(weights) - 1.0) < 1e-10
        
        # Proportional to market caps
        assert weights[0] > weights[1] > weights[2]
    
    def test_posterior_no_views(self, simple_inputs: tuple) -> None:
        """Test posterior with no views."""
        market_caps, cov = simple_inputs
        
        model = BlackLittermanModel(
            market_caps=market_caps,
            covariance=cov,
        )
        
        result = model.posterior(views=[], confidences=[])
        
        # Posterior should equal equilibrium with no views
        np.testing.assert_array_almost_equal(
            result.posterior_returns,
            result.equilibrium_returns,
        )
    
    def test_posterior_absolute_view(self, simple_inputs: tuple) -> None:
        """Test posterior with absolute view."""
        market_caps, cov = simple_inputs
        
        model = BlackLittermanModel(
            market_caps=market_caps,
            covariance=cov,
        )
        
        # View: Asset 0 will return 10%
        views = [(0, 0.10)]
        confidences = [0.5]  # 50% confidence
        
        result = model.posterior(views=views, confidences=confidences)
        
        # Posterior return for asset 0 should be between equilibrium and view
        eq = result.equilibrium_returns[0]
        view = 0.10
        post = result.posterior_returns[0]
        
        assert min(eq, view) <= post <= max(eq, view)
    
    def test_posterior_relative_view(self, simple_inputs: tuple) -> None:
        """Test posterior with relative view."""
        market_caps, cov = simple_inputs
        
        model = BlackLittermanModel(
            market_caps=market_caps,
            covariance=cov,
        )
        
        # View: Asset 0 outperforms asset 2 by 2%
        views = [([0, 2], 0.02)]  # [assets], excess return
        confidences = [0.6]
        
        result = model.posterior(views=views, confidences=confidences)
        
        # Posterior returns should reflect the relative view
        spread = result.posterior_returns[0] - result.posterior_returns[2]
        
        # Should be closer to 2% than equilibrium spread
        eq_spread = result.equilibrium_returns[0] - result.equilibrium_returns[2]
        assert abs(spread - 0.02) <= abs(eq_spread - 0.02)
    
    def test_high_confidence_view(self, simple_inputs: tuple) -> None:
        """Test that high confidence pushes posterior toward view."""
        market_caps, cov = simple_inputs
        
        model = BlackLittermanModel(
            market_caps=market_caps,
            covariance=cov,
        )
        
        # Very confident view
        views = [(0, 0.15)]
        confidences = [0.99]
        
        result = model.posterior(views=views, confidences=confidences)
        
        # Posterior should be close to view
        assert abs(result.posterior_returns[0] - 0.15) < 0.03
    
    def test_low_confidence_view(self, simple_inputs: tuple) -> None:
        """Test that low confidence keeps posterior near equilibrium."""
        market_caps, cov = simple_inputs
        
        model = BlackLittermanModel(
            market_caps=market_caps,
            covariance=cov,
        )
        
        # Very low confidence view
        views = [(0, 0.15)]
        confidences = [0.01]
        
        result = model.posterior(views=views, confidences=confidences)
        
        # Posterior should be close to equilibrium
        eq = result.equilibrium_returns[0]
        assert abs(result.posterior_returns[0] - eq) < abs(0.15 - eq)
    
    def test_optimal_weights_sum_to_one(self, simple_inputs: tuple) -> None:
        """Test that optimal weights sum to one."""
        market_caps, cov = simple_inputs
        
        model = BlackLittermanModel(
            market_caps=market_caps,
            covariance=cov,
        )
        
        views = [(0, 0.12)]
        confidences = [0.5]
        
        result = model.posterior(views=views, confidences=confidences)
        
        assert abs(sum(result.optimal_weights) - 1.0) < 1e-4
    
    def test_multiple_views(self, simple_inputs: tuple) -> None:
        """Test with multiple views."""
        market_caps, cov = simple_inputs
        
        model = BlackLittermanModel(
            market_caps=market_caps,
            covariance=cov,
        )
        
        # Multiple views
        views = [
            (0, 0.12),  # Asset 0 returns 12%
            (1, 0.08),  # Asset 1 returns 8%
        ]
        confidences = [0.6, 0.5]
        
        result = model.posterior(views=views, confidences=confidences)
        
        # Should incorporate both views
        assert len(result.posterior_returns) == 3
        assert result.optimal_weights is not None
    
    def test_posterior_covariance_positive_definite(self, simple_inputs: tuple) -> None:
        """Test that posterior covariance is positive definite."""
        market_caps, cov = simple_inputs
        
        model = BlackLittermanModel(
            market_caps=market_caps,
            covariance=cov,
        )
        
        views = [(0, 0.10)]
        confidences = [0.5]
        
        result = model.posterior(views=views, confidences=confidences)
        
        # Check positive definite via eigenvalues
        eigenvalues = np.linalg.eigvalsh(result.posterior_covariance)
        assert all(e > 0 for e in eigenvalues)

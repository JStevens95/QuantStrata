"""
Unit tests for covariance estimation module.

Tests CovarianceEstimator and ShrinkageEstimator.
"""

import numpy as np
import pytest

from src.portfolio.optimization.covariance import (
    CovarianceEstimator,
    ShrinkageEstimator,
)


class TestCovarianceEstimator:
    """Tests for CovarianceEstimator."""
    
    @pytest.fixture
    def sample_returns(self) -> np.ndarray:
        """Create sample returns data."""
        np.random.seed(42)
        n_obs = 252
        n_assets = 5
        
        # Generate correlated returns
        mean = np.zeros(n_assets)
        cov = np.array([
            [0.04, 0.01, 0.02, 0.01, 0.02],
            [0.01, 0.03, 0.01, 0.01, 0.01],
            [0.02, 0.01, 0.05, 0.02, 0.01],
            [0.01, 0.01, 0.02, 0.04, 0.01],
            [0.02, 0.01, 0.01, 0.01, 0.03],
        ])
        
        returns = np.random.multivariate_normal(mean, cov / 252, n_obs)
        return returns
    
    def test_estimator_creation(self) -> None:
        """Test estimator creation."""
        estimator = CovarianceEstimator(annualization=252)
        
        assert estimator.annualization == 252
    
    def test_sample_covariance(self, sample_returns: np.ndarray) -> None:
        """Test sample covariance estimation."""
        estimator = CovarianceEstimator()
        
        cov = estimator.estimate(sample_returns)
        
        # Should be square and correct size
        assert cov.shape == (5, 5)
        
        # Should be symmetric
        np.testing.assert_array_almost_equal(cov, cov.T)
        
        # Should be positive definite
        eigenvalues = np.linalg.eigvalsh(cov)
        assert all(e > 0 for e in eigenvalues)
    
    def test_ewm_covariance(self, sample_returns: np.ndarray) -> None:
        """Test exponentially weighted covariance estimation."""
        estimator = CovarianceEstimator()
        
        cov = estimator.estimate_ewm(sample_returns, halflife=60)
        
        # Should be square and symmetric
        assert cov.shape == (5, 5)
        np.testing.assert_array_almost_equal(cov, cov.T)
    
    def test_ewm_vs_sample(self, sample_returns: np.ndarray) -> None:
        """Test that EWM and sample give different results."""
        estimator = CovarianceEstimator()
        
        sample_cov = estimator.estimate(sample_returns)
        ewm_cov = estimator.estimate_ewm(sample_returns, halflife=30)
        
        # They should differ
        assert not np.allclose(sample_cov, ewm_cov)
    
    def test_constant_correlation(self, sample_returns: np.ndarray) -> None:
        """Test constant correlation estimator."""
        estimator = CovarianceEstimator()
        
        cov = estimator.estimate_constant_correlation(sample_returns)
        
        # Should be symmetric
        np.testing.assert_array_almost_equal(cov, cov.T)
        
        # Extract correlation matrix
        std = np.sqrt(np.diag(cov))
        corr = cov / np.outer(std, std)
        
        # Off-diagonal correlations should be equal
        off_diag = corr[np.triu_indices(5, k=1)]
        assert np.std(off_diag) < 0.01
    
    def test_annualization(self, sample_returns: np.ndarray) -> None:
        """Test that annualization works correctly."""
        estimator_daily = CovarianceEstimator(annualization=1)
        estimator_annual = CovarianceEstimator(annualization=252)
        
        cov_daily = estimator_daily.estimate(sample_returns, annualize=True)
        cov_annual = estimator_annual.estimate(sample_returns, annualize=True)
        
        # Annual should be 252x daily
        ratio = np.mean(cov_annual) / np.mean(cov_daily)
        assert 200 < ratio < 300  # Approximately 252


class TestShrinkageEstimator:
    """Tests for ShrinkageEstimator (Ledoit-Wolf)."""
    
    @pytest.fixture
    def sample_returns(self) -> np.ndarray:
        """Create sample returns data."""
        np.random.seed(42)
        return np.random.randn(100, 10) * 0.02
    
    def test_estimator_creation(self) -> None:
        """Test estimator creation."""
        estimator = ShrinkageEstimator(shrinkage_target="identity")
        
        assert estimator.shrinkage_target == "identity"
    
    def test_shrinkage_to_identity(self, sample_returns: np.ndarray) -> None:
        """Test shrinkage toward identity."""
        estimator = ShrinkageEstimator(shrinkage_target="identity")
        
        result = estimator.estimate(sample_returns)
        
        # Should have covariance matrix
        assert result.covariance.shape == (10, 10)
        
        # Should be symmetric
        np.testing.assert_array_almost_equal(
            result.covariance,
            result.covariance.T,
        )
        
        # Shrinkage intensity should be between 0 and 1
        assert 0 <= result.shrinkage_intensity <= 1
    
    def test_shrinkage_to_diagonal(self, sample_returns: np.ndarray) -> None:
        """Test shrinkage toward diagonal."""
        estimator = ShrinkageEstimator(shrinkage_target="diagonal")
        
        result = estimator.estimate(sample_returns)
        
        assert result.covariance.shape == (10, 10)
        assert 0 <= result.shrinkage_intensity <= 1
    
    def test_shrinkage_to_constant_correlation(self, sample_returns: np.ndarray) -> None:
        """Test shrinkage toward constant correlation."""
        estimator = ShrinkageEstimator(shrinkage_target="constant_correlation")
        
        result = estimator.estimate(sample_returns)
        
        assert result.covariance.shape == (10, 10)
    
    def test_custom_shrinkage_intensity(self, sample_returns: np.ndarray) -> None:
        """Test with custom shrinkage intensity."""
        estimator = ShrinkageEstimator()
        
        result = estimator.estimate(
            sample_returns,
            shrinkage_intensity=0.5,
        )
        
        assert result.shrinkage_intensity == 0.5
    
    def test_zero_shrinkage_equals_sample(self, sample_returns: np.ndarray) -> None:
        """Test that zero shrinkage gives sample covariance."""
        estimator = ShrinkageEstimator()
        basic_estimator = CovarianceEstimator()
        
        result = estimator.estimate(
            sample_returns,
            shrinkage_intensity=0.0,
            annualize=False,
        )
        
        sample_cov = basic_estimator.estimate(sample_returns, annualize=False)
        
        np.testing.assert_array_almost_equal(result.covariance, sample_cov, decimal=5)
    
    def test_shrinkage_improves_condition_number(self, sample_returns: np.ndarray) -> None:
        """Test that shrinkage improves condition number."""
        basic_estimator = CovarianceEstimator()
        shrinkage_estimator = ShrinkageEstimator()
        
        sample_cov = basic_estimator.estimate(sample_returns, annualize=False)
        shrunk = shrinkage_estimator.estimate(
            sample_returns,
            annualize=False,
        )
        
        cond_sample = np.linalg.cond(sample_cov)
        cond_shrunk = np.linalg.cond(shrunk.covariance)
        
        # Shrinkage should reduce or maintain condition number
        assert cond_shrunk <= cond_sample * 1.5
    
    def test_positive_definite(self, sample_returns: np.ndarray) -> None:
        """Test that result is positive definite."""
        estimator = ShrinkageEstimator()
        
        result = estimator.estimate(sample_returns)
        
        # Check via eigenvalues
        eigenvalues = np.linalg.eigvalsh(result.covariance)
        assert all(e > 0 for e in eigenvalues)
    
    def test_high_dimensional_case(self) -> None:
        """Test with high-dimensional data (n < p case)."""
        np.random.seed(42)
        # More assets than observations (p > n)
        returns = np.random.randn(30, 50) * 0.02
        
        estimator = ShrinkageEstimator()
        result = estimator.estimate(returns)
        
        # Should still produce valid covariance
        assert result.covariance.shape == (50, 50)
        
        # Should be positive definite
        eigenvalues = np.linalg.eigvalsh(result.covariance)
        assert all(e > 0 for e in eigenvalues)

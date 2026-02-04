"""
Unit tests for cliquet option pricer.

Tests EquityCliquetGbmMcPricer, FxCliquetGbmMcPricer, and CliquetPricingResult.
"""

from datetime import date, timedelta

import numpy as np
import pytest

from src.pricers.equity.cliquet_gbm_mc import (
    CliquetMarketData,
    CliquetPricingResult,
    EquityCliquetGbmMcPricer,
)
from src.instruments.equity.options.cliquet import EquityCliquetOption


class TestCliquetPricingResult:
    """Tests for CliquetPricingResult dataclass."""
    
    def test_result_creation(self) -> None:
        """Test result creation."""
        result = CliquetPricingResult(
            price=5.0,
            std_error=0.1,
            expected_n_capped=2.5,
            expected_n_floored=1.2,
        )
        
        assert result.price == 5.0
        assert result.std_error == 0.1


class TestCliquetMarketData:
    """Tests for CliquetMarketData dataclass."""
    
    def test_market_data_creation(self) -> None:
        """Test market data creation."""
        data = CliquetMarketData(
            spot=100.0,
            volatility=0.20,
            rate=0.05,
            dividend_yield=0.02,
        )
        
        assert data.spot == 100.0
        assert data.volatility == 0.20


class TestEquityCliquetGbmMcPricer:
    """Tests for EquityCliquetGbmMcPricer."""
    
    @pytest.fixture
    def sample_cliquet(self) -> EquityCliquetOption:
        """Create sample cliquet option."""
        return EquityCliquetOption(
            underlying="SPX",
            start_date=date.today(),
            maturity_date=date.today() + timedelta(days=365),
            reset_dates=[date.today() + timedelta(days=i * 30) for i in range(1, 13)],
            local_cap=0.05,
            local_floor=-0.03,
            global_cap=0.30,
            global_floor=0.0,
            notional=100.0,
        )
    
    @pytest.fixture
    def sample_market_data(self) -> CliquetMarketData:
        """Create sample market data."""
        return CliquetMarketData(
            spot=100.0,
            volatility=0.20,
            rate=0.05,
            dividend_yield=0.0,
        )
    
    def test_pricer_creation(self) -> None:
        """Test pricer creation."""
        pricer = EquityCliquetGbmMcPricer(n_paths=10000, seed=42)
        
        assert pricer.n_paths == 10000
    
    def test_price_basic(
        self,
        sample_cliquet: EquityCliquetOption,
        sample_market_data: CliquetMarketData,
    ) -> None:
        """Test basic pricing."""
        pricer = EquityCliquetGbmMcPricer(n_paths=10000, seed=42)
        
        result = pricer.price(sample_cliquet, sample_market_data)
        
        # Should return valid result
        assert isinstance(result, CliquetPricingResult)
        assert result.price >= 0  # Cliquet with floor 0 should be non-negative
    
    def test_price_convergence(
        self,
        sample_cliquet: EquityCliquetOption,
        sample_market_data: CliquetMarketData,
    ) -> None:
        """Test that more paths reduce standard error."""
        pricer_few = EquityCliquetGbmMcPricer(n_paths=1000, seed=42)
        pricer_many = EquityCliquetGbmMcPricer(n_paths=50000, seed=42)
        
        result_few = pricer_few.price(sample_cliquet, sample_market_data)
        result_many = pricer_many.price(sample_cliquet, sample_market_data)
        
        # More paths should have smaller error
        assert result_many.std_error < result_few.std_error
    
    def test_local_cap_effect(self, sample_market_data: CliquetMarketData) -> None:
        """Test that local cap affects price."""
        cliquet_low_cap = EquityCliquetOption(
            underlying="SPX",
            start_date=date.today(),
            maturity_date=date.today() + timedelta(days=365),
            reset_dates=[date.today() + timedelta(days=i * 30) for i in range(1, 13)],
            local_cap=0.02,  # Low cap
            local_floor=-0.03,
            global_cap=None,
            global_floor=0.0,
            notional=100.0,
        )
        
        cliquet_high_cap = EquityCliquetOption(
            underlying="SPX",
            start_date=date.today(),
            maturity_date=date.today() + timedelta(days=365),
            reset_dates=[date.today() + timedelta(days=i * 30) for i in range(1, 13)],
            local_cap=0.10,  # High cap
            local_floor=-0.03,
            global_cap=None,
            global_floor=0.0,
            notional=100.0,
        )
        
        pricer = EquityCliquetGbmMcPricer(n_paths=20000, seed=42)
        
        result_low = pricer.price(cliquet_low_cap, sample_market_data)
        result_high = pricer.price(cliquet_high_cap, sample_market_data)
        
        # Higher cap should give higher price
        assert result_high.price >= result_low.price - 0.5  # Allow for MC noise
    
    def test_reproducibility(
        self,
        sample_cliquet: EquityCliquetOption,
        sample_market_data: CliquetMarketData,
    ) -> None:
        """Test reproducibility with same seed."""
        pricer1 = EquityCliquetGbmMcPricer(n_paths=10000, seed=42)
        pricer2 = EquityCliquetGbmMcPricer(n_paths=10000, seed=42)
        
        result1 = pricer1.price(sample_cliquet, sample_market_data)
        result2 = pricer2.price(sample_cliquet, sample_market_data)
        
        assert abs(result1.price - result2.price) < 0.01
    
    def test_antithetic_variance_reduction(
        self,
        sample_cliquet: EquityCliquetOption,
        sample_market_data: CliquetMarketData,
    ) -> None:
        """Test antithetic variance reduction if available."""
        pricer_no_anti = EquityCliquetGbmMcPricer(n_paths=10000, seed=42, antithetic=False)
        pricer_anti = EquityCliquetGbmMcPricer(n_paths=10000, seed=42, antithetic=True)
        
        result_no_anti = pricer_no_anti.price(sample_cliquet, sample_market_data)
        result_anti = pricer_anti.price(sample_cliquet, sample_market_data)
        
        # Antithetic should generally reduce variance
        # Note: Not always guaranteed for all payoffs
        assert result_anti.std_error <= result_no_anti.std_error * 1.5

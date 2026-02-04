"""
Unit tests for variance swap pricing module.

Tests VarianceSwap, VarianceSwapPricer, and VarianceSwapResult.
"""

from datetime import date

import numpy as np
import pytest

from src.volatility.trading.variance_swap import (
    VarianceSwap,
    VarianceSwapPricer,
    VarianceSwapResult,
    calculate_realized_variance,
)


class TestVarianceSwap:
    """Tests for VarianceSwap instrument."""
    
    def test_basic_creation(self) -> None:
        """Test basic swap creation."""
        swap = VarianceSwap(
            strike_var=0.04,
            maturity=0.5,
            notional=100_000,
        )
        
        assert swap.strike_var == 0.04
        assert swap.maturity == 0.5
        assert swap.notional == 100_000
    
    def test_strike_vol_property(self) -> None:
        """Test strike volatility property."""
        swap = VarianceSwap(strike_var=0.04, maturity=1.0)
        
        # sqrt(0.04) = 0.2
        assert abs(swap.strike_vol - 0.2) < 1e-10
    
    def test_vega_notional(self) -> None:
        """Test vega notional calculation."""
        swap = VarianceSwap(
            strike_var=0.04,  # 20% vol
            maturity=1.0,
            notional=100_000,
        )
        
        # vega_notional = notional * 2 * strike_vol = 100_000 * 2 * 0.2 = 40_000
        assert abs(swap.vega_notional - 40_000) < 1e-6
    
    def test_from_vol_constructor(self) -> None:
        """Test creating swap from vol parameters."""
        swap = VarianceSwap.from_vol(
            strike_vol=0.20,
            maturity=1.0,
            vega_notional=40_000,
        )
        
        assert abs(swap.strike_vol - 0.20) < 1e-10
        assert swap.maturity == 1.0
    
    def test_observation_frequency_default(self) -> None:
        """Test default observation frequency."""
        swap = VarianceSwap(strike_var=0.04, maturity=1.0)
        
        assert swap.observation_frequency == "daily"


class TestVarianceSwapResult:
    """Tests for VarianceSwapResult."""
    
    def test_result_creation(self) -> None:
        """Test result creation."""
        result = VarianceSwapResult(
            fair_variance=0.042,
            strike_variance=0.04,
            mtm=5000.0,
            notional=100_000,
            maturity=0.5,
        )
        
        assert result.fair_variance == 0.042
        assert result.mtm == 5000.0
    
    def test_fair_vol_property(self) -> None:
        """Test fair volatility property."""
        result = VarianceSwapResult(
            fair_variance=0.04,
            strike_variance=0.04,
            mtm=0.0,
            notional=100_000,
            maturity=1.0,
        )
        
        assert abs(result.fair_vol - 0.20) < 1e-10


class TestVarianceSwapPricer:
    """Tests for VarianceSwapPricer."""
    
    def test_pricer_creation(self) -> None:
        """Test pricer creation."""
        pricer = VarianceSwapPricer(n_integration_points=100)
        
        assert pricer.n_integration_points == 100
    
    def test_price_with_option_chain(self) -> None:
        """Test pricing with option chain."""
        pricer = VarianceSwapPricer()
        swap = VarianceSwap(strike_var=0.04, maturity=1.0, notional=100_000)
        
        # Create synthetic option chain
        spot = 100.0
        forward = 102.0  # Small forward premium
        
        # Strike range around forward
        strikes = np.linspace(80, 120, 21)
        
        # Generate synthetic option prices (simplified)
        # In practice, these would come from market quotes
        atm_vol = 0.20
        call_prices = np.maximum(0, forward - strikes) * 0.5  # Simplified intrinsic-ish
        put_prices = np.maximum(0, strikes - forward) * 0.5
        
        # Add some time value
        call_prices = call_prices + atm_vol * np.sqrt(swap.maturity) * 5
        put_prices = put_prices + atm_vol * np.sqrt(swap.maturity) * 5
        
        result = pricer.price(
            swap=swap,
            spot=spot,
            forward=forward,
            option_strikes=strikes,
            option_prices_call=call_prices,
            option_prices_put=put_prices,
            rate=0.05,
        )
        
        # Fair variance should be positive
        assert result.fair_variance > 0
        
        # Result should have all required fields
        assert result.notional == swap.notional
        assert result.maturity == swap.maturity
    
    def test_price_at_the_money_strike(self) -> None:
        """Test that ATM strike gives reasonable results."""
        pricer = VarianceSwapPricer()
        
        # ATM variance swap
        swap = VarianceSwap(strike_var=0.0, maturity=1.0, notional=100_000)
        
        spot = 100.0
        forward = 100.0
        strikes = np.linspace(70, 130, 31)
        
        # Simplified BS-like prices with 20% vol
        vol = 0.20
        from scipy.stats import norm
        
        d1 = (np.log(forward / strikes) + 0.5 * vol**2) / vol
        d2 = d1 - vol
        
        call_prices = forward * norm.cdf(d1) - strikes * norm.cdf(d2)
        put_prices = strikes * norm.cdf(-d2) - forward * norm.cdf(-d1)
        
        call_prices = np.maximum(call_prices, 0.01)
        put_prices = np.maximum(put_prices, 0.01)
        
        result = pricer.price(
            swap=swap,
            spot=spot,
            forward=forward,
            option_strikes=strikes,
            option_prices_call=call_prices,
            option_prices_put=put_prices,
            rate=0.0,
        )
        
        # Fair variance should be close to 0.04 (20%^2)
        # Allow for approximation errors
        assert 0.01 < result.fair_variance < 0.10
    
    def test_mtm_positive_when_fair_above_strike(self) -> None:
        """Test MTM is positive when fair variance > strike."""
        pricer = VarianceSwapPricer()
        
        # Low strike variance
        swap = VarianceSwap(strike_var=0.01, maturity=1.0, notional=100_000)
        
        spot = 100.0
        forward = 100.0
        strikes = np.linspace(70, 130, 21)
        
        # Use higher vol for options
        vol = 0.25
        call_prices = np.maximum(forward - strikes, 0) + vol * 10
        put_prices = np.maximum(strikes - forward, 0) + vol * 10
        
        result = pricer.price(
            swap=swap,
            spot=spot,
            forward=forward,
            option_strikes=strikes,
            option_prices_call=call_prices,
            option_prices_put=put_prices,
            rate=0.0,
        )
        
        # If fair var > strike var, MTM should be positive (long position profits)
        if result.fair_variance > swap.strike_var:
            assert result.mtm >= 0


class TestCalculateRealizedVariance:
    """Tests for realized variance calculation."""
    
    def test_zero_variance_flat_prices(self) -> None:
        """Test that flat prices give zero variance."""
        prices = np.ones(100) * 100.0
        
        realized_var = calculate_realized_variance(prices, annualization=252)
        
        assert abs(realized_var) < 1e-10
    
    def test_positive_variance(self) -> None:
        """Test that varying prices give positive variance."""
        np.random.seed(42)
        returns = np.random.randn(252) * 0.01
        prices = 100 * np.cumprod(1 + returns)
        
        realized_var = calculate_realized_variance(prices)
        
        assert realized_var > 0
    
    def test_annualization(self) -> None:
        """Test that annualization affects result."""
        np.random.seed(42)
        returns = np.random.randn(252) * 0.01
        prices = 100 * np.cumprod(1 + returns)
        
        var_daily = calculate_realized_variance(prices, annualization=1)
        var_annual = calculate_realized_variance(prices, annualization=252)
        
        # Annual should be approximately 252x daily
        assert abs(var_annual / var_daily - 252) < 10
    
    def test_known_volatility(self) -> None:
        """Test with synthetic data of known volatility."""
        np.random.seed(42)
        true_vol = 0.20
        daily_vol = true_vol / np.sqrt(252)
        
        # Generate 1000 days of returns
        returns = np.random.randn(1000) * daily_vol
        prices = 100 * np.cumprod(1 + returns)
        
        realized_var = calculate_realized_variance(prices, annualization=252)
        realized_vol = np.sqrt(realized_var)
        
        # Should be close to true vol (with some sampling error)
        assert abs(realized_vol - true_vol) < 0.05

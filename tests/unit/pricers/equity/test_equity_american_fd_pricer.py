"""
Unit tests for equity American vanilla FD pricer.

Tests early exercise premium, put-call asymmetry, and convergence.

Author: QuantStrata Team
"""

import math
import pytest
import numpy as np

from src.instruments.equity.options.vanilla import (
    EuropeanEquityVanillaOption,
    AmericanEquityVanillaOption,
)
from src.pricers.equity.american_fd import EquityAmericanVanillaFdPricer
from src.pricers.equity.european_fd import EquityEuropeanVanillaFdPricer
from src.pricers.equity.european_bsm import EquityEuropeanVanillaBsmPricer
from src.marketdata.core.ids import MarketId
from src.marketdata.core.interfaces import Quote
from src.marketdata.core.market import Market
from src.marketdata.curves.term_structure import FlatZeroRateCurve
from src.marketdata.surfaces.vol_surface import FlatVolSurface


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def spot_id() -> MarketId:
    return MarketId(asset_class="EQ", mkt_type="SPOT", name="AAPL")


@pytest.fixture
def vol_id() -> MarketId:
    return MarketId(asset_class="EQ", mkt_type="VOL", name="AAPL")


@pytest.fixture
def curve_id() -> MarketId:
    return MarketId(asset_class="IR", mkt_type="CURVE", name="USD_OIS")


@pytest.fixture
def standard_market(spot_id, vol_id, curve_id) -> Market:
    """Standard market: S=100, r=5%, vol=20%."""
    return Market(
        asof="2026-01-28",
        quotes={spot_id: Quote(value=100.0)},
        curves={curve_id: FlatZeroRateCurve(continuously_compounded_rate=0.05)},
        vols={vol_id: FlatVolSurface(sigma=0.20)},
    )


# =============================================================================
# Basic Pricing Tests
# =============================================================================

class TestBasicPricing:
    """Tests for basic American pricing."""
    
    def test_call_positive_value(self, standard_market, spot_id, vol_id, curve_id):
        """American call should have positive value."""
        pricer = EquityAmericanVanillaFdPricer(n_space=201, n_time_steps=100)
        opt = AmericanEquityVanillaOption(
            ticker="AAPL", option_type="call", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        pv = pricer.price(opt, standard_market)
        assert pv > 0.0
    
    def test_put_positive_value(self, standard_market, spot_id, vol_id, curve_id):
        """American put should have positive value."""
        pricer = EquityAmericanVanillaFdPricer(n_space=201, n_time_steps=100)
        opt = AmericanEquityVanillaOption(
            ticker="AAPL", option_type="put", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        pv = pricer.price(opt, standard_market)
        assert pv > 0.0
    
    def test_notional_scaling(self, standard_market, spot_id, vol_id, curve_id):
        """Price should scale linearly with notional."""
        pricer = EquityAmericanVanillaFdPricer(n_space=201, n_time_steps=100)
        
        opt_1 = AmericanEquityVanillaOption(
            ticker="AAPL", option_type="put", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        opt_100 = AmericanEquityVanillaOption(
            ticker="AAPL", option_type="put", strike=100.0, expiry=1.0,
            notional=100, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        
        pv_1 = pricer.price(opt_1, standard_market)
        pv_100 = pricer.price(opt_100, standard_market)
        
        assert abs(pv_100 - 100 * pv_1) < 1e-8


# =============================================================================
# Early Exercise Premium Tests
# =============================================================================

class TestEarlyExercisePremium:
    """Tests for American >= European (early exercise premium)."""
    
    def test_american_put_ge_european(self, standard_market, spot_id, vol_id, curve_id):
        """American put should be >= European put."""
        am_pricer = EquityAmericanVanillaFdPricer(n_space=201, n_time_steps=100)
        eu_pricer = EquityEuropeanVanillaFdPricer(n_space=201, n_time_steps=100)
        
        am_opt = AmericanEquityVanillaOption(
            ticker="AAPL", option_type="put", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        eu_opt = EuropeanEquityVanillaOption(
            ticker="AAPL", option_type="put", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        
        am_pv = am_pricer.price(am_opt, standard_market)
        eu_pv = eu_pricer.price(eu_opt, standard_market)
        
        assert am_pv >= eu_pv - 1e-8  # Allow small numerical tolerance
    
    def test_american_call_no_dividend_equals_european(self, standard_market, spot_id, vol_id, curve_id):
        """American call on non-dividend stock should equal European."""
        am_pricer = EquityAmericanVanillaFdPricer(n_space=301, n_time_steps=150)
        eu_pricer = EquityEuropeanVanillaFdPricer(n_space=301, n_time_steps=150)
        
        am_opt = AmericanEquityVanillaOption(
            ticker="AAPL", option_type="call", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        eu_opt = EuropeanEquityVanillaOption(
            ticker="AAPL", option_type="call", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        
        am_pv = am_pricer.price(am_opt, standard_market)
        eu_pv = eu_pricer.price(eu_opt, standard_market)
        
        # Should be very close (never optimal to exercise early)
        rel_diff = abs(am_pv - eu_pv) / eu_pv
        assert rel_diff < 0.01  # Within 1%
    
    def test_deep_itm_put_early_exercise_premium(self, spot_id, vol_id, curve_id):
        """Deep ITM put should have significant early exercise premium."""
        # Market with higher rate makes early exercise more valuable
        market = Market(
            asof="2026-01-28",
            quotes={spot_id: Quote(value=50.0)},  # Deep ITM for K=100
            curves={curve_id: FlatZeroRateCurve(continuously_compounded_rate=0.10)},
            vols={vol_id: FlatVolSurface(sigma=0.20)},
        )
        
        am_pricer = EquityAmericanVanillaFdPricer(n_space=201, n_time_steps=100)
        eu_pricer = EquityEuropeanVanillaFdPricer(n_space=201, n_time_steps=100)
        
        am_opt = AmericanEquityVanillaOption(
            ticker="AAPL", option_type="put", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        eu_opt = EuropeanEquityVanillaOption(
            ticker="AAPL", option_type="put", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        
        am_pv = am_pricer.price(am_opt, market)
        eu_pv = eu_pricer.price(eu_opt, market)
        
        # American should be noticeably higher
        assert am_pv > eu_pv + 0.5  # At least 0.5 premium
    
    def test_american_call_with_dividend(self, spot_id, vol_id, curve_id):
        """American call with high dividend may have early exercise premium."""
        # High dividend yield makes early call exercise potentially valuable
        market = Market(
            asof="2026-01-28",
            quotes={spot_id: Quote(value=100.0)},
            curves={curve_id: FlatZeroRateCurve(continuously_compounded_rate=0.05)},
            vols={vol_id: FlatVolSurface(sigma=0.20)},
        )
        
        am_pricer = EquityAmericanVanillaFdPricer(n_space=201, n_time_steps=100)
        eu_pricer = EquityEuropeanVanillaFdPricer(n_space=201, n_time_steps=100)
        
        # High dividend yield
        q = 0.08
        
        am_opt = AmericanEquityVanillaOption(
            ticker="AAPL", option_type="call", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=q, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        eu_opt = EuropeanEquityVanillaOption(
            ticker="AAPL", option_type="call", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=q, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        
        am_pv = am_pricer.price(am_opt, market)
        eu_pv = eu_pricer.price(eu_opt, market)
        
        # American should be >= European
        assert am_pv >= eu_pv - 1e-8


# =============================================================================
# Intrinsic Value Tests
# =============================================================================

class TestIntrinsicValue:
    """Tests that American value >= intrinsic."""
    
    def test_value_ge_intrinsic_put(self, standard_market, spot_id, vol_id, curve_id):
        """American put should be >= intrinsic value."""
        pricer = EquityAmericanVanillaFdPricer(n_space=201, n_time_steps=100)
        
        # Deep ITM put
        opt = AmericanEquityVanillaOption(
            ticker="AAPL", option_type="put", strike=120.0, expiry=1.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        pv = pricer.price(opt, standard_market)
        
        intrinsic = max(120.0 - 100.0, 0.0)  # K - S
        assert pv >= intrinsic - 1e-8
    
    def test_value_ge_intrinsic_call(self, standard_market, spot_id, vol_id, curve_id):
        """American call should be >= intrinsic value."""
        pricer = EquityAmericanVanillaFdPricer(n_space=201, n_time_steps=100)
        
        # Deep ITM call
        opt = AmericanEquityVanillaOption(
            ticker="AAPL", option_type="call", strike=80.0, expiry=1.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        pv = pricer.price(opt, standard_market)
        
        intrinsic = max(100.0 - 80.0, 0.0)  # S - K
        assert pv >= intrinsic - 1e-8


# =============================================================================
# Greeks Tests
# =============================================================================

class TestGreeks:
    """Tests for American Greeks."""
    
    def test_put_delta_negative(self, standard_market, spot_id, vol_id, curve_id):
        """American put delta should be negative."""
        pricer = EquityAmericanVanillaFdPricer(n_space=201, n_time_steps=100)
        opt = AmericanEquityVanillaOption(
            ticker="AAPL", option_type="put", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        greeks = pricer.greeks(opt, standard_market)
        assert greeks["delta"] < 0.0
    
    def test_call_delta_positive(self, standard_market, spot_id, vol_id, curve_id):
        """American call delta should be positive."""
        pricer = EquityAmericanVanillaFdPricer(n_space=201, n_time_steps=100)
        opt = AmericanEquityVanillaOption(
            ticker="AAPL", option_type="call", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        greeks = pricer.greeks(opt, standard_market)
        assert greeks["delta"] > 0.0
    
    def test_gamma_positive(self, standard_market, spot_id, vol_id, curve_id):
        """American gamma should be positive."""
        pricer = EquityAmericanVanillaFdPricer(n_space=201, n_time_steps=100)
        opt = AmericanEquityVanillaOption(
            ticker="AAPL", option_type="put", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        greeks = pricer.greeks(opt, standard_market)
        assert greeks["gamma"] > 0.0


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_zero_expiry(self, standard_market, spot_id, vol_id, curve_id):
        """At expiry, return intrinsic value."""
        pricer = EquityAmericanVanillaFdPricer(n_space=201, n_time_steps=100)
        
        opt_call = AmericanEquityVanillaOption(
            ticker="AAPL", option_type="call", strike=90.0, expiry=0.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        pv_call = pricer.price(opt_call, standard_market)
        assert abs(pv_call - 10.0) < 1e-10  # S - K = 100 - 90
        
        opt_put = AmericanEquityVanillaOption(
            ticker="AAPL", option_type="put", strike=110.0, expiry=0.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        pv_put = pricer.price(opt_put, standard_market)
        assert abs(pv_put - 10.0) < 1e-10  # K - S = 110 - 100
    
    def test_diagnostics_returns_grids(self, standard_market, spot_id, vol_id, curve_id):
        """diagnostics() should return grid information."""
        pricer = EquityAmericanVanillaFdPricer(n_space=51, n_time_steps=25)
        opt = AmericanEquityVanillaOption(
            ticker="AAPL", option_type="put", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        
        diag = pricer.diagnostics(opt, standard_market, store_surface=False)
        
        assert diag.spot_grid is not None
        assert len(diag.spot_grid) == 51

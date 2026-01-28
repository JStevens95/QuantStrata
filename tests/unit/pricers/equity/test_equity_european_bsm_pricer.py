"""
Unit tests for equity European vanilla BSM pricer.

Tests pricing accuracy, Greeks, and edge cases.

Author: QuantStrata Team
"""

import math
import pytest
import numpy as np

from src.instruments.equity.options.vanilla import EuropeanEquityVanillaOption
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


@pytest.fixture
def pricer() -> EquityEuropeanVanillaBsmPricer:
    return EquityEuropeanVanillaBsmPricer()


# =============================================================================
# Basic Pricing Tests
# =============================================================================

class TestBasicPricing:
    """Tests for basic pricing functionality."""
    
    def test_atm_call_positive_value(self, pricer, standard_market, spot_id, vol_id, curve_id):
        """ATM call should have positive value."""
        opt = EuropeanEquityVanillaOption(
            ticker="AAPL", option_type="call", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        pv = pricer.price(opt, standard_market)
        assert pv > 0.0
    
    def test_atm_put_positive_value(self, pricer, standard_market, spot_id, vol_id, curve_id):
        """ATM put should have positive value."""
        opt = EuropeanEquityVanillaOption(
            ticker="AAPL", option_type="put", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        pv = pricer.price(opt, standard_market)
        assert pv > 0.0
    
    def test_deep_itm_call(self, pricer, standard_market, spot_id, vol_id, curve_id):
        """Deep ITM call should be close to intrinsic value."""
        opt = EuropeanEquityVanillaOption(
            ticker="AAPL", option_type="call", strike=50.0, expiry=1.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        pv = pricer.price(opt, standard_market)
        # Should be greater than discounted intrinsic
        intrinsic = 100.0 - 50.0
        assert pv >= intrinsic * math.exp(-0.05)
    
    def test_deep_otm_call(self, pricer, standard_market, spot_id, vol_id, curve_id):
        """Deep OTM call should be small."""
        opt = EuropeanEquityVanillaOption(
            ticker="AAPL", option_type="call", strike=200.0, expiry=1.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        pv = pricer.price(opt, standard_market)
        assert pv < 1.0  # Should be very small
        assert pv > 0.0  # But positive
    
    def test_notional_scaling(self, pricer, standard_market, spot_id, vol_id, curve_id):
        """Price should scale linearly with notional."""
        opt_1 = EuropeanEquityVanillaOption(
            ticker="AAPL", option_type="call", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        opt_100 = EuropeanEquityVanillaOption(
            ticker="AAPL", option_type="call", strike=100.0, expiry=1.0,
            notional=100, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        pv_1 = pricer.price(opt_1, standard_market)
        pv_100 = pricer.price(opt_100, standard_market)
        assert abs(pv_100 - 100 * pv_1) < 1e-10


# =============================================================================
# Put-Call Parity Tests
# =============================================================================

class TestPutCallParity:
    """Tests for put-call parity."""
    
    def test_parity_no_dividend(self, pricer, standard_market, spot_id, vol_id, curve_id):
        """C - P = S - K * exp(-rT) when q=0."""
        S = 100.0
        K = 100.0
        T = 1.0
        r = 0.05
        
        call = EuropeanEquityVanillaOption(
            ticker="AAPL", option_type="call", strike=K, expiry=T,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        put = EuropeanEquityVanillaOption(
            ticker="AAPL", option_type="put", strike=K, expiry=T,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        
        c = pricer.price(call, standard_market)
        p = pricer.price(put, standard_market)
        
        # C - P = S - K * exp(-rT)
        expected = S - K * math.exp(-r * T)
        assert abs((c - p) - expected) < 1e-10
    
    def test_parity_with_dividend(self, spot_id, vol_id, curve_id):
        """C - P = S*exp(-qT) - K*exp(-rT) with dividend yield."""
        S = 100.0
        K = 100.0
        T = 1.0
        r = 0.05
        q = 0.02
        
        market = Market(
            asof="2026-01-28",
            quotes={spot_id: Quote(value=S)},
            curves={curve_id: FlatZeroRateCurve(continuously_compounded_rate=r)},
            vols={vol_id: FlatVolSurface(sigma=0.20)},
        )
        
        pricer = EquityEuropeanVanillaBsmPricer()
        
        call = EuropeanEquityVanillaOption(
            ticker="AAPL", option_type="call", strike=K, expiry=T,
            notional=1, dividend_yield=q, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        put = EuropeanEquityVanillaOption(
            ticker="AAPL", option_type="put", strike=K, expiry=T,
            notional=1, dividend_yield=q, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        
        c = pricer.price(call, market)
        p = pricer.price(put, market)
        
        # C - P = S*exp(-qT) - K*exp(-rT)
        expected = S * math.exp(-q * T) - K * math.exp(-r * T)
        assert abs((c - p) - expected) < 1e-10


# =============================================================================
# Greeks Tests
# =============================================================================

class TestGreeks:
    """Tests for Greeks computation."""
    
    def test_call_delta_positive(self, pricer, standard_market, spot_id, vol_id, curve_id):
        """Call delta should be positive and between 0 and 1."""
        opt = EuropeanEquityVanillaOption(
            ticker="AAPL", option_type="call", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        greeks = pricer.greeks(opt, standard_market)
        assert 0.0 < greeks["delta"] < 1.0
    
    def test_put_delta_negative(self, pricer, standard_market, spot_id, vol_id, curve_id):
        """Put delta should be negative and between -1 and 0."""
        opt = EuropeanEquityVanillaOption(
            ticker="AAPL", option_type="put", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        greeks = pricer.greeks(opt, standard_market)
        assert -1.0 < greeks["delta"] < 0.0
    
    def test_gamma_positive(self, pricer, standard_market, spot_id, vol_id, curve_id):
        """Gamma should be positive for both calls and puts."""
        for opt_type in ["call", "put"]:
            opt = EuropeanEquityVanillaOption(
                ticker="AAPL", option_type=opt_type, strike=100.0, expiry=1.0,
                notional=1, dividend_yield=0.0, spot_id=spot_id,
                vol_id=vol_id, curve_id=curve_id,
            )
            greeks = pricer.greeks(opt, standard_market)
            assert greeks["gamma"] > 0.0
    
    def test_vega_positive(self, pricer, standard_market, spot_id, vol_id, curve_id):
        """Vega should be positive for both calls and puts."""
        for opt_type in ["call", "put"]:
            opt = EuropeanEquityVanillaOption(
                ticker="AAPL", option_type=opt_type, strike=100.0, expiry=1.0,
                notional=1, dividend_yield=0.0, spot_id=spot_id,
                vol_id=vol_id, curve_id=curve_id,
            )
            greeks = pricer.greeks(opt, standard_market)
            assert greeks["vega"] > 0.0
    
    def test_greeks_notional_scaling(self, pricer, standard_market, spot_id, vol_id, curve_id):
        """Greeks should scale with notional."""
        opt_1 = EuropeanEquityVanillaOption(
            ticker="AAPL", option_type="call", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        opt_100 = EuropeanEquityVanillaOption(
            ticker="AAPL", option_type="call", strike=100.0, expiry=1.0,
            notional=100, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        greeks_1 = pricer.greeks(opt_1, standard_market)
        greeks_100 = pricer.greeks(opt_100, standard_market)
        
        for key in ["delta", "gamma", "vega", "rho"]:
            assert abs(greeks_100[key] - 100 * greeks_1[key]) < 1e-10


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_zero_expiry(self, pricer, standard_market, spot_id, vol_id, curve_id):
        """At expiry, price should equal intrinsic value."""
        opt_call = EuropeanEquityVanillaOption(
            ticker="AAPL", option_type="call", strike=90.0, expiry=0.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        pv_call = pricer.price(opt_call, standard_market)
        assert abs(pv_call - 10.0) < 1e-10  # S - K = 100 - 90 = 10
        
        opt_put = EuropeanEquityVanillaOption(
            ticker="AAPL", option_type="put", strike=110.0, expiry=0.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        pv_put = pricer.price(opt_put, standard_market)
        assert abs(pv_put - 10.0) < 1e-10  # K - S = 110 - 100 = 10
    
    def test_zero_expiry_greeks(self, pricer, standard_market, spot_id, vol_id, curve_id):
        """At expiry, greeks should be zero (stable policy)."""
        opt = EuropeanEquityVanillaOption(
            ticker="AAPL", option_type="call", strike=100.0, expiry=0.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        greeks = pricer.greeks(opt, standard_market)
        for key in ["delta", "gamma", "vega", "rho", "theta"]:
            assert greeks[key] == 0.0
    
    def test_high_dividend_reduces_call_value(self, spot_id, vol_id, curve_id):
        """Higher dividend yield should reduce call value."""
        pricer = EquityEuropeanVanillaBsmPricer()
        
        market = Market(
            asof="2026-01-28",
            quotes={spot_id: Quote(value=100.0)},
            curves={curve_id: FlatZeroRateCurve(continuously_compounded_rate=0.05)},
            vols={vol_id: FlatVolSurface(sigma=0.20)},
        )
        
        call_q0 = EuropeanEquityVanillaOption(
            ticker="AAPL", option_type="call", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.0, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        call_q5 = EuropeanEquityVanillaOption(
            ticker="AAPL", option_type="call", strike=100.0, expiry=1.0,
            notional=1, dividend_yield=0.05, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        
        pv_q0 = pricer.price(call_q0, market)
        pv_q5 = pricer.price(call_q5, market)
        
        assert pv_q5 < pv_q0  # Higher dividend reduces call value

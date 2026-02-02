# tests/unit/pricers/test_spread_bch_pricer.py
"""
Unit tests for FX and Equity Spread Option Bachelier Pricers.

Tests cover:
- FX and Equity spread option pricing using Bachelier model
- Call vs put options
- Put-call parity
- Greeks computation
- Finite difference validation
- Edge cases (ATM, zero vol, negative strikes)

Author: QuantStrata Team
"""
from __future__ import annotations

import math
import pytest

from src.instruments.fx.options.spread import (
    FxSpreadEuropeanOption,
    FxSpreadEuropeanOptionSimple,
)
from src.instruments.equity.options.spread import (
    EquitySpreadEuropeanOption,
    EquitySpreadEuropeanOptionSimple,
)
from src.pricers.fx.european_bch import (
    FxSpreadEuropeanOptionBchPricer,
    FxSpreadEuropeanOptionBchPricerSimple,
)
from src.pricers.equity.european_bch import (
    EquitySpreadEuropeanOptionBchPricer,
    EquitySpreadEuropeanOptionBchPricerSimple,
)
from src.marketdata.core.ids import MarketId
from src.marketdata.core.market import Market
from src.marketdata.curves.term_structure import FlatZeroRateCurve
from src.marketdata.surfaces.vol_surface import FlatVolSurface


# =============================================================================
# FX SPREAD OPTION FIXTURES
# =============================================================================


@pytest.fixture
def fx_spread_params():
    """Base parameters for an FX spread option."""
    return {
        "notional": 1_000_000,
        "strike": 0.10,           # 10% spread strike
        "expiry": 0.5,            # 6 months
        "forward_spread": 0.12,   # Forward spread (S1 - S2)
        "vol": 0.05,              # 5% normal vol
        "discount_factor": 0.975,
        "option_type": "call",
    }


@pytest.fixture
def fx_call_spread(fx_spread_params):
    """Create an FX spread call option."""
    return FxSpreadEuropeanOptionSimple(**fx_spread_params)


@pytest.fixture
def fx_put_spread(fx_spread_params):
    """Create an FX spread put option."""
    params = {**fx_spread_params, "option_type": "put"}
    return FxSpreadEuropeanOptionSimple(**params)


# =============================================================================
# EQUITY SPREAD OPTION FIXTURES
# =============================================================================


@pytest.fixture
def eq_spread_params():
    """Base parameters for an Equity spread option."""
    return {
        "notional": 100_000,
        "strike": 5.0,            # $5 spread strike
        "expiry": 0.25,           # 3 months
        "forward_spread": 7.5,    # Forward spread (S1 - S2)
        "vol": 10.0,              # $10 normal vol
        "discount_factor": 0.99,
        "option_type": "call",
    }


@pytest.fixture
def eq_call_spread(eq_spread_params):
    """Create an Equity spread call option."""
    return EquitySpreadEuropeanOptionSimple(**eq_spread_params)


@pytest.fixture
def eq_put_spread(eq_spread_params):
    """Create an Equity spread put option."""
    params = {**eq_spread_params, "option_type": "put"}
    return EquitySpreadEuropeanOptionSimple(**params)


# =============================================================================
# FX SPREAD OPTION VALIDATION TESTS
# =============================================================================


class TestFxSpreadOptionValidation:
    """Tests for FX spread option instrument validation."""
    
    def test_valid_fx_spread(self, fx_spread_params):
        """Valid FX spread option should be created without errors."""
        spread = FxSpreadEuropeanOptionSimple(**fx_spread_params)
        assert spread.notional == fx_spread_params["notional"]
        assert spread.strike == fx_spread_params["strike"]
    
    def test_zero_notional_raises(self, fx_spread_params):
        """Zero notional should raise."""
        fx_spread_params["notional"] = 0.0
        with pytest.raises(ValueError, match="notional must be non-zero"):
            FxSpreadEuropeanOptionSimple(**fx_spread_params)
    
    def test_negative_expiry_raises(self, fx_spread_params):
        """Negative expiry should raise."""
        fx_spread_params["expiry"] = -0.1
        with pytest.raises(ValueError, match="expiry must be >= 0"):
            FxSpreadEuropeanOptionSimple(**fx_spread_params)
    
    def test_is_itm_call(self, fx_call_spread):
        """Call is ITM when forward_spread > strike."""
        assert fx_call_spread.is_in_the_money  # 0.12 > 0.10
    
    def test_is_itm_put(self, fx_spread_params):
        """Put is ITM when forward_spread < strike."""
        params = {**fx_spread_params, "option_type": "put", "forward_spread": 0.08}
        put = FxSpreadEuropeanOptionSimple(**params)
        assert put.is_in_the_money  # 0.08 < 0.10


# =============================================================================
# FX SPREAD OPTION PRICING TESTS
# =============================================================================


class TestFxSpreadOptionPricing:
    """Tests for FX spread option pricing."""
    
    def test_call_price_positive(self, fx_call_spread):
        """Call spread option price should be positive."""
        pricer = FxSpreadEuropeanOptionBchPricerSimple()
        pv = pricer.price(fx_call_spread)
        assert pv > 0
    
    def test_put_price_positive(self, fx_put_spread):
        """Put spread option price should be positive."""
        pricer = FxSpreadEuropeanOptionBchPricerSimple()
        pv = pricer.price(fx_put_spread)
        assert pv > 0
    
    def test_itm_higher_than_otm(self, fx_spread_params):
        """ITM option should be more valuable than OTM."""
        pricer = FxSpreadEuropeanOptionBchPricerSimple()
        
        # ITM call: forward_spread > strike
        itm_params = {**fx_spread_params, "forward_spread": 0.20, "strike": 0.10}
        itm = FxSpreadEuropeanOptionSimple(**itm_params)
        
        # OTM call: forward_spread < strike
        otm_params = {**fx_spread_params, "forward_spread": 0.05, "strike": 0.10}
        otm = FxSpreadEuropeanOptionSimple(**otm_params)
        
        assert pricer.price(itm) > pricer.price(otm)
    
    def test_price_scales_with_notional(self, fx_spread_params):
        """Price should scale linearly with notional."""
        pricer = FxSpreadEuropeanOptionBchPricerSimple()
        
        sp1 = FxSpreadEuropeanOptionSimple(**fx_spread_params)
        
        params2 = {**fx_spread_params, "notional": 2_000_000}
        sp2 = FxSpreadEuropeanOptionSimple(**params2)
        
        pv1 = pricer.price(sp1)
        pv2 = pricer.price(sp2)
        
        assert abs(pv2 - 2 * pv1) < 1e-6


# =============================================================================
# EQUITY SPREAD OPTION PRICING TESTS
# =============================================================================


class TestEquitySpreadOptionPricing:
    """Tests for Equity spread option pricing."""
    
    def test_call_price_positive(self, eq_call_spread):
        """Call spread option price should be positive."""
        pricer = EquitySpreadEuropeanOptionBchPricerSimple()
        pv = pricer.price(eq_call_spread)
        assert pv > 0
    
    def test_put_price_positive(self, eq_put_spread):
        """Put spread option price should be positive."""
        pricer = EquitySpreadEuropeanOptionBchPricerSimple()
        pv = pricer.price(eq_put_spread)
        assert pv > 0
    
    def test_price_scales_with_notional(self, eq_spread_params):
        """Price should scale linearly with notional."""
        pricer = EquitySpreadEuropeanOptionBchPricerSimple()
        
        sp1 = EquitySpreadEuropeanOptionSimple(**eq_spread_params)
        
        params2 = {**eq_spread_params, "notional": 200_000}
        sp2 = EquitySpreadEuropeanOptionSimple(**params2)
        
        pv1 = pricer.price(sp1)
        pv2 = pricer.price(sp2)
        
        assert abs(pv2 - 2 * pv1) < 1e-6


# =============================================================================
# PUT-CALL PARITY TESTS
# =============================================================================


class TestPutCallParity:
    """Tests for put-call parity."""
    
    def test_fx_put_call_parity(self, fx_spread_params):
        """
        Call - Put = DF × (F - K)
        
        For spread options under Bachelier.
        """
        call = FxSpreadEuropeanOptionSimple(**fx_spread_params)
        put_params = {**fx_spread_params, "option_type": "put"}
        put = FxSpreadEuropeanOptionSimple(**put_params)
        
        pricer = FxSpreadEuropeanOptionBchPricerSimple()
        
        call_pv = pricer.price(call)
        put_pv = pricer.price(put)
        
        # Expected: N × DF × (F - K)
        N = fx_spread_params["notional"]
        df = fx_spread_params["discount_factor"]
        F = fx_spread_params["forward_spread"]
        K = fx_spread_params["strike"]
        
        expected_diff = N * df * (F - K)
        actual_diff = call_pv - put_pv
        
        assert abs(actual_diff - expected_diff) < 1e-6, \
            f"Parity failed: {actual_diff} != {expected_diff}"
    
    def test_eq_put_call_parity(self, eq_spread_params):
        """Equity spread put-call parity."""
        call = EquitySpreadEuropeanOptionSimple(**eq_spread_params)
        put_params = {**eq_spread_params, "option_type": "put"}
        put = EquitySpreadEuropeanOptionSimple(**put_params)
        
        pricer = EquitySpreadEuropeanOptionBchPricerSimple()
        
        call_pv = pricer.price(call)
        put_pv = pricer.price(put)
        
        N = eq_spread_params["notional"]
        df = eq_spread_params["discount_factor"]
        F = eq_spread_params["forward_spread"]
        K = eq_spread_params["strike"]
        
        expected_diff = N * df * (F - K)
        actual_diff = call_pv - put_pv
        
        assert abs(actual_diff - expected_diff) < 1e-6


# =============================================================================
# GREEKS TESTS
# =============================================================================


class TestSpreadOptionGreeks:
    """Tests for spread option Greeks."""
    
    def test_fx_greeks_exist(self, fx_call_spread):
        """Greeks should be computed."""
        pricer = FxSpreadEuropeanOptionBchPricerSimple()
        greeks = pricer.greeks(fx_call_spread)
        
        assert "delta" in greeks
        assert "gamma" in greeks
        assert "vega" in greeks
        assert "theta" in greeks
        assert "rho" in greeks
    
    def test_fx_call_delta_positive(self, fx_call_spread):
        """Call delta should be positive."""
        pricer = FxSpreadEuropeanOptionBchPricerSimple()
        greeks = pricer.greeks(fx_call_spread)
        assert greeks["delta"] > 0
    
    def test_fx_put_delta_negative(self, fx_put_spread):
        """Put delta should be negative."""
        pricer = FxSpreadEuropeanOptionBchPricerSimple()
        greeks = pricer.greeks(fx_put_spread)
        assert greeks["delta"] < 0
    
    def test_gamma_positive(self, fx_call_spread):
        """Gamma should be positive."""
        pricer = FxSpreadEuropeanOptionBchPricerSimple()
        greeks = pricer.greeks(fx_call_spread)
        assert greeks["gamma"] > 0
    
    def test_vega_positive(self, fx_call_spread):
        """Vega should be positive."""
        pricer = FxSpreadEuropeanOptionBchPricerSimple()
        greeks = pricer.greeks(fx_call_spread)
        assert greeks["vega"] > 0
    
    def test_eq_greeks_exist(self, eq_call_spread):
        """Equity spread Greeks should be computed."""
        pricer = EquitySpreadEuropeanOptionBchPricerSimple()
        greeks = pricer.greeks(eq_call_spread)
        
        assert "delta" in greeks
        assert "gamma" in greeks
        assert "vega" in greeks


# =============================================================================
# FINITE DIFFERENCE TESTS
# =============================================================================


class TestFiniteDifferenceGreeks:
    """Finite difference validation for Greeks."""
    
    def test_fx_delta_fd(self, fx_spread_params):
        """FX Delta should match finite difference approximation."""
        pricer = FxSpreadEuropeanOptionBchPricerSimple()
        
        bump = 1e-5
        
        params_up = {**fx_spread_params, "forward_spread": fx_spread_params["forward_spread"] + bump}
        params_dn = {**fx_spread_params, "forward_spread": fx_spread_params["forward_spread"] - bump}
        
        sp_up = FxSpreadEuropeanOptionSimple(**params_up)
        sp_dn = FxSpreadEuropeanOptionSimple(**params_dn)
        sp_mid = FxSpreadEuropeanOptionSimple(**fx_spread_params)
        
        pv_up = pricer.price(sp_up)
        pv_dn = pricer.price(sp_dn)
        
        fd_delta = (pv_up - pv_dn) / (2 * bump)
        
        greeks = pricer.greeks(sp_mid)
        analytic_delta = greeks["delta"]
        
        assert abs(fd_delta - analytic_delta) / abs(analytic_delta) < 0.01
    
    def test_fx_vega_fd(self, fx_spread_params):
        """FX Vega should match finite difference approximation."""
        pricer = FxSpreadEuropeanOptionBchPricerSimple()
        
        bump = 1e-6
        
        params_up = {**fx_spread_params, "vol": fx_spread_params["vol"] + bump}
        params_dn = {**fx_spread_params, "vol": fx_spread_params["vol"] - bump}
        
        sp_up = FxSpreadEuropeanOptionSimple(**params_up)
        sp_dn = FxSpreadEuropeanOptionSimple(**params_dn)
        sp_mid = FxSpreadEuropeanOptionSimple(**fx_spread_params)
        
        pv_up = pricer.price(sp_up)
        pv_dn = pricer.price(sp_dn)
        
        fd_vega = (pv_up - pv_dn) / (2 * bump)
        
        greeks = pricer.greeks(sp_mid)
        analytic_vega = greeks["vega"]
        
        assert abs(fd_vega - analytic_vega) / abs(analytic_vega) < 0.01
    
    def test_eq_delta_fd(self, eq_spread_params):
        """Equity Delta should match finite difference approximation."""
        pricer = EquitySpreadEuropeanOptionBchPricerSimple()
        
        bump = 1e-4
        
        params_up = {**eq_spread_params, "forward_spread": eq_spread_params["forward_spread"] + bump}
        params_dn = {**eq_spread_params, "forward_spread": eq_spread_params["forward_spread"] - bump}
        
        sp_up = EquitySpreadEuropeanOptionSimple(**params_up)
        sp_dn = EquitySpreadEuropeanOptionSimple(**params_dn)
        sp_mid = EquitySpreadEuropeanOptionSimple(**eq_spread_params)
        
        pv_up = pricer.price(sp_up)
        pv_dn = pricer.price(sp_dn)
        
        fd_delta = (pv_up - pv_dn) / (2 * bump)
        
        greeks = pricer.greeks(sp_mid)
        analytic_delta = greeks["delta"]
        
        assert abs(fd_delta - analytic_delta) / abs(analytic_delta) < 0.01


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_atm_spread_option(self, fx_spread_params):
        """ATM spread option should have positive value."""
        params = {**fx_spread_params, "forward_spread": fx_spread_params["strike"]}
        spread = FxSpreadEuropeanOptionSimple(**params)
        
        pricer = FxSpreadEuropeanOptionBchPricerSimple()
        pv = pricer.price(spread)
        
        assert pv > 0
    
    def test_zero_vol_spread_option(self, fx_spread_params):
        """Zero vol spread option should equal intrinsic."""
        params = {**fx_spread_params, "vol": 0.0}
        spread = FxSpreadEuropeanOptionSimple(**params)
        
        pricer = FxSpreadEuropeanOptionBchPricerSimple()
        pv = pricer.price(spread)
        
        # Intrinsic for call = N × DF × max(F - K, 0)
        N = params["notional"]
        df = params["discount_factor"]
        F = params["forward_spread"]
        K = params["strike"]
        intrinsic = N * df * max(F - K, 0)
        
        assert abs(pv - intrinsic) < 1e-6
    
    def test_negative_strike(self, fx_spread_params):
        """Bachelier handles negative strikes (spread can be negative)."""
        params = {**fx_spread_params, "strike": -0.05, "forward_spread": -0.02}
        spread = FxSpreadEuropeanOptionSimple(**params)
        
        pricer = FxSpreadEuropeanOptionBchPricerSimple()
        pv = pricer.price(spread)
        
        # Should still produce valid positive price
        assert pv > 0
    
    def test_negative_forward_spread(self, fx_spread_params):
        """Spread can be negative (S1 < S2)."""
        params = {**fx_spread_params, "forward_spread": -0.05, "strike": 0.0}
        spread = FxSpreadEuropeanOptionSimple(**params)
        
        pricer = FxSpreadEuropeanOptionBchPricerSimple()
        pv = pricer.price(spread)
        
        # OTM call should have positive value due to optionality
        assert pv > 0
    
    def test_expired_option(self, fx_spread_params):
        """Expired option should return intrinsic value."""
        params = {**fx_spread_params, "expiry": 0.0}
        spread = FxSpreadEuropeanOptionSimple(**params)
        
        pricer = FxSpreadEuropeanOptionBchPricerSimple()
        pv = pricer.price(spread)
        
        # ITM call at expiry
        N = params["notional"]
        df = params["discount_factor"]
        F = params["forward_spread"]
        K = params["strike"]
        expected = N * df * max(F - K, 0)
        
        assert abs(pv - expected) < 1e-6

# tests/unit/pricers/equity/test_equity_european_b76_pricer.py
"""
Unit tests for Equity Futures Option Black76 Pricer.

Tests cover:
- Basic pricing functionality
- Put-call parity
- Greeks computation
- Finite difference validation
- Edge cases

Author: QuantStrata Team
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.marketdata.core.ids import MarketId
from src.marketdata.core.market import Market
from src.marketdata.core.interfaces import Quote
from src.marketdata.surfaces.vol_surface import FlatVolSurface
from src.marketdata.curves.term_structure import FlatZeroRateCurve

from src.instruments.equity.options.futures import (
    EquityFuturesEuropeanOption,
    EquityFuturesEuropeanOptionSimple,
)
from src.pricers.equity.european_b76 import (
    EquityFuturesEuropeanOptionB76Pricer,
    EquityFuturesEuropeanOptionB76PricerSimple,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def market_ids():
    """Standard MarketIds for SPX futures options."""
    return {
        "spot": MarketId("EQUITY", "SPOT", "SPX", (("ccy", "USD"),)),
        "vol": MarketId("EQUITY", "VOL", "SPX", (("ccy", "USD"),)),
        "curve": MarketId("IR", "CURVE", "USD.OIS", (("ccy", "USD"),)),
    }


@pytest.fixture
def base_market(market_ids):
    """
    Create a base market for SPX futures option pricing.

    Spot: 5000
    Rate: 5%
    Dividend yield: 1.5%
    Vol: 18%

    Futures (1Y): 5000 × exp((0.05 - 0.015) × 1) ≈ 5178.5
    """
    return Market(
        asof="2026-01-15",
        quotes={
            market_ids["spot"]: Quote(value=5000.0),
        },
        curves={
            market_ids["curve"]: FlatZeroRateCurve(continuously_compounded_rate=0.05),
        },
        vols={
            market_ids["vol"]: FlatVolSurface(sigma=0.18),
        },
    )


@pytest.fixture
def pricer():
    """Create Black76 pricer."""
    return EquityFuturesEuropeanOptionB76Pricer()


@pytest.fixture
def simple_pricer():
    """Create simple Black76 pricer."""
    return EquityFuturesEuropeanOptionB76PricerSimple()


# =============================================================================
# INSTRUMENT VALIDATION TESTS
# =============================================================================


class TestEuropeanEquityFuturesOptionValidation:
    """Test instrument validation."""

    def test_invalid_option_type(self, market_ids):
        """Should reject invalid option type."""
        with pytest.raises(ValueError, match="option_type"):
            EquityFuturesEuropeanOption(
                ticker="SPX",
                option_type="invalid",
                strike=5000.0,
                expiry=0.25,
                futures_expiry=0.25,
                notional=50.0,
                spot_id=market_ids["spot"],
                vol_id=market_ids["vol"],
                curve_id=market_ids["curve"],
                dividend_yield=0.015,
            )

    def test_empty_ticker(self, market_ids):
        """Should reject empty ticker."""
        with pytest.raises(ValueError, match="ticker"):
            EquityFuturesEuropeanOption(
                ticker="",
                option_type="call",
                strike=5000.0,
                expiry=0.25,
                futures_expiry=0.25,
                notional=50.0,
                spot_id=market_ids["spot"],
                vol_id=market_ids["vol"],
                curve_id=market_ids["curve"],
                dividend_yield=0.015,
            )

    def test_negative_strike(self, market_ids):
        """Should reject negative strike."""
        with pytest.raises(ValueError, match="strike"):
            EquityFuturesEuropeanOption(
                ticker="SPX",
                option_type="call",
                strike=-5000.0,
                expiry=0.25,
                futures_expiry=0.25,
                notional=50.0,
                spot_id=market_ids["spot"],
                vol_id=market_ids["vol"],
                curve_id=market_ids["curve"],
                dividend_yield=0.015,
            )

    def test_futures_expiry_before_option_expiry(self, market_ids):
        """Should reject futures_expiry < expiry."""
        with pytest.raises(ValueError, match="futures_expiry"):
            EquityFuturesEuropeanOption(
                ticker="SPX",
                option_type="call",
                strike=5000.0,
                expiry=0.5,
                futures_expiry=0.25,  # Before option expiry
                notional=50.0,
                spot_id=market_ids["spot"],
                vol_id=market_ids["vol"],
                curve_id=market_ids["curve"],
                dividend_yield=0.015,
            )


# =============================================================================
# BASIC PRICING TESTS
# =============================================================================


class TestEquityFuturesOptionBlack76Pricing:
    """Test basic pricing functionality."""

    def test_call_price_positive(self, market_ids, base_market, pricer):
        """Call price should be positive."""
        option = EquityFuturesEuropeanOption(
            ticker="SPX",
            option_type="call",
            strike=5200.0,
            expiry=1.0,
            futures_expiry=1.0,
            notional=50.0,
            spot_id=market_ids["spot"],
            vol_id=market_ids["vol"],
            curve_id=market_ids["curve"],
            dividend_yield=0.015,
        )

        pv = pricer.price(option, base_market)

        assert pv > 0, "Call option should have positive value."

    def test_put_price_positive(self, market_ids, base_market, pricer):
        """Put price should be positive."""
        option = EquityFuturesEuropeanOption(
            ticker="SPX",
            option_type="put",
            strike=5200.0,
            expiry=1.0,
            futures_expiry=1.0,
            notional=50.0,
            spot_id=market_ids["spot"],
            vol_id=market_ids["vol"],
            curve_id=market_ids["curve"],
            dividend_yield=0.015,
        )

        pv = pricer.price(option, base_market)

        assert pv > 0, "Put option should have positive value."

    def test_notional_scaling(self, market_ids, base_market, pricer):
        """Price should scale linearly with notional."""
        option_50 = EquityFuturesEuropeanOption(
            ticker="SPX",
            option_type="call",
            strike=5200.0,
            expiry=1.0,
            futures_expiry=1.0,
            notional=50.0,
            spot_id=market_ids["spot"],
            vol_id=market_ids["vol"],
            curve_id=market_ids["curve"],
            dividend_yield=0.015,
        )

        option_100 = EquityFuturesEuropeanOption(
            ticker="SPX",
            option_type="call",
            strike=5200.0,
            expiry=1.0,
            futures_expiry=1.0,
            notional=100.0,
            spot_id=market_ids["spot"],
            vol_id=market_ids["vol"],
            curve_id=market_ids["curve"],
            dividend_yield=0.015,
        )

        pv_50 = pricer.price(option_50, base_market)
        pv_100 = pricer.price(option_100, base_market)

        assert pytest.approx(pv_100, rel=1e-10) == 2.0 * pv_50


# =============================================================================
# PUT-CALL PARITY TESTS
# =============================================================================


class TestPutCallParity:
    """
    Test put-call parity for Black76.

    For Black76: C - P = DF × (F - K)
    """

    def test_put_call_parity_atm(self, market_ids, base_market, pricer):
        """Put-call parity should hold at ATM."""
        # Compute futures price.
        spot = 5000.0
        r, q = 0.05, 0.015
        t = 1.0
        futures = spot * math.exp((r - q) * t)
        df = math.exp(-r * t)

        # Use futures as strike (ATM forward).
        strike = futures

        call = EquityFuturesEuropeanOption(
            ticker="SPX",
            option_type="call",
            strike=strike,
            expiry=t,
            futures_expiry=t,
            notional=1.0,
            spot_id=market_ids["spot"],
            vol_id=market_ids["vol"],
            curve_id=market_ids["curve"],
            dividend_yield=q,
        )

        put = EquityFuturesEuropeanOption(
            ticker="SPX",
            option_type="put",
            strike=strike,
            expiry=t,
            futures_expiry=t,
            notional=1.0,
            spot_id=market_ids["spot"],
            vol_id=market_ids["vol"],
            curve_id=market_ids["curve"],
            dividend_yield=q,
        )

        call_pv = pricer.price(call, base_market)
        put_pv = pricer.price(put, base_market)

        # At ATM forward, C - P = DF × (F - K) = 0.
        assert pytest.approx(call_pv - put_pv, abs=1e-8) == 0.0

    def test_put_call_parity_various_strikes(self, market_ids, base_market, pricer):
        """Put-call parity should hold at various strikes."""
        spot = 5000.0
        r, q = 0.05, 0.015
        t = 1.0
        futures = spot * math.exp((r - q) * t)
        df = math.exp(-r * t)

        strikes = [4500, 4800, 5000, 5200, 5500]

        for strike in strikes:
            call = EquityFuturesEuropeanOption(
                ticker="SPX",
                option_type="call",
                strike=float(strike),
                expiry=t,
                futures_expiry=t,
                notional=1.0,
                spot_id=market_ids["spot"],
                vol_id=market_ids["vol"],
                curve_id=market_ids["curve"],
                dividend_yield=q,
            )

            put = EquityFuturesEuropeanOption(
                ticker="SPX",
                option_type="put",
                strike=float(strike),
                expiry=t,
                futures_expiry=t,
                notional=1.0,
                spot_id=market_ids["spot"],
                vol_id=market_ids["vol"],
                curve_id=market_ids["curve"],
                dividend_yield=q,
            )

            call_pv = pricer.price(call, base_market)
            put_pv = pricer.price(put, base_market)

            expected = df * (futures - strike)

            assert pytest.approx(call_pv - put_pv, rel=1e-6) == expected, (
                f"Put-call parity failed at strike {strike}"
            )


# =============================================================================
# GREEKS TESTS
# =============================================================================


class TestGreeks:
    """Test Greeks computation."""

    def test_greeks_exist(self, market_ids, base_market, pricer):
        """All Greeks should be computed."""
        option = EquityFuturesEuropeanOption(
            ticker="SPX",
            option_type="call",
            strike=5200.0,
            expiry=1.0,
            futures_expiry=1.0,
            notional=50.0,
            spot_id=market_ids["spot"],
            vol_id=market_ids["vol"],
            curve_id=market_ids["curve"],
            dividend_yield=0.015,
        )

        greeks = pricer.greeks(option, base_market)

        expected_keys = {
            "delta_futures",
            "delta_spot",
            "gamma",
            "vega",
            "theta",
            "rho",
        }

        assert set(greeks.keys()) == expected_keys

    def test_call_delta_positive(self, market_ids, base_market, pricer):
        """Call delta should be positive."""
        option = EquityFuturesEuropeanOption(
            ticker="SPX",
            option_type="call",
            strike=5200.0,
            expiry=1.0,
            futures_expiry=1.0,
            notional=50.0,
            spot_id=market_ids["spot"],
            vol_id=market_ids["vol"],
            curve_id=market_ids["curve"],
            dividend_yield=0.015,
        )

        greeks = pricer.greeks(option, base_market)

        assert greeks["delta_futures"] > 0
        assert greeks["delta_spot"] > 0

    def test_put_delta_negative(self, market_ids, base_market, pricer):
        """Put delta should be negative."""
        option = EquityFuturesEuropeanOption(
            ticker="SPX",
            option_type="put",
            strike=5200.0,
            expiry=1.0,
            futures_expiry=1.0,
            notional=50.0,
            spot_id=market_ids["spot"],
            vol_id=market_ids["vol"],
            curve_id=market_ids["curve"],
            dividend_yield=0.015,
        )

        greeks = pricer.greeks(option, base_market)

        assert greeks["delta_futures"] < 0
        assert greeks["delta_spot"] < 0

    def test_gamma_positive(self, market_ids, base_market, pricer):
        """Gamma should be positive for both calls and puts."""
        call = EquityFuturesEuropeanOption(
            ticker="SPX",
            option_type="call",
            strike=5200.0,
            expiry=1.0,
            futures_expiry=1.0,
            notional=50.0,
            spot_id=market_ids["spot"],
            vol_id=market_ids["vol"],
            curve_id=market_ids["curve"],
            dividend_yield=0.015,
        )

        call_greeks = pricer.greeks(call, base_market)

        assert call_greeks["gamma"] > 0

    def test_vega_positive(self, market_ids, base_market, pricer):
        """Vega should be positive."""
        option = EquityFuturesEuropeanOption(
            ticker="SPX",
            option_type="call",
            strike=5200.0,
            expiry=1.0,
            futures_expiry=1.0,
            notional=50.0,
            spot_id=market_ids["spot"],
            vol_id=market_ids["vol"],
            curve_id=market_ids["curve"],
            dividend_yield=0.015,
        )

        greeks = pricer.greeks(option, base_market)

        assert greeks["vega"] > 0


# =============================================================================
# FINITE DIFFERENCE VALIDATION
# =============================================================================


class TestFiniteDifferenceValidation:
    """Validate Greeks using finite differences."""

    def test_delta_spot_fd(self, market_ids, pricer):
        """Delta spot should match finite difference."""
        spot_base = 5000.0
        r, q = 0.05, 0.015
        vol = 0.18
        bump = 1.0  # $1 bump

        def make_market(spot_val):
            return Market(
                asof="2026-01-15",
                quotes={
                    market_ids["spot"]: Quote(value=spot_val),
                },
                curves={
                    market_ids["curve"]: FlatZeroRateCurve(continuously_compounded_rate=r),
                },
                vols={
                    market_ids["vol"]: FlatVolSurface(sigma=vol),
                },
            )

        option = EquityFuturesEuropeanOption(
            ticker="SPX",
            option_type="call",
            strike=5200.0,
            expiry=1.0,
            futures_expiry=1.0,
            notional=1.0,
            spot_id=market_ids["spot"],
            vol_id=market_ids["vol"],
            curve_id=market_ids["curve"],
            dividend_yield=q,
        )

        market_up = make_market(spot_base + bump)
        market_dn = make_market(spot_base - bump)
        market_mid = make_market(spot_base)

        pv_up = pricer.price(option, market_up)
        pv_dn = pricer.price(option, market_dn)

        delta_fd = (pv_up - pv_dn) / (2 * bump)
        delta_analytic = pricer.greeks(option, market_mid)["delta_spot"]

        assert pytest.approx(delta_fd, rel=0.01) == delta_analytic

    def test_vega_fd(self, market_ids, pricer):
        """Vega should match finite difference."""
        spot = 5000.0
        r, q = 0.05, 0.015
        vol_base = 0.18
        bump = 0.001  # 10 bps

        def make_market(vol_val):
            return Market(
                asof="2026-01-15",
                quotes={
                    market_ids["spot"]: Quote(value=spot),
                },
                curves={
                    market_ids["curve"]: FlatZeroRateCurve(continuously_compounded_rate=r),
                },
                vols={
                    market_ids["vol"]: FlatVolSurface(sigma=vol_val),
                },
            )

        option = EquityFuturesEuropeanOption(
            ticker="SPX",
            option_type="call",
            strike=5200.0,
            expiry=1.0,
            futures_expiry=1.0,
            notional=1.0,
            spot_id=market_ids["spot"],
            vol_id=market_ids["vol"],
            curve_id=market_ids["curve"],
            dividend_yield=q,
        )

        market_up = make_market(vol_base + bump)
        market_dn = make_market(vol_base - bump)
        market_mid = make_market(vol_base)

        pv_up = pricer.price(option, market_up)
        pv_dn = pricer.price(option, market_dn)

        vega_fd = (pv_up - pv_dn) / (2 * bump)
        vega_analytic = pricer.greeks(option, market_mid)["vega"]

        assert pytest.approx(vega_fd, rel=0.01) == vega_analytic


# =============================================================================
# SIMPLE PRICER TESTS
# =============================================================================


class TestSimplePricer:
    """Test the simplified pricer."""

    def test_simple_pricer_matches_full(self, market_ids, base_market, pricer, simple_pricer):
        """Simple pricer should match full pricer when given same inputs."""
        # Compute futures and discount factor from market.
        spot = 5000.0
        r, q = 0.05, 0.015
        t = 1.0
        futures = spot * math.exp((r - q) * t)
        df = math.exp(-r * t)
        vol = 0.18

        # Full pricer option.
        option_full = EquityFuturesEuropeanOption(
            ticker="SPX",
            option_type="call",
            strike=5200.0,
            expiry=t,
            futures_expiry=t,
            notional=50.0,
            spot_id=market_ids["spot"],
            vol_id=market_ids["vol"],
            curve_id=market_ids["curve"],
            dividend_yield=q,
        )

        # Simple pricer option.
        option_simple = EquityFuturesEuropeanOptionSimple(
            ticker="SPX",
            option_type="call",
            strike=5200.0,
            expiry=t,
            futures_price=futures,
            vol=vol,
            discount_factor=df,
            notional=50.0,
        )

        pv_full = pricer.price(option_full, base_market)
        pv_simple = simple_pricer.price(option_simple)

        assert pytest.approx(pv_full, rel=1e-8) == pv_simple

    def test_simple_greeks(self, simple_pricer):
        """Simple pricer should compute Greeks."""
        option = EquityFuturesEuropeanOptionSimple(
            ticker="SPX",
            option_type="call",
            strike=5200.0,
            expiry=1.0,
            futures_price=5178.5,
            vol=0.18,
            discount_factor=0.9512,
            notional=50.0,
        )

        greeks = simple_pricer.greeks(option)

        assert "delta" in greeks
        assert "gamma" in greeks
        assert "vega" in greeks
        assert "theta" in greeks
        assert "rho" in greeks


# =============================================================================
# EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Test edge cases."""

    def test_deep_itm_call(self, market_ids, base_market, pricer):
        """Deep ITM call should be close to DF × (F - K)."""
        spot = 5000.0
        r, q = 0.05, 0.015
        t = 1.0
        futures = spot * math.exp((r - q) * t)  # ≈ 5178.5
        df = math.exp(-r * t)
        strike = 4000.0  # Very deep ITM

        option = EquityFuturesEuropeanOption(
            ticker="SPX",
            option_type="call",
            strike=strike,
            expiry=t,
            futures_expiry=t,
            notional=1.0,
            spot_id=market_ids["spot"],
            vol_id=market_ids["vol"],
            curve_id=market_ids["curve"],
            dividend_yield=q,
        )

        pv = pricer.price(option, base_market)
        intrinsic = df * (futures - strike)

        # Deep ITM call should be very close to intrinsic.
        assert pv > intrinsic * 0.99

    def test_deep_otm_call(self, market_ids, base_market, pricer):
        """Deep OTM call should be close to zero."""
        strike = 8000.0  # Very deep OTM (60%+ above forward)

        option = EquityFuturesEuropeanOption(
            ticker="SPX",
            option_type="call",
            strike=strike,
            expiry=1.0,
            futures_expiry=1.0,
            notional=1.0,
            spot_id=market_ids["spot"],
            vol_id=market_ids["vol"],
            curve_id=market_ids["curve"],
            dividend_yield=0.015,
        )

        pv = pricer.price(option, base_market)

        # Deep OTM call should be very small relative to strike.
        assert pv < 10.0  # Less than $10 for very deep OTM

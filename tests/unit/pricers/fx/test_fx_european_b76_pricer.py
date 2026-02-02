# tests/unit/pricers/fx/test_fx_forward_option_black76.py
"""
Unit tests for FX Forward Option Black76 Pricer.

Tests cover:
- Basic pricing functionality
- Put-call parity
- Greeks computation
- Finite difference validation
- Edge cases (ATM, deep ITM/OTM, zero vol, expiry)

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

from src.instruments.fx.options.forward import (
    FxForwardEuropeanOption,
    FxForwardEuropeanOptionSimple,
)
from src.pricers.fx.european_b76 import (
    FxForwardEuropeanOptionB76Pricer,
    FxForwardEuropeanOptionB76PricerSimple,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def market_ids():
    """Standard MarketIds for EURUSD."""
    return {
        "spot": MarketId("FX", "SPOT", "EURUSD", (("dom", "USD"), ("for", "EUR"))),
        "vol": MarketId("FX", "VOL", "EURUSD", (("dom", "USD"), ("for", "EUR"))),
        "dom_curve": MarketId("IR", "CURVE", "USD.OIS", (("ccy", "USD"),)),
        "for_curve": MarketId("IR", "CURVE", "EUR.OIS", (("ccy", "EUR"),)),
    }


@pytest.fixture
def base_market(market_ids):
    """
    Create a base market for EURUSD forward option pricing.

    Spot: 1.10
    Domestic rate (USD): 5%
    Foreign rate (EUR): 3%
    Vol: 8%

    Forward (1Y): 1.10 × exp((0.05 - 0.03) × 1) ≈ 1.1222
    """
    return Market(
        asof="2026-01-15",
        quotes={
            market_ids["spot"]: Quote(value=1.10),
        },
        curves={
            market_ids["dom_curve"]: FlatZeroRateCurve(continuously_compounded_rate=0.05),
            market_ids["for_curve"]: FlatZeroRateCurve(continuously_compounded_rate=0.03),
        },
        vols={
            market_ids["vol"]: FlatVolSurface(sigma=0.08),
        },
    )


@pytest.fixture
def pricer():
    """Create Black76 pricer."""
    return FxForwardEuropeanOptionB76Pricer()


@pytest.fixture
def simple_pricer():
    """Create simple Black76 pricer."""
    return FxForwardEuropeanOptionB76PricerSimple()


# =============================================================================
# INSTRUMENT VALIDATION TESTS
# =============================================================================


class TestFxForwardEuropeanOptionValidation:
    """Test instrument validation."""

    def test_invalid_option_type(self, market_ids):
        """Should reject invalid option type."""
        with pytest.raises(ValueError, match="option_type"):
            FxForwardEuropeanOption(
                option_type="invalid",
                notional=1_000_000,
                strike=1.12,
                expiry=0.5,
                forward_expiry=0.5,
                spot_id=market_ids["spot"],
                vol_id=market_ids["vol"],
                domestic_curve_id=market_ids["dom_curve"],
                foreign_curve_id=market_ids["for_curve"],
            )

    def test_zero_notional(self, market_ids):
        """Should reject zero notional."""
        with pytest.raises(ValueError, match="notional"):
            FxForwardEuropeanOption(
                option_type="call",
                notional=0.0,
                strike=1.12,
                expiry=0.5,
                forward_expiry=0.5,
                spot_id=market_ids["spot"],
                vol_id=market_ids["vol"],
                domestic_curve_id=market_ids["dom_curve"],
                foreign_curve_id=market_ids["for_curve"],
            )

    def test_negative_strike(self, market_ids):
        """Should reject negative strike."""
        with pytest.raises(ValueError, match="strike"):
            FxForwardEuropeanOption(
                option_type="call",
                notional=1_000_000,
                strike=-1.12,
                expiry=0.5,
                forward_expiry=0.5,
                spot_id=market_ids["spot"],
                vol_id=market_ids["vol"],
                domestic_curve_id=market_ids["dom_curve"],
                foreign_curve_id=market_ids["for_curve"],
            )

    def test_forward_expiry_before_option_expiry(self, market_ids):
        """Should reject forward_expiry < expiry."""
        with pytest.raises(ValueError, match="forward_expiry"):
            FxForwardEuropeanOption(
                option_type="call",
                notional=1_000_000,
                strike=1.12,
                expiry=1.0,
                forward_expiry=0.5,  # Before option expiry
                spot_id=market_ids["spot"],
                vol_id=market_ids["vol"],
                domestic_curve_id=market_ids["dom_curve"],
                foreign_curve_id=market_ids["for_curve"],
            )


# =============================================================================
# BASIC PRICING TESTS
# =============================================================================


class TestFxForwardOptionBlack76Pricing:
    """Test basic pricing functionality."""

    def test_call_price_positive(self, market_ids, base_market, pricer):
        """Call price should be positive."""
        option = FxForwardEuropeanOption(
            option_type="call",
            notional=1_000_000,
            strike=1.12,
            expiry=1.0,
            forward_expiry=1.0,
            spot_id=market_ids["spot"],
            vol_id=market_ids["vol"],
            domestic_curve_id=market_ids["dom_curve"],
            foreign_curve_id=market_ids["for_curve"],
        )

        pv = pricer.price(option, base_market)

        assert pv > 0, "Call option should have positive value."

    def test_put_price_positive(self, market_ids, base_market, pricer):
        """Put price should be positive."""
        option = FxForwardEuropeanOption(
            option_type="put",
            notional=1_000_000,
            strike=1.12,
            expiry=1.0,
            forward_expiry=1.0,
            spot_id=market_ids["spot"],
            vol_id=market_ids["vol"],
            domestic_curve_id=market_ids["dom_curve"],
            foreign_curve_id=market_ids["for_curve"],
        )

        pv = pricer.price(option, base_market)

        assert pv > 0, "Put option should have positive value."

    def test_notional_scaling(self, market_ids, base_market, pricer):
        """Price should scale linearly with notional."""
        option_1m = FxForwardEuropeanOption(
            option_type="call",
            notional=1_000_000,
            strike=1.12,
            expiry=1.0,
            forward_expiry=1.0,
            spot_id=market_ids["spot"],
            vol_id=market_ids["vol"],
            domestic_curve_id=market_ids["dom_curve"],
            foreign_curve_id=market_ids["for_curve"],
        )

        option_2m = FxForwardEuropeanOption(
            option_type="call",
            notional=2_000_000,
            strike=1.12,
            expiry=1.0,
            forward_expiry=1.0,
            spot_id=market_ids["spot"],
            vol_id=market_ids["vol"],
            domestic_curve_id=market_ids["dom_curve"],
            foreign_curve_id=market_ids["for_curve"],
        )

        pv_1m = pricer.price(option_1m, base_market)
        pv_2m = pricer.price(option_2m, base_market)

        assert pytest.approx(pv_2m, rel=1e-10) == 2.0 * pv_1m


# =============================================================================
# PUT-CALL PARITY TESTS
# =============================================================================


class TestPutCallParity:
    """
    Test put-call parity for Black76.

    For Black76: C - P = DF × (F - K)

    Where:
        C = call price
        P = put price
        DF = discount factor
        F = forward rate
        K = strike
    """

    def test_put_call_parity_atm(self, market_ids, base_market, pricer):
        """Put-call parity should hold at ATM."""
        # Compute forward rate.
        spot = 1.10
        r_d, r_f = 0.05, 0.03
        t = 1.0
        forward = spot * math.exp((r_d - r_f) * t)
        df = math.exp(-r_d * t)

        # Use forward as strike (ATM forward).
        strike = forward

        call = FxForwardEuropeanOption(
            option_type="call",
            notional=1.0,
            strike=strike,
            expiry=t,
            forward_expiry=t,
            spot_id=market_ids["spot"],
            vol_id=market_ids["vol"],
            domestic_curve_id=market_ids["dom_curve"],
            foreign_curve_id=market_ids["for_curve"],
        )

        put = FxForwardEuropeanOption(
            option_type="put",
            notional=1.0,
            strike=strike,
            expiry=t,
            forward_expiry=t,
            spot_id=market_ids["spot"],
            vol_id=market_ids["vol"],
            domestic_curve_id=market_ids["dom_curve"],
            foreign_curve_id=market_ids["for_curve"],
        )

        call_pv = pricer.price(call, base_market)
        put_pv = pricer.price(put, base_market)

        # At ATM forward, C - P = DF × (F - K) = 0.
        assert pytest.approx(call_pv - put_pv, abs=1e-10) == 0.0

    def test_put_call_parity_various_strikes(self, market_ids, base_market, pricer):
        """Put-call parity should hold at various strikes."""
        spot = 1.10
        r_d, r_f = 0.05, 0.03
        t = 1.0
        forward = spot * math.exp((r_d - r_f) * t)
        df = math.exp(-r_d * t)

        strikes = [1.05, 1.10, 1.12, 1.15, 1.20, 1.25]

        for strike in strikes:
            call = FxForwardEuropeanOption(
                option_type="call",
                notional=1.0,
                strike=strike,
                expiry=t,
                forward_expiry=t,
                spot_id=market_ids["spot"],
                vol_id=market_ids["vol"],
                domestic_curve_id=market_ids["dom_curve"],
                foreign_curve_id=market_ids["for_curve"],
            )

            put = FxForwardEuropeanOption(
                option_type="put",
                notional=1.0,
                strike=strike,
                expiry=t,
                forward_expiry=t,
                spot_id=market_ids["spot"],
                vol_id=market_ids["vol"],
                domestic_curve_id=market_ids["dom_curve"],
                foreign_curve_id=market_ids["for_curve"],
            )

            call_pv = pricer.price(call, base_market)
            put_pv = pricer.price(put, base_market)

            expected = df * (forward - strike)

            assert pytest.approx(call_pv - put_pv, rel=1e-8) == expected, (
                f"Put-call parity failed at strike {strike}"
            )


# =============================================================================
# GREEKS TESTS
# =============================================================================


class TestGreeks:
    """Test Greeks computation."""

    def test_greeks_exist(self, market_ids, base_market, pricer):
        """All Greeks should be computed."""
        option = FxForwardEuropeanOption(
            option_type="call",
            notional=1_000_000,
            strike=1.12,
            expiry=1.0,
            forward_expiry=1.0,
            spot_id=market_ids["spot"],
            vol_id=market_ids["vol"],
            domestic_curve_id=market_ids["dom_curve"],
            foreign_curve_id=market_ids["for_curve"],
        )

        greeks = pricer.greeks(option, base_market)

        expected_keys = {
            "delta_forward",
            "delta_spot",
            "gamma",
            "vega",
            "theta",
            "rho_domestic",
            "rho_foreign",
        }

        assert set(greeks.keys()) == expected_keys

    def test_call_delta_positive(self, market_ids, base_market, pricer):
        """Call delta should be positive."""
        option = FxForwardEuropeanOption(
            option_type="call",
            notional=1_000_000,
            strike=1.12,
            expiry=1.0,
            forward_expiry=1.0,
            spot_id=market_ids["spot"],
            vol_id=market_ids["vol"],
            domestic_curve_id=market_ids["dom_curve"],
            foreign_curve_id=market_ids["for_curve"],
        )

        greeks = pricer.greeks(option, base_market)

        assert greeks["delta_forward"] > 0
        assert greeks["delta_spot"] > 0

    def test_put_delta_negative(self, market_ids, base_market, pricer):
        """Put delta should be negative."""
        option = FxForwardEuropeanOption(
            option_type="put",
            notional=1_000_000,
            strike=1.12,
            expiry=1.0,
            forward_expiry=1.0,
            spot_id=market_ids["spot"],
            vol_id=market_ids["vol"],
            domestic_curve_id=market_ids["dom_curve"],
            foreign_curve_id=market_ids["for_curve"],
        )

        greeks = pricer.greeks(option, base_market)

        assert greeks["delta_forward"] < 0
        assert greeks["delta_spot"] < 0

    def test_gamma_positive(self, market_ids, base_market, pricer):
        """Gamma should be positive for both calls and puts."""
        call = FxForwardEuropeanOption(
            option_type="call",
            notional=1_000_000,
            strike=1.12,
            expiry=1.0,
            forward_expiry=1.0,
            spot_id=market_ids["spot"],
            vol_id=market_ids["vol"],
            domestic_curve_id=market_ids["dom_curve"],
            foreign_curve_id=market_ids["for_curve"],
        )

        put = FxForwardEuropeanOption(
            option_type="put",
            notional=1_000_000,
            strike=1.12,
            expiry=1.0,
            forward_expiry=1.0,
            spot_id=market_ids["spot"],
            vol_id=market_ids["vol"],
            domestic_curve_id=market_ids["dom_curve"],
            foreign_curve_id=market_ids["for_curve"],
        )

        call_greeks = pricer.greeks(call, base_market)
        put_greeks = pricer.greeks(put, base_market)

        assert call_greeks["gamma"] > 0
        assert put_greeks["gamma"] > 0

    def test_vega_positive(self, market_ids, base_market, pricer):
        """Vega should be positive for both calls and puts."""
        call = FxForwardEuropeanOption(
            option_type="call",
            notional=1_000_000,
            strike=1.12,
            expiry=1.0,
            forward_expiry=1.0,
            spot_id=market_ids["spot"],
            vol_id=market_ids["vol"],
            domestic_curve_id=market_ids["dom_curve"],
            foreign_curve_id=market_ids["for_curve"],
        )

        call_greeks = pricer.greeks(call, base_market)

        assert call_greeks["vega"] > 0


# =============================================================================
# FINITE DIFFERENCE VALIDATION
# =============================================================================


class TestFiniteDifferenceValidation:
    """Validate Greeks using finite differences."""

    def test_delta_spot_fd(self, market_ids, pricer):
        """Delta spot should match finite difference."""
        spot_base = 1.10
        r_d, r_f = 0.05, 0.03
        vol = 0.08
        bump = 0.001

        def make_market(spot_val):
            return Market(
                asof="2026-01-15",
                quotes={
                    market_ids["spot"]: Quote(value=spot_val),
                },
                curves={
                    market_ids["dom_curve"]: FlatZeroRateCurve(continuously_compounded_rate=r_d),
                    market_ids["for_curve"]: FlatZeroRateCurve(continuously_compounded_rate=r_f),
                },
                vols={
                    market_ids["vol"]: FlatVolSurface(sigma=vol),
                },
            )

        option = FxForwardEuropeanOption(
            option_type="call",
            notional=1.0,
            strike=1.12,
            expiry=1.0,
            forward_expiry=1.0,
            spot_id=market_ids["spot"],
            vol_id=market_ids["vol"],
            domestic_curve_id=market_ids["dom_curve"],
            foreign_curve_id=market_ids["for_curve"],
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
        spot = 1.10
        r_d, r_f = 0.05, 0.03
        vol_base = 0.08
        bump = 0.001  # 10 bps

        def make_market(vol_val):
            return Market(
                asof="2026-01-15",
                quotes={
                    market_ids["spot"]: Quote(value=spot),
                },
                curves={
                    market_ids["dom_curve"]: FlatZeroRateCurve(continuously_compounded_rate=r_d),
                    market_ids["for_curve"]: FlatZeroRateCurve(continuously_compounded_rate=r_f),
                },
                vols={
                    market_ids["vol"]: FlatVolSurface(sigma=vol_val),
                },
            )

        option = FxForwardEuropeanOption(
            option_type="call",
            notional=1.0,
            strike=1.12,
            expiry=1.0,
            forward_expiry=1.0,
            spot_id=market_ids["spot"],
            vol_id=market_ids["vol"],
            domestic_curve_id=market_ids["dom_curve"],
            foreign_curve_id=market_ids["for_curve"],
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
        # Compute forward and discount factor from market.
        spot = 1.10
        r_d, r_f = 0.05, 0.03
        t = 1.0
        forward = spot * math.exp((r_d - r_f) * t)
        df = math.exp(-r_d * t)
        vol = 0.08

        # Full pricer option.
        option_full = FxForwardEuropeanOption(
            option_type="call",
            notional=1_000_000,
            strike=1.12,
            expiry=t,
            forward_expiry=t,
            spot_id=market_ids["spot"],
            vol_id=market_ids["vol"],
            domestic_curve_id=market_ids["dom_curve"],
            foreign_curve_id=market_ids["for_curve"],
        )

        # Simple pricer option.
        option_simple = FxForwardEuropeanOptionSimple(
            option_type="call",
            notional=1_000_000,
            strike=1.12,
            expiry=t,
            forward_rate=forward,
            vol=vol,
            discount_factor=df,
        )

        pv_full = pricer.price(option_full, base_market)
        pv_simple = simple_pricer.price(option_simple)

        assert pytest.approx(pv_full, rel=1e-8) == pv_simple

    def test_simple_greeks(self, simple_pricer):
        """Simple pricer should compute Greeks."""
        option = FxForwardEuropeanOptionSimple(
            option_type="call",
            notional=1_000_000,
            strike=1.12,
            expiry=1.0,
            forward_rate=1.1222,
            vol=0.08,
            discount_factor=0.9512,
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

    def test_zero_expiry(self, market_ids, base_market, pricer):
        """At expiry, price should be intrinsic value."""
        # Forward at expiry = spot (approximately, ignoring day fraction).
        forward = 1.10 * math.exp((0.05 - 0.03) * 0.0)  # = 1.10
        strike = 1.08  # ITM call

        option = FxForwardEuropeanOption(
            option_type="call",
            notional=1.0,
            strike=strike,
            expiry=0.0,
            forward_expiry=0.0,
            spot_id=market_ids["spot"],
            vol_id=market_ids["vol"],
            domestic_curve_id=market_ids["dom_curve"],
            foreign_curve_id=market_ids["for_curve"],
        )

        pv = pricer.price(option, base_market)

        # Intrinsic = max(F - K, 0) = max(1.10 - 1.08, 0) = 0.02.
        # DF at T=0 is 1.0.
        expected = max(1.10 - strike, 0)
        assert pytest.approx(pv, abs=1e-10) == expected

    def test_deep_itm_call(self, market_ids, base_market, pricer):
        """Deep ITM call should be close to DF × (F - K)."""
        spot = 1.10
        r_d, r_f = 0.05, 0.03
        t = 1.0
        forward = spot * math.exp((r_d - r_f) * t)  # ≈ 1.1222
        df = math.exp(-r_d * t)
        strike = 0.90  # Very deep ITM

        option = FxForwardEuropeanOption(
            option_type="call",
            notional=1.0,
            strike=strike,
            expiry=t,
            forward_expiry=t,
            spot_id=market_ids["spot"],
            vol_id=market_ids["vol"],
            domestic_curve_id=market_ids["dom_curve"],
            foreign_curve_id=market_ids["for_curve"],
        )

        pv = pricer.price(option, base_market)
        intrinsic = df * (forward - strike)

        # Deep ITM call should be very close to intrinsic.
        assert pv > intrinsic * 0.99

    def test_deep_otm_call(self, market_ids, base_market, pricer):
        """Deep OTM call should be close to zero."""
        strike = 1.50  # Very deep OTM

        option = FxForwardEuropeanOption(
            option_type="call",
            notional=1.0,
            strike=strike,
            expiry=1.0,
            forward_expiry=1.0,
            spot_id=market_ids["spot"],
            vol_id=market_ids["vol"],
            domestic_curve_id=market_ids["dom_curve"],
            foreign_curve_id=market_ids["for_curve"],
        )

        pv = pricer.price(option, base_market)

        # Deep OTM call should be very small.
        assert pv < 0.001

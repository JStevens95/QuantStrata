"""
Unit tests for equity vanilla options.

Tests instrument construction, validation, and basic properties.

Author: QuantStrata Team
"""

import pytest
from src.instruments.equity.options.vanilla import (
    EquityVanillaEuropeanOption, EquityVanillaAmericanOption,
)
from src.instruments.equity.linear.spot import EquitySpot
from src.instruments.equity.linear.forward import EquityForward
from src.marketdata.core.ids import MarketId


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def spot_id() -> MarketId:
    """Standard AAPL spot market ID."""
    return MarketId(asset_class="EQ", mkt_type="SPOT", name="AAPL")


@pytest.fixture
def vol_id() -> MarketId:
    """Standard AAPL vol market ID."""
    return MarketId(asset_class="EQ", mkt_type="VOL", name="AAPL")


@pytest.fixture
def curve_id() -> MarketId:
    """Standard USD discount curve ID."""
    return MarketId(asset_class="IR", mkt_type="CURVE", name="USD_OIS")


# =============================================================================
# EquitySpot Tests
# =============================================================================

class TestEquitySpot:
    """Tests for EquitySpot instrument."""

    def test_construction_valid(self, spot_id: MarketId):
        """Test valid spot construction."""
        spot = EquitySpot(
            ticker="AAPL",
            quantity=100,
            spot_id=spot_id,
        )
        assert spot.ticker == "AAPL"
        assert spot.quantity == 100
        assert spot.spot_id == spot_id

    def test_construction_negative_quantity(self, spot_id: MarketId):
        """Test short position (negative quantity)."""
        spot = EquitySpot(
            ticker="AAPL",
            quantity=-50,
            spot_id=spot_id,
        )
        assert spot.quantity == -50

    def test_construction_zero_quantity(self, spot_id: MarketId):
        """Test flat position (zero quantity)."""
        spot = EquitySpot(
            ticker="AAPL",
            quantity=0,
            spot_id=spot_id,
        )
        assert spot.quantity == 0

    def test_invalid_ticker_empty(self, spot_id: MarketId):
        """Test that empty ticker raises error."""
        with pytest.raises(ValueError, match="ticker"):
            EquitySpot(ticker="", quantity=100, spot_id=spot_id)

    def test_invalid_spot_id_type(self):
        """Test that invalid spot_id type raises error."""
        with pytest.raises(ValueError, match="spot_id"):
            EquitySpot(ticker="AAPL", quantity=100, spot_id="not_a_market_id")  # type: ignore

    def test_immutable(self, spot_id: MarketId):
        """Test that spot is immutable."""
        spot = EquitySpot(ticker="AAPL", quantity=100, spot_id=spot_id)
        with pytest.raises(AttributeError):
            spot.quantity = 200  # type: ignore


# =============================================================================
# EquityForward Tests
# =============================================================================

class TestEquityForward:
    """Tests for EquityForward instrument."""

    def test_construction_valid(self, spot_id: MarketId, curve_id: MarketId):
        """Test valid forward construction."""
        fwd = EquityForward(
            ticker="AAPL",
            strike=150.0,
            expiry=1.0,
            notional=100,
            dividend_yield=0.005,
            spot_id=spot_id,
            curve_id=curve_id,
        )
        assert fwd.ticker == "AAPL"
        assert fwd.strike == 150.0
        assert fwd.expiry == 1.0
        assert fwd.notional == 100
        assert fwd.dividend_yield == 0.005

    def test_invalid_strike_zero(self, spot_id: MarketId, curve_id: MarketId):
        """Test that zero strike raises error."""
        with pytest.raises(ValueError, match="strike"):
            EquityForward(
                ticker="AAPL", strike=0.0, expiry=1.0, notional=100,
                dividend_yield=0.005, spot_id=spot_id, curve_id=curve_id,
            )

    def test_invalid_strike_negative(self, spot_id: MarketId, curve_id: MarketId):
        """Test that negative strike raises error."""
        with pytest.raises(ValueError, match="strike"):
            EquityForward(
                ticker="AAPL", strike=-10.0, expiry=1.0, notional=100,
                dividend_yield=0.005, spot_id=spot_id, curve_id=curve_id,
            )

    def test_invalid_expiry_negative(self, spot_id: MarketId, curve_id: MarketId):
        """Test that negative expiry raises error."""
        with pytest.raises(ValueError, match="expiry"):
            EquityForward(
                ticker="AAPL", strike=150.0, expiry=-1.0, notional=100,
                dividend_yield=0.005, spot_id=spot_id, curve_id=curve_id,
            )

    def test_invalid_dividend_yield_negative(self, spot_id: MarketId, curve_id: MarketId):
        """Test that negative dividend yield raises error."""
        with pytest.raises(ValueError, match="dividend_yield"):
            EquityForward(
                ticker="AAPL", strike=150.0, expiry=1.0, notional=100,
                dividend_yield=-0.01, spot_id=spot_id, curve_id=curve_id,
            )

    def test_zero_expiry_valid(self, spot_id: MarketId, curve_id: MarketId):
        """Test that zero expiry is valid."""
        fwd = EquityForward(
            ticker="AAPL", strike=150.0, expiry=0.0, notional=100,
            dividend_yield=0.005, spot_id=spot_id, curve_id=curve_id,
        )
        assert fwd.expiry == 0.0


# =============================================================================
# EuropeanEquityVanillaOption Tests
# =============================================================================

class TestEuropeanEquityVanillaOption:
    """Tests for European equity vanilla option."""

    def test_construction_call(self, spot_id: MarketId, vol_id: MarketId, curve_id: MarketId):
        """Test valid call option construction."""
        opt = EquityVanillaEuropeanOption(
            ticker="AAPL",
            option_type="call",
            strike=150.0,
            expiry=1.0,
            notional=100,
            dividend_yield=0.005,
            spot_id=spot_id,
            vol_id=vol_id,
            curve_id=curve_id,
        )
        assert opt.option_type == "call"
        assert opt.strike == 150.0
        assert opt.expiry == 1.0

    def test_construction_put(self, spot_id: MarketId, vol_id: MarketId, curve_id: MarketId):
        """Test valid put option construction."""
        opt = EquityVanillaEuropeanOption(
            ticker="AAPL",
            option_type="put",
            strike=150.0,
            expiry=1.0,
            notional=100,
            dividend_yield=0.005,
            spot_id=spot_id,
            vol_id=vol_id,
            curve_id=curve_id,
        )
        assert opt.option_type == "put"

    def test_invalid_option_type(self, spot_id: MarketId, vol_id: MarketId, curve_id: MarketId):
        """Test that invalid option type raises error."""
        with pytest.raises(ValueError, match="option_type"):
            EquityVanillaEuropeanOption(
                ticker="AAPL", option_type="invalid", strike=150.0, expiry=1.0,
                notional=100, dividend_yield=0.005, spot_id=spot_id,
                vol_id=vol_id, curve_id=curve_id,
            )

    def test_invalid_strike_zero(self, spot_id: MarketId, vol_id: MarketId, curve_id: MarketId):
        """Test that zero strike raises error."""
        with pytest.raises(ValueError, match="strike"):
            EquityVanillaEuropeanOption(
                ticker="AAPL", option_type="call", strike=0.0, expiry=1.0,
                notional=100, dividend_yield=0.005, spot_id=spot_id,
                vol_id=vol_id, curve_id=curve_id,
            )

    def test_invalid_notional_zero(self, spot_id: MarketId, vol_id: MarketId, curve_id: MarketId):
        """Test that zero notional raises error."""
        with pytest.raises(ValueError, match="notional"):
            EquityVanillaEuropeanOption(
                ticker="AAPL", option_type="call", strike=150.0, expiry=1.0,
                notional=0.0, dividend_yield=0.005, spot_id=spot_id,
                vol_id=vol_id, curve_id=curve_id,
            )

    def test_short_position_negative_notional(self, spot_id: MarketId, vol_id: MarketId, curve_id: MarketId):
        """Test that negative notional creates short position."""
        opt = EquityVanillaEuropeanOption(
            ticker="AAPL", option_type="call", strike=150.0, expiry=1.0,
            notional=-100, dividend_yield=0.005, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        assert opt.notional == -100


# =============================================================================
# AmericanEquityVanillaOption Tests
# =============================================================================

class TestAmericanEquityVanillaOption:
    """Tests for American equity vanilla option."""

    def test_construction_call(self, spot_id: MarketId, vol_id: MarketId, curve_id: MarketId):
        """Test valid American call construction."""
        opt = EquityVanillaAmericanOption(
            ticker="AAPL",
            option_type="call",
            strike=150.0,
            expiry=1.0,
            notional=100,
            dividend_yield=0.005,
            spot_id=spot_id,
            vol_id=vol_id,
            curve_id=curve_id,
        )
        assert opt.option_type == "call"

    def test_construction_put(self, spot_id: MarketId, vol_id: MarketId, curve_id: MarketId):
        """Test valid American put construction."""
        opt = EquityVanillaAmericanOption(
            ticker="AAPL",
            option_type="put",
            strike=150.0,
            expiry=1.0,
            notional=100,
            dividend_yield=0.005,
            spot_id=spot_id,
            vol_id=vol_id,
            curve_id=curve_id,
        )
        assert opt.option_type == "put"

    def test_invalid_expiry_negative(self, spot_id: MarketId, vol_id: MarketId, curve_id: MarketId):
        """Test that negative expiry raises error."""
        with pytest.raises(ValueError, match="expiry"):
            EquityVanillaAmericanOption(
                ticker="AAPL", option_type="call", strike=150.0, expiry=-1.0,
                notional=100, dividend_yield=0.005, spot_id=spot_id,
                vol_id=vol_id, curve_id=curve_id,
            )

    def test_immutable(self, spot_id: MarketId, vol_id: MarketId, curve_id: MarketId):
        """Test that option is immutable."""
        opt = EquityVanillaAmericanOption(
            ticker="AAPL", option_type="call", strike=150.0, expiry=1.0,
            notional=100, dividend_yield=0.005, spot_id=spot_id,
            vol_id=vol_id, curve_id=curve_id,
        )
        with pytest.raises(AttributeError):
            opt.strike = 200.0  # type: ignore

"""
Unit tests for FX Vanilla Option instruments.

Tests cover:
1. Construction with valid parameters
2. Validation of invalid parameters
3. Equality and immutability
"""

import pytest
from dataclasses import FrozenInstanceError

from src.instruments.fx.options.vanilla import FxVanillaEuropeanOption
from src.marketdata.core.ids import MarketId


class TestFxVanillaEuropeanOption:
    """Tests for FxVanillaEuropeanOption instrument."""

    @pytest.fixture
    def sample_option(self) -> FxVanillaEuropeanOption:
        """Sample FX vanilla option for testing."""
        return FxVanillaEuropeanOption(
            option_type="call",
            notional=1_000_000.0,
            strike=1.10,
            expiry=0.5,
            spot_id=MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD"),
            vol_id=MarketId(asset_class="FX", mkt_type="VOL", name="EURUSD"),
            domestic_curve_id=MarketId(asset_class="IR", mkt_type="CURVE", name="USD"),
            foreign_curve_id=MarketId(asset_class="IR", mkt_type="CURVE", name="EUR"),
        )

    def test_construction_call(self, sample_option: FxVanillaEuropeanOption) -> None:
        """Test call option construction."""
        assert sample_option.option_type == "call"
        assert sample_option.notional == 1_000_000.0
        assert sample_option.strike == 1.10
        assert sample_option.expiry == 0.5

    def test_construction_put(self) -> None:
        """Test put option construction."""
        option = FxVanillaEuropeanOption(
            option_type="put",
            notional=500_000.0,
            strike=1.05,
            expiry=1.0,
            spot_id=MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD"),
            vol_id=MarketId(asset_class="FX", mkt_type="VOL", name="EURUSD"),
            domestic_curve_id=MarketId(asset_class="IR", mkt_type="CURVE", name="USD"),
            foreign_curve_id=MarketId(asset_class="IR", mkt_type="CURVE", name="EUR"),
        )
        assert option.option_type == "put"

    def test_negative_notional_allowed_for_shorts(self) -> None:
        """Test that negative notional is allowed (represents short position)."""
        option = FxVanillaEuropeanOption(
            option_type="call",
            notional=-100.0,  # Short position
            strike=1.10,
            expiry=0.5,
            spot_id=MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD"),
            vol_id=MarketId(asset_class="FX", mkt_type="VOL", name="EURUSD"),
            domestic_curve_id=MarketId(asset_class="IR", mkt_type="CURVE", name="USD"),
            foreign_curve_id=MarketId(asset_class="IR", mkt_type="CURVE", name="EUR"),
        )
        assert option.notional == -100.0

    def test_negative_strike_raises(self) -> None:
        """Test that negative strike raises error."""
        with pytest.raises(ValueError, match="strike"):
            FxVanillaEuropeanOption(
                option_type="call",
                notional=100.0,
                strike=-1.10,  # Invalid
                expiry=0.5,
                spot_id=MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD"),
                vol_id=MarketId(asset_class="FX", mkt_type="VOL", name="EURUSD"),
                domestic_curve_id=MarketId(asset_class="IR", mkt_type="CURVE", name="USD"),
                foreign_curve_id=MarketId(asset_class="IR", mkt_type="CURVE", name="EUR"),
            )

    def test_negative_expiry_raises(self) -> None:
        """Test that negative expiry raises error."""
        with pytest.raises(ValueError, match="expiry"):
            FxVanillaEuropeanOption(
                option_type="call",
                notional=100.0,
                strike=1.10,
                expiry=-0.5,  # Invalid
                spot_id=MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD"),
                vol_id=MarketId(asset_class="FX", mkt_type="VOL", name="EURUSD"),
                domestic_curve_id=MarketId(asset_class="IR", mkt_type="CURVE", name="USD"),
                foreign_curve_id=MarketId(asset_class="IR", mkt_type="CURVE", name="EUR"),
            )

    def test_frozen_dataclass(self, sample_option: FxVanillaEuropeanOption) -> None:
        """Test that the dataclass is frozen (immutable)."""
        with pytest.raises(FrozenInstanceError):
            sample_option.strike = 1.20  # type: ignore

    def test_equality(self, sample_option: FxVanillaEuropeanOption) -> None:
        """Test equality comparison."""
        option2 = FxVanillaEuropeanOption(
            option_type="call",
            notional=1_000_000.0,
            strike=1.10,
            expiry=0.5,
            spot_id=MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD"),
            vol_id=MarketId(asset_class="FX", mkt_type="VOL", name="EURUSD"),
            domestic_curve_id=MarketId(asset_class="IR", mkt_type="CURVE", name="USD"),
            foreign_curve_id=MarketId(asset_class="IR", mkt_type="CURVE", name="EUR"),
        )
        assert sample_option == option2

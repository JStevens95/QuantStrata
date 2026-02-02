"""
Unit tests for multi-asset instruments.

Tests cover instrument definitions and validation.
"""

import pytest
import numpy as np

from src.marketdata.core.ids import MarketId
from src.instruments.multi_asset import (
    MultiAssetBasketEuropeanOption,
    MultiAssetSpreadEuropeanOption,
    MultiAssetExchangeEuropeanOption,
    MultiAssetBestOfEuropeanOption,
    MultiAssetWorstOfEuropeanOption,
)


def make_id(name: str) -> MarketId:
    """Helper to create MarketId for tests."""
    return MarketId(asset_class="EQ", mkt_type="SPOT", name=name)


# =============================================================================
# MultiAssetBasketEuropeanOption Tests
# =============================================================================

class TestMultiAssetBasketEuropeanOption:
    """Tests for MultiAssetBasketEuropeanOption instrument."""

    def test_valid_call_creation(self):
        """Valid basket call creation."""
        inst = MultiAssetBasketEuropeanOption(
            option_type="call",
            underlyings=(make_id("AAPL"), make_id("GOOGL")),
            weights=(0.5, 0.5),
            strike=100.0,
            expiry=1.0,
        )
        assert inst.n_assets == 2
        assert inst.option_type == "call"

    def test_valid_put_creation(self):
        """Valid basket put creation."""
        inst = MultiAssetBasketEuropeanOption(
            option_type="put",
            underlyings=(make_id("AAPL"), make_id("GOOGL")),
            weights=(0.5, 0.5),
            strike=100.0,
            expiry=1.0,
        )
        assert inst.n_assets == 2
        assert inst.option_type == "put"

    def test_from_lists(self):
        """Create from lists."""
        inst = MultiAssetBasketEuropeanOption.from_lists(
            option_type="call",
            underlyings=[make_id("A"), make_id("B"), make_id("C")],
            weights=[0.4, 0.35, 0.25],
            strike=100.0,
            expiry=0.5,
        )
        assert inst.n_assets == 3

    def test_invalid_option_type_raises(self):
        """Invalid option type should raise."""
        with pytest.raises(ValueError, match="option_type"):
            MultiAssetBasketEuropeanOption(
                option_type="invalid",
                underlyings=(make_id("A"), make_id("B")),
                weights=(0.5, 0.5),
                strike=100.0,
                expiry=1.0,
            )

    def test_single_underlying_raises(self):
        """Single underlying should raise."""
        with pytest.raises(ValueError, match="at least 2"):
            MultiAssetBasketEuropeanOption(
                option_type="call",
                underlyings=(make_id("AAPL"),),
                weights=(1.0,),
                strike=100.0,
                expiry=1.0,
            )

    def test_mismatched_weights_raises(self):
        """Mismatched weights should raise."""
        with pytest.raises(ValueError, match="same length"):
            MultiAssetBasketEuropeanOption(
                option_type="call",
                underlyings=(make_id("A"), make_id("B")),
                weights=(0.5,),
                strike=100.0,
                expiry=1.0,
            )

    def test_negative_strike_raises(self):
        """Negative strike should raise."""
        with pytest.raises(ValueError, match="non-negative"):
            MultiAssetBasketEuropeanOption(
                option_type="call",
                underlyings=(make_id("A"), make_id("B")),
                weights=(0.5, 0.5),
                strike=-10.0,
                expiry=1.0,
            )

    def test_zero_expiry_raises(self):
        """Zero expiry should raise."""
        with pytest.raises(ValueError, match="positive"):
            MultiAssetBasketEuropeanOption(
                option_type="call",
                underlyings=(make_id("A"), make_id("B")),
                weights=(0.5, 0.5),
                strike=100.0,
                expiry=0.0,
            )


# =============================================================================
# MultiAssetSpreadEuropeanOption Tests
# =============================================================================

class TestMultiAssetSpreadEuropeanOption:
    """Tests for MultiAssetSpreadEuropeanOption instrument."""

    def test_valid_call_creation(self):
        """Valid spread call creation."""
        inst = MultiAssetSpreadEuropeanOption(
            option_type="call",
            underlying1=make_id("CL"),
            underlying2=make_id("HO"),
            strike=5.0,
            expiry=0.5,
        )
        assert inst.expiry == 0.5
        assert inst.option_type == "call"

    def test_valid_put_creation(self):
        """Valid spread put creation."""
        inst = MultiAssetSpreadEuropeanOption(
            option_type="put",
            underlying1=make_id("CL"),
            underlying2=make_id("HO"),
            strike=5.0,
            expiry=0.5,
        )
        assert inst.option_type == "put"

    def test_invalid_option_type_raises(self):
        """Invalid option type should raise."""
        with pytest.raises(ValueError, match="option_type"):
            MultiAssetSpreadEuropeanOption(
                option_type="invalid",
                underlying1=make_id("A"),
                underlying2=make_id("B"),
                strike=5.0,
                expiry=0.5,
            )

    def test_zero_expiry_raises(self):
        """Zero expiry should raise."""
        with pytest.raises(ValueError, match="positive"):
            MultiAssetSpreadEuropeanOption(
                option_type="call",
                underlying1=make_id("A"),
                underlying2=make_id("B"),
                strike=0.0,
                expiry=0.0,
            )


# =============================================================================
# MultiAssetExchangeEuropeanOption Tests
# =============================================================================

class TestMultiAssetExchangeEuropeanOption:
    """Tests for MultiAssetExchangeEuropeanOption instrument."""

    def test_valid_creation(self):
        """Valid exchange option creation."""
        inst = MultiAssetExchangeEuropeanOption(
            underlying1=make_id("A"),
            underlying2=make_id("B"),
            expiry=1.0,
        )
        assert inst.expiry == 1.0

    def test_zero_expiry_raises(self):
        """Zero expiry should raise."""
        with pytest.raises(ValueError, match="positive"):
            MultiAssetExchangeEuropeanOption(
                underlying1=make_id("A"),
                underlying2=make_id("B"),
                expiry=0.0,
            )


# =============================================================================
# MultiAssetBestOfEuropeanOption Tests
# =============================================================================

class TestMultiAssetBestOfEuropeanOption:
    """Tests for MultiAssetBestOfEuropeanOption instrument."""

    def test_valid_call_creation(self):
        """Valid best-of call creation."""
        inst = MultiAssetBestOfEuropeanOption(
            option_type="call",
            underlyings=(make_id("A"), make_id("B"), make_id("C")),
            strike=100.0,
            expiry=1.0,
        )
        assert inst.n_assets == 3
        assert inst.option_type == "call"

    def test_valid_put_creation(self):
        """Valid best-of put creation."""
        inst = MultiAssetBestOfEuropeanOption(
            option_type="put",
            underlyings=(make_id("A"), make_id("B")),
            strike=100.0,
            expiry=1.0,
        )
        assert inst.option_type == "put"

    def test_from_list(self):
        """Create from list."""
        inst = MultiAssetBestOfEuropeanOption.from_list(
            option_type="call",
            underlyings=[make_id("A"), make_id("B")],
            strike=100.0,
            expiry=0.5,
        )
        assert inst.n_assets == 2

    def test_invalid_option_type_raises(self):
        """Invalid option type should raise."""
        with pytest.raises(ValueError, match="option_type"):
            MultiAssetBestOfEuropeanOption(
                option_type="invalid",
                underlyings=(make_id("A"), make_id("B")),
                strike=100.0,
                expiry=1.0,
            )

    def test_single_underlying_raises(self):
        """Single underlying should raise."""
        with pytest.raises(ValueError, match="at least 2"):
            MultiAssetBestOfEuropeanOption(
                option_type="call",
                underlyings=(make_id("A"),),
                strike=100.0,
                expiry=1.0,
            )


# =============================================================================
# MultiAssetWorstOfEuropeanOption Tests
# =============================================================================

class TestMultiAssetWorstOfEuropeanOption:
    """Tests for MultiAssetWorstOfEuropeanOption instrument."""

    def test_valid_call_creation(self):
        """Valid worst-of call creation."""
        inst = MultiAssetWorstOfEuropeanOption(
            option_type="call",
            underlyings=(make_id("A"), make_id("B")),
            strike=100.0,
            expiry=1.0,
        )
        assert inst.n_assets == 2
        assert inst.option_type == "call"

    def test_valid_put_creation(self):
        """Valid worst-of put creation."""
        inst = MultiAssetWorstOfEuropeanOption(
            option_type="put",
            underlyings=(make_id("A"), make_id("B")),
            strike=100.0,
            expiry=1.0,
        )
        assert inst.option_type == "put"

    def test_from_list(self):
        """Create from list."""
        inst = MultiAssetWorstOfEuropeanOption.from_list(
            option_type="put",
            underlyings=[make_id("A"), make_id("B"), make_id("C")],
            strike=100.0,
            expiry=0.5,
        )
        assert inst.n_assets == 3

    def test_invalid_option_type_raises(self):
        """Invalid option type should raise."""
        with pytest.raises(ValueError, match="option_type"):
            MultiAssetWorstOfEuropeanOption(
                option_type="invalid",
                underlyings=(make_id("A"), make_id("B")),
                strike=100.0,
                expiry=1.0,
            )

    def test_single_underlying_raises(self):
        """Single underlying should raise."""
        with pytest.raises(ValueError, match="at least 2"):
            MultiAssetWorstOfEuropeanOption(
                option_type="call",
                underlyings=(make_id("A"),),
                strike=100.0,
                expiry=1.0,
            )

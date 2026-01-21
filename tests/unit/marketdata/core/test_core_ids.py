from __future__ import annotations

import pytest

from src.marketdata.core.ids import MarketId


def test_marketid_key_without_qualifiers() -> None:
    mid = MarketId(asset_class="fx", mkt_type="spot", name="eurusd")
    assert mid.key() == "FX.SPOT.eurusd"


def test_marketid_key_with_qualifiers_preserves_order() -> None:
    mid = MarketId(
        asset_class="IR",
        mkt_type="FIXING",
        name="USD.SOFR",
        qualifiers=(("tenor", "3M"), ("cut", "NY")),
    )
    assert mid.key() == "IR.FIXING.USD.SOFR|tenor=3M|cut=NY"


def test_marketid_parse_roundtrip() -> None:
    raw = "FX.VOL.EURUSD|cut=LDN|convention=delta25"
    mid = MarketId.parse(raw)
    assert mid.key() == "FX.VOL.EURUSD|cut=LDN|convention=delta25"


def test_marketid_with_qualifier_appends() -> None:
    mid = MarketId("FX", "VOL", "EURUSD")
    mid2 = mid.with_qualifier("cut", "LDN")
    assert mid2.key() == "FX.VOL.EURUSD|cut=LDN"


def test_marketid_empty_qualifier_key_raises() -> None:
    with pytest.raises(ValueError, match="qualifier key must be non-empty"):
        MarketId("FX", "SPOT", "EURUSD", qualifiers=(("", "x"),))
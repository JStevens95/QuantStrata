# tests/unit/marketdata/test_market_id.py

from __future__ import annotations

import pytest

from src.marketdata.ids import MarketId


def test_market_id_normalization_and_key_roundtrip() -> None:
    mid = MarketId(asset_class=" fx ", mkt_type=" spot ", name=" EURUSD ")
    assert mid.asset_class == "FX"
    assert mid.mkt_type == "SPOT"
    assert mid.name == "EURUSD"

    key = mid.key()
    assert key == "FX.SPOT.EURUSD"

    mid2 = MarketId.parse(key)
    assert mid2 == mid


def test_market_id_qualifiers_in_key_and_roundtrip() -> None:
    mid = MarketId("IR", "FIXING", "USD.SOFR").with_qualifier("tenor", "3M").with_qualifier("source", "TEST")
    key = mid.key()
    assert key == "IR.FIXING.USD.SOFR|tenor=3M|source=TEST"

    mid2 = MarketId.parse(key)
    assert mid2 == mid


@pytest.mark.parametrize(
    "text",
    [
        "",                          # empty
        "FX.SPOT",                   # missing name
        "FX.SPOT.EURUSD|badqual",     # qualifier must be k=v
        "FX.SPOT.EURUSD|=x",          # empty key
    ],
)
def test_market_id_parse_invalid_raises(text: str) -> None:
    with pytest.raises(ValueError):
        MarketId.parse(text)
# tests/unit/marketdata/core/test_market.py

import numpy as np
import pytest

from src.marketdata.core.ids import MarketId
from src.marketdata.core.interfaces import Quote
from src.marketdata.core.market import Market


class DummyCurve:
    def df(self, t: float) -> float:
        return float(np.exp(-0.01 * float(t)))

    def zero_rate(self, t: float) -> float:
        return 0.01

    def forward_rate(self, t1: float, t2: float) -> float:
        return 0.01


class DummyVol:
    def implied_vol(self, expiry: float, strike: float) -> float:
        return 0.2

    def vol(self, expiry: float, strike: float) -> float:
        return self.implied_vol(expiry, strike)


def test_market_accessors_work() -> None:
    spot_id = MarketId("FX", "SPOT", "EURUSD")
    c_id = MarketId("IR", "CURVE", "USD.OIS")
    v_id = MarketId("FX", "VOL", "EURUSD")

    m = Market(
        asof="2026-01-07",
        quotes={spot_id: Quote(1.10)},
        curves={c_id: DummyCurve()},
        vols={v_id: DummyVol()},
    )

    assert m.quote(spot_id) == 1.10
    assert m.curve(c_id) is m.curves[c_id]
    assert m.vol_surface(v_id) is m.vols[v_id]
    assert m.has(spot_id)
    assert m.has(c_id)
    assert m.has(v_id)


def test_market_missing_quote_raises_keyerror() -> None:
    spot_id = MarketId("FX", "SPOT", "EURUSD")
    missing = MarketId("FX", "SPOT", "GBPUSD")

    m = Market(asof="2026-01-07", quotes={spot_id: Quote(1.10)}, curves={}, vols={})

    with pytest.raises(KeyError) as exc:
        _ = m.quote(missing)
    assert missing.key() in str(exc.value)
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pytest

from src.marketdata.core.interfaces import Quote
from src.marketdata.core.market import Market
from src.marketdata.core.ids import MarketId
from src.marketdata.scenarios.interfaces import ScenarioPack
from src.marketdata.scenarios.shocks import SpotShock


def _make_market_id(key: str) -> MarketId:
    """
    Helper that tries a few common MarketId construction patterns so tests
    don't depend on a single MarketId API choice.
    """
    # 1) classmethod constructors
    for attr in ("parse", "from_key", "from_string"):
        fn = getattr(MarketId, attr, None)
        if callable(fn):
            try:
                return fn(key)  # type: ignore[misc]
            except Exception:
                pass

    # 2) direct string constructor: MarketId("...")
    try:
        return MarketId(key)  # type: ignore[call-arg]
    except Exception:
        pass

    # 3) dataclass-like constructor variants
    for kwargs in (
        {"key": key},
        {"name": key},
        {"raw": key},
    ):
        try:
            return MarketId(**kwargs)  # type: ignore[arg-type]
        except Exception:
            pass

    raise RuntimeError(
        f"Could not construct MarketId from key={key!r}. "
        "Update _make_market_id() to match your MarketId API."
    )


@dataclass(frozen=True, slots=True)
class _FlatCurve:
    r: float

    def df(self, t: float) -> float:
        import math
        t = float(t)
        return float(math.exp(-self.r * t))

    def zero_rate(self, t: float) -> float:
        return float(self.r)

    def forward_rate(self, t1: float, t2: float) -> float:
        return float(self.r)


@dataclass(frozen=True, slots=True)
class _FlatVolSurface:
    sigma: float

    def implied_vol(self, expiry: float, strike: float) -> float:
        return float(self.sigma)

    def vol(self, expiry: float, strike: float) -> float:
        return float(self.sigma)


def _make_base_market() -> tuple[Market, Dict[str, Any]]:
    spot_id = _make_market_id("FX.SPOT.EURUSD")
    other_quote_id = _make_market_id("FX.SPOT.GBPUSD")
    curve_id = _make_market_id("IR.CURVE.USD.OIS")
    vol_id = _make_market_id("FX.VOL.EURUSD")

    base = Market(
        asof="2026-01-01",
        quotes={
            spot_id: Quote(100.0),
            other_quote_id: Quote(200.0),
        },
        curves={curve_id: _FlatCurve(r=0.02)},
        vols={vol_id: _FlatVolSurface(sigma=0.20)},
        meta={"unit_test": True},
    )
    return base, {
        "spot_id": spot_id,
        "other_quote_id": other_quote_id,
        "curve_id": curve_id,
        "vol_id": vol_id,
    }


def test_scenario_pack_apply_all_returns_named_views() -> None:
    base, ids = _make_base_market()

    pack = ScenarioPack(
        scenarios={
            "spot_up_10pct": SpotShock(
                name="spot_up_10pct",
                spot_id=ids["spot_id"],
                bump=0.10,
                bump_mode="relative",
            )
        }
    )

    shocked = pack.apply_all(base)
    assert set(shocked.keys()) == {"spot_up_10pct"}

    m = shocked["spot_up_10pct"]
    assert pytest.approx(m.quote(ids["spot_id"])) == 110.0
    # Ensure other quote untouched
    assert pytest.approx(m.quote(ids["other_quote_id"])) == 200.0
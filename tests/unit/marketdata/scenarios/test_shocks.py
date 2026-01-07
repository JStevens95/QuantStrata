from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pytest

from src.marketdata.core.interfaces import Quote
from src.marketdata.core.market import Market
from src.marketdata.core.ids import MarketId

from src.marketdata.scenarios.shocks import (
    SpotShock,
    VolShock,
    ParallelRateShock,
)


def _make_market_id(key: str) -> MarketId:
    # Keep in sync with test_base.py (duplicated intentionally to avoid test coupling).
    for attr in ("parse", "from_key", "from_string"):
        fn = getattr(MarketId, attr, None)
        if callable(fn):
            try:
                return fn(key)  # type: ignore[misc]
            except Exception:
                pass

    try:
        return MarketId(key)  # type: ignore[call-arg]
    except Exception:
        pass

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
    name: str = "BASE_CURVE"

    def df(self, t: float) -> float:
        import math
        t = float(t)
        if t < 0.0:
            raise ValueError("t must be >= 0")
        return float(math.exp(-self.r * t))

    def zero_rate(self, t: float) -> float:
        return float(self.r)

    def forward_rate(self, t1: float, t2: float) -> float:
        return float(self.r)


@dataclass(frozen=True, slots=True)
class _FlatVolSurface:
    sigma: float
    surface_name: str = "BASE_SURFACE"

    def implied_vol(self, expiry: float, strike: float) -> float:
        return float(self.sigma)

    def vol(self, expiry: float, strike: float) -> float:
        return float(self.sigma)


def _make_base_market() -> tuple[Market, Dict[str, Any]]:
    spot_id = _make_market_id("FX.SPOT.EURUSD")
    curve_id = _make_market_id("IR.CURVE.USD.OIS")
    vol_id = _make_market_id("FX.VOL.EURUSD")

    base = Market(
        asof="2026-01-01",
        quotes={spot_id: Quote(100.0)},
        curves={curve_id: _FlatCurve(r=0.02)},
        vols={vol_id: _FlatVolSurface(sigma=0.20)},
        meta=None,
    )
    return base, {"spot_id": spot_id, "curve_id": curve_id, "vol_id": vol_id}


def test_spot_shock_relative() -> None:
    base, ids = _make_base_market()

    shocked = SpotShock(
        name="spot_up_10pct",
        spot_id=ids["spot_id"],
        bump=0.10,
        bump_mode="relative",
    ).apply(base)

    assert pytest.approx(shocked.quote(ids["spot_id"])) == 110.0
    # Non-overridden components still resolve
    _ = shocked.curve(ids["curve_id"])
    _ = shocked.vol_surface(ids["vol_id"])


def test_spot_shock_absolute() -> None:
    base, ids = _make_base_market()

    shocked = SpotShock(
        name="spot_up_5",
        spot_id=ids["spot_id"],
        bump=5.0,
        bump_mode="absolute",
    ).apply(base)

    assert pytest.approx(shocked.quote(ids["spot_id"])) == 105.0


def test_spot_shock_invalid_mode_raises() -> None:
    base, ids = _make_base_market()
    with pytest.raises(ValueError, match="bump_mode"):
        _ = SpotShock(
            name="bad",
            spot_id=ids["spot_id"],
            bump=0.1,
            bump_mode="nope",  # type: ignore[arg-type]
        ).apply(base)


def test_vol_shock_relative_and_alias_vol() -> None:
    base, ids = _make_base_market()

    shocked = VolShock(
        name="vol_up_10pct",
        vol_id=ids["vol_id"],
        bump=0.10,
        bump_mode="relative",
        vol_floor=1e-8,
    ).apply(base)

    surf = shocked.vol_surface(ids["vol_id"])
    assert pytest.approx(surf.implied_vol(1.0, 100.0)) == 0.22
    # Alias must match interface
    assert pytest.approx(surf.vol(1.0, 100.0)) == 0.22

    # Delegation: base surface metadata should still be visible
    assert getattr(surf, "surface_name") == "BASE_SURFACE"


def test_vol_shock_absolute_with_floor() -> None:
    base, ids = _make_base_market()

    shocked = VolShock(
        name="vol_down_big",
        vol_id=ids["vol_id"],
        bump=-1.0,
        bump_mode="absolute",
        vol_floor=1e-6,
    ).apply(base)

    surf = shocked.vol_surface(ids["vol_id"])
    assert pytest.approx(surf.implied_vol(1.0, 100.0)) == 1e-6


def test_vol_shock_invalid_mode_raises() -> None:
    base, ids = _make_base_market()
    with pytest.raises(ValueError, match="bump_mode"):
        _ = VolShock(
            name="bad",
            vol_id=ids["vol_id"],
            bump=0.1,
            bump_mode="nope",  # type: ignore[arg-type]
        ).apply(base)


def test_parallel_rate_shock_df_zero_forward_and_delegation() -> None:
    base, ids = _make_base_market()

    shocked = ParallelRateShock(
        name="rates_up_100bp",
        curve_id=ids["curve_id"],
        rate_shift=0.01,
    ).apply(base)

    c = shocked.curve(ids["curve_id"])

    # base r=0.02, shift=0.01 => effective r=0.03
    assert pytest.approx(c.df(1.0), rel=1e-12) == pytest.approx(base.curve(ids["curve_id"]).df(1.0) * 0.99004983375, rel=1e-9)
    assert pytest.approx(c.zero_rate(5.0), rel=1e-12) == 0.03
    assert pytest.approx(c.forward_rate(1.0, 2.0), rel=1e-12) == 0.03

    # Delegation for curve-specific attributes
    assert getattr(c, "name") == "BASE_CURVE"


def test_shocks_compose_sequentially() -> None:
    base, ids = _make_base_market()

    m1 = SpotShock(
        name="spot_up_10pct",
        spot_id=ids["spot_id"],
        bump=0.10,
        bump_mode="relative",
    ).apply(base)

    m2 = VolShock(
        name="vol_up_10pct",
        vol_id=ids["vol_id"],
        bump=0.10,
        bump_mode="relative",
    ).apply(m1)

    assert pytest.approx(m2.quote(ids["spot_id"])) == 110.0
    surf = m2.vol_surface(ids["vol_id"])
    assert pytest.approx(surf.implied_vol(1.0, 100.0)) == 0.22
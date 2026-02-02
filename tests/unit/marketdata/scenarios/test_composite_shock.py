"""Unit tests for CompositeShock (multi-factor scenario)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.marketdata.core.ids import MarketId
from src.marketdata.scenarios.interfaces import MarketView, ScenarioShock
from src.marketdata.scenarios.shocks import CompositeShock, ParallelRateShock, SpotShock, VolShock


@dataclass
class _FakeMarket:
    """Minimal MarketView for testing."""

    spot: float = 1.1
    vol: float = 0.15
    rate: float = 0.05

    def quote(self, market_id: MarketId) -> float:
        if "SPOT" in str(market_id):
            return self.spot
        return 0.0

    def curve(self, market_id: MarketId):
        return None

    def vol_surface(self, market_id: MarketId):
        return None


def test_composite_shock_applies_in_order() -> None:
    spot_id = MarketId.parse("FX.SPOT.EURUSD")
    shock1 = SpotShock(name="s1", spot_id=spot_id, bump=0.10, bump_mode="relative")
    shock2 = SpotShock(name="s2", spot_id=spot_id, bump=-0.05, bump_mode="relative")
    composite = CompositeShock(name="combo", shocks=[shock1, shock2])
    base = _FakeMarket(spot=1.0)
    result = composite.apply(base)
    # First: 1.0 * 1.10 = 1.10; second: 1.10 * 0.95 = 1.045
    assert result.quote(spot_id) == pytest.approx(1.045)


def test_composite_shock_single_shock() -> None:
    spot_id = MarketId.parse("FX.SPOT.EURUSD")
    single = SpotShock(name="s", spot_id=spot_id, bump=0.01, bump_mode="relative")
    composite = CompositeShock(name="wrap", shocks=[single])
    base = _FakeMarket(spot=1.0)
    result = composite.apply(base)
    assert result.quote(spot_id) == pytest.approx(1.01)


def test_composite_shock_implements_protocol() -> None:
    spot_id = MarketId.parse("FX.SPOT.EURUSD")
    composite = CompositeShock(
        name="c",
        shocks=[SpotShock("s", spot_id=spot_id, bump=0.0, bump_mode="relative")],
    )
    assert isinstance(composite, ScenarioShock)
    assert composite.name == "c"
    assert hasattr(composite, "apply")

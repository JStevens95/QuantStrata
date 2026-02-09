"""Unit tests for scenario generation (presets and historical)."""

from __future__ import annotations

import numpy as np
import pytest

from src.marketdata.core.ids import MarketId
from src.marketdata.scenarios.interfaces import ScenarioPack
from src.marketdata.scenarios.shocks import CompositeShock
from src.risk.scenarios.generation import (
    composite_from_preset,
    preset_stress_pack,
    shocks_from_historical_series,
)


@pytest.fixture
def spot_id() -> MarketId:
    return MarketId.parse("FX.SPOT.EURUSD")


@pytest.fixture
def vol_id() -> MarketId:
    return MarketId.parse("FX.VOL.EURUSD")


@pytest.fixture
def domestic_curve_id() -> MarketId:
    return MarketId.parse("IR.CURVE.USD.OIS")


def test_preset_spot_down_10(spot_id: MarketId) -> None:
    pack = preset_stress_pack("spot_down_10", spot_id=spot_id)
    assert isinstance(pack, ScenarioPack)
    assert "spot_down_10" in pack.scenarios
    shock = pack.scenarios["spot_down_10"]
    assert shock.name == "spot_down_10"
    assert hasattr(shock, "apply")


def test_preset_spot_down_10_missing_spot_raises() -> None:
    with pytest.raises(ValueError, match="requires spot_id"):
        preset_stress_pack("spot_down_10")


def test_preset_crisis_style(spot_id: MarketId, vol_id: MarketId, domestic_curve_id: MarketId) -> None:
    pack = preset_stress_pack(
        "crisis_style",
        spot_id=spot_id,
        vol_id=vol_id,
        domestic_curve_id=domestic_curve_id,
    )
    assert "crisis_style" in pack.scenarios
    shock = pack.scenarios["crisis_style"]
    assert isinstance(shock, CompositeShock)
    assert len(shock.shocks) == 3


def test_preset_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown preset name"):
        preset_stress_pack("unknown_preset", spot_id=MarketId.parse("FX.SPOT.EURUSD"))


def test_shocks_from_historical_series() -> None:
    spot_id = MarketId.parse("FX.SPOT.EURUSD")
    # Simulate 100 days of spot; worst 1-day move at 5th percentile
    np.random.seed(1)
    series = 1.1 + np.cumsum(np.random.randn(100) * 0.01)
    series_by_id = {spot_id: series}
    shocks = shocks_from_historical_series(series_by_id, percentile=5.0, use_relative=True)
    assert len(shocks) == 1
    assert shocks[0].name.startswith("hist_spot_")


def test_shocks_from_historical_series_short_series() -> None:
    spot_id = MarketId.parse("FX.SPOT.EURUSD")
    series_by_id = {spot_id: np.array([1.0, 1.01])}  # only 2 points, horizon=1 needs 2+
    shocks = shocks_from_historical_series(series_by_id, horizon=1)
    assert len(shocks) == 1


def test_composite_from_preset(spot_id: MarketId) -> None:
    composite = composite_from_preset("spot_down_10", spot_id=spot_id)
    assert isinstance(composite, CompositeShock)
    assert len(composite.shocks) == 1

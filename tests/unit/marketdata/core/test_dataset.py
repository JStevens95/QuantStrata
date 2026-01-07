# tests/unit/marketdata/core/test_dataset.py

from __future__ import annotations

import numpy as np
import pytest

from src.marketdata.core.dataset import MarketDataset
from src.marketdata.core.ids import MarketId
from src.marketdata.core.panel import Panel


class DummyCurve:
    def __init__(self, params: np.ndarray) -> None:
        self.params = np.asarray(params, dtype=float)

    def df(self, t: float) -> float:
        return 1.0

    def zero_rate(self, t: float) -> float:
        return 0.0

    def forward_rate(self, t1: float, t2: float) -> float:
        return 0.0


class DummyVol:
    def __init__(self, params: np.ndarray) -> None:
        self.params = np.asarray(params, dtype=float)

    def implied_vol(self, expiry: float, strike: float) -> float:
        return float(self.params.reshape(-1)[0])

    def vol(self, expiry: float, strike: float) -> float:
        return self.implied_vol(expiry, strike)


class DummyCurveFactory:
    def build(self, params: np.ndarray) -> DummyCurve:
        return DummyCurve(params=params)


class DummyVolFactory:
    def build(self, params: np.ndarray) -> DummyVol:
        return DummyVol(params=params)


def test_dataset_snapshot_builds_market_correctly() -> None:
    # ---- IDs ----
    spot_id = MarketId("FX", "SPOT", "EURUSD")
    curve_id = MarketId("IR", "CURVE", "USD.OIS")
    vol_id = MarketId("FX", "VOL", "EURUSD")

    # ---- Data shapes ----
    # T=2, S=2
    dates = ["2026-01-06", "2026-01-07"]
    n_scenarios = 2

    # Quote panel [T,S]
    spot_panel = Panel(
        data=np.array([[1.10, 1.11], [1.20, 1.21]], dtype=float),
        axis_names=("time", "scenario"),
    )

    # Curve params panel [T,S,K] (block params)
    curve_params_panel = Panel(
        data=np.array(
            [
                [[0.01, 0.02, 0.03], [0.11, 0.12, 0.13]],
                [[0.21, 0.22, 0.23], [0.31, 0.32, 0.33]],
            ],
            dtype=float,
        ),
        axis_names=("time", "scenario", "k"),
    )

    # Vol params panel [T] (scalar-like params)
    vol_params_panel = Panel(
        data=np.array([0.20, 0.25], dtype=float),
        axis_names=("time",),
    )

    ds = MarketDataset(
        dates=dates,
        n_scenarios=n_scenarios,
        panels={spot_id: spot_panel},
        curve_params={curve_id: curve_params_panel},
        curve_factories={curve_id: DummyCurveFactory()},
        vol_params={vol_id: vol_params_panel},
        vol_factories={vol_id: DummyVolFactory()},
    )

    m = ds.snapshot(time_idx=1, scenario_idx=0)

    assert m.asof == "2026-01-07"
    assert m.quote(spot_id) == 1.20

    curve = m.curve(curve_id)
    assert isinstance(curve, DummyCurve)
    # time_idx=1, scenario_idx=0 -> params should be [0.21,0.22,0.23]
    np.testing.assert_allclose(curve.params, np.array([0.21, 0.22, 0.23], dtype=float))

    vol = m.vol_surface(vol_id)
    assert isinstance(vol, DummyVol)
    # vol params at time_idx=1 -> 0.25
    assert vol.implied_vol(1.0, 1.0) == 0.25


def test_dataset_missing_factories_raises() -> None:
    curve_id = MarketId("IR", "CURVE", "USD.OIS")

    with pytest.raises(ValueError, match="Missing curve factories"):
        MarketDataset(
            dates=["2026-01-07"],
            n_scenarios=1,
            panels={},
            curve_params={curve_id: Panel(np.array([[1.0]]), ("time", "k"))},
            curve_factories={},  # missing
            vol_params={},
            vol_factories={},
        )


def test_dataset_missing_curve_factory_raises_on_init() -> None:
    curve_id = MarketId("IR", "CURVE", "USD.OIS")
    with pytest.raises(ValueError, match="Missing curve factories"):
        MarketDataset(
            dates=["2026-01-07"],
            n_scenarios=1,
            panels={},
            curve_params={curve_id: Panel(np.array([[1.0]]), ("time", "k"))},
            curve_factories={},
            vol_params={},
            vol_factories={},
        )


def test_dataset_quote_panel_wrong_ndim_raises() -> None:
    spot_id = MarketId("FX", "SPOT", "EURUSD")

    bad_quote_panel = Panel(
        data=np.zeros((2, 2, 2), dtype=float),
        axis_names=("time", "scenario", "x"),
    )

    with pytest.raises(ValueError, match="Quote panel must have ndim 1 or 2"):
        MarketDataset(
            dates=["2026-01-06", "2026-01-07"],
            n_scenarios=2,
            panels={spot_id: bad_quote_panel},
            curve_params={},
            curve_factories={},
            vol_params={},
            vol_factories={},
        )


def test_dataset_time_axis_mismatch_raises() -> None:
    spot_id = MarketId("FX", "SPOT", "EURUSD")

    # dates length = 2, but panel has T=3
    spot_panel = Panel(
        data=np.array([1.0, 2.0, 3.0], dtype=float),
        axis_names=("time",),
    )

    with pytest.raises(ValueError, match="time axis mismatch"):
        MarketDataset(
            dates=["2026-01-06", "2026-01-07"],
            n_scenarios=1,
            panels={spot_id: spot_panel},
            curve_params={},
            curve_factories={},
            vol_params={},
            vol_factories={},
        )


def test_dataset_scenario_axis_mismatch_raises() -> None:
    spot_id = MarketId("FX", "SPOT", "EURUSD")

    # n_scenarios = 2, but panel has S=3
    spot_panel = Panel(
        data=np.array([[1.0, 2.0, 3.0]], dtype=float),
        axis_names=("time", "scenario"),
    )

    with pytest.raises(ValueError, match="scenario axis mismatch"):
        MarketDataset(
            dates=["2026-01-07"],
            n_scenarios=2,
            panels={spot_id: spot_panel},
            curve_params={},
            curve_factories={},
            vol_params={},
            vol_factories={},
        )
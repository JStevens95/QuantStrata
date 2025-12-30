from __future__ import annotations

import pytest
import numpy as np

from src.marketdata.ids import MarketId
from src.marketdata.interfaces import Panel
from src.marketdata.dataset import MarketDataset
from src.marketdata.curves.factories import FlatCurveFactory


def test_marketdataset_snapshot_builds_curve_from_panel_block() -> None:
    # MarketId for the curve we want to build
    curve_id = MarketId(asset_class="IR", mkt_type="CURVE", name="USD.OIS")

    # Store a flat rate time series as a curve parameter panel.
    # axis_names follow your convention: first axis time, optional scenario.
    # For a flat curve factory, the parameter block at time t is a scalar.
    curve_rate_series = np.array([0.01, 0.015, 0.02], dtype=float)  # T=3
    curve_panel = Panel(data=curve_rate_series, axis_names=("time",))

    dataset = MarketDataset(
        dates=["2025-01-01", "2025-01-02", "2025-01-03"],
        n_scenarios=1,
        panels={},  # no scalar quotes needed for this test
        curve_params={curve_id: curve_panel},
        curve_factories={curve_id: FlatCurveFactory()},
        vol_params={},
        vol_factories={},
        meta=None,
    )

    market_t1 = dataset.snapshot(time_idx=1, scenario_idx=0)
    curve = market_t1.curve(curve_id)

    # For the rate at time_idx=1 we stored 0.015
    assert curve.zero_rate(0.5) == pytest.approx(0.015)
    assert curve.df(1.0) == pytest.approx(float(np.exp(-0.015 * 1.0)))

from __future__ import annotations

import pytest
import numpy as np

from src.marketdata.ids import MarketId
from src.marketdata.interfaces import Panel
from src.marketdata.dataset import MarketDataset
from src.marketdata.curves.factories import FlatCurveFactory


def test_snapshot_uses_scenario_axis_for_scalar_params() -> None:
    curve_id = MarketId("IR", "CURVE", "USD.OIS")

    # Shape [T,S] with axis_names indicating scenario.
    rates = np.asarray(
        [
            [0.01, 0.02],
            [0.015, 0.025],
            [0.02, 0.03],
        ],
        dtype=float,
    )
    curve_panel = Panel(data=rates, axis_names=("time", "scenario"))

    ds = MarketDataset(
        dates=["2025-01-01", "2025-01-02", "2025-01-03"],
        n_scenarios=2,
        panels={},
        curve_params={curve_id: curve_panel},
        curve_factories={curve_id: FlatCurveFactory()},
        vol_params={},
        vol_factories={},
    )

    m0 = ds.snapshot(time_idx=1, scenario_idx=0)
    m1 = ds.snapshot(time_idx=1, scenario_idx=1)

    assert m0.curve(curve_id).zero_rate(1.0) == pytest.approx(0.015)
    assert m1.curve(curve_id).zero_rate(1.0) == pytest.approx(0.025)
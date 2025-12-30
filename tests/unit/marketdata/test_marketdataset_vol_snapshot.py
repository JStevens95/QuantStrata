from __future__ import annotations

import pytest
import numpy as np

from src.marketdata.ids import MarketId
from src.marketdata.interfaces import Panel
from src.marketdata.dataset import MarketDataset
from src.marketdata.surfaces.factories import FlatVolFactory


def test_marketdataset_snapshot_builds_volsurface_from_panel_block() -> None:
    vol_id = MarketId(asset_class="FX", mkt_type="VOL", name="EURUSD")

    # Store a flat vol time series as [T]. Your dataset slicing must accept ndim=1.
    vol_series = np.asarray([0.10, 0.11, 0.12], dtype=float)  # T=3
    vol_panel = Panel(data=vol_series, axis_names=("time",))

    ds = MarketDataset(
        dates=["2025-01-01", "2025-01-02", "2025-01-03"],
        n_scenarios=1,
        panels={},  # no scalar quotes needed here
        curve_params={},
        curve_factories={},
        vol_params={vol_id: vol_panel},
        vol_factories={vol_id: FlatVolFactory()},
    )

    market_t1 = ds.snapshot(time_idx=1, scenario_idx=0)
    surface = market_t1.vol_surface(vol_id)

    assert surface.vol(expiry=1.0, strike=1.0) == pytest.approx(0.11)
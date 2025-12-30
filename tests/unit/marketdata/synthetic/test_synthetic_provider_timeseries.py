from __future__ import annotations

from src.marketdata.ids import MarketId
from src.marketdata.requests import TimeseriesRequest, Universe
from src.marketdata.synthetic.provider import SyntheticProvider


def test_get_timeseries_returns_dataset_and_snapshot_works() -> None:
    """
    Ensure SyntheticProvider.get_timeseries() returns a MarketDataset with:
    - correct time axis (dates)
    - correct scenario count
    - a working snapshot() that reconstructs Market objects.
    """
    spot_id = MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD")
    vol_id = MarketId(asset_class="FX", mkt_type="VOL", name="EURUSD")
    curve_id = MarketId(asset_class="IR", mkt_type="CURVE", name="USD.OIS")

    provider = SyntheticProvider(seed=7)

    dataset = provider.get_timeseries(
        TimeseriesRequest(
            start="2025-01-01",
            end="2025-01-10",
            freq="B",
            universe=Universe([spot_id, vol_id, curve_id]),
            scenarios=2,
        )
    )

    assert len(dataset.dates) > 0
    assert dataset.n_scenarios == 2

    # Snapshot a non-zero scenario to validate scenario slicing.
    market = dataset.snapshot(time_idx=0, scenario_idx=1)

    assert market.quote(spot_id) > 0.0
    assert market.vol_surface(vol_id).vol(expiry=0.5, strike=1.0) > 0.0
    assert market.curve(curve_id).df(1.0) > 0.0
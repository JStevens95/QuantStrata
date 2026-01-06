from __future__ import annotations

import pytest
import numpy as np

from src.marketdata.ids import MarketId
from src.marketdata.requests import TimeseriesRequest, Universe
from src.marketdata.synthetic.provider import SyntheticProvider
from src.marketdata.synthetic.config import SyntheticProviderConfig, CurveZeroSpec, VolGridSmileSpec



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


def test_timeseries_panel_shapes_and_axis_names() -> None:
    spot_id = MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD")
    vol_id = MarketId(asset_class="FX", mkt_type="VOL", name="EURUSD")
    curve_id = MarketId(asset_class="IR", mkt_type="CURVE", name="USD.OIS")

    provider = SyntheticProvider(seed=7)

    ds = provider.get_timeseries(
        TimeseriesRequest(
            start="2025-01-01",
            end="2025-01-10",
            freq="B",
            universe=Universe([spot_id, vol_id, curve_id]),
            scenarios=3,
        )
    )

    # Spot panels are scalar quotes stored as [T,S]
    spot_panel = ds.panels[spot_id]
    assert spot_panel.axis_names == ("time", "scenario")
    assert spot_panel.data.shape == (len(ds.dates), ds.n_scenarios)

    curve_panel = ds.curve_params[curve_id]
    vol_panel = ds.vol_params[vol_id]

    # Curve/vol params may be blocks (e.g. curve grids [T,S,K,2], vol grids [T,S,n_exp,n_k]).
    # Only require that the first two axes are time/scenario and the leading shapes match.
    assert curve_panel.axis_names[:2] == ("time", "scenario")
    assert curve_panel.data.shape[:2] == (len(ds.dates), ds.n_scenarios)

    assert vol_panel.axis_names[:2] == ("time", "scenario")
    assert vol_panel.data.shape[:2] == (len(ds.dates), ds.n_scenarios)


def test_fixing_copies_spot_when_spot_present() -> None:
    spot_id = MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD")
    fixing_id = MarketId(asset_class="FX", mkt_type="FIXING", name="EURUSD")

    provider = SyntheticProvider(seed=7)

    ds = provider.get_timeseries(
        TimeseriesRequest(
            start="2025-01-01",
            end="2025-01-05",
            freq="D",
            universe=Universe([spot_id, fixing_id]),
            scenarios=2,
        )
    )

    spot = ds.panels[spot_id].data
    fixing = ds.panels[fixing_id].data

    np.testing.assert_allclose(fixing, spot, rtol=0.0, atol=0.0)


def test_snapshot_bounds_raise() -> None:
    spot_id = MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD")

    provider = SyntheticProvider(seed=7)
    ds = provider.get_timeseries(
        TimeseriesRequest(
            start="2025-01-01",
            end="2025-01-03",
            freq="D",
            universe=Universe([spot_id]),
            scenarios=2,
        )
    )

    with pytest.raises(IndexError):
        _ = ds.snapshot(time_idx=len(ds.dates), scenario_idx=0)

    with pytest.raises(IndexError):
        _ = ds.snapshot(time_idx=0, scenario_idx=ds.n_scenarios)


@pytest.mark.parametrize("freq", ["X", "", "  "])
def test_timeseries_invalid_freq_raises(freq: str) -> None:
    spot_id = MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD")
    provider = SyntheticProvider(seed=7)

    with pytest.raises(ValueError):
        provider.get_timeseries(
            TimeseriesRequest(
                start="2025-01-01",
                end="2025-01-03",
                freq=freq,
                universe=Universe([spot_id]),
                scenarios=1,
            )
        )


def test_timeseries_start_after_end_raises() -> None:
    spot_id = MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD")
    provider = SyntheticProvider(seed=7)

    with pytest.raises(ValueError):
        provider.get_timeseries(
            TimeseriesRequest(
                start="2025-01-10",
                end="2025-01-01",
                freq="D",
                universe=Universe([spot_id]),
                scenarios=1,
            )
        )


def test_timeseries_spot_is_not_flat_over_time() -> None:
    spot_id = MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD")

    provider = SyntheticProvider(seed=7)

    ds = provider.get_timeseries(
        TimeseriesRequest(
            start="2025-01-01",
            end="2025-01-20",
            freq="B",
            universe=Universe([spot_id]),
            scenarios=3,
        )
    )

    x = ds.panels[spot_id].data  # [T,S]
    assert x.shape[0] > 2
    # at least one time step differs from t=0 for at least one scenario
    assert bool(np.any(np.abs(x[1:, :] - x[0:1, :]) > 0.0))


def test_timeseries_curve_is_term_structured_not_flat() -> None:
    curve_id = MarketId(asset_class="IR", mkt_type="CURVE", name="USD.OIS")

    cfg = SyntheticProviderConfig(
        curve=CurveZeroSpec(tenors=np.array([0.25, 0.5, 1.0, 2.0, 5.0]), base_rate=0.02, slope=0.01, curvature=0.00)
    )
    provider = SyntheticProvider(seed=7, config=cfg)

    ds = provider.get_timeseries(
        TimeseriesRequest(
            start="2025-01-01",
            end="2025-01-01",
            freq="D",
            universe=Universe([curve_id]),
            scenarios=1,
        )
    )

    m = ds.snapshot(time_idx=0, scenario_idx=0)
    c = m.curve(curve_id)

    r1 = c.zero_rate(1.0)
    r5 = c.zero_rate(5.0)
    assert r5 > r1  # positive slope => upward curve


def test_timeseries_vol_surface_is_smile_shaped() -> None:
    vol_id = MarketId(asset_class="FX", mkt_type="VOL", name="EURUSD")

    cfg = SyntheticProviderConfig(
        vol=VolGridSmileSpec(
            expiries=np.array([0.5]),
            strikes=np.array([0.90, 1.00, 1.10]),
            atm_vol=0.12,
            skew=-0.10,
            smile=0.50,
            term=0.0,
            noise_scale=0.0,
        )
    )
    provider = SyntheticProvider(seed=7, config=cfg)

    ds = provider.get_timeseries(
        TimeseriesRequest(
            start="2025-01-01",
            end="2025-01-01",
            freq="D",
            universe=Universe([vol_id]),
            scenarios=1,
        )
    )

    m = ds.snapshot(time_idx=0, scenario_idx=0)
    s = m.vol_surface(vol_id)

    # Smile: edge strikes >= atm (for positive curvature)
    v_low = s.vol(expiry=0.5, strike=0.90)
    v_atm = s.vol(expiry=0.5, strike=1.00)
    v_high = s.vol(expiry=0.5, strike=1.10)

    # With negative skew, right wing can be below ATM; test convexity instead:
    assert v_low + v_high >= 2.0 * v_atm
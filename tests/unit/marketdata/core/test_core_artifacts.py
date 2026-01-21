from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.testing as npt

from src.marketdata.core.artifacts import load_market_dataset, save_market_dataset
from src.marketdata.core.ids import MarketId
from src.marketdata.core.requests import TimeseriesRequest, Universe
from src.marketdata.providers.synthetic.provider import SyntheticProvider
from src.marketdata.curves.factories import ZeroRateCurveFactory
from src.marketdata.surfaces.factories import GridVolFactory


def test_market_dataset_artifact_roundtrip(tmp_path: Path) -> None:
    """
    Ensure we can save/load a MarketDataset and preserve:
    - dates, scenarios, meta
    - all panels numerically
    - factories (type + params)
    - snapshot() behavior
    """
    provider = SyntheticProvider(seed=7)

    spot = MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD", qualifiers=(("dom", "USD"),))
    curve = MarketId(asset_class="IR", mkt_type="CURVE", name="USD", qualifiers=(("curve", "OIS"),))
    vol = MarketId(asset_class="FX", mkt_type="VOL", name="EURUSD", qualifiers=(("dom", "USD"),))

    ds = provider.get_timeseries(
        TimeseriesRequest(
            start="2026-01-01",
            end="2026-01-03",
            freq="D",
            universe=Universe([spot, curve, vol]),
            scenarios=2,
        )
    )

    out_dir = tmp_path / "ds_artifact"
    save_market_dataset(ds, out_dir, overwrite=False)

    ds2 = load_market_dataset(out_dir)

    assert ds2.dates == ds.dates
    assert ds2.n_scenarios == ds.n_scenarios
    assert dict(ds2.meta or {}).get("freq") == dict(ds.meta or {}).get("freq")

    # Panels numeric equality
    for mid, p in ds.panels.items():
        assert mid in ds2.panels
        assert ds2.panels[mid].axis_names == p.axis_names
        npt.assert_allclose(ds2.panels[mid].data, p.data, rtol=0.0, atol=0.0)

    for mid, p in ds.curve_params.items():
        assert mid in ds2.curve_params
        assert ds2.curve_params[mid].axis_names == p.axis_names
        npt.assert_allclose(ds2.curve_params[mid].data, p.data, rtol=0.0, atol=0.0)

    for mid, p in ds.vol_params.items():
        assert mid in ds2.vol_params
        assert ds2.vol_params[mid].axis_names == p.axis_names
        npt.assert_allclose(ds2.vol_params[mid].data, p.data, rtol=0.0, atol=0.0)

    # Factories preserved (type checks)
    for mid, f in ds.curve_factories.items():
        assert mid in ds2.curve_factories
        assert type(ds2.curve_factories[mid]) is type(f)

    for mid, f in ds.vol_factories.items():
        assert mid in ds2.vol_factories
        assert type(ds2.vol_factories[mid]) is type(f)

    # Optional: check a couple of factory params where relevant
    # (your synthetic stack typically uses these)
    for mid, f in ds2.curve_factories.items():
        if isinstance(f, ZeroRateCurveFactory):
            assert str(f.extrapolation)

    for mid, f in ds2.vol_factories.items():
        if isinstance(f, GridVolFactory):
            assert np.asarray(f.expiries).size > 0
            assert np.asarray(f.strikes).size > 0

    # Snapshot behavior equality on a couple of values
    m1 = ds.snapshot(time_idx=0, scenario_idx=0)
    m2 = ds2.snapshot(time_idx=0, scenario_idx=0)

    assert abs(m1.quote(spot) - m2.quote(spot)) == 0.0
    assert abs(m1.curve(curve).df(1.0) - m2.curve(curve).df(1.0)) == 0.0
    assert abs(m1.vol_surface(vol).vol(0.5, 1.0) - m2.vol_surface(vol).vol(0.5, 1.0)) == 0.0
from __future__ import annotations

import numpy as np
import pytest

from src.marketdata.core.ids import MarketId
from src.marketdata.core.requests import MarketRequest, TimeseriesRequest, Universe
from src.marketdata.providers.synthetic.config import SyntheticProviderConfig
from src.marketdata.providers.synthetic.provider import SyntheticProvider
from src.marketdata.providers.static.provider import StaticProvider


def _make_universe() -> Universe:
    spot = MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD")
    vol = MarketId(
        asset_class="FX",
        mkt_type="VOL",
        name="EURUSD",
        qualifiers=(("dom", "USD"), ("for", "EUR")),
    )
    return Universe(ids=[spot, vol])


def _make_store_dataset():
    cfg = SyntheticProviderConfig(curve_method="zeros")
    synth = SyntheticProvider(seed=7, config=cfg)

    req = TimeseriesRequest(
        start="2026-01-01",
        end="2026-01-10",
        freq="D",
        universe=_make_universe(),
        scenarios=4,
    )
    return synth.get_timeseries(req)


def test_static_provider_get_timeseries_slices_time_and_scenarios() -> None:
    store = _make_store_dataset()
    provider = StaticProvider(dataset=store)

    req = TimeseriesRequest(
        start="2026-01-03",
        end="2026-01-05",
        freq="D",
        universe=_make_universe(),
        scenarios=2,
    )

    ds = provider.get_timeseries(req)

    assert ds.dates == ["2026-01-03", "2026-01-04", "2026-01-05"]
    assert int(ds.n_scenarios) == 2

    spot_mid = MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD")
    assert spot_mid in ds.panels
    spot = np.asarray(ds.panels[spot_mid].data, dtype=float)
    assert spot.shape == (3, 2)

    # snapshot should work end-to-end
    mkt = ds.snapshot(time_idx=0, scenario_idx=0)
    assert mkt.asof == "2026-01-03"
    assert float(mkt.quote(spot_mid)) == pytest.approx(float(spot[0, 0]))


def test_static_provider_get_market_returns_snapshot_at_asof_and_scenario() -> None:
    store = _make_store_dataset()
    provider = StaticProvider(dataset=store)

    spot_mid = MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD")
    u = _make_universe()

    mkt = provider.get_market(MarketRequest(asof="2026-01-04", universe=u, scenario=1))
    assert mkt.asof == "2026-01-04"
    assert np.isfinite(mkt.quote(spot_mid))


def test_static_provider_raises_if_requested_date_not_in_store() -> None:
    store = _make_store_dataset()
    provider = StaticProvider(dataset=store)

    req = TimeseriesRequest(
        start="2025-12-31",
        end="2026-01-02",
        freq="D",
        universe=_make_universe(),
        scenarios=1,
    )
    with pytest.raises(ValueError, match="missing requested date"):
        provider.get_timeseries(req)


def test_static_provider_raises_if_requested_scenarios_exceed_store() -> None:
    store = _make_store_dataset()
    provider = StaticProvider(dataset=store)

    req = TimeseriesRequest(
        start="2026-01-01",
        end="2026-01-02",
        freq="D",
        universe=_make_universe(),
        scenarios=999,
    )
    with pytest.raises(ValueError, match="exceeds stored scenarios"):
        provider.get_timeseries(req)


def test_static_provider_raises_if_universe_id_missing_from_store() -> None:
    store = _make_store_dataset()
    provider = StaticProvider(dataset=store)

    missing_mid = MarketId(asset_class="EQ", mkt_type="SPOT", name="SPX")
    bad_universe = Universe(ids=[missing_mid])

    req = TimeseriesRequest(
        start="2026-01-01",
        end="2026-01-02",
        freq="D",
        universe=bad_universe,
        scenarios=1,
    )
    with pytest.raises(ValueError, match="store missing MarketIds"):
        provider.get_timeseries(req)
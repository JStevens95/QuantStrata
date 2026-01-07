from __future__ import annotations

import numpy as np

from src.marketdata.core.ids import MarketId
from src.marketdata.core.requests import Universe, TimeseriesRequest, MarketRequest
from src.marketdata.providers.synthetic.config import SyntheticProviderConfig
from src.marketdata.providers.synthetic.provider import SyntheticProvider
from src.marketdata.providers.synthetic.specs import CurveBootstrapSpec, CurveZeroSpec


def _make_universe_basic() -> Universe:
    spot = MarketId("FX", "SPOT", "EURUSD")
    fixing = MarketId("FX", "FIXING", "EURUSD")
    curve = MarketId("IR", "CURVE", "USD.OIS")
    vol = MarketId("FX", "VOL", "EURUSD")
    return Universe([spot, fixing, curve, vol])


# -----------------------------------------------------------------------------
# get_timeseries() tests (explicit)
# -----------------------------------------------------------------------------

def test_get_timeseries_returns_dataset_with_expected_shapes_and_factories() -> None:
    provider = SyntheticProvider(seed=7)
    uni = _make_universe_basic()

    ds = provider.get_timeseries(
        TimeseriesRequest(
            start="2026-01-01",
            end="2026-01-03",
            freq="D",
            universe=uni,
            scenarios=3,
        )
    )

    # dataset basics
    assert ds.n_scenarios == 3
    assert ds.dates == ["2026-01-01", "2026-01-02", "2026-01-03"]

    # ids
    spot_id = MarketId("FX", "SPOT", "EURUSD")
    fixing_id = MarketId("FX", "FIXING", "EURUSD")
    curve_id = MarketId("IR", "CURVE", "USD.OIS")
    vol_id = MarketId("FX", "VOL", "EURUSD")

    # quote panels exist + have expected shape
    assert spot_id in ds.panels
    assert fixing_id in ds.panels
    assert ds.panels[spot_id].data.shape == (3, 3)

    # curve params + factory exist + have expected shape
    assert curve_id in ds.curve_params
    assert curve_id in ds.curve_factories
    curve_params = ds.curve_params[curve_id].data
    assert curve_params.ndim == 4
    assert curve_params.shape[0] == 3  # time
    assert curve_params.shape[1] == 3  # scenario
    assert curve_params.shape[-1] == 2  # [tenor, zero_rate]

    # vol params + factory exist + have expected shape
    assert vol_id in ds.vol_params
    assert vol_id in ds.vol_factories
    vol_params = ds.vol_params[vol_id].data
    assert vol_params.ndim == 3
    assert vol_params.shape[0] == 3
    assert vol_params.shape[1] == 3
    assert vol_params.shape[2] > 0  # flattened params length


def test_get_timeseries_fixing_reuses_spot_when_present() -> None:
    provider = SyntheticProvider(seed=11)
    ds = provider.get_timeseries(
        TimeseriesRequest(
            start="2026-01-01",
            end="2026-01-05",
            freq="D",
            universe=_make_universe_basic(),
            scenarios=2,
        )
    )

    spot_id = MarketId("FX", "SPOT", "EURUSD")
    fixing_id = MarketId("FX", "FIXING", "EURUSD")

    np.testing.assert_allclose(ds.panels[fixing_id].data, ds.panels[spot_id].data)


def test_get_timeseries_is_deterministic_per_marketid_independent_of_universe_order() -> None:
    provider = SyntheticProvider(seed=123)

    spot = MarketId("FX", "SPOT", "EURUSD")
    curve = MarketId("IR", "CURVE", "USD.OIS")
    vol = MarketId("FX", "VOL", "EURUSD")

    uni_a = Universe([spot, curve, vol])
    uni_b = Universe([vol, spot, curve])

    ds_a = provider.get_timeseries(
        TimeseriesRequest(start="2026-01-01", end="2026-01-04", freq="D", universe=uni_a, scenarios=4)
    )
    ds_b = provider.get_timeseries(
        TimeseriesRequest(start="2026-01-01", end="2026-01-04", freq="D", universe=uni_b, scenarios=4)
    )

    np.testing.assert_allclose(ds_a.panels[spot].data, ds_b.panels[spot].data)
    np.testing.assert_allclose(ds_a.curve_params[curve].data, ds_b.curve_params[curve].data)
    np.testing.assert_allclose(ds_a.vol_params[vol].data, ds_b.vol_params[vol].data)


# -----------------------------------------------------------------------------
# get_market() tests (explicit)
# -----------------------------------------------------------------------------

def test_get_market_returns_market_snapshot_and_matches_timeseries_snapshot_for_same_asof_and_scenario() -> None:
    provider = SyntheticProvider(seed=7)
    uni = _make_universe_basic()

    # get_market path
    req = MarketRequest(asof="2026-01-02", universe=uni, scenario=1)
    mkt = provider.get_market(req)
    assert mkt.asof == "2026-01-02"

    # compare against get_timeseries(...).snapshot(...)
    ds = provider.get_timeseries(
        TimeseriesRequest(
            start="2026-01-02",
            end="2026-01-02",
            freq="D",
            universe=uni,
            scenarios=2,  # must include scenario_idx=1
        )
    )
    mkt_expected = ds.snapshot(time_idx=0, scenario_idx=1)

    spot_id = MarketId("FX", "SPOT", "EURUSD")
    assert abs(mkt.quote(spot_id) - mkt_expected.quote(spot_id)) < 1e-12


def test_get_market_requests_enough_scenarios_when_scenario_index_is_high() -> None:
    provider = SyntheticProvider(seed=7)
    uni = _make_universe_basic()

    # scenario=3 implies provider must generate at least 4 scenarios internally
    req = MarketRequest(asof="2026-01-02", universe=uni, scenario=3)
    mkt = provider.get_market(req)

    spot_id = MarketId("FX", "SPOT", "EURUSD")
    assert np.isfinite(mkt.quote(spot_id))


# -----------------------------------------------------------------------------
# curve bootstrap path
# -----------------------------------------------------------------------------

def test_provider_curve_bootstrap_path_generates_curve_params_and_market_curve() -> None:
    curve_id = MarketId("IR", "CURVE", "USD.OIS")

    cfg = SyntheticProviderConfig(
        curve_method="bootstrap",
        curve_bootstrap=CurveBootstrapSpec(
            deposit_maturities=(0.25, 0.5),
            swap_maturities=(1.0, 2.0),
            pay_freq=2,
            noise_scale=0.0,
            engine="native",
        ),
        curve_zero=CurveZeroSpec(tenors=np.array([0.25, 0.5, 1.0, 2.0], dtype=float)),
    )

    provider = SyntheticProvider(seed=99, config=cfg)

    ds = provider.get_timeseries(
        TimeseriesRequest(
            start="2026-01-01",
            end="2026-01-02",
            freq="D",
            universe=Universe([curve_id]),
            scenarios=2,
        )
    )

    assert curve_id in ds.curve_params
    params = ds.curve_params[curve_id].data
    assert params.ndim == 4
    assert params.shape[0] == 2  # time
    assert params.shape[1] == 2  # scenario
    assert params.shape[-1] == 2  # [tenor, zero_rate]

    mkt = ds.snapshot(time_idx=0, scenario_idx=0)
    curve = mkt.curve(curve_id)

    df_2y = float(curve.df(2.0))
    assert np.isfinite(df_2y)
    assert df_2y > 0.0
from __future__ import annotations

import numpy as np
import pytest

from src.marketdata.core.ids import MarketId
from src.marketdata.builders.panels import (
    make_time_grid,
    make_quote_panel,
    make_zero_curve_panel,
    make_grid_vol_panel,
)
from src.marketdata.builders.datasets import build_marketdataset, validate_dataset_layout
from src.marketdata.curves.factories import ZeroRateCurveFactory
from src.marketdata.surfaces.factories import GridVolFactory


def _ids() -> dict[str, MarketId]:
    return {
        "spot": MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD"),
        "df_dom": MarketId(asset_class="IR", mkt_type="CURVE", name="USD.OIS", qualifiers=(("ccy", "USD"),)),
        "fx_vol": MarketId(asset_class="FX", mkt_type="VOL", name="EURUSD", qualifiers=(("cut", "NY"), ("conv", "delta25"))),
    }


def test_build_marketdataset_snapshot_roundtrip() -> None:
    ids = _ids()
    dates = make_time_grid(start="2026-01-07", n_t=3)
    n_s = 2

    quote_panels = {
        ids["spot"]: make_quote_panel(n_t=3, n_s=n_s, values=lambda ti, si: 1.10 + 0.01 * ti + 0.02 * si),
    }

    tenors = np.array([0.0, 0.5, 1.0, 2.0], dtype=float)

    def z_fn(ti: int, si: int) -> np.ndarray:
        base = np.array([0.03, 0.031, 0.032, 0.033], dtype=float)
        return base + 0.0005 * ti + 0.001 * si

    curve_panels = {
        ids["df_dom"]: make_zero_curve_panel(dates=dates, n_s=n_s, tenors=tenors, zero_rates=z_fn),
    }

    curve_factories = {
        ids["df_dom"]: ZeroRateCurveFactory(extrapolation="flat"),
    }

    expiries = np.array([0.5, 1.0], dtype=float)
    strikes = np.array([1.00, 1.10, 1.20], dtype=float)

    vol_panels = {
        ids["fx_vol"]: make_grid_vol_panel(
            dates=dates,
            n_s=n_s,
            spot_panel=quote_panels[ids["spot"]],
            expiries=expiries,
            strikes=strikes,
        )
    }

    vol_factories = {
        ids["fx_vol"]: GridVolFactory(expiries=expiries, strikes=strikes, extrapolation="flat"),
    }

    ds = build_marketdataset(
        dates=dates,
        n_scenarios=n_s,
        quote_panels=quote_panels,
        curve_param_panels=curve_panels,
        curve_factories=curve_factories,
        vol_param_panels=vol_panels,
        vol_factories=vol_factories,
        meta={"source": "unit_test"},
        validate=True,
    )

    mkt = ds.snapshot(time_idx=1, scenario_idx=1)
    assert mkt.asof == dates[1]

    s = mkt.quote(ids["spot"])
    assert np.isfinite(s)

    c = mkt.curve(ids["df_dom"])
    assert np.isfinite(c.df(1.0))
    assert np.isfinite(c.zero_rate(1.0))
    assert np.isfinite(c.forward_rate(1.0, 2.0))

    v = mkt.vol_surface(ids["fx_vol"])
    assert np.isfinite(v.implied_vol(1.0, s))


def test_validate_dataset_layout_rejects_bad_zero_curve_panel_last_dim() -> None:
    ids = _ids()
    dates = make_time_grid(start="2026-01-07", n_t=2)
    n_s = 2

    quote_panels = {
        ids["spot"]: make_quote_panel(n_t=2, n_s=n_s, values=lambda ti, si: 1.10),
    }

    # Bad curve params: last dim should be 2, but is 1 here
    bad_curve = np.zeros((2, 2, 3, 1), dtype=float)
    curve_panels = {
        ids["df_dom"]:  # still a Panel, but wrong shape for ZeroRateCurveFactory
            __import__("src.marketdata.core.panel", fromlist=["Panel"]).Panel(data=bad_curve, axis_names=("time", "scenario", "tenor", "field")),
    }

    curve_factories = {ids["df_dom"]: ZeroRateCurveFactory(extrapolation="flat")}

    expiries = np.array([0.5], dtype=float)
    strikes = np.array([1.0], dtype=float)
    vol_panels = {
        ids["fx_vol"]: make_grid_vol_panel(
            dates=dates,
            n_s=n_s,
            spot_panel=quote_panels[ids["spot"]],
            expiries=expiries,
            strikes=strikes,
        )
    }
    vol_factories = {ids["fx_vol"]: GridVolFactory(expiries=expiries, strikes=strikes)}

    ds = build_marketdataset(
        dates=dates,
        n_scenarios=n_s,
        quote_panels=quote_panels,
        curve_param_panels=curve_panels,
        curve_factories=curve_factories,
        vol_param_panels=vol_panels,
        vol_factories=vol_factories,
        validate=False,  # build first
    )

    with pytest.raises(ValueError, match="last dim size 2"):
        validate_dataset_layout(ds)
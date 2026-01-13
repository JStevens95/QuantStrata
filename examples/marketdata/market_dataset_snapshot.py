# examples/marketdata/market_dataset_snapshot.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Tuple

import numpy as np
import matplotlib.pyplot as plt

from src.marketdata.core.ids import MarketId
from src.marketdata.core.dataset import MarketDataset
from src.marketdata.core.panel import Panel

# ---- marketdata factories ----
from src.marketdata.curves.factories import ZeroRateCurveFactory
from src.marketdata.surfaces.factories import GridVolFactory

# ---- Plotting helpers ----
from src.core.reporting.plots.utils import render_fig, PlotConfig
from src.core.reporting.plots.marketdata.curves import plot_curve_df, plot_curve_zero_rate
from src.core.reporting.plots.marketdata.quotes import plot_quotes
from src.core.reporting.plots.marketdata.surfaces import plot_vol_surface_heatmap, plot_vol_smile_slices, plot_vol_surface


# =============================================================================
# 1) Market IDs (canonical identifiers)
# =============================================================================

@dataclass(frozen=True, slots=True)
class ExampleIds:
    spot: MarketId
    df_dom: MarketId
    df_for: MarketId
    fx_vol: MarketId


def build_market_ids() -> ExampleIds:
    """
    Canonical IDs used by both:
      - MarketDataset storage keys
      - pricers (Market.quote/curve/vol_surface lookups)
    """
    return ExampleIds(
        spot=MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD"),
        df_dom=MarketId(asset_class="IR", mkt_type="CURVE", name="USD.OIS", qualifiers=(("ccy", "USD"),)),
        df_for=MarketId(asset_class="IR", mkt_type="CURVE", name="EUR.OIS", qualifiers=(("ccy", "EUR"),)),
        fx_vol=MarketId(asset_class="FX", mkt_type="VOL", name="EURUSD", qualifiers=(("cut", "NY"), ("conv", "delta25"))),
    )


# =============================================================================
# 2) Panel builders (what gets stored in MarketDataset)
# =============================================================================

def build_dates(n_t: int) -> list[str]:
    base = np.datetime64("2026-01-07")
    return [(base + np.timedelta64(i, "D")).astype(str) for i in range(int(n_t))]


def build_quote_panels(ids: ExampleIds, *, n_t: int, n_s: int) -> Mapping[MarketId, Panel]:
    """
    Quotes are scalar panels:
      [T] or [T,S] (axis_names must declare scenario if 2D)
    """
    t = np.arange(n_t, dtype=float)

    # Simple scenario-dependent spot evolution (toy, but non-flat)
    spot_s0 = 1.10 * (1.0 + 0.0010 * t)
    spot_s1 = 1.10 * (1.0 + 0.0015 * t)
    spot_s2 = 1.10 * (1.0 + 0.0005 * t)

    spot_ts = np.stack([spot_s0, spot_s1, spot_s2], axis=1)[:, :n_s]  # [T,S]

    return {
        ids.spot: Panel(data=spot_ts, axis_names=("time", "scenario")),
    }


def build_curve_param_panels(
    ids: ExampleIds,
    *,
    dates: list[str],
    n_s: int,
    tenors: np.ndarray,
) -> Mapping[MarketId, Panel]:
    """
    Curve params must match ZeroRateCurveFactory expectations.

    ZeroRateCurveFactory expects params shaped:
      - [K,2] columns [tenor, zero_rate]   OR
      - [2,K] rows    [tenor, zero_rate]

    Therefore we store curve params as a block panel:
      [T, S, K, 2]  where [:,:, :,0]=tenor and [:,:, :,1]=zero_rate.

    Then MarketDataset.snapshot(...) slices to [K,2], which the factory consumes directly.
    """
    n_t = len(dates)
    tenors = np.asarray(tenors, dtype=float).reshape(-1)
    k = int(tenors.size)

    # Base continuous zero rates at each tenor (must match len(tenors))
    usd_base = np.array([0.045, 0.046, 0.047, 0.048, 0.049, 0.050, 0.050], dtype=float)
    eur_base = np.array([0.032, 0.033, 0.034, 0.035, 0.036, 0.037, 0.037], dtype=float)

    if usd_base.size != k or eur_base.size != k:
        raise ValueError("Base curve arrays must match len(tenors).")

    usd = np.zeros((n_t, n_s, k, 2), dtype=float)
    eur = np.zeros((n_t, n_s, k, 2), dtype=float)

    for ti in range(n_t):
        time_bump = 0.0002 * float(ti)
        for si in range(n_s):
            scen_bump = 0.0005 * float(si)

            usd_zeros = usd_base + time_bump + scen_bump
            eur_zeros = eur_base + 0.5 * time_bump + 0.5 * scen_bump

            usd[ti, si, :, 0] = tenors
            usd[ti, si, :, 1] = usd_zeros

            eur[ti, si, :, 0] = tenors
            eur[ti, si, :, 1] = eur_zeros

    return {
        ids.df_dom: Panel(data=usd, axis_names=("time", "scenario", "tenor", "field")),
        ids.df_for: Panel(data=eur, axis_names=("time", "scenario", "tenor", "field")),
    }


def _toy_vol_grid(expiries: np.ndarray, strikes: np.ndarray, *, spot: float, atm: float) -> np.ndarray:
    """
    Toy surface used to demonstrate the workflow of:
      (block panel + factory) -> VolSurface snapshot.

    Real smile building is covered by the FX calibration example.
    """
    log_m = np.log(strikes / float(spot))
    out = np.zeros((expiries.size, strikes.size), dtype=float)

    for i, t in enumerate(expiries.tolist()):
        skew = -0.06 * np.sqrt(float(t))
        curv = 0.18 / (1.0 + float(t))
        out[i, :] = float(atm) + skew * log_m + curv * (log_m ** 2)

    return np.maximum(out, 1e-8)


def build_vol_param_panels_and_factories(
    ids: ExampleIds,
    *,
    quote_panels: Mapping[MarketId, Panel],
    expiries: np.ndarray,
    strikes: np.ndarray,
) -> Tuple[Mapping[MarketId, Panel], Mapping[MarketId, GridVolFactory]]:
    """
    Vol params are block panels:
      [T,S,n_exp,n_k]

    Factory reconstructs GridVolSurface from the sliced [n_exp,n_k] block.
    """
    spot_ts = np.asarray(quote_panels[ids.spot].data, dtype=float)  # [T,S]
    n_t, n_s = spot_ts.shape
    n_exp = expiries.size
    n_k = strikes.size

    vol_params = np.zeros((n_t, n_s, n_exp, n_k), dtype=float)

    for ti in range(n_t):
        for si in range(n_s):
            spot = float(spot_ts[ti, si])
            atm = 0.12 + 0.002 * float(ti) / max(n_t - 1, 1) + 0.002 * float(si)
            vol_params[ti, si, :, :] = _toy_vol_grid(expiries, strikes, spot=spot, atm=atm)

    panels = {
        ids.fx_vol: Panel(data=vol_params, axis_names=("time", "scenario", "expiry", "strike")),
    }

    factories = {
        ids.fx_vol: GridVolFactory(expiries=expiries, strikes=strikes, extrapolation="flat"),
    }

    return panels, factories


# =============================================================================
# 3) Build MarketDataset (the production container)
# =============================================================================

def build_dataset(*, n_t: int = 6, n_s: int = 3) -> Tuple[MarketDataset, ExampleIds]:
    ids = build_market_ids()
    dates = build_dates(n_t=n_t)

    # Quotes
    quote_panels = build_quote_panels(ids, n_t=n_t, n_s=n_s)

    # Curves: params + factories
    tenors = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0], dtype=float)
    curve_params = build_curve_param_panels(ids, dates=dates, n_s=n_s, tenors=tenors)

    curve_factories = {
        ids.df_dom: ZeroRateCurveFactory(extrapolation="flat"),
        ids.df_for: ZeroRateCurveFactory(extrapolation="flat"),
    }

    # Vols: params + factories
    expiries = np.array([0.25, 0.5, 1.0, 2.0], dtype=float)
    strikes = 1.10 * np.exp(np.linspace(-0.25, 0.25, 31, dtype=float))

    vol_params, vol_factories = build_vol_param_panels_and_factories(
        ids,
        quote_panels=quote_panels,
        expiries=expiries,
        strikes=strikes,
    )

    ds = MarketDataset(
        dates=dates,
        n_scenarios=n_s,
        panels=quote_panels,
        curve_params=curve_params,
        curve_factories=curve_factories,
        vol_params=vol_params,
        vol_factories=vol_factories,
        meta={"source": "demo", "purpose": "marketdataset_snapshot"},
    )
    return ds, ids


# =============================================================================
# 4) Snapshot + reporting
# =============================================================================

def print_snapshot(ds: MarketDataset, ids: ExampleIds, *, time_idx: int, scenario_idx: int) -> None:
    mkt = ds.snapshot(time_idx=time_idx, scenario_idx=scenario_idx)

    print("\n" + "=" * 90)
    print(f"MARKET SNAPSHOT (from MarketDataset)  t={time_idx}  s={scenario_idx}  asof={mkt.asof}")
    print("=" * 90)

    s0 = mkt.quote(ids.spot)
    print(f"\nSpot: {s0:.6f}")

    c_dom = mkt.curve(ids.df_dom)
    c_for = mkt.curve(ids.df_for)
    print("\nCurves:")
    print(f"  USD df(1y)={c_dom.df(1.0):.8f}  r(1y)={c_dom.zero_rate(1.0):.6f}  f(1y,2y)={c_dom.forward_rate(1.0,2.0):.6f}")
    print(f"  EUR df(1y)={c_for.df(1.0):.8f}  r(1y)={c_for.zero_rate(1.0):.6f}  f(1y,2y)={c_for.forward_rate(1.0,2.0):.6f}")

    v = mkt.vol_surface(ids.fx_vol)
    print("\nVol surface samples:")
    print(f"  sigma(1y, ATM~spot)={v.implied_vol(1.0, s0):.6f}")
    print(f"  sigma(1y, 90%spot) ={v.implied_vol(1.0, 0.90*s0):.6f}")
    print(f"  sigma(1y,110%spot) ={v.implied_vol(1.0, 1.10*s0):.6f}")


def plot_snapshot(ds: MarketDataset, ids: ExampleIds, *, time_idx: int, scenario_idx: int, show: bool, save: bool) -> None:
    # Ensure matplotlib stays open until we explicitly show once at the end.
    plt.ioff()

    mkt = ds.snapshot(time_idx=time_idx, scenario_idx=scenario_idx)
    out_dir = Path("outputs/marketdata_example_02")
    if save:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Quotes
    fig_q = plot_quotes(mkt.quotes, title=f"Snapshot Quotes (t={time_idx}, s={scenario_idx})")
    if save:
        render_fig(fig=fig_q, cfg=PlotConfig(show=show, save=save, out_dir=out_dir, filename="01_quotes.png"))

    # Curves
    fig_df_dom = plot_curve_df(mkt.curve(ids.df_dom), title="USD OIS: DF(t)")
    if save:
        render_fig(fig=fig_df_dom, cfg=PlotConfig(show=show, save=save, out_dir=out_dir, filename="02_usd_df.png"))

    fig_z_dom = plot_curve_zero_rate(mkt.curve(ids.df_dom), title="USD OIS: r(t)")
    if save:
        render_fig(fig=fig_z_dom, cfg=PlotConfig(show=show, save=save, out_dir=out_dir, filename="03_usd_zero.png"))

    fig_df_for = plot_curve_df(mkt.curve(ids.df_for), title="EUR OIS: DF(t)")
    if save:
        render_fig(fig=fig_df_for, cfg=PlotConfig(show=show, save=save, out_dir=out_dir, filename="04_eur_df.png"))

    fig_z_for = plot_curve_zero_rate(mkt.curve(ids.df_for), title="EUR OIS: r(t)")
    if save:
        render_fig(fig=fig_z_for, cfg=PlotConfig(show=show, save=save, out_dir=out_dir, filename="05_eur_zero.png"))

    # Vol surface
    vol = mkt.vol_surface(ids.fx_vol)
    fig_hm = plot_vol_surface_heatmap(vol, title="EURUSD: Vol Surface Heatmap")
    if save:
        render_fig(fig=fig_hm, cfg=PlotConfig(show=show, save=save, out_dir=out_dir, filename="06_vol_heatmap.png"))

    fig_smile = plot_vol_smile_slices(vol, title="EURUSD: Smile slices σ(T,K)")
    if save:
        render_fig(fig=fig_smile, cfg=PlotConfig(show=show, save=save, out_dir=out_dir, filename="07_vol_smiles.png"))

    fig_surface = plot_vol_surface(vol, title="EURUSD: Vol Surface")
    if save:
        render_fig(fig=fig_surface, cfg=PlotConfig(show=show, save=save, out_dir=out_dir, filename="08_vol_surface.png"))

    if show:
        # One blocking call at the very end so all figures appear together.
        plt.show(block=True)


def main(plot_config: PlotConfig, time_idx: int = 2, scenario_idx: int = 1) -> None:
    # build market data set.
    ds, ids = build_dataset(n_t=6, n_s=3)

    # print summary of markerdata snapshot
    print_snapshot(
        ds, ids, time_idx=time_idx, scenario_idx=scenario_idx
    )

    # plot market data snapshot
    plot_snapshot(
        ds, ids, time_idx=time_idx, scenario_idx=scenario_idx, show=plot_config.show, save=plot_config.save
    )


if __name__ == "__main__":
    # define plot configuration.
    cfg = PlotConfig()

    # run main script.
    main(plot_config=cfg, time_idx=2, scenario_idx=1)
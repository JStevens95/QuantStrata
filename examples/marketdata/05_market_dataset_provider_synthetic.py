from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

from src.core.reporting.plots.utils import PlotConfig, render_fig
from src.core.reporting.plots.marketdata.quotes import plot_quotes
from src.core.reporting.plots.marketdata.curves import plot_curve_df, plot_curve_zero_rate
from src.core.reporting.plots.marketdata.surfaces import (
    plot_vol_surface_heatmap,
    plot_vol_smile_slices,
    plot_vol_surface,
)

from src.marketdata.core.ids import MarketId
from src.marketdata.core.market import Market
from src.marketdata.core.requests import TimeseriesRequest, Universe
from src.marketdata.synthetic.config import SyntheticProviderConfig
from src.marketdata.providers.synthetic.provider import SyntheticProvider
from src.marketdata.synthetic.specs import CurveZeroSpec, VolGridSmileSpec


# ======================================================================================
# 0) Example IDs + Universe
# ======================================================================================

@dataclass(frozen=True, slots=True)
class ExampleIds:
    """
    Canonical MarketIds used in this example.

    What this exercises
    -------------------
    - FX SPOT (explicit request)
    - FX VOL  (explicit request)
      -> dependency closure should auto-add IR curves for dom/for currencies
    """
    fx_spot_eurusd: MarketId
    fx_vol_eurusd: MarketId


def make_example_ids() -> ExampleIds:
    """
    Construct the minimal set of requested MarketIds for this example.

    Notes
    -----
    - We encode dom/for explicitly in qualifiers so FX VOL prerequisites are unambiguous.
    - Qualifiers are tuple-of-pairs (hashable, deterministic).
    """
    spot = MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD")

    vol = MarketId(
        asset_class="FX",
        mkt_type="VOL",
        name="EURUSD",
        qualifiers=(("dom", "USD"), ("for", "EUR"), ("cut", "NY"), ("conv", "delta25")),
    )

    return ExampleIds(fx_spot_eurusd=spot, fx_vol_eurusd=vol)


def make_universe(ids: ExampleIds) -> Universe:
    """
    Build the Universe to request from the provider.

    Important
    ---------
    We only include what the user asked for (SPOT + VOL).
    Dependency closure will pull in IR curves automatically.
    """
    return Universe(ids=[ids.fx_spot_eurusd, ids.fx_vol_eurusd])


# ======================================================================================
# 1) Provider config + creation
# ======================================================================================

def build_provider(*, seed: int) -> SyntheticProvider:
    """
    Construct a SyntheticProvider with stable defaults.

    Current production-stable settings
    ----------------------------------
    - curve_method="zeros" (desk-ready and deterministic)
    """
    # ----------------------------
    # Curve tenor grid (years)
    # Dense front-end, then taper.
    # ----------------------------
    curve_tenors = np.array(
        [
            1/365, 7/365, 14/365, 1/12, 2/12, 3/12, 6/12, 9/12,
            1, 2, 3, 4, 5, 7, 10, 12, 15, 20, 25, 30
        ],
        dtype=float,
    )

    # ----------------------------
    # Vol expiry grid (years)
    # Front-end dense, then longer dated.
    # ----------------------------
    vol_expiries = np.array(
        [
            1/12, 2/12, 3/12, 6/12, 9/12,
            1, 2, 3, 4, 5, 7, 10
        ],
        dtype=float,
    )

    # ----------------------------
    # Strike grid (absolute strikes)
    # Use log-spacing around spot0.
    # IMPORTANT: match spot0 to your SpotGbmSpec(initial_level=...)
    # ----------------------------
    spot0 = 1.10
    log_mny = np.linspace(-0.40, 0.40, 61, dtype=float)  # 61 strikes, +/-40% log-moneyness
    vol_strikes = spot0 * np.exp(log_mny)

    cfg = SyntheticProviderConfig(
        curve_method="zeros",
        curve_zero=CurveZeroSpec(tenors=curve_tenors),
        vol=VolGridSmileSpec(expiries=vol_expiries, strikes=vol_strikes),
    )

    return SyntheticProvider(seed=int(seed), config=cfg)


# ======================================================================================
# 2) Dataset generation + debug prints
# ======================================================================================

def build_dataset(
    *,
    provider: SyntheticProvider,
    universe: Universe,
    start: str,
    end: str,
    freq: str,
    scenarios: int,
):
    """
    Generate a MarketDataset from the provider.

    This is the canonical user entry-point for synthetic market data.
    """
    request = TimeseriesRequest(
        start=str(start),
        end=str(end),
        freq=str(freq),
        universe=universe,
        scenarios=int(scenarios),
    )
    return provider.get_timeseries(request)


def print_dataset_layout(ds) -> None:
    """
    Print shapes + axis_names so users can understand storage immediately.
    """
    print("\n" + "=" * 110)
    print(f"SYNTHETIC DATASET  T={len(ds.dates)}  S={int(ds.n_scenarios)}  dates=[{ds.dates[0]}..{ds.dates[-1]}]")
    print("=" * 110)

    def _shape(x) -> Tuple[int, ...]:
        return tuple(np.asarray(x).shape)

    print("\n[Quote panels]")
    for mid, p in ds.panels.items():
        print(f"  {mid.key():70s}  shape={_shape(p.data)}  axis={p.axis_names}")

    print("\n[Curve param panels]")
    for mid, p in ds.curve_params.items():
        fac = ds.curve_factories.get(mid)
        print(f"  {mid.key():70s}  shape={_shape(p.data)}  axis={p.axis_names}  factory={type(fac).__name__}")

    print("\n[Vol param panels]")
    for mid, p in ds.vol_params.items():
        fac = ds.vol_factories.get(mid)
        print(f"  {mid.key():70s}  shape={_shape(p.data)}  axis={p.axis_names}  factory={type(fac).__name__}")


# ======================================================================================
# 3) Dependency IDs + snapshot usage
# ======================================================================================

def expected_dependency_ids(vol_mid: MarketId) -> Dict[str, MarketId]:
    """
    Compute the curve MarketIds expected from FX VOL dependency closure.

    We read dom/for currencies from qualifiers and map them to:
      IR.CURVE.<CCY>.OIS|ccy=<CCY>
    """
    q = dict((str(k).strip().lower(), str(v).strip().upper()) for k, v in (vol_mid.qualifiers or ()))
    dom = q.get("dom")
    foreign = q.get("for")

    out: Dict[str, MarketId] = {}
    if dom is not None:
        out["dom_curve"] = MarketId(asset_class="IR", mkt_type="CURVE", name=f"{dom}.OIS", qualifiers=(("ccy", dom),))
    if foreign is not None:
        out["for_curve"] = MarketId(asset_class="IR", mkt_type="CURVE", name=f"{foreign}.OIS", qualifiers=(("ccy", foreign),))
    return out


def print_snapshot_summary(*, market: Market, ids: ExampleIds, time_idx: int, scenario_idx: int) -> None:
    """
    Print a pricing-friendly summary from a Market snapshot.
    """
    spot = float(market.quote(ids.fx_spot_eurusd))

    deps = expected_dependency_ids(ids.fx_vol_eurusd)
    dom_curve = market.curve(deps["dom_curve"])
    for_curve = market.curve(deps["for_curve"])

    r_dom_1y = float(dom_curve.zero_rate(1.0))
    r_for_1y = float(for_curve.zero_rate(1.0))

    vol = market.vol_surface(ids.fx_vol_eurusd)
    sig_atm_1y = float(vol.implied_vol(1.0, spot))

    print("\n" + "-" * 110)
    print(f"SNAPSHOT  t={time_idx}  s={scenario_idx}  asof={market.asof}")
    print("-" * 110)
    print(f"Spot(EURUSD)                : {spot:.6f}")
    print(f"USD curve r(1y) (dom)        : {r_dom_1y:.6f}")
    print(f"EUR curve r(1y) (for)        : {r_for_1y:.6f}")
    print(f"Vol sigma(1y, K~spot)        : {sig_atm_1y:.6f}")


# ======================================================================================
# 4) Plot helpers (more plots)
# ======================================================================================

def plot_spot_paths_panel(*, ds, spot_mid: MarketId, title: str) -> plt.Figure:
    """
    Plot EURUSD spot paths across scenarios using the raw dataset panel.

    Why this exists
    ---------------
    - Users often want to see the full scenario paths over time (not just a snapshot).
    """
    spot = np.asarray(ds.panels[spot_mid].data, dtype=float)  # [T,S]
    t = np.arange(spot.shape[0], dtype=int)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for s in range(spot.shape[1]):
        ax.plot(t, spot[:, s], label=f"scen {s}")

    ax.set_title(title)
    ax.set_xlabel("time index")
    ax.set_ylabel("spot")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_atm_term_structure(*, market: Market, vol_mid: MarketId, spot: float, expiries: np.ndarray, title: str) -> plt.Figure:
    """
    Plot ATM term structure sigma(T, K=spot) for a given Market snapshot.

    Notes
    -----
    - We sample vol.implied_vol(T, spot) at a set of expiries.
    - This is an intuitive “sanity plot” for users.
    """
    vol = market.vol_surface(vol_mid)

    x = np.asarray(expiries, dtype=float).reshape(-1)
    y = np.array([float(vol.implied_vol(float(T), float(spot))) for T in x], dtype=float)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y, marker="o")
    ax.set_title(title)
    ax.set_xlabel("expiry (years)")
    ax.set_ylabel("implied vol (ATM)")
    fig.tight_layout()
    return fig


def render_snapshot_plots(
    *,
    market: Market,
    ids: ExampleIds,
    plot_cfg: PlotConfig,
    title_suffix: str,
) -> None:
    """
    Render a bundle of snapshot plots using your reporting plot library.
    """
    figures = []

    # Quotes panel overview (whatever is present in market.quotes).
    figures.append(("01_quotes.png", plot_quotes(market.quotes, title=f"Snapshot Quotes {title_suffix}")))

    # Curves (dom + for) generated by dependency closure.
    deps = expected_dependency_ids(ids.fx_vol_eurusd)

    figures.append(("02_dom_curve_df.png", plot_curve_df(market.curve(deps["dom_curve"]), title=f"DOM Curve DF(t) {title_suffix}")))
    figures.append(("03_dom_curve_zero.png", plot_curve_zero_rate(market.curve(deps["dom_curve"]), title=f"DOM Curve r(t) {title_suffix}")))

    figures.append(("04_for_curve_df.png", plot_curve_df(market.curve(deps["for_curve"]), title=f"FOR Curve DF(t) {title_suffix}")))
    figures.append(("05_for_curve_zero.png", plot_curve_zero_rate(market.curve(deps["for_curve"]), title=f"FOR Curve r(t) {title_suffix}")))

    # Vol plots (surface, smiles, heatmap).
    vol = market.vol_surface(ids.fx_vol_eurusd)
    figures.append(("06_vol_heatmap.png", plot_vol_surface_heatmap(vol, title=f"FX VOL Heatmap {title_suffix}")))
    figures.append(("07_vol_smiles.png", plot_vol_smile_slices(vol, title=f"FX VOL Smile Slices {title_suffix}")))
    figures.append(("08_vol_surface_3d.png", plot_vol_surface(vol, title=f"FX VOL Surface (3D) {title_suffix}")))

    # Render all figures via your standard helper.
    for filename, fig in figures:
        render_fig(fig, cfg=plot_cfg, filename=filename)


# ======================================================================================
# 5) Determinism demo
# ======================================================================================

def assert_determinism_demo(*, seed: int, universe: Universe) -> None:
    """
    Demonstrate strict determinism for same seed/config (exact array equality).
    """
    provider_a = build_provider(seed=seed)
    provider_b = build_provider(seed=seed)

    ds_a = build_dataset(provider=provider_a, universe=universe, start="2026-01-01", end="2026-01-03", freq="D", scenarios=2)
    ds_b = build_dataset(provider=provider_b, universe=universe, start="2026-01-01", end="2026-01-03", freq="D", scenarios=2)

    spot_mid = MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD")
    a = np.asarray(ds_a.panels[spot_mid].data, dtype=float)
    b = np.asarray(ds_b.panels[spot_mid].data, dtype=float)

    if not np.array_equal(a, b):
        raise AssertionError("Determinism check failed: SPOT differs for same seed/config.")

    print("\nDeterminism demo: PASS (same seed/config => identical SPOT panel).")


# ======================================================================================
# 6) Main
# ======================================================================================

def main() -> None:
    """
    Example: SyntheticProvider -> MarketDataset -> snapshot -> pricing objects (+ lots of plots)

    What the user learns
    --------------------
    1) Request SPOT + VOL only
    2) Dependency closure auto-adds required IR curves
    3) Snapshot reconstructs Curve/VolSurface for pricing
    4) Plots demonstrate realism + debugging hooks
    """
    # Configure outputs (standard pattern used across your examples).
    plot_cfg = PlotConfig(
        show=True,   # Set False in CI environments
        save=False,   # Save files so users can inspect results
        out_dir=Path("outputs/marketdata/05_market_dataset_provider_synthetic"),
        block=True,
        close=False,
        dpi=160,
    )

    ids = make_example_ids()
    universe = make_universe(ids)

    provider = build_provider(seed=7)

    ds = build_dataset(
        provider=provider,
        universe=universe,
        start="2025-12-01",
        end="2026-01-14",
        freq="D",
        scenarios=8,
    )

    # Debug: show what was produced and how it is stored.
    print_dataset_layout(ds)

    # Sanity: dependency closure should have created the two curves.
    deps = expected_dependency_ids(ids.fx_vol_eurusd)
    assert deps["dom_curve"] in ds.curve_params
    assert deps["for_curve"] in ds.curve_params

    # Snapshot -> Market.
    time_idx = 3
    scenario_idx = 1
    mkt = ds.snapshot(time_idx=int(time_idx), scenario_idx=int(scenario_idx))

    # Print a pricing-friendly summary.
    print_snapshot_summary(market=mkt, ids=ids, time_idx=time_idx, scenario_idx=scenario_idx)

    # Extra plot: SPOT paths across scenarios (time series).
    fig_spot = plot_spot_paths_panel(ds=ds, spot_mid=ids.fx_spot_eurusd, title="EURUSD SPOT paths across scenarios")
    render_fig(fig_spot, cfg=plot_cfg, filename="00_spot_paths.png")

    # Snapshot bundle plots (quotes + curves + vol surface plots).
    title_suffix = f"(t={time_idx}, s={scenario_idx})"
    render_snapshot_plots(market=mkt, ids=ids, plot_cfg=plot_cfg, title_suffix=title_suffix)

    # Extra plot: ATM vol term structure sampled at expiries.
    # We use the vol factory grid if available; otherwise fall back to a sensible list.
    spot = float(mkt.quote(ids.fx_spot_eurusd))
    vol_factory = ds.vol_factories.get(ids.fx_vol_eurusd)
    expiries = np.asarray(getattr(vol_factory, "expiries", np.array([0.25, 0.5, 1.0, 2.0], dtype=float)), dtype=float)

    fig_atm = plot_atm_term_structure(
        market=mkt,
        vol_mid=ids.fx_vol_eurusd,
        spot=spot,
        expiries=expiries,
        title=f"ATM term structure σ(T, K=spot) {title_suffix}",
    )
    render_fig(fig_atm, cfg=plot_cfg, filename="09_atm_term_structure.png")

    # Determinism demo (strict equality).
    assert_determinism_demo(seed=7, universe=universe)


if __name__ == "__main__":
    main()
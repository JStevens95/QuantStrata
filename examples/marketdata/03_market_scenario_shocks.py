# examples/marketdata/03_market_dataset_scenarios.py
from __future__ import annotations

from src.marketdata.core.dataset import MarketDataset

"""
Example 03: Apply scenario shocks to a MarketDataset snapshot (ScenarioRunner workflow).

What this teaches
-----------------
1) You start from a MarketDataset (time × scenario panels + factories).
2) You "snapshot" it into a MarketView via:
       runner.base_snapshot(time_idx, scenario_idx)
3) You apply ScenarioShocks as *non-mutating wrappers* (views), e.g.:
       SpotShock, VolShock, ParallelRateShock
4) You validate the shock actually changes numbers and produce plots that
   reflect the shocked market.

Important note about plotting
-----------------------------
Some plotting helpers read GridVolSurface attributes like `.implied_vols`.
But a shocked vol surface may be a wrapper that overrides `implied_vol(T,K)`
without changing `.implied_vols`. If you plot `.implied_vols`, the shocked plot
can look identical.

To avoid that, this example plots smiles by *sampling* `implied_vol(T,K)` across
a strike grid, which correctly reflects the shock.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.core.reporting.plots.utils import PlotConfig, render_fig
from src.marketdata.builders.datasets import build_marketdataset
from src.marketdata.builders.panels import make_grid_vol_panel, make_quote_panel, make_time_grid, make_zero_curve_panel
from src.marketdata.core.ids import MarketId
from src.marketdata.scenarios.interfaces import MarketView, ScenarioPack
from src.marketdata.scenarios.runner import ScenarioRunner
from src.marketdata.scenarios.shocks import ParallelRateShock, SpotShock, VolShock

from src.marketdata.curves.factory import ZeroRateCurveFactory
from src.marketdata.surfaces.factory import GridVolFactory


# ======================================================================================
# 1) Canonical MarketIds used by the dataset (storage keys) and by snapshots (lookups)
# ======================================================================================

@dataclass(frozen=True, slots=True)
class ExampleIds:
    """Convenience bundle of MarketIds used by this example."""
    spot_eurusd: MarketId
    usd_ois_curve: MarketId
    eurusd_vol: MarketId


def make_example_ids() -> ExampleIds:
    """
    Construct canonical MarketIds.

    Notes
    -----
    - "USD.OIS" here is just a *name string* used in the example. In real market
      data ingestion you'd likely have additional qualifiers (index, curve family, etc.).
    """
    return ExampleIds(
        spot_eurusd=MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD"),
        usd_ois_curve=MarketId(
            asset_class="IR",
            mkt_type="CURVE",
            name="USD.OIS",
            qualifiers=(("ccy", "USD"),),
        ),
        eurusd_vol=MarketId(
            asset_class="FX",
            mkt_type="VOL",
            name="EURUSD",
            qualifiers=(("cut", "NY"), ("conv", "delta25")),
        ),
    )


# ======================================================================================
# 2) Build a small demo MarketDataset using the production builders (panels + factories)
# ======================================================================================

def build_demo_dataset(*, n_time: int, n_scenarios: int) -> Tuple[MarketDataset, ExampleIds]:
    """
    Build a small but complete MarketDataset (quotes + curve params + vol params).

    This is intentionally "provider-like":
    - we generate arrays/panels
    - we supply factories that reconstruct rich objects at snapshot-time
    """
    ids = make_example_ids()

    # ---- Time grid (MarketDataset.dates) ----
    dates = make_time_grid(start="2026-01-07", n_t=int(n_time), step="D")

    # ---- Quote panel: EURUSD spot(t, scenario) ----
    # A simple drift by time and scenario so the dataset isn't flat.
    spot_panel = make_quote_panel(
        n_t=int(n_time),
        n_s=int(n_scenarios),
        values=lambda ti, si: 1.10 * (1.0 + 0.0010 * ti + 0.0005 * si),
    )
    quote_panels = {ids.spot_eurusd: spot_panel}

    # ---- Curve panel: USD OIS-style zero curve params [T,S,K,2] ----
    tenors = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0], dtype=float)
    usd_zeros_base = np.array([0.045, 0.047, 0.048, 0.049, 0.050, 0.050], dtype=float)

    curve_panel = make_zero_curve_panel(
        dates=dates,
        n_s=int(n_scenarios),
        tenors=tenors,
        zero_rates=lambda ti, si: usd_zeros_base + 0.0002 * ti + 0.0005 * si,
    )
    curve_param_panels = {ids.usd_ois_curve: curve_panel}

    # Factory that consumes a [K,2] block (tenor, zero_rate)
    curve_factories = {ids.usd_ois_curve: ZeroRateCurveFactory(extrapolation="flat")}

    # ---- Vol panel: EURUSD grid vol params [T,S,n_exp,n_k] ----
    expiries = np.array([0.25, 0.5, 1.0, 2.0], dtype=float)

    # Strike grid is built around the *initial* spot level for clarity.
    # (At snapshot-time, the surface itself can still be sampled at any K.)
    strikes = 1.10 * np.exp(np.linspace(-0.25, 0.25, 31, dtype=float))

    vol_panel = make_grid_vol_panel(
        dates=dates,
        n_s=int(n_scenarios),
        spot_panel=spot_panel,
        expiries=expiries,
        strikes=strikes,
        # Optional: provide a custom generator; we leave it None to use the default demo smile.
        vol_grid=None,
    )
    vol_param_panels = {ids.eurusd_vol: vol_panel}

    vol_factories = {ids.eurusd_vol: GridVolFactory(expiries=expiries, strikes=strikes, extrapolation="flat")}

    dataset = build_marketdataset(
        dates=dates,
        n_scenarios=int(n_scenarios),
        quote_panels=quote_panels,
        curve_param_panels=curve_param_panels,
        curve_factories=curve_factories,
        vol_param_panels=vol_param_panels,
        vol_factories=vol_factories,
        meta={"source": "demo", "purpose": "03_market_dataset_scenarios"},
        validate=True,
    )
    return dataset, ids


# ======================================================================================
# 3) Scenarios (shocks) to apply
# ======================================================================================

def build_demo_scenario_pack(ids: ExampleIds) -> ScenarioPack:
    """
    Build a ScenarioPack containing multiple named shocks.

    Notes
    -----
    - bump_mode="relative" for vol means: sigma_new = sigma_old * (1 + bump)
    - For rates, we use an additive parallel shift in rate units (e.g. 25bp = 0.0025).
    """
    scenarios = {
        "spot_up_1pct": SpotShock(
            name="spot_up_1pct",
            spot_id=ids.spot_eurusd,
            bump=0.01,
            bump_mode="relative",
        ),
        "vol_up_10pct": VolShock(
            name="vol_up_10pct",
            vol_id=ids.eurusd_vol,
            bump=0.10,
            bump_mode="relative",
        ),
        "rates_up_25bp": ParallelRateShock(
            name="rates_up_25bp",
            curve_id=ids.usd_ois_curve,
            rate_shift=0.0025,
        ),
    }
    return ScenarioPack(scenarios=scenarios)


# ======================================================================================
# 4) Debug/diagnostic printing
# ======================================================================================

def print_snapshot_summary(mkt: MarketView, ids: ExampleIds, *, label: str) -> None:
    """
    Print a small, stable diagnostic summary for a MarketView.

    This is intentionally "repricer-friendly": we query through the MarketView API.
    """
    spot = float(mkt.quote(ids.spot_eurusd))

    curve = mkt.curve(ids.usd_ois_curve)
    r_1y = float(curve.zero_rate(1.0))
    df_1y = float(curve.df(1.0))

    vol = mkt.vol_surface(ids.eurusd_vol)
    sig_atm_1y = float(vol.implied_vol(1.0, spot))

    print("\n" + "-" * 90)
    print(f"{label}")
    print("-" * 90)
    print(f"Spot(EURUSD)                 : {spot:.6f}")
    print(f"USD curve df(1y)             : {df_1y:.8f}")
    print(f"USD curve zero_rate(1y)      : {r_1y:.6f}")
    print(f"EURUSD vol implied_vol(1y,ATM): {sig_atm_1y:.6f}")


# ======================================================================================
# 5) Plotting that reflects shocks correctly (sample implied_vol(T,K), not implied_vols)
# ======================================================================================

def plot_smile_overlay_sampled(
    *,
    base: MarketView,
    shocked: MarketView,
    ids: ExampleIds,
    expiries: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0),
    n_strikes: int = 51,
    strike_width: float = 0.25,
    title: str,
) -> plt.Figure:
    """
    Overlay base vs shocked smiles by sampling implied_vol(T,K).

    Parameters
    ----------
    strike_width:
        Width in log-strike space around ATM. We build strikes as:
            K = spot * exp(linspace(-strike_width, +strike_width, n_strikes))
    """
    base_spot = float(base.quote(ids.spot_eurusd))

    # Build a strike grid centered on the *base* ATM.
    strike_grid = base_spot * np.exp(np.linspace(-float(strike_width), float(strike_width), int(n_strikes)))

    base_vol = base.vol_surface(ids.eurusd_vol)
    shocked_vol = shocked.vol_surface(ids.eurusd_vol)

    fig, ax = plt.subplots()

    for T in expiries:
        # Sample via implied_vol(...) so wrappers/shocks are respected.
        y_base = np.array([float(base_vol.implied_vol(float(T), float(k))) for k in strike_grid], dtype=float)
        y_shck = np.array([float(shocked_vol.implied_vol(float(T), float(k))) for k in strike_grid], dtype=float)

        ax.plot(strike_grid, y_base, label=f"base  T={T:g}")
        ax.plot(strike_grid, y_shck, linestyle="--", label=f"shock T={T:g}")

    ax.set_title(title)
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Implied vol σ")
    ax.grid(True)
    ax.legend(ncol=2)

    fig.tight_layout()
    return fig


def plot_curve_overlay(
    *,
    base: MarketView,
    shocked: MarketView,
    ids: ExampleIds,
    tenors: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
    title: str,
) -> plt.Figure:
    """Overlay base vs shocked zero-rate curve r(T)."""
    t_grid = np.asarray(tenors, dtype=float)

    c0 = base.curve(ids.usd_ois_curve)
    c1 = shocked.curve(ids.usd_ois_curve)

    r0 = np.array([float(c0.zero_rate(float(t))) for t in t_grid], dtype=float)
    r1 = np.array([float(c1.zero_rate(float(t))) for t in t_grid], dtype=float)

    fig, ax = plt.subplots()
    ax.plot(t_grid, r0, label="base")
    ax.plot(t_grid, r1, linestyle="--", label="shock")

    ax.set_title(title)
    ax.set_xlabel("Tenor T (years)")
    ax.set_ylabel("Zero rate r(T)")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    return fig


# ======================================================================================
# 6) Main: build dataset → snapshot → apply scenarios → print + plot comparisons
# ======================================================================================

def main() -> None:
    """
    Run the dataset + ScenarioRunner workflow and produce plots that visibly differ.

    Design choice
    -------------
    - Create all figures first (so everything exists in memory).
    - Optionally save figures (cfg.save=True).
    - Show all figures with ONE blocking call at the very end.
      This avoids “script stops at first figure” behaviour.
    """
    # Turn interactive mode off so figure lifecycle is fully controlled by the script.
    plt.ioff()

    # Use non-blocking show during individual render calls.
    # We will do one final blocking plt.show(...) after all figures are rendered.
    plot_cfg = PlotConfig(
        show=True,
        save=False,  # set True if you want PNGs written to out_dir
        out_dir=Path("outputs/marketdata/03_market_dataset_scenarios"),
        block=False,  # IMPORTANT: do NOT block per figure
        close=False,
        dpi=160,
    )

    # ---- Build dataset (production builders) ----
    dataset, ids = build_demo_dataset(n_time=6, n_scenarios=3)

    # ---- Scenario runner bridges dataset snapshots -> MarketView ----
    runner = ScenarioRunner(dataset=dataset)

    # Choose one snapshot (time × scenario) to demonstrate.
    time_idx = 2
    scenario_idx = 1

    # Create the base snapshot (MarketView).
    base_market = runner.base_snapshot(time_idx=time_idx, scenario_idx=scenario_idx)

    # Build and apply a pack of scenario shocks.
    pack = build_demo_scenario_pack(ids)
    shocked_markets: Mapping[str, MarketView] = runner.apply_pack(
        time_idx=time_idx,
        scenario_idx=scenario_idx,
        pack=pack,
    )

    # ---- Print diagnostics (base + each scenario) ----
    print_snapshot_summary(base_market, ids, label=f"BASE snapshot (t={time_idx}, s={scenario_idx})")
    for scenario_name, shocked_market in shocked_markets.items():
        print_snapshot_summary(shocked_market, ids, label=f"SCENARIO: {scenario_name}")

    # ---- Build figures first (so we can show/save consistently) ----
    figures: list[tuple[str, plt.Figure]] = []

    # Base vs vol shock (smile overlay)
    vol_shocked = shocked_markets["vol_up_10pct"]
    fig_smile = plot_smile_overlay_sampled(
        base=base_market,
        shocked=vol_shocked,
        ids=ids,
        title=f"EURUSD Smile Overlay (base vs vol_up_10pct)  t={time_idx}, s={scenario_idx}",
    )
    figures.append(("01_smile_overlay_base_vs_vol_up.png", fig_smile))

    # Base vs rates shock (curve overlay)
    rate_shocked = shocked_markets["rates_up_25bp"]
    fig_curve = plot_curve_overlay(
        base=base_market,
        shocked=rate_shocked,
        ids=ids,
        title=f"USD Curve Overlay (base vs rates_up_25bp)  t={time_idx}, s={scenario_idx}",
    )
    figures.append(("02_curve_overlay_base_vs_rates_up.png", fig_curve))

    # Base vs spot shock (smile overlay; useful to show ATM shift effect)
    spot_shocked = shocked_markets["spot_up_1pct"]
    fig_smile_spot = plot_smile_overlay_sampled(
        base=base_market,
        shocked=spot_shocked,
        ids=ids,
        title=f"EURUSD Smile Overlay (base vs spot_up_1pct)  t={time_idx}, s={scenario_idx}",
    )
    figures.append(("03_smile_overlay_base_vs_spot_up.png", fig_smile_spot))

    # ---- Render (save/show) all figures using the same config ----
    for filename, fig in figures:
        # filename is only used if cfg.save=True (render_fig will ignore it otherwise)
        render_fig(fig, cfg=plot_cfg, filename=filename)

    # ---- One final blocking show keeps all windows alive together ----
    if plot_cfg.show:
        plt.show(block=True)

if __name__ == "__main__":
    main()
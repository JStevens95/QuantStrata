# examples/marketdata/02_market_dataset_snapshot.py
from __future__ import annotations

"""
Example 02: Build a MarketDataset (time × scenario panels + factories) and snapshot it.

What this teaches
-----------------
1) Why panels exist:
   - Panels are compact numeric “cubes” (numpy arrays) indexed by (time, scenario, …).
   - They are the *storage layer* for large datasets.

2) Why factories exist:
   - Factories reconstruct rich objects (Curve / VolSurface / Quote) from sliced panel blocks.
   - Snapshot-time reconstruction keeps storage lightweight and pricing-time convenient.

3) The key workflow:
       MarketDataset.snapshot(time_idx, scenario_idx) -> Market
   Then you use Market.quote(...) / Market.curve(...) / Market.vol_surface(...) like normal.

Notes
-----
- This example intentionally uses the production builders in:
    src/marketdata/builders/panels.py
    src/marketdata/builders/datasets.py
  so users see the “official” construction path.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.core.reporting.plots.utils import PlotConfig, render_fig
from src.core.reporting.plots.marketdata.curves import plot_curve_df, plot_curve_zero_rate
from src.core.reporting.plots.marketdata.quotes import plot_quotes
from src.core.reporting.plots.marketdata.surfaces import plot_vol_smile_slices, plot_vol_surface, plot_vol_surface_heatmap

from src.marketdata.builders.datasets import build_marketdataset
from src.marketdata.builders.panels import make_grid_vol_panel, make_quote_panel, make_time_grid, make_zero_curve_panel
from src.marketdata.core.dataset import MarketDataset
from src.marketdata.core.ids import MarketId
from src.marketdata.core.market import Market
from src.marketdata.core.panel import Panel

from src.marketdata.curves.factories import ZeroRateCurveFactory
from src.marketdata.surfaces.factories import GridVolFactory


# ======================================================================================
# 1) Canonical MarketIds used as:
#    - dict keys in the MarketDataset
#    - lookups inside the Market snapshot
# ======================================================================================

@dataclass(frozen=True, slots=True)
class ExampleIds:
    """Collection of MarketId keys used throughout this example."""
    spot_eurusd: MarketId
    usd_curve: MarketId
    eur_curve: MarketId
    eurusd_vol: MarketId


def make_example_ids() -> ExampleIds:
    """
    Create canonical MarketIds.

    These IDs are the “spine” of the marketdata layer:
    - providers produce data keyed by MarketId
    - MarketDataset stores Panels keyed by MarketId
    - pricers query a Market snapshot using MarketId
    """
    return ExampleIds(
        spot_eurusd=MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD"),
        usd_curve=MarketId(asset_class="IR", mkt_type="CURVE", name="USD.OIS", qualifiers=(("ccy", "USD"),)),
        eur_curve=MarketId(asset_class="IR", mkt_type="CURVE", name="EUR.OIS", qualifiers=(("ccy", "EUR"),)),
        eurusd_vol=MarketId(asset_class="FX", mkt_type="VOL", name="EURUSD", qualifiers=(("cut", "NY"), ("conv", "delta25"))),
    )


# ======================================================================================
# 2) Build a MarketDataset using production builders (panels + factories)
# ======================================================================================

def build_demo_dataset(*, n_time: int, n_scenarios: int) -> Tuple[MarketDataset, ExampleIds]:
    """
    Build a small-but-complete MarketDataset:
    - Quotes: spot panel
    - Curves: parameter panels shaped [T,S,K,2] + ZeroRateCurveFactory
    - Vols:   parameter panels shaped [T,S,n_exp,n_k] + GridVolFactory

    Parameters
    ----------
    n_time:
        Number of time points (T).
    n_scenarios:
        Number of scenarios (S).

    Returns
    -------
    (MarketDataset, ExampleIds)
        The dataset plus the IDs used to query snapshots.
    """
    # Create the canonical IDs used everywhere in the example.
    ids = make_example_ids()

    # Build deterministic dates for ds.dates (so output is stable and testable).
    dates: List[str] = make_time_grid(start="2026-01-07", n_t=int(n_time), step="D")

    # -------------------------------
    # Quote panels (scalar data)
    # -------------------------------
    # Build a spot(t,scenario) panel using a callable generator.
    # (Callable is convenient because you can inject time/scenario behaviour.)
    spot_panel: Panel = make_quote_panel(
        n_t=int(n_time),
        n_s=int(n_scenarios),
        values=lambda ti, si: 1.10 * (1.0 + 0.0010 * ti + 0.0005 * si),
    )

    # Store quote panels keyed by MarketId.
    quote_panels: Dict[MarketId, Panel] = {
        ids.spot_eurusd: spot_panel,
    }

    # -------------------------------
    # Curve panels (block data)
    # -------------------------------
    # Tenor grid in years (example only; in production you’d align with instruments).
    tenors = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0], dtype=float)

    # Base zero rates (continuously compounded) for USD and EUR.
    usd_zero_base = np.array([0.045, 0.046, 0.047, 0.048, 0.049, 0.050, 0.050], dtype=float)
    eur_zero_base = np.array([0.032, 0.033, 0.034, 0.035, 0.036, 0.037, 0.037], dtype=float)

    # Build curve panels with shape [T,S,K,2] where [:,:,:,0]=tenor and [:,:,:,1]=zero_rate.
    usd_curve_panel: Panel = make_zero_curve_panel(
        dates=dates,
        n_s=int(n_scenarios),
        tenors=tenors,
        zero_rates=lambda ti, si: usd_zero_base + 0.0002 * ti + 0.0005 * si,
    )
    eur_curve_panel: Panel = make_zero_curve_panel(
        dates=dates,
        n_s=int(n_scenarios),
        tenors=tenors,
        zero_rates=lambda ti, si: eur_zero_base + 0.0001 * ti + 0.0003 * si,
    )

    # Store curve parameter panels keyed by MarketId.
    curve_param_panels: Dict[MarketId, Panel] = {
        ids.usd_curve: usd_curve_panel,
        ids.eur_curve: eur_curve_panel,
    }

    # Factories reconstruct Curve objects from each sliced [K,2] block.
    curve_factories = {
        ids.usd_curve: ZeroRateCurveFactory(extrapolation="flat"),
        ids.eur_curve: ZeroRateCurveFactory(extrapolation="flat"),
    }

    # -------------------------------
    # Vol panels (block data)
    # -------------------------------
    # Define the surface grid (expiry in years, strike in absolute space).
    expiries = np.array([0.25, 0.5, 1.0, 2.0], dtype=float)

    # Build a strike grid around an “initial” spot (for a stable grid in the dataset).
    strikes = 1.10 * np.exp(np.linspace(-0.25, 0.25, 31, dtype=float))

    # Build vol panel with shape [T,S,n_exp,n_k] using the spot_panel as input.
    eurusd_vol_panel: Panel = make_grid_vol_panel(
        dates=dates,
        n_s=int(n_scenarios),
        spot_panel=spot_panel,
        expiries=expiries,
        strikes=strikes,
        vol_grid=None,  # None => use builder’s default demo smile (skew + curvature)
    )

    # Store vol parameter panels keyed by MarketId.
    vol_param_panels: Dict[MarketId, Panel] = {
        ids.eurusd_vol: eurusd_vol_panel,
    }

    # Factory reconstructs GridVolSurface from a sliced [n_exp,n_k] block.
    vol_factories = {
        ids.eurusd_vol: GridVolFactory(expiries=expiries, strikes=strikes, extrapolation="flat"),
    }

    # -------------------------------
    # Wire everything into MarketDataset
    # -------------------------------
    dataset: MarketDataset = build_marketdataset(
        dates=dates,
        n_scenarios=int(n_scenarios),
        quote_panels=quote_panels,
        curve_param_panels=curve_param_panels,
        curve_factories=curve_factories,
        vol_param_panels=vol_param_panels,
        vol_factories=vol_factories,
        meta={"source": "demo", "purpose": "02_market_dataset_snapshot"},
        validate=True,  # run opinionated layout checks so mistakes fail loudly
    )

    return dataset, ids


# ======================================================================================
# 3) Snapshot + print diagnostics
# ======================================================================================

def print_snapshot_summary(
    *,
    market: Market,
    ids: ExampleIds,
    time_idx: int,
    scenario_idx: int,
) -> None:
    """
    Print a compact summary of a single Market snapshot.

    This is meant to show a user the “output object” they will price off.
    """
    # Read the scalar spot quote from the snapshot.
    spot = float(market.quote(ids.spot_eurusd))

    # Read curve objects from the snapshot and compute a few standard quantities.
    usd_curve = market.curve(ids.usd_curve)
    eur_curve = market.curve(ids.eur_curve)

    # Read vol surface object from the snapshot and sample some points.
    vol = market.vol_surface(ids.eurusd_vol)

    print("\n" + "=" * 95)
    print(f"MARKET SNAPSHOT  t={time_idx}  s={scenario_idx}  asof={market.asof}")
    print("=" * 95)
    print(f"Spot(EURUSD): {spot:.6f}")
    print("")
    print("USD curve diagnostics:")
    print(f"  df(1y)={usd_curve.df(1.0):.8f}   r(1y)={usd_curve.zero_rate(1.0):.6f}   f(1y,2y)={usd_curve.forward_rate(1.0, 2.0):.6f}")
    print("EUR curve diagnostics:")
    print(f"  df(1y)={eur_curve.df(1.0):.8f}   r(1y)={eur_curve.zero_rate(1.0):.6f}   f(1y,2y)={eur_curve.forward_rate(1.0, 2.0):.6f}")
    print("")
    print("EURUSD vol samples:")
    print(f"  sigma(1y, ATM~spot) = {vol.implied_vol(1.0, spot):.6f}")
    print(f"  sigma(1y,  90%spot) = {vol.implied_vol(1.0, 0.90 * spot):.6f}")
    print(f"  sigma(1y, 110%spot) = {vol.implied_vol(1.0, 1.10 * spot):.6f}")


# ======================================================================================
# 4) Plot snapshot objects (quotes / curves / vols)
# ======================================================================================

def plot_snapshot_objects(
    *,
    market: Market,
    ids: ExampleIds,
    time_idx: int,
    scenario_idx: int,
    plot_cfg: PlotConfig,
) -> None:
    """
    Plot a snapshot’s market objects.

    Implementation detail
    ---------------------
    We create all figures first, then do a single blocking show (via render_fig),
    which avoids “flash then disappear” behaviour in many IDEs.
    """
    # Turn interactive mode off so figures don’t auto-close / auto-refresh unexpectedly.
    plt.ioff()

    # Collect (filename, figure) pairs so we can optionally save all.
    figures: List[Tuple[str, plt.Figure]] = []

    # --- Quotes ---
    figures.append((
        "01_quotes.png",
        plot_quotes(market.quotes, title=f"Snapshot Quotes (t={time_idx}, s={scenario_idx})"),
    ))

    # --- Curves (USD) ---
    figures.append((
        "02_usd_df.png",
        plot_curve_df(market.curve(ids.usd_curve), title="USD curve: Discount Factor DF(t)"),
    ))
    figures.append((
        "03_usd_zero.png",
        plot_curve_zero_rate(market.curve(ids.usd_curve), title="USD curve: Zero Rate r(t)"),
    ))

    # --- Curves (EUR) ---
    figures.append((
        "04_eur_df.png",
        plot_curve_df(market.curve(ids.eur_curve), title="EUR curve: Discount Factor DF(t)"),
    ))
    figures.append((
        "05_eur_zero.png",
        plot_curve_zero_rate(market.curve(ids.eur_curve), title="EUR curve: Zero Rate r(t)"),
    ))

    # --- Vol surface ---
    vol = market.vol_surface(ids.eurusd_vol)

    figures.append((
        "06_vol_heatmap.png",
        plot_vol_surface_heatmap(vol, title="EURUSD vol surface (heatmap)"),
    ))
    figures.append((
        "07_vol_smiles.png",
        plot_vol_smile_slices(vol, title="EURUSD smile slices σ(T,K)"),
    ))
    figures.append((
        "08_vol_surface_3d.png",
        plot_vol_surface(vol, title="EURUSD vol surface (3D)"),
    ))

    # If saving is enabled, write every figure to disk (without showing).
    if plot_cfg.save:
        for filename, fig in figures:
            render_fig(
                fig,
                cfg=PlotConfig(
                    show=False,
                    save=True,
                    out_dir=Path(plot_cfg.out_dir),
                    filename=None,
                    dpi=plot_cfg.dpi,
                    block=False,
                    close=False,
                ),
                filename=filename,
            )

    # If showing is enabled, show figures.
    # We use render_fig so behaviour is consistent with the rest of the project.
    if plot_cfg.show:
        # Show all figures (matplotlib will manage multiple windows).
        for _, fig in figures:
            render_fig(fig, cfg=PlotConfig(show=True, save=False, out_dir=plot_cfg.out_dir, block=False))
        # One final blocking call keeps all windows alive in script mode.
        plt.show(block=bool(plot_cfg.block))


# ======================================================================================
# 5) Entry point
# ======================================================================================

def main() -> None:
    """Build a dataset, snapshot one point, print diagnostics, and plot the snapshot objects."""
    plot_cfg = PlotConfig(
        show=True,   # toggle on/off for interactive plot windows
        save=False,  # toggle on/off to save to disk
        out_dir=Path("outputs/marketdata/02_market_dataset_snapshot"),
        block=True,  # keeps windows alive in script/IDE runs
        close=False,
        dpi=160,
    )

    # Build the dataset (time × scenario × instruments).
    dataset, ids = build_demo_dataset(n_time=6, n_scenarios=3)

    # Select which (time,scenario) we want to snapshot.
    time_idx = 2
    scenario_idx = 1

    # Convert the dataset slice into a rich Market snapshot.
    market = dataset.snapshot(time_idx=time_idx, scenario_idx=scenario_idx)

    # Print a compact summary so a user can sanity check values quickly.
    print_snapshot_summary(market=market, ids=ids, time_idx=time_idx, scenario_idx=scenario_idx)

    # Plot the snapshot objects (quotes, curves, vols).
    plot_snapshot_objects(market=market, ids=ids, time_idx=time_idx, scenario_idx=scenario_idx, plot_cfg=plot_cfg)


if __name__ == "__main__":
    main()
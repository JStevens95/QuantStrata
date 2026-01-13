from __future__ import annotations

"""
Example 01: Build a single immutable Market snapshot (static, one asof date).

What this teaches:
- How to define canonical MarketId keys
- How to build:
    * quotes: MarketId -> Quote
    * curves: MarketId -> Curve (ZeroRateCurve)
    * vol surfaces: MarketId -> VolSurface (GridVolSurface)
- How to inspect the Market and plot each object category

This example is "snapshot-first":
You already have a single asof market snapshot and want to price on it immediately.
"""
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Tuple

from src.marketdata.core.ids import MarketId
from src.marketdata.core.interfaces import Quote, Curve, VolSurface
from src.marketdata.core.market import Market

from src.marketdata.curves.term_structure import ZeroRateCurve
from src.marketdata.surfaces.vol_surface import GridVolSurface

from src.core.reporting.plots.utils import PlotConfig, render_fig
from src.core.reporting.plots.marketdata.quotes import plot_quotes
from src.core.reporting.plots.marketdata.curves import plot_curve_df, plot_curve_zero_rate
from src.core.reporting.plots.marketdata.surfaces import (
    plot_vol_surface_heatmap, plot_vol_smile_slices, plot_vol_surface,
)


# =============================================================================
# 1) Canonical Market IDs
# =============================================================================

@dataclass(frozen=True, slots=True)
class ExampleMarketIds:
    """
    Strongly-typed container for the MarketId objects used in this example.

    Using a dataclass avoids “stringly-typed” mistakes and makes it obvious
    what instruments this example defines.
    """
    spot_eurusd: MarketId
    curve_usd_ois: MarketId
    curve_eur_ois: MarketId
    vol_eurusd: MarketId


def build_example_market_ids() -> ExampleMarketIds:
    """
    Construct canonical MarketId keys.

    Notes
    -----
    - qualifiers must be an iterable of (key,value) pairs, or omitted.
      Use tuples like: (("ccy","USD"),)
    """
    return ExampleMarketIds(
        spot_eurusd=MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD"),
        curve_usd_ois=MarketId(asset_class="IR", mkt_type="CURVE", name="USD.OIS", qualifiers=(("ccy", "USD"),)),
        curve_eur_ois=MarketId(asset_class="IR", mkt_type="CURVE", name="EUR.OIS", qualifiers=(("ccy", "EUR"),)),
        vol_eurusd=MarketId(asset_class="FX", mkt_type="VOL", name="EURUSD", qualifiers=(("cut", "NY"), ("conv", "delta25"))),
    )


# =============================================================================
# 2) Build market objects (quotes / curves / vols)
# =============================================================================

def build_example_quotes(ids: ExampleMarketIds) -> Dict[MarketId, Quote]:
    """
    Build the quote dictionary for Market(...).

    Quotes are scalar values wrapped in Quote(value=...).
    """
    # One scalar quote: spot EURUSD
    spot_value = 1.10
    return {ids.spot_eurusd: Quote(value=spot_value)}


def build_example_curves(ids: ExampleMarketIds) -> Dict[MarketId, Curve]:
    """
    Build curve objects for the Market snapshot.

    We use ZeroRateCurve:
    - Inputs: tenors + continuously compounded zero rates r(T)
    - Methods derived:
        df(t)         = exp(-r(t) * t)
        zero_rate(t)  = r(t) (interpolated)
        forward_rate  derived from DF ratio
    """
    # Tenor grid in year fractions.
    tenors = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0], dtype=float)

    # Simple synthetic zero curves for demo.
    usd_zero_rates = np.array([0.045, 0.046, 0.047, 0.048, 0.049, 0.050, 0.050], dtype=float)
    eur_zero_rates = np.array([0.032, 0.033, 0.034, 0.035, 0.036, 0.037, 0.037], dtype=float)

    # Return the MarketId -> Curve mapping.
    return {
        ids.curve_usd_ois: ZeroRateCurve(tenors=tenors, zero_rates=usd_zero_rates, extrapolation="flat"),
        ids.curve_eur_ois: ZeroRateCurve(tenors=tenors, zero_rates=eur_zero_rates, extrapolation="flat"),
    }


def _build_demo_vol_grid(
    expiries: np.ndarray,
    strikes: np.ndarray,
    *,
    spot: float,
    atm_by_expiry: Mapping[float, float],
) -> np.ndarray:
    """
    Build a synthetic implied-vol grid for demonstration purposes.

    We generate a mild smile / skew:
        vol(T,K) = ATM(T) + skew(T)*log(K/S) + curv(T)*log(K/S)^2

    This is NOT calibration — just a visually nice surface for plotting.
    """
    # Allocate output array: [n_expiry, n_strike]
    vol_grid = np.zeros((expiries.size, strikes.size), dtype=float)

    # log-moneyness, vector across strikes
    log_m = np.log(strikes / float(spot))

    # Fill each expiry slice
    for i, t in enumerate(expiries.tolist()):
        t = float(t)
        atm = float(atm_by_expiry[t])
        skew = -0.06 * np.sqrt(t)      # more skew at longer expiries
        curv = 0.18 / (1.0 + t)        # curvature decays with expiry

        vol_grid[i, :] = atm + skew * log_m + curv * (log_m ** 2)
        vol_grid[i, :] = np.maximum(vol_grid[i, :], 1e-8)  # enforce positivity floor

    return vol_grid


def build_example_vol_surfaces(ids: ExampleMarketIds, *, spot: float) -> Dict[MarketId, VolSurface]:
    """
    Build a GridVolSurface for the Market snapshot.

    GridVolSurface is a concrete implementation of VolSurface with:
      - expiries: 1D array
      - strikes:  1D array
      - implied_vols: 2D array [expiry, strike]
    """
    expiries = np.array([0.25, 0.5, 1.0, 2.0], dtype=float)

    # Build strikes around spot (log-spaced).
    strikes = spot * np.exp(np.linspace(-0.25, 0.25, 31, dtype=float))

    # Define an ATM term-structure (expiry -> ATM vol).
    atm_by_expiry = {0.25: 0.115, 0.5: 0.120, 1.0: 0.125, 2.0: 0.130}

    # Generate the full grid.
    vol_grid = _build_demo_vol_grid(expiries, strikes, spot=spot, atm_by_expiry=atm_by_expiry)

    return {
        ids.vol_eurusd: GridVolSurface(
            expiries=expiries,
            strikes=strikes,
            implied_vols=vol_grid,
            extrapolation="flat",
            strike_space="absolute",
            surface_id="FX.EURUSD.DEMO_GRID",
        )
    }


# =============================================================================
# 3) Assemble Market snapshot
# =============================================================================

def build_market_snapshot(*, asof: str = "2026-01-07") -> Tuple[Market, ExampleMarketIds]:
    """
    Construct a fully-populated Market snapshot.

    Returns
    -------
    market:
        The immutable Market object pricers will consume.
    ids:
        The ExampleMarketIds container so downstream code doesn’t hardcode strings.
    """
    ids = build_example_market_ids()

    # Build quotes (scalar market observables)
    quotes = build_example_quotes(ids)

    # Build curves (term structures)
    curves = build_example_curves(ids)

    # Pull spot from quotes (used to place strike grid around the spot)
    spot = float(quotes[ids.spot_eurusd].value)

    # Build vol surfaces (implied vol objects)
    vols = build_example_vol_surfaces(ids, spot=spot)

    # Assemble the Market snapshot
    market = Market(
        asof=asof,
        quotes=quotes,
        curves=curves,
        vols=vols,
        meta={"source": "demo", "purpose": "01_market_snapshot_static_build"},
    )
    return market, ids


# =============================================================================
# 4) Reporting helpers (print + plots)
# =============================================================================

def print_market_summary(market: Market, ids: ExampleMarketIds) -> None:
    """
    Print a small diagnostic summary so users can sanity-check objects quickly.
    """
    print("\n" + "=" * 90)
    print(f"MARKET SNAPSHOT  asof={market.asof}")
    print("=" * 90)

    # Quotes
    print("\n[Quotes]")
    for mid, q in market.quotes.items():
        print(f"  {mid.key():45s}  value={q.value:.8f}")

    # Curves
    print("\n[Curves] (diagnostics at 1y and forward(1y,2y))")
    for mid, curve in market.curves.items():
        df_1y = curve.df(1.0)
        r_1y = curve.zero_rate(1.0)
        f_1y_2y = curve.forward_rate(1.0, 2.0)
        print(f"  {mid.key():45s}  df(1y)={df_1y:.8f}  r(1y)={r_1y:.6f}  f(1y,2y)={f_1y_2y:.6f}")

    # Vol surface samples
    print("\n[Vol Surfaces] (samples at 1y for K=spot, 90%spot, 110%spot)")
    spot = market.quote(ids.spot_eurusd)
    vol = market.vol_surface(ids.vol_eurusd)
    print(f"  {ids.vol_eurusd.key():45s}  "
          f"sigma(1y,ATM)={vol.implied_vol(1.0, spot):.6f}  "
          f"sigma(1y,90%)={vol.implied_vol(1.0, 0.90 * spot):.6f}  "
          f"sigma(1y,110%)={vol.implied_vol(1.0, 1.10 * spot):.6f}")


def plot_market_objects(market: Market, ids: ExampleMarketIds, *, plot_cfg: PlotConfig) -> None:
    """
    Generate plots for each Market object category.

    Design choice:
    - create all figures first
    - optionally save them
    - call ONE plt.show(block=True) at the end (prevents IDE “flash then disappear”)
    """
    # Ensure we control when figures show (important for IDE scripts).
    plt.ioff()

    # ---- Build figures ----
    figures = [
        ("01_quotes.png", plot_quotes(market.quotes, title="Market Quotes (scalar)")),
        ("02_usd_df.png", plot_curve_df(market.curve(ids.curve_usd_ois), title="USD OIS: Discount Factor DF(t)")),
        ("03_usd_zero.png", plot_curve_zero_rate(market.curve(ids.curve_usd_ois), title="USD OIS: Zero Rate r(t)")),
        ("04_eur_df.png", plot_curve_df(market.curve(ids.curve_eur_ois), title="EUR OIS: Discount Factor DF(t)")),
        ("05_eur_zero.png", plot_curve_zero_rate(market.curve(ids.curve_eur_ois), title="EUR OIS: Zero Rate r(t)")),
    ]

    vol = market.vol_surface(ids.vol_eurusd)
    figures.extend([
        ("06_vol_heatmap.png", plot_vol_surface_heatmap(vol, title="EURUSD: Grid Vol Surface (heatmap)")),
        ("07_vol_smiles.png", plot_vol_smile_slices(vol, title="EURUSD: Smile slices σ(T,K) vs K")),
        ("08_vol_surface_3d.png", plot_vol_surface(vol, title="EURUSD: Grid Vol Surface (3D)")),
    ])

    # ---- Save (optional) ----
    if plot_cfg.save:
        for filename, fig in figures:
            render_fig(fig, cfg=PlotConfig(show=False, save=True, out_dir=plot_cfg.out_dir, dpi=plot_cfg.dpi), filename=filename)

    # ---- Show (optional) ----
    if plot_cfg.show:
        plt.show(block=plot_cfg.block)


def main() -> None:
    """
    Entry point for running in IDE or terminal.

    Toggle plotting behavior by editing PlotConfig below.
    """
    plot_cfg = PlotConfig(
        show=True,
        save=False,
        out_dir=Path("outputs/marketdata/01_market_snapshot_static_build"),
        block=True,
    )

    market, ids = build_market_snapshot(asof="2026-01-07")
    print_market_summary(market, ids)
    plot_market_objects(market, ids, plot_cfg=plot_cfg)


if __name__ == "__main__":
    main()
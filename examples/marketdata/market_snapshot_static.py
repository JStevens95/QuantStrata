from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np

from src.marketdata.core.ids import MarketId
from src.marketdata.core.interfaces import Quote, Curve, VolSurface
from src.marketdata.core.market import Market

from src.marketdata.curves.term_structure import ZeroRateCurve
from src.marketdata.surfaces.vol_surface import GridVolSurface

from src.core.reporting.plots.utils import display_fig, save_fig
from src.core.reporting.plots.marketdata.curves import plot_curve_df, plot_curve_zero_rate
from src.core.reporting.plots.marketdata.quotes import plot_quotes
from src.core.reporting.plots.marketdata.surfaces import plot_vol_surface_heatmap, plot_vol_smile_slices, plot_vol_surface


# =============================================================================
# IDs
# =============================================================================

@dataclass(frozen=True, slots=True)
class ExampleIds:
    spot: MarketId
    df_dom: MarketId
    df_for: MarketId
    fx_vol: MarketId


def build_market_ids() -> ExampleIds:
    """
    Adjust ONLY this function if your MarketId constructor differs.
    """
    return ExampleIds(
        spot=MarketId(
            asset_class="FX", mkt_type="SPOT", name="EURUSD", qualifiers=None
        ),
        df_dom=MarketId(
            asset_class="IR", mkt_type="CURVE", name="USD.OIS", qualifiers={"ccy": "USD"}
        ),
        df_for=MarketId(
            asset_class="IR", mkt_type="CURVE", name="EUR.OIS", qualifiers={"ccy": "EUR"}
        ),
        fx_vol=MarketId(
            asset_class="FX", mkt_type="VOL", name="EURUSD", qualifiers=(("cut", "NY"), ("conv", "delta25"))
        ),
    )


# =============================================================================
# Build market objects (quotes / curves / vols)
# =============================================================================

def build_quotes(ids: ExampleIds) -> Dict[MarketId, Quote]:
    # One scalar quote (spot)
    return {ids.spot: Quote(value=1.10)}


def build_curves(ids: ExampleIds) -> Dict[MarketId, Curve]:
    # “ZeroRateCurve”: store r(T); derive df(T) & forwards from it
    tenors = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0], dtype=float)

    usd_ois_zero = np.array([0.045, 0.046, 0.047, 0.048, 0.049, 0.050, 0.050], dtype=float)
    eur_ois_zero = np.array([0.032, 0.033, 0.034, 0.035, 0.036, 0.037, 0.037], dtype=float)

    return {
        ids.df_dom: ZeroRateCurve(tenors=tenors, zero_rates=usd_ois_zero, extrapolation="flat"),
        ids.df_for: ZeroRateCurve(tenors=tenors, zero_rates=eur_ois_zero, extrapolation="flat"),
    }


def _build_example_vol_grid(
    expiries: np.ndarray,
    strikes: np.ndarray,
    atm_by_t: Mapping[float, float],
    spot: float,
) -> np.ndarray:
    """
    Small synthetic smile:
      vol(T,K) = ATM(T) + skew(T) * log(K/spot) + curvature(T) * log(K/spot)^2
    This is only for demo/plots; calibration example is separate.
    """
    vols = np.zeros((expiries.size, strikes.size), dtype=float)
    log_m = np.log(strikes / float(spot))

    for i, t in enumerate(expiries.tolist()):
        atm = float(atm_by_t[float(t)])
        skew = -0.06 * np.sqrt(float(t))        # mild left skew
        curv = 0.18 / (1.0 + float(t))          # mild curvature

        vols[i, :] = atm + skew * log_m + curv * (log_m ** 2)
        vols[i, :] = np.maximum(vols[i, :], 1e-8)
    return vols


def build_vol_surfaces(ids: ExampleIds, spot: float) -> Dict[MarketId, VolSurface]:
    expiries = np.array([0.25, 0.5, 1.0, 2.0], dtype=float)
    strikes = spot * np.exp(np.linspace(-0.25, 0.25, 31, dtype=float))

    atm_by_t = {0.25: 0.115, 0.5: 0.120, 1.0: 0.125, 2.0: 0.130}
    vol_grid = _build_example_vol_grid(expiries=expiries, strikes=strikes, atm_by_t=atm_by_t, spot=spot)

    return {
        ids.fx_vol: GridVolSurface(
            expiries=expiries,
            strikes=strikes,
            implied_vols=vol_grid,
            extrapolation="flat",
            strike_space="absolute",
            surface_id="FX.EURUSD.DEMO_GRID",
        )
    }


# =============================================================================
# Assemble Market snapshot
# =============================================================================

def build_market_snapshot(asof: str = "2026-01-07") -> Tuple[Market, ExampleIds]:
    ids = build_market_ids()

    quotes = build_quotes(ids)
    curves = build_curves(ids)
    spot = float(quotes[ids.spot].value)

    vols = build_vol_surfaces(ids, spot=spot)

    market = Market(
        asof=asof,
        quotes=quotes,
        curves=curves,
        vols=vols,
        meta={"source": "demo", "purpose": "market_snapshot_static_build"},
    )
    return market, ids


# =============================================================================
# Print helpers (make the example “readable”)
# =============================================================================

def print_market_summary(market: Market, ids: ExampleIds) -> None:
    print("\n" + "=" * 80)
    print(f"MARKET SNAPSHOT  asof={market.asof}")
    print("=" * 80)

    # Quotes
    print("\n[Quotes]")
    for mid, q in market.quotes.items():
        print(f"  {mid.key():40s}  value={q.value:.8f}")

    # Curves (show a few diagnostics)
    print("\n[Curves]")
    for mid, c in market.curves.items():
        df_1y = c.df(1.0)
        r_1y = c.zero_rate(1.0)
        f_1y_2y = c.forward_rate(1.0, 2.0)
        print(f"  {mid.key():40s}  df(1y)={df_1y:.8f}  r(1y)={r_1y:.6f}  f(1y,2y)={f_1y_2y:.6f}")

    # Vol surface (sample a couple of points)
    print("\n[Vol Surfaces]")
    for mid, v in market.vols.items():
        s = market.quote(ids.spot)
        sig_atm_1y = v.implied_vol(1.0, s)
        sig_low_1y = v.implied_vol(1.0, s * 0.90)
        sig_high_1y = v.implied_vol(1.0, s * 1.10)
        print(f"  {mid.key():40s}  sigma(1y,ATM)={sig_atm_1y:.6f}  sigma(1y,90%)={sig_low_1y:.6f}  sigma(1y,110%)={sig_high_1y:.6f}")


# =============================================================================
# Plot helpers (one plot per object “category”)
# =============================================================================

def _maybe_render(fig, *, show: bool, save: bool, out_dir: Path, filename: str) -> None:
    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
        save_fig(fig, path=str(out_dir / filename))
    if show:
        display_fig(fig, block=False)


def plot_market_objects(market: Market, ids: ExampleIds, *, show: bool, save: bool, out_dir: Path) -> None:
    # Quotes
    fig_q = plot_quotes(market.quotes, title="Market Quotes (scalar)")
    _maybe_render(fig_q, show=show, save=save, out_dir=out_dir, filename="01_quotes.png")

    # Domestic / foreign curves
    fig_df_dom = plot_curve_df(market.curve(ids.df_dom), title="USD OIS: Discount Factor DF(t)")
    _maybe_render(fig_df_dom, show=show, save=save, out_dir=out_dir, filename="02_curve_usd_df.png")

    fig_z_dom = plot_curve_zero_rate(market.curve(ids.df_dom), title="USD OIS: Zero Rate r(t)")
    _maybe_render(fig_z_dom, show=show, save=save, out_dir=out_dir, filename="03_curve_usd_zero.png")

    fig_df_for = plot_curve_df(market.curve(ids.df_for), title="EUR OIS: Discount Factor DF(t)")
    _maybe_render(fig_df_for, show=show, save=save, out_dir=out_dir, filename="04_curve_eur_df.png")

    fig_z_for = plot_curve_zero_rate(market.curve(ids.df_for), title="EUR OIS: Zero Rate r(t)")
    _maybe_render(fig_z_for, show=show, save=save, out_dir=out_dir, filename="05_curve_eur_zero.png")

    # Vol surface
    vol = market.vol_surface(ids.fx_vol)
    fig_hm = plot_vol_surface_heatmap(vol, title="EURUSD: Grid Vol Surface (heatmap)")
    _maybe_render(fig_hm, show=show, save=save, out_dir=out_dir, filename="06_vol_surface_heatmap.png")

    fig_smile = plot_vol_smile_slices(vol, title="EURUSD: Smile slices σ(T,K) vs K")
    _maybe_render(fig_smile, show=show, save=save, out_dir=out_dir, filename="07_vol_surface_smiles.png")

    fig_surface = plot_vol_surface(vol, title="EURUSD: Grid Vol Surface (3D)")
    _maybe_render(fig_surface, show=show, save=save, out_dir=out_dir, filename="08_vol_surface_3d.png")


# =============================================================================
# Entry point (IDE friendly)
# =============================================================================

def main(show: bool = True, save: bool = False) -> None:
    market, ids = build_market_snapshot()
    print_market_summary(market, ids)
    plot_market_objects(market, ids, show=show, save=save, out_dir=Path("outputs/marketdata_example_01"))


if __name__ == "__main__":
    main(show=True, save=False)
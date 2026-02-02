#!/usr/bin/env python3
"""
QuantStrata Example — Single FX European Vanilla Pricing

This example prices a single European FX vanilla option using:
  1) Black–Scholes–Merton (closed form)
  2) Monte Carlo
  3) Finite Difference (PDE) [optional]

It prints a clear summary, runs convergence sweeps, and produces plots.

Design rules:
- Use QuantStrata Market/MarketId objects (no ad-hoc dicts of floats).
- Treat BSM as the benchmark for vanilla Europeans under flat vol.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Ensure repo root is on sys.path (script may be launched from anywhere).
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]  # .../QuantStrata
sys.path.insert(0, str(REPO_ROOT))

# QuantStrata imports
from src.marketdata.core.ids import MarketId
from src.marketdata.core.interfaces import Quote
from src.marketdata.core.market import Market
from src.marketdata.curves.term_structure import FlatZeroRateCurve
from src.marketdata.surfaces.vol_surface import FlatVolSurface

from src.instruments.fx.options.vanilla import EuropeanFxVanillaOption
from src.pricers.fx.european_bsm import FxEuropeanVanillaBsmPricer
from src.pricers.fx.european_bsm_mc import FxEuropeanVanillaMcPricer

# FD pricer may not exist / may be WIP: import defensively.
try:
    from src.pricers.fx.european_bsm_fde import FxEuropeanVanillaFdPricer  # type: ignore
except Exception:  # pragma: no cover
    FxEuropeanVanillaFdPricer = None  # type: ignore


# -----------------------------------------------------------------------------
# Plot configuration (keep consistent across examples)
# -----------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.figsize": (12, 5),
        "font.size": 11,
        "axes.titlesize": 13,
        "lines.linewidth": 2,
    }
)

COLORS = {"bsm": "#2E86AB", "mc": "#E94F37", "fd": "#8B5CF6"}


# -----------------------------------------------------------------------------
# Small helpers (formatting + safe FD)
# -----------------------------------------------------------------------------
def _banner(title: str) -> None:
    print("=" * 80)
    print(title)
    print("=" * 80)


def _safe_fd_price(option, market, n_spot, n_time, *, verbose: bool = False):
    if FxEuropeanVanillaFdPricer is None:
        if verbose:
            print("[FD] Import failed: FxEuropeanVanillaFdPricer is None")
        return None
    try:
        pricer = FxEuropeanVanillaFdPricer(n_spot=n_spot, n_time=n_time)
        return float(pricer.price(option, market))
    except Exception as e:
        if verbose:
            print(f"[FD] Failed for grid {n_spot}x{n_time}: {type(e).__name__}: {e}")
        return None


@dataclass(frozen=True, slots=True)
class ConvergenceRow:
    x: int
    pv: float
    abs_err: float
    rel_err: float


# -----------------------------------------------------------------------------
# 1) Setup: Define Market + Option
# -----------------------------------------------------------------------------
def build_market_and_option() -> Tuple[Market, EuropeanFxVanillaOption, dict]:
    """
    Construct:
      - Market snapshot (spot, 2 curves, flat vol)
      - European FX vanilla option referencing MarketIds

    Returns (market, option, params_dict).
    """
    asof = "2026-01-28"

    # Market IDs (canonical within QuantStrata)
    eurusd_spot_id = MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD")
    usd_curve_id = MarketId(asset_class="IR", mkt_type="CURVE", name="USD_OIS")
    eur_curve_id = MarketId(asset_class="IR", mkt_type="CURVE", name="EUR_OIS")
    eurusd_vol_id = MarketId(asset_class="FX", mkt_type="VOL", name="EURUSD")

    # Market levels
    spot = 1.0850
    r_domestic = 0.05  # USD
    r_foreign = 0.02   # EUR
    vol = 0.10

    # Option terms
    strike = 1.1000
    expiry = 1.0
    notional = 1_000_000.0
    option_type = "call"

    # Build Market using project objects
    market = Market(
        asof=asof,
        quotes={eurusd_spot_id: Quote(value=float(spot))},
        curves={
            usd_curve_id: FlatZeroRateCurve(continuously_compounded_rate=float(r_domestic)),
            eur_curve_id: FlatZeroRateCurve(continuously_compounded_rate=float(r_foreign)),
        },
        vols={eurusd_vol_id: FlatVolSurface(sigma=float(vol))},
        meta={"source": "example", "name": "flat market"},
    )

    # IMPORTANT: use the actual EuropeanFxVanillaOption signature in your repo
    option = EuropeanFxVanillaOption(
        option_type=option_type,
        strike=float(strike),
        expiry=float(expiry),
        notional=float(notional),
        spot_id=eurusd_spot_id,
        vol_id=eurusd_vol_id,
        domestic_curve_id=usd_curve_id,
        foreign_curve_id=eur_curve_id,
    )

    params = {
        "asof": asof,
        "spot_id": eurusd_spot_id,
        "vol_id": eurusd_vol_id,
        "curve_dom_id": usd_curve_id,
        "curve_for_id": eur_curve_id,
        "spot": float(spot),
        "strike": float(strike),
        "expiry": float(expiry),
        "notional": float(notional),
        "r_domestic": float(r_domestic),
        "r_foreign": float(r_foreign),
        "vol": float(vol),
        "option_type": option_type,
    }
    return market, option, params


# -----------------------------------------------------------------------------
# 2) Run pricing + convergence
# -----------------------------------------------------------------------------
def run(save_fig: bool = False) -> None:
    market, option, p = build_market_and_option()

    _banner("QuantStrata Example — Single FX European Vanilla Pricing")
    print(f"As-of:        {p['asof']}")
    print(f"SpotId:       {p['spot_id'].key()}")
    print(f"VolId:        {p['vol_id'].key()}")
    print(f"Curve rd:     {p['curve_dom_id'].key()}  (domestic)")
    print(f"Curve rf:     {p['curve_for_id'].key()}  (foreign)")
    print("-" * 80)
    print(f"Spot S0:      {p['spot']:.6f}")
    print(f"Strike K:     {p['strike']:.6f}")
    print(f"Expiry T:     {p['expiry']:.6f} years")
    print(f"rd / rf:      {p['r_domestic']:.4%} / {p['r_foreign']:.4%}")
    print(f"Vol sigma:    {p['vol']:.4%}")
    print(f"Notional:     {p['notional']:,.0f}")
    print(f"Option type:  {p['option_type']}")
    print("-" * 80)

    # Forward under continuous carry: F = S * exp((rd-rf)T)
    fwd = p["spot"] * float(np.exp((p["r_domestic"] - p["r_foreign"]) * p["expiry"]))
    print(f"Forward F:    {fwd:.6f}")
    print(f"Moneyness:    K/F = {p['strike'] / fwd:.4f}")
    print()

    # --- BSM (benchmark) ---
    bsm = FxEuropeanVanillaBsmPricer()
    bsm_pv = float(bsm.price(option, market))
    bsm_greeks = bsm.greeks(option, market)

    # --- MC ---
    mc = FxEuropeanVanillaMcPricer(n_paths=100_000, seed=42)
    mc_pv = float(mc.price(option, market))

    # --- FD (optional) ---
    fd_pv = _safe_fd_price(option, market, n_spot=200, n_time=100, verbose=True)

    print("PVs")
    print("-" * 80)
    print(f"BSM PV:       {bsm_pv:,.6f}")
    print(f"MC  PV:       {mc_pv:,.6f}")
    print(f"MC error:     {abs(mc_pv - bsm_pv):,.6f}  ({abs(mc_pv / bsm_pv - 1.0) * 100.0:.4f}%)")
    if fd_pv is None:
        print("FD  PV:       [skipped / unavailable]")
    else:
        print(f"FD  PV:       {fd_pv:,.6f}")
        print(f"FD error:     {abs(fd_pv - bsm_pv):,.6f}  ({abs(fd_pv / bsm_pv - 1.0) * 100.0:.4f}%)")

    print("\nGreeks (BSM)")
    print("-" * 80)
    print(f"delta:        {float(bsm_greeks['delta']): .6f}")
    print(f"gamma:        {float(bsm_greeks['gamma']): .6f}")
    print(f"vega:         {float(bsm_greeks['vega']): .6f}")
    print(f"theta:        {float(bsm_greeks['theta']): .6f}")
    # FX rho is often split; use .get for backward compatibility
    print(f"rho_domestic: {float(bsm_greeks.get('rho_domestic', 0.0)): .6f}")
    print(f"rho_foreign:  {float(bsm_greeks.get('rho_foreign', 0.0)): .6f}")

    # --- MC convergence sweep ---
    path_counts = [1_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000]
    mc_rows: List[ConvergenceRow] = []
    for n in path_counts:
        pv_n = float(FxEuropeanVanillaMcPricer(n_paths=n, seed=42).price(option, market))
        abs_err = abs(pv_n - bsm_pv)
        rel_err = abs_err / abs(bsm_pv) if bsm_pv != 0.0 else float("nan")
        mc_rows.append(ConvergenceRow(x=n, pv=pv_n, abs_err=abs_err, rel_err=rel_err))

    print("\nMC convergence")
    print("-" * 80)
    print(f"{'Paths':>12} {'PV':>18} {'AbsErr':>18} {'RelErr':>12}")
    for r in mc_rows:
        print(f"{r.x:>12,} {r.pv:>18,.6f} {r.abs_err:>18,.6f} {r.rel_err*100:>11.4f}%")

    # --- FD convergence sweep (optional) ---
    grid_sizes: List[Tuple[int, int]] = [(50, 25), (100, 50), (200, 100), (400, 200)]
    fd_rows: List[ConvergenceRow] = []
    for n_spot, n_time in grid_sizes:
        pv = _safe_fd_price(option, market, n_spot=n_spot, n_time=n_time)
        if pv is None:
            continue
        abs_err = abs(pv - bsm_pv)
        rel_err = abs_err / abs(bsm_pv) if bsm_pv != 0.0 else float("nan")
        fd_rows.append(ConvergenceRow(x=n_spot, pv=pv, abs_err=abs_err, rel_err=rel_err))

    print("\nFD convergence")
    print("-" * 80)
    if not fd_rows:
        print("FD convergence skipped (FD pricer unavailable or failed).")
    else:
        print(f"{'n_spot':>12} {'PV':>18} {'AbsErr':>18} {'RelErr':>12}")
        for row, (n_spot, n_time) in zip(fd_rows, grid_sizes):
            print(f"{n_spot:>12} {row.pv:>18,.6f} {row.abs_err:>18,.6f} {row.rel_err*100:>11.4f}%")

    # -----------------------------------------------------------------------------
    # 3) Visualisation (same as your original script, but consistent & robust)
    # -----------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # (1) Call unit price vs strike (BSM)
    ax = axes[0]
    strikes_plot = np.linspace(0.95, 1.25, 20)
    unit_prices = []
    for k in strikes_plot:
        # Rebuild option with same MarketIds but new strike
        opt_k = EuropeanFxVanillaOption(
            option_type=p["option_type"],
            strike=float(k),
            expiry=p["expiry"],
            notional=p["notional"],
            spot_id=p["spot_id"],
            vol_id=p["vol_id"],
            domestic_curve_id=p["curve_dom_id"],
            foreign_curve_id=p["curve_for_id"],
        )
        unit_prices.append(float(bsm.price(opt_k, market)) / p["notional"])

    ax.plot(strikes_plot, unit_prices, "-", color=COLORS["bsm"], linewidth=2, label="BSM (unit PV)")
    ax.axvline(p["spot"], color="gray", linestyle="--", alpha=0.5, label=f"Spot = {p['spot']:.4f}")
    ax.axvline(fwd, color="gray", linestyle=":", alpha=0.5, label=f"Forward = {fwd:.4f}")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Unit Price (PV / Notional)")
    ax.set_title("Call Price vs Strike")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (2) MC convergence: absolute error vs paths (log-log)
    ax = axes[1]
    mc_abs_errs = np.array([r.abs_err for r in mc_rows], dtype=float)
    ax.loglog(path_counts, mc_abs_errs, "o-", color=COLORS["mc"], linewidth=2, markersize=8)

    # Reference O(1/sqrt(N)) line anchored at the first point
    ref_x = np.array(path_counts, dtype=float)
    ref_y = mc_abs_errs[0] * np.sqrt(ref_x[0]) / np.sqrt(ref_x)
    ax.loglog(ref_x, ref_y, "--", color="gray", alpha=0.7, label=r"$O(1/\sqrt{N})$")

    ax.set_xlabel("Number of Paths")
    ax.set_ylabel("Absolute Error (USD)")
    ax.set_title("MC Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (3) FD convergence (if available)
    ax = axes[2]
    if not fd_rows:
        ax.text(0.5, 0.5, "FD unavailable", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        grid_labels = [f"{s}×{t}" for s, t in grid_sizes[: len(fd_rows)]]
        fd_abs_errs = [r.abs_err for r in fd_rows]
        ax.semilogy(range(len(fd_abs_errs)), fd_abs_errs, "o-", color=COLORS["fd"], linewidth=2, markersize=8)
        ax.set_xticks(range(len(fd_abs_errs)))
        ax.set_xticklabels(grid_labels)
        ax.set_xlabel("Grid Size (Spot × Time)")
        ax.set_ylabel("Absolute Error (USD)")
        ax.set_title("FD Convergence")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    out_path = Path.cwd() / "single_fx_vanilla_pricing.png"
    if save_fig:
        print(f"\nFigure saved to: {out_path}")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    run(save_fig=False)
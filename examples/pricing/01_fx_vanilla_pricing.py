#!/usr/bin/env python3
"""
===============================================================================
FX European Vanilla Option Pricing
===============================================================================

This example prices a single European FX vanilla option using multiple methods:
- Black-Scholes-Merton (closed-form analytic)
- Monte Carlo simulation
- Finite Difference (PDE) method

Learning Objectives
-------------------
1. **BSM Pricing**: Understand the Garman-Kohlhagen extension for FX
2. **Method Comparison**: Compare analytic, MC, and FDE approaches
3. **Convergence Analysis**: Study how numerical methods converge to BSM
4. **Greeks Computation**: Extract risk sensitivities from BSM

Mathematical Framework
----------------------
For FX options under Garman-Kohlhagen:

    C = S·e^(-r_f·T)·N(d1) - K·e^(-r_d·T)·N(d2)
    P = K·e^(-r_d·T)·N(-d2) - S·e^(-r_f·T)·N(-d1)

Where:
    d1 = [ln(S/K) + (r_d - r_f + σ²/2)T] / (σ√T)
    d2 = d1 - σ√T

Forward price under continuous carry:
    F = S · exp((r_d - r_f) · T)

Production Context
------------------
At a hedge fund:
- BSM is the benchmark for European vanilla options
- MC is used for path-dependent exotics or complex payoffs
- FDE is used for American exercise and early exercise boundaries
- Greeks drive hedging decisions and risk limits

Prerequisites
-------------
- Examples in fundamentals/ folder
- Understanding of Black-Scholes model

Run This Example
----------------
    cd /path/to/QuantStrata
    PYTHONPATH=. python examples/pricing/01_fx_vanilla_pricing.py

Author: QuantStrata Team
===============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

from __future__ import annotations  # Enable modern type hints

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# Path setup: Ensure imports work when running as script
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# -----------------------------------------------------------------------------
# QuantStrata imports
# -----------------------------------------------------------------------------
from src.marketdata.core.ids import MarketId
from src.marketdata.core.interfaces import Quote
from src.marketdata.core.market import Market
from src.marketdata.curves.term_structure import FlatZeroRateCurve
from src.marketdata.surfaces.vol_surface import FlatVolSurface

from src.instruments.fx.options.vanilla import FxVanillaEuropeanOption
from src.pricers.fx.european_bsm import FxVanillaEuropeanOptionBsmPricer

# Try to import MC pricer (may have different name)
try:
    from src.pricers.fx.european_mc import FxVanillaEuropeanOptionMcPricer
except ImportError:
    try:
        from src.pricers.fx.european_bsm_mc import FxVanillaEuropeanOptionMcPricer
    except ImportError:
        FxVanillaEuropeanOptionMcPricer = None  # type: ignore

# Try to import FD pricer (optional)
try:
    from src.pricers.fx.european_fde import FxVanillaEuropeanOptionFdPricer
except ImportError:
    try:
        from src.pricers.fx.european_bsm_fde import FxVanillaEuropeanOptionFdPricer
    except ImportError:
        FxVanillaEuropeanOptionFdPricer = None  # type: ignore


# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

ENABLE_PLOTTING = True

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available - plotting disabled")


# =============================================================================
# CONSTANTS
# =============================================================================

# Color scheme for plots
COLORS = {
    "bsm": "#2E86AB",
    "mc": "#E94F37",
    "fd": "#8B5CF6",
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True, slots=True)
class ConvergenceRow:
    """
    A single row of convergence data.
    
    Attributes
    ----------
    x : int
        The x-axis value (e.g., number of paths or grid size).
    pv : float
        The computed present value.
    abs_err : float
        Absolute error vs benchmark.
    rel_err : float
        Relative error vs benchmark.
    """
    x: int
    pv: float
    abs_err: float
    rel_err: float


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def safe_fd_price(
    option: FxVanillaEuropeanOption,
    market: Market,
    n_spot: int,
    n_time: int,
    verbose: bool = False,
) -> Optional[float]:
    """
    Safely price using FD pricer, returning None if unavailable or failed.
    
    Parameters
    ----------
    option : FxVanillaEuropeanOption
        The option to price.
    market : Market
        Market snapshot.
    n_spot : int
        Number of spot grid points.
    n_time : int
        Number of time steps.
    verbose : bool
        Whether to log failures.
    
    Returns
    -------
    Optional[float]
        FD price or None if failed.
    """
    if FxVanillaEuropeanOptionFdPricer is None:
        if verbose:
            logger.warning("FD pricer not available")
        return None
    
    try:
        pricer = FxVanillaEuropeanOptionFdPricer(n_spot=n_spot, n_time=n_time)
        return float(pricer.price(option, market))
    except Exception as e:
        if verbose:
            logger.warning(f"FD failed for grid {n_spot}x{n_time}: {e}")
        return None


# =============================================================================
# SECTION 1: Setup - Market and Option
# =============================================================================

def build_market_and_option() -> Tuple[Market, FxVanillaEuropeanOption, dict]:
    """
    Construct the market snapshot and option instrument.
    
    Returns
    -------
    Tuple[Market, FxVanillaEuropeanOption, dict]
        Market, option, and parameters dictionary.
    
    Market Setup
    ------------
    - Spot: 1.0850 EUR/USD
    - USD Rate (domestic): 5%
    - EUR Rate (foreign): 2%
    - Volatility: 10%
    
    Option Terms
    ------------
    - Strike: 1.1000 (OTM call)
    - Expiry: 1 year
    - Notional: 1,000,000 EUR
    - Type: Call
    
    Production Notes
    ----------------
    - Always use MarketId for identifying market data
    - Use FlatZeroRateCurve for testing, ZeroRateCurve for production
    - Notional is in foreign currency (EUR) for FX options
    """
    asof = "2026-01-28"
    
    # -------------------------------------------------------------------------
    # Define Market IDs
    # These follow the convention: asset_class.mkt_type.name
    # -------------------------------------------------------------------------
    eurusd_spot_id = MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD")
    usd_curve_id = MarketId(asset_class="IR", mkt_type="CURVE", name="USD_OIS")
    eur_curve_id = MarketId(asset_class="IR", mkt_type="CURVE", name="EUR_OIS")
    eurusd_vol_id = MarketId(asset_class="FX", mkt_type="VOL", name="EURUSD")
    
    # -------------------------------------------------------------------------
    # Market parameters
    # -------------------------------------------------------------------------
    spot = 1.0850          # EUR/USD spot rate
    r_domestic = 0.05      # USD (domestic) risk-free rate
    r_foreign = 0.02       # EUR (foreign) risk-free rate
    vol = 0.10             # Implied volatility (10%)
    
    # -------------------------------------------------------------------------
    # Option terms
    # -------------------------------------------------------------------------
    strike = 1.1000        # OTM call (spot < strike)
    expiry = 1.0           # 1 year to expiry
    notional = 1_000_000.0 # 1 million EUR
    option_type = "call"
    
    # -------------------------------------------------------------------------
    # Build Market using QuantStrata objects
    # -------------------------------------------------------------------------
    market = Market(
        asof=asof,
        quotes={
            eurusd_spot_id: Quote(value=float(spot)),
        },
        curves={
            usd_curve_id: FlatZeroRateCurve(continuously_compounded_rate=float(r_domestic)),
            eur_curve_id: FlatZeroRateCurve(continuously_compounded_rate=float(r_foreign)),
        },
        vols={
            eurusd_vol_id: FlatVolSurface(sigma=float(vol)),
        },
        meta={
            "source": "example",
            "description": "Flat market for vanilla FX option pricing",
        },
    )
    
    # -------------------------------------------------------------------------
    # Create the option instrument
    # -------------------------------------------------------------------------
    option = FxVanillaEuropeanOption(
        option_type=option_type,
        strike=float(strike),
        expiry=float(expiry),
        notional=float(notional),
        spot_id=eurusd_spot_id,
        vol_id=eurusd_vol_id,
        domestic_curve_id=usd_curve_id,
        foreign_curve_id=eur_curve_id,
    )
    
    # -------------------------------------------------------------------------
    # Parameters dictionary for convenience
    # -------------------------------------------------------------------------
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


# =============================================================================
# SECTION 2: Pricing and Analysis
# =============================================================================

def run_pricing(market: Market, option: FxVanillaEuropeanOption, params: dict) -> None:
    """
    Run pricing with BSM, MC, and FD methods and analyze convergence.
    
    Parameters
    ----------
    market : Market
        Market snapshot.
    option : FxVanillaEuropeanOption
        The option to price.
    params : dict
        Parameters dictionary.
    """
    logger.info("=" * 80)
    logger.info("QuantStrata Example — Single FX European Vanilla Pricing")
    logger.info("=" * 80)
    
    # -------------------------------------------------------------------------
    # Display setup
    # -------------------------------------------------------------------------
    logger.info(f"As-of:        {params['asof']}")
    logger.info(f"SpotId:       {params['spot_id'].key()}")
    logger.info(f"VolId:        {params['vol_id'].key()}")
    logger.info(f"Curve rd:     {params['curve_dom_id'].key()}  (domestic)")
    logger.info(f"Curve rf:     {params['curve_for_id'].key()}  (foreign)")
    logger.info("-" * 80)
    logger.info(f"Spot S0:      {params['spot']:.6f}")
    logger.info(f"Strike K:     {params['strike']:.6f}")
    logger.info(f"Expiry T:     {params['expiry']:.6f} years")
    logger.info(f"rd / rf:      {params['r_domestic']:.4%} / {params['r_foreign']:.4%}")
    logger.info(f"Vol sigma:    {params['vol']:.4%}")
    logger.info(f"Notional:     {params['notional']:,.0f}")
    logger.info(f"Option type:  {params['option_type']}")
    logger.info("-" * 80)
    
    # -------------------------------------------------------------------------
    # Compute forward price
    # F = S * exp((r_d - r_f) * T)
    # -------------------------------------------------------------------------
    fwd = params["spot"] * np.exp(
        (params["r_domestic"] - params["r_foreign"]) * params["expiry"]
    )
    logger.info(f"Forward F:    {fwd:.6f}")
    logger.info(f"Moneyness:    K/F = {params['strike'] / fwd:.4f}")
    logger.info("")
    
    # -------------------------------------------------------------------------
    # BSM pricing (benchmark)
    # -------------------------------------------------------------------------
    bsm = FxVanillaEuropeanOptionBsmPricer()
    bsm_pv = float(bsm.price(option, market))
    bsm_greeks = bsm.greeks(option, market)
    
    # -------------------------------------------------------------------------
    # Monte Carlo pricing
    # -------------------------------------------------------------------------
    if FxVanillaEuropeanOptionMcPricer is not None:
        mc = FxVanillaEuropeanOptionMcPricer(n_paths=100_000, seed=42)
        mc_pv = float(mc.price(option, market))
    else:
        mc_pv = None
        logger.warning("MC pricer not available")
    
    # -------------------------------------------------------------------------
    # Finite Difference pricing (optional)
    # -------------------------------------------------------------------------
    fd_pv = safe_fd_price(option, market, n_spot=200, n_time=100, verbose=True)
    
    # -------------------------------------------------------------------------
    # Display PVs
    # -------------------------------------------------------------------------
    logger.info("PVs")
    logger.info("-" * 80)
    logger.info(f"BSM PV:       {bsm_pv:,.6f}")
    
    if mc_pv is not None:
        logger.info(f"MC  PV:       {mc_pv:,.6f}")
        logger.info(f"MC error:     {abs(mc_pv - bsm_pv):,.6f}  ({abs(mc_pv / bsm_pv - 1.0) * 100.0:.4f}%)")
    
    if fd_pv is not None:
        logger.info(f"FD  PV:       {fd_pv:,.6f}")
        logger.info(f"FD error:     {abs(fd_pv - bsm_pv):,.6f}  ({abs(fd_pv / bsm_pv - 1.0) * 100.0:.4f}%)")
    else:
        logger.info("FD  PV:       [skipped / unavailable]")
    
    # -------------------------------------------------------------------------
    # Display Greeks
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("Greeks (BSM)")
    logger.info("-" * 80)
    logger.info(f"delta:        {float(bsm_greeks['delta']): .6f}")
    logger.info(f"gamma:        {float(bsm_greeks['gamma']): .6f}")
    logger.info(f"vega:         {float(bsm_greeks['vega']): .6f}")
    logger.info(f"theta:        {float(bsm_greeks['theta']): .6f}")
    logger.info(f"rho_domestic: {float(bsm_greeks.get('rho_domestic', 0.0)): .6f}")
    logger.info(f"rho_foreign:  {float(bsm_greeks.get('rho_foreign', 0.0)): .6f}")
    
    # -------------------------------------------------------------------------
    # MC Convergence sweep
    # -------------------------------------------------------------------------
    if FxVanillaEuropeanOptionMcPricer is not None:
        run_mc_convergence(option, market, bsm_pv)
    
    # -------------------------------------------------------------------------
    # FD Convergence sweep
    # -------------------------------------------------------------------------
    run_fd_convergence(option, market, bsm_pv)
    
    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------
    if MATPLOTLIB_AVAILABLE and ENABLE_PLOTTING:
        visualize_results(option, market, params, bsm, bsm_pv)


def run_mc_convergence(
    option: FxVanillaEuropeanOption,
    market: Market,
    bsm_pv: float,
) -> List[ConvergenceRow]:
    """
    Run MC convergence sweep.
    
    Parameters
    ----------
    option : FxVanillaEuropeanOption
        The option to price.
    market : Market
        Market snapshot.
    bsm_pv : float
        BSM benchmark price.
    
    Returns
    -------
    List[ConvergenceRow]
        Convergence data.
    """
    path_counts = [1_000, 10_000, 50_000, 100_000, 500_000, 1_000_000]
    mc_rows: List[ConvergenceRow] = []
    
    for n in path_counts:
        pv_n = float(FxVanillaEuropeanOptionMcPricer(n_paths=n, seed=42).price(option, market))
        abs_err = abs(pv_n - bsm_pv)
        rel_err = abs_err / abs(bsm_pv) if bsm_pv != 0.0 else float("nan")
        mc_rows.append(ConvergenceRow(x=n, pv=pv_n, abs_err=abs_err, rel_err=rel_err))
    
    logger.info("")
    logger.info("MC Convergence")
    logger.info("-" * 80)
    logger.info(f"{'Paths':>12} {'PV':>18} {'AbsErr':>18} {'RelErr':>12}")
    
    for r in mc_rows:
        logger.info(f"{r.x:>12,} {r.pv:>18,.6f} {r.abs_err:>18,.6f} {r.rel_err*100:>11.4f}%")
    
    return mc_rows


def run_fd_convergence(
    option: FxVanillaEuropeanOption,
    market: Market,
    bsm_pv: float,
) -> List[ConvergenceRow]:
    """
    Run FD convergence sweep.
    
    Parameters
    ----------
    option : FxVanillaEuropeanOption
        The option to price.
    market : Market
        Market snapshot.
    bsm_pv : float
        BSM benchmark price.
    
    Returns
    -------
    List[ConvergenceRow]
        Convergence data.
    """
    grid_sizes = [(50, 25), (100, 50), (200, 100), (400, 200)]
    fd_rows: List[ConvergenceRow] = []
    
    for n_spot, n_time in grid_sizes:
        pv = safe_fd_price(option, market, n_spot=n_spot, n_time=n_time)
        if pv is None:
            continue
        abs_err = abs(pv - bsm_pv)
        rel_err = abs_err / abs(bsm_pv) if bsm_pv != 0.0 else float("nan")
        fd_rows.append(ConvergenceRow(x=n_spot, pv=pv, abs_err=abs_err, rel_err=rel_err))
    
    logger.info("")
    logger.info("FD Convergence")
    logger.info("-" * 80)
    
    if not fd_rows:
        logger.info("FD convergence skipped (FD pricer unavailable or failed).")
    else:
        logger.info(f"{'n_spot':>12} {'PV':>18} {'AbsErr':>18} {'RelErr':>12}")
        for row, (n_spot, n_time) in zip(fd_rows, grid_sizes):
            logger.info(f"{n_spot:>12} {row.pv:>18,.6f} {row.abs_err:>18,.6f} {row.rel_err*100:>11.4f}%")
    
    return fd_rows


# =============================================================================
# SECTION 3: Visualization
# =============================================================================

def visualize_results(
    option: FxVanillaEuropeanOption,
    market: Market,
    params: dict,
    bsm: FxVanillaEuropeanOptionBsmPricer,
    bsm_pv: float,
) -> None:
    """
    Create visualization of pricing results.
    
    Parameters
    ----------
    option : FxVanillaEuropeanOption
        The option.
    market : Market
        Market snapshot.
    params : dict
        Parameters dictionary.
    bsm : FxVanillaEuropeanOptionBsmPricer
        BSM pricer.
    bsm_pv : float
        BSM price.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.figsize": (12, 5),
        "font.size": 11,
        "axes.titlesize": 13,
        "lines.linewidth": 2,
    })
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    fwd = params["spot"] * np.exp(
        (params["r_domestic"] - params["r_foreign"]) * params["expiry"]
    )
    
    # -------------------------------------------------------------------------
    # Plot 1: Call unit price vs strike (BSM)
    # -------------------------------------------------------------------------
    ax = axes[0]
    strikes_plot = np.linspace(0.95, 1.25, 20)
    unit_prices = []
    
    for k in strikes_plot:
        opt_k = FxVanillaEuropeanOption(
            option_type=params["option_type"],
            strike=float(k),
            expiry=params["expiry"],
            notional=params["notional"],
            spot_id=params["spot_id"],
            vol_id=params["vol_id"],
            domestic_curve_id=params["curve_dom_id"],
            foreign_curve_id=params["curve_for_id"],
        )
        unit_prices.append(float(bsm.price(opt_k, market)) / params["notional"])
    
    ax.plot(strikes_plot, unit_prices, "-", color=COLORS["bsm"], linewidth=2, label="BSM (unit PV)")
    ax.axvline(params["spot"], color="gray", linestyle="--", alpha=0.5, label=f"Spot = {params['spot']:.4f}")
    ax.axvline(fwd, color="gray", linestyle=":", alpha=0.5, label=f"Forward = {fwd:.4f}")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Unit Price (PV / Notional)")
    ax.set_title("Call Price vs Strike")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 2: MC convergence (log-log)
    # -------------------------------------------------------------------------
    ax = axes[1]
    if FxVanillaEuropeanOptionMcPricer is not None:
        path_counts = [1_000, 10_000, 50_000, 100_000, 500_000, 1_000_000]
        mc_abs_errs = []
        
        for n in path_counts:
            pv_n = float(FxVanillaEuropeanOptionMcPricer(n_paths=n, seed=42).price(option, market))
            mc_abs_errs.append(abs(pv_n - bsm_pv))
        
        mc_abs_errs = np.array(mc_abs_errs)
        ax.loglog(path_counts, mc_abs_errs, "o-", color=COLORS["mc"], linewidth=2, markersize=8)
        
        # Reference O(1/sqrt(N)) line
        ref_x = np.array(path_counts, dtype=float)
        ref_y = mc_abs_errs[0] * np.sqrt(ref_x[0]) / np.sqrt(ref_x)
        ax.loglog(ref_x, ref_y, "--", color="gray", alpha=0.7, label=r"$O(1/\sqrt{N})$")
        
        ax.set_xlabel("Number of Paths")
        ax.set_ylabel("Absolute Error (USD)")
        ax.set_title("MC Convergence")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "MC unavailable", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    
    # -------------------------------------------------------------------------
    # Plot 3: FD convergence
    # -------------------------------------------------------------------------
    ax = axes[2]
    grid_sizes = [(50, 25), (100, 50), (200, 100), (400, 200)]
    fd_abs_errs = []
    
    for n_spot, n_time in grid_sizes:
        pv = safe_fd_price(option, market, n_spot=n_spot, n_time=n_time)
        if pv is not None:
            fd_abs_errs.append(abs(pv - bsm_pv))
    
    if fd_abs_errs:
        grid_labels = [f"{s}×{t}" for s, t in grid_sizes[:len(fd_abs_errs)]]
        ax.semilogy(range(len(fd_abs_errs)), fd_abs_errs, "o-", color=COLORS["fd"], linewidth=2, markersize=8)
        ax.set_xticks(range(len(fd_abs_errs)))
        ax.set_xticklabels(grid_labels)
        ax.set_xlabel("Grid Size (Spot × Time)")
        ax.set_ylabel("Absolute Error (USD)")
        ax.set_title("FD Convergence")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "FD unavailable", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    
    plt.tight_layout()
    plt.show(block=True)
    
    logger.info("")
    logger.info("Visualization complete")


# =============================================================================
# SUMMARY
# =============================================================================

def print_summary() -> None:
    """Print summary of key concepts."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    
    summary = """
    ┌─────────────────────────────────────────────────────────────────────┐
    │                         KEY TAKEAWAYS                                │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                      │
    │  1. FX Option Pricing (Garman-Kohlhagen):                           │
    │     - Domestic rate (r_d) discounts the premium                     │
    │     - Foreign rate (r_f) determines carry                           │
    │     - Forward = Spot × exp((r_d - r_f) × T)                         │
    │                                                                      │
    │  2. Pricing Methods:                                                │
    │     - BSM: Analytic, O(1) complexity, exact for European            │
    │     - MC: O(N×T) complexity, converges as O(1/√N)                   │
    │     - FD: O(M²×T) complexity, handles early exercise                │
    │                                                                      │
    │  3. Greeks:                                                         │
    │     - delta: Sensitivity to spot                                    │
    │     - gamma: Convexity (second derivative to spot)                  │
    │     - vega: Sensitivity to volatility                               │
    │     - theta: Time decay                                             │
    │     - rho: Sensitivity to interest rates                            │
    │                                                                      │
    │  NEXT: See 01_equity_vanilla_pricing.py for equity options          │
    │                                                                      │
    └─────────────────────────────────────────────────────────────────────┘
    """
    logger.info(summary)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main(args: argparse.Namespace) -> None:
    """
    Main entry point for the example.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    """
    global ENABLE_PLOTTING
    ENABLE_PLOTTING = args.plot
    
    try:
        # Build market and option
        market, option, params = build_market_and_option()
        
        # Run pricing
        run_pricing(market, option, params)
        
        # Summary
        print_summary()
        
        logger.info("Example completed successfully!")
        
    except Exception as e:
        logger.exception(f"Example failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FX Vanilla Option Pricing Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=True,
        help="Enable plotting (default: True)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_false",
        dest="plot",
        help="Disable plotting",
    )
    
    args = parser.parse_args()
    main(args)

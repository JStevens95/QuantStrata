#!/usr/bin/env python3
"""
===============================================================================
European Vanilla Option Pricing: BSM, Monte Carlo, and Finite Difference
===============================================================================

Production-grade showcase for European vanilla FX options using QuantStrata
library pricers, realistic market data, and publication-quality plots.
Designed for front-office quant / hedge fund use.

Requirements
------------
- Python 3.10+ (3.12 recommended); library uses dataclass(slots=True).

Learning Objectives
-------------------
1. **Multi-Method Pricing**: BSM analytical, Monte Carlo, and Finite Difference
2. **Library Components**: FxVanillaEuropeanOptionBsmPricer, McPricer, FdPricer;
   Market with FlatZeroRateCurve / GridVolSurface
3. **Realistic Market Data**: EUR/USD spot, rates, and vol surface (flat or smile)
4. **Greeks & Risk**: Per-unit and position Delta, Gamma, Vega, Theta
5. **Convergence**: MC vs path count, FD vs grid size; library reporting plots
6. **Plots**: Method comparison, MC/FD convergence, price vs spot/strike/expiry,
   Greeks vs spot; all saved to PNG

Mathematical Framework
----------------------
Garman-Kohlhagen (FX):
    C = S·e^(-r_f·T)·N(d1) - K·e^(-r_d·T)·N(d2)
    d1,2 = [ln(S/K) + (r_d - r_f ± σ²/2)T] / (σ√T)
Put-Call Parity: C - P = S·e^(-r_f·T) - K·e^(-r_d·T)

Production Context
------------------
- BSM for vanilla European pricing and Greeks
- MC for path-dependent validation and convergence studies
- FD for American exercise and barriers
- GridVolSurface for smile-aware pricing (production vol input)

Run This Example
----------------
    cd /path/to/QuantStrata
    python examples/showcase/01_european_vanilla_pricing.py [--no-plot]

Author: QuantStrata Team
===============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np

# -----------------------------------------------------------------------------
# Python version: library uses dataclass(slots=True) and requires 3.10+
# -----------------------------------------------------------------------------
if sys.version_info < (3, 10):
    print("This example requires Python 3.10+ (3.12 recommended).")
    print(f"Current version: {sys.version_info.major}.{sys.version_info.minor}")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# -----------------------------------------------------------------------------
# QuantStrata imports - using library pricers and market data
# -----------------------------------------------------------------------------
from src.marketdata.core.ids import MarketId
from src.marketdata.core.interfaces import Quote
from src.marketdata.core.market import Market
from src.marketdata.curves.term_structure import FlatZeroRateCurve
from src.marketdata.surfaces.vol_surface import FlatVolSurface, GridVolSurface

from src.instruments.fx.options.vanilla import FxVanillaEuropeanOption

# Library pricers - production implementations
from src.pricers.fx.european_bsm import FxVanillaEuropeanOptionBsmPricer
from src.pricers.fx.european_bsm_mc import FxVanillaEuropeanOptionMcPricer
from src.pricers.fx.european_bsm_fde import FxVanillaEuropeanOptionFdPricer

# BSM model functions for direct Greeks computation
from src.models.analytic.black_scholes_merton.base import (
    vanilla_delta,
    vanilla_gamma,
    vanilla_vega,
    vanilla_theta,
)


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
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available - plotting disabled")


# =============================================================================
# CONSTANTS
# =============================================================================

# Market IDs
EURUSD_SPOT = MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD")
USD_CURVE = MarketId(asset_class="IR", mkt_type="CURVE", name="USD_OIS")
EUR_CURVE = MarketId(asset_class="IR", mkt_type="CURVE", name="EUR_OIS")
EURUSD_VOL = MarketId(asset_class="FX", mkt_type="VOL", name="EURUSD")

# Color palette
COLORS = {
    'bsm': '#2E86AB',      # Blue
    'mc': '#A23B72',       # Magenta
    'fd': '#F18F01',       # Orange
    'call': '#2E86AB',
    'put': '#E94F37',
    'gamma': '#8B5CF6',
    'vega': '#10B981',
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class OptionParams:
    """Container for option parameters. Defaults reflect realistic EUR/USD market."""
    spot: float = 1.0850           # EUR/USD spot (realistic level)
    strike: float = 1.0850        # ATM strike
    expiry: float = 1.0           # 1Y expiry
    vol: float = 0.095            # 9.5% ATM vol (typical G10)
    r_dom: float = 0.045           # USD rate (domestic)
    r_for: float = 0.030           # EUR rate (foreign)
    notional: float = 1_000_000   # 1M EUR notional


@dataclass
class PricingResult:
    """Result from a pricer."""
    method: str
    call_price: float
    put_price: float
    time_ms: float
    std_error: float = 0.0


# =============================================================================
# MARKET AND INSTRUMENT SETUP
# =============================================================================

def create_market(
    params: OptionParams,
    vol_surface: Union[FlatVolSurface, GridVolSurface, None] = None,
) -> Market:
    """
    Create market snapshot from parameters.

    Uses FlatVolSurface by default. For production-style pricing with a smile,
    pass a GridVolSurface (see create_eurusd_smile_surface).
    """
    if vol_surface is None:
        vol_surface = FlatVolSurface(sigma=params.vol)
    return Market(
        asof=date.today().isoformat(),
        quotes={EURUSD_SPOT: Quote(value=params.spot)},
        curves={
            USD_CURVE: FlatZeroRateCurve(continuously_compounded_rate=params.r_dom),
            EUR_CURVE: FlatZeroRateCurve(continuously_compounded_rate=params.r_for),
        },
        vols={EURUSD_VOL: vol_surface},
    )


def create_eurusd_smile_surface(spot: float, atm_vol: float = 0.095) -> GridVolSurface:
    """
    Build a production-style EUR/USD vol surface with smile (expiry x strike).

    In a front-office setup this would come from market data or a vol builder.
    Here we use a simple parametric smile: ATM + 25D put/call wings.
    """
    expiries = np.array([0.25, 0.5, 1.0, 2.0])
    # Strikes as moneyness then converted to absolute
    moneyness = np.array([0.90, 0.95, 1.0, 1.05, 1.10])
    strikes = (moneyness * spot).tolist()
    # Smile: higher vol for OTM put/call (stylized)
    n_exp, n_strike = len(expiries), len(strikes)
    implied_vols = np.zeros((n_exp, n_strike))
    for i in range(n_exp):
        for j in range(n_strike):
            wing = abs(moneyness[j] - 1.0)
            implied_vols[i, j] = atm_vol * (1.0 + 0.3 * wing + 0.1 * wing**2)
    return GridVolSurface(
        expiries=expiries,
        strikes=np.array(strikes),
        implied_vols=implied_vols,
        extrapolation="flat",
    )


def create_option(
    params: OptionParams,
    option_type: str = "call",
) -> FxVanillaEuropeanOption:
    """Create FX vanilla option."""
    return FxVanillaEuropeanOption(
        option_type=option_type,
        spot_id=EURUSD_SPOT,
        domestic_curve_id=USD_CURVE,
        foreign_curve_id=EUR_CURVE,
        vol_id=EURUSD_VOL,
        strike=params.strike,
        expiry=params.expiry,
        notional=params.notional,
    )


# =============================================================================
# PRICING FUNCTIONS USING LIBRARY PRICERS
# =============================================================================

def price_with_bsm(params: OptionParams) -> PricingResult:
    """Price using library BSM pricer."""
    market = create_market(params)
    pricer = FxVanillaEuropeanOptionBsmPricer()
    
    call_option = create_option(params, "call")
    put_option = create_option(params, "put")
    
    start = time.perf_counter()
    call_price = pricer.price(call_option, market)
    put_price = pricer.price(put_option, market)
    elapsed = (time.perf_counter() - start) * 1000
    
    return PricingResult(
        method="BSM (Analytical)",
        call_price=call_price / params.notional,  # Per unit
        put_price=put_price / params.notional,
        time_ms=elapsed,
    )


def price_with_mc(
    params: OptionParams,
    n_paths: int = 100_000,
) -> PricingResult:
    """Price using library Monte Carlo pricer."""
    market = create_market(params)
    pricer = FxVanillaEuropeanOptionMcPricer(
        n_paths=n_paths,
        seed=42,
        antithetic=True,
    )
    
    call_option = create_option(params, "call")
    put_option = create_option(params, "put")
    
    start = time.perf_counter()
    call_result = pricer.run(call_option, market)
    put_result = pricer.run(put_option, market)
    elapsed = (time.perf_counter() - start) * 1000
    
    call_price = call_result.discounted_payoffs.mean()
    put_price = put_result.discounted_payoffs.mean()
    
    # Standard error
    call_stderr = call_result.discounted_payoffs.std() / np.sqrt(len(call_result.discounted_payoffs))
    
    return PricingResult(
        method="Monte Carlo",
        call_price=call_price / params.notional,
        put_price=put_price / params.notional,
        time_ms=elapsed,
        std_error=call_stderr / params.notional,
    )


def price_with_fd(params: OptionParams) -> PricingResult:
    """Price using library Finite Difference pricer."""
    market = create_market(params)
    pricer = FxVanillaEuropeanOptionFdPricer(
        n_space=401,
        n_time_steps=200,
        theta=0.5,  # Crank-Nicolson
    )
    
    call_option = create_option(params, "call")
    put_option = create_option(params, "put")
    
    start = time.perf_counter()
    call_price = pricer.price(call_option, market)
    put_price = pricer.price(put_option, market)
    elapsed = (time.perf_counter() - start) * 1000
    
    return PricingResult(
        method="Finite Difference",
        call_price=call_price / params.notional,
        put_price=put_price / params.notional,
        time_ms=elapsed,
    )


def compute_greeks(params: OptionParams) -> Dict[str, float]:
    """Compute Greeks using library BSM pricer."""
    market = create_market(params)
    pricer = FxVanillaEuropeanOptionBsmPricer()
    call_option = create_option(params, "call")
    
    greeks = pricer.greeks(call_option, market)
    
    return {
        'delta': greeks.get('delta', 0.0),
        'gamma': greeks.get('gamma', 0.0),
        'vega': greeks.get('vega', 0.0),
        'theta': greeks.get('theta', 0.0),
    }


def check_put_call_parity(params: OptionParams) -> Dict[str, float]:
    """Verify put-call parity using library pricer."""
    bsm_result = price_with_bsm(params)
    
    # Put-call parity: C - P = S·e^(-r_f·T) - K·e^(-r_d·T)
    parity_lhs = bsm_result.call_price - bsm_result.put_price
    parity_rhs = (
        params.spot * np.exp(-params.r_for * params.expiry) -
        params.strike * np.exp(-params.r_dom * params.expiry)
    )
    
    return {
        "C - P": parity_lhs,
        "S·e^(-r_f·T) - K·e^(-r_d·T)": parity_rhs,
        "parity_error": abs(parity_lhs - parity_rhs),
    }


# =============================================================================
# CONVERGENCE ANALYSIS
# =============================================================================

def _mc_convergence(params: OptionParams) -> Tuple[List[int], List[float], List[float]]:
    """Test MC convergence vs BSM."""
    bsm_result = price_with_bsm(params)
    bsm_call = bsm_result.call_price
    
    path_counts = [1000, 5000, 10000, 50000, 100000, 500000]
    mc_prices = []
    mc_errors = []
    
    for n in path_counts:
        mc_result = price_with_mc(params, n_paths=n)
        mc_prices.append(mc_result.call_price)
        mc_errors.append(mc_result.std_error)
    
    return path_counts, mc_prices, mc_errors


def _fd_convergence(params: OptionParams) -> Tuple[List[int], List[float]]:
    """Test FD convergence vs BSM."""
    bsm_result = price_with_bsm(params)
    bsm_call = bsm_result.call_price
    
    grid_sizes = [51, 101, 201, 401, 801]
    fd_errors = []
    
    market = create_market(params)
    call_option = create_option(params, "call")
    
    for n in grid_sizes:
        pricer = FxVanillaEuropeanOptionFdPricer(
            n_space=n,
            n_time_steps=n // 2,
            theta=0.5,
        )
        fd_price = pricer.price(call_option, market) / params.notional
        fd_errors.append(abs(fd_price - bsm_call))
    
    return grid_sizes, fd_errors


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def run_pricing_showcase() -> Tuple[List[PricingResult], Dict]:
    """
    Run the complete pricing showcase.
    """
    logger.info("=" * 70)
    logger.info("EUROPEAN VANILLA OPTION PRICING SHOWCASE")
    logger.info("=" * 70)
    
    params = OptionParams()
    
    # Section 1: Parameters
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 1: Option Parameters")
    logger.info("=" * 70)
    
    logger.info("")
    logger.info(f"  Spot (EUR/USD):      {params.spot}")
    logger.info(f"  Strike:              {params.strike}")
    logger.info(f"  Expiry:              {params.expiry} year")
    logger.info(f"  Volatility:          {params.vol:.1%}")
    logger.info(f"  Domestic rate (USD): {params.r_dom:.2%}")
    logger.info(f"  Foreign rate (EUR):  {params.r_for:.2%}")
    logger.info(f"  Notional:            {params.notional:,.0f} EUR")
    
    # Section 2: Pricing
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 2: Multi-Method Pricing (Library Pricers)")
    logger.info("=" * 70)
    
    results = []
    
    # BSM
    bsm_result = price_with_bsm(params)
    results.append(bsm_result)
    
    # Monte Carlo
    mc_result = price_with_mc(params)
    results.append(mc_result)
    
    # Finite Difference
    fd_result = price_with_fd(params)
    results.append(fd_result)
    
    logger.info("")
    logger.info(f"{'Method':<20} {'Call Price':>12} {'Put Price':>12} {'Time (ms)':>12}")
    logger.info("-" * 60)
    
    for r in results:
        logger.info(f"{r.method:<20} {r.call_price:>12.6f} {r.put_price:>12.6f} {r.time_ms:>11.2f}")
    
    # Section 2b: Production market (GridVolSurface with smile)
    logger.info("")
    logger.info("  Production-style pricing with GridVolSurface (smile):")
    smile_surface = create_eurusd_smile_surface(params.spot, params.vol)
    market_smile = create_market(params, vol_surface=smile_surface)
    pricer_bsm = FxVanillaEuropeanOptionBsmPricer()
    call_smile = pricer_bsm.price(create_option(params, "call"), market_smile) / params.notional
    put_smile = pricer_bsm.price(create_option(params, "put"), market_smile) / params.notional
    logger.info(f"  BSM with GridVolSurface: call (per unit) = {call_smile:.6f}, put = {put_smile:.6f}")
    
    # Section 3: Greeks
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 3: Greeks (BSM Pricer)")
    logger.info("=" * 70)
    
    greeks = compute_greeks(params)
    n = params.notional
    logger.info("")
    logger.info("  (Per unit notional — scale by notional for position Greeks)")
    logger.info(f"  Delta:  {greeks['delta']/n:.6f}  (position delta: {greeks['delta']:,.0f} EUR)")
    logger.info(f"  Gamma:  {greeks['gamma']/n:.6f}  (position gamma: {greeks['gamma']:,.2f})")
    logger.info(f"  Vega:   {greeks['vega']/n:.6f}  (per 1%% vol; position vega: {greeks['vega']:,.0f})")
    logger.info(f"  Theta:  {greeks['theta']/n:.8f}  (per day: {greeks['theta']/365:,.2f})")
    
    # Section 4: Put-Call Parity
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 4: Put-Call Parity Check")
    logger.info("=" * 70)
    
    parity = check_put_call_parity(params)
    
    logger.info("")
    logger.info(f"  C - P:                      {parity['C - P']:.6f}")
    logger.info(f"  S·e^(-r_f·T) - K·e^(-r_d·T): {parity['S·e^(-r_f·T) - K·e^(-r_d·T)']:.6f}")
    logger.info(f"  Parity Error:               {parity['parity_error']:.2e}")
    
    # Section 5: Method Errors
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 5: Error vs BSM Analytical")
    logger.info("=" * 70)
    
    bsm_call = bsm_result.call_price
    
    logger.info("")
    logger.info(f"{'Method':<20} {'Error vs BSM':>15}")
    logger.info("-" * 40)
    logger.info(f"{'BSM (reference)':<20} {0.0:>15.6f}")
    logger.info(f"{'Monte Carlo':<20} {abs(mc_result.call_price - bsm_call):>15.6f}")
    logger.info(f"{'Finite Difference':<20} {abs(fd_result.call_price - bsm_call):>15.6f}")
    
    return results, {
        "params": params,
        "greeks": greeks,
        "parity": parity,
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def _build_mc_convergence_points(params: OptionParams, path_counts: List[int]):
    """Build MC convergence points (PV in domestic) for library plot."""
    from src.core.reporting.plots.pricers.monte_carlo import McConvergencePoint
    bsm_result = price_with_bsm(params)
    pv_benchmark = bsm_result.call_price * params.notional  # domestic
    points = []
    for n in path_counts:
        mc_result = price_with_mc(params, n_paths=n)
        pv_mean = mc_result.call_price * params.notional
        stderr = mc_result.std_error * params.notional
        z = 1.96
        points.append(McConvergencePoint(
            n_paths=n,
            pv_mean=pv_mean,
            pv_ci_lo=pv_mean - z * stderr,
            pv_ci_hi=pv_mean + z * stderr,
            pv_stderr=stderr,
        ))
    return points


def visualize_pricing(results: List[PricingResult], analysis: Dict) -> None:
    """Production-style visualizations: method comparison, convergence, profiles, Greeks."""
    if not MATPLOTLIB_AVAILABLE or not ENABLE_PLOTTING:
        logger.info("Skipping plots")
        return
    
    if "params" not in analysis:
        raise KeyError(
            "analysis must contain 'params'. "
            "Ensure run_pricing_showcase() returns (results, {'params': params, 'greeks': ..., 'parity': ...})."
        )
    
    from src.core.reporting.plots.style import apply_report_style, get_report_figsize, report_rc
    from src.core.reporting.plots.pricers.monte_carlo import plot_mc_convergence_vs_paths
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 6: Visualization")
    logger.info("=" * 70)
    
    params = analysis["params"]
    bsm_result = results[0]
    out_dir = Path.cwd()
    
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")
    
    with report_rc():
        # ----- Figure 1: Method comparison + Time -----
        fig1, axes = plt.subplots(1, 2, figsize=(14, 5))
        methods = [r.method for r in results]
        call_prices = [r.call_price for r in results]
        colors = [COLORS["bsm"], COLORS["mc"], COLORS["fd"]]
        
        ax = axes[0]
        ax.bar(methods, call_prices, color=colors)
        ax.axhline(bsm_result.call_price, color="black", linestyle="--", linewidth=1, label="BSM ref")
        ax.set_ylabel("Call Price (per unit)")
        ax.set_title("Pricing Method Comparison")
        ax.legend()
        apply_report_style(ax)
        
        ax = axes[1]
        ax.bar(methods, [r.time_ms for r in results], color=colors)
        ax.set_ylabel("Time (ms)")
        ax.set_title("Execution Time")
        ax.set_yscale("log")
        apply_report_style(ax)
        
        plt.tight_layout()
        fig1.savefig(out_dir / "european_vanilla_method_comparison.png", dpi=150, bbox_inches="tight")
        logger.info("  Saved: european_vanilla_method_comparison.png")

        # ----- Figure 2: MC convergence (library plot) -----
        path_counts = [1000, 5000, 10000, 25000, 50000, 100000]
        mc_points = _build_mc_convergence_points(params, path_counts)
        pv_bsm_domestic = bsm_result.call_price * params.notional
        fig2, ax2 = plt.subplots(figsize=get_report_figsize())
        plot_mc_convergence_vs_paths(
            points=mc_points,
            pv_benchmark=pv_bsm_domestic,
            benchmark_label="BSM",
            title="Monte Carlo Convergence: Call PV vs Path Count",
            ylabel="PV (USD)",
            use_log_x=True,
            ax=ax2,
        )
        apply_report_style(ax2)
        fig2.savefig(out_dir / "european_vanilla_mc_convergence.png", dpi=150, bbox_inches="tight")
        plt.close(fig2)
        logger.info("  Saved: european_vanilla_mc_convergence.png")
        
        # ----- Figure 3: FD convergence -----
        grid_sizes, fd_errors = _fd_convergence(params)
        fig3, ax3 = plt.subplots(figsize=get_report_figsize())
        ax3.semilogy(grid_sizes, fd_errors, "o-", color=COLORS["fd"], linewidth=2)
        ax3.set_xlabel("Grid size (space points)")
        ax3.set_ylabel("|FD price - BSM price| (per unit)")
        ax3.set_title("Finite Difference Convergence vs BSM")
        ax3.grid(True, alpha=0.5)
        apply_report_style(ax3)
        fig3.savefig(out_dir / "european_vanilla_fd_convergence.png", dpi=150, bbox_inches="tight")
        plt.close(fig3)
        logger.info("  Saved: european_vanilla_fd_convergence.png")
        
        # ----- Figure 4: Price vs Spot + Price vs Strike -----
        fig4, axes = plt.subplots(1, 2, figsize=(14, 5))
        spots = np.linspace(params.spot * 0.7, params.spot * 1.3, 50)
        call_vs_spot, put_vs_spot = [], []
        for s in spots:
            p = OptionParams(spot=s, strike=params.strike, expiry=params.expiry, vol=params.vol,
                            r_dom=params.r_dom, r_for=params.r_for, notional=1.0)
            r = price_with_bsm(p)
            call_vs_spot.append(r.call_price)
            put_vs_spot.append(r.put_price)
        ax = axes[0]
        ax.plot(spots, call_vs_spot, color=COLORS["call"], linewidth=2, label="Call")
        ax.plot(spots, put_vs_spot, color=COLORS["put"], linewidth=2, label="Put")
        ax.axvline(params.strike, color="gray", linestyle="--", alpha=0.6)
        ax.set_xlabel("Spot (EUR/USD)")
        ax.set_ylabel("Option price (per unit)")
        ax.set_title("Price vs Spot")
        ax.legend()
        apply_report_style(ax)
        
        strikes = np.linspace(params.spot * 0.85, params.spot * 1.15, 30)
        call_vs_k, put_vs_k = [], []
        for k in strikes:
            p = OptionParams(spot=params.spot, strike=float(k), expiry=params.expiry, vol=params.vol,
                             r_dom=params.r_dom, r_for=params.r_for, notional=1.0)
            r = price_with_bsm(p)
            call_vs_k.append(r.call_price)
            put_vs_k.append(r.put_price)
        ax = axes[1]
        ax.plot(strikes / params.spot, call_vs_k, color=COLORS["call"], linewidth=2, label="Call")
        ax.plot(strikes / params.spot, put_vs_k, color=COLORS["put"], linewidth=2, label="Put")
        ax.axvline(1.0, color="gray", linestyle="--", alpha=0.6)
        ax.set_xlabel("Moneyness (K/S)")
        ax.set_ylabel("Option price (per unit)")
        ax.set_title("Price vs Strike (Smile)")
        ax.legend()
        apply_report_style(ax)
        plt.tight_layout()
        fig4.savefig(out_dir / "european_vanilla_price_profiles.png", dpi=150, bbox_inches="tight")
        plt.close(fig4)
        logger.info("  Saved: european_vanilla_price_profiles.png")
        
        # ----- Figure 5: Term structure (price vs expiry) -----
        expiries = np.linspace(0.1, 2.0, 25)
        call_vs_t, put_vs_t = [], []
        for t in expiries:
            p = OptionParams(spot=params.spot, strike=params.strike, expiry=float(t), vol=params.vol,
                            r_dom=params.r_dom, r_for=params.r_for, notional=1.0)
            r = price_with_bsm(p)
            call_vs_t.append(r.call_price)
            put_vs_t.append(r.put_price)
        fig5, ax5 = plt.subplots(figsize=get_report_figsize())
        ax5.plot(expiries, call_vs_t, color=COLORS["call"], linewidth=2, label="Call")
        ax5.plot(expiries, put_vs_t, color=COLORS["put"], linewidth=2, label="Put")
        ax5.set_xlabel("Expiry (years)")
        ax5.set_ylabel("Option price (per unit)")
        ax5.set_title("Price vs Expiry (Term Structure)")
        ax5.legend()
        apply_report_style(ax5)
        fig5.savefig(out_dir / "european_vanilla_term_structure.png", dpi=150, bbox_inches="tight")
        plt.close(fig5)
        logger.info("  Saved: european_vanilla_term_structure.png")
        
        # ----- Figure 6: Delta, Gamma, Vega, Theta vs Spot -----
        fig6, axes = plt.subplots(2, 2, figsize=(12, 10))
        carry = params.r_dom - params.r_for
        deltas, gammas, vegas, thetas = [], [], [], []
        for s in spots:
            deltas.append(vanilla_delta(option_type="call", spot=float(s), strike=params.strike,
                                        expiry=params.expiry, discount_rate=params.r_dom, carry=carry, vol=params.vol))
            gammas.append(vanilla_gamma(option_type="call", spot=float(s), strike=params.strike,
                                        expiry=params.expiry, discount_rate=params.r_dom, carry=carry, vol=params.vol))
            vegas.append(vanilla_vega(option_type="call", spot=float(s), strike=params.strike,
                                      expiry=params.expiry, discount_rate=params.r_dom, carry=carry, vol=params.vol))
            thetas.append(vanilla_theta(option_type="call", spot=float(s), strike=params.strike,
                                       expiry=params.expiry, discount_rate=params.r_dom, carry=carry, vol=params.vol))
        axes[0, 0].plot(spots, deltas, color=COLORS["bsm"], linewidth=2)
        axes[0, 0].axvline(params.strike, color="gray", linestyle="--", alpha=0.5)
        axes[0, 0].set_xlabel("Spot"); axes[0, 0].set_ylabel("Delta"); axes[0, 0].set_title("Delta vs Spot")
        apply_report_style(axes[0, 0])
        axes[0, 1].plot(spots, gammas, color=COLORS["gamma"], linewidth=2)
        axes[0, 1].axvline(params.strike, color="gray", linestyle="--", alpha=0.5)
        axes[0, 1].set_xlabel("Spot"); axes[0, 1].set_ylabel("Gamma"); axes[0, 1].set_title("Gamma vs Spot")
        apply_report_style(axes[0, 1])
        axes[1, 0].plot(spots, vegas, color=COLORS["vega"], linewidth=2)
        axes[1, 0].axvline(params.strike, color="gray", linestyle="--", alpha=0.5)
        axes[1, 0].set_xlabel("Spot"); axes[1, 0].set_ylabel("Vega"); axes[1, 0].set_title("Vega vs Spot")
        apply_report_style(axes[1, 0])
        axes[1, 1].plot(spots, thetas, color=COLORS["fd"], linewidth=2)
        axes[1, 1].axvline(params.strike, color="gray", linestyle="--", alpha=0.5)
        axes[1, 1].set_xlabel("Spot"); axes[1, 1].set_ylabel("Theta"); axes[1, 1].set_title("Theta vs Spot")
        apply_report_style(axes[1, 1])
        plt.tight_layout()
        fig6.savefig(out_dir / "european_vanilla_greeks_vs_spot.png", dpi=150, bbox_inches="tight")
        plt.close(fig6)
        logger.info("  Saved: european_vanilla_greeks_vs_spot.png")
        
        plt.show(block=True)
    
    logger.info("Visualization complete.")


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
    │  1. Library components (Python 3.10+ / 3.12):                       │
    │     - FxVanillaEuropeanOptionBsmPricer, McPricer, FdPricer          │
    │     - Market + FlatZeroRateCurve; FlatVolSurface / GridVolSurface   │
    │     - Reporting: apply_report_style, plot_mc_convergence_vs_paths   │
    │                                                                      │
    │  2. Realistic market: EUR/USD spot, rates, ATM vol; smile via       │
    │     GridVolSurface for production-style pricing.                    │
    │                                                                      │
    │  3. Put-Call Parity: C - P = S·e^(-r_f·T) - K·e^(-r_d·T)           │
    │     Greeks: per-unit and position (delta, gamma, vega, theta).       │
    │                                                                      │
    │  4. Plots: method comparison, MC/FD convergence, price vs           │
    │     spot/strike/expiry, Greeks vs spot (saved as PNG).               │
    │                                                                      │
    └─────────────────────────────────────────────────────────────────────┘
    """
    logger.info(summary)


# =============================================================================
# MAIN
# =============================================================================

def main(args: argparse.Namespace) -> None:
    """Main entry point."""
    global ENABLE_PLOTTING
    ENABLE_PLOTTING = args.plot
    
    try:
        results, analysis = run_pricing_showcase()
        # Ensure analysis has 'params' (in case of an old or partial return shape)
        if not isinstance(analysis, dict):
            analysis = {}
        if "params" not in analysis:
            analysis = {**analysis, "params": OptionParams(), "greeks": analysis.get("greeks", {}), "parity": analysis.get("parity", {})}
        visualize_pricing(results, analysis)
        print_summary()
        logger.info("Example completed successfully!")
        
    except Exception as e:
        logger.exception(f"Example failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="European Vanilla Pricing Showcase")
    parser.add_argument("--plot", action="store_true", default=True)
    parser.add_argument("--no-plot", action="store_false", dest="plot")
    
    args = parser.parse_args()
    main(args)

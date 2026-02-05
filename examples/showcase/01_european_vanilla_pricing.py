#!/usr/bin/env python3
"""
===============================================================================
European Vanilla Option Pricing: BSM, Monte Carlo, and Finite Difference
===============================================================================

This example demonstrates pricing European vanilla FX options using three
different methods via QuantStrata's production pricers.

Learning Objectives
-------------------
1. **Multi-Method Pricing**: Compare BSM analytical, MC, and FD approaches
2. **Library Integration**: Use production FX pricers consistently
3. **Greeks Analysis**: Compute and visualize risk sensitivities
4. **Convergence Testing**: Validate numerical methods against analytical

Mathematical Framework
----------------------
For FX options under Garman-Kohlhagen:

    C = S·e^(-r_f·T)·N(d1) - K·e^(-r_d·T)·N(d2)
    P = K·e^(-r_d·T)·N(-d2) - S·e^(-r_f·T)·N(-d1)

Where:
    d1 = [ln(S/K) + (r_d - r_f + σ²/2)T] / (σ√T)
    d2 = d1 - σ√T

Put-Call Parity:
    C - P = S·e^(-r_f·T) - K·e^(-r_d·T)

Production Context
------------------
At a hedge fund:
- FX options are highly liquid G10 instruments
- BSM is the market standard for quoting
- MC is used for path-dependent exotics validation
- FD is preferred for American exercise and barriers

Prerequisites
-------------
- Examples in fundamentals/
- Understanding of FX conventions

Run This Example
----------------
    cd /path/to/QuantStrata
    PYTHONPATH=. python examples/showcase/01_european_vanilla_pricing.py

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
from typing import Dict, List, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# -----------------------------------------------------------------------------
# QuantStrata imports - using library pricers
# -----------------------------------------------------------------------------
from src.marketdata.core.ids import MarketId
from src.marketdata.core.interfaces import Quote
from src.marketdata.core.market import Market
from src.marketdata.curves.term_structure import FlatZeroRateCurve
from src.marketdata.surfaces.vol_surface import FlatVolSurface

from src.instruments.fx.options.vanilla import FxVanillaEuropeanOption

# Library pricers - production implementations
from src.pricers.fx.european_bsm import FxVanillaEuropeanOptionBsmPricer
from src.pricers.fx.european_bsm_mc import FxVanillaEuropeanOptionMcPricer
from src.pricers.fx.european_bsm_fde import FxVanillaEuropeanOptionFdPricer

# BSM model functions for direct Greeks computation
from src.models.analytic.black_scholes_merton.base import (
    vanilla_price,
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
    """Container for option parameters."""
    spot: float = 1.3000          # EUR/USD spot
    strike: float = 1.3000        # ATM strike
    expiry: float = 1.0           # 1 year
    vol: float = 0.10             # 10% volatility
    r_dom: float = 0.05           # USD rate
    r_for: float = 0.02           # EUR rate
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

def create_market(params: OptionParams) -> Market:
    """Create market snapshot from parameters."""
    return Market(
        val_date=date.today(),
        quotes={EURUSD_SPOT: Quote(value=params.spot)},
        curves={
            USD_CURVE: FlatZeroRateCurve(USD_CURVE, params.r_dom),
            EUR_CURVE: FlatZeroRateCurve(EUR_CURVE, params.r_for),
        },
        vol_surfaces={EURUSD_VOL: FlatVolSurface(EURUSD_VOL, params.vol)},
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

def test_mc_convergence(params: OptionParams) -> Tuple[List[int], List[float], List[float]]:
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


def test_fd_convergence(params: OptionParams) -> Tuple[List[int], List[float]]:
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
    
    # Section 3: Greeks
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 3: Greeks (BSM Pricer)")
    logger.info("=" * 70)
    
    greeks = compute_greeks(params)
    
    logger.info("")
    logger.info(f"  Delta:  {greeks['delta']:.4f}")
    logger.info(f"  Gamma:  {greeks['gamma']:.4f}")
    logger.info(f"  Vega:   {greeks['vega']:.4f}")
    logger.info(f"  Theta:  {greeks['theta']:.6f}")
    
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

def visualize_pricing(results: List[PricingResult], analysis: Dict) -> None:
    """Visualize pricing results."""
    if not MATPLOTLIB_AVAILABLE or not ENABLE_PLOTTING:
        logger.info("Skipping plots")
        return
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 6: Visualization")
    logger.info("=" * 70)
    
    params = analysis["params"]
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.figsize': (14, 8),
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'legend.fontsize': 10,
        'lines.linewidth': 2,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Method Comparison
    ax = axes[0, 0]
    methods = [r.method for r in results]
    call_prices = [r.call_price for r in results]
    colors = [COLORS['bsm'], COLORS['mc'], COLORS['fd']]
    
    bars = ax.bar(methods, call_prices, color=colors)
    ax.axhline(call_prices[0], color='black', linestyle='--', linewidth=1, label='BSM Reference')
    ax.set_ylabel('Call Price (per unit)')
    ax.set_title('Pricing Method Comparison')
    ax.legend()
    
    # Plot 2: Price vs Spot
    ax = axes[0, 1]
    spots = np.linspace(params.spot * 0.7, params.spot * 1.3, 50)
    
    call_prices_bsm = []
    put_prices_bsm = []
    
    for s in spots:
        test_params = OptionParams(
            spot=s,
            strike=params.strike,
            expiry=params.expiry,
            vol=params.vol,
            r_dom=params.r_dom,
            r_for=params.r_for,
            notional=1.0,
        )
        result = price_with_bsm(test_params)
        call_prices_bsm.append(result.call_price)
        put_prices_bsm.append(result.put_price)
    
    ax.plot(spots, call_prices_bsm, color=COLORS['call'], linewidth=2.5, label='Call')
    ax.plot(spots, put_prices_bsm, color=COLORS['put'], linewidth=2.5, label='Put')
    ax.axvline(params.strike, color='gray', linestyle='--', alpha=0.5, label='Strike')
    ax.set_xlabel('Spot Price')
    ax.set_ylabel('Option Price')
    ax.set_title('Option Price vs Spot (BSM)')
    ax.legend()
    
    # Plot 3: Greeks vs Spot
    ax = axes[1, 0]
    deltas = []
    gammas = []
    
    for s in spots:
        # Use library BSM functions directly
        carry = params.r_dom - params.r_for
        delta = vanilla_delta(
            option_type="call",
            spot=s,
            strike=params.strike,
            expiry=params.expiry,
            discount_rate=params.r_dom,
            carry=carry,
            vol=params.vol,
        )
        gamma = vanilla_gamma(
            spot=s,
            strike=params.strike,
            expiry=params.expiry,
            discount_rate=params.r_dom,
            carry=carry,
            vol=params.vol,
        )
        deltas.append(delta)
        gammas.append(gamma)
    
    ax2 = ax.twinx()
    ax.plot(spots, deltas, color=COLORS['bsm'], linewidth=2.5, label='Delta')
    ax2.plot(spots, gammas, color=COLORS['gamma'], linewidth=2.5, linestyle='--', label='Gamma')
    
    ax.set_xlabel('Spot Price')
    ax.set_ylabel('Delta', color=COLORS['bsm'])
    ax2.set_ylabel('Gamma', color=COLORS['gamma'])
    ax.set_title('Delta and Gamma vs Spot')
    ax.axvline(params.strike, color='gray', linestyle='--', alpha=0.5)
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    # Plot 4: Execution Time Comparison
    ax = axes[1, 1]
    times = [r.time_ms for r in results]
    
    bars = ax.bar(methods, times, color=colors)
    ax.set_ylabel('Time (ms)')
    ax.set_title('Execution Time Comparison')
    ax.set_yscale('log')
    
    for bar, t in zip(bars, times):
        ax.annotate(f'{t:.1f}ms',
                   (bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show(block=True)
    
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
    │  1. Library Pricers Used:                                           │
    │     - FxVanillaEuropeanOptionBsmPricer (analytical)                 │
    │     - FxVanillaEuropeanOptionMcPricer (Monte Carlo)                 │
    │     - FxVanillaEuropeanOptionFdPricer (Finite Difference)           │
    │                                                                      │
    │  2. Method Comparison:                                              │
    │     - BSM: Fastest, exact for European vanilla                      │
    │     - MC: Flexible, scales to complex payoffs                       │
    │     - FD: Handles early exercise, American options                  │
    │                                                                      │
    │  3. Put-Call Parity:                                                │
    │     C - P = S·e^(-r_f·T) - K·e^(-r_d·T)                            │
    │     All methods satisfy this relationship                           │
    │                                                                      │
    │  4. Production Usage:                                               │
    │     - Use BSM for vanilla European pricing/Greeks                   │
    │     - Use MC for path-dependent validation                          │
    │     - Use FD for American exercise/barriers                         │
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

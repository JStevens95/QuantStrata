#!/usr/bin/env python3
"""
===============================================================================
Sensitivities Computation: Greeks and Risk Factors
===============================================================================

This example demonstrates comprehensive Greeks computation - the foundation
of derivatives risk management and hedging.

Learning Objectives
-------------------
1. **Analytical Greeks**: Closed-form BSM sensitivities
2. **Bump-and-Reprice**: Numerical Greeks via finite differences
3. **Bump Size Analysis**: Trade-off between accuracy and numerical noise
4. **Dollar Greeks**: Converting sensitivities to P&L estimates

Mathematical Framework
----------------------
For a derivative V(S, σ, r, t), the Greeks are partial derivatives:

    Delta (δ) = ∂V/∂S            # Spot sensitivity
    Gamma (Γ) = ∂²V/∂S²          # Convexity
    Vega (ν)  = ∂V/∂σ            # Volatility sensitivity
    Theta (Θ) = -∂V/∂t           # Time decay
    Rho (ρ)   = ∂V/∂r            # Rate sensitivity

Finite Difference Approximations:
    
    Central: δ ≈ [V(S+h) - V(S-h)] / (2h)
    Gamma:   Γ ≈ [V(S+h) - 2V(S) + V(S-h)] / h²

Production Context
------------------
At a hedge fund:
- Greeks drive daily hedging decisions
- Limits are set on Greeks (max delta, max vega)
- P&L attribution decomposes daily P&L into Greek contributions
- Greeks are computed multiple times per day

Prerequisites
-------------
- Examples in fundamentals/ and pricing/
- Understanding of scenario shocks

Run This Example
----------------
    cd /path/to/QuantStrata
    PYTHONPATH=. python examples/risk/02_sensitivities_computation.py

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
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# Path setup
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
from src.marketdata.scenarios.shocks import SpotShock, VolShock, ParallelRateShock

from src.instruments.fx.options.vanilla import FxVanillaEuropeanOption
from src.portfolio.core import Portfolio, Position
from src.portfolio.portfolio import PortfolioPricer
from src.pricers.registry import PricerRegistry
from src.pricers.fx.european_bsm import FxVanillaEuropeanOptionBsmPricer

# Try to import sensitivities engine (may not exist in all versions)
try:
    from src.risk.sensitivities.engine import compute_sensitivities
    from src.risk.sensitivities.config import SensitivitiesConfig, BumpConfig
    SENS_ENGINE_AVAILABLE = True
except ImportError:
    SENS_ENGINE_AVAILABLE = False


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

# Market IDs - use mkt_type (not data_type)
EURUSD_SPOT = MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD")
USD_CURVE = MarketId(asset_class="IR", mkt_type="CURVE", name="USD_OIS")
EUR_CURVE = MarketId(asset_class="IR", mkt_type="CURVE", name="EUR_OIS")
EURUSD_VOL = MarketId(asset_class="FX", mkt_type="VOL", name="EURUSD")


# =============================================================================
# SETUP: Market and Portfolio
# =============================================================================

def create_market_and_portfolio() -> Tuple[Market, Portfolio, PortfolioPricer, FxVanillaEuropeanOptionBsmPricer, float, dict]:
    """
    Create market, portfolio, and pricers for sensitivity analysis.
    
    Returns
    -------
    Tuple
        market, portfolio, portfolio_pricer, bsm_pricer, base_pv, analytic_greeks
    """
    logger.info("=" * 70)
    logger.info("SETUP: Market and Portfolio")
    logger.info("=" * 70)
    
    # -------------------------------------------------------------------------
    # Market parameters
    # -------------------------------------------------------------------------
    spot = 1.0850
    r_usd = 0.05
    r_eur = 0.02
    vol = 0.10
    
    # -------------------------------------------------------------------------
    # Build Market with correct API
    # -------------------------------------------------------------------------
    market = Market(
        asof="2026-01-28",
        quotes={EURUSD_SPOT: Quote(value=spot)},
        curves={
            USD_CURVE: FlatZeroRateCurve(continuously_compounded_rate=r_usd),
            EUR_CURVE: FlatZeroRateCurve(continuously_compounded_rate=r_eur),
        },
        vols={EURUSD_VOL: FlatVolSurface(sigma=vol)},
    )
    
    # -------------------------------------------------------------------------
    # Create option with correct API
    # -------------------------------------------------------------------------
    option = FxVanillaEuropeanOption(
        option_type="call",
        strike=1.10,
        expiry=1.0,
        notional=10_000_000,
        spot_id=EURUSD_SPOT,
        domestic_curve_id=USD_CURVE,
        foreign_curve_id=EUR_CURVE,
        vol_id=EURUSD_VOL,
    )
    
    portfolio = Portfolio(positions=[
        Position(position_id="OTM_CALL", instrument=option, quantity=1.0)
    ])
    
    # -------------------------------------------------------------------------
    # Setup pricers
    # -------------------------------------------------------------------------
    registry = PricerRegistry()
    bsm_pricer = FxVanillaEuropeanOptionBsmPricer()
    registry.register(FxVanillaEuropeanOption, bsm_pricer)
    portfolio_pricer = PortfolioPricer(pricer_registry=registry)
    
    # -------------------------------------------------------------------------
    # Compute base results
    # -------------------------------------------------------------------------
    base_result = portfolio_pricer.price(portfolio, market)
    base_pv = base_result.totals.pv
    analytic_greeks = bsm_pricer.greeks(option, market)
    
    logger.info("")
    logger.info(f"Option: OTM EUR Call, K=1.10, T=1Y, Notional=10M EUR")
    logger.info(f"  Spot: {spot}, Vol: {vol:.1%}")
    logger.info(f"  Base PV: {base_pv:,.2f} USD")
    
    logger.info("")
    logger.info("Analytical Greeks (BSM):")
    for greek, value in analytic_greeks.items():
        logger.info(f"  {greek}: {value:,.4f}")
    
    return market, portfolio, portfolio_pricer, bsm_pricer, base_pv, analytic_greeks


# =============================================================================
# SECTION 1: Manual Bump-and-Reprice Greeks
# =============================================================================

def bump_and_reprice_delta(
    portfolio: Portfolio,
    market: Market,
    pricer: PortfolioPricer,
    spot_id: MarketId,
    bump_pct: float = 0.01,
) -> float:
    """
    Compute delta via central difference.
    
    Parameters
    ----------
    portfolio : Portfolio
        Portfolio to price.
    market : Market
        Base market.
    pricer : PortfolioPricer
        Portfolio pricer.
    spot_id : MarketId
        Spot market ID.
    bump_pct : float
        Relative bump size (e.g., 0.01 = 1%).
    
    Returns
    -------
    float
        Numerical delta.
    
    Formula
    -------
    δ ≈ [V(S·(1+h)) - V(S·(1-h))] / (2·S·h)
    """
    spot = market.quote(spot_id)
    h = spot * bump_pct
    
    # Bump up
    shock_up = SpotShock(name="up", spot_id=spot_id, bump=bump_pct, bump_mode="relative")
    market_up = shock_up.apply(market)
    pv_up = pricer.price(portfolio, market_up).totals.pv
    
    # Bump down
    shock_down = SpotShock(name="down", spot_id=spot_id, bump=-bump_pct, bump_mode="relative")
    market_down = shock_down.apply(market)
    pv_down = pricer.price(portfolio, market_down).totals.pv
    
    delta = (pv_up - pv_down) / (2 * h)
    return delta


def bump_and_reprice_gamma(
    portfolio: Portfolio,
    market: Market,
    pricer: PortfolioPricer,
    spot_id: MarketId,
    bump_pct: float = 0.01,
) -> float:
    """
    Compute gamma via central difference second derivative.
    
    Formula
    -------
    Γ ≈ [V(S+h) - 2V(S) + V(S-h)] / h²
    """
    spot = market.quote(spot_id)
    h = spot * bump_pct
    
    shock_up = SpotShock(name="up", spot_id=spot_id, bump=bump_pct, bump_mode="relative")
    shock_down = SpotShock(name="down", spot_id=spot_id, bump=-bump_pct, bump_mode="relative")
    
    pv_up = pricer.price(portfolio, shock_up.apply(market)).totals.pv
    pv_down = pricer.price(portfolio, shock_down.apply(market)).totals.pv
    pv_base = pricer.price(portfolio, market).totals.pv
    
    gamma = (pv_up - 2 * pv_base + pv_down) / (h**2)
    return gamma


def bump_and_reprice_vega(
    portfolio: Portfolio,
    market: Market,
    pricer: PortfolioPricer,
    vol_id: MarketId,
    bump_abs: float = 0.01,
) -> float:
    """
    Compute vega via central difference.
    
    Formula
    -------
    ν ≈ [V(σ+h) - V(σ-h)] / (2h)
    """
    shock_up = VolShock(name="up", vol_id=vol_id, bump=bump_abs, bump_mode="absolute")
    shock_down = VolShock(name="down", vol_id=vol_id, bump=-bump_abs, bump_mode="absolute")
    
    pv_up = pricer.price(portfolio, shock_up.apply(market)).totals.pv
    pv_down = pricer.price(portfolio, shock_down.apply(market)).totals.pv
    
    vega = (pv_up - pv_down) / (2 * bump_abs)
    return vega


def run_manual_bump_and_reprice(
    portfolio: Portfolio,
    market: Market,
    pricer: PortfolioPricer,
    analytic_greeks: dict,
) -> Tuple[float, float, float]:
    """
    Run manual bump-and-reprice and compare to analytical Greeks.
    
    Returns
    -------
    Tuple[float, float, float]
        FD delta, gamma, vega.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 1: Manual Bump-and-Reprice Greeks")
    logger.info("=" * 70)
    
    # Compute numerical Greeks
    fd_delta = bump_and_reprice_delta(portfolio, market, pricer, EURUSD_SPOT, 0.01)
    fd_gamma = bump_and_reprice_gamma(portfolio, market, pricer, EURUSD_SPOT, 0.01)
    fd_vega = bump_and_reprice_vega(portfolio, market, pricer, EURUSD_VOL, 0.01)
    
    logger.info("")
    logger.info("Bump-and-Reprice Greeks (1% bump):")
    logger.info(f"  Delta: {fd_delta:,.4f}")
    logger.info(f"  Gamma: {fd_gamma:,.4f}")
    logger.info(f"  Vega:  {fd_vega:,.4f}")
    
    # Compare to analytical
    logger.info("")
    logger.info("Comparison (FD vs Analytical):")
    logger.info(f"{'Greek':<10} {'Analytical':<15} {'FD':<15} {'Diff':<15} {'Diff %':<10}")
    logger.info("-" * 65)
    
    greeks_compare = [
        ('Delta', analytic_greeks.get('delta', 0), fd_delta),
        ('Gamma', analytic_greeks.get('gamma', 0), fd_gamma),
        ('Vega', analytic_greeks.get('vega', 0), fd_vega),
    ]
    
    for name, ana, fd in greeks_compare:
        diff = fd - ana
        diff_pct = (diff / ana * 100) if ana != 0 else 0
        logger.info(f"{name:<10} {ana:<15,.4f} {fd:<15,.4f} {diff:<15,.6f} {diff_pct:<10.4f}%")
    
    return fd_delta, fd_gamma, fd_vega


# =============================================================================
# SECTION 2: Bump Size Sensitivity
# =============================================================================

def run_bump_size_analysis(
    portfolio: Portfolio,
    market: Market,
    pricer: PortfolioPricer,
    analytic_greeks: dict,
) -> List[Tuple[float, float, float]]:
    """
    Analyze how bump size affects numerical accuracy.
    
    Returns
    -------
    List[Tuple[float, float, float]]
        List of (bump, fd_delta, error_pct) tuples.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 2: Bump Size Sensitivity Analysis")
    logger.info("=" * 70)
    
    bump_sizes = [0.001, 0.005, 0.01, 0.02, 0.05]
    delta_by_bump: List[Tuple[float, float, float]] = []
    
    analytical_delta = analytic_greeks.get('delta', 0)
    
    logger.info("")
    logger.info("Delta vs Bump Size:")
    logger.info(f"{'Bump %':<10} {'FD Delta':<15} {'Analytical':<15} {'Error %':<10}")
    logger.info("-" * 50)
    
    for bump in bump_sizes:
        fd_d = bump_and_reprice_delta(portfolio, market, pricer, EURUSD_SPOT, bump)
        error = (fd_d - analytical_delta) / analytical_delta * 100 if analytical_delta != 0 else 0
        delta_by_bump.append((bump, fd_d, error))
        logger.info(f"{bump*100:<10.2f} {fd_d:<15,.4f} {analytical_delta:<15,.4f} {error:<10.4f}%")
    
    logger.info("""
    Observation:
    - Smaller bumps have higher accuracy but more numerical noise
    - Larger bumps have more truncation error but less noise
    - Sweet spot typically around 0.5-1% for spot bumps
    """)
    
    return delta_by_bump


# =============================================================================
# SECTION 3: Dollar Greeks
# =============================================================================

def run_dollar_greeks_analysis(
    spot: float,
    analytic_greeks: dict,
) -> None:
    """
    Convert Greeks to dollar P&L estimates.
    
    Dollar Greeks show the P&L impact of market moves:
    - Delta P&L ≈ δ × S × ΔS/S
    - Gamma P&L ≈ ½ × Γ × (S × ΔS/S)²
    - Vega P&L ≈ ν × Δσ
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 3: Dollar Greeks (Scenario P&L)")
    logger.info("=" * 70)
    
    delta = analytic_greeks.get('delta', 0)
    gamma = analytic_greeks.get('gamma', 0)
    vega = analytic_greeks.get('vega', 0)
    
    logger.info(f"""
    Dollar Greeks translate sensitivities to P&L for market moves:
    
      Δ P&L (1% spot move) ≈ Delta × Spot × 0.01 + 0.5 × Gamma × (Spot × 0.01)²
    
    Position:
      Delta = {delta:,.2f}
      Gamma = {gamma:,.2f}
      Vega  = {vega:,.2f}
    
    Scenario P&L Estimates:
    """)
    
    scenarios = [
        ("Spot +1%", 0.01, 0),
        ("Spot -1%", -0.01, 0),
        ("Spot +5%", 0.05, 0),
        ("Vol +1pt", 0, 0.01),
        ("Vol -1pt", 0, -0.01),
    ]
    
    logger.info(f"{'Scenario':<15} {'Delta P&L':<15} {'Gamma P&L':<15} {'Vega P&L':<15} {'Total':<15}")
    logger.info("-" * 75)
    
    for name, spot_move, vol_move in scenarios:
        delta_pnl = delta * spot * spot_move if spot_move else 0
        gamma_pnl = 0.5 * gamma * (spot * spot_move)**2 if spot_move else 0
        vega_pnl = vega * vol_move if vol_move else 0
        total = delta_pnl + gamma_pnl + vega_pnl
        
        logger.info(f"{name:<15} {delta_pnl:>+12,.2f}   {gamma_pnl:>+12,.2f}   {vega_pnl:>+12,.2f}   {total:>+12,.2f}")


# =============================================================================
# SECTION 4: Visualization
# =============================================================================

def visualize_greeks(
    market: Market,
    portfolio: Portfolio,
    bsm_pricer: FxVanillaEuropeanOptionBsmPricer,
    delta_by_bump: List[Tuple[float, float, float]],
    fd_delta: float,
    fd_gamma: float,
    fd_vega: float,
    analytic_greeks: dict,
) -> None:
    """
    Create comprehensive Greeks visualizations.
    """
    if not MATPLOTLIB_AVAILABLE or not ENABLE_PLOTTING:
        logger.info("Skipping plots (matplotlib not available or disabled)")
        return
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 4: Visualization")
    logger.info("=" * 70)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    spot = market.quote(EURUSD_SPOT)
    
    # -------------------------------------------------------------------------
    # Plot 1: Bump size convergence
    # -------------------------------------------------------------------------
    ax = axes[0, 0]
    bumps_plot = [b * 100 for b, _, _ in delta_by_bump]
    errors_plot = [abs(e) for _, _, e in delta_by_bump]
    
    ax.semilogy(bumps_plot, errors_plot, 'o-', color='#2E86AB', linewidth=2, markersize=8)
    ax.set_xlabel('Bump Size (%)')
    ax.set_ylabel('|Error| (%)')
    ax.set_title('FD Delta Error vs Bump Size')
    ax.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 2: Greeks comparison bar chart
    # -------------------------------------------------------------------------
    ax = axes[0, 1]
    greek_names = ['Delta', 'Gamma\n(×1000)', 'Vega']
    ana_values = [
        analytic_greeks.get('delta', 0),
        analytic_greeks.get('gamma', 0) * 1000,
        analytic_greeks.get('vega', 0),
    ]
    fd_values = [fd_delta, fd_gamma * 1000, fd_vega]
    
    x = np.arange(len(greek_names))
    width = 0.35
    
    ax.bar(x - width/2, ana_values, width, label='Analytical', color='#2E86AB')
    ax.bar(x + width/2, fd_values, width, label='FD (1%)', color='#E94F37')
    ax.set_xticks(x)
    ax.set_xticklabels(greek_names)
    ax.set_ylabel('Greek Value')
    ax.set_title('Analytical vs FD Greeks')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # -------------------------------------------------------------------------
    # Plot 3: Delta profile vs spot
    # -------------------------------------------------------------------------
    ax = axes[1, 0]
    spot_range = np.linspace(spot * 0.85, spot * 1.15, 50)
    deltas = []
    
    # Get the option from the portfolio
    option = portfolio.positions[0].instrument
    
    for s in spot_range:
        shocked_market = Market(
            asof=market.asof,
            quotes={EURUSD_SPOT: Quote(value=s)},
            curves=market.curves,
            vols=market.vols,
        )
        greeks = bsm_pricer.greeks(option, shocked_market)
        deltas.append(greeks.get('delta', 0))
    
    ax.plot(spot_range, deltas, color='#2E86AB', linewidth=2)
    ax.axvline(spot, color='gray', linestyle='--', alpha=0.7, label=f'Current: {spot}')
    ax.axvline(option.strike, color='red', linestyle=':', alpha=0.7, label=f'Strike: {option.strike}')
    ax.set_xlabel('Spot')
    ax.set_ylabel('Delta')
    ax.set_title('Delta vs Spot (shows Gamma shape)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 4: Vega profile by expiry
    # -------------------------------------------------------------------------
    ax = axes[1, 1]
    expiries = np.linspace(0.1, 2.0, 30)
    vegas = []
    
    for exp in expiries:
        temp_option = FxVanillaEuropeanOption(
            option_type="call",
            strike=option.strike,
            expiry=exp,
            notional=option.notional,
            spot_id=EURUSD_SPOT,
            domestic_curve_id=USD_CURVE,
            foreign_curve_id=EUR_CURVE,
            vol_id=EURUSD_VOL,
        )
        greeks = bsm_pricer.greeks(temp_option, market)
        vegas.append(greeks.get('vega', 0))
    
    ax.plot(expiries, np.array(vegas) / 1000, color='#8B5CF6', linewidth=2)
    ax.axvline(option.expiry, color='gray', linestyle='--', alpha=0.7, label=f'Current: {option.expiry}Y')
    ax.set_xlabel('Time to Expiry (years)')
    ax.set_ylabel('Vega (thousands)')
    ax.set_title('Vega vs Expiry')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
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
    │  1. Greek Computation Methods:                                      │
    │     - Analytical: Exact, fast, but model-specific                   │
    │     - FD (bump-and-reprice): Universal, but slower                  │
    │                                                                      │
    │  2. Bump Size Trade-off:                                            │
    │     - Small bumps: More accurate, more numerical noise              │
    │     - Large bumps: More truncation error, less noise                │
    │     - Optimal: ~0.5-1% for spot, ~1pt for vol                       │
    │                                                                      │
    │  3. Dollar Greeks:                                                  │
    │     - Delta P&L ≈ δ × S × ΔS/S                                      │
    │     - Gamma P&L ≈ ½ × Γ × (S × ΔS/S)²                               │
    │     - Vega P&L  ≈ ν × Δσ                                            │
    │                                                                      │
    │  4. Greek Profiles:                                                 │
    │     - Delta increases towards ITM (S → ∞)                           │
    │     - Gamma peaks at ATM                                            │
    │     - Vega peaks for longer expiries, ATM                           │
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
        # Setup
        market, portfolio, portfolio_pricer, bsm_pricer, base_pv, analytic_greeks = (
            create_market_and_portfolio()
        )
        
        # Section 1: Manual bump-and-reprice
        fd_delta, fd_gamma, fd_vega = run_manual_bump_and_reprice(
            portfolio, market, portfolio_pricer, analytic_greeks
        )
        
        # Section 2: Bump size analysis
        delta_by_bump = run_bump_size_analysis(
            portfolio, market, portfolio_pricer, analytic_greeks
        )
        
        # Section 3: Dollar Greeks
        spot = market.quote(EURUSD_SPOT)
        run_dollar_greeks_analysis(spot, analytic_greeks)
        
        # Section 4: Visualization
        visualize_greeks(
            market, portfolio, bsm_pricer, delta_by_bump,
            fd_delta, fd_gamma, fd_vega, analytic_greeks
        )
        
        # Summary
        print_summary()
        
        logger.info("Example completed successfully!")
        
    except Exception as e:
        logger.exception(f"Example failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sensitivities Computation Example",
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

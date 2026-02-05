#!/usr/bin/env python3
"""
===============================================================================
Portfolio Pricing: Aggregating Positions
===============================================================================

This example demonstrates portfolio-level pricing - a core capability for
managing multi-instrument trading books.

Learning Objectives
-------------------
1. **Portfolio Construction**: Build portfolios with multiple positions
2. **PortfolioPricer**: Price all positions in a single call
3. **Greeks Aggregation**: Sum sensitivities across positions
4. **P&L Profile Analysis**: Compute portfolio P&L under market moves

Mathematical Framework
----------------------
Portfolio valuation is additive:

    PV_portfolio = Σ (quantity_i × PV_i)

Greeks aggregate linearly:
    
    Δ_portfolio = Σ (quantity_i × Δ_i)
    Γ_portfolio = Σ (quantity_i × Γ_i)
    ν_portfolio = Σ (quantity_i × ν_i)

Production Context
------------------
At a hedge fund:
- Portfolios contain hundreds to thousands of positions
- Greeks are aggregated for hedging and limit monitoring
- P&L attribution decomposes daily P&L by position and Greek
- Portfolio limits (max delta, max vega) control risk

Prerequisites
-------------
- Examples in fundamentals/ and pricing/01_fx_vanilla_pricing.py
- Understanding of Greeks

Run This Example
----------------
    cd /path/to/QuantStrata
    PYTHONPATH=. python examples/pricing/03_portfolio_pricing.py

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
from typing import List, Tuple

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

from src.instruments.fx.options.vanilla import FxVanillaEuropeanOption
from src.portfolio.core import Portfolio, Position
from src.portfolio.portfolio import PortfolioPricer
from src.pricers.registry import PricerRegistry
from src.pricers.fx.european_bsm import FxVanillaEuropeanOptionBsmPricer


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
# SECTION 1: Market Setup
# =============================================================================

def create_market() -> Tuple[Market, dict]:
    """
    Create market snapshot for portfolio pricing.
    
    Returns
    -------
    Tuple[Market, dict]
        Market and parameters dictionary.
    """
    logger.info("=" * 70)
    logger.info("SECTION 1: Market Setup")
    logger.info("=" * 70)
    
    # Market data
    spot = 1.0850
    r_usd = 0.05
    r_eur = 0.02
    vol = 0.10
    
    market = Market(
        asof="2026-01-28",
        quotes={EURUSD_SPOT: Quote(value=spot)},
        curves={
            USD_CURVE: FlatZeroRateCurve(continuously_compounded_rate=r_usd),
            EUR_CURVE: FlatZeroRateCurve(continuously_compounded_rate=r_eur),
        },
        vols={EURUSD_VOL: FlatVolSurface(sigma=vol)},
    )
    
    params = {"spot": spot, "r_usd": r_usd, "r_eur": r_eur, "vol": vol}
    
    logger.info("")
    logger.info("Market data:")
    logger.info(f"  EUR/USD Spot: {spot}")
    logger.info(f"  USD Rate: {r_usd:.2%}")
    logger.info(f"  EUR Rate: {r_eur:.2%}")
    logger.info(f"  Volatility: {vol:.2%}")
    
    return market, params


# =============================================================================
# SECTION 2: Build Portfolio
# =============================================================================

def create_option(
    strike: float,
    expiry: float,
    is_call: bool,
    notional: float,
) -> FxVanillaEuropeanOption:
    """
    Create an FX vanilla option with correct API.
    
    Parameters
    ----------
    strike : float
        Option strike.
    expiry : float
        Time to expiry in years.
    is_call : bool
        True for call, False for put.
    notional : float
        Notional in foreign currency (EUR).
    
    Returns
    -------
    EuropeanFxVanillaOption
        The option instrument.
    """
    return FxVanillaEuropeanOption(
        option_type="call" if is_call else "put",
        strike=strike,
        expiry=expiry,
        notional=notional,
        spot_id=EURUSD_SPOT,
        domestic_curve_id=USD_CURVE,
        foreign_curve_id=EUR_CURVE,
        vol_id=EURUSD_VOL,
    )


def build_portfolio() -> Portfolio:
    """
    Build a portfolio with multiple positions.
    
    Portfolio Structure
    -------------------
    - Long ATM call (directional upside)
    - Short OTM call (cap upside, reduce cost)
    - Short OTM put (collect premium, take downside risk)
    - Long 1Y call (longer-dated exposure)
    
    This structure is a call spread plus a short put, typical for
    expressing a moderately bullish view while collecting premium.
    
    Returns
    -------
    Portfolio
        The constructed portfolio.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 2: Building Portfolio")
    logger.info("=" * 70)
    
    # Portfolio: Long call spread + short put
    positions = [
        Position(
            position_id="LONG_CALL_ATM",
            instrument=create_option(strike=1.08, expiry=0.5, is_call=True, notional=5_000_000),
            quantity=1.0,
        ),
        Position(
            position_id="SHORT_CALL_OTM",
            instrument=create_option(strike=1.12, expiry=0.5, is_call=True, notional=5_000_000),
            quantity=-1.0,  # Short position
        ),
        Position(
            position_id="SHORT_PUT_OTM",
            instrument=create_option(strike=1.04, expiry=0.5, is_call=False, notional=2_000_000),
            quantity=-1.0,
        ),
        Position(
            position_id="LONG_CALL_1Y",
            instrument=create_option(strike=1.10, expiry=1.0, is_call=True, notional=3_000_000),
            quantity=1.0,
        ),
    ]
    
    portfolio = Portfolio(positions=positions)
    
    logger.info("")
    logger.info(f"Portfolio created with {len(positions)} positions:")
    logger.info("")
    logger.info(f"{'Position ID':<20} {'Type':<12} {'Strike':<10} {'Expiry':<10} {'Notional':<15} {'Qty':<6}")
    logger.info("-" * 73)
    
    for pos in positions:
        opt = pos.instrument
        opt_type = "Call" if opt.option_type == "call" else "Put"
        logger.info(
            f"{pos.position_id:<20} {opt_type:<12} {opt.strike:<10.4f} "
            f"{opt.expiry:<10.2f} {opt.notional:>12,.0f}   {pos.quantity:+.0f}"
        )
    
    return portfolio


# =============================================================================
# SECTION 3: Setup Pricer Registry
# =============================================================================

def setup_pricer_registry() -> Tuple[PricerRegistry, PortfolioPricer]:
    """
    Configure pricer registry and portfolio pricer.
    
    The PricerRegistry maps instrument types to pricers, allowing
    the PortfolioPricer to automatically select the correct pricer
    for each instrument.
    
    Returns
    -------
    Tuple[PricerRegistry, PortfolioPricer]
        Registry and portfolio pricer.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 3: Pricer Registry Setup")
    logger.info("=" * 70)
    
    # Create registry and register pricers
    registry = PricerRegistry()
    registry.register(FxVanillaEuropeanOption, FxVanillaEuropeanOptionBsmPricer())
    
    # Create portfolio pricer
    portfolio_pricer = PortfolioPricer(pricer_registry=registry)
    
    logger.info("")
    logger.info("Pricer registry configured:")
    logger.info("  EuropeanFxVanillaOption -> FxEuropeanVanillaBsmPricer")
    
    return registry, portfolio_pricer


# =============================================================================
# SECTION 4: Price Portfolio
# =============================================================================

def price_portfolio(
    portfolio: Portfolio,
    market: Market,
    portfolio_pricer: PortfolioPricer,
) -> Tuple[float, dict]:
    """
    Price the portfolio and extract results.
    
    Returns
    -------
    Tuple[float, dict]
        Total PV and totals Greeks.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 4: Portfolio Pricing Results")
    logger.info("=" * 70)
    
    # Price portfolio
    result = portfolio_pricer.price(portfolio, market)
    
    logger.info("")
    logger.info(f"{'Position ID':<20} {'PV (USD)':<15} {'Delta':<12} {'Gamma':<12} {'Vega':<12}")
    logger.info("-" * 71)
    
    for pos_result in result.per_position:
        pv = pos_result.pv
        delta = pos_result.greeks.get('delta', 0)
        gamma = pos_result.greeks.get('gamma', 0)
        vega = pos_result.greeks.get('vega', 0)
        logger.info(
            f"{pos_result.position_id:<20} {pv:>12,.2f}   {delta:>10,.2f} {gamma:>10,.2f} {vega:>10,.2f}"
        )
    
    logger.info("-" * 71)
    total_delta = result.totals.greeks.get('delta', 0)
    total_gamma = result.totals.greeks.get('gamma', 0)
    total_vega = result.totals.greeks.get('vega', 0)
    logger.info(
        f"{'TOTAL':<20} {result.totals.pv:>12,.2f}   "
        f"{total_delta:>10,.2f} {total_gamma:>10,.2f} {total_vega:>10,.2f}"
    )
    
    return result.totals.pv, result.totals.greeks


# =============================================================================
# SECTION 5: Portfolio Analytics
# =============================================================================

def run_portfolio_analytics(
    portfolio: Portfolio,
    total_pv: float,
    total_greeks: dict,
    params: dict,
) -> None:
    """
    Compute portfolio-level analytics.
    
    Analytics include:
    - Total notional exposure
    - PV as percentage of notional
    - Dollar Greeks (P&L for 1% moves)
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 5: Portfolio Analytics")
    logger.info("=" * 70)
    
    spot = params["spot"]
    
    total_notional = sum(abs(pos.quantity * pos.instrument.notional) for pos in portfolio.positions)
    total_delta = total_greeks.get('delta', 0)
    total_gamma = total_greeks.get('gamma', 0)
    total_vega = total_greeks.get('vega', 0)
    
    logger.info("")
    logger.info("Portfolio Summary:")
    logger.info(f"  Total Notional (abs): {total_notional:>15,.0f} EUR")
    logger.info(f"  Total PV:             {total_pv:>15,.2f} USD")
    logger.info(f"  PV as % of Notional:  {total_pv/total_notional*100:>14.3f}%")
    
    logger.info("")
    logger.info("Risk Metrics:")
    logger.info(f"  Delta (total):        {total_delta:>15,.2f}")
    logger.info(f"  Gamma (total):        {total_gamma:>15,.2f}")
    logger.info(f"  Vega (total):         {total_vega:>15,.2f}")
    
    # Dollar Greeks (sensitivity to 1% move)
    delta_1pct = total_delta * spot * 0.01
    gamma_1pct = 0.5 * total_gamma * (spot * 0.01)**2
    vega_1pt = total_vega * 0.01
    
    logger.info("")
    logger.info("Scenario P&L Estimates:")
    logger.info(f"  Spot +1%:  Delta P&L = {delta_1pct:>+12,.2f} USD")
    logger.info(f"  Spot +1%:  Gamma P&L = {gamma_1pct:>+12,.2f} USD")
    logger.info(f"  Vol +1pt:  Vega P&L  = {vega_1pt:>+12,.2f} USD")


# =============================================================================
# SECTION 6: P&L Profile Analysis
# =============================================================================

def compute_pnl_profile(
    portfolio: Portfolio,
    market: Market,
    portfolio_pricer: PortfolioPricer,
    total_pv: float,
    params: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute P&L profile across spot range.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Spot range and P&L profile.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 6: P&L Profile Analysis")
    logger.info("=" * 70)
    
    spot = params["spot"]
    
    # Compute P&L across spot range
    spot_range = np.linspace(spot * 0.90, spot * 1.10, 50)
    pnl_profile = []
    
    for s in spot_range:
        shocked_market = Market(
            asof=market.asof,
            quotes={EURUSD_SPOT: Quote(value=s)},
            curves=market.curves,
            vols=market.vols,
        )
        shocked_result = portfolio_pricer.price(portfolio, shocked_market)
        pnl = shocked_result.totals.pv - total_pv
        pnl_profile.append(pnl)
    
    pnl_profile = np.array(pnl_profile)
    
    # Find breakeven points
    zero_crossings = np.where(np.diff(np.sign(pnl_profile)))[0]
    
    logger.info("")
    logger.info("P&L Profile:")
    logger.info(f"  Spot range: {spot_range[0]:.4f} to {spot_range[-1]:.4f}")
    logger.info(f"  P&L range: {pnl_profile.min():+,.2f} to {pnl_profile.max():+,.2f}")
    
    if len(zero_crossings) > 0:
        logger.info("")
        logger.info("  Breakeven points:")
        for idx in zero_crossings:
            be_spot = (spot_range[idx] + spot_range[idx + 1]) / 2
            logger.info(f"    {be_spot:.4f} ({(be_spot / spot - 1) * 100:+.2f}% from current)")
    
    return spot_range, pnl_profile


# =============================================================================
# SECTION 7: Visualization
# =============================================================================

def visualize_results(
    portfolio: Portfolio,
    market: Market,
    portfolio_pricer: PortfolioPricer,
    total_pv: float,
    total_greeks: dict,
    spot_range: np.ndarray,
    pnl_profile: np.ndarray,
    params: dict,
) -> None:
    """
    Create comprehensive visualizations.
    """
    if not MATPLOTLIB_AVAILABLE or not ENABLE_PLOTTING:
        logger.info("Skipping plots (matplotlib not available or disabled)")
        return
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 7: Visualization")
    logger.info("=" * 70)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    spot = params["spot"]
    result = portfolio_pricer.price(portfolio, market)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # -------------------------------------------------------------------------
    # Plot 1: Position PV breakdown
    # -------------------------------------------------------------------------
    ax = axes[0, 0]
    position_ids = [r.position_id for r in result.per_position]
    pvs = [r.pv for r in result.per_position]
    colors = ['#10B981' if pv > 0 else '#E94F37' for pv in pvs]
    
    bars = ax.barh(position_ids, pvs, color=colors)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('PV (USD)')
    ax.set_title('Position PV Breakdown')
    ax.grid(True, alpha=0.3, axis='x')
    
    for bar, pv in zip(bars, pvs):
        ax.text(
            pv + (5000 if pv > 0 else -5000),
            bar.get_y() + bar.get_height() / 2,
            f'{pv:,.0f}',
            ha='left' if pv > 0 else 'right',
            va='center',
            fontsize=9,
        )
    
    # -------------------------------------------------------------------------
    # Plot 2: Greeks breakdown
    # -------------------------------------------------------------------------
    ax = axes[0, 1]
    total_delta = total_greeks.get('delta', 0)
    total_gamma = total_greeks.get('gamma', 0)
    total_vega = total_greeks.get('vega', 0)
    
    greek_names = ['Delta', 'Gamma', 'Vega']
    greek_values = [total_delta, total_gamma * 1000, total_vega]  # Scale gamma
    
    x = np.arange(len(greek_names))
    width = 0.6
    colors = ['#2E86AB' if v > 0 else '#E94F37' for v in greek_values]
    
    bars = ax.bar(x, greek_values, width, color=colors)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(greek_names)
    ax.set_ylabel('Greek Value')
    ax.set_title('Portfolio Greeks (Gamma ×1000)')
    ax.grid(True, alpha=0.3, axis='y')
    
    # -------------------------------------------------------------------------
    # Plot 3: P&L Profile
    # -------------------------------------------------------------------------
    ax = axes[1, 0]
    ax.plot(spot_range, pnl_profile / 1000, color='#2E86AB', linewidth=2)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(spot, color='gray', linestyle='--', alpha=0.7, label=f'Current: {spot}')
    ax.fill_between(
        spot_range, pnl_profile / 1000, 0,
        where=(pnl_profile > 0), alpha=0.3, color='#10B981', label='Profit',
    )
    ax.fill_between(
        spot_range, pnl_profile / 1000, 0,
        where=(pnl_profile <= 0), alpha=0.3, color='#E94F37', label='Loss',
    )
    ax.set_xlabel('EUR/USD Spot')
    ax.set_ylabel('P&L (USD thousands)')
    ax.set_title('Portfolio P&L Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 4: Delta profile
    # -------------------------------------------------------------------------
    ax = axes[1, 1]
    delta_profile = []
    
    for s in spot_range:
        shocked_market = Market(
            asof=market.asof,
            quotes={EURUSD_SPOT: Quote(value=s)},
            curves=market.curves,
            vols=market.vols,
        )
        shocked_result = portfolio_pricer.price(portfolio, shocked_market)
        delta_profile.append(shocked_result.totals.greeks.get('delta', 0))
    
    ax.plot(spot_range, delta_profile, color='#8B5CF6', linewidth=2)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(spot, color='gray', linestyle='--', alpha=0.7, label=f'Current: {spot}')
    ax.set_xlabel('EUR/USD Spot')
    ax.set_ylabel('Portfolio Delta')
    ax.set_title('Delta Profile (shows Gamma)')
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
    │  1. Portfolio Structure:                                            │
    │     - Portfolio contains a list of Positions                        │
    │     - Position = Instrument + Quantity + ID                         │
    │     - Quantity can be negative (short positions)                    │
    │                                                                      │
    │  2. Pricer Registry:                                                │
    │     - Maps instrument types to pricers                              │
    │     - Allows flexible pricer selection                              │
    │                                                                      │
    │  3. PortfolioPricer:                                                │
    │     - Prices all positions in a single call                         │
    │     - Aggregates PV and Greeks                                      │
    │     - Returns per-position and total results                        │
    │                                                                      │
    │  4. Portfolio Analytics:                                            │
    │     - P&L profile shows profit/loss at different spots              │
    │     - Delta profile shows how delta changes (gamma effect)          │
    │     - Breakeven analysis identifies key levels                      │
    │                                                                      │
    │  5. Greeks Aggregation:                                             │
    │     - Greeks add linearly across positions                          │
    │     - Sign of quantity affects Greek direction                      │
    │                                                                      │
    │  NEXT: See examples/risk/ for scenario analysis                     │
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
        # Section 1: Market setup
        market, params = create_market()
        
        # Section 2: Build portfolio
        portfolio = build_portfolio()
        
        # Section 3: Setup pricer registry
        registry, portfolio_pricer = setup_pricer_registry()
        
        # Section 4: Price portfolio
        total_pv, total_greeks = price_portfolio(portfolio, market, portfolio_pricer)
        
        # Section 5: Portfolio analytics
        run_portfolio_analytics(portfolio, total_pv, total_greeks, params)
        
        # Section 6: P&L profile
        spot_range, pnl_profile = compute_pnl_profile(
            portfolio, market, portfolio_pricer, total_pv, params
        )
        
        # Section 7: Visualization
        visualize_results(
            portfolio, market, portfolio_pricer, total_pv, total_greeks,
            spot_range, pnl_profile, params
        )
        
        # Summary
        print_summary()
        
        logger.info("Example completed successfully!")
        
    except Exception as e:
        logger.exception(f"Example failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Portfolio Pricing Example",
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

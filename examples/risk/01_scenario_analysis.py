#!/usr/bin/env python3
"""
===============================================================================
Scenario Analysis: Running Portfolios Through Market Shocks
===============================================================================

This example demonstrates comprehensive scenario analysis - a core risk
management capability for evaluating portfolio P&L under stressed conditions.

Learning Objectives
-------------------
1. **Single Shocks**: Apply spot, vol, and rate shocks individually
2. **Combined Scenarios**: Build realistic stress scenarios (risk-off, crisis)
3. **Scenario Ladders**: Systematic sensitivity analysis across bump ranges
4. **P&L Analysis**: Interpret scenario results for risk management

Mathematical Framework
----------------------
Scenario P&L is computed as:

    P&L_scenario = PV_shocked - PV_base

For Greeks-based approximation:
    
    ΔP&L ≈ δ·ΔS + ½Γ·(ΔS)² + ν·Δσ + ρ·Δr

Production Context
------------------
At a hedge fund:
- Daily stress testing against historical crisis scenarios
- Regulatory stress tests (CCAR, FRTB scenarios)
- Pre-trade risk checks against scenario limits
- VaR backtesting via scenario ladders

Prerequisites
-------------
- Examples in fundamentals/
- Understanding of shocks from 06_scenario_shocks.py

Run This Example
----------------
    cd /path/to/QuantStrata
    PYTHONPATH=. python examples/risk/01_scenario_analysis.py

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

from src.instruments.fx.options.vanilla import EuropeanFxVanillaOption
from src.portfolio.core import Portfolio, Position
from src.portfolio.portfolio import PortfolioPricer
from src.pricers.registry import PricerRegistry
from src.pricers.fx.european_bsm import FxEuropeanVanillaBsmPricer

# Try to import scenario runner (may not exist)
try:
    from src.risk.scenarios.runner import run_portfolio_scenarios
    RUNNER_AVAILABLE = True
except ImportError:
    RUNNER_AVAILABLE = False


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
# SETUP: Market, Portfolio, Pricer
# =============================================================================

def create_market_and_portfolio() -> Tuple[Market, Portfolio, PortfolioPricer, float]:
    """
    Create the market, portfolio, and pricer for scenario analysis.
    
    Returns
    -------
    Tuple[Market, Portfolio, PortfolioPricer, float]
        Market, portfolio, pricer, and base PV.
    
    Portfolio Structure
    -------------------
    Long ATM straddle (10M EUR):
    - Long 1x ATM Call
    - Long 1x ATM Put
    
    This is a typical vol-trading position that profits from large moves
    in either direction and from volatility increases.
    
    Production Notes
    ----------------
    - Straddle is delta-neutral at inception
    - Long gamma, long vega
    - Theta negative (time decay)
    """
    logger.info("=" * 70)
    logger.info("SETUP: Market, Portfolio, and Pricer")
    logger.info("=" * 70)
    
    # -------------------------------------------------------------------------
    # Market parameters
    # -------------------------------------------------------------------------
    spot = 1.0850
    r_usd = 0.05   # USD rate (domestic)
    r_eur = 0.02   # EUR rate (foreign)
    vol = 0.10     # 10% implied volatility
    
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
    
    logger.info(f"Base market:")
    logger.info(f"  EUR/USD: {spot}, Vol: {vol:.1%}, USD: {r_usd:.1%}, EUR: {r_eur:.1%}")
    
    # -------------------------------------------------------------------------
    # Create option helper
    # -------------------------------------------------------------------------
    def create_option(
        strike: float,
        expiry: float,
        is_call: bool,
        notional: float,
    ) -> EuropeanFxVanillaOption:
        """Create an FX vanilla option with correct API."""
        return EuropeanFxVanillaOption(
            option_type="call" if is_call else "put",
            strike=strike,
            expiry=expiry,
            notional=notional,
            spot_id=EURUSD_SPOT,
            domestic_curve_id=USD_CURVE,
            foreign_curve_id=EUR_CURVE,
            vol_id=EURUSD_VOL,
        )
    
    # -------------------------------------------------------------------------
    # Build portfolio: Long straddle
    # -------------------------------------------------------------------------
    positions = [
        Position(
            position_id="LONG_CALL",
            instrument=create_option(strike=1.085, expiry=0.5, is_call=True, notional=10_000_000),
            quantity=1.0,
        ),
        Position(
            position_id="LONG_PUT",
            instrument=create_option(strike=1.085, expiry=0.5, is_call=False, notional=10_000_000),
            quantity=1.0,
        ),
    ]
    portfolio = Portfolio(positions=positions)
    
    # -------------------------------------------------------------------------
    # Setup pricer
    # -------------------------------------------------------------------------
    registry = PricerRegistry()
    registry.register(EuropeanFxVanillaOption, FxEuropeanVanillaBsmPricer())
    portfolio_pricer = PortfolioPricer(pricer_registry=registry)
    
    # -------------------------------------------------------------------------
    # Compute base PV
    # -------------------------------------------------------------------------
    base_result = portfolio_pricer.price(portfolio, market)
    base_pv = base_result.totals.pv
    
    logger.info("")
    logger.info("Portfolio: Long ATM Straddle (10M EUR)")
    logger.info(f"  Base PV: {base_pv:,.2f} USD")
    logger.info(f"  Delta: {base_result.totals.greeks.get('delta', 0):,.2f}")
    logger.info(f"  Gamma: {base_result.totals.greeks.get('gamma', 0):,.2f}")
    logger.info(f"  Vega:  {base_result.totals.greeks.get('vega', 0):,.2f}")
    
    return market, portfolio, portfolio_pricer, base_pv


# =============================================================================
# SECTION 1: Single Shock Scenarios
# =============================================================================

def run_single_shock_scenarios(
    market: Market,
    portfolio: Portfolio,
    pricer: PortfolioPricer,
    base_pv: float,
) -> Dict[str, float]:
    """
    Run single-factor shock scenarios.
    
    Parameters
    ----------
    market : Market
        Base market.
    portfolio : Portfolio
        Portfolio to price.
    pricer : PortfolioPricer
        Portfolio pricer.
    base_pv : float
        Base PV for P&L calculation.
    
    Returns
    -------
    Dict[str, float]
        P&L by scenario name.
    
    Single Shocks Tested
    --------------------
    - Spot ±1%, ±5%
    - Vol ±1 point
    - USD rates ±50bp
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 1: Single Shock Scenarios")
    logger.info("=" * 70)
    
    # -------------------------------------------------------------------------
    # Define single shocks
    # -------------------------------------------------------------------------
    single_shocks = [
        SpotShock(name="Spot +1%", spot_id=EURUSD_SPOT, bump=0.01, bump_mode="relative"),
        SpotShock(name="Spot -1%", spot_id=EURUSD_SPOT, bump=-0.01, bump_mode="relative"),
        SpotShock(name="Spot +5%", spot_id=EURUSD_SPOT, bump=0.05, bump_mode="relative"),
        SpotShock(name="Spot -5%", spot_id=EURUSD_SPOT, bump=-0.05, bump_mode="relative"),
        VolShock(name="Vol +1pt", vol_id=EURUSD_VOL, bump=0.01, bump_mode="absolute"),
        VolShock(name="Vol -1pt", vol_id=EURUSD_VOL, bump=-0.01, bump_mode="absolute"),
        ParallelRateShock(name="USD +50bp", curve_id=USD_CURVE, rate_shift=0.005),
        ParallelRateShock(name="USD -50bp", curve_id=USD_CURVE, rate_shift=-0.005),
    ]
    
    # -------------------------------------------------------------------------
    # Run scenarios
    # -------------------------------------------------------------------------
    results: Dict[str, float] = {"BASE": 0.0}
    
    logger.info("")
    logger.info("Scenario Results:")
    logger.info(f"{'Scenario':<15} {'PV (USD)':<15} {'P&L (USD)':<15} {'P&L %':<10}")
    logger.info("-" * 55)
    
    for shock in single_shocks:
        shocked_market = shock.apply(market)
        result = pricer.price(portfolio, shocked_market)
        pnl = result.totals.pv - base_pv
        pnl_pct = (pnl / base_pv) * 100 if base_pv != 0 else 0
        
        results[shock.name] = pnl
        logger.info(f"{shock.name:<15} {result.totals.pv:>12,.2f}   {pnl:>+12,.2f}   {pnl_pct:>+8.2f}%")
    
    return results


# =============================================================================
# SECTION 2: Combined Stress Scenarios
# =============================================================================

def run_combined_stress_scenarios(
    market: Market,
    portfolio: Portfolio,
    pricer: PortfolioPricer,
    base_pv: float,
) -> Dict[str, float]:
    """
    Run combined stress scenarios representing market crises.
    
    Parameters
    ----------
    market : Market
        Base market.
    portfolio : Portfolio
        Portfolio to price.
    pricer : PortfolioPricer
        Portfolio pricer.
    base_pv : float
        Base PV for P&L calculation.
    
    Returns
    -------
    Dict[str, float]
        P&L by scenario name.
    
    Stress Scenarios
    ----------------
    - Risk-Off: Spot down, vol up, rates down (flight to quality)
    - Risk-On: Spot up, vol down, rates up (risk appetite)
    - Vol Spike: Pure volatility shock
    - EUR Crisis: Major EUR selloff with vol spike
    - USD Weakness: USD selloff scenario
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 2: Combined Stress Scenarios")
    logger.info("=" * 70)
    
    explanation = """
    Stress scenarios combine multiple shocks to simulate market crises.
    Each scenario represents a coherent market narrative.
    """
    logger.info(explanation)
    
    # -------------------------------------------------------------------------
    # Define stress scenarios: (spot_bump, vol_bump, rate_bump)
    # -------------------------------------------------------------------------
    stress_scenarios = {
        "Risk-Off": (-0.05, 0.05, -0.01),      # Spot down, vol up, rates down
        "Risk-On": (0.03, -0.02, 0.005),        # Spot up, vol down, rates up
        "Vol Spike": (0.0, 0.10, 0.0),          # Pure vol spike (+10 points)
        "EUR Crisis": (-0.10, 0.08, -0.02),     # Major EUR selloff
        "USD Weakness": (0.08, 0.03, -0.01),    # USD selloff
    }
    
    stress_pnls: Dict[str, float] = {}
    
    logger.info("")
    logger.info("Stress Scenario Results:")
    logger.info(f"{'Scenario':<15} {'Spot Δ':<10} {'Vol Δ':<10} {'Rate Δ':<10} {'P&L (USD)':<15}")
    logger.info("-" * 60)
    
    for scenario_name, (spot_bump, vol_bump, rate_bump) in stress_scenarios.items():
        # Apply shocks sequentially
        shocked_market = market
        
        if spot_bump != 0:
            shock = SpotShock(
                name="spot",
                spot_id=EURUSD_SPOT,
                bump=spot_bump,
                bump_mode="relative",
            )
            shocked_market = shock.apply(shocked_market)
        
        if vol_bump != 0:
            shock = VolShock(
                name="vol",
                vol_id=EURUSD_VOL,
                bump=vol_bump,
                bump_mode="absolute",
            )
            shocked_market = shock.apply(shocked_market)
        
        if rate_bump != 0:
            shock = ParallelRateShock(
                name="rate",
                curve_id=USD_CURVE,
                rate_shift=rate_bump,
            )
            shocked_market = shock.apply(shocked_market)
        
        # Price and compute P&L
        result = pricer.price(portfolio, shocked_market)
        pnl = result.totals.pv - base_pv
        stress_pnls[scenario_name] = pnl
        
        logger.info(
            f"{scenario_name:<15} {spot_bump*100:>+8.1f}%  {vol_bump*100:>+7.1f}pt  "
            f"{rate_bump*100:>+7.0f}bp  {pnl:>+12,.2f}"
        )
    
    return stress_pnls


# =============================================================================
# SECTION 3: Spot Ladder Analysis
# =============================================================================

def run_spot_ladder(
    market: Market,
    portfolio: Portfolio,
    pricer: PortfolioPricer,
    base_pv: float,
) -> List[Tuple[float, float, float]]:
    """
    Run spot ladder analysis for systematic sensitivity.
    
    Parameters
    ----------
    market : Market
        Base market.
    portfolio : Portfolio
        Portfolio to price.
    pricer : PortfolioPricer
        Portfolio pricer.
    base_pv : float
        Base PV for P&L calculation.
    
    Returns
    -------
    List[Tuple[float, float, float]]
        List of (bump, spot, pnl) tuples.
    
    Spot Ladder
    -----------
    Systematic bumps: -10%, -5%, -2%, -1%, 0%, +1%, +2%, +5%, +10%
    
    Used for:
    - Validating delta and gamma Greeks
    - Finding breakeven points
    - Estimating scenario limits
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 3: Spot Ladder Analysis")
    logger.info("=" * 70)
    
    spot = market.quote(EURUSD_SPOT)
    spot_bumps = [-0.10, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.10]
    
    ladder_data: List[Tuple[float, float, float]] = []
    
    logger.info("")
    logger.info("Spot Ladder:")
    logger.info(f"{'Bump':<10} {'Spot':<12} {'PV':<15} {'P&L':<15} {'Δ P&L %':<10}")
    logger.info("-" * 62)
    
    for bump in spot_bumps:
        if bump == 0:
            pv = base_pv
            pnl = 0.0
        else:
            shock = SpotShock(
                name=f"spot_{bump*100:+.0f}pct",
                spot_id=EURUSD_SPOT,
                bump=bump,
                bump_mode="relative",
            )
            shocked_market = shock.apply(market)
            result = pricer.price(portfolio, shocked_market)
            pv = result.totals.pv
            pnl = pv - base_pv
        
        spotted = spot * (1 + bump)
        pnl_pct = pnl / base_pv * 100 if base_pv != 0 else 0
        ladder_data.append((bump, spotted, pnl))
        
        logger.info(f"{bump*100:>+8.0f}%  {spotted:<12.4f} {pv:>12,.2f}   {pnl:>+12,.2f}   {pnl_pct:>+8.2f}%")
    
    return ladder_data


# =============================================================================
# SECTION 4: Vol Ladder Analysis
# =============================================================================

def run_vol_ladder(
    market: Market,
    portfolio: Portfolio,
    pricer: PortfolioPricer,
    base_pv: float,
) -> List[Tuple[float, float, float]]:
    """
    Run vol ladder analysis for vega sensitivity.
    
    Parameters
    ----------
    market : Market
        Base market.
    portfolio : Portfolio
        Portfolio to price.
    pricer : PortfolioPricer
        Portfolio pricer.
    base_pv : float
        Base PV for P&L calculation.
    
    Returns
    -------
    List[Tuple[float, float, float]]
        List of (bump, vol, pnl) tuples.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 4: Volatility Ladder Analysis")
    logger.info("=" * 70)
    
    base_vol = 0.10  # 10%
    vol_bumps = [-0.05, -0.03, -0.01, 0.0, 0.01, 0.03, 0.05]
    
    ladder_data: List[Tuple[float, float, float]] = []
    
    logger.info("")
    logger.info("Vol Ladder:")
    logger.info(f"{'Bump':<10} {'Vol':<12} {'PV':<15} {'P&L':<15}")
    logger.info("-" * 52)
    
    for bump in vol_bumps:
        if bump == 0:
            pv = base_pv
            pnl = 0.0
        else:
            shock = VolShock(
                name=f"vol_{bump*100:+.0f}pt",
                vol_id=EURUSD_VOL,
                bump=bump,
                bump_mode="absolute",
            )
            shocked_market = shock.apply(market)
            result = pricer.price(portfolio, shocked_market)
            pv = result.totals.pv
            pnl = pv - base_pv
        
        vol_level = base_vol + bump
        ladder_data.append((bump, vol_level, pnl))
        
        logger.info(f"{bump*100:>+8.0f}pt  {vol_level*100:<10.1f}%  {pv:>12,.2f}   {pnl:>+12,.2f}")
    
    return ladder_data


# =============================================================================
# SECTION 5: Visualization
# =============================================================================

def visualize_scenarios(
    single_results: Dict[str, float],
    stress_results: Dict[str, float],
    spot_ladder: List[Tuple[float, float, float]],
    vol_ladder: List[Tuple[float, float, float]],
    spot: float,
    vol: float,
) -> None:
    """
    Create comprehensive scenario analysis visualizations.
    
    Four plots:
    1. Single shock P&L (horizontal bar)
    2. Stress scenario P&L (horizontal bar)
    3. Spot ladder P&L profile
    4. Vol ladder P&L profile
    """
    if not MATPLOTLIB_AVAILABLE or not ENABLE_PLOTTING:
        logger.info("Skipping plots (matplotlib not available or disabled)")
        return
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 5: Visualization")
    logger.info("=" * 70)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # -------------------------------------------------------------------------
    # Plot 1: Single shock P&L
    # -------------------------------------------------------------------------
    ax = axes[0, 0]
    shock_names = [n for n in single_results.keys() if n != "BASE"]
    pnls = [single_results[n] / 1000 for n in shock_names]  # In thousands
    colors = ['#10B981' if p > 0 else '#E94F37' for p in pnls]
    
    ax.barh(shock_names, pnls, color=colors)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('P&L (USD thousands)')
    ax.set_title('Single Shock P&L')
    ax.grid(True, alpha=0.3, axis='x')
    
    # -------------------------------------------------------------------------
    # Plot 2: Stress scenario P&L
    # -------------------------------------------------------------------------
    ax = axes[0, 1]
    stress_names = list(stress_results.keys())
    stress_vals = [stress_results[n] / 1000 for n in stress_names]
    colors = ['#10B981' if p > 0 else '#E94F37' for p in stress_vals]
    
    ax.barh(stress_names, stress_vals, color=colors)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('P&L (USD thousands)')
    ax.set_title('Stress Scenario P&L')
    ax.grid(True, alpha=0.3, axis='x')
    
    # -------------------------------------------------------------------------
    # Plot 3: Spot ladder P&L profile
    # -------------------------------------------------------------------------
    ax = axes[1, 0]
    spot_levels = [s for _, s, _ in spot_ladder]
    spot_pnls = [p / 1000 for _, _, p in spot_ladder]
    
    ax.plot(spot_levels, spot_pnls, 'o-', color='#2E86AB', linewidth=2, markersize=8)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(spot, color='gray', linestyle='--', alpha=0.7, label=f'Current: {spot}')
    ax.fill_between(
        spot_levels, spot_pnls, 0,
        where=(np.array(spot_pnls) > 0), alpha=0.3, color='#10B981',
    )
    ax.fill_between(
        spot_levels, spot_pnls, 0,
        where=(np.array(spot_pnls) <= 0), alpha=0.3, color='#E94F37',
    )
    ax.set_xlabel('EUR/USD Spot')
    ax.set_ylabel('P&L (USD thousands)')
    ax.set_title('Spot Ladder P&L Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 4: Vol ladder P&L profile
    # -------------------------------------------------------------------------
    ax = axes[1, 1]
    vol_levels = [v * 100 for _, v, _ in vol_ladder]
    vol_pnls = [p / 1000 for _, _, p in vol_ladder]
    
    ax.plot(vol_levels, vol_pnls, 's-', color='#8B5CF6', linewidth=2, markersize=8)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(vol * 100, color='gray', linestyle='--', alpha=0.7, label=f'Current: {vol*100:.0f}%')
    ax.fill_between(
        vol_levels, vol_pnls, 0,
        where=(np.array(vol_pnls) > 0), alpha=0.3, color='#10B981',
    )
    ax.fill_between(
        vol_levels, vol_pnls, 0,
        where=(np.array(vol_pnls) <= 0), alpha=0.3, color='#E94F37',
    )
    ax.set_xlabel('Implied Volatility (%)')
    ax.set_ylabel('P&L (USD thousands)')
    ax.set_title('Vol Ladder P&L Profile')
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
    │  1. Single Shocks:                                                  │
    │     - SpotShock: Perturb spot prices                                │
    │     - VolShock: Perturb volatility                                  │
    │     - ParallelRateShock: Shift yield curves                         │
    │                                                                      │
    │  2. Stress Testing:                                                 │
    │     - Combine multiple shocks for realistic scenarios               │
    │     - Model crisis events (risk-off, vol spike, etc.)               │
    │     - Apply shocks sequentially via .apply()                        │
    │                                                                      │
    │  3. Ladder Analysis:                                                │
    │     - Systematic bumps to single risk factor                        │
    │     - Shows P&L profile and breakeven points                        │
    │     - Validates Greeks (delta from spot ladder, vega from vol)      │
    │                                                                      │
    │  4. Straddle Characteristics:                                       │
    │     - Long gamma, long vega, delta-neutral                          │
    │     - Profits from large spot moves (either direction)              │
    │     - Profits from volatility increase                              │
    │                                                                      │
    │  NEXT: See 02_sensitivities_computation.py for formal Greeks        │
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
        market, portfolio, pricer, base_pv = create_market_and_portfolio()
        
        # Section 1: Single shocks
        single_results = run_single_shock_scenarios(market, portfolio, pricer, base_pv)
        
        # Section 2: Combined stress
        stress_results = run_combined_stress_scenarios(market, portfolio, pricer, base_pv)
        
        # Section 3: Spot ladder
        spot_ladder = run_spot_ladder(market, portfolio, pricer, base_pv)
        
        # Section 4: Vol ladder
        vol_ladder = run_vol_ladder(market, portfolio, pricer, base_pv)
        
        # Section 5: Visualization
        spot = market.quote(EURUSD_SPOT)
        vol = 0.10
        visualize_scenarios(single_results, stress_results, spot_ladder, vol_ladder, spot, vol)
        
        # Summary
        print_summary()
        
        logger.info("Example completed successfully!")
        
    except Exception as e:
        logger.exception(f"Example failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scenario Analysis Example",
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

#!/usr/bin/env python3
"""
===============================================================================
P&L Attribution: Greek-Based Explain
===============================================================================

This example demonstrates P&L attribution - decomposing portfolio P&L into
contributions from individual risk factors using Greeks.

Learning Objectives
-------------------
1. **P&L Attribution**: Understand how to explain P&L by risk factor
2. **Greek-Based Decomposition**: Use delta, gamma, vega, rho for explain
3. **Residual Analysis**: Identify unexplained P&L and model limitations
4. **Production Workflow**: End-of-day P&L explain process

Mathematical Framework
----------------------
First-order Taylor expansion of portfolio value:

    ΔV ≈ Δ·ΔS + ½Γ·(ΔS)² + ν·Δσ + ρ·Δr + Θ·Δt

P&L Attribution:
    - Delta P&L:    Δ × ΔS (spot move contribution)
    - Gamma P&L:    ½ × Γ × (ΔS)² (convexity contribution)
    - Vega P&L:     ν × Δσ (vol move contribution)
    - Rho P&L:      ρ × Δr (rate move contribution)
    - Theta P&L:    Θ × Δt (time decay)
    - Residual:     Actual P&L - Sum of attributed

Residual represents:
    - Higher-order terms (vanna, volga, charm, etc.)
    - Model error
    - Discrete hedging effects

Production Context
------------------
At a hedge fund:
- P&L explain is mandatory at end of day
- Large residuals trigger investigation
- Attribution drives risk factor analysis
- Used for performance attribution to traders

Prerequisites
-------------
- Understanding of Greeks (examples/risk/02_sensitivities_computation.py)
- Understanding of scenarios (examples/risk/01_scenario_analysis.py)

Run This Example
----------------
    cd /path/to/QuantStrata
    PYTHONPATH=. python examples/risk/03_pnl_attribution.py

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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

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
from src.marketdata.scenarios.shocks import SpotShock, VolShock

from src.instruments.fx.options.vanilla import EuropeanFxVanillaOption
from src.portfolio.core import Portfolio, Position
from src.portfolio.portfolio import PortfolioPricer
from src.pricers.registry import PricerRegistry
from src.pricers.fx.european_bsm import FxEuropeanVanillaBsmPricer

# Try to import attribution module
try:
    from src.risk.attribution.runner import (
        attribute_portfolio_scenarios,
        AttributionConfig,
    )
    ATTRIBUTION_AVAILABLE = True
except ImportError:
    ATTRIBUTION_AVAILABLE = False


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

# Market IDs
EURUSD_SPOT = MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD")
USD_CURVE = MarketId(asset_class="IR", mkt_type="CURVE", name="USD_OIS")
EUR_CURVE = MarketId(asset_class="IR", mkt_type="CURVE", name="EUR_OIS")
EURUSD_VOL = MarketId(asset_class="FX", mkt_type="VOL", name="EURUSD")


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MarketMove:
    """Represents a market move from T-1 to T."""
    spot_t0: float
    spot_t1: float
    vol_t0: float
    vol_t1: float
    r_dom_t0: float
    r_dom_t1: float
    r_for_t0: float
    r_for_t1: float
    
    @property
    def delta_spot(self) -> float:
        return self.spot_t1 - self.spot_t0
    
    @property
    def delta_vol(self) -> float:
        return self.vol_t1 - self.vol_t0
    
    @property
    def delta_r_dom(self) -> float:
        return self.r_dom_t1 - self.r_dom_t0
    
    @property
    def delta_r_for(self) -> float:
        return self.r_for_t1 - self.r_for_t0


@dataclass
class AttributionResult:
    """P&L attribution breakdown."""
    actual_pnl: float
    delta_pnl: float
    gamma_pnl: float
    vega_pnl: float
    theta_pnl: float
    rho_dom_pnl: float
    rho_for_pnl: float
    residual: float
    
    @property
    def explained_pnl(self) -> float:
        return (
            self.delta_pnl + self.gamma_pnl + self.vega_pnl +
            self.theta_pnl + self.rho_dom_pnl + self.rho_for_pnl
        )
    
    @property
    def explain_ratio(self) -> float:
        if abs(self.actual_pnl) < 1e-10:
            return 1.0 if abs(self.residual) < 1e-10 else 0.0
        return self.explained_pnl / self.actual_pnl


# =============================================================================
# SECTION 1: Setup
# =============================================================================

def create_market_t0() -> Tuple[Market, Dict[str, float]]:
    """
    Create T-1 (yesterday's) market snapshot.
    
    Returns
    -------
    Tuple[Market, Dict]
        Market and parameters.
    """
    logger.info("=" * 70)
    logger.info("SECTION 1: Market Setup")
    logger.info("=" * 70)
    
    # T-1 market data
    spot = 1.0850
    vol = 0.10
    r_dom = 0.05
    r_for = 0.04
    
    market = Market(
        asof="2026-01-27",
        quotes={EURUSD_SPOT: Quote(value=spot)},
        curves={
            USD_CURVE: FlatZeroRateCurve(continuously_compounded_rate=r_dom),
            EUR_CURVE: FlatZeroRateCurve(continuously_compounded_rate=r_for),
        },
        vols={EURUSD_VOL: FlatVolSurface(sigma=vol)},
    )
    
    params = {"spot": spot, "vol": vol, "r_dom": r_dom, "r_for": r_for}
    
    logger.info("")
    logger.info("T-1 (Yesterday) Market:")
    logger.info(f"  EUR/USD Spot: {spot:.4f}")
    logger.info(f"  Volatility:   {vol:.2%}")
    logger.info(f"  USD Rate:     {r_dom:.2%}")
    logger.info(f"  EUR Rate:     {r_for:.2%}")
    
    return market, params


def create_market_t1(move: MarketMove) -> Market:
    """
    Create T (today's) market snapshot after market moves.
    
    Parameters
    ----------
    move : MarketMove
        Market move from T-1 to T.
    
    Returns
    -------
    Market
        Today's market.
    """
    return Market(
        asof="2026-01-28",
        quotes={EURUSD_SPOT: Quote(value=move.spot_t1)},
        curves={
            USD_CURVE: FlatZeroRateCurve(continuously_compounded_rate=move.r_dom_t1),
            EUR_CURVE: FlatZeroRateCurve(continuously_compounded_rate=move.r_for_t1),
        },
        vols={EURUSD_VOL: FlatVolSurface(sigma=move.vol_t1)},
    )


def build_portfolio() -> Portfolio:
    """
    Build a portfolio for P&L attribution.
    
    Returns
    -------
    Portfolio
        Sample portfolio.
    """
    positions = [
        Position(
            position_id="EURUSD_CALL_ATM",
            instrument=EuropeanFxVanillaOption(
                option_type="call",
                strike=1.08,
                expiry=0.5,
                notional=10_000_000,
                spot_id=EURUSD_SPOT,
                domestic_curve_id=USD_CURVE,
                foreign_curve_id=EUR_CURVE,
                vol_id=EURUSD_VOL,
            ),
            quantity=1,
        ),
        Position(
            position_id="EURUSD_PUT_OTM",
            instrument=EuropeanFxVanillaOption(
                option_type="put",
                strike=1.05,
                expiry=0.5,
                notional=5_000_000,
                spot_id=EURUSD_SPOT,
                domestic_curve_id=USD_CURVE,
                foreign_curve_id=EUR_CURVE,
                vol_id=EURUSD_VOL,
            ),
            quantity=1,
        ),
    ]
    
    logger.info("")
    logger.info("Portfolio:")
    for pos in positions:
        opt = pos.instrument
        logger.info(f"  {pos.position_id}: {opt.option_type.upper()} K={opt.strike}, T={opt.expiry}Y, N={opt.notional:,.0f}")
    
    return Portfolio(positions=positions)


def setup_pricer() -> Tuple[PricerRegistry, PortfolioPricer]:
    """Setup pricer infrastructure."""
    registry = PricerRegistry()
    registry.register(EuropeanFxVanillaOption, FxEuropeanVanillaBsmPricer())
    portfolio_pricer = PortfolioPricer(pricer_registry=registry)
    return registry, portfolio_pricer


# =============================================================================
# SECTION 2: P&L Attribution Calculation
# =============================================================================

def compute_greeks(
    portfolio: Portfolio,
    market: Market,
    portfolio_pricer: PortfolioPricer,
) -> Dict[str, float]:
    """
    Compute portfolio Greeks at T-1.
    
    Returns
    -------
    Dict[str, float]
        Portfolio Greeks.
    """
    result = portfolio_pricer.price(portfolio, market)
    greeks = result.totals.greeks
    
    return {
        "pv": result.totals.pv,
        "delta": greeks.get("delta", 0),
        "gamma": greeks.get("gamma", 0),
        "vega": greeks.get("vega", 0),
        "theta": greeks.get("theta", 0),
        "rho_domestic": greeks.get("rho_domestic", greeks.get("rho", 0)),
        "rho_foreign": greeks.get("rho_foreign", 0),
    }


def attribute_pnl(
    portfolio: Portfolio,
    market_t0: Market,
    market_t1: Market,
    move: MarketMove,
    portfolio_pricer: PortfolioPricer,
) -> AttributionResult:
    """
    Attribute P&L to risk factors using Greeks.
    
    Parameters
    ----------
    portfolio : Portfolio
        The portfolio.
    market_t0 : Market
        Yesterday's market.
    market_t1 : Market
        Today's market.
    move : MarketMove
        Market move data.
    portfolio_pricer : PortfolioPricer
        Portfolio pricer.
    
    Returns
    -------
    AttributionResult
        P&L attribution breakdown.
    
    Mathematical Details
    --------------------
    We use Taylor expansion around T-1 values:
    
        ΔV ≈ Δ·ΔS + ½Γ·(ΔS)² + ν·Δσ + Θ·Δt + ρ_d·Δr_d + ρ_f·Δr_f
    
    where Greeks are computed at T-1 market.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 2: P&L Attribution")
    logger.info("=" * 70)
    
    # Get T-1 and T portfolio values
    pv_t0 = portfolio_pricer.price(portfolio, market_t0).totals.pv
    pv_t1 = portfolio_pricer.price(portfolio, market_t1).totals.pv
    
    actual_pnl = pv_t1 - pv_t0
    
    # Get Greeks at T-1
    greeks = compute_greeks(portfolio, market_t0, portfolio_pricer)
    
    logger.info("")
    logger.info("Portfolio Values:")
    logger.info(f"  PV (T-1): ${pv_t0:,.2f}")
    logger.info(f"  PV (T):   ${pv_t1:,.2f}")
    logger.info(f"  Actual P&L: ${actual_pnl:,.2f}")
    
    logger.info("")
    logger.info("T-1 Greeks:")
    logger.info(f"  Delta (Δ):        {greeks['delta']:,.2f}")
    logger.info(f"  Gamma (Γ):        {greeks['gamma']:,.2f}")
    logger.info(f"  Vega (ν):         {greeks['vega']:,.2f}")
    logger.info(f"  Theta (Θ):        {greeks['theta']:,.2f}")
    logger.info(f"  Rho Domestic:     {greeks['rho_domestic']:,.2f}")
    logger.info(f"  Rho Foreign:      {greeks['rho_foreign']:,.2f}")
    
    # Compute attributed P&L
    delta_pnl = greeks["delta"] * move.delta_spot
    gamma_pnl = 0.5 * greeks["gamma"] * move.delta_spot ** 2
    vega_pnl = greeks["vega"] * move.delta_vol
    theta_pnl = greeks["theta"] * (1 / 252)  # 1 day
    rho_dom_pnl = greeks["rho_domestic"] * move.delta_r_dom
    rho_for_pnl = greeks["rho_foreign"] * move.delta_r_for
    
    explained = delta_pnl + gamma_pnl + vega_pnl + theta_pnl + rho_dom_pnl + rho_for_pnl
    residual = actual_pnl - explained
    
    logger.info("")
    logger.info("Market Moves:")
    logger.info(f"  ΔS (spot):        {move.delta_spot:+.4f} ({move.delta_spot/move.spot_t0*100:+.2f}%)")
    logger.info(f"  Δσ (vol):         {move.delta_vol*100:+.2f} vol points")
    logger.info(f"  Δr_dom:           {move.delta_r_dom*100:+.2f} bps")
    logger.info(f"  Δr_for:           {move.delta_r_for*100:+.2f} bps")
    
    return AttributionResult(
        actual_pnl=actual_pnl,
        delta_pnl=delta_pnl,
        gamma_pnl=gamma_pnl,
        vega_pnl=vega_pnl,
        theta_pnl=theta_pnl,
        rho_dom_pnl=rho_dom_pnl,
        rho_for_pnl=rho_for_pnl,
        residual=residual,
    )


# =============================================================================
# SECTION 3: Display Results
# =============================================================================

def display_attribution(result: AttributionResult) -> None:
    """Display P&L attribution breakdown."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 3: P&L Attribution Breakdown")
    logger.info("=" * 70)
    
    logger.info("")
    logger.info("P&L Attribution:")
    logger.info("-" * 50)
    logger.info(f"{'Factor':<20} {'P&L':>15} {'% of Total':>15}")
    logger.info("-" * 50)
    
    total = abs(result.actual_pnl) if abs(result.actual_pnl) > 1e-10 else 1
    
    logger.info(f"{'Delta':<20} ${result.delta_pnl:>14,.2f} {result.delta_pnl/total*100:>14.1f}%")
    logger.info(f"{'Gamma':<20} ${result.gamma_pnl:>14,.2f} {result.gamma_pnl/total*100:>14.1f}%")
    logger.info(f"{'Vega':<20} ${result.vega_pnl:>14,.2f} {result.vega_pnl/total*100:>14.1f}%")
    logger.info(f"{'Theta':<20} ${result.theta_pnl:>14,.2f} {result.theta_pnl/total*100:>14.1f}%")
    logger.info(f"{'Rho (Domestic)':<20} ${result.rho_dom_pnl:>14,.2f} {result.rho_dom_pnl/total*100:>14.1f}%")
    logger.info(f"{'Rho (Foreign)':<20} ${result.rho_for_pnl:>14,.2f} {result.rho_for_pnl/total*100:>14.1f}%")
    logger.info("-" * 50)
    logger.info(f"{'Explained':<20} ${result.explained_pnl:>14,.2f} {result.explained_pnl/total*100:>14.1f}%")
    logger.info(f"{'Residual':<20} ${result.residual:>14,.2f} {result.residual/total*100:>14.1f}%")
    logger.info("-" * 50)
    logger.info(f"{'ACTUAL P&L':<20} ${result.actual_pnl:>14,.2f} {'100.0':>14}%")
    
    logger.info("")
    logger.info("Quality Metrics:")
    logger.info(f"  Explain Ratio:    {result.explain_ratio:.1%}")
    logger.info(f"  Residual Ratio:   {abs(result.residual)/total*100:.1f}%")
    
    if abs(result.residual / total) > 0.10:
        logger.warning("  ⚠ Large residual (>10%) - investigate higher-order effects")
    else:
        logger.info("  ✓ Residual within acceptable bounds")


# =============================================================================
# SECTION 4: Multiple Scenarios
# =============================================================================

def run_scenario_attribution(
    portfolio: Portfolio,
    market_t0: Market,
    params: Dict[str, float],
    portfolio_pricer: PortfolioPricer,
) -> List[Tuple[str, MarketMove, AttributionResult]]:
    """
    Run attribution for multiple market scenarios.
    
    Returns
    -------
    List[Tuple[str, MarketMove, AttributionResult]]
        List of (scenario_name, move, attribution).
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 4: Multi-Scenario Attribution")
    logger.info("=" * 70)
    
    scenarios = [
        ("Spot +1%", MarketMove(
            spot_t0=params["spot"], spot_t1=params["spot"] * 1.01,
            vol_t0=params["vol"], vol_t1=params["vol"],
            r_dom_t0=params["r_dom"], r_dom_t1=params["r_dom"],
            r_for_t0=params["r_for"], r_for_t1=params["r_for"],
        )),
        ("Spot -1%", MarketMove(
            spot_t0=params["spot"], spot_t1=params["spot"] * 0.99,
            vol_t0=params["vol"], vol_t1=params["vol"],
            r_dom_t0=params["r_dom"], r_dom_t1=params["r_dom"],
            r_for_t0=params["r_for"], r_for_t1=params["r_for"],
        )),
        ("Vol +2pts", MarketMove(
            spot_t0=params["spot"], spot_t1=params["spot"],
            vol_t0=params["vol"], vol_t1=params["vol"] + 0.02,
            r_dom_t0=params["r_dom"], r_dom_t1=params["r_dom"],
            r_for_t0=params["r_for"], r_for_t1=params["r_for"],
        )),
        ("Spot +2%, Vol +1pt", MarketMove(
            spot_t0=params["spot"], spot_t1=params["spot"] * 1.02,
            vol_t0=params["vol"], vol_t1=params["vol"] + 0.01,
            r_dom_t0=params["r_dom"], r_dom_t1=params["r_dom"],
            r_for_t0=params["r_for"], r_for_t1=params["r_for"],
        )),
        ("Rates +25bp", MarketMove(
            spot_t0=params["spot"], spot_t1=params["spot"],
            vol_t0=params["vol"], vol_t1=params["vol"],
            r_dom_t0=params["r_dom"], r_dom_t1=params["r_dom"] + 0.0025,
            r_for_t0=params["r_for"], r_for_t1=params["r_for"] + 0.0025,
        )),
    ]
    
    results = []
    
    logger.info("")
    logger.info("Scenario Attribution Summary:")
    logger.info("-" * 90)
    logger.info(f"{'Scenario':<20} {'Actual':>12} {'Delta':>10} {'Gamma':>10} {'Vega':>10} {'Residual':>10} {'Explain':>10}")
    logger.info("-" * 90)
    
    for name, move in scenarios:
        market_t1 = create_market_t1(move)
        attr = attribute_pnl(portfolio, market_t0, market_t1, move, portfolio_pricer)
        results.append((name, move, attr))
        
        logger.info(
            f"{name:<20} ${attr.actual_pnl:>10,.0f} ${attr.delta_pnl:>8,.0f} "
            f"${attr.gamma_pnl:>8,.0f} ${attr.vega_pnl:>8,.0f} "
            f"${attr.residual:>8,.0f} {attr.explain_ratio:>9.1%}"
        )
    
    logger.info("-" * 90)
    
    return results


# =============================================================================
# SECTION 5: Visualization
# =============================================================================

def visualize_attribution(
    result: AttributionResult,
    scenario_results: List[Tuple[str, MarketMove, AttributionResult]],
) -> None:
    """Create attribution visualizations."""
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
    # Plot 1: P&L Attribution Waterfall
    # -------------------------------------------------------------------------
    ax = axes[0, 0]
    
    factors = ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho Dom', 'Rho For', 'Residual']
    values = [
        result.delta_pnl, result.gamma_pnl, result.vega_pnl,
        result.theta_pnl, result.rho_dom_pnl, result.rho_for_pnl, result.residual
    ]
    
    # Waterfall chart
    cumulative = 0
    for i, (factor, value) in enumerate(zip(factors, values)):
        color = '#10B981' if value >= 0 else '#E94F37'
        ax.bar(i, value, bottom=cumulative, color=color, alpha=0.8)
        cumulative += value
    
    # Add total bar
    ax.bar(len(factors), result.actual_pnl, color='#2E86AB', alpha=0.8)
    factors.append('Total')
    
    ax.set_xticks(range(len(factors)))
    ax.set_xticklabels(factors, rotation=45, ha='right')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel('P&L ($)')
    ax.set_title('P&L Attribution Waterfall')
    ax.grid(True, alpha=0.3, axis='y')
    
    # -------------------------------------------------------------------------
    # Plot 2: Attribution Pie Chart
    # -------------------------------------------------------------------------
    ax = axes[0, 1]
    
    abs_values = [abs(v) for v in values[:-1]]  # Exclude residual
    labels = factors[:-2]  # Exclude residual and total
    colors = ['#2E86AB', '#8B5CF6', '#F59E0B', '#6B7280', '#10B981', '#E94F37']
    
    if sum(abs_values) > 0:
        ax.pie(abs_values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Attribution by Factor (Absolute)')
    else:
        ax.text(0.5, 0.5, "No P&L to attribute", ha='center', va='center')
    
    # -------------------------------------------------------------------------
    # Plot 3: Scenario Comparison
    # -------------------------------------------------------------------------
    ax = axes[1, 0]
    
    scenario_names = [r[0] for r in scenario_results]
    actuals = [r[2].actual_pnl for r in scenario_results]
    explained = [r[2].explained_pnl for r in scenario_results]
    
    x = np.arange(len(scenario_names))
    width = 0.35
    
    ax.bar(x - width/2, actuals, width, label='Actual P&L', color='#2E86AB')
    ax.bar(x + width/2, explained, width, label='Explained P&L', color='#10B981')
    
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, rotation=45, ha='right')
    ax.set_ylabel('P&L ($)')
    ax.set_title('Actual vs Explained P&L by Scenario')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # -------------------------------------------------------------------------
    # Plot 4: Explain Ratio by Scenario
    # -------------------------------------------------------------------------
    ax = axes[1, 1]
    
    explain_ratios = [r[2].explain_ratio * 100 for r in scenario_results]
    
    colors = ['#10B981' if r >= 90 else '#F59E0B' if r >= 80 else '#E94F37' for r in explain_ratios]
    
    bars = ax.barh(scenario_names, explain_ratios, color=colors)
    ax.axvline(100, color='gray', linestyle='--', alpha=0.7)
    ax.axvline(90, color='#F59E0B', linestyle='--', alpha=0.5, label='90% threshold')
    
    ax.set_xlabel('Explain Ratio (%)')
    ax.set_title('Attribution Quality by Scenario')
    ax.set_xlim(0, 110)
    ax.grid(True, alpha=0.3, axis='x')
    
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
    │  1. P&L Attribution:                                                │
    │     - Decomposes P&L into risk factor contributions                 │
    │     - Uses Taylor expansion: ΔV ≈ Δ·ΔS + ½Γ·(ΔS)² + ν·Δσ + ...      │
    │                                                                      │
    │  2. Key Components:                                                 │
    │     - Delta P&L: First-order spot sensitivity                       │
    │     - Gamma P&L: Convexity (second-order) effect                    │
    │     - Vega P&L: Volatility move contribution                        │
    │     - Theta P&L: Time decay (always negative for long options)      │
    │                                                                      │
    │  3. Residual Analysis:                                              │
    │     - Residual = Actual - Explained                                 │
    │     - Large residual indicates higher-order effects or model error  │
    │     - Target: Explain ratio > 90%                                   │
    │                                                                      │
    │  4. Production Use:                                                 │
    │     - End-of-day P&L explain is mandatory                           │
    │     - Drives investigation into large moves                         │
    │     - Performance attribution by trader/strategy                    │
    │                                                                      │
    │  NEXT: See 04_delta_hedging.py for dynamic hedging                  │
    │                                                                      │
    └─────────────────────────────────────────────────────────────────────┘
    """
    logger.info(summary)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main(args: argparse.Namespace) -> None:
    """
    Main entry point.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    """
    global ENABLE_PLOTTING
    ENABLE_PLOTTING = args.plot
    
    try:
        # Section 1: Setup
        market_t0, params = create_market_t0()
        portfolio = build_portfolio()
        registry, portfolio_pricer = setup_pricer()
        
        # Define a sample market move
        move = MarketMove(
            spot_t0=params["spot"], spot_t1=params["spot"] * 1.015,  # +1.5%
            vol_t0=params["vol"], vol_t1=params["vol"] + 0.005,      # +0.5 vol pts
            r_dom_t0=params["r_dom"], r_dom_t1=params["r_dom"],
            r_for_t0=params["r_for"], r_for_t1=params["r_for"],
        )
        
        market_t1 = create_market_t1(move)
        
        # Section 2: P&L Attribution
        result = attribute_pnl(portfolio, market_t0, market_t1, move, portfolio_pricer)
        
        # Section 3: Display
        display_attribution(result)
        
        # Section 4: Multi-scenario
        scenario_results = run_scenario_attribution(
            portfolio, market_t0, params, portfolio_pricer
        )
        
        # Section 5: Visualization
        visualize_attribution(result, scenario_results)
        
        # Summary
        print_summary()
        
        logger.info("Example completed successfully!")
        
    except Exception as e:
        logger.exception(f"Example failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="P&L Attribution Example",
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

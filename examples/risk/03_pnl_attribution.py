#!/usr/bin/env python3
"""
===============================================================================
P&L Attribution: Explaining Daily P&L by Risk Factors
===============================================================================

This example demonstrates P&L attribution - decomposing portfolio P&L into
contributions from each risk factor (Greeks-based explain).

Learning Objectives
-------------------
1. **P&L Explain**: Decompose P&L into delta, gamma, vega, theta, and residual
2. **First-Order Attribution**: P&L ≈ Δ·ΔS + ν·Δσ + Θ·Δt
3. **Second-Order Effects**: Include gamma for large moves
4. **Residual Analysis**: Identify unexplained P&L (model error, higher order)

Mathematical Framework
----------------------
Taylor expansion of option value V(S, σ, t):

    ΔV ≈ ∂V/∂S · ΔS + ∂V/∂σ · Δσ + ∂V/∂t · Δt + ½ ∂²V/∂S² · (ΔS)²

In terms of Greeks:
    ΔV ≈ Δ·ΔS + ν·Δσ + Θ·Δt + ½Γ·(ΔS)²

Where:
    - Delta P&L:  Δ · ΔS
    - Vega P&L:   ν · Δσ  
    - Theta P&L:  Θ · Δt (typically negative for long options)
    - Gamma P&L:  ½Γ · (ΔS)²
    - Residual:   Actual P&L - Explained P&L (higher-order terms, model error)

Production Context
------------------
At a hedge fund:
- P&L explain is run daily to validate portfolio P&L
- Large unexplained P&L triggers investigation (model issues, data errors)
- Attribution drives hedging decisions (which risk factors dominate?)
- Risk management monitors breakdown to ensure limits aren't breached

Prerequisites
-------------
- Understanding of Greeks (examples/risk/02_sensitivities_computation.py)
- Portfolio pricing (examples/pricing/03_portfolio_pricing.py)

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

from src.instruments.fx.options.vanilla import FxVanillaEuropeanOption
from src.portfolio.core import Portfolio, Position
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

# Market IDs
EURUSD_SPOT = MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD")
USD_CURVE = MarketId(asset_class="IR", mkt_type="CURVE", name="USD_OIS")
EUR_CURVE = MarketId(asset_class="IR", mkt_type="CURVE", name="EUR_OIS")
EURUSD_VOL = MarketId(asset_class="FX", mkt_type="VOL", name="EURUSD")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class MarketMove:
    """Represents a market move from T to T+1."""
    spot_t0: float
    spot_t1: float
    vol_t0: float
    vol_t1: float
    dt: float  # Time elapsed (in years)
    
    @property
    def delta_spot(self) -> float:
        """Absolute spot change."""
        return self.spot_t1 - self.spot_t0
    
    @property
    def delta_vol(self) -> float:
        """Absolute vol change (in decimal)."""
        return self.vol_t1 - self.vol_t0
    
    @property
    def spot_return(self) -> float:
        """Spot return (percentage)."""
        return (self.spot_t1 - self.spot_t0) / self.spot_t0


@dataclass
class PnLAttribution:
    """P&L attribution breakdown."""
    delta_pnl: float
    gamma_pnl: float
    vega_pnl: float
    theta_pnl: float
    explained_pnl: float
    actual_pnl: float
    residual: float
    
    @property
    def explain_ratio(self) -> float:
        """Ratio of explained to actual P&L."""
        if abs(self.actual_pnl) < 1e-10:
            return 1.0
        return self.explained_pnl / self.actual_pnl


# =============================================================================
# SECTION 1: Market Setup
# =============================================================================

def create_markets(
    spot_t0: float,
    spot_t1: float,
    vol_t0: float,
    vol_t1: float,
    r_usd: float,
    r_eur: float,
) -> Tuple[Market, Market]:
    """
    Create T0 and T1 market snapshots.
    
    Returns
    -------
    Tuple[Market, Market]
        Markets at T0 and T1.
    """
    # T0 market
    market_t0 = Market(
        asof="2026-01-27",
        quotes={EURUSD_SPOT: Quote(value=spot_t0)},
        curves={
            USD_CURVE: FlatZeroRateCurve(continuously_compounded_rate=r_usd),
            EUR_CURVE: FlatZeroRateCurve(continuously_compounded_rate=r_eur),
        },
        vols={EURUSD_VOL: FlatVolSurface(sigma=vol_t0)},
    )
    
    # T1 market
    market_t1 = Market(
        asof="2026-01-28",
        quotes={EURUSD_SPOT: Quote(value=spot_t1)},
        curves={
            USD_CURVE: FlatZeroRateCurve(continuously_compounded_rate=r_usd),
            EUR_CURVE: FlatZeroRateCurve(continuously_compounded_rate=r_eur),
        },
        vols={EURUSD_VOL: FlatVolSurface(sigma=vol_t1)},
    )
    
    return market_t0, market_t1


# =============================================================================
# SECTION 2: Portfolio Construction
# =============================================================================

def build_portfolio() -> Portfolio:
    """
    Build a sample portfolio for P&L attribution.
    
    Returns
    -------
    Portfolio
        Portfolio with multiple positions.
    """
    positions = [
        # Long ATM call
        Position(
            position_id="LONG_CALL_ATM",
            instrument=FxVanillaEuropeanOption(
                option_type="call",
                strike=1.0850,
                expiry=0.25,  # 3 months
                notional=10_000_000,
                spot_id=EURUSD_SPOT,
                domestic_curve_id=USD_CURVE,
                foreign_curve_id=EUR_CURVE,
                vol_id=EURUSD_VOL,
            ),
            quantity=1.0,
        ),
        # Short OTM put
        Position(
            position_id="SHORT_PUT_OTM",
            instrument=FxVanillaEuropeanOption(
                option_type="put",
                strike=1.0500,
                expiry=0.25,
                notional=5_000_000,
                spot_id=EURUSD_SPOT,
                domestic_curve_id=USD_CURVE,
                foreign_curve_id=EUR_CURVE,
                vol_id=EURUSD_VOL,
            ),
            quantity=-1.0,
        ),
        # Long 1Y straddle
        Position(
            position_id="LONG_STRADDLE_CALL",
            instrument=FxVanillaEuropeanOption(
                option_type="call",
                strike=1.0850,
                expiry=1.0,
                notional=3_000_000,
                spot_id=EURUSD_SPOT,
                domestic_curve_id=USD_CURVE,
                foreign_curve_id=EUR_CURVE,
                vol_id=EURUSD_VOL,
            ),
            quantity=1.0,
        ),
        Position(
            position_id="LONG_STRADDLE_PUT",
            instrument=FxVanillaEuropeanOption(
                option_type="put",
                strike=1.0850,
                expiry=1.0,
                notional=3_000_000,
                spot_id=EURUSD_SPOT,
                domestic_curve_id=USD_CURVE,
                foreign_curve_id=EUR_CURVE,
                vol_id=EURUSD_VOL,
            ),
            quantity=1.0,
        ),
    ]
    
    return Portfolio(positions=positions)


# =============================================================================
# SECTION 3: P&L Attribution Engine
# =============================================================================

def compute_portfolio_greeks(
    portfolio: Portfolio,
    market: Market,
    pricer: FxVanillaEuropeanOptionBsmPricer,
) -> Dict[str, float]:
    """
    Compute aggregate portfolio Greeks.
    
    Returns
    -------
    Dict[str, float]
        Total delta, gamma, vega, theta.
    """
    total_delta = 0.0
    total_gamma = 0.0
    total_vega = 0.0
    total_theta = 0.0
    
    for pos in portfolio.positions:
        greeks = pricer.greeks(pos.instrument, market)
        total_delta += pos.quantity * greeks.get('delta', 0)
        total_gamma += pos.quantity * greeks.get('gamma', 0)
        total_vega += pos.quantity * greeks.get('vega', 0)
        total_theta += pos.quantity * greeks.get('theta', 0)
    
    return {
        'delta': total_delta,
        'gamma': total_gamma,
        'vega': total_vega,
        'theta': total_theta,
    }


def compute_portfolio_pv(
    portfolio: Portfolio,
    market: Market,
    pricer: FxVanillaEuropeanOptionBsmPricer,
) -> float:
    """Compute total portfolio PV."""
    total_pv = 0.0
    for pos in portfolio.positions:
        pv = pricer.price(pos.instrument, market)
        total_pv += pos.quantity * pv
    return total_pv


def compute_pnl_attribution(
    portfolio: Portfolio,
    market_t0: Market,
    market_t1: Market,
    market_move: MarketMove,
    pricer: FxVanillaEuropeanOptionBsmPricer,
) -> PnLAttribution:
    """
    Compute P&L attribution using Taylor expansion.
    
    Parameters
    ----------
    portfolio : Portfolio
        The portfolio.
    market_t0 : Market
        Market at T0.
    market_t1 : Market
        Market at T1.
    market_move : MarketMove
        Market move details.
    pricer : FxVanillaEuropeanOptionBsmPricer
        The pricer.
    
    Returns
    -------
    PnLAttribution
        P&L breakdown.
    """
    # Compute Greeks at T0
    greeks_t0 = compute_portfolio_greeks(portfolio, market_t0, pricer)
    
    # Compute actual P&L
    pv_t0 = compute_portfolio_pv(portfolio, market_t0, pricer)
    pv_t1 = compute_portfolio_pv(portfolio, market_t1, pricer)
    actual_pnl = pv_t1 - pv_t0
    
    # Taylor expansion attribution
    # Delta P&L: Δ × ΔS
    delta_pnl = greeks_t0['delta'] * market_move.delta_spot
    
    # Gamma P&L: ½Γ × (ΔS)²
    gamma_pnl = 0.5 * greeks_t0['gamma'] * (market_move.delta_spot ** 2)
    
    # Vega P&L: ν × Δσ (vega is per 1% vol move, so multiply by 100)
    vega_pnl = greeks_t0['vega'] * market_move.delta_vol * 100
    
    # Theta P&L: Θ × Δt (theta is daily decay)
    theta_pnl = greeks_t0['theta'] * market_move.dt * 365  # Convert to daily
    
    # Explained P&L
    explained_pnl = delta_pnl + gamma_pnl + vega_pnl + theta_pnl
    
    # Residual (unexplained)
    residual = actual_pnl - explained_pnl
    
    return PnLAttribution(
        delta_pnl=delta_pnl,
        gamma_pnl=gamma_pnl,
        vega_pnl=vega_pnl,
        theta_pnl=theta_pnl,
        explained_pnl=explained_pnl,
        actual_pnl=actual_pnl,
        residual=residual,
    )


# =============================================================================
# SECTION 4: Run Attribution
# =============================================================================

def run_pnl_attribution() -> Tuple[PnLAttribution, Dict[str, float]]:
    """
    Run P&L attribution analysis.
    
    Returns
    -------
    Tuple[PnLAttribution, Dict[str, float]]
        Attribution results and Greeks.
    """
    logger.info("=" * 70)
    logger.info("SECTION 1: P&L Attribution Setup")
    logger.info("=" * 70)
    
    # Market parameters
    spot_t0 = 1.0850
    spot_t1 = 1.0920  # Spot moved up 70 pips
    vol_t0 = 0.10
    vol_t1 = 0.105  # Vol increased 50 bps
    r_usd = 0.05
    r_eur = 0.04
    dt = 1 / 365  # 1 day
    
    market_move = MarketMove(
        spot_t0=spot_t0,
        spot_t1=spot_t1,
        vol_t0=vol_t0,
        vol_t1=vol_t1,
        dt=dt,
    )
    
    logger.info("")
    logger.info("Market Move (T0 → T1):")
    logger.info(f"  Spot:      {spot_t0:.4f} → {spot_t1:.4f} (Δ = {market_move.delta_spot:+.4f}, {market_move.spot_return*100:+.2f}%)")
    logger.info(f"  Vol:       {vol_t0*100:.1f}% → {vol_t1*100:.1f}% (Δ = {market_move.delta_vol*100:+.2f}%)")
    logger.info(f"  Time:      1 day")
    
    # Create markets
    market_t0, market_t1 = create_markets(
        spot_t0, spot_t1, vol_t0, vol_t1, r_usd, r_eur
    )
    
    # Build portfolio
    portfolio = build_portfolio()
    
    logger.info("")
    logger.info(f"Portfolio: {len(portfolio.positions)} positions")
    for pos in portfolio.positions:
        opt = pos.instrument
        logger.info(f"  {pos.position_id}: {opt.option_type} K={opt.strike:.4f} T={opt.expiry:.2f}y qty={pos.quantity:+.0f}")
    
    # Create pricer
    pricer = FxVanillaEuropeanOptionBsmPricer()
    
    # Compute Greeks at T0
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 2: Portfolio Greeks at T0")
    logger.info("=" * 70)
    
    greeks_t0 = compute_portfolio_greeks(portfolio, market_t0, pricer)
    
    logger.info("")
    logger.info("Aggregate Greeks:")
    logger.info(f"  Delta (Δ): {greeks_t0['delta']:>15,.2f}")
    logger.info(f"  Gamma (Γ): {greeks_t0['gamma']:>15,.2f}")
    logger.info(f"  Vega (ν):  {greeks_t0['vega']:>15,.2f}")
    logger.info(f"  Theta (Θ): {greeks_t0['theta']:>15,.2f}")
    
    # Compute attribution
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 3: P&L Attribution")
    logger.info("=" * 70)
    
    attribution = compute_pnl_attribution(
        portfolio, market_t0, market_t1, market_move, pricer
    )
    
    pv_t0 = compute_portfolio_pv(portfolio, market_t0, pricer)
    pv_t1 = compute_portfolio_pv(portfolio, market_t1, pricer)
    
    logger.info("")
    logger.info("Portfolio Value:")
    logger.info(f"  PV (T0):        ${pv_t0:>15,.2f}")
    logger.info(f"  PV (T1):        ${pv_t1:>15,.2f}")
    logger.info(f"  Actual P&L:     ${attribution.actual_pnl:>15,.2f}")
    
    logger.info("")
    logger.info("P&L Decomposition:")
    logger.info("-" * 50)
    logger.info(f"  Delta P&L:      ${attribution.delta_pnl:>15,.2f}  (Δ × ΔS)")
    logger.info(f"  Gamma P&L:      ${attribution.gamma_pnl:>15,.2f}  (½Γ × ΔS²)")
    logger.info(f"  Vega P&L:       ${attribution.vega_pnl:>15,.2f}  (ν × Δσ)")
    logger.info(f"  Theta P&L:      ${attribution.theta_pnl:>15,.2f}  (Θ × Δt)")
    logger.info("-" * 50)
    logger.info(f"  Explained P&L:  ${attribution.explained_pnl:>15,.2f}")
    logger.info(f"  Actual P&L:     ${attribution.actual_pnl:>15,.2f}")
    logger.info(f"  Residual:       ${attribution.residual:>15,.2f}")
    logger.info("")
    logger.info(f"  Explain Ratio:  {attribution.explain_ratio*100:>14.1f}%")
    
    return attribution, greeks_t0


# =============================================================================
# SECTION 5: Multi-Day Attribution
# =============================================================================

def run_multiday_attribution() -> List[PnLAttribution]:
    """
    Run attribution over multiple days.
    
    Returns
    -------
    List[PnLAttribution]
        Attribution for each day.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 4: Multi-Day P&L Attribution")
    logger.info("=" * 70)
    
    # Simulate 5 days of market moves
    np.random.seed(42)
    
    spot_t0 = 1.0850
    vol_t0 = 0.10
    r_usd = 0.05
    r_eur = 0.04
    
    # Daily returns and vol changes
    daily_returns = np.random.normal(0, 0.007, 5)  # ~70bps daily vol
    vol_changes = np.random.normal(0, 0.003, 5)  # Vol changes
    
    portfolio = build_portfolio()
    pricer = FxVanillaEuropeanOptionBsmPricer()
    
    attributions: List[PnLAttribution] = []
    
    logger.info("")
    logger.info(f"{'Day':<6} {'ΔSpot':>10} {'ΔVol':>10} {'Actual':>12} {'Explained':>12} {'Residual':>12}")
    logger.info("-" * 72)
    
    for day in range(5):
        spot_t1 = spot_t0 * (1 + daily_returns[day])
        vol_t1 = vol_t0 + vol_changes[day]
        vol_t1 = max(0.05, min(0.30, vol_t1))  # Bound vol
        
        market_move = MarketMove(
            spot_t0=spot_t0,
            spot_t1=spot_t1,
            vol_t0=vol_t0,
            vol_t1=vol_t1,
            dt=1/365,
        )
        
        market_t0, market_t1 = create_markets(
            spot_t0, spot_t1, vol_t0, vol_t1, r_usd, r_eur
        )
        
        attr = compute_pnl_attribution(
            portfolio, market_t0, market_t1, market_move, pricer
        )
        attributions.append(attr)
        
        logger.info(
            f"Day {day+1:<3} {market_move.delta_spot*10000:>+8.1f}bp {vol_changes[day]*100:>+8.2f}% "
            f"${attr.actual_pnl:>10,.0f} ${attr.explained_pnl:>10,.0f} ${attr.residual:>10,.0f}"
        )
        
        # Roll forward
        spot_t0 = spot_t1
        vol_t0 = vol_t1
    
    # Summary
    total_actual = sum(a.actual_pnl for a in attributions)
    total_explained = sum(a.explained_pnl for a in attributions)
    total_residual = sum(a.residual for a in attributions)
    
    logger.info("-" * 72)
    logger.info(
        f"{'TOTAL':<6} {'':<10} {'':<10} "
        f"${total_actual:>10,.0f} ${total_explained:>10,.0f} ${total_residual:>10,.0f}"
    )
    
    return attributions


# =============================================================================
# SECTION 6: Visualization
# =============================================================================

def visualize_attribution(
    attribution: PnLAttribution,
    multiday: List[PnLAttribution],
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
    # Plot 1: Single-day attribution breakdown
    # -------------------------------------------------------------------------
    ax = axes[0, 0]
    components = ['Delta', 'Gamma', 'Vega', 'Theta', 'Residual']
    values = [
        attribution.delta_pnl,
        attribution.gamma_pnl,
        attribution.vega_pnl,
        attribution.theta_pnl,
        attribution.residual,
    ]
    colors = ['#2E86AB' if v >= 0 else '#E94F37' for v in values]
    
    bars = ax.barh(components, values, color=colors)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('P&L ($)')
    ax.set_title('Single-Day P&L Attribution')
    ax.grid(True, alpha=0.3, axis='x')
    
    for bar, val in zip(bars, values):
        ax.text(
            val + (500 if val >= 0 else -500),
            bar.get_y() + bar.get_height() / 2,
            f'${val:,.0f}',
            ha='left' if val >= 0 else 'right',
            va='center',
            fontsize=9,
        )
    
    # -------------------------------------------------------------------------
    # Plot 2: Multi-day cumulative P&L
    # -------------------------------------------------------------------------
    ax = axes[0, 1]
    days = range(1, len(multiday) + 1)
    cum_actual = np.cumsum([a.actual_pnl for a in multiday])
    cum_explained = np.cumsum([a.explained_pnl for a in multiday])
    
    ax.plot(days, cum_actual / 1000, 'b-', linewidth=2, label='Actual')
    ax.plot(days, cum_explained / 1000, 'g--', linewidth=2, label='Explained')
    ax.fill_between(days, cum_actual / 1000, cum_explained / 1000, alpha=0.3, color='red', label='Residual')
    ax.set_xlabel('Day')
    ax.set_ylabel('Cumulative P&L ($000s)')
    ax.set_title('Multi-Day P&L: Actual vs Explained')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 3: Attribution breakdown over time
    # -------------------------------------------------------------------------
    ax = axes[1, 0]
    delta_pnls = [a.delta_pnl for a in multiday]
    gamma_pnls = [a.gamma_pnl for a in multiday]
    vega_pnls = [a.vega_pnl for a in multiday]
    theta_pnls = [a.theta_pnl for a in multiday]
    
    width = 0.2
    x = np.arange(len(multiday))
    
    ax.bar(x - 1.5*width, delta_pnls, width, label='Delta', color='#2E86AB')
    ax.bar(x - 0.5*width, gamma_pnls, width, label='Gamma', color='#10B981')
    ax.bar(x + 0.5*width, vega_pnls, width, label='Vega', color='#8B5CF6')
    ax.bar(x + 1.5*width, theta_pnls, width, label='Theta', color='#F59E0B')
    
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Day')
    ax.set_ylabel('P&L ($)')
    ax.set_title('Daily P&L by Component')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Day {i+1}' for i in range(len(multiday))])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # -------------------------------------------------------------------------
    # Plot 4: Explain ratio
    # -------------------------------------------------------------------------
    ax = axes[1, 1]
    explain_ratios = [a.explain_ratio * 100 for a in multiday]
    
    bars = ax.bar(days, explain_ratios, color=['#10B981' if r > 90 else '#F59E0B' if r > 80 else '#E94F37' for r in explain_ratios])
    ax.axhline(100, color='black', linestyle='--', alpha=0.5)
    ax.axhline(90, color='green', linestyle=':', alpha=0.5, label='90% threshold')
    ax.set_xlabel('Day')
    ax.set_ylabel('Explain Ratio (%)')
    ax.set_title('Daily Explain Ratio (>90% is good)')
    ax.set_ylim(0, 120)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
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
    │  1. P&L Attribution Formula:                                        │
    │     ΔV ≈ Δ·ΔS + ½Γ·(ΔS)² + ν·Δσ + Θ·Δt                            │
    │                                                                      │
    │  2. Component Interpretation:                                       │
    │     - Delta P&L: Directional exposure to spot                       │
    │     - Gamma P&L: Convexity benefit/cost from large moves            │
    │     - Vega P&L: Vol exposure                                        │
    │     - Theta P&L: Time decay (usually negative for long options)     │
    │                                                                      │
    │  3. Residual Analysis:                                              │
    │     - Small residual = model explains P&L well                      │
    │     - Large residual = investigate! (model error, data issue)       │
    │     - Target: >90% explain ratio                                    │
    │                                                                      │
    │  4. Production Use:                                                 │
    │     - Daily P&L validation                                          │
    │     - Risk factor contribution monitoring                           │
    │     - Hedging effectiveness analysis                                │
    │                                                                      │
    │  NEXT: See 04_delta_hedging.py for hedging workflow                 │
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
        # Single-day attribution
        attribution, greeks = run_pnl_attribution()
        
        # Multi-day attribution
        multiday = run_multiday_attribution()
        
        # Visualization
        visualize_attribution(attribution, multiday)
        
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

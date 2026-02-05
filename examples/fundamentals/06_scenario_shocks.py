#!/usr/bin/env python3
"""
===============================================================================
Scenario Shocks: Applying Market Perturbations
===============================================================================

This example covers how to apply shocks to market data for scenario analysis
and stress testing - a core capability for risk management.

Learning Objectives
-------------------
1. **SpotShock**: Perturb spot prices (relative or absolute)
2. **VolShock**: Bump entire volatility surfaces
3. **ParallelRateShock**: Shift yield curves in parallel
4. **Combining Shocks**: Build complex stress scenarios

Mathematical Framework
----------------------
Shocks transform a base Market into a shocked Market:

    SpotShock (relative): S_shocked = S_base × (1 + bump)
    SpotShock (absolute): S_shocked = S_base + bump
    
    VolShock (absolute):  σ_shocked(K,T) = σ_base(K,T) + bump
    VolShock (relative):  σ_shocked(K,T) = σ_base(K,T) × (1 + bump)
    
    ParallelRateShock:    df_shocked(t) = df_base(t) × exp(-Δr × t)

Production Context
------------------
At a hedge fund, scenario shocks are used for:
- Greeks computation via finite differences (bump-and-reprice)
- Regulatory stress testing (CCAR, FRTB)
- Custom stress scenarios (market crash, vol spike)
- Sensitivity ladders for risk limits

Prerequisites
-------------
- Example 01-05: Market fundamentals and datasets

Run This Example
----------------
    cd /path/to/QuantStrata
    PYTHONPATH=. python examples/fundamentals/06_scenario_shocks.py

Author: QuantStrata Team
===============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

from __future__ import annotations  # Enable modern type hints (PEP 604)

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
from src.marketdata.curves.term_structure import ZeroRateCurve
from src.marketdata.surfaces.vol_surface import GridVolSurface
from src.marketdata.scenarios.shocks import (
    SpotShock,
    VolShock,
    ParallelRateShock,
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
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available - plotting disabled")


# =============================================================================
# CONSTANTS
# =============================================================================

EURUSD_SPOT = MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD")
USD_CURVE = MarketId(asset_class="IR", mkt_type="CURVE", name="USD_OIS")
EUR_CURVE = MarketId(asset_class="IR", mkt_type="CURVE", name="EUR_OIS")
EURUSD_VOL = MarketId(asset_class="FX", mkt_type="VOL", name="EURUSD")


# =============================================================================
# HELPER: Create Base Market
# =============================================================================

def create_base_market() -> Market:
    """
    Create a complete base market for shock demonstrations.
    
    Returns
    -------
    Market
        A market with spot, curves, and vol surface.
    
    This market represents typical EUR/USD trading conditions:
    - Spot at 1.0850
    - USD rates higher than EUR rates (carry differential)
    - Vol surface with typical FX smile/skew
    """
    # -------------------------------------------------------------------------
    # Create interest rate curves
    # -------------------------------------------------------------------------
    tenors = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
    usd_rates = np.array([0.050, 0.051, 0.052, 0.053, 0.055, 0.057])  # Higher USD
    eur_rates = np.array([0.035, 0.036, 0.038, 0.040, 0.042, 0.044])  # Lower EUR
    
    usd_curve = ZeroRateCurve(tenors=tenors, zero_rates=usd_rates)
    eur_curve = ZeroRateCurve(tenors=tenors, zero_rates=eur_rates)
    
    # -------------------------------------------------------------------------
    # Create vol surface with smile
    # -------------------------------------------------------------------------
    vol_expiries = np.array([0.25, 0.5, 1.0, 2.0])
    vol_strikes = np.linspace(0.95, 1.20, 11)
    base_vol = 0.08  # 8% ATM vol
    
    vol_grid = np.zeros((len(vol_expiries), len(vol_strikes)))
    for i, exp in enumerate(vol_expiries):
        for j, k in enumerate(vol_strikes):
            # Simple smile model: skew + convexity
            moneyness = np.log(k / 1.085)
            vol_grid[i, j] = base_vol - 0.15 * moneyness + 0.10 * moneyness**2
    
    eurusd_vol = GridVolSurface(
        expiries=vol_expiries,
        strikes=vol_strikes,
        implied_vols=vol_grid,
    )
    
    # -------------------------------------------------------------------------
    # Assemble market
    # -------------------------------------------------------------------------
    return Market(
        asof="2026-01-28",
        quotes={EURUSD_SPOT: Quote(value=1.0850)},
        curves={USD_CURVE: usd_curve, EUR_CURVE: eur_curve},
        vols={EURUSD_VOL: eurusd_vol},
    )


# =============================================================================
# SETUP: Display Base Market
# =============================================================================

def display_base_market(market: Market) -> None:
    """
    Display the base market state.
    
    Parameters
    ----------
    market : Market
        The market to display.
    """
    logger.info("=" * 70)
    logger.info("SETUP: Creating Base Market")
    logger.info("=" * 70)
    
    logger.info("")
    logger.info("Base market created:")
    logger.info(f"  As-of: {market.asof}")
    logger.info(f"  EUR/USD spot: {market.quote(EURUSD_SPOT):.4f}")
    logger.info(f"  USD 1Y rate: {market.curve(USD_CURVE).zero_rate(1.0):.4%}")
    logger.info(f"  EUR 1Y rate: {market.curve(EUR_CURVE).zero_rate(1.0):.4%}")
    logger.info(f"  ATM 1Y vol: {market.vol_surface(EURUSD_VOL).implied_vol(1.0, 1.085):.2%}")


# =============================================================================
# SECTION 1: SpotShock
# =============================================================================

def demonstrate_spot_shock(base_market: Market) -> tuple[Market, Market]:
    """
    Demonstrate SpotShock for perturbing spot prices.
    
    Parameters
    ----------
    base_market : Market
        The market to shock.
    
    Returns
    -------
    tuple[Market, Market]
        Spot-up and spot-down markets.
    
    SpotShock Modes
    ---------------
    relative: new_spot = old_spot × (1 + bump)
              +1% bump → 1.0850 × 1.01 = 1.0959
              
    absolute: new_spot = old_spot + bump
              +0.01 bump → 1.0850 + 0.01 = 1.0950
    
    Production Notes
    ----------------
    - Relative bumps are standard for spot delta
    - Absolute bumps useful for fixed pip moves
    - Other market data remains unchanged
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 1: SpotShock - Perturbing Spot Prices")
    logger.info("=" * 70)
    
    explanation = """
    SpotShock applies a bump to a spot price quote.
    
    Bump modes:
      relative: new_spot = old_spot × (1 + bump)   [+1% → ×1.01]
      absolute: new_spot = old_spot + bump         [+0.01 → +0.01]
    """
    logger.info(explanation)
    
    # -------------------------------------------------------------------------
    # Create spot shocks
    # -------------------------------------------------------------------------
    spot_up = SpotShock(
        name="spot_up_1pct",          # Descriptive name for logging
        spot_id=EURUSD_SPOT,          # Which spot to shock
        bump=0.01,                    # +1% relative bump
        bump_mode="relative",         # Multiplicative
    )
    
    spot_down = SpotShock(
        name="spot_down_1pct",
        spot_id=EURUSD_SPOT,
        bump=-0.01,                   # -1% relative bump
        bump_mode="relative",
    )
    
    # -------------------------------------------------------------------------
    # Apply shocks
    # shock.apply(market) returns a new Market (immutable pattern)
    # -------------------------------------------------------------------------
    market_spot_up = spot_up.apply(base_market)
    market_spot_down = spot_down.apply(base_market)
    
    logger.info("Spot shock results:")
    logger.info(f"  Base spot:  {base_market.quote(EURUSD_SPOT):.4f}")
    logger.info(f"  Spot +1%:   {market_spot_up.quote(EURUSD_SPOT):.4f}")
    logger.info(f"  Spot -1%:   {market_spot_down.quote(EURUSD_SPOT):.4f}")
    
    # -------------------------------------------------------------------------
    # Verify other market data unchanged
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("Other market data remains unchanged:")
    logger.info(f"  USD curve (spot up): {market_spot_up.curve(USD_CURVE).zero_rate(1.0):.4%}")
    logger.info(f"  Vol surface (spot up): {market_spot_up.vol_surface(EURUSD_VOL).implied_vol(1.0, 1.085):.2%}")
    
    return market_spot_up, market_spot_down


# =============================================================================
# SECTION 2: VolShock
# =============================================================================

def demonstrate_vol_shock(base_market: Market) -> tuple[Market, Market]:
    """
    Demonstrate VolShock for bumping volatility surfaces.
    
    Parameters
    ----------
    base_market : Market
        The market to shock.
    
    Returns
    -------
    tuple[Market, Market]
        Vol-up and vol-down markets.
    
    VolShock Modes
    --------------
    absolute: new_vol = old_vol + bump
              +1pt bump → 8% + 1% = 9%
              
    relative: new_vol = old_vol × (1 + bump)
              +10% bump → 8% × 1.10 = 8.8%
    
    Production Notes
    ----------------
    - Absolute bumps are standard for vega computation
    - Shock applies uniformly across all strikes and expiries
    - For localized bumps, use more specific shock types
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 2: VolShock - Perturbing Volatility Surfaces")
    logger.info("=" * 70)
    
    explanation = """
    VolShock applies a bump to an entire volatility surface.
    
    Bump modes:
      absolute: new_vol = old_vol + bump   [+1pt → 8% + 1% = 9%]
      relative: new_vol = old_vol × (1 + bump)
    """
    logger.info(explanation)
    
    # -------------------------------------------------------------------------
    # Create vol shocks
    # -------------------------------------------------------------------------
    vol_up = VolShock(
        name="vol_up_1pt",
        vol_id=EURUSD_VOL,
        bump=0.01,                    # +1 vol point (absolute)
        bump_mode="absolute",
    )
    
    vol_down = VolShock(
        name="vol_down_1pt",
        vol_id=EURUSD_VOL,
        bump=-0.01,                   # -1 vol point
        bump_mode="absolute",
    )
    
    # -------------------------------------------------------------------------
    # Apply shocks
    # -------------------------------------------------------------------------
    market_vol_up = vol_up.apply(base_market)
    market_vol_down = vol_down.apply(base_market)
    
    logger.info("Vol shock results (ATM 1Y):")
    logger.info(f"  Base vol:  {base_market.vol_surface(EURUSD_VOL).implied_vol(1.0, 1.085):.2%}")
    logger.info(f"  Vol +1pt:  {market_vol_up.vol_surface(EURUSD_VOL).implied_vol(1.0, 1.085):.2%}")
    logger.info(f"  Vol -1pt:  {market_vol_down.vol_surface(EURUSD_VOL).implied_vol(1.0, 1.085):.2%}")
    
    # -------------------------------------------------------------------------
    # Vol shock affects all strikes uniformly
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("Vol shock affects all strikes (1Y expiry):")
    logger.info(f"{'Strike':<10} {'Base':<10} {'Vol Up':<10} {'Vol Down':<10}")
    logger.info("-" * 40)
    
    for k in [1.00, 1.05, 1.085, 1.12, 1.18]:
        base_v = base_market.vol_surface(EURUSD_VOL).implied_vol(1.0, k)
        up_v = market_vol_up.vol_surface(EURUSD_VOL).implied_vol(1.0, k)
        down_v = market_vol_down.vol_surface(EURUSD_VOL).implied_vol(1.0, k)
        logger.info(f"{k:<10.3f} {base_v:<10.2%} {up_v:<10.2%} {down_v:<10.2%}")
    
    return market_vol_up, market_vol_down


# =============================================================================
# SECTION 3: ParallelRateShock
# =============================================================================

def demonstrate_rate_shock(base_market: Market) -> tuple[Market, Market]:
    """
    Demonstrate ParallelRateShock for shifting yield curves.
    
    Parameters
    ----------
    base_market : Market
        The market to shock.
    
    Returns
    -------
    tuple[Market, Market]
        Rate-up and rate-down markets.
    
    ParallelRateShock Mathematics
    -----------------------------
    The shock is applied to discount factors:
    
        df_shocked(t) = df_base(t) × exp(-Δr × t)
    
    This is equivalent to adding Δr to all zero rates.
    
    Example: +100bp shift
        r_base(5Y) = 5.5% → r_shocked(5Y) = 6.5%
        df_shocked(5) = df_base(5) × exp(-0.01 × 5)
    
    Production Notes
    ----------------
    - Parallel shifts are first-order approximation
    - Key rate duration (KRD) shocks are more granular
    - PCA-based shocks (level/slope/curve) capture real dynamics
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 3: ParallelRateShock - Shifting Yield Curves")
    logger.info("=" * 70)
    
    explanation = """
    ParallelRateShock applies a parallel shift to a yield curve.
    
    Mathematics: df_shocked(t) = df_base(t) × exp(-Δr × t)
    
    Example: rate_shift=0.01 → +100bp parallel shift
    """
    logger.info(explanation)
    
    # -------------------------------------------------------------------------
    # Create rate shocks
    # -------------------------------------------------------------------------
    usd_up = ParallelRateShock(
        name="usd_up_100bp",
        curve_id=USD_CURVE,
        rate_shift=0.01,              # +100bp (1%)
    )
    
    usd_down = ParallelRateShock(
        name="usd_down_50bp",
        curve_id=USD_CURVE,
        rate_shift=-0.005,            # -50bp
    )
    
    # -------------------------------------------------------------------------
    # Apply shocks
    # -------------------------------------------------------------------------
    market_rate_up = usd_up.apply(base_market)
    market_rate_down = usd_down.apply(base_market)
    
    logger.info("Rate shock results (USD OIS):")
    logger.info(f"{'Tenor':<10} {'Base':<12} {'Up 100bp':<12} {'Down 50bp':<12}")
    logger.info("-" * 46)
    
    for t in [0.25, 1.0, 5.0, 10.0]:
        base_r = base_market.curve(USD_CURVE).zero_rate(t)
        up_r = market_rate_up.curve(USD_CURVE).zero_rate(t)
        down_r = market_rate_down.curve(USD_CURVE).zero_rate(t)
        logger.info(f"{t:<10.2f} {base_r:<12.4%} {up_r:<12.4%} {down_r:<12.4%}")
    
    return market_rate_up, market_rate_down


# =============================================================================
# SECTION 4: Combining Multiple Shocks
# =============================================================================

def demonstrate_combined_shocks(base_market: Market) -> Market:
    """
    Demonstrate combining multiple shocks for stress scenarios.
    
    Parameters
    ----------
    base_market : Market
        The market to shock.
    
    Returns
    -------
    Market
        The stressed market with all shocks applied.
    
    Stress Scenario Design
    ----------------------
    Real market stress events typically involve:
    - Equity down / vol up (leverage effect)
    - FX down (risk-off → USD strength)
    - Rates down (flight to quality)
    
    Shocks are applied sequentially:
        market = shock1.apply(base)
        market = shock2.apply(market)
        market = shock3.apply(market)
    
    Production Notes
    ----------------
    - Order of application doesn't matter for independent shocks
    - Consider correlated shocks for realism
    - Document stress scenario rationale
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 4: Combining Multiple Shocks")
    logger.info("=" * 70)
    
    explanation = """
    Shocks can be combined by applying them sequentially.
    The result is a market with all shocks applied.
    
    Example stress scenario (risk-off event):
      - Spot: -5% (EUR weakens)
      - Vol: +3 points (volatility spike)
      - USD rates: -100bp (flight to quality)
    """
    logger.info(explanation)
    
    # -------------------------------------------------------------------------
    # Define stress scenario shocks
    # -------------------------------------------------------------------------
    stress_spot = SpotShock(
        name="stress_spot",
        spot_id=EURUSD_SPOT,
        bump=-0.05,                   # -5% spot
        bump_mode="relative",
    )
    
    stress_vol = VolShock(
        name="stress_vol",
        vol_id=EURUSD_VOL,
        bump=0.03,                    # +3 vol points
        bump_mode="absolute",
    )
    
    stress_rate = ParallelRateShock(
        name="stress_rate",
        curve_id=USD_CURVE,
        rate_shift=-0.01,             # -100bp
    )
    
    # -------------------------------------------------------------------------
    # Apply sequentially
    # -------------------------------------------------------------------------
    stressed_market = stress_spot.apply(base_market)
    stressed_market = stress_vol.apply(stressed_market)
    stressed_market = stress_rate.apply(stressed_market)
    
    logger.info("Combined stress scenario:")
    logger.info("  Shocks: Spot -5%, Vol +3pt, USD rates -100bp")
    logger.info("")
    logger.info(f"{'Metric':<20} {'Base':<15} {'Stressed':<15} {'Change':<15}")
    logger.info("-" * 65)
    
    # Spot
    base_spot = base_market.quote(EURUSD_SPOT)
    stress_spot_val = stressed_market.quote(EURUSD_SPOT)
    logger.info(f"{'EUR/USD Spot':<20} {base_spot:<15.4f} {stress_spot_val:<15.4f} {(stress_spot_val/base_spot-1)*100:+.1f}%")
    
    # Vol
    base_vol = base_market.vol_surface(EURUSD_VOL).implied_vol(1.0, 1.085)
    stress_vol_val = stressed_market.vol_surface(EURUSD_VOL).implied_vol(1.0, 1.085)
    logger.info(f"{'ATM 1Y Vol':<20} {base_vol:<15.2%} {stress_vol_val:<15.2%} {(stress_vol_val-base_vol)*100:+.1f}pt")
    
    # Rate
    base_rate = base_market.curve(USD_CURVE).zero_rate(1.0)
    stress_rate_val = stressed_market.curve(USD_CURVE).zero_rate(1.0)
    logger.info(f"{'USD 1Y Rate':<20} {base_rate:<15.4%} {stress_rate_val:<15.4%} {(stress_rate_val-base_rate)*10000:+.0f}bp")
    
    return stressed_market


# =============================================================================
# SECTION 5: Creating Scenario Ladders
# =============================================================================

def demonstrate_scenario_ladders(base_market: Market) -> None:
    """
    Demonstrate scenario ladders for sensitivity analysis.
    
    Parameters
    ----------
    base_market : Market
        The market to shock.
    
    Scenario Ladders
    ----------------
    A series of bumps used to:
    - Compute Greeks via finite differences
    - Build risk factor sensitivity profiles
    - Analyze P&L convexity
    
    Example spot ladder: [-5%, -2%, -1%, 0%, +1%, +2%, +5%]
    
    Production Notes
    ----------------
    - Central difference: delta ≈ (V+ - V-) / (2 × bump)
    - Gamma from second derivative
    - Store scenarios for audit trail
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 5: Creating Scenario Ladders")
    logger.info("=" * 70)
    
    explanation = """
    Scenario ladders are used to compute sensitivities:
      - Apply a series of bumps (e.g., -5%, -2%, 0%, +2%, +5%)
      - Price under each scenario
      - Analyze the P&L profile
    
    Used for: Greeks computation, sensitivity limits, convexity analysis
    """
    logger.info(explanation)
    
    # -------------------------------------------------------------------------
    # Spot ladder
    # -------------------------------------------------------------------------
    spot_bumps = [-0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05]
    spot_ladder: List[Tuple[float, float]] = []
    base_spot = base_market.quote(EURUSD_SPOT)
    
    logger.info("")
    logger.info("Spot ladder:")
    logger.info(f"{'Bump':<10} {'Spot':<12} {'Change':<12}")
    logger.info("-" * 34)
    
    for bump in spot_bumps:
        if bump == 0.0:
            market = base_market
        else:
            shock = SpotShock(
                name=f"spot_{bump:+.0%}",
                spot_id=EURUSD_SPOT,
                bump=bump,
                bump_mode="relative",
            )
            market = shock.apply(base_market)
        
        spot = market.quote(EURUSD_SPOT)
        spot_ladder.append((bump, spot))
        logger.info(f"{bump:+.0%}       {spot:<12.4f} {(spot/base_spot-1)*100:+.2f}%")
    
    # -------------------------------------------------------------------------
    # Vol ladder
    # -------------------------------------------------------------------------
    vol_bumps = [-0.03, -0.01, 0.0, 0.01, 0.03]
    vol_ladder: List[Tuple[float, float]] = []
    base_vol = base_market.vol_surface(EURUSD_VOL).implied_vol(1.0, 1.085)
    
    logger.info("")
    logger.info("Vol ladder (ATM 1Y):")
    logger.info(f"{'Bump':<10} {'Vol':<12} {'Change':<12}")
    logger.info("-" * 34)
    
    for bump in vol_bumps:
        if bump == 0.0:
            market = base_market
        else:
            shock = VolShock(
                name=f"vol_{bump*100:+.0f}pt",
                vol_id=EURUSD_VOL,
                bump=bump,
                bump_mode="absolute",
            )
            market = shock.apply(base_market)
        
        vol = market.vol_surface(EURUSD_VOL).implied_vol(1.0, 1.085)
        vol_ladder.append((bump, vol))
        logger.info(f"{bump*100:+.0f}pt      {vol:<12.2%} {(vol-base_vol)*100:+.1f}pt")


# =============================================================================
# SECTION 6: Visualization
# =============================================================================

def visualize_shocks(
    base_market: Market,
    market_rate_up: Market,
    market_rate_down: Market,
    market_vol_up: Market,
    market_vol_down: Market,
) -> None:
    """
    Create visualizations of shock effects.
    
    Parameters
    ----------
    base_market : Market
        Base market.
    market_rate_up, market_rate_down : Market
        Rate-shocked markets.
    market_vol_up, market_vol_down : Market
        Vol-shocked markets.
    """
    if not MATPLOTLIB_AVAILABLE or not ENABLE_PLOTTING:
        logger.info("Skipping plots (matplotlib not available or disabled)")
        return
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 6: Visualization")
    logger.info("=" * 70)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    base_spot = base_market.quote(EURUSD_SPOT)
    
    # -------------------------------------------------------------------------
    # Plot 1: Spot scenarios (bar chart)
    # -------------------------------------------------------------------------
    ax = axes[0]
    spot_bumps = [-0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05]
    spots = [base_spot * (1 + b) for b in spot_bumps]
    colors = ['#E94F37' if b < 0 else '#10B981' if b > 0 else '#2E86AB' for b in spot_bumps]
    
    ax.bar(range(len(spot_bumps)), spots, color=colors)
    ax.axhline(base_spot, color='gray', linestyle='--', alpha=0.7, label=f'Base: {base_spot:.4f}')
    ax.set_xticks(range(len(spot_bumps)))
    ax.set_xticklabels([f'{b*100:+.0f}%' for b in spot_bumps])
    ax.set_xlabel('Spot Shock')
    ax.set_ylabel('EUR/USD')
    ax.set_title('Spot Scenario Ladder')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 2: Rate curve comparison
    # -------------------------------------------------------------------------
    ax = axes[1]
    t_grid = np.linspace(0.1, 10, 50)
    base_rates = [base_market.curve(USD_CURVE).zero_rate(t) * 100 for t in t_grid]
    up_rates = [market_rate_up.curve(USD_CURVE).zero_rate(t) * 100 for t in t_grid]
    down_rates = [market_rate_down.curve(USD_CURVE).zero_rate(t) * 100 for t in t_grid]
    
    ax.plot(t_grid, base_rates, 'k-', linewidth=2, label='Base')
    ax.plot(t_grid, up_rates, '--', color='#E94F37', linewidth=2, label='+100bp')
    ax.plot(t_grid, down_rates, '--', color='#10B981', linewidth=2, label='-50bp')
    ax.fill_between(t_grid, down_rates, up_rates, alpha=0.2, color='gray')
    ax.set_xlabel('Tenor (years)')
    ax.set_ylabel('Zero Rate (%)')
    ax.set_title('Rate Curve Scenarios')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 3: Vol smile comparison
    # -------------------------------------------------------------------------
    ax = axes[2]
    strikes = np.linspace(0.98, 1.18, 30)
    base_vols = [base_market.vol_surface(EURUSD_VOL).implied_vol(1.0, k) * 100 for k in strikes]
    up_vols = [market_vol_up.vol_surface(EURUSD_VOL).implied_vol(1.0, k) * 100 for k in strikes]
    down_vols = [market_vol_down.vol_surface(EURUSD_VOL).implied_vol(1.0, k) * 100 for k in strikes]
    
    ax.plot(strikes, base_vols, 'k-', linewidth=2, label='Base')
    ax.plot(strikes, up_vols, '--', color='#E94F37', linewidth=2, label='+1pt')
    ax.plot(strikes, down_vols, '--', color='#10B981', linewidth=2, label='-1pt')
    ax.axvline(1.085, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Strike')
    ax.set_ylabel('Implied Vol (%)')
    ax.set_title('Vol Smile Scenarios (1Y)')
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
    │  1. Shock Types:                                                    │
    │     - SpotShock: Bump spot prices (relative or absolute)            │
    │     - VolShock: Bump entire vol surface (relative or absolute)      │
    │     - ParallelRateShock: Parallel shift yield curves                │
    │                                                                      │
    │  2. Bump Modes:                                                     │
    │     - relative: new = old × (1 + bump)                              │
    │     - absolute: new = old + bump                                    │
    │                                                                      │
    │  3. Applying Shocks:                                                │
    │     shocked_market = shock.apply(base_market)                       │
    │     - Returns a NEW Market (immutable pattern)                      │
    │     - Original market unchanged                                     │
    │                                                                      │
    │  4. Combining Shocks:                                               │
    │     market = shock1.apply(base_market)                              │
    │     market = shock2.apply(market)                                   │
    │     - Apply sequentially for combined scenarios                     │
    │                                                                      │
    │  5. Scenario Ladders:                                               │
    │     - Series of bumps for sensitivity analysis                      │
    │     - Useful for Greeks via finite differences                      │
    │                                                                      │
    │  These tools form the foundation for the risk module.               │
    │                                                                      │
    │  NEXT: See 07_timeseries_generation.py for scenario simulation      │
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
        # Setup: Create base market
        base_market = create_base_market()
        display_base_market(base_market)
        
        # Section 1: SpotShock
        market_spot_up, market_spot_down = demonstrate_spot_shock(base_market)
        
        # Section 2: VolShock
        market_vol_up, market_vol_down = demonstrate_vol_shock(base_market)
        
        # Section 3: ParallelRateShock
        market_rate_up, market_rate_down = demonstrate_rate_shock(base_market)
        
        # Section 4: Combined Shocks
        stressed_market = demonstrate_combined_shocks(base_market)
        
        # Section 5: Scenario Ladders
        demonstrate_scenario_ladders(base_market)
        
        # Section 6: Visualization
        visualize_shocks(
            base_market,
            market_rate_up, market_rate_down,
            market_vol_up, market_vol_down,
        )
        
        # Summary
        print_summary()
        
        logger.info("Example completed successfully!")
        
    except Exception as e:
        logger.exception(f"Example failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scenario Shocks Example",
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

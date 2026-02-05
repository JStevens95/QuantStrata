#!/usr/bin/env python3
"""
===============================================================================
Curves and Term Structures: Discount Factors and Interest Rates
===============================================================================

This example covers the construction and use of discount curves - the foundation
of all derivatives pricing and the mechanism for computing present values.

Learning Objectives
-------------------
1. **Discount Factors**: Understand df(t) and its role in present value
2. **Zero Rates**: Learn continuous compounding and rate extraction
3. **Forward Rates**: Understand the term structure of interest rates
4. **FX Forwards**: Apply interest rate parity to currency pricing

Mathematical Framework
----------------------
The fundamental relationship between discount factors and rates:

    df(t) = exp(-r(t) * t)           # Zero rate definition
    r(t) = -ln(df(t)) / t            # Rate from discount factor

    f(t1, t2) = ln(df(t1)/df(t2)) / (t2-t1)   # Forward rate

For FX forwards (Covered Interest Rate Parity):

    F(T) = S * df_foreign(T) / df_domestic(T)

Production Context
------------------
At a hedge fund:
- Curves are calibrated daily from market instruments (OIS, swaps)
- Multiple curves per currency (discounting vs projection)
- Curves drive all NPV calculations and risk sensitivities
- Curve risk (DV01, key rate duration) is a primary risk metric

Prerequisites
-------------
- Example 01: Market IDs and Quotes

Run This Example
----------------
    cd /path/to/QuantStrata
    PYTHONPATH=. python examples/fundamentals/02_curves_and_term_structures.py

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
# Path setup: Ensure imports work when running as script
# This resolves the repo root and adds it to sys.path
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# -----------------------------------------------------------------------------
# QuantStrata imports
# -----------------------------------------------------------------------------
from src.marketdata.core.ids import MarketId          # Unique identifiers
from src.marketdata.core.interfaces import Quote      # Scalar values
from src.marketdata.core.market import Market         # Market snapshot
from src.marketdata.curves.term_structure import (
    FlatZeroRateCurve,    # Constant rate (for testing)
    ZeroRateCurve,        # Interpolated term structure (PRODUCTION)
)

# =============================================================================
# LOGGING SETUP
# =============================================================================

# Configure structured logging (production standard)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Optional plotting (requires matplotlib)
ENABLE_PLOTTING = True

# Try to import matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available - plotting disabled")


# =============================================================================
# SECTION 1: Discount Factors - The Time Value of Money
# =============================================================================

def demonstrate_discount_factors() -> FlatZeroRateCurve:
    """
    Demonstrate discount factors and their fundamental role in pricing.
    
    A discount factor df(t) represents the present value of $1 received at
    time t in the future. It encapsulates:
    - The time value of money
    - Credit/counterparty risk (if applicable)
    - Funding costs
    
    Returns
    -------
    FlatZeroRateCurve
        A simple flat curve for comparison in later sections.
    
    Mathematical Background
    -----------------------
    For continuous compounding at rate r:
        df(t) = exp(-r * t)
    
    Example: At 5% annual rate, df(1) = exp(-0.05) ≈ 0.9512
    This means $1 received in 1 year is worth $0.9512 today.
    
    Production Notes
    ----------------
    - Flat curves are used for testing and benchmarking only
    - Production systems use calibrated curves from market data
    - Different curves for discounting vs. projection (multi-curve framework)
    """
    logger.info("=" * 70)
    logger.info("SECTION 1: Discount Factors - The Time Value of Money")
    logger.info("=" * 70)
    
    # -------------------------------------------------------------------------
    # Explain the concept of discount factors
    # This is fundamental to all fixed income and derivatives pricing
    # -------------------------------------------------------------------------
    explanation = """
    A discount factor df(t) represents the present value of $1 received at time t.
    
    Key relationships:
      df(0) = 1        → No discounting for money received today
      df(t) < 1        → Future money is worth less than today's money
      df(t) = exp(-r·t) → For continuous compounding at rate r
    
    Example: If df(1) = 0.95, then $1 in 1 year is worth $0.95 today.
    """
    logger.info(explanation)
    
    # -------------------------------------------------------------------------
    # Create a flat curve (constant rate across all tenors)
    # This is useful for testing and pedagogical purposes
    # -------------------------------------------------------------------------
    flat_rate = 0.05  # 5% annual continuously compounded rate
    
    flat_curve = FlatZeroRateCurve(
        continuously_compounded_rate=flat_rate  # Rate applies to all tenors
    )
    
    # -------------------------------------------------------------------------
    # Examine discount factors at standard tenors
    # These tenors represent common pricing/hedging points
    # -------------------------------------------------------------------------
    tenors = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    logger.info(f"Flat curve at {flat_rate:.1%} continuous rate:")
    logger.info(f"{'Tenor (years)':<15} {'Discount Factor':<18} {'Zero Rate':<12}")
    logger.info("-" * 45)
    
    for t in tenors:
        # df(t) returns the discount factor at tenor t
        df = flat_curve.df(t)
        
        # zero_rate(t) returns the continuously compounded zero rate
        # Note: At t=0, zero_rate is undefined; we display 0 for clarity
        zr = flat_curve.zero_rate(t) if t > 0 else 0.0
        
        logger.info(f"{t:<15.2f} {df:<18.6f} {zr:<12.4%}")
    
    # -------------------------------------------------------------------------
    # Demonstrate the fundamental relationship: df(t) = exp(-r * t)
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("Verification: df(t) = exp(-r * t)")
    test_tenor = 5.0
    expected_df = np.exp(-flat_rate * test_tenor)
    actual_df = flat_curve.df(test_tenor)
    logger.info(f"  For t={test_tenor}, r={flat_rate}:")
    logger.info(f"  Expected: exp(-{flat_rate}*{test_tenor}) = {expected_df:.6f}")
    logger.info(f"  Actual:   flat_curve.df({test_tenor}) = {actual_df:.6f}")
    logger.info(f"  Match: {np.isclose(expected_df, actual_df)}")
    
    return flat_curve


# =============================================================================
# SECTION 2: Zero Rates vs Forward Rates
# =============================================================================

def demonstrate_zero_vs_forward_rates() -> ZeroRateCurve:
    """
    Demonstrate the difference between zero rates and forward rates.
    
    Returns
    -------
    ZeroRateCurve
        An upward-sloping curve for use in later sections.
    
    Definitions
    -----------
    Zero Rate r(t):
        The annualized rate for lending/borrowing from TODAY to time t.
        Also called "spot rate" or "zero-coupon rate".
        
    Forward Rate f(t1, t2):
        The rate locked in TODAY for lending/borrowing from t1 to t2.
        Represents future expectations embedded in the term structure.
    
    Mathematical Relationships
    --------------------------
    df(t) = exp(-r(t) * t)
    
    f(t1, t2) = [r(t2)*t2 - r(t1)*t1] / (t2 - t1)
              = ln(df(t1)/df(t2)) / (t2 - t1)
    
    Production Notes
    ----------------
    - Forward rates reveal market expectations about future rates
    - The shape of the forward curve is critical for rate trading strategies
    - PCA decomposition (level, slope, curve) often applied to forwards
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 2: Zero Rates vs Forward Rates")
    logger.info("=" * 70)
    
    # -------------------------------------------------------------------------
    # Explain the concepts
    # -------------------------------------------------------------------------
    explanation = """
    Zero Rate r(t):
      The annualized rate for borrowing/lending from today to time t.
      df(t) = exp(-r(t) * t)
    
    Forward Rate f(t1, t2):
      The rate locked in today for borrowing/lending from t1 to t2.
      df(t1)/df(t2) = exp(f(t1,t2) * (t2 - t1))
    """
    logger.info(explanation)
    
    # -------------------------------------------------------------------------
    # Create an upward-sloping zero curve
    # This is typical of a "normal" yield curve where longer rates are higher
    # -------------------------------------------------------------------------
    tenors_input = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0])  # Standard tenors
    zero_rates_input = np.array([0.045, 0.047, 0.050, 0.052, 0.055, 0.057])  # Upward slope
    
    # ZeroRateCurve interpolates between input points
    # This is the PRODUCTION class for real curves
    zero_curve = ZeroRateCurve(
        tenors=tenors_input,      # Tenor points in years
        zero_rates=zero_rates_input,  # Corresponding zero rates
    )
    
    # -------------------------------------------------------------------------
    # Display zero rates, discount factors, and forward rates
    # -------------------------------------------------------------------------
    logger.info("Upward-sloping zero curve:")
    logger.info(f"{'Tenor':<10} {'Zero Rate':<12} {'DF':<15} {'Fwd to 10Y':<15}")
    logger.info("-" * 55)
    
    for t in [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]:
        # Extract zero rate at this tenor (interpolated if needed)
        zr = zero_curve.zero_rate(t)
        
        # Compute discount factor
        df = zero_curve.df(t)
        
        # Compute forward rate from t to 10Y
        # forward_rate(t1, t2) returns the rate from t1 to t2
        if t < 10.0:
            fwd = zero_curve.forward_rate(t, 10.0)
        else:
            fwd = zr  # At 10Y, forward to 10Y is the spot rate
        
        logger.info(f"{t:<10.2f} {zr:<12.4%} {df:<15.6f} {fwd:<15.4%}")
    
    # -------------------------------------------------------------------------
    # Explain the term structure shape
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("Interpretation:")
    logger.info("  - Zero rates increase with tenor → upward-sloping curve")
    logger.info("  - Forward rates > zero rates → market expects rates to rise")
    logger.info("  - This is a 'normal' yield curve shape")
    
    return zero_curve


# =============================================================================
# SECTION 3: Visualizing the Term Structure
# =============================================================================

def visualize_term_structure(
    zero_curve: ZeroRateCurve,
    tenors_input: np.ndarray,
    zero_rates_input: np.ndarray,
) -> None:
    """
    Create visualization of the term structure.
    
    Three plots:
    1. Zero rate curve with input points
    2. Discount factor curve
    3. Instantaneous forward rate curve
    
    Parameters
    ----------
    zero_curve : ZeroRateCurve
        The curve to visualize.
    tenors_input : np.ndarray
        Original input tenor points.
    zero_rates_input : np.ndarray
        Original input zero rates.
    
    Production Notes
    ----------------
    - Visualization is critical for curve validation
    - Check for arbitrage (negative forward rates)
    - Verify interpolation behaves sensibly
    """
    if not MATPLOTLIB_AVAILABLE or not ENABLE_PLOTTING:
        logger.info("Skipping plots (matplotlib not available or disabled)")
        return
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 3: Visualizing the Term Structure")
    logger.info("=" * 70)
    
    # -------------------------------------------------------------------------
    # Configure matplotlib style
    # -------------------------------------------------------------------------
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.figsize': (12, 5),
        'font.size': 11,
        'axes.titlesize': 13,
        'lines.linewidth': 2,
    })
    
    # Dense grid for smooth plots
    t_grid = np.linspace(0.01, 10.0, 100)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # -------------------------------------------------------------------------
    # Plot 1: Zero Rates
    # Shows the term structure with interpolation
    # -------------------------------------------------------------------------
    ax = axes[0]
    zero_rates = [zero_curve.zero_rate(t) for t in t_grid]
    ax.plot(
        t_grid, 
        np.array(zero_rates) * 100,  # Convert to percentage
        color='#2E86AB', 
        linewidth=2,
        label='Interpolated curve',
    )
    ax.scatter(
        tenors_input, 
        zero_rates_input * 100, 
        color='#E94F37', 
        s=60, 
        zorder=5,
        label='Input points',
    )
    ax.set_xlabel('Tenor (years)')
    ax.set_ylabel('Zero Rate (%)')
    ax.set_title('Zero Rate Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 2: Discount Factors
    # Shows the exponential decay of present value
    # -------------------------------------------------------------------------
    ax = axes[1]
    dfs = [zero_curve.df(t) for t in t_grid]
    ax.plot(t_grid, dfs, color='#8B5CF6', linewidth=2)
    ax.set_xlabel('Tenor (years)')
    ax.set_ylabel('Discount Factor')
    ax.set_title('Discount Factor Curve')
    ax.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 3: Instantaneous Forward Rates
    # Computed numerically as: f(t) ≈ -d(ln df)/dt
    # -------------------------------------------------------------------------
    ax = axes[2]
    dt = 0.01  # Small increment for numerical derivative
    fwd_rates = []
    for t in t_grid[:-1]:
        # f(t) = (df(t)/df(t+dt) - 1) / dt ≈ instantaneous forward
        fwd = (zero_curve.df(t) / zero_curve.df(t + dt) - 1) / dt
        fwd_rates.append(fwd)
    ax.plot(
        t_grid[:-1], 
        np.array(fwd_rates) * 100, 
        color='#10B981', 
        linewidth=2,
    )
    ax.set_xlabel('Tenor (years)')
    ax.set_ylabel('Instantaneous Forward Rate (%)')
    ax.set_title('Forward Rate Curve')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show(block=True)
    
    logger.info("Term structure visualization complete")


# =============================================================================
# SECTION 4: Using Curves in a Market Object
# =============================================================================

def demonstrate_market_with_curves() -> Market:
    """
    Demonstrate integrating curves into a Market object.
    
    Returns
    -------
    Market
        Complete market snapshot with curves.
    
    Production Notes
    ----------------
    - Market objects are the primary input to pricers
    - Multiple curves per currency in production (OIS, LIBOR, etc.)
    - Curve IDs follow the convention: IR.CURVE.{CCY}_{TYPE}
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 4: Using Curves in a Market Object")
    logger.info("=" * 70)
    
    # -------------------------------------------------------------------------
    # Create MarketIds for curves
    # Convention: asset_class="IR", mkt_type="CURVE", name="{CCY}_{TYPE}"
    # -------------------------------------------------------------------------
    usd_curve_id = MarketId(
        asset_class="IR",
        mkt_type="CURVE",
        name="USD_OIS",  # USD Overnight Index Swap curve
    )
    
    eur_curve_id = MarketId(
        asset_class="IR",
        mkt_type="CURVE",
        name="EUR_OIS",  # EUR Overnight Index Swap curve
    )
    
    # -------------------------------------------------------------------------
    # Create USD curve (higher rates - typical of USD vs EUR)
    # -------------------------------------------------------------------------
    usd_tenors = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
    usd_rates = np.array([0.050, 0.051, 0.052, 0.053, 0.055, 0.057])
    
    usd_curve = ZeroRateCurve(
        tenors=usd_tenors,
        zero_rates=usd_rates,
    )
    
    # -------------------------------------------------------------------------
    # Create EUR curve (lower rates)
    # The rate differential drives FX forward pricing
    # -------------------------------------------------------------------------
    eur_tenors = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
    eur_rates = np.array([0.035, 0.036, 0.038, 0.040, 0.042, 0.044])
    
    eur_curve = ZeroRateCurve(
        tenors=eur_tenors,
        zero_rates=eur_rates,
    )
    
    # -------------------------------------------------------------------------
    # Create FX spot quote
    # -------------------------------------------------------------------------
    eurusd_spot_id = MarketId(
        asset_class="FX",
        mkt_type="SPOT",
        name="EURUSD",
    )
    
    # -------------------------------------------------------------------------
    # Assemble the Market object
    # This is the standard structure passed to pricers
    # -------------------------------------------------------------------------
    market = Market(
        asof="2026-01-28",                            # Valuation date
        quotes={eurusd_spot_id: Quote(value=1.0850)}, # FX spot
        curves={                                       # Discount curves
            usd_curve_id: usd_curve,
            eur_curve_id: eur_curve,
        },
        vols={},  # No vol surfaces in this example
        meta={
            "source": "Example data",
            "description": "Two-currency market for FX forward pricing",
        },
    )
    
    logger.info(f"Market created with curves:")
    logger.info(f"  As-of: {market.asof}")
    logger.info(f"  Spot EUR/USD: {market.quote(eurusd_spot_id):.4f}")
    logger.info(f"  USD 5Y DF: {market.curve(usd_curve_id).df(5.0):.6f}")
    logger.info(f"  EUR 5Y DF: {market.curve(eur_curve_id).df(5.0):.6f}")
    
    # -------------------------------------------------------------------------
    # Calculate rate differential (drives FX forwards)
    # -------------------------------------------------------------------------
    usd_5y_rate = market.curve(usd_curve_id).zero_rate(5.0)
    eur_5y_rate = market.curve(eur_curve_id).zero_rate(5.0)
    rate_diff = (usd_5y_rate - eur_5y_rate) * 10000  # In basis points
    
    logger.info(f"")
    logger.info(f"5Y rate differential (USD - EUR): {rate_diff:.1f} bps")
    logger.info(f"  → EUR/USD forwards will be ABOVE spot (USD rates higher)")
    
    return market


# =============================================================================
# SECTION 5: FX Forward Pricing with Curves
# =============================================================================

def demonstrate_fx_forward_pricing(market: Market) -> None:
    """
    Demonstrate FX forward pricing using interest rate parity.
    
    Parameters
    ----------
    market : Market
        Market containing spot and curves.
    
    Covered Interest Rate Parity
    ----------------------------
    The FX forward rate is determined by no-arbitrage:
    
        F(T) = S * exp((r_d - r_f) * T)
             = S * df_foreign(T) / df_domestic(T)
    
    Where:
        S   = Spot rate (domestic per foreign)
        r_d = Domestic interest rate
        r_f = Foreign interest rate
        T   = Time to maturity
    
    For EUR/USD:
        - Domestic = USD (denominator currency)
        - Foreign = EUR (numerator currency)
        - Higher USD rates → forward ABOVE spot
    
    Production Notes
    ----------------
    - Forward points = (Forward - Spot) * 10,000 (in pips for FX)
    - Forward points drive FX swap markets
    - Crucial for hedging currency exposure
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 5: FX Forward Pricing with Curves")
    logger.info("=" * 70)
    
    # -------------------------------------------------------------------------
    # Explain the theory
    # -------------------------------------------------------------------------
    explanation = """
    The FX forward rate is determined by interest rate parity:
    
      F(T) = S * exp((r_domestic - r_foreign) * T)
           = S * df_foreign(T) / df_domestic(T)
    
    Where:
      S   = Spot rate
      r_d = Domestic (USD) rate
      r_f = Foreign (EUR) rate
      T   = Time to maturity
    """
    logger.info(explanation)
    
    # -------------------------------------------------------------------------
    # Extract market data
    # -------------------------------------------------------------------------
    eurusd_spot_id = MarketId("FX", "SPOT", "EURUSD")
    usd_curve_id = MarketId("IR", "CURVE", "USD_OIS")
    eur_curve_id = MarketId("IR", "CURVE", "EUR_OIS")
    
    spot = market.quote(eurusd_spot_id)
    usd_curve = market.curve(usd_curve_id)
    eur_curve = market.curve(eur_curve_id)
    
    # -------------------------------------------------------------------------
    # Calculate forwards at standard tenors
    # -------------------------------------------------------------------------
    logger.info("EUR/USD Forward Rates:")
    logger.info(f"{'Tenor':<10} {'USD DF':<12} {'EUR DF':<12} {'Forward':<12} {'Fwd Points':<12}")
    logger.info("-" * 58)
    
    for T in [0.25, 0.5, 1.0, 2.0, 5.0]:
        # Get discount factors from curves
        df_usd = usd_curve.df(T)  # Domestic discount factor
        df_eur = eur_curve.df(T)  # Foreign discount factor
        
        # Apply covered interest rate parity
        # Forward = Spot * (df_foreign / df_domestic)
        forward = spot * (df_eur / df_usd)
        
        # Forward points = (Forward - Spot) * 10000 (standard FX convention)
        fwd_points = (forward - spot) * 10000
        
        logger.info(f"{T:<10.2f} {df_usd:<12.6f} {df_eur:<12.6f} {forward:<12.4f} {fwd_points:<+12.1f}")
    
    # -------------------------------------------------------------------------
    # Explain the result
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("Interpretation:")
    logger.info("  USD rates > EUR rates → df_EUR > df_USD → Forward > Spot")
    logger.info("  Positive forward points → EUR/USD forward is above spot")


# =============================================================================
# SECTION 6: Curve Comparison Visualization
# =============================================================================

def visualize_curve_comparison(market: Market) -> None:
    """
    Visualize USD vs EUR curves and the resulting forward curve.
    
    Parameters
    ----------
    market : Market
        Market containing curves.
    
    Production Notes
    ----------------
    - Curve comparison is essential for relative value trading
    - Forward curve shape drives basis swaps
    - Visualization helps identify calibration issues
    """
    if not MATPLOTLIB_AVAILABLE or not ENABLE_PLOTTING:
        logger.info("Skipping plots (matplotlib not available or disabled)")
        return
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 6: Curve Comparison - USD vs EUR")
    logger.info("=" * 70)
    
    # -------------------------------------------------------------------------
    # Extract data from market
    # -------------------------------------------------------------------------
    eurusd_spot_id = MarketId("FX", "SPOT", "EURUSD")
    usd_curve_id = MarketId("IR", "CURVE", "USD_OIS")
    eur_curve_id = MarketId("IR", "CURVE", "EUR_OIS")
    
    spot = market.quote(eurusd_spot_id)
    usd_curve = market.curve(usd_curve_id)
    eur_curve = market.curve(eur_curve_id)
    
    # -------------------------------------------------------------------------
    # Configure matplotlib
    # -------------------------------------------------------------------------
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    t_grid = np.linspace(0.1, 10.0, 100)
    
    # -------------------------------------------------------------------------
    # Plot 1: Zero rate comparison
    # -------------------------------------------------------------------------
    ax = axes[0]
    usd_zeros = [usd_curve.zero_rate(t) * 100 for t in t_grid]
    eur_zeros = [eur_curve.zero_rate(t) * 100 for t in t_grid]
    
    ax.plot(t_grid, usd_zeros, color='#2E86AB', linewidth=2, label='USD OIS')
    ax.plot(t_grid, eur_zeros, color='#E94F37', linewidth=2, label='EUR OIS')
    ax.fill_between(
        t_grid, eur_zeros, usd_zeros, 
        alpha=0.2, color='gray',
        label='Rate differential',
    )
    ax.set_xlabel('Tenor (years)')
    ax.set_ylabel('Zero Rate (%)')
    ax.set_title('Zero Rate Curves: USD vs EUR')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 2: FX forward curve
    # -------------------------------------------------------------------------
    ax = axes[1]
    forwards = [spot * (eur_curve.df(t) / usd_curve.df(t)) for t in t_grid]
    
    ax.plot(t_grid, forwards, color='#8B5CF6', linewidth=2, label='Forward curve')
    ax.axhline(spot, color='gray', linestyle='--', alpha=0.7, label=f'Spot = {spot:.4f}')
    ax.set_xlabel('Tenor (years)')
    ax.set_ylabel('EUR/USD Forward Rate')
    ax.set_title('EUR/USD Forward Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show(block=True)
    
    logger.info("Curve comparison visualization complete")


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
    │  1. Discount Factor df(t):                                          │
    │     - Present value of $1 received at time t                        │
    │     - df(t) = exp(-r(t) * t) for continuous compounding             │
    │                                                                      │
    │  2. Zero Rate r(t):                                                 │
    │     - Annualized rate from today to time t                          │
    │     - Inverted from: r(t) = -ln(df(t)) / t                          │
    │                                                                      │
    │  3. Forward Rate f(t1, t2):                                         │
    │     - Rate locked today for period [t1, t2]                         │
    │     - f(t1,t2) = ln(df(t1)/df(t2)) / (t2-t1)                        │
    │                                                                      │
    │  4. Curve Types:                                                    │
    │     - FlatZeroRateCurve: Constant rate (testing only)               │
    │     - ZeroRateCurve: Interpolated term structure (PRODUCTION)       │
    │                                                                      │
    │  5. FX Forwards via Interest Rate Parity:                           │
    │     - F(T) = S * df_foreign(T) / df_domestic(T)                     │
    │     - Higher domestic rates → forward above spot                    │
    │                                                                      │
    │  NEXT: See 03_volatility_surface.py for volatility surfaces         │
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
        # Section 1: Discount Factors
        flat_curve = demonstrate_discount_factors()
        
        # Section 2: Zero vs Forward Rates
        zero_curve = demonstrate_zero_vs_forward_rates()
        
        # Section 3: Visualization
        tenors_input = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
        zero_rates_input = np.array([0.045, 0.047, 0.050, 0.052, 0.055, 0.057])
        visualize_term_structure(zero_curve, tenors_input, zero_rates_input)
        
        # Section 4: Market with Curves
        market = demonstrate_market_with_curves()
        
        # Section 5: FX Forward Pricing
        demonstrate_fx_forward_pricing(market)
        
        # Section 6: Curve Comparison
        visualize_curve_comparison(market)
        
        # Summary
        print_summary()
        
        logger.info("Example completed successfully!")
        
    except Exception as e:
        logger.exception(f"Example failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Parse command-line arguments
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Curves and Term Structures Example",
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

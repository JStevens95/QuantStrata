#!/usr/bin/env python3
"""
===============================================================================
Volatility Surfaces: Implied Volatility and the Smile
===============================================================================

This example covers volatility surfaces - the market's view of future 
uncertainty and a critical input to all option pricing.

Learning Objectives
-------------------
1. **Implied Volatility**: Understand IV and its market interpretation
2. **Volatility Smile**: Learn why IV varies with strike
3. **Term Structure**: Understand how IV varies with expiry
4. **FX Conventions**: Learn delta-based quoting for FX options

Mathematical Framework
----------------------
Implied volatility is the value of σ that makes Black-Scholes price equal 
market price:

    C_market = BS(S, K, T, r, σ_implied)

The volatility smile reflects:
- Fat tails in return distributions (kurtosis > 3)
- Crash risk premium (especially for equities)
- Supply/demand for OTM options

Vol Surface Parameterization (SABR model):

    σ(K) ≈ σ_ATM * [1 + ρ*ν*(K-F)/σ_ATM + (2-3ρ²)*ν²*(K-F)²/6σ_ATM²]

Production Context
------------------
At a hedge fund:
- Vol surfaces are calibrated from option market prices
- Interpolation across strikes/expiries is critical
- Vol risk (vega, volga, vanna) is a primary concern
- FX desks quote in delta terms (25D, ATM, 10D)

Prerequisites
-------------
- Example 01: Market IDs and Quotes
- Example 02: Curves and Term Structures

Run This Example
----------------
    cd /path/to/QuantStrata
    PYTHONPATH=. python examples/fundamentals/03_volatility_surface.py

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
from typing import Dict, List, Tuple, Optional

import numpy as np

# -----------------------------------------------------------------------------
# Path setup: Ensure imports work when running as script
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# -----------------------------------------------------------------------------
# QuantStrata imports
# -----------------------------------------------------------------------------
from src.marketdata.core.ids import MarketId          # Unique identifiers
from src.marketdata.core.interfaces import Quote      # Scalar values
from src.marketdata.core.market import Market         # Market snapshot
from src.marketdata.surfaces.vol_surface import (
    FlatVolSurface,    # Constant vol (for testing)
    GridVolSurface,    # Interpolated vol surface (PRODUCTION)
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

# Optional plotting
ENABLE_PLOTTING = True

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available - plotting disabled")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def generate_smile_vol(
    expiry: float,
    strike: float,
    spot: float = 100.0,
    atm_vol: float = 0.15,
    skew: float = -0.1,
    smile: float = 0.05,
) -> float:
    """
    Generate implied volatility with smile and skew characteristics.
    
    This is a simplified parametric model for illustration. Production
    systems typically use SABR, SVI, or neural network interpolation.
    
    Parameters
    ----------
    expiry : float
        Time to expiry in years.
    strike : float
        Option strike price.
    spot : float, default 100.0
        Current spot price.
    atm_vol : float, default 0.15
        At-the-money volatility level.
    skew : float, default -0.1
        Linear skew coefficient (negative = puts more expensive).
    smile : float, default 0.05
        Quadratic smile coefficient (positive = wings more expensive).
    
    Returns
    -------
    float
        Implied volatility for the given expiry/strike.
    
    Model Specification
    -------------------
    σ(K,T) = σ_ATM + skew * ln(K/S) + smile * ln(K/S)²
             × term_structure_adjustment
    
    The term structure factor dampens vol at longer expiries,
    reflecting mean reversion in volatility.
    """
    # Log-moneyness: ln(K/S), positive for OTM calls, negative for OTM puts
    moneyness = np.log(strike / spot)
    
    # Base vol with skew (linear in moneyness)
    # Negative skew means IV increases as strike decreases (typical equity skew)
    vol = atm_vol + skew * moneyness
    
    # Add smile (quadratic in moneyness)
    # Creates the "U-shape" around ATM
    vol += smile * moneyness**2
    
    # Term structure adjustment (vol decreases slightly with expiry)
    # Reflects mean reversion in volatility
    vol *= (1.0 - 0.1 * np.log(expiry + 0.1))
    
    # Floor at 5% to prevent numerical issues
    return max(0.05, vol)


# =============================================================================
# SECTION 1: What is Implied Volatility?
# =============================================================================

def demonstrate_implied_volatility() -> None:
    """
    Explain the concept of implied volatility.
    
    Implied volatility (IV) is the market's expectation of future volatility,
    backed out from option prices using Black-Scholes or similar models.
    
    Key points:
    - IV is NOT historical (realized) volatility
    - Higher IV = more expensive options
    - IV varies with strike (smile) and expiry (term structure)
    - The Black-Scholes constant-vol assumption is violated
    
    Production Notes
    ----------------
    - IV is quoted as an annualized standard deviation
    - VIX is derived from S&P 500 option IV
    - IV typically spikes during market stress
    """
    logger.info("=" * 70)
    logger.info("SECTION 1: What is Implied Volatility?")
    logger.info("=" * 70)
    
    explanation = """
    Implied Volatility (IV) is the market's expectation of future volatility
    implied by option prices.
    
    Key points:
    ┌────────────────────────────────────────────────────────────────┐
    │ - IV is backed out from option prices using Black-Scholes     │
    │ - It's NOT the same as historical (realized) volatility       │
    │ - Higher IV means more expensive options                      │
    │ - IV varies by strike (smile) and expiry (term structure)     │
    │                                                                │
    │ The Black-Scholes assumption of constant volatility is        │
    │ violated in practice. The "volatility surface" captures how   │
    │ IV varies across strikes and expiries.                        │
    └────────────────────────────────────────────────────────────────┘
    
    Mathematical relationship:
        C_market = BS(S, K, T, r, σ_implied)
        
    Where σ_implied is found by numerical inversion.
    """
    logger.info(explanation)


# =============================================================================
# SECTION 2: Flat Volatility Surface
# =============================================================================

def demonstrate_flat_surface() -> FlatVolSurface:
    """
    Demonstrate the simplest volatility surface: constant vol.
    
    Returns
    -------
    FlatVolSurface
        A flat surface for testing and benchmarking.
    
    Production Notes
    ----------------
    - Flat surfaces are used ONLY for testing
    - They correspond to the Black-Scholes assumption
    - Real markets exhibit smile and term structure
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 2: Flat Volatility Surface")
    logger.info("=" * 70)
    
    # -------------------------------------------------------------------------
    # Create a flat vol surface (same vol everywhere)
    # This is the Black-Scholes world
    # -------------------------------------------------------------------------
    flat_vol = 0.15  # 15% annualized volatility
    
    flat_surface = FlatVolSurface(sigma=flat_vol)
    
    logger.info(f"Flat volatility surface at {flat_vol:.1%}:")
    logger.info(f"{'Expiry':<10} {'Strike':<10} {'Implied Vol':<12}")
    logger.info("-" * 32)
    
    # Demonstrate that IV is constant everywhere
    for expiry in [0.25, 0.5, 1.0]:
        for strike in [90, 100, 110]:
            iv = flat_surface.implied_vol(expiry, strike)
            logger.info(f"{expiry:<10.2f} {strike:<10.0f} {iv:<12.2%}")
    
    logger.info("")
    logger.info("Note: Flat surfaces are for TESTING only - real markets have smile!")
    
    return flat_surface


# =============================================================================
# SECTION 3: The Volatility Smile
# =============================================================================

def demonstrate_volatility_smile() -> GridVolSurface:
    """
    Demonstrate the volatility smile phenomenon.
    
    Returns
    -------
    GridVolSurface
        A realistic vol surface with smile characteristics.
    
    The Volatility Smile
    --------------------
    In practice, IV varies with strike:
    - Higher for OTM puts (low strikes) - "skew"
    - Higher for OTM calls (high strikes) - "wing"
    - Lowest around ATM strikes
    - Creates a "smile" or "smirk" shape
    
    This reflects:
    - Fat tails in real return distributions (kurtosis > 3)
    - Crash risk premium (especially for equities)
    - Supply/demand imbalances
    
    Production Notes
    ----------------
    - SABR model is standard for FX/IR options
    - SVI model is common for equity options
    - Machine learning increasingly used for interpolation
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 3: The Volatility Smile")
    logger.info("=" * 70)
    
    explanation = """
    The volatility smile shows that IV typically:
    ┌────────────────────────────────────────────────────────────────┐
    │ - Is higher for OTM puts (low strikes) - "skew"               │
    │ - Is higher for OTM calls (high strikes) - "wing"             │
    │ - Is lowest around ATM strikes                                 │
    │ - Creates a "smile" or "smirk" shape                          │
    │                                                                │
    │ This reflects:                                                 │
    │   • Fat tails in real return distributions                    │
    │   • Crash risk premium (especially for equities)              │
    │   • Supply/demand imbalances                                  │
    └────────────────────────────────────────────────────────────────┘
    """
    logger.info(explanation)
    
    # -------------------------------------------------------------------------
    # Define the vol surface grid
    # -------------------------------------------------------------------------
    spot = 100.0  # Reference spot price
    expiries = np.array([0.25, 0.5, 1.0, 2.0])  # Standard expiry tenors
    strikes = np.linspace(80, 120, 21)  # Range around ATM
    
    # -------------------------------------------------------------------------
    # Generate vol grid with smile characteristics
    # -------------------------------------------------------------------------
    vol_grid = np.zeros((len(expiries), len(strikes)))
    
    for i, exp in enumerate(expiries):
        for j, k in enumerate(strikes):
            vol_grid[i, j] = generate_smile_vol(
                expiry=exp,
                strike=k,
                spot=spot,
                atm_vol=0.15,    # 15% ATM vol
                skew=-0.1,       # Negative skew (puts more expensive)
                smile=0.05,      # Quadratic smile
            )
    
    # -------------------------------------------------------------------------
    # Create the GridVolSurface
    # This is the PRODUCTION class for vol surfaces
    # -------------------------------------------------------------------------
    smile_surface = GridVolSurface(
        expiries=expiries,
        strikes=strikes,
        implied_vols=vol_grid,
    )
    
    # -------------------------------------------------------------------------
    # Display sample points
    # -------------------------------------------------------------------------
    logger.info(f"Smile surface at 1Y expiry:")
    logger.info(f"{'Strike':<10} {'Moneyness':<12} {'Implied Vol':<12}")
    logger.info("-" * 34)
    
    test_strikes = [85, 95, 100, 105, 115]
    for strike in test_strikes:
        iv = smile_surface.implied_vol(1.0, strike)
        moneyness = np.log(strike / spot) * 100  # In percent
        logger.info(f"{strike:<10.0f} {moneyness:<+12.1f}% {iv:<12.2%}")
    
    return smile_surface


# =============================================================================
# SECTION 4: Visualizing the Volatility Surface
# =============================================================================

def visualize_volatility_surface(smile_surface: GridVolSurface) -> None:
    """
    Create comprehensive visualization of the vol surface.
    
    Parameters
    ----------
    smile_surface : GridVolSurface
        The vol surface to visualize.
    
    Three views:
    1. Smile at different expiries (strike vs IV)
    2. Term structure at different strikes (expiry vs IV)
    3. 3D surface view
    
    Production Notes
    ----------------
    - Visualization is critical for detecting arbitrage
    - Look for calendar spread arbitrage (term structure)
    - Look for butterfly arbitrage (smile convexity)
    """
    if not MATPLOTLIB_AVAILABLE or not ENABLE_PLOTTING:
        logger.info("Skipping plots (matplotlib not available or disabled)")
        return
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 4: Visualizing the Volatility Surface")
    logger.info("=" * 70)
    
    # -------------------------------------------------------------------------
    # Configure matplotlib
    # -------------------------------------------------------------------------
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.figsize': (12, 5),
        'font.size': 11,
        'axes.titlesize': 13,
        'lines.linewidth': 2,
    })
    
    spot = 100.0
    expiries = np.array([0.25, 0.5, 1.0, 2.0])
    strikes = np.linspace(80, 120, 21)
    
    fig = plt.figure(figsize=(15, 5))
    
    # -------------------------------------------------------------------------
    # Plot 1: Smile at different expiries
    # Shows how IV varies with strike for different maturities
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(131)
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(expiries)))
    
    for exp, color in zip(expiries, colors):
        vols = [smile_surface.implied_vol(exp, k) * 100 for k in strikes]
        ax1.plot(strikes, vols, color=color, linewidth=2, label=f'T = {exp}Y')
    
    ax1.axvline(spot, color='gray', linestyle='--', alpha=0.5, label='ATM')
    ax1.set_xlabel('Strike')
    ax1.set_ylabel('Implied Volatility (%)')
    ax1.set_title('Volatility Smile by Expiry')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 2: Term structure at different strikes
    # Shows how IV varies with expiry for different strikes
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(132)
    exp_grid = np.linspace(0.1, 2.0, 50)
    test_strikes_plot = [85, 95, 100, 105, 115]
    colors = plt.cm.RdYlBu(np.linspace(0.1, 0.9, len(test_strikes_plot)))
    
    for strike, color in zip(test_strikes_plot, colors):
        vols = [smile_surface.implied_vol(t, strike) * 100 for t in exp_grid]
        ax2.plot(exp_grid, vols, color=color, linewidth=2, label=f'K = {strike}')
    
    ax2.set_xlabel('Time to Expiry (years)')
    ax2.set_ylabel('Implied Volatility (%)')
    ax2.set_title('Volatility Term Structure by Strike')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 3: 3D surface view
    # Shows the complete vol surface
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(133, projection='3d')
    K_grid, T_grid = np.meshgrid(strikes, expiries)
    V_grid = np.array([
        [smile_surface.implied_vol(t, k) * 100 for k in strikes]
        for t in expiries
    ])
    
    surf = ax3.plot_surface(K_grid, T_grid, V_grid, cmap='viridis', alpha=0.8)
    ax3.set_xlabel('Strike')
    ax3.set_ylabel('Expiry')
    ax3.set_zlabel('IV (%)')
    ax3.set_title('3D Volatility Surface')
    ax3.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    plt.show(block=True)
    
    logger.info("Volatility surface visualization complete")


# =============================================================================
# SECTION 5: Using Vol Surfaces in a Market Object
# =============================================================================

def demonstrate_market_with_vol_surface() -> Market:
    """
    Demonstrate integrating a vol surface into a Market object.
    
    Returns
    -------
    Market
        Complete market snapshot with vol surface.
    
    Production Notes
    ----------------
    - Vol surfaces are identified by MarketId with mkt_type="VOL"
    - Multiple surfaces per underlying in production
    - Surface calibration runs daily (or more frequently)
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 5: Using Vol Surfaces in a Market Object")
    logger.info("=" * 70)
    
    # -------------------------------------------------------------------------
    # Create MarketIds
    # -------------------------------------------------------------------------
    eurusd_spot_id = MarketId("FX", "SPOT", "EURUSD")
    eurusd_vol_id = MarketId("FX", "VOL", "EURUSD")
    
    # -------------------------------------------------------------------------
    # Create FX vol surface
    # Scale strikes to FX level (spot around 1.08)
    # -------------------------------------------------------------------------
    fx_spot = 1.0850
    expiries = np.array([0.25, 0.5, 1.0, 2.0])
    fx_strikes = np.linspace(0.95, 1.20, 21)  # FX strike range
    
    fx_vol_grid = np.zeros((len(expiries), len(fx_strikes)))
    for i, exp in enumerate(expiries):
        for j, k in enumerate(fx_strikes):
            fx_vol_grid[i, j] = generate_smile_vol(
                expiry=exp,
                strike=k,
                spot=fx_spot,
                atm_vol=0.08,     # 8% ATM vol (typical FX)
                skew=-0.15,       # Moderate skew
                smile=0.10,       # Smile effect
            )
    
    fx_vol_surface = GridVolSurface(
        expiries=expiries,
        strikes=fx_strikes,
        implied_vols=fx_vol_grid,
    )
    
    # -------------------------------------------------------------------------
    # Create Market object
    # -------------------------------------------------------------------------
    market = Market(
        asof="2026-01-28",
        quotes={eurusd_spot_id: Quote(value=fx_spot)},
        curves={},  # Curves from example 02
        vols={eurusd_vol_id: fx_vol_surface},
        meta={"source": "Example data"},
    )
    
    logger.info(f"Market created with EUR/USD vol surface:")
    logger.info(f"  As-of: {market.asof}")
    logger.info(f"  Spot: {market.quote(eurusd_spot_id):.4f}")
    
    # -------------------------------------------------------------------------
    # Query vol surface from market
    # -------------------------------------------------------------------------
    vol_surface = market.vol_surface(eurusd_vol_id)
    
    logger.info("")
    logger.info("EUR/USD Implied Volatility:")
    logger.info(f"{'Expiry':<10} {'Strike':<12} {'IV':<10}")
    logger.info("-" * 32)
    
    for exp in [0.25, 0.5, 1.0]:
        for strike in [1.00, 1.08, 1.15]:
            iv = vol_surface.implied_vol(exp, strike)
            logger.info(f"{exp:<10.2f} {strike:<12.4f} {iv:<10.2%}")
    
    return market


# =============================================================================
# SECTION 6: Delta-Based Quoting Convention (FX Markets)
# =============================================================================

def demonstrate_delta_quoting() -> None:
    """
    Explain the FX market convention for quoting volatility.
    
    FX options are quoted in terms of delta rather than strike:
    - 25D Put: 25 delta put (OTM, ~1 std dev below ATM)
    - ATM: At-the-money (50 delta or ATMF/DNS)
    - 25D Call: 25 delta call (OTM, ~1 std dev above ATM)
    
    Market quotes typically include:
    - ATM volatility
    - 25D Risk Reversal: σ(25D Call) - σ(25D Put)
    - 25D Butterfly: [σ(25D Call) + σ(25D Put)]/2 - σ(ATM)
    
    Risk Reversal captures the skew
    Butterfly captures the smile curvature
    
    Production Notes
    ----------------
    - FX desks quote in delta terms
    - Strike conversion requires option pricing model
    - Different ATM conventions (DNS, ATMF, spot-delta)
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 6: Delta-Based Quoting Convention (FX Markets)")
    logger.info("=" * 70)
    
    explanation = """
    FX options are quoted in terms of delta rather than strike:
    
    ┌────────────────────────────────────────────────────────────────┐
    │ 25D Put   - 25 delta put (OTM put, ~1 std dev below ATM)      │
    │ ATM       - At-the-money (50 delta, or DNS/ATMF)              │
    │ 25D Call  - 25 delta call (OTM call, ~1 std dev above ATM)    │
    └────────────────────────────────────────────────────────────────┘
    
    Market quotes typically include:
      • ATM volatility
      • 25D Risk Reversal: σ(25D Call) - σ(25D Put)
      • 25D Butterfly: [σ(25D Call) + σ(25D Put)]/2 - σ(ATM)
    
    Risk Reversal captures the skew (puts vs calls premium)
    Butterfly captures the smile curvature (wings vs center)
    """
    logger.info(explanation)
    
    # -------------------------------------------------------------------------
    # Illustrative delta-based quotes
    # -------------------------------------------------------------------------
    atm_vol = 0.08       # 8% ATM vol
    rr_25d = -0.012      # Negative = puts more expensive (typical FX skew)
    bf_25d = 0.005       # Positive = wings more expensive (smile)
    
    # Convert to strike vols
    # σ(25D Put) = ATM + BF - RR/2
    # σ(25D Call) = ATM + BF + RR/2
    vol_25d_put = atm_vol + bf_25d - rr_25d / 2
    vol_25d_call = atm_vol + bf_25d + rr_25d / 2
    
    logger.info("")
    logger.info("Delta-based EUR/USD Vol Quotes (1Y):")
    logger.info(f"  ATM:        {atm_vol:.2%}")
    logger.info(f"  25D RR:     {rr_25d*100:+.2f} vol points")
    logger.info(f"  25D BF:     {bf_25d*100:.2f} vol points")
    logger.info("")
    logger.info("Implied strike vols:")
    logger.info(f"  25D Put:    {vol_25d_put:.2%}")
    logger.info(f"  ATM:        {atm_vol:.2%}")
    logger.info(f"  25D Call:   {vol_25d_call:.2%}")
    
    # -------------------------------------------------------------------------
    # Interpretation
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("Interpretation:")
    logger.info("  - Negative RR → puts more expensive → downside protection demand")
    logger.info("  - Positive BF → wings more expensive → tail risk premium")
    logger.info("  - These values are typical for EUR/USD")


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
    │  1. Implied Volatility (IV):                                        │
    │     - Market's expectation of future volatility                     │
    │     - Backed out from option prices via Black-Scholes               │
    │     - Not the same as historical volatility                         │
    │                                                                      │
    │  2. Volatility Smile:                                               │
    │     - IV varies with strike (smile/skew)                            │
    │     - IV varies with expiry (term structure)                        │
    │     - Reflects fat tails and crash risk premium                     │
    │                                                                      │
    │  3. Vol Surface Types:                                              │
    │     - FlatVolSurface: Constant vol (testing only)                   │
    │     - GridVolSurface: Interpolated grid (PRODUCTION)                │
    │                                                                      │
    │  4. FX Quoting Convention:                                          │
    │     - Delta-based quotes (25D, ATM, 10D)                            │
    │     - Risk Reversal = skew measure                                  │
    │     - Butterfly = curvature measure                                 │
    │                                                                      │
    │  5. Market Integration:                                             │
    │     - Vol surfaces stored in Market.vols dictionary                 │
    │     - Accessed via market.vol_surface(market_id)                    │
    │                                                                      │
    │  NEXT: See 04_timeseries_datasets.py for time series data           │
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
        # Section 1: Implied Volatility Concept
        demonstrate_implied_volatility()
        
        # Section 2: Flat Vol Surface
        flat_surface = demonstrate_flat_surface()
        
        # Section 3: Volatility Smile
        smile_surface = demonstrate_volatility_smile()
        
        # Section 4: Visualization
        visualize_volatility_surface(smile_surface)
        
        # Section 5: Market Integration
        market = demonstrate_market_with_vol_surface()
        
        # Section 6: FX Delta Quoting
        demonstrate_delta_quoting()
        
        # Summary
        print_summary()
        
        logger.info("Example completed successfully!")
        
    except Exception as e:
        logger.exception(f"Example failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Volatility Surface Example",
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

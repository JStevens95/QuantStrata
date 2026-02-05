#!/usr/bin/env python3
"""
===============================================================================
Market Snapshots: From Dataset to Pricing
===============================================================================

This example demonstrates how to extract Market snapshots from MarketDataset
and use them for pricing - the critical bridge between data and valuation.

Learning Objectives
-------------------
1. **Snapshot Extraction**: Get Market objects from MarketDataset
2. **Iteration Patterns**: Loop over time, scenarios, or both
3. **Efficient Access**: Direct panel access vs snapshot extraction
4. **Statistics**: Compute VaR and risk metrics from scenario distributions

Snapshot Coordinates
--------------------
    market = dataset.snapshot(time_idx, scenario_idx)
    
    time_idx:     0 to len(dates)-1
    scenario_idx: 0 to n_scenarios-1

The returned Market object is ready for pricing with full curves and vols.

Production Context
------------------
At a hedge fund, snapshot iteration is used for:
- Backtesting: Fixed scenario (=0), loop over time
- Monte Carlo VaR: Fixed time (=horizon), loop over scenarios
- Full P&L attribution: Loop over both dimensions

Prerequisites
-------------
- Example 01-04: Market fundamentals
- Understanding of MarketDataset structure

Run This Example
----------------
    cd /path/to/QuantStrata
    PYTHONPATH=. python examples/fundamentals/05_market_snapshot.py

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
from typing import Dict, List, Optional

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
from src.marketdata.core.panel import Panel
from src.marketdata.core.dataset import MarketDataset
from src.marketdata.core.market import Market
from src.marketdata.curves.factory import ZeroRateCurveFactory
from src.marketdata.surfaces.factory import GridVolFactory

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
EURUSD_VOL = MarketId(asset_class="FX", mkt_type="VOL", name="EURUSD")


# =============================================================================
# HELPER: Create Sample Dataset
# =============================================================================

def create_sample_dataset(
    n_dates: int = 10,
    n_scenarios: int = 50,
) -> MarketDataset:
    """
    Create a sample dataset for demonstration.
    
    Parameters
    ----------
    n_dates : int
        Number of dates in the time series.
    n_scenarios : int
        Number of scenarios per date.
    
    Returns
    -------
    MarketDataset
        A complete dataset with spot, curves, and vols.
    
    Production Notes
    ----------------
    In production, this would load from:
    - Database (historical data)
    - Scenario generator (Monte Carlo)
    - Stress testing framework (specific scenarios)
    """
    # -------------------------------------------------------------------------
    # Generate dates
    # -------------------------------------------------------------------------
    dates = [f"2026-01-{20+i:02d}" for i in range(n_dates)]
    
    np.random.seed(42)  # Reproducibility
    
    # -------------------------------------------------------------------------
    # Spot panel [T, S]
    # Base with drift + random shocks
    # -------------------------------------------------------------------------
    spot_base = 1.0850 + np.arange(n_dates).reshape(-1, 1) * 0.001  # Small drift
    spot_scenarios = spot_base + np.random.randn(n_dates, n_scenarios) * 0.005
    spot_panel = Panel(data=spot_scenarios, axis_names=("time", "scenario"))
    
    # -------------------------------------------------------------------------
    # Curve panel [T, S, K, 2]
    # -------------------------------------------------------------------------
    tenors = np.array([0.25, 0.5, 1.0, 2.0, 5.0])
    base_rates = np.array([0.045, 0.048, 0.050, 0.053, 0.057])
    
    curve_params = np.zeros((n_dates, n_scenarios, len(tenors), 2))
    for t in range(n_dates):
        for s in range(n_scenarios):
            rate_shock = np.random.randn() * 0.002
            curve_params[t, s, :, 0] = tenors
            curve_params[t, s, :, 1] = base_rates + rate_shock
    
    curve_panel = Panel(
        data=curve_params,
        axis_names=("time", "scenario", "tenor", "cols"),
    )
    curve_factory = ZeroRateCurveFactory()
    
    # -------------------------------------------------------------------------
    # Vol panel [T, S, n_exp, n_k]
    # -------------------------------------------------------------------------
    vol_expiries = np.array([0.25, 0.5, 1.0])
    vol_strikes = np.linspace(0.95, 1.20, 11)
    base_vol = 0.08
    
    vol_params = np.zeros((n_dates, n_scenarios, len(vol_expiries), len(vol_strikes)))
    for t in range(n_dates):
        for s in range(n_scenarios):
            vol_shock = np.random.randn() * 0.003
            vol_params[t, s, :, :] = base_vol + vol_shock
    
    vol_panel = Panel(
        data=vol_params,
        axis_names=("time", "scenario", "expiry", "strike"),
    )
    vol_factory = GridVolFactory(expiries=vol_expiries, strikes=vol_strikes)
    
    return MarketDataset(
        dates=dates,
        n_scenarios=n_scenarios,
        panels={EURUSD_SPOT: spot_panel},
        curve_params={USD_CURVE: curve_panel},
        curve_factories={USD_CURVE: curve_factory},
        vol_params={EURUSD_VOL: vol_panel},
        vol_factories={EURUSD_VOL: vol_factory},
    )


# =============================================================================
# SECTION 1: Basic Snapshot Extraction
# =============================================================================

def demonstrate_basic_extraction(dataset: MarketDataset) -> None:
    """
    Demonstrate basic snapshot extraction from dataset.
    
    Parameters
    ----------
    dataset : MarketDataset
        The dataset to extract from.
    
    The snapshot() method returns a complete Market object:
    - quotes: Scalar values (spot prices)
    - curves: Interest rate curves
    - vols: Volatility surfaces
    
    Production Notes
    ----------------
    - Snapshot extraction is lazy: curves/vols built on demand
    - Market objects are immutable snapshots
    - Safe for parallel processing
    """
    logger.info("=" * 70)
    logger.info("SECTION 1: Basic Snapshot Extraction")
    logger.info("=" * 70)
    
    # -------------------------------------------------------------------------
    # Extract a single snapshot
    # snapshot(time_idx, scenario_idx) -> Market
    # -------------------------------------------------------------------------
    market = dataset.snapshot(time_idx=0, scenario_idx=0)
    
    logger.info("")
    logger.info("Single snapshot extracted:")
    logger.info(f"  As-of date: {market.asof}")
    logger.info(f"  EUR/USD spot: {market.quote(EURUSD_SPOT):.4f}")
    logger.info(f"  USD 1Y rate: {market.curve(USD_CURVE).zero_rate(1.0):.4%}")
    logger.info(f"  ATM vol (1Y): {market.vol_surface(EURUSD_VOL).implied_vol(1.0, 1.085):.2%}")
    
    # -------------------------------------------------------------------------
    # Extract at different coordinates
    # Demonstrates the (time_idx, scenario_idx) coordinate system
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("Snapshots at different coordinates:")
    logger.info(f"{'[t, s]':<12} {'Date':<15} {'Spot':<12}")
    logger.info("-" * 39)
    
    for time_idx in [0, 5, 9]:
        for scenario_idx in [0, 25, 49]:
            m = dataset.snapshot(time_idx=time_idx, scenario_idx=scenario_idx)
            spot = m.quote(EURUSD_SPOT)
            logger.info(f"[{time_idx}, {scenario_idx}]     {m.asof:<15} {spot:<12.4f}")


# =============================================================================
# SECTION 2: Iterating Over Time (Fixed Scenario)
# =============================================================================

def demonstrate_time_iteration(dataset: MarketDataset) -> tuple[List[float], List[float]]:
    """
    Demonstrate iterating over time with a fixed scenario.
    
    Parameters
    ----------
    dataset : MarketDataset
        The dataset to iterate over.
    
    Returns
    -------
    tuple[List[float], List[float]]
        Spot prices and rates over time.
    
    Use Case: Historical Backtesting
    --------------------------------
    - scenario_idx=0 represents the realized market path
    - Loop over dates to compute daily P&L
    - This pattern is used for performance attribution
    
    Pattern:
        for time_idx in range(len(dates)):
            market = dataset.snapshot(time_idx, scenario_idx=0)
            # price portfolio, compute P&L
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 2: Iterating Over Time (Fixed Scenario)")
    logger.info("=" * 70)
    
    explanation = """
    Pattern: Loop over dates with a fixed scenario.
    Use case: Historical backtesting, P&L attribution.
    
        for time_idx in range(len(dates)):
            market = dataset.snapshot(time_idx, scenario_idx=0)
            # Compute P&L at this date
    """
    logger.info(explanation)
    
    scenario_idx = 0  # Use first scenario (or historical path)
    spots_over_time: List[float] = []
    rates_over_time: List[float] = []
    
    for time_idx in range(len(dataset.dates)):
        # Extract market at this date
        market = dataset.snapshot(time_idx=time_idx, scenario_idx=scenario_idx)
        
        # Extract market data for analysis
        spots_over_time.append(market.quote(EURUSD_SPOT))
        rates_over_time.append(market.curve(USD_CURVE).zero_rate(1.0))
    
    logger.info(f"Time series (scenario {scenario_idx}):")
    logger.info(f"{'Date':<12} {'Spot':<12} {'1Y Rate':<12}")
    logger.info("-" * 36)
    
    for i, date in enumerate(dataset.dates):
        logger.info(f"{date:<12} {spots_over_time[i]:<12.4f} {rates_over_time[i]:<12.4%}")
    
    return spots_over_time, rates_over_time


# =============================================================================
# SECTION 3: Iterating Over Scenarios (Fixed Date)
# =============================================================================

def demonstrate_scenario_iteration(dataset: MarketDataset) -> tuple[np.ndarray, np.ndarray]:
    """
    Demonstrate iterating over scenarios with a fixed date.
    
    Parameters
    ----------
    dataset : MarketDataset
        The dataset to iterate over.
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Spot and vol arrays across scenarios.
    
    Use Case: Monte Carlo VaR
    -------------------------
    - time_idx=horizon represents the risk horizon
    - Loop over scenarios to build P&L distribution
    - Compute VaR and ES from distribution
    
    Pattern:
        for scenario_idx in range(n_scenarios):
            market = dataset.snapshot(time_idx=horizon, scenario_idx=scenario_idx)
            # price portfolio, accumulate P&L
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 3: Iterating Over Scenarios (Fixed Date)")
    logger.info("=" * 70)
    
    explanation = """
    Pattern: Loop over scenarios with a fixed date.
    Use case: Monte Carlo VaR calculation.
    
        for scenario_idx in range(n_scenarios):
            market = dataset.snapshot(time_idx=horizon, scenario_idx=scenario_idx)
            # Compute P&L for this scenario
    """
    logger.info(explanation)
    
    time_idx = 5  # Middle of the dataset (risk horizon)
    spots_by_scenario: List[float] = []
    vols_by_scenario: List[float] = []
    
    for scenario_idx in range(dataset.n_scenarios):
        # Extract market for this scenario
        market = dataset.snapshot(time_idx=time_idx, scenario_idx=scenario_idx)
        
        # Collect market data
        spots_by_scenario.append(market.quote(EURUSD_SPOT))
        vols_by_scenario.append(market.vol_surface(EURUSD_VOL).implied_vol(1.0, 1.085))
    
    spots_array = np.array(spots_by_scenario)
    vols_array = np.array(vols_by_scenario)
    
    logger.info("")
    logger.info(f"Scenario statistics (date {dataset.dates[time_idx]}):")
    logger.info(f"  Spot range: [{spots_array.min():.4f}, {spots_array.max():.4f}]")
    logger.info(f"  Spot mean:  {spots_array.mean():.4f}")
    logger.info(f"  Spot std:   {spots_array.std():.4f}")
    logger.info(f"  Vol range:  [{vols_array.min():.2%}, {vols_array.max():.2%}]")
    logger.info(f"  Vol mean:   {vols_array.mean():.2%}")
    
    return spots_array, vols_array


# =============================================================================
# SECTION 4: Full Grid Iteration
# =============================================================================

def demonstrate_full_grid_iteration(dataset: MarketDataset) -> np.ndarray:
    """
    Demonstrate iterating over the full (time, scenario) grid.
    
    Parameters
    ----------
    dataset : MarketDataset
        The dataset to iterate over.
    
    Returns
    -------
    np.ndarray
        P&L grid [T, S].
    
    Use Case: Full Backtesting with Multiple Paths
    -----------------------------------------------
    - Each path represents a possible market evolution
    - Compute P&L at each (date, scenario) coordinate
    - Analyze path-dependent effects
    
    Production Notes
    ----------------
    - This is computationally expensive (T × S iterations)
    - Consider parallel processing for large grids
    - Direct panel access is faster for simple calculations
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 4: Full Grid Iteration")
    logger.info("=" * 70)
    
    explanation = """
    Pattern: Loop over all (date, scenario) combinations.
    Use case: Full backtesting with multiple paths, comprehensive P&L analysis.
    
        for t in range(len(dates)):
            for s in range(n_scenarios):
                market = dataset.snapshot(t, s)
                # Compute P&L
    """
    logger.info(explanation)
    
    # Compute P&L proxy (spot change from initial) for each path
    pnl_grid = np.zeros((len(dataset.dates), dataset.n_scenarios))
    initial_spot = 1.0850
    
    for t in range(len(dataset.dates)):
        for s in range(dataset.n_scenarios):
            market = dataset.snapshot(time_idx=t, scenario_idx=s)
            pnl_grid[t, s] = market.quote(EURUSD_SPOT) - initial_spot
    
    logger.info(f"P&L grid shape: {pnl_grid.shape}")
    logger.info("")
    logger.info("Mean P&L by date:")
    
    for t, date in enumerate(dataset.dates):
        mean_pnl = pnl_grid[t, :].mean() * 10000  # In pips
        std_pnl = pnl_grid[t, :].std() * 10000
        logger.info(f"  {date}: {mean_pnl:+.1f} pips (std: {std_pnl:.1f})")
    
    return pnl_grid


# =============================================================================
# SECTION 5: Efficient Batch Access
# =============================================================================

def demonstrate_efficient_access(dataset: MarketDataset) -> None:
    """
    Demonstrate efficient direct panel access vs snapshot extraction.
    
    Parameters
    ----------
    dataset : MarketDataset
        The dataset to access.
    
    Performance Comparison
    ----------------------
    - Direct panel access: O(1) array indexing
    - Snapshot extraction: Curve/vol reconstruction overhead
    
    Use direct access when:
    - Only need raw quote data (no curves/vols)
    - Processing many scenarios in bulk
    - Simple calculations (no pricing)
    
    Use snapshot extraction when:
    - Need curves/vols for pricing
    - Working with pricers that expect Market objects
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 5: Efficient Batch Access")
    logger.info("=" * 70)
    
    explanation = """
    For performance, access raw Panel data directly instead of
    extracting snapshots one by one.
    
    Direct access:  spot = dataset.panels[EURUSD_SPOT].data[t, s]
    Snapshot:       spot = dataset.snapshot(t, s).quote(EURUSD_SPOT)
    
    Direct access is faster for bulk operations on quote data.
    """
    logger.info(explanation)
    
    # -------------------------------------------------------------------------
    # Direct panel access
    # -------------------------------------------------------------------------
    spot_panel = dataset.panels[EURUSD_SPOT]
    
    logger.info(f"Direct panel access:")
    logger.info(f"  Panel shape: {spot_panel.data.shape}")
    logger.info(f"  All spots at t=0: {spot_panel.data[0, :5]}... (first 5 scenarios)")
    
    # -------------------------------------------------------------------------
    # Verify equivalence
    # -------------------------------------------------------------------------
    for s in range(5):
        market = dataset.snapshot(time_idx=0, scenario_idx=s)
        direct_value = spot_panel.data[0, s]
        snapshot_value = market.quote(EURUSD_SPOT)
        assert abs(snapshot_value - direct_value) < 1e-10, "Mismatch!"
    
    logger.info(f"  Verified: snapshot extraction matches direct access ✓")


# =============================================================================
# SECTION 6: Visualization
# =============================================================================

def visualize_snapshots(dataset: MarketDataset) -> None:
    """
    Create comprehensive visualizations of snapshot data.
    
    Parameters
    ----------
    dataset : MarketDataset
        The dataset to visualize.
    
    Four plots:
    1. Time series paths (fan chart)
    2. Terminal distribution (histogram)
    3. Heatmap of all scenarios
    4. Terminal P&L distribution
    """
    if not MATPLOTLIB_AVAILABLE or not ENABLE_PLOTTING:
        logger.info("Skipping plots (matplotlib not available or disabled)")
        return
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 6: Visualization")
    logger.info("=" * 70)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Get raw data for efficient access
    spot_data = dataset.panels[EURUSD_SPOT].data
    
    # -------------------------------------------------------------------------
    # Plot 1: Time series paths (fan chart)
    # Shows scenario uncertainty over time
    # -------------------------------------------------------------------------
    ax = axes[0, 0]
    for s in range(min(30, dataset.n_scenarios)):
        ax.plot(
            spot_data[:, s],
            alpha=0.3,
            linewidth=0.8,
            color='#2E86AB',
        )
    ax.plot(
        spot_data.mean(axis=1),
        'k-',
        linewidth=2,
        label='Mean',
    )
    ax.fill_between(
        range(len(dataset.dates)),
        np.percentile(spot_data, 5, axis=1),
        np.percentile(spot_data, 95, axis=1),
        alpha=0.2,
        color='gray',
        label='5-95% CI',
    )
    ax.set_xticks(range(len(dataset.dates)))
    ax.set_xticklabels([d[-2:] for d in dataset.dates])
    ax.set_xlabel('Day')
    ax.set_ylabel('EUR/USD')
    ax.set_title('Spot Paths Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 2: Terminal distribution
    # Used for VaR calculation
    # -------------------------------------------------------------------------
    ax = axes[0, 1]
    terminal_spots = spot_data[-1, :]
    ax.hist(
        terminal_spots,
        bins=20,
        color='#8B5CF6',
        alpha=0.7,
        edgecolor='white',
        density=True,
    )
    ax.axvline(
        terminal_spots.mean(),
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Mean: {terminal_spots.mean():.4f}',
    )
    ax.axvline(
        np.percentile(terminal_spots, 5),
        color='orange',
        linestyle=':',
        linewidth=2,
        label='5% VaR',
    )
    ax.set_xlabel('EUR/USD')
    ax.set_ylabel('Density')
    ax.set_title('Terminal Spot Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 3: Heatmap of all paths
    # Visual overview of scenario evolution
    # -------------------------------------------------------------------------
    ax = axes[1, 0]
    im = ax.imshow(
        spot_data.T,
        aspect='auto',
        cmap='RdYlBu_r',
        extent=[0, len(dataset.dates)-1, 0, dataset.n_scenarios],
    )
    ax.set_xlabel('Day')
    ax.set_ylabel('Scenario')
    ax.set_title('Spot Heatmap (all scenarios)')
    plt.colorbar(im, ax=ax, label='EUR/USD')
    
    # -------------------------------------------------------------------------
    # Plot 4: Terminal P&L distribution
    # Core risk metric
    # -------------------------------------------------------------------------
    ax = axes[1, 1]
    terminal_pnl = (terminal_spots - spot_data[0, :]) * 10000  # In pips
    ax.hist(
        terminal_pnl,
        bins=20,
        color='#10B981',
        alpha=0.7,
        edgecolor='white',
    )
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.axvline(
        terminal_pnl.mean(),
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Mean: {terminal_pnl.mean():+.1f} pips',
    )
    ax.set_xlabel('P&L (pips)')
    ax.set_ylabel('Frequency')
    ax.set_title('Terminal P&L Distribution')
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
    │  1. Snapshot Extraction:                                            │
    │     market = dataset.snapshot(time_idx, scenario_idx)               │
    │     Returns a complete Market object for pricing                    │
    │                                                                      │
    │  2. Iteration Patterns:                                             │
    │     - Fixed scenario, vary time → Backtesting                       │
    │     - Fixed time, vary scenario → Monte Carlo VaR                   │
    │     - Full grid → Comprehensive analysis                            │
    │                                                                      │
    │  3. Efficient Access:                                               │
    │     - Direct panel: dataset.panels[id].data[t, s]                   │
    │     - Snapshot: dataset.snapshot(t, s).quote(id)                    │
    │     - Use direct access for bulk quote operations                   │
    │                                                                      │
    │  4. Statistics:                                                     │
    │     - Mean, std, percentiles across scenarios                       │
    │     - VaR from terminal distribution percentile                     │
    │                                                                      │
    │  5. Coordinates:                                                    │
    │     - time_idx: 0 to len(dates)-1                                   │
    │     - scenario_idx: 0 to n_scenarios-1                              │
    │                                                                      │
    │  NEXT: See 06_scenario_shocks.py for applying market shocks         │
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
        # Create sample dataset
        logger.info("Creating sample dataset...")
        dataset = create_sample_dataset(n_dates=10, n_scenarios=50)
        
        # Section 1: Basic Extraction
        demonstrate_basic_extraction(dataset)
        
        # Section 2: Time Iteration
        spots, rates = demonstrate_time_iteration(dataset)
        
        # Section 3: Scenario Iteration
        spots_array, vols_array = demonstrate_scenario_iteration(dataset)
        
        # Section 4: Full Grid
        pnl_grid = demonstrate_full_grid_iteration(dataset)
        
        # Section 5: Efficient Access
        demonstrate_efficient_access(dataset)
        
        # Section 6: Visualization
        visualize_snapshots(dataset)
        
        # Summary
        print_summary()
        
        logger.info("Example completed successfully!")
        
    except Exception as e:
        logger.exception(f"Example failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Market Snapshots Example",
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

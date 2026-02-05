#!/usr/bin/env python3
"""
===============================================================================
Time Series Datasets: Multi-Day Market Data with Scenarios
===============================================================================

This example covers the MarketDataset structure - the central data container
for multi-day, multi-scenario market data used in pricing, risk, and backtesting.

Learning Objectives
-------------------
1. **MarketDataset**: Understand the container for time series market data
2. **Panel**: Learn the N-dimensional array structure with named axes
3. **Factories**: See how curves and vol surfaces are reconstructed
4. **Snapshot Extraction**: Extract Market objects for pricing

Data Structure Overview
-----------------------
MarketDataset stores data in the following structure:

    ┌─────────────────────────────────────────────────────────────────────┐
    │ MarketDataset                                                        │
    ├─────────────────────────────────────────────────────────────────────┤
    │ dates: ["2026-01-20", "2026-01-21", ...]   # T dates                │
    │ n_scenarios: 100                            # S scenarios per date   │
    │                                                                      │
    │ panels: {                                   # Quote data             │
    │   MarketId("FX","SPOT","EURUSD"): Panel[T,S]                        │
    │ }                                                                    │
    │                                                                      │
    │ curve_params: {                             # Curve parameters       │
    │   MarketId("IR","CURVE","USD"): Panel[T,S,K,2]  # (tenor, rate)     │
    │ }                                                                    │
    │ curve_factories: { ... }                    # Curve builders         │
    │                                                                      │
    │ vol_params: {                               # Vol surface parameters │
    │   MarketId("FX","VOL","EURUSD"): Panel[T,S,E,K]  # (expiry, strike) │
    │ }                                                                    │
    │ vol_factories: { ... }                      # Surface builders       │
    └─────────────────────────────────────────────────────────────────────┘

Production Context
------------------
At a hedge fund, MarketDataset is used for:
- Historical backtesting: 1 scenario per date (actual market data)
- Monte Carlo VaR: Many scenarios per date (simulated risk factors)
- Stress testing: Specific extreme scenarios
- P&L attribution: Daily market data for performance analysis

Prerequisites
-------------
- Example 01: Market IDs and Quotes
- Example 02: Curves and Term Structures
- Example 03: Volatility Surfaces

Run This Example
----------------
    cd /path/to/QuantStrata
    PYTHONPATH=. python examples/fundamentals/04_timeseries_datasets.py

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
# Path setup: Ensure imports work when running as script
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# -----------------------------------------------------------------------------
# QuantStrata imports
# -----------------------------------------------------------------------------
from src.marketdata.core.ids import MarketId          # Unique identifiers
from src.marketdata.core.panel import Panel           # N-dimensional arrays
from src.marketdata.core.dataset import MarketDataset # Time series container
from src.marketdata.curves.factory import ZeroRateCurveFactory  # Curve builder
from src.marketdata.surfaces.factory import GridVolFactory      # Vol surface builder

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

# Standard MarketIds used throughout this example
EURUSD_SPOT_ID = MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD")
USD_CURVE_ID = MarketId(asset_class="IR", mkt_type="CURVE", name="USD_OIS")
EURUSD_VOL_ID = MarketId(asset_class="FX", mkt_type="VOL", name="EURUSD")


# =============================================================================
# SECTION 1: Understanding MarketDataset
# =============================================================================

def explain_market_dataset() -> None:
    """
    Explain the purpose and structure of MarketDataset.
    
    MarketDataset is the central data container for:
    - Time series of market data (multiple dates)
    - Multiple scenarios per date (for Monte Carlo or stress testing)
    - Curve and vol surface parameters with reconstruction factories
    
    The key method is:
        dataset.snapshot(time_idx, scenario_idx) -> Market
    
    This extracts a single Market object for pricing at a specific coordinate.
    
    Production Notes
    ----------------
    - MarketDataset supports both historical (1 scenario) and simulated data
    - Efficient storage using numpy arrays with named axes (Panel)
    - Lazy reconstruction of curves/surfaces on snapshot extraction
    """
    logger.info("=" * 70)
    logger.info("SECTION 1: Understanding MarketDataset")
    logger.info("=" * 70)
    
    explanation = """
    MarketDataset is a container for multi-day, multi-scenario market data.
    
    Structure:
    ┌────────────────────────────────────────────────────────────────┐
    │ dates:         List of date strings ["2026-01-01", ...]       │
    │ n_scenarios:   Number of scenarios per date (1 for historical) │
    │ panels:        Quote data stored in Panel objects              │
    │ curve_params:  + curve_factories → For reconstructing curves   │
    │ vol_params:    + vol_factories → For reconstructing surfaces   │
    └────────────────────────────────────────────────────────────────┘
    
    Key method:
        market = dataset.snapshot(time_idx, scenario_idx) -> Market
      
    This extracts a single Market object for pricing at (date, scenario).
    """
    logger.info(explanation)


# =============================================================================
# SECTION 2: Panels - Multi-Dimensional Data Storage
# =============================================================================

def demonstrate_panels() -> tuple[Panel, Panel]:
    """
    Demonstrate the Panel class for storing multi-dimensional data.
    
    Returns
    -------
    tuple[Panel, Panel]
        1D panel (time series) and 2D panel (with scenarios).
    
    Panel is a thin wrapper around numpy arrays with:
    - Named axes for clarity (e.g., "time", "scenario", "tenor")
    - Shape validation
    - Integration with MarketDataset
    
    Common Panel Shapes
    -------------------
    [T]           - Single value per date (historical only)
    [T, S]        - Value per (date, scenario) - quotes, spot prices
    [T, S, K, 2]  - Curve parameters (K tenors, 2 = tenor + rate)
    [T, S, E, K]  - Vol surface (E expiries, K strikes)
    
    Production Notes
    ----------------
    - Axis names are critical for debugging and validation
    - Use consistent naming across the codebase
    - Panel.data provides direct numpy array access
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 2: Panels - Multi-Dimensional Data Storage")
    logger.info("=" * 70)
    
    explanation = """
    Panel is a container for numpy arrays with named axes.
    
    Common shapes:
      [T]           Single value per date (no scenarios)
      [T, S]        Value per date and scenario
      [T, S, K, 2]  Curve params: K tenors, 2 columns (tenor, rate)
      [T, S, E, K]  Vol surface: E expiries, K strikes
    
    Axis names describe the dimensions:
      ("time",)                     Time series only
      ("time", "scenario")          With scenarios
      ("time", "scenario", "tenor") Curve parameters
    """
    logger.info(explanation)
    
    # -------------------------------------------------------------------------
    # Create a 1D panel (time series of spot prices, no scenarios)
    # -------------------------------------------------------------------------
    dates = ["2026-01-20", "2026-01-21", "2026-01-22", "2026-01-23", "2026-01-24"]
    n_dates = len(dates)
    
    # Simulated spot prices (deterministic for this example)
    spot_values = np.array([1.0850, 1.0875, 1.0820, 1.0890, 1.0860])
    
    spot_panel_1d = Panel(
        data=spot_values,                # 1D numpy array
        axis_names=("time",),            # Single axis: time
    )
    
    logger.info("")
    logger.info("1D Panel (time series only):")
    logger.info(f"  Shape: {spot_panel_1d.data.shape}")
    logger.info(f"  Axis names: {spot_panel_1d.axis_names}")
    logger.info(f"  Data: {spot_panel_1d.data}")
    
    # -------------------------------------------------------------------------
    # Create a 2D panel (time series with scenarios)
    # This is the standard format for Monte Carlo / VaR
    # -------------------------------------------------------------------------
    n_scenarios = 100
    np.random.seed(42)  # Reproducibility
    
    # Generate scenarios: base values + random shocks
    spot_base = spot_values.reshape(-1, 1)              # [T, 1] - broadcast ready
    spot_shocks = np.random.randn(n_dates, n_scenarios) * 0.005  # Random shocks
    spot_scenarios = spot_base + spot_shocks            # [T, S] via broadcasting
    
    spot_panel_2d = Panel(
        data=spot_scenarios,                # 2D numpy array [T, S]
        axis_names=("time", "scenario"),    # Two axes
    )
    
    logger.info("")
    logger.info("2D Panel (with scenarios):")
    logger.info(f"  Shape: {spot_panel_2d.data.shape}")
    logger.info(f"  Axis names: {spot_panel_2d.axis_names}")
    logger.info(f"  Day 0, Scenario 0:  {spot_panel_2d.data[0, 0]:.4f}")
    logger.info(f"  Day 0, Scenario 99: {spot_panel_2d.data[0, 99]:.4f}")
    
    return spot_panel_1d, spot_panel_2d


# =============================================================================
# SECTION 3: Building a Simple MarketDataset
# =============================================================================

def build_simple_dataset(spot_panel: Panel) -> MarketDataset:
    """
    Build a simple MarketDataset with only quote panels.
    
    Parameters
    ----------
    spot_panel : Panel
        The spot price panel [T, S].
    
    Returns
    -------
    MarketDataset
        A basic dataset with quotes only (no curves/vols).
    
    Production Notes
    ----------------
    - This is the simplest possible dataset
    - curves={}, vols={} indicates no term structures
    - Useful for equity spot modeling
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 3: Building a Simple MarketDataset")
    logger.info("=" * 70)
    
    dates = ["2026-01-20", "2026-01-21", "2026-01-22", "2026-01-23", "2026-01-24"]
    n_scenarios = spot_panel.data.shape[1]  # Extract from panel shape
    
    # Build the dataset with quotes only
    dataset = MarketDataset(
        dates=dates,
        n_scenarios=n_scenarios,
        panels={EURUSD_SPOT_ID: spot_panel},  # Quote panels
        curve_params={},                       # No curves
        curve_factories={},
        vol_params={},                         # No vol surfaces
        vol_factories={},
        meta={                                 # Optional metadata
            "source": "Synthetic data",
            "description": "Simple example with quotes only",
        },
    )
    
    logger.info("MarketDataset created:")
    logger.info(f"  Number of dates: {len(dataset.dates)}")
    logger.info(f"  Number of scenarios: {dataset.n_scenarios}")
    logger.info(f"  Date range: {dataset.dates[0]} to {dataset.dates[-1]}")
    logger.info(f"  Quote panels: {list(dataset.panels.keys())}")
    
    # -------------------------------------------------------------------------
    # Extract snapshots at different coordinates
    # snapshot(time_idx, scenario_idx) returns a Market object
    # -------------------------------------------------------------------------
    market_0_0 = dataset.snapshot(time_idx=0, scenario_idx=0)
    market_0_50 = dataset.snapshot(time_idx=0, scenario_idx=50)
    market_2_0 = dataset.snapshot(time_idx=2, scenario_idx=0)
    
    logger.info("")
    logger.info("Snapshot extraction:")
    logger.info(f"  [t=0, s=0]:  date={market_0_0.asof}, spot={market_0_0.quote(EURUSD_SPOT_ID):.4f}")
    logger.info(f"  [t=0, s=50]: date={market_0_50.asof}, spot={market_0_50.quote(EURUSD_SPOT_ID):.4f}")
    logger.info(f"  [t=2, s=0]:  date={market_2_0.asof}, spot={market_2_0.quote(EURUSD_SPOT_ID):.4f}")
    
    return dataset


# =============================================================================
# SECTION 4: Adding Curves to the Dataset
# =============================================================================

def add_curves_to_dataset(
    dates: List[str],
    n_scenarios: int,
    spot_panel: Panel,
) -> tuple[MarketDataset, Panel]:
    """
    Add interest rate curves to the dataset.
    
    Parameters
    ----------
    dates : List[str]
        List of date strings.
    n_scenarios : int
        Number of scenarios per date.
    spot_panel : Panel
        The spot price panel.
    
    Returns
    -------
    tuple[MarketDataset, Panel]
        Dataset with curves and the curve panel.
    
    Curve Parameter Structure
    -------------------------
    Shape: [T, S, K, 2]
      T = number of dates
      S = number of scenarios
      K = number of tenor points
      2 = (tenor_value, zero_rate)
    
    The factory reconstructs ZeroRateCurve from this format.
    
    Production Notes
    ----------------
    - Multiple curves per currency in production (OIS, LIBOR, etc.)
    - Curve shocks often applied as parallel shifts or key rate bumps
    - Consider PCA decomposition for correlated rate scenarios
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 4: Adding Curves to the Dataset")
    logger.info("=" * 70)
    
    n_dates = len(dates)
    
    # -------------------------------------------------------------------------
    # Define curve tenor grid
    # Standard tenors for interest rate curves
    # -------------------------------------------------------------------------
    tenors = np.array([0.25, 0.5, 1.0, 2.0, 5.0])  # 3M, 6M, 1Y, 2Y, 5Y
    n_tenors = len(tenors)
    
    # Base zero rates (upward sloping normal curve)
    base_rates = np.array([0.045, 0.048, 0.050, 0.053, 0.057])
    
    # -------------------------------------------------------------------------
    # Generate curve scenarios
    # Shape: [T, S, K, 2] where 2 = (tenor, zero_rate)
    # -------------------------------------------------------------------------
    np.random.seed(42)  # Reproducibility
    
    curve_params = np.zeros((n_dates, n_scenarios, n_tenors, 2))
    
    for t in range(n_dates):
        for s in range(n_scenarios):
            # Parallel shift scenario (simplified)
            # Production would use PCA or Hull-White simulation
            rate_shock = np.random.randn() * 0.002  # ±20bp shock
            
            # Apply shock and small time drift
            rates = base_rates + rate_shock + t * 0.001
            
            # Store (tenor, rate) pairs
            curve_params[t, s, :, 0] = tenors       # Column 0: tenors
            curve_params[t, s, :, 1] = rates        # Column 1: zero rates
    
    curve_panel = Panel(
        data=curve_params,
        axis_names=("time", "scenario", "tenor", "cols"),
    )
    
    logger.info("Curve panel created:")
    logger.info(f"  Shape: {curve_panel.data.shape}")
    logger.info(f"  Axis names: {curve_panel.axis_names}")
    logger.info(f"  Interpretation: [dates, scenarios, tenors, (tenor,rate)]")
    
    # -------------------------------------------------------------------------
    # Create factory for curve reconstruction
    # ZeroRateCurveFactory builds ZeroRateCurve from (tenor, rate) pairs
    # -------------------------------------------------------------------------
    curve_factory = ZeroRateCurveFactory()
    
    # Build dataset with curves
    dataset_with_curves = MarketDataset(
        dates=dates,
        n_scenarios=n_scenarios,
        panels={EURUSD_SPOT_ID: spot_panel},
        curve_params={USD_CURVE_ID: curve_panel},      # Curve parameters
        curve_factories={USD_CURVE_ID: curve_factory}, # How to build curves
        vol_params={},
        vol_factories={},
    )
    
    # -------------------------------------------------------------------------
    # Extract and verify curve from snapshot
    # -------------------------------------------------------------------------
    market = dataset_with_curves.snapshot(time_idx=0, scenario_idx=0)
    curve = market.curve(USD_CURVE_ID)
    
    logger.info("")
    logger.info("Extracted curve from dataset:")
    logger.info(f"  5Y discount factor: {curve.df(5.0):.6f}")
    logger.info(f"  5Y zero rate: {curve.zero_rate(5.0):.4%}")
    
    return dataset_with_curves, curve_panel


# =============================================================================
# SECTION 5: Adding Vol Surfaces to the Dataset
# =============================================================================

def add_vol_surfaces(
    dates: List[str],
    n_scenarios: int,
    spot_panel: Panel,
    curve_panel: Panel,
) -> tuple[MarketDataset, Panel]:
    """
    Add volatility surfaces to the dataset.
    
    Parameters
    ----------
    dates : List[str]
        List of date strings.
    n_scenarios : int
        Number of scenarios per date.
    spot_panel : Panel
        The spot price panel.
    curve_panel : Panel
        The curve parameter panel.
    
    Returns
    -------
    tuple[MarketDataset, Panel]
        Complete dataset and vol panel.
    
    Vol Surface Parameter Structure
    -------------------------------
    Shape: [T, S, E, K]
      T = number of dates
      S = number of scenarios
      E = number of expiry tenors
      K = number of strikes
    
    The factory uses fixed expiry/strike grids defined at creation.
    
    Production Notes
    ----------------
    - Vol shocks often correlated with spot moves (leverage effect)
    - SABR or SVI parameterization more common than grid storage
    - Vol-of-vol generates fat-tailed PnL distributions
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 5: Adding Vol Surfaces to the Dataset")
    logger.info("=" * 70)
    
    n_dates = len(dates)
    
    # -------------------------------------------------------------------------
    # Define vol surface grid
    # -------------------------------------------------------------------------
    vol_expiries = np.array([0.25, 0.5, 1.0])         # 3M, 6M, 1Y
    vol_strikes = np.linspace(0.95, 1.20, 11)         # Strike range
    n_exp = len(vol_expiries)
    n_strikes = len(vol_strikes)
    
    # -------------------------------------------------------------------------
    # Helper to generate smile-shaped vol surface
    # -------------------------------------------------------------------------
    def make_smile_vol(
        exp: float,
        strike: float,
        spot: float = 1.0850,
        atm: float = 0.08,
        skew: float = -0.15,
        convex: float = 0.10,
    ) -> float:
        """Generate IV with smile characteristics."""
        m = np.log(strike / spot)  # Log-moneyness
        return atm + skew * m + convex * m**2
    
    # Base vol surface
    base_vols = np.zeros((n_exp, n_strikes))
    for i, exp in enumerate(vol_expiries):
        for j, k in enumerate(vol_strikes):
            base_vols[i, j] = make_smile_vol(exp, k)
    
    # -------------------------------------------------------------------------
    # Generate vol scenarios
    # Shape: [T, S, E, K]
    # -------------------------------------------------------------------------
    np.random.seed(123)  # Different seed for vol
    
    vol_params = np.zeros((n_dates, n_scenarios, n_exp, n_strikes))
    
    for t in range(n_dates):
        for s in range(n_scenarios):
            # Vol-of-vol shock (parallel shift to surface)
            vol_shock = np.random.randn() * 0.005  # ±0.5% vol shock
            vol_params[t, s, :, :] = base_vols + vol_shock
    
    vol_panel = Panel(
        data=vol_params,
        axis_names=("time", "scenario", "expiry", "strike"),
    )
    
    logger.info("Vol panel created:")
    logger.info(f"  Shape: {vol_panel.data.shape}")
    logger.info(f"  Axis names: {vol_panel.axis_names}")
    logger.info(f"  Interpretation: [dates, scenarios, expiries, strikes]")
    
    # -------------------------------------------------------------------------
    # Create factory for vol surface reconstruction
    # GridVolFactory needs the expiry/strike grid at initialization
    # -------------------------------------------------------------------------
    vol_factory = GridVolFactory(
        expiries=vol_expiries,
        strikes=vol_strikes,
    )
    
    # Reconstruct curve factory
    curve_factory = ZeroRateCurveFactory()
    
    return vol_panel, vol_factory


# =============================================================================
# SECTION 6: Complete Dataset with All Components
# =============================================================================

def build_complete_dataset() -> MarketDataset:
    """
    Build a complete MarketDataset with quotes, curves, and vol surfaces.
    
    Returns
    -------
    MarketDataset
        Production-ready dataset structure.
    
    This is the standard format for:
    - FX option pricing
    - VaR and scenario analysis
    - Greeks computation
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 6: Complete Dataset with All Components")
    logger.info("=" * 70)
    
    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    dates = ["2026-01-20", "2026-01-21", "2026-01-22", "2026-01-23", "2026-01-24"]
    n_dates = len(dates)
    n_scenarios = 100
    
    np.random.seed(42)
    
    # -------------------------------------------------------------------------
    # Spot panel [T, S]
    # -------------------------------------------------------------------------
    spot_base = np.array([1.0850, 1.0875, 1.0820, 1.0890, 1.0860]).reshape(-1, 1)
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
            curve_params[t, s, :, 1] = base_rates + rate_shock + t * 0.001
    
    curve_panel = Panel(
        data=curve_params,
        axis_names=("time", "scenario", "tenor", "cols"),
    )
    curve_factory = ZeroRateCurveFactory()
    
    # -------------------------------------------------------------------------
    # Vol panel [T, S, E, K]
    # -------------------------------------------------------------------------
    vol_expiries = np.array([0.25, 0.5, 1.0])
    vol_strikes = np.linspace(0.95, 1.20, 11)
    base_vol = 0.08
    
    vol_params = np.zeros((n_dates, n_scenarios, len(vol_expiries), len(vol_strikes)))
    for t in range(n_dates):
        for s in range(n_scenarios):
            vol_shock = np.random.randn() * 0.005
            vol_params[t, s, :, :] = base_vol + vol_shock
    
    vol_panel = Panel(
        data=vol_params,
        axis_names=("time", "scenario", "expiry", "strike"),
    )
    vol_factory = GridVolFactory(expiries=vol_expiries, strikes=vol_strikes)
    
    # -------------------------------------------------------------------------
    # Build complete dataset
    # -------------------------------------------------------------------------
    dataset = MarketDataset(
        dates=dates,
        n_scenarios=n_scenarios,
        panels={EURUSD_SPOT_ID: spot_panel},
        curve_params={USD_CURVE_ID: curve_panel},
        curve_factories={USD_CURVE_ID: curve_factory},
        vol_params={EURUSD_VOL_ID: vol_panel},
        vol_factories={EURUSD_VOL_ID: vol_factory},
        meta={
            "source": "Synthetic generator",
            "description": "Complete FX dataset with curves and vols",
            "ccy_pair": "EURUSD",
        },
    )
    
    logger.info("Complete MarketDataset:")
    logger.info(f"  Dates: {len(dataset.dates)}")
    logger.info(f"  Scenarios: {dataset.n_scenarios}")
    logger.info(f"  Quotes: {list(dataset.panels.keys())}")
    logger.info(f"  Curves: {list(dataset.curve_params.keys())}")
    logger.info(f"  Vols: {list(dataset.vol_params.keys())}")
    logger.info(f"  Meta: {dataset.meta}")
    
    # -------------------------------------------------------------------------
    # Extract complete market snapshot
    # -------------------------------------------------------------------------
    market = dataset.snapshot(time_idx=2, scenario_idx=42)
    
    logger.info("")
    logger.info("Full snapshot (day 2, scenario 42):")
    logger.info(f"  As-of: {market.asof}")
    logger.info(f"  EUR/USD spot: {market.quote(EURUSD_SPOT_ID):.4f}")
    logger.info(f"  USD 1Y rate: {market.curve(USD_CURVE_ID).zero_rate(1.0):.4%}")
    logger.info(f"  ATM 1Y vol: {market.vol_surface(EURUSD_VOL_ID).implied_vol(1.0, 1.085):.2%}")
    
    return dataset


# =============================================================================
# SECTION 7: Visualizing the Dataset
# =============================================================================

def visualize_dataset(dataset: MarketDataset) -> None:
    """
    Create visualizations of the dataset.
    
    Parameters
    ----------
    dataset : MarketDataset
        The dataset to visualize.
    
    Three plots:
    1. Spot scenarios over time (path fan)
    2. Rate distribution at a single date
    3. Vol distribution at a single date
    """
    if not MATPLOTLIB_AVAILABLE or not ENABLE_PLOTTING:
        logger.info("Skipping plots (matplotlib not available or disabled)")
        return
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 7: Visualizing the Dataset")
    logger.info("=" * 70)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Get raw data
    spot_data = dataset.panels[EURUSD_SPOT_ID].data
    curve_data = dataset.curve_params[USD_CURVE_ID].data
    vol_data = dataset.vol_params[EURUSD_VOL_ID].data
    
    n_dates = len(dataset.dates)
    n_scenarios = dataset.n_scenarios
    
    # -------------------------------------------------------------------------
    # Plot 1: Spot scenarios over time (path fan)
    # -------------------------------------------------------------------------
    ax = axes[0]
    for s in range(min(20, n_scenarios)):
        ax.plot(
            range(n_dates),
            spot_data[:, s],
            alpha=0.3,
            linewidth=0.8,
            color='#2E86AB',
        )
    ax.plot(
        range(n_dates),
        spot_data.mean(axis=1),
        'k-',
        linewidth=2,
        label='Mean',
    )
    ax.set_xticks(range(n_dates))
    ax.set_xticklabels([d[-5:] for d in dataset.dates])  # MM-DD
    ax.set_xlabel('Date')
    ax.set_ylabel('EUR/USD Spot')
    ax.set_title('Spot Scenarios Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 2: Rate distribution at a single date
    # Extract 1Y rate (index 2) for all scenarios at day 0
    # -------------------------------------------------------------------------
    ax = axes[1]
    rates_day0 = curve_data[0, :, 2, 1]  # [t=0, all scenarios, tenor=2, column=1=rate]
    ax.hist(
        rates_day0 * 100,
        bins=30,
        color='#2E86AB',
        alpha=0.7,
        edgecolor='white',
    )
    ax.axvline(
        rates_day0.mean() * 100,
        color='red',
        linestyle='--',
        linewidth=2,
        label='Mean',
    )
    ax.set_xlabel('1Y Zero Rate (%)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Rate Distribution ({dataset.dates[0]})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 3: Vol distribution at a single date
    # Extract ATM vol (expiry=1, strike index=5) for all scenarios at day 0
    # -------------------------------------------------------------------------
    ax = axes[2]
    vols_day0 = vol_data[0, :, 1, 5]  # [t=0, all scenarios, expiry=1, strike=5]
    ax.hist(
        vols_day0 * 100,
        bins=30,
        color='#8B5CF6',
        alpha=0.7,
        edgecolor='white',
    )
    ax.axvline(
        vols_day0.mean() * 100,
        color='red',
        linestyle='--',
        linewidth=2,
        label='Mean',
    )
    ax.set_xlabel('ATM Implied Vol (%)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Vol Distribution ({dataset.dates[0]})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show(block=True)
    
    logger.info("Dataset visualization complete")


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
    │  1. MarketDataset:                                                  │
    │     - Container for multi-day, multi-scenario market data           │
    │     - Produces Market via .snapshot(time_idx, scenario_idx)         │
    │                                                                      │
    │  2. Panel:                                                          │
    │     - N-dimensional array with named axes                           │
    │     - Shapes: [T], [T,S], [T,S,K], [T,S,K1,K2]                      │
    │     - Axis names describe dimensions                                │
    │                                                                      │
    │  3. Factories:                                                      │
    │     - ZeroRateCurveFactory: Builds curves from (tenor, rate)        │
    │     - GridVolFactory: Builds surfaces from grids                    │
    │                                                                      │
    │  4. Dataset Components:                                             │
    │     - panels: Quote data (spot prices)                              │
    │     - curve_params + curve_factories: Yield curves                  │
    │     - vol_params + vol_factories: Vol surfaces                      │
    │                                                                      │
    │  5. Use Cases:                                                      │
    │     - Historical backtesting (1 scenario per date)                  │
    │     - Monte Carlo VaR (many scenarios per date)                     │
    │     - Stress testing (specific extreme scenarios)                   │
    │                                                                      │
    │  NEXT: See 05_market_snapshot.py for snapshot patterns              │
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
        # Section 1: Explain MarketDataset
        explain_market_dataset()
        
        # Section 2: Demonstrate Panels
        spot_panel_1d, spot_panel_2d = demonstrate_panels()
        
        # Section 3: Build Simple Dataset
        simple_dataset = build_simple_dataset(spot_panel_2d)
        
        # Section 4-6: Build Complete Dataset
        complete_dataset = build_complete_dataset()
        
        # Section 7: Visualization
        visualize_dataset(complete_dataset)
        
        # Summary
        print_summary()
        
        logger.info("Example completed successfully!")
        
    except Exception as e:
        logger.exception(f"Example failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Time Series Datasets Example",
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

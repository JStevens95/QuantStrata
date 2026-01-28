#!/usr/bin/env python3
"""
Time Series Datasets: Multi-Day Market Data with Scenarios

This example covers the MarketDataset structure for handling:
- Time series of market data (multiple dates)
- Multiple scenarios per date (for Monte Carlo or stress testing)
- Panels for storing multi-dimensional data
- Factories for reconstructing curves and vol surfaces

MarketDataset is the bridge between raw data and pricing snapshots.

Author: QuantStrata Team
"""

import sys
sys.path.insert(0, '../..')

import numpy as np
import matplotlib.pyplot as plt

from src.marketdata.core.ids import MarketId
from src.marketdata.core.panel import Panel
from src.marketdata.core.dataset import MarketDataset
from src.marketdata.curves.factory import ZeroRateCurveFactory
from src.marketdata.surfaces.factory import GridVolFactory

# Plot configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (12, 5),
    'font.size': 11,
    'axes.titlesize': 13,
    'lines.linewidth': 2,
})

# =============================================================================
# 1. Understanding MarketDataset
# =============================================================================

# save figures
SAVE_FIGURES = False

print("=" * 70)
print("1. Understanding MarketDataset")
print("=" * 70)

print("""
MarketDataset is a container for multi-day, multi-scenario market data.

Structure:
  - dates: List of date strings ["2026-01-01", "2026-01-02", ...]
  - n_scenarios: Number of scenarios per date (1 for historical, >1 for MC)
  - panels: Quote data stored in Panel objects
  - curve_params + curve_factories: For reconstructing curves
  - vol_params + vol_factories: For reconstructing vol surfaces

Key method:
  dataset.snapshot(time_idx, scenario_idx) -> Market
  
This extracts a single Market object for pricing at a specific (date, scenario).
""")

# =============================================================================
# 2. Panels: Multi-Dimensional Data Storage
# =============================================================================

print("\n" + "=" * 70)
print("2. Panels: Multi-Dimensional Data Storage")
print("=" * 70)

print("""
Panel is a container for numpy arrays with named axes.

Common shapes:
  [T]       - Single value per date (no scenarios)
  [T, S]    - Value per date and scenario
  [T, S, K] - Matrix per date and scenario (e.g., vol grid)

Axis names describe the dimensions:
  ("time",)                    - Time series
  ("time", "scenario")         - With scenarios
  ("time", "scenario", "tenor")- Curve parameters
""")

# Create a simple 1D panel (time series of spot prices)
dates = ["2026-01-20", "2026-01-21", "2026-01-22", "2026-01-23", "2026-01-24"]
n_dates = len(dates)

# Simulated spot prices
spot_values = np.array([1.0850, 1.0875, 1.0820, 1.0890, 1.0860])

spot_panel = Panel(
    data=spot_values,
    axis_names=("time",),
)

print(f"\n1D Panel (time series):")
print(f"  Shape: {spot_panel.data.shape}")
print(f"  Axis names: {spot_panel.axis_names}")
print(f"  Data: {spot_panel.data}")

# Create a 2D panel (time series with scenarios)
n_scenarios = 100
np.random.seed(42)

# Generate scenarios: base + random shocks
spot_base = spot_values.reshape(-1, 1)  # [T, 1]
spot_shocks = np.random.randn(n_dates, n_scenarios) * 0.005  # Random shocks
spot_scenarios = spot_base + spot_shocks  # [T, S]

spot_panel_2d = Panel(
    data=spot_scenarios,
    axis_names=("time", "scenario"),
)

print(f"\n2D Panel (with scenarios):")
print(f"  Shape: {spot_panel_2d.data.shape}")
print(f"  Axis names: {spot_panel_2d.axis_names}")
print(f"  Day 0, Scenario 0: {spot_panel_2d.data[0, 0]:.4f}")
print(f"  Day 0, Scenario 99: {spot_panel_2d.data[0, 99]:.4f}")

# =============================================================================
# 3. Building a Simple MarketDataset
# =============================================================================

print("\n" + "=" * 70)
print("3. Building a Simple MarketDataset")
print("=" * 70)

# Define market IDs
eurusd_spot_id = MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD")

# For simplicity, we'll use just quote panels (no curves/vols for now)
dataset_simple = MarketDataset(
    dates=dates,
    n_scenarios=n_scenarios,
    panels={eurusd_spot_id: spot_panel_2d},
    curve_params={},
    curve_factories={},
    vol_params={},
    vol_factories={},
    meta={"source": "Synthetic data", "description": "Simple example"},
)

print(f"\nMarketDataset created:")
print(f"  Number of dates: {len(dataset_simple.dates)}")
print(f"  Number of scenarios: {dataset_simple.n_scenarios}")
print(f"  Date range: {dataset_simple.dates[0]} to {dataset_simple.dates[-1]}")
print(f"  Quote panels: {list(dataset_simple.panels.keys())}")

# Extract a snapshot
market_0_0 = dataset_simple.snapshot(time_idx=0, scenario_idx=0)
market_0_50 = dataset_simple.snapshot(time_idx=0, scenario_idx=50)
market_2_0 = dataset_simple.snapshot(time_idx=2, scenario_idx=0)

print(f"\nSnapshots:")
print(f"  Day 0, Scenario 0:  EURUSD = {market_0_0.quote(eurusd_spot_id):.4f}")
print(f"  Day 0, Scenario 50: EURUSD = {market_0_50.quote(eurusd_spot_id):.4f}")
print(f"  Day 2, Scenario 0:  EURUSD = {market_2_0.quote(eurusd_spot_id):.4f}")

# =============================================================================
# 4. Adding Curves to the Dataset
# =============================================================================

print("\n" + "=" * 70)
print("4. Adding Curves to the Dataset")
print("=" * 70)

# Define curve parameters
# Shape: [T, S, K, 2] where K is number of tenors, 2 = (tenor, zero_rate)
tenors = np.array([0.25, 0.5, 1.0, 2.0, 5.0])
n_tenors = len(tenors)

# Base zero rates (upward sloping)
base_rates = np.array([0.045, 0.048, 0.050, 0.053, 0.057])

# Generate curve scenarios
curve_params = np.zeros((n_dates, n_scenarios, n_tenors, 2))
for t in range(n_dates):
    for s in range(n_scenarios):
        # Parallel shift scenario
        rate_shock = np.random.randn() * 0.002  # ±20bp shock
        rates = base_rates + rate_shock + t * 0.001  # Small drift
        curve_params[t, s, :, 0] = tenors
        curve_params[t, s, :, 1] = rates

usd_curve_id = MarketId(asset_class="IR", mkt_type="CURVE", name="USD_OIS")

curve_panel = Panel(
    data=curve_params,
    axis_names=("time", "scenario", "tenor", "cols"),
)

# Create factory for reconstructing curves
curve_factory = ZeroRateCurveFactory(tenors=tenors)

print(f"\nCurve panel created:")
print(f"  Shape: {curve_panel.data.shape}")
print(f"  Axis names: {curve_panel.axis_names}")

# Create dataset with curves
dataset_with_curves = MarketDataset(
    dates=dates,
    n_scenarios=n_scenarios,
    panels={eurusd_spot_id: spot_panel_2d},
    curve_params={usd_curve_id: curve_panel},
    curve_factories={usd_curve_id: curve_factory},
    vol_params={},
    vol_factories={},
)

# Extract and use curve
market_with_curve = dataset_with_curves.snapshot(time_idx=0, scenario_idx=0)
curve = market_with_curve.curve(usd_curve_id)

print(f"\nExtracted curve from dataset:")
print(f"  5Y discount factor: {curve.df(5.0):.6f}")
print(f"  5Y zero rate: {curve.zero_rate(5.0):.4%}")

# =============================================================================
# 5. Adding Vol Surfaces to the Dataset
# =============================================================================

print("\n" + "=" * 70)
print("5. Adding Vol Surfaces to the Dataset")
print("=" * 70)

# Define vol surface parameters
vol_expiries = np.array([0.25, 0.5, 1.0])
vol_strikes = np.linspace(0.95, 1.20, 11)
n_exp = len(vol_expiries)
n_strikes = len(vol_strikes)

# Base vol surface (smile shape)
def make_smile_vol(exp, strike, spot=1.0850, atm=0.08, skew=-0.15, convex=0.10):
    m = np.log(strike / spot)
    return atm + skew * m + convex * m**2

base_vols = np.zeros((n_exp, n_strikes))
for i, exp in enumerate(vol_expiries):
    for j, k in enumerate(vol_strikes):
        base_vols[i, j] = make_smile_vol(exp, k)

# Generate vol scenarios
vol_params = np.zeros((n_dates, n_scenarios, n_exp, n_strikes))
for t in range(n_dates):
    for s in range(n_scenarios):
        vol_shock = np.random.randn() * 0.005  # ±0.5% vol shock
        vol_params[t, s, :, :] = base_vols + vol_shock

eurusd_vol_id = MarketId(asset_class="FX", mkt_type="VOL", name="EURUSD")

vol_panel = Panel(
    data=vol_params,
    axis_names=("time", "scenario", "expiry", "strike"),
)

# Create factory for reconstructing vol surfaces
vol_factory = GridVolFactory(expiries=vol_expiries, strikes=vol_strikes)

print(f"\nVol panel created:")
print(f"  Shape: {vol_panel.data.shape}")
print(f"  Axis names: {vol_panel.axis_names}")

# =============================================================================
# 6. Complete Dataset with All Components
# =============================================================================

print("\n" + "=" * 70)
print("6. Complete Dataset with All Components")
print("=" * 70)

# Build complete dataset
complete_dataset = MarketDataset(
    dates=dates,
    n_scenarios=n_scenarios,
    panels={eurusd_spot_id: spot_panel_2d},
    curve_params={usd_curve_id: curve_panel},
    curve_factories={usd_curve_id: curve_factory},
    vol_params={eurusd_vol_id: vol_panel},
    vol_factories={eurusd_vol_id: vol_factory},
    meta={
        "source": "Synthetic generator",
        "description": "Complete FX dataset with curves and vols",
        "ccy_pair": "EURUSD",
    },
)

print(f"\nComplete MarketDataset:")
print(f"  Dates: {len(complete_dataset.dates)}")
print(f"  Scenarios: {complete_dataset.n_scenarios}")
print(f"  Quotes: {list(complete_dataset.panels.keys())}")
print(f"  Curves: {list(complete_dataset.curve_params.keys())}")
print(f"  Vols: {list(complete_dataset.vol_params.keys())}")
print(f"  Meta: {complete_dataset.meta}")

# Extract complete market snapshot
market_full = complete_dataset.snapshot(time_idx=2, scenario_idx=42)

print(f"\nFull snapshot (day 2, scenario 42):")
print(f"  As-of: {market_full.asof}")
print(f"  EUR/USD spot: {market_full.quote(eurusd_spot_id):.4f}")
print(f"  USD 1Y rate: {market_full.curve(usd_curve_id).zero_rate(1.0):.4%}")
print(f"  ATM 1Y vol: {market_full.vol_surface(eurusd_vol_id).implied_vol(1.0, 1.085):.2%}")

# =============================================================================
# 7. Visualizing the Dataset
# =============================================================================

print("\n" + "=" * 70)
print("7. Visualizing the Dataset")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Spot scenarios over time
ax = axes[0]
for s in range(min(20, n_scenarios)):
    ax.plot(range(n_dates), spot_panel_2d.data[:, s], alpha=0.3, linewidth=0.8)
ax.plot(range(n_dates), spot_panel_2d.data.mean(axis=1), 'k-', linewidth=2, label='Mean')
ax.set_xticks(range(n_dates))
ax.set_xticklabels([d[-5:] for d in dates])  # Show MM-DD
ax.set_xlabel('Date')
ax.set_ylabel('EUR/USD Spot')
ax.set_title('Spot Scenarios Over Time')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Rate distribution at a single date
ax = axes[1]
rates_day0 = curve_params[0, :, 2, 1]  # 1Y rate, all scenarios
ax.hist(rates_day0 * 100, bins=30, color='#2E86AB', alpha=0.7, edgecolor='white')
ax.axvline(rates_day0.mean() * 100, color='red', linestyle='--', linewidth=2, label='Mean')
ax.set_xlabel('1Y Zero Rate (%)')
ax.set_ylabel('Frequency')
ax.set_title(f'Rate Distribution ({dates[0]})')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Vol distribution at a single date
ax = axes[2]
vols_day0 = vol_params[0, :, 1, 5]  # 0.5Y expiry, ATM strike, all scenarios
ax.hist(vols_day0 * 100, bins=30, color='#8B5CF6', alpha=0.7, edgecolor='white')
ax.axvline(vols_day0.mean() * 100, color='red', linestyle='--', linewidth=2, label='Mean')
ax.set_xlabel('ATM Implied Vol (%)')
ax.set_ylabel('Frequency')
ax.set_title(f'Vol Distribution ({dates[0]})')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
if SAVE_FIGURES:
    plt.savefig('dataset_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nPlot saved to dataset_visualization.png")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)

print("""
Key Takeaways:

1. MarketDataset:
   - Container for multi-day, multi-scenario market data
   - Produces Market snapshots via .snapshot(time_idx, scenario_idx)

2. Panel:
   - N-dimensional array with named axes
   - Shapes: [T], [T,S], [T,S,K], [T,S,K1,K2]
   - Axis names describe dimensions

3. Factories:
   - ZeroRateCurveFactory: Reconstructs curves from [tenor, rate] pairs
   - GridVolFactory: Reconstructs vol surfaces from grids

4. Dataset Components:
   - panels: Quote data (spot prices, single rates)
   - curve_params + curve_factories: Yield curves
   - vol_params + vol_factories: Volatility surfaces

5. Use Cases:
   - Historical backtesting (1 scenario per date)
   - Monte Carlo simulation (many scenarios per date)
   - Stress testing (specific scenarios)

Next: See 05_market_snapshots.py for snapshot extraction patterns.
""")

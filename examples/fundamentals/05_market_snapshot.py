#!/usr/bin/env python3
"""
Market Snapshots: From Dataset to Pricing

This example demonstrates how to extract Market snapshots from
MarketDataset for use in pricing:
- Extracting snapshots at specific (date, scenario) coordinates
- Iterating over dates and scenarios
- Using snapshots with pricers
- Time series analysis patterns

Author: QuantStrata Team
"""

import sys
sys.path.insert(0, '../..')

import numpy as np
import matplotlib.pyplot as plt
from typing import List

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

# save figures
SAVE_FIGURES = False

# =============================================================================
# Helper: Create a sample dataset
# =============================================================================

def create_sample_dataset(n_dates: int = 10, n_scenarios: int = 50) -> MarketDataset:
    """Create a sample dataset for demonstration."""
    
    # Generate dates
    dates = [f"2026-01-{20+i:02d}" for i in range(n_dates)]
    
    # Market IDs
    eurusd_spot_id = MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD")
    usd_curve_id = MarketId(asset_class="IR", mkt_type="CURVE", name="USD_OIS")
    eurusd_vol_id = MarketId(asset_class="FX", mkt_type="VOL", name="EURUSD")
    
    np.random.seed(42)
    
    # Spot panel [T, S]
    spot_base = 1.0850 + np.arange(n_dates).reshape(-1, 1) * 0.001
    spot_scenarios = spot_base + np.random.randn(n_dates, n_scenarios) * 0.005
    spot_panel = Panel(data=spot_scenarios, axis_names=("time", "scenario"))
    
    # Curve panel [T, S, K, 2]
    tenors = np.array([0.25, 0.5, 1.0, 2.0, 5.0])
    base_rates = np.array([0.045, 0.048, 0.050, 0.053, 0.057])
    curve_params = np.zeros((n_dates, n_scenarios, len(tenors), 2))
    for t in range(n_dates):
        for s in range(n_scenarios):
            rate_shock = np.random.randn() * 0.002
            curve_params[t, s, :, 0] = tenors
            curve_params[t, s, :, 1] = base_rates + rate_shock
    curve_panel = Panel(data=curve_params, axis_names=("time", "scenario", "tenor", "cols"))
    curve_factory = ZeroRateCurveFactory()
    
    # Vol panel [T, S, n_exp, n_k]
    vol_expiries = np.array([0.25, 0.5, 1.0])
    vol_strikes = np.linspace(0.95, 1.20, 11)
    base_vol = 0.08
    vol_params = np.zeros((n_dates, n_scenarios, len(vol_expiries), len(vol_strikes)))
    for t in range(n_dates):
        for s in range(n_scenarios):
            vol_shock = np.random.randn() * 0.003
            vol_params[t, s, :, :] = base_vol + vol_shock
    vol_panel = Panel(data=vol_params, axis_names=("time", "scenario", "expiry", "strike"))
    vol_factory = GridVolFactory(expiries=vol_expiries, strikes=vol_strikes)
    
    return MarketDataset(
        dates=dates,
        n_scenarios=n_scenarios,
        panels={eurusd_spot_id: spot_panel},
        curve_params={usd_curve_id: curve_panel},
        curve_factories={usd_curve_id: curve_factory},
        vol_params={eurusd_vol_id: vol_panel},
        vol_factories={eurusd_vol_id: vol_factory},
    )

# Create dataset
dataset = create_sample_dataset(n_dates=10, n_scenarios=50)

# Market IDs for access
EURUSD_SPOT = MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD")
USD_CURVE = MarketId(asset_class="IR", mkt_type="CURVE", name="USD_OIS")
EURUSD_VOL = MarketId(asset_class="FX", mkt_type="VOL", name="EURUSD")

# =============================================================================
# 1. Basic Snapshot Extraction
# =============================================================================

print("=" * 70)
print("1. Basic Snapshot Extraction")
print("=" * 70)

# Extract a single snapshot
market = dataset.snapshot(time_idx=0, scenario_idx=0)

print(f"\nSingle snapshot extracted:")
print(f"  As-of date: {market.asof}")
print(f"  EUR/USD spot: {market.quote(EURUSD_SPOT):.4f}")
print(f"  USD 1Y rate: {market.curve(USD_CURVE).zero_rate(1.0):.4%}")
print(f"  ATM vol (1Y): {market.vol_surface(EURUSD_VOL).implied_vol(1.0, 1.085):.2%}")

# Extract at different coordinates
print(f"\nSnapshots at different coordinates:")
for time_idx in [0, 5, 9]:
    for scenario_idx in [0, 25, 49]:
        m = dataset.snapshot(time_idx=time_idx, scenario_idx=scenario_idx)
        spot = m.quote(EURUSD_SPOT)
        print(f"  [{time_idx}, {scenario_idx}]: date={m.asof}, spot={spot:.4f}")

# =============================================================================
# 2. Iterating Over Time (Fixed Scenario)
# =============================================================================

print("\n" + "=" * 70)
print("2. Iterating Over Time (Fixed Scenario)")
print("=" * 70)

print("""
Pattern: Loop over dates with a fixed scenario.
Use case: Historical backtesting, P&L attribution.
""")

scenario_idx = 0  # Use first scenario
spots_over_time: List[float] = []
rates_over_time: List[float] = []

for time_idx in range(len(dataset.dates)):
    market = dataset.snapshot(time_idx=time_idx, scenario_idx=scenario_idx)
    spots_over_time.append(market.quote(EURUSD_SPOT))
    rates_over_time.append(market.curve(USD_CURVE).zero_rate(1.0))

print(f"\nTime series (scenario {scenario_idx}):")
print(f"{'Date':<12} {'Spot':<12} {'1Y Rate':<12}")
print("-" * 36)
for i, date in enumerate(dataset.dates):
    print(f"{date:<12} {spots_over_time[i]:<12.4f} {rates_over_time[i]:<12.4%}")

# =============================================================================
# 3. Iterating Over Scenarios (Fixed Date)
# =============================================================================

print("\n" + "=" * 70)
print("3. Iterating Over Scenarios (Fixed Date)")
print("=" * 70)

print("""
Pattern: Loop over scenarios with a fixed date.
Use case: Monte Carlo simulation, VaR calculation.
""")

time_idx = 5  # Middle of the dataset
spots_by_scenario: List[float] = []
vols_by_scenario: List[float] = []

for scenario_idx in range(dataset.n_scenarios):
    market = dataset.snapshot(time_idx=time_idx, scenario_idx=scenario_idx)
    spots_by_scenario.append(market.quote(EURUSD_SPOT))
    vols_by_scenario.append(market.vol_surface(EURUSD_VOL).implied_vol(1.0, 1.085))

spots_array = np.array(spots_by_scenario)
vols_array = np.array(vols_by_scenario)

print(f"\nScenario statistics (date {dataset.dates[time_idx]}):")
print(f"  Spot range: [{spots_array.min():.4f}, {spots_array.max():.4f}]")
print(f"  Spot mean:  {spots_array.mean():.4f}")
print(f"  Spot std:   {spots_array.std():.4f}")
print(f"  Vol range:  [{vols_array.min():.2%}, {vols_array.max():.2%}]")
print(f"  Vol mean:   {vols_array.mean():.2%}")

# =============================================================================
# 4. Full Grid Iteration
# =============================================================================

print("\n" + "=" * 70)
print("4. Full Grid Iteration")
print("=" * 70)

print("""
Pattern: Loop over all (date, scenario) combinations.
Use case: Full backtesting with multiple paths.
""")

# Example: compute P&L proxy (spot change) for each path
pnl_grid = np.zeros((len(dataset.dates), dataset.n_scenarios))

for t in range(len(dataset.dates)):
    for s in range(dataset.n_scenarios):
        market = dataset.snapshot(time_idx=t, scenario_idx=s)
        pnl_grid[t, s] = market.quote(EURUSD_SPOT) - 1.0850  # Change from initial

print(f"\nP&L grid shape: {pnl_grid.shape}")
print(f"Mean P&L by date:")
for t, date in enumerate(dataset.dates):
    print(f"  {date}: {pnl_grid[t, :].mean()*10000:+.1f} pips (std: {pnl_grid[t, :].std()*10000:.1f})")

# =============================================================================
# 5. Efficient Batch Access
# =============================================================================

print("\n" + "=" * 70)
print("5. Efficient Batch Access")
print("=" * 70)

print("""
For performance, you can access raw Panel data directly instead of
extracting snapshots one by one.
""")

# Direct panel access (much faster for bulk operations)
spot_panel = dataset.panels[EURUSD_SPOT]

print(f"\nDirect panel access:")
print(f"  Panel shape: {spot_panel.data.shape}")
print(f"  All spots at t=0: {spot_panel.data[0, :5]}... (first 5 scenarios)")

# Equivalent to extracting each snapshot
for s in range(5):
    market = dataset.snapshot(time_idx=0, scenario_idx=s)
    assert abs(market.quote(EURUSD_SPOT) - spot_panel.data[0, s]) < 1e-10
print(f"  Verified: snapshot extraction matches direct access ✓")

# =============================================================================
# 6. Visualization: Time Series and Scenario Distributions
# =============================================================================

print("\n" + "=" * 70)
print("6. Visualization")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: Time series paths
ax = axes[0, 0]
spot_data = dataset.panels[EURUSD_SPOT].data
for s in range(min(30, dataset.n_scenarios)):
    ax.plot(spot_data[:, s], alpha=0.3, linewidth=0.8, color='#2E86AB')
ax.plot(spot_data.mean(axis=1), 'k-', linewidth=2, label='Mean')
ax.fill_between(range(len(dataset.dates)),
                np.percentile(spot_data, 5, axis=1),
                np.percentile(spot_data, 95, axis=1),
                alpha=0.2, color='gray', label='5-95% CI')
ax.set_xticks(range(len(dataset.dates)))
ax.set_xticklabels([d[-2:] for d in dataset.dates])
ax.set_xlabel('Day')
ax.set_ylabel('EUR/USD')
ax.set_title('Spot Paths Over Time')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Terminal distribution
ax = axes[0, 1]
terminal_spots = spot_data[-1, :]
ax.hist(terminal_spots, bins=20, color='#8B5CF6', alpha=0.7, edgecolor='white', density=True)
ax.axvline(terminal_spots.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {terminal_spots.mean():.4f}')
ax.axvline(np.percentile(terminal_spots, 5), color='orange', linestyle=':', linewidth=2, label='5% VaR')
ax.set_xlabel('EUR/USD')
ax.set_ylabel('Density')
ax.set_title('Terminal Spot Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Heatmap of all paths
ax = axes[1, 0]
im = ax.imshow(spot_data.T, aspect='auto', cmap='RdYlBu_r', 
               extent=[0, len(dataset.dates)-1, 0, dataset.n_scenarios])
ax.set_xlabel('Day')
ax.set_ylabel('Scenario')
ax.set_title('Spot Heatmap (all scenarios)')
plt.colorbar(im, ax=ax, label='EUR/USD')

# Plot 4: P&L distribution
ax = axes[1, 1]
terminal_pnl = (terminal_spots - spot_data[0, :]) * 10000  # In pips
ax.hist(terminal_pnl, bins=20, color='#10B981', alpha=0.7, edgecolor='white')
ax.axvline(0, color='black', linestyle='-', linewidth=1)
ax.axvline(terminal_pnl.mean(), color='red', linestyle='--', linewidth=2, 
           label=f'Mean: {terminal_pnl.mean():+.1f} pips')
ax.set_xlabel('P&L (pips)')
ax.set_ylabel('Frequency')
ax.set_title('Terminal P&L Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
if SAVE_FIGURES:
    plt.savefig('snapshot_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nPlot saved to snapshot_analysis.png")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)

print("""
Key Takeaways:

1. Snapshot Extraction:
   market = dataset.snapshot(time_idx, scenario_idx)
   - Returns a complete Market object for pricing

2. Iteration Patterns:
   - Fixed scenario, vary time → backtesting
   - Fixed time, vary scenario → Monte Carlo
   - Full grid → comprehensive analysis

3. Efficient Access:
   - Direct panel access for bulk operations
   - Snapshot extraction for pricing integration

4. Statistics:
   - Mean, std, percentiles across scenarios
   - VaR calculation from terminal distribution

5. Coordinates:
   - time_idx: 0 to len(dates)-1
   - scenario_idx: 0 to n_scenarios-1

Next: See 06_scenario_shocks.py for applying market shocks.
""")

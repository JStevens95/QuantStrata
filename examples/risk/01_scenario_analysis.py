#!/usr/bin/env python3
"""
Scenario Analysis: Running Portfolios Through Market Shocks

This example demonstrates how to:
- Apply scenario shocks to a market
- Run a portfolio through multiple scenarios
- Analyze scenario P&L
- Create stress test reports

Author: QuantStrata Team
"""

import sys
sys.path.insert(0, '../..')

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

from src.marketdata.core.ids import MarketId
from src.marketdata.core.interfaces import Quote
from src.marketdata.core.market import Market
from src.marketdata.curves.term_structure import FlatCurve
from src.marketdata.surfaces.vol_surface import FlatVolSurface
from src.marketdata.scenarios.shocks import SpotShock, VolShock, ParallelRateShock

from src.instruments.fx.options.vanilla import EuropeanFxVanillaOption
from src.portfolio.core import Portfolio, Position
from src.portfolio.portfolio import PortfolioPricer
from src.pricers.registry import PricerRegistry
from src.pricers.fx.european_bsm import EuropeanFxVanillaBsmPricer

from src.risk.scenarios.runner import run_portfolio_scenarios

# Plot configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (14, 6),
    'font.size': 11,
    'axes.titlesize': 13,
    'lines.linewidth': 2,
})

# =============================================================================
# Setup: Market, Portfolio, Pricer
# =============================================================================

print("=" * 70)
print("Setup: Market, Portfolio, and Pricer")
print("=" * 70)

# Market IDs
EURUSD_SPOT = MarketId(asset_class="FX", data_type="SPOT", name="EURUSD")
USD_CURVE = MarketId(asset_class="IR", data_type="CURVE", name="USD_OIS")
EUR_CURVE = MarketId(asset_class="IR", data_type="CURVE", name="EUR_OIS")
EURUSD_VOL = MarketId(asset_class="FX", data_type="VOL", name="EURUSD")

# Base market
spot = 1.0850
r_usd = 0.05
r_eur = 0.02
vol = 0.10

base_market = Market(
    asof="2026-01-28",
    quotes={EURUSD_SPOT: Quote(value=spot)},
    curves={USD_CURVE: FlatCurve(rate=r_usd), EUR_CURVE: FlatCurve(rate=r_eur)},
    vols={EURUSD_VOL: FlatVolSurface(vol=vol)},
)

# Create option helper
def create_option(strike: float, expiry: float, is_call: bool, notional: float) -> EuropeanFxVanillaOption:
    return EuropeanFxVanillaOption(
        ccy_pair="EURUSD", strike=strike, expiry_years=expiry, is_call=is_call,
        notional=notional, spot_id=EURUSD_SPOT, domestic_curve_id=USD_CURVE,
        foreign_curve_id=EUR_CURVE, vol_id=EURUSD_VOL,
    )

# Portfolio: Long straddle
positions = [
    Position(position_id="LONG_CALL", 
             instrument=create_option(1.085, 0.5, True, 10_000_000),
             quantity=1.0),
    Position(position_id="LONG_PUT",
             instrument=create_option(1.085, 0.5, False, 10_000_000),
             quantity=1.0),
]
portfolio = Portfolio(positions=positions)

# Pricer setup
registry = PricerRegistry()
registry.register(EuropeanFxVanillaOption, EuropeanFxVanillaBsmPricer())
portfolio_pricer = PortfolioPricer(pricer_registry=registry)

# Base price
base_result = portfolio_pricer.price(portfolio, base_market)
base_pv = base_result.totals.pv

print(f"\nBase market:")
print(f"  EUR/USD: {spot}, Vol: {vol:.1%}, USD: {r_usd:.1%}, EUR: {r_eur:.1%}")
print(f"\nPortfolio: Long ATM Straddle (10M EUR)")
print(f"  Base PV: {base_pv:,.2f} USD")
print(f"  Delta: {base_result.totals.greeks.get('delta', 0):,.2f}")
print(f"  Gamma: {base_result.totals.greeks.get('gamma', 0):,.2f}")
print(f"  Vega:  {base_result.totals.greeks.get('vega', 0):,.2f}")

# =============================================================================
# 1. Single Shock Scenarios
# =============================================================================

print("\n" + "=" * 70)
print("1. Single Shock Scenarios")
print("=" * 70)

# Define single shocks
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

# Run scenarios
result = run_portfolio_scenarios(portfolio, base_market, portfolio_pricer, single_shocks)

print(f"\nScenario Results:")
print(f"{'Scenario':<15} {'PV (USD)':<15} {'P&L (USD)':<15} {'P&L %':<10}")
print("-" * 55)

for name in result.scenario_names:
    pv = result.pv_by_scenario[name]
    pnl = result.pnl_by_scenario[name]
    pnl_pct = (pnl / base_pv) * 100 if base_pv != 0 else 0
    print(f"{name:<15} {pv:>12,.2f}   {pnl:>+12,.2f}   {pnl_pct:>+8.2f}%")

# =============================================================================
# 2. Combined Stress Scenarios
# =============================================================================

print("\n" + "=" * 70)
print("2. Combined Stress Scenarios")
print("=" * 70)

print("""
Stress scenarios combine multiple shocks to simulate market crises.
""")

# Define stress scenarios
def create_stress_scenario(name: str, spot_bump: float, vol_bump: float, rate_bump: float):
    """Create a combined stress scenario."""
    shocks = []
    if spot_bump != 0:
        shocks.append(SpotShock(name=f"{name}_spot", spot_id=EURUSD_SPOT, 
                                bump=spot_bump, bump_mode="relative"))
    if vol_bump != 0:
        shocks.append(VolShock(name=f"{name}_vol", vol_id=EURUSD_VOL,
                              bump=vol_bump, bump_mode="absolute"))
    if rate_bump != 0:
        shocks.append(ParallelRateShock(name=f"{name}_rate", curve_id=USD_CURVE,
                                        rate_shift=rate_bump))
    return shocks

# Apply stress scenarios
stress_scenarios = {
    "Risk-Off": (-0.05, 0.05, -0.01),      # Spot down, vol up, rates down
    "Risk-On": (0.03, -0.02, 0.005),        # Spot up, vol down, rates up
    "Vol Spike": (0.0, 0.10, 0.0),          # Pure vol spike
    "EUR Crisis": (-0.10, 0.08, -0.02),     # Major EUR selloff
    "USD Weakness": (0.08, 0.03, -0.01),    # USD selloff
}

print(f"\nStress Scenario Results:")
print(f"{'Scenario':<15} {'Spot Δ':<10} {'Vol Δ':<10} {'Rate Δ':<10} {'P&L (USD)':<15}")
print("-" * 60)

stress_pnls = {}

for scenario_name, (spot_bump, vol_bump, rate_bump) in stress_scenarios.items():
    # Apply shocks sequentially
    shocked_market = base_market
    if spot_bump != 0:
        shock = SpotShock(name="spot", spot_id=EURUSD_SPOT, bump=spot_bump, bump_mode="relative")
        shocked_market = shock.apply(shocked_market)
    if vol_bump != 0:
        shock = VolShock(name="vol", vol_id=EURUSD_VOL, bump=vol_bump, bump_mode="absolute")
        shocked_market = shock.apply(shocked_market)
    if rate_bump != 0:
        shock = ParallelRateShock(name="rate", curve_id=USD_CURVE, rate_shift=rate_bump)
        shocked_market = shock.apply(shocked_market)
    
    # Price
    stressed_result = portfolio_pricer.price(portfolio, shocked_market)
    pnl = stressed_result.totals.pv - base_pv
    stress_pnls[scenario_name] = pnl
    
    print(f"{scenario_name:<15} {spot_bump*100:>+8.1f}%  {vol_bump*100:>+7.1f}pt  {rate_bump*100:>+7.0f}bp  {pnl:>+12,.2f}")

# =============================================================================
# 3. Spot Ladder Analysis
# =============================================================================

print("\n" + "=" * 70)
print("3. Spot Ladder Analysis")
print("=" * 70)

# Create spot ladder
spot_bumps = [-0.10, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.10]
spot_shocks = [
    SpotShock(name=f"Spot {b*100:+.0f}%", spot_id=EURUSD_SPOT, bump=b, bump_mode="relative")
    for b in spot_bumps if b != 0
]

ladder_result = run_portfolio_scenarios(portfolio, base_market, portfolio_pricer, spot_shocks)

print(f"\nSpot Ladder:")
print(f"{'Bump':<10} {'Spot':<12} {'PV':<15} {'P&L':<15} {'Δ P&L %':<10}")
print("-" * 62)

for bump in spot_bumps:
    if bump == 0:
        name = "BASE"
    else:
        name = f"Spot {bump*100:+.0f}%"
    
    pv = ladder_result.pv_by_scenario[name]
    pnl = ladder_result.pnl_by_scenario[name]
    spotted = spot * (1 + bump)
    
    print(f"{bump*100:>+8.0f}%  {spotted:<12.4f} {pv:>12,.2f}   {pnl:>+12,.2f}   {pnl/base_pv*100:>+8.2f}%")

# =============================================================================
# 4. Volatility Surface Analysis
# =============================================================================

print("\n" + "=" * 70)
print("4. Volatility Ladder Analysis")
print("=" * 70)

vol_bumps = [-0.05, -0.03, -0.01, 0.0, 0.01, 0.03, 0.05]
vol_shocks = [
    VolShock(name=f"Vol {b*100:+.0f}pt", vol_id=EURUSD_VOL, bump=b, bump_mode="absolute")
    for b in vol_bumps if b != 0
]

vol_ladder_result = run_portfolio_scenarios(portfolio, base_market, portfolio_pricer, vol_shocks)

print(f"\nVol Ladder:")
print(f"{'Bump':<10} {'Vol':<12} {'PV':<15} {'P&L':<15}")
print("-" * 52)

for bump in vol_bumps:
    if bump == 0:
        name = "BASE"
    else:
        name = f"Vol {bump*100:+.0f}pt"
    
    pv = vol_ladder_result.pv_by_scenario[name]
    pnl = vol_ladder_result.pnl_by_scenario[name]
    
    print(f"{bump*100:>+8.0f}pt  {(vol + bump)*100:<10.1f}%  {pv:>12,.2f}   {pnl:>+12,.2f}")

# =============================================================================
# 5. Visualization
# =============================================================================

print("\n" + "=" * 70)
print("5. Visualization")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Single shock P&L
ax = axes[0, 0]
shock_names = [name for name in result.scenario_names if name != "BASE"]
pnls = [result.pnl_by_scenario[name] for name in shock_names]
colors = ['#10B981' if p > 0 else '#E94F37' for p in pnls]

bars = ax.barh(shock_names, np.array(pnls)/1000, color=colors)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_xlabel('P&L (USD thousands)')
ax.set_title('Single Shock P&L')
ax.grid(True, alpha=0.3, axis='x')

# Plot 2: Stress scenario P&L
ax = axes[0, 1]
stress_names = list(stress_pnls.keys())
stress_vals = [stress_pnls[n]/1000 for n in stress_names]
colors = ['#10B981' if p > 0 else '#E94F37' for p in stress_vals]

bars = ax.barh(stress_names, stress_vals, color=colors)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_xlabel('P&L (USD thousands)')
ax.set_title('Stress Scenario P&L')
ax.grid(True, alpha=0.3, axis='x')

# Plot 3: Spot ladder P&L profile
ax = axes[1, 0]
spot_levels = [spot * (1 + b) for b in spot_bumps]
spot_pnls = [ladder_result.pnl_by_scenario.get(
    f"Spot {b*100:+.0f}%" if b != 0 else "BASE", 0)/1000 for b in spot_bumps]

ax.plot(spot_levels, spot_pnls, 'o-', color='#2E86AB', linewidth=2, markersize=8)
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(spot, color='gray', linestyle='--', alpha=0.7, label=f'Current: {spot}')
ax.fill_between(spot_levels, spot_pnls, 0, 
                where=(np.array(spot_pnls) > 0), alpha=0.3, color='#10B981')
ax.fill_between(spot_levels, spot_pnls, 0,
                where=(np.array(spot_pnls) <= 0), alpha=0.3, color='#E94F37')
ax.set_xlabel('EUR/USD Spot')
ax.set_ylabel('P&L (USD thousands)')
ax.set_title('Spot Ladder P&L Profile')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Vol ladder P&L profile
ax = axes[1, 1]
vol_levels = [(vol + b)*100 for b in vol_bumps]
vol_pnls = [vol_ladder_result.pnl_by_scenario.get(
    f"Vol {b*100:+.0f}pt" if b != 0 else "BASE", 0)/1000 for b in vol_bumps]

ax.plot(vol_levels, vol_pnls, 's-', color='#8B5CF6', linewidth=2, markersize=8)
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(vol*100, color='gray', linestyle='--', alpha=0.7, label=f'Current: {vol*100:.0f}%')
ax.fill_between(vol_levels, vol_pnls, 0,
                where=(np.array(vol_pnls) > 0), alpha=0.3, color='#10B981')
ax.fill_between(vol_levels, vol_pnls, 0,
                where=(np.array(vol_pnls) <= 0), alpha=0.3, color='#E94F37')
ax.set_xlabel('Implied Volatility (%)')
ax.set_ylabel('P&L (USD thousands)')
ax.set_title('Vol Ladder P&L Profile')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('scenario_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nPlot saved to scenario_analysis.png")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)

print("""
Key Takeaways:

1. Scenario Runner:
   - run_portfolio_scenarios() prices under multiple shocks
   - Returns PV and P&L by scenario name
   - Includes BASE scenario automatically

2. Shock Types:
   - SpotShock: Perturb spot prices
   - VolShock: Perturb volatility
   - ParallelRateShock: Shift yield curves

3. Stress Testing:
   - Combine multiple shocks for realistic scenarios
   - Apply shocks sequentially via .apply()
   - Model crisis events (risk-off, vol spike, etc.)

4. Ladder Analysis:
   - Systematic bumps to single risk factor
   - Shows P&L profile and breakeven points
   - Validates Greeks (delta, vega)

5. Portfolio Characteristics:
   - Straddle: Long gamma, long vega, delta-neutral
   - Profits from large spot moves (either direction)
   - Profits from volatility increase

Next: See 02_sensitivities_computation.py for formal Greeks.
""")

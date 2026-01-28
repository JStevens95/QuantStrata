#!/usr/bin/env python3
"""
Portfolio Pricing: Aggregating Positions

This example demonstrates how to:
- Build a Portfolio with multiple positions
- Use the PortfolioPricer to price all positions
- Aggregate Greeks across the portfolio
- Analyze portfolio-level risk metrics

Author: QuantStrata Team
"""

import sys
sys.path.insert(0, '../..')

import numpy as np
import matplotlib.pyplot as plt
from typing import List

from src.marketdata.core.ids import MarketId
from src.marketdata.core.interfaces import Quote
from src.marketdata.core.market import Market
from src.marketdata.curves.term_structure import FlatCurve
from src.marketdata.surfaces.vol_surface import FlatVolSurface

from src.instruments.fx.options.vanilla import EuropeanFxVanillaOption
from src.portfolio.core import Portfolio, Position
from src.portfolio.portfolio import PortfolioPricer
from src.pricers.registry import PricerRegistry
from src.pricers.fx.european_bsm import EuropeanFxVanillaBsmPricer

# Plot configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (12, 6),
    'font.size': 11,
    'axes.titlesize': 13,
    'lines.linewidth': 2,
})

# =============================================================================
# 1. Setup: Create Market
# =============================================================================

print("=" * 70)
print("1. Market Setup")
print("=" * 70)

# Market IDs
EURUSD_SPOT = MarketId(asset_class="FX", data_type="SPOT", name="EURUSD")
USD_CURVE = MarketId(asset_class="IR", data_type="CURVE", name="USD_OIS")
EUR_CURVE = MarketId(asset_class="IR", data_type="CURVE", name="EUR_OIS")
EURUSD_VOL = MarketId(asset_class="FX", data_type="VOL", name="EURUSD")

# Market data
spot = 1.0850
r_usd = 0.05
r_eur = 0.02
vol = 0.10

market = Market(
    asof="2026-01-28",
    quotes={EURUSD_SPOT: Quote(value=spot)},
    curves={
        USD_CURVE: FlatCurve(rate=r_usd),
        EUR_CURVE: FlatCurve(rate=r_eur),
    },
    vols={EURUSD_VOL: FlatVolSurface(vol=vol)},
)

print(f"\nMarket data:")
print(f"  EUR/USD Spot: {spot}")
print(f"  USD Rate: {r_usd:.2%}")
print(f"  EUR Rate: {r_eur:.2%}")
print(f"  Volatility: {vol:.2%}")

# =============================================================================
# 2. Build Portfolio
# =============================================================================

print("\n" + "=" * 70)
print("2. Building Portfolio")
print("=" * 70)

# Create options
def create_option(strike: float, expiry: float, is_call: bool, notional: float) -> EuropeanFxVanillaOption:
    return EuropeanFxVanillaOption(
        ccy_pair="EURUSD",
        strike=strike,
        expiry_years=expiry,
        is_call=is_call,
        notional=notional,
        spot_id=EURUSD_SPOT,
        domestic_curve_id=USD_CURVE,
        foreign_curve_id=EUR_CURVE,
        vol_id=EURUSD_VOL,
    )

# Portfolio: Long call spread + short put
positions = [
    Position(
        position_id="LONG_CALL_ATM",
        instrument=create_option(strike=1.08, expiry=0.5, is_call=True, notional=5_000_000),
        quantity=1.0,
    ),
    Position(
        position_id="SHORT_CALL_OTM",
        instrument=create_option(strike=1.12, expiry=0.5, is_call=True, notional=5_000_000),
        quantity=-1.0,  # Short position
    ),
    Position(
        position_id="SHORT_PUT_OTM",
        instrument=create_option(strike=1.04, expiry=0.5, is_call=False, notional=2_000_000),
        quantity=-1.0,
    ),
    Position(
        position_id="LONG_CALL_1Y",
        instrument=create_option(strike=1.10, expiry=1.0, is_call=True, notional=3_000_000),
        quantity=1.0,
    ),
]

portfolio = Portfolio(positions=positions)

print(f"\nPortfolio created with {len(positions)} positions:")
print(f"\n{'Position ID':<20} {'Type':<12} {'Strike':<10} {'Expiry':<10} {'Notional':<15} {'Qty':<6}")
print("-" * 73)
for pos in positions:
    opt = pos.instrument
    opt_type = "Call" if opt.is_call else "Put"
    print(f"{pos.position_id:<20} {opt_type:<12} {opt.strike:<10.4f} {opt.expiry_years:<10.2f} {opt.notional:>12,.0f}   {pos.quantity:+.0f}")

# =============================================================================
# 3. Setup Pricer Registry
# =============================================================================

print("\n" + "=" * 70)
print("3. Pricer Registry Setup")
print("=" * 70)

# Create registry and register pricers
registry = PricerRegistry()
registry.register(EuropeanFxVanillaOption, EuropeanFxVanillaBsmPricer())

# Create portfolio pricer
portfolio_pricer = PortfolioPricer(pricer_registry=registry)

print(f"\nPricer registry configured:")
print(f"  EuropeanFxVanillaOption -> EuropeanFxVanillaBsmPricer")

# =============================================================================
# 4. Price Portfolio
# =============================================================================

print("\n" + "=" * 70)
print("4. Portfolio Pricing Results")
print("=" * 70)

# Price portfolio
result = portfolio_pricer.price(portfolio, market)

print(f"\n{'Position ID':<20} {'PV (USD)':<15} {'Delta':<12} {'Gamma':<12} {'Vega':<12}")
print("-" * 71)

for pos_result in result.per_position:
    pv = pos_result.pv
    delta = pos_result.greeks.get('delta', 0)
    gamma = pos_result.greeks.get('gamma', 0)
    vega = pos_result.greeks.get('vega', 0)
    print(f"{pos_result.position_id:<20} {pv:>12,.2f}   {delta:>10,.2f} {gamma:>10,.2f} {vega:>10,.2f}")

print("-" * 71)
print(f"{'TOTAL':<20} {result.totals.pv:>12,.2f}   "
      f"{result.totals.greeks.get('delta', 0):>10,.2f} "
      f"{result.totals.greeks.get('gamma', 0):>10,.2f} "
      f"{result.totals.greeks.get('vega', 0):>10,.2f}")

# =============================================================================
# 5. Portfolio Analytics
# =============================================================================

print("\n" + "=" * 70)
print("5. Portfolio Analytics")
print("=" * 70)

total_notional = sum(abs(pos.quantity * pos.instrument.notional) for pos in positions)
total_pv = result.totals.pv
total_delta = result.totals.greeks.get('delta', 0)
total_gamma = result.totals.greeks.get('gamma', 0)
total_vega = result.totals.greeks.get('vega', 0)

print(f"\nPortfolio Summary:")
print(f"  Total Notional (abs): {total_notional:>15,.0f} EUR")
print(f"  Total PV:             {total_pv:>15,.2f} USD")
print(f"  PV as % of Notional:  {total_pv/total_notional*100:>14.3f}%")

print(f"\nRisk Metrics:")
print(f"  Delta (total):        {total_delta:>15,.2f}")
print(f"  Gamma (total):        {total_gamma:>15,.2f}")
print(f"  Vega (total):         {total_vega:>15,.2f}")

# Dollar Greeks (sensitivity to 1% move)
delta_1pct = total_delta * spot * 0.01
gamma_1pct = 0.5 * total_gamma * (spot * 0.01)**2
vega_1pt = total_vega * 0.01

print(f"\nScenario P&L Estimates:")
print(f"  Spot +1%:  Delta P&L = {delta_1pct:>+12,.2f} USD")
print(f"  Spot +1%:  Gamma P&L = {gamma_1pct:>+12,.2f} USD")
print(f"  Vol +1pt:  Vega P&L  = {vega_1pt:>+12,.2f} USD")

# =============================================================================
# 6. P&L Profile Analysis
# =============================================================================

print("\n" + "=" * 70)
print("6. P&L Profile Analysis")
print("=" * 70)

# Compute P&L across spot range
spot_range = np.linspace(spot * 0.90, spot * 1.10, 50)
pnl_profile = []

for s in spot_range:
    shocked_market = Market(
        asof=market.asof,
        quotes={EURUSD_SPOT: Quote(value=s)},
        curves=market.curves,
        vols=market.vols,
    )
    shocked_result = portfolio_pricer.price(portfolio, shocked_market)
    pnl = shocked_result.totals.pv - total_pv
    pnl_profile.append(pnl)

pnl_profile = np.array(pnl_profile)

# Find breakeven points
spot_changes = (spot_range / spot - 1) * 100
zero_crossings = np.where(np.diff(np.sign(pnl_profile)))[0]

print(f"\nP&L Profile:")
print(f"  Spot range: {spot_range[0]:.4f} to {spot_range[-1]:.4f}")
print(f"  P&L range: {pnl_profile.min():+,.2f} to {pnl_profile.max():+,.2f}")

if len(zero_crossings) > 0:
    print(f"\n  Breakeven points:")
    for idx in zero_crossings:
        be_spot = (spot_range[idx] + spot_range[idx+1]) / 2
        print(f"    {be_spot:.4f} ({(be_spot/spot - 1)*100:+.2f}% from current)")

# =============================================================================
# 7. Visualization
# =============================================================================

print("\n" + "=" * 70)
print("7. Visualization")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Position PV breakdown
ax = axes[0, 0]
position_ids = [r.position_id for r in result.per_position]
pvs = [r.pv for r in result.per_position]
colors = ['#10B981' if pv > 0 else '#E94F37' for pv in pvs]

bars = ax.barh(position_ids, pvs, color=colors)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_xlabel('PV (USD)')
ax.set_title('Position PV Breakdown')
ax.grid(True, alpha=0.3, axis='x')

for bar, pv in zip(bars, pvs):
    ax.text(pv + (5000 if pv > 0 else -5000), bar.get_y() + bar.get_height()/2,
           f'{pv:,.0f}', ha='left' if pv > 0 else 'right', va='center', fontsize=9)

# Plot 2: Greeks breakdown
ax = axes[0, 1]
greek_names = ['Delta', 'Gamma', 'Vega']
greek_values = [total_delta, total_gamma * 1000, total_vega]  # Scale gamma for visibility

x = np.arange(len(greek_names))
width = 0.6
colors = ['#2E86AB' if v > 0 else '#E94F37' for v in greek_values]

bars = ax.bar(x, greek_values, width, color=colors)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(greek_names)
ax.set_ylabel('Greek Value')
ax.set_title('Portfolio Greeks (Gamma ×1000)')
ax.grid(True, alpha=0.3, axis='y')

# Plot 3: P&L Profile
ax = axes[1, 0]
ax.plot(spot_range, pnl_profile / 1000, color='#2E86AB', linewidth=2)
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(spot, color='gray', linestyle='--', alpha=0.7, label=f'Current: {spot}')
ax.fill_between(spot_range, pnl_profile/1000, 0, 
                where=(pnl_profile > 0), alpha=0.3, color='#10B981', label='Profit')
ax.fill_between(spot_range, pnl_profile/1000, 0,
                where=(pnl_profile <= 0), alpha=0.3, color='#E94F37', label='Loss')
ax.set_xlabel('EUR/USD Spot')
ax.set_ylabel('P&L (USD thousands)')
ax.set_title('Portfolio P&L Profile')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Delta profile
ax = axes[1, 1]
delta_profile = []

for s in spot_range:
    shocked_market = Market(
        asof=market.asof,
        quotes={EURUSD_SPOT: Quote(value=s)},
        curves=market.curves,
        vols=market.vols,
    )
    shocked_result = portfolio_pricer.price(portfolio, shocked_market)
    delta_profile.append(shocked_result.totals.greeks.get('delta', 0))

ax.plot(spot_range, delta_profile, color='#8B5CF6', linewidth=2)
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(spot, color='gray', linestyle='--', alpha=0.7, label=f'Current: {spot}')
ax.set_xlabel('EUR/USD Spot')
ax.set_ylabel('Portfolio Delta')
ax.set_title('Delta Profile (shows Gamma)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('portfolio_pricing.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nPlot saved to portfolio_pricing.png")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)

print("""
Key Takeaways:

1. Portfolio Structure:
   - Portfolio contains a list of Positions
   - Position = Instrument + Quantity + ID
   - Quantity can be negative (short positions)

2. Pricer Registry:
   - Maps instrument types to pricers
   - Allows flexible pricer selection

3. PortfolioPricer:
   - Prices all positions in a single call
   - Aggregates PV and Greeks
   - Returns per-position and total results

4. Portfolio Analytics:
   - P&L profile shows profit/loss at different spots
   - Delta profile shows how delta changes (gamma effect)
   - Breakeven analysis identifies key levels

5. Greeks Aggregation:
   - Greeks add linearly across positions
   - Sign of quantity affects Greek direction

Next: See examples/risk/ for scenario analysis and sensitivities.
""")

#!/usr/bin/env python3
"""
Scenario Shocks: Applying Market Perturbations

This example covers how to apply shocks to market data for
scenario analysis and stress testing:
- SpotShock: Perturb spot prices
- VolShock: Perturb volatility surfaces
- ParallelRateShock: Shift interest rate curves
- Combining multiple shocks

These tools are essential for risk management and sensitivity analysis.

Author: QuantStrata Team
"""

import sys
sys.path.insert(0, '../..')

import numpy as np
import matplotlib.pyplot as plt

from src.marketdata.core.ids import MarketId
from src.marketdata.core.interfaces import Quote
from src.marketdata.core.market import Market
from src.marketdata.curves.term_structure import ZeroRateCurve
from src.marketdata.surfaces.vol_surface import GridVolSurface
from src.marketdata.scenarios.shocks import (
    SpotShock, VolShock, ParallelRateShock
)

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
# Setup: Create a Base Market
# =============================================================================

print("=" * 70)
print("Setup: Creating Base Market")
print("=" * 70)

# Define market IDs
EURUSD_SPOT = MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD")
USD_CURVE = MarketId(asset_class="IR", mkt_type="CURVE", name="USD_OIS")
EUR_CURVE = MarketId(asset_class="IR", mkt_type="CURVE", name="EUR_OIS")
EURUSD_VOL = MarketId(asset_class="FX", mkt_type="VOL", name="EURUSD")

# Create curves
tenors = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
usd_rates = np.array([0.050, 0.051, 0.052, 0.053, 0.055, 0.057])
eur_rates = np.array([0.035, 0.036, 0.038, 0.040, 0.042, 0.044])

usd_curve = ZeroRateCurve(tenors=tenors, zero_rates=usd_rates)
eur_curve = ZeroRateCurve(tenors=tenors, zero_rates=eur_rates)

# Create vol surface
vol_expiries = np.array([0.25, 0.5, 1.0, 2.0])
vol_strikes = np.linspace(0.95, 1.20, 11)
base_vol = 0.08

# Simple smile
vol_grid = np.zeros((len(vol_expiries), len(vol_strikes)))
for i, exp in enumerate(vol_expiries):
    for j, k in enumerate(vol_strikes):
        moneyness = np.log(k / 1.085)
        vol_grid[i, j] = base_vol - 0.15 * moneyness + 0.10 * moneyness**2

eurusd_vol = GridVolSurface(expiries=vol_expiries, strikes=vol_strikes, implied_vols=vol_grid)

# Create base market
base_market = Market(
    asof="2026-01-28",
    quotes={EURUSD_SPOT: Quote(value=1.0850)},
    curves={USD_CURVE: usd_curve, EUR_CURVE: eur_curve},
    vols={EURUSD_VOL: eurusd_vol},
)

print(f"\nBase market created:")
print(f"  As-of: {base_market.asof}")
print(f"  EUR/USD spot: {base_market.quote(EURUSD_SPOT):.4f}")
print(f"  USD 1Y rate: {base_market.curve(USD_CURVE).zero_rate(1.0):.4%}")
print(f"  EUR 1Y rate: {base_market.curve(EUR_CURVE).zero_rate(1.0):.4%}")
print(f"  ATM 1Y vol: {base_market.vol_surface(EURUSD_VOL).implied_vol(1.0, 1.085):.2%}")

# =============================================================================
# 1. SpotShock: Perturbing Spot Prices
# =============================================================================

print("\n" + "=" * 70)
print("1. SpotShock: Perturbing Spot Prices")
print("=" * 70)

print("""
SpotShock applies a bump to a spot price quote.

Bump modes:
- "relative": new_spot = old_spot * (1 + bump)
- "absolute": new_spot = old_spot + bump

Example: bump=0.01, relative → +1% spot move
""")

# Relative shock: +1%
spot_up = SpotShock(
    name="spot_up_1pct",
    spot_id=EURUSD_SPOT,
    bump=0.01,
    bump_mode="relative",
)

# Relative shock: -1%
spot_down = SpotShock(
    name="spot_down_1pct",
    spot_id=EURUSD_SPOT,
    bump=-0.01,
    bump_mode="relative",
)

# Apply shocks
market_spot_up = spot_up.apply(base_market)
market_spot_down = spot_down.apply(base_market)

print(f"\nSpot shock results:")
print(f"  Base spot:      {base_market.quote(EURUSD_SPOT):.4f}")
print(f"  Spot +1%:       {market_spot_up.quote(EURUSD_SPOT):.4f}")
print(f"  Spot -1%:       {market_spot_down.quote(EURUSD_SPOT):.4f}")

# Verify other market data is unchanged
print(f"\nOther market data unchanged:")
print(f"  USD curve (spot up):  {market_spot_up.curve(USD_CURVE).zero_rate(1.0):.4%}")
print(f"  Vol surface (spot up): {market_spot_up.vol_surface(EURUSD_VOL).implied_vol(1.0, 1.085):.2%}")

# =============================================================================
# 2. VolShock: Perturbing Volatility Surfaces
# =============================================================================

print("\n" + "=" * 70)
print("2. VolShock: Perturbing Volatility Surfaces")
print("=" * 70)

print("""
VolShock applies a bump to an entire volatility surface.

Bump modes:
- "absolute": new_vol = old_vol + bump (e.g., bump=0.01 → +1 vol point)
- "relative": new_vol = old_vol * (1 + bump) (e.g., bump=0.10 → +10% of vol)
""")

# Absolute shock: +1 vol point
vol_up = VolShock(
    name="vol_up_1pt",
    vol_id=EURUSD_VOL,
    bump=0.01,  # +1% absolute
    bump_mode="absolute",
)

# Absolute shock: -1 vol point
vol_down = VolShock(
    name="vol_down_1pt",
    vol_id=EURUSD_VOL,
    bump=-0.01,
    bump_mode="absolute",
)

# Apply shocks
market_vol_up = vol_up.apply(base_market)
market_vol_down = vol_down.apply(base_market)

print(f"\nVol shock results (ATM 1Y):")
print(f"  Base vol:   {base_market.vol_surface(EURUSD_VOL).implied_vol(1.0, 1.085):.2%}")
print(f"  Vol +1pt:   {market_vol_up.vol_surface(EURUSD_VOL).implied_vol(1.0, 1.085):.2%}")
print(f"  Vol -1pt:   {market_vol_down.vol_surface(EURUSD_VOL).implied_vol(1.0, 1.085):.2%}")

# Vol shock affects all strikes
print(f"\nVol shock across strikes (1Y expiry):")
print(f"{'Strike':<10} {'Base':<10} {'Vol Up':<10} {'Vol Down':<10}")
print("-" * 40)
for k in [1.00, 1.05, 1.085, 1.12, 1.18]:
    base_v = base_market.vol_surface(EURUSD_VOL).implied_vol(1.0, k)
    up_v = market_vol_up.vol_surface(EURUSD_VOL).implied_vol(1.0, k)
    down_v = market_vol_down.vol_surface(EURUSD_VOL).implied_vol(1.0, k)
    print(f"{k:<10.3f} {base_v:<10.2%} {up_v:<10.2%} {down_v:<10.2%}")

# =============================================================================
# 3. ParallelRateShock: Shifting Yield Curves
# =============================================================================

print("\n" + "=" * 70)
print("3. ParallelRateShock: Shifting Yield Curves")
print("=" * 70)

print("""
ParallelRateShock applies a parallel shift to a yield curve.

The shift is applied to continuous rates:
  df_shocked(t) = df_base(t) * exp(-rate_shift * t)

Example: rate_shift=0.01 → +100bp parallel shift
""")

# +100bp shift to USD curve
usd_up = ParallelRateShock(
    name="usd_up_100bp",
    curve_id=USD_CURVE,
    rate_shift=0.01,  # +100bp
)

# -50bp shift to USD curve
usd_down = ParallelRateShock(
    name="usd_down_50bp",
    curve_id=USD_CURVE,
    rate_shift=-0.005,  # -50bp
)

# Apply shocks
market_rate_up = usd_up.apply(base_market)
market_rate_down = usd_down.apply(base_market)

print(f"\nRate shock results (USD OIS):")
print(f"{'Tenor':<10} {'Base':<12} {'Up 100bp':<12} {'Down 50bp':<12}")
print("-" * 46)
for t in [0.25, 1.0, 5.0, 10.0]:
    base_r = base_market.curve(USD_CURVE).zero_rate(t)
    up_r = market_rate_up.curve(USD_CURVE).zero_rate(t)
    down_r = market_rate_down.curve(USD_CURVE).zero_rate(t)
    print(f"{t:<10.2f} {base_r:<12.4%} {up_r:<12.4%} {down_r:<12.4%}")

# =============================================================================
# 4. Combining Multiple Shocks
# =============================================================================

print("\n" + "=" * 70)
print("4. Combining Multiple Shocks")
print("=" * 70)

print("""
Shocks can be combined by applying them sequentially.
The result is a market with all shocks applied.
""")

# Stress scenario: spot down, vol up, rates down
stress_spot = SpotShock(name="stress_spot", spot_id=EURUSD_SPOT, bump=-0.05, bump_mode="relative")
stress_vol = VolShock(name="stress_vol", vol_id=EURUSD_VOL, bump=0.03, bump_mode="absolute")
stress_rate = ParallelRateShock(name="stress_rate", curve_id=USD_CURVE, rate_shift=-0.01)

# Apply sequentially
stressed_market = stress_spot.apply(base_market)
stressed_market = stress_vol.apply(stressed_market)
stressed_market = stress_rate.apply(stressed_market)

print(f"\nCombined stress scenario:")
print(f"  Shocks: Spot -5%, Vol +3pt, USD rates -100bp")
print(f"\n{'Metric':<20} {'Base':<15} {'Stressed':<15} {'Change':<15}")
print("-" * 65)

base_spot = base_market.quote(EURUSD_SPOT)
stress_spot_val = stressed_market.quote(EURUSD_SPOT)
print(f"{'EUR/USD Spot':<20} {base_spot:<15.4f} {stress_spot_val:<15.4f} {(stress_spot_val/base_spot-1)*100:+.1f}%")

base_vol_val = base_market.vol_surface(EURUSD_VOL).implied_vol(1.0, 1.085)
stress_vol_val = stressed_market.vol_surface(EURUSD_VOL).implied_vol(1.0, 1.085)
print(f"{'ATM 1Y Vol':<20} {base_vol_val:<15.2%} {stress_vol_val:<15.2%} {(stress_vol_val-base_vol_val)*100:+.1f}pt")

base_rate = base_market.curve(USD_CURVE).zero_rate(1.0)
stress_rate_val = stressed_market.curve(USD_CURVE).zero_rate(1.0)
print(f"{'USD 1Y Rate':<20} {base_rate:<15.4%} {stress_rate_val:<15.4%} {(stress_rate_val-base_rate)*10000:+.0f}bp")

# =============================================================================
# 5. Creating Scenario Ladders
# =============================================================================

print("\n" + "=" * 70)
print("5. Creating Scenario Ladders")
print("=" * 70)

print("""
Scenario ladders are used to compute sensitivities:
- Apply a series of bumps (e.g., -5%, -2%, 0%, +2%, +5%)
- Price under each scenario
- Analyze the P&L profile
""")

# Spot ladder
spot_bumps = [-0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05]
spot_ladder = []

print(f"\nSpot ladder:")
print(f"{'Bump':<10} {'Spot':<12} {'Change':<12}")
print("-" * 34)

for bump in spot_bumps:
    if bump == 0.0:
        market = base_market
    else:
        shock = SpotShock(name=f"spot_{bump:+.0%}", spot_id=EURUSD_SPOT, 
                         bump=bump, bump_mode="relative")
        market = shock.apply(base_market)
    
    spot = market.quote(EURUSD_SPOT)
    spot_ladder.append((bump, spot))
    print(f"{bump:+.0%}       {spot:<12.4f} {(spot/base_spot-1)*100:+.2f}%")

# Vol ladder
vol_bumps = [-0.03, -0.01, 0.0, 0.01, 0.03]
vol_ladder = []

print(f"\nVol ladder (ATM 1Y):")
print(f"{'Bump':<10} {'Vol':<12} {'Change':<12}")
print("-" * 34)

for bump in vol_bumps:
    if bump == 0.0:
        market = base_market
    else:
        shock = VolShock(name=f"vol_{bump:+.0%}", vol_id=EURUSD_VOL,
                        bump=bump, bump_mode="absolute")
        market = shock.apply(base_market)
    
    vol = market.vol_surface(EURUSD_VOL).implied_vol(1.0, 1.085)
    vol_ladder.append((bump, vol))
    print(f"{bump*100:+.0f}pt      {vol:<12.2%} {(vol-base_vol_val)*100:+.1f}pt")

# =============================================================================
# 6. Visualization
# =============================================================================

print("\n" + "=" * 70)
print("6. Visualization")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Spot scenarios
ax = axes[0]
bumps_pct = [b*100 for b, _ in spot_ladder]
spots = [s for _, s in spot_ladder]
colors = ['#E94F37' if b < 0 else '#10B981' if b > 0 else '#2E86AB' for b, _ in spot_ladder]
ax.bar(range(len(spot_ladder)), spots, color=colors)
ax.axhline(base_spot, color='gray', linestyle='--', alpha=0.7, label=f'Base: {base_spot:.4f}')
ax.set_xticks(range(len(spot_ladder)))
ax.set_xticklabels([f'{b:+.0f}%' for b in bumps_pct])
ax.set_xlabel('Spot Shock')
ax.set_ylabel('EUR/USD')
ax.set_title('Spot Scenario Ladder')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Rate curve comparison
ax = axes[1]
t_grid = np.linspace(0.1, 10, 50)
base_rates_plot = [base_market.curve(USD_CURVE).zero_rate(t) * 100 for t in t_grid]
up_rates = [market_rate_up.curve(USD_CURVE).zero_rate(t) * 100 for t in t_grid]
down_rates = [market_rate_down.curve(USD_CURVE).zero_rate(t) * 100 for t in t_grid]

ax.plot(t_grid, base_rates_plot, 'k-', linewidth=2, label='Base')
ax.plot(t_grid, up_rates, '--', color='#E94F37', linewidth=2, label='+100bp')
ax.plot(t_grid, down_rates, '--', color='#10B981', linewidth=2, label='-50bp')
ax.fill_between(t_grid, down_rates, up_rates, alpha=0.2, color='gray')
ax.set_xlabel('Tenor (years)')
ax.set_ylabel('Zero Rate (%)')
ax.set_title('Rate Curve Scenarios')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Vol smile comparison
ax = axes[2]
strikes_plot = np.linspace(0.98, 1.18, 30)
base_vols_plot = [base_market.vol_surface(EURUSD_VOL).implied_vol(1.0, k) * 100 for k in strikes_plot]
up_vols = [market_vol_up.vol_surface(EURUSD_VOL).implied_vol(1.0, k) * 100 for k in strikes_plot]
down_vols = [market_vol_down.vol_surface(EURUSD_VOL).implied_vol(1.0, k) * 100 for k in strikes_plot]

ax.plot(strikes_plot, base_vols_plot, 'k-', linewidth=2, label='Base')
ax.plot(strikes_plot, up_vols, '--', color='#E94F37', linewidth=2, label='+1pt')
ax.plot(strikes_plot, down_vols, '--', color='#10B981', linewidth=2, label='-1pt')
ax.axvline(1.085, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Strike')
ax.set_ylabel('Implied Vol (%)')
ax.set_title('Vol Smile Scenarios (1Y)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
if SAVE_FIGURES:
    plt.savefig('scenario_shocks.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nPlot saved to scenario_shocks.png")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)

print("""
Key Takeaways:

1. Shock Types:
   - SpotShock: Bump spot prices (relative or absolute)
   - VolShock: Bump entire vol surface (relative or absolute)
   - ParallelRateShock: Parallel shift yield curves

2. Bump Modes:
   - "relative": new = old * (1 + bump)
   - "absolute": new = old + bump

3. Applying Shocks:
   shocked_market = shock.apply(base_market)
   - Returns a new MarketView
   - Original market unchanged (immutable)

4. Combining Shocks:
   market = shock1.apply(base_market)
   market = shock2.apply(market)
   - Apply sequentially for combined scenarios

5. Scenario Ladders:
   - Series of bumps for sensitivity analysis
   - Useful for computing Greeks via finite differences

These tools form the foundation for the risk module's
scenario analysis and sensitivity computation.

Next: See examples/pricing/ for using markets with pricers.
""")

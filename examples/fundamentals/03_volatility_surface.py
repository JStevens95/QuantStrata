#!/usr/bin/env python3
"""
Volatility Surfaces: Implied Volatility and the Smile

This example covers volatility surfaces used in option pricing:
- Implied volatility and its interpretation
- The volatility smile and skew
- Building and querying vol surfaces
- Relationship to option prices

Volatility surfaces are essential for pricing options at
different strikes and maturities.

Author: QuantStrata Team
"""

import sys
sys.path.insert(0, '../..')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.marketdata.core.ids import MarketId
from src.marketdata.core.interfaces import Quote
from src.marketdata.core.market import Market
from src.marketdata.surfaces.vol_surface import FlatVolSurface, GridVolSurface

# Plot configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (12, 5),
    'font.size': 11,
    'axes.titlesize': 13,
    'lines.linewidth': 2,
})

# =============================================================================
# 1. What is Implied Volatility?
# =============================================================================

print("=" * 70)
print("1. What is Implied Volatility?")
print("=" * 70)

print("""
Implied Volatility (IV) is the market's expectation of future volatility
implied by option prices.

Key points:
- IV is backed out from option prices using Black-Scholes
- It's NOT the same as historical (realized) volatility
- Higher IV means more expensive options
- IV varies by strike (smile) and expiry (term structure)

The Black-Scholes assumption of constant volatility is violated in practice.
The "volatility surface" captures how IV varies across strikes and expiries.
""")

# =============================================================================
# 2. Flat Volatility Surface
# =============================================================================

print("\n" + "=" * 70)
print("2. Flat Volatility Surface")
print("=" * 70)

# Create a flat vol surface (same vol everywhere)
flat_vol = 0.15  # 15% volatility
flat_surface = FlatVolSurface(vol=flat_vol)

print(f"\nFlat volatility surface at {flat_vol:.1%}:")
print(f"{'Expiry':<10} {'Strike':<10} {'Implied Vol':<12}")
print("-" * 32)

for expiry in [0.25, 0.5, 1.0]:
    for strike in [90, 100, 110]:
        iv = flat_surface.implied_vol(expiry, strike)
        print(f"{expiry:<10.2f} {strike:<10.0f} {iv:<12.2%}")

# =============================================================================
# 3. The Volatility Smile
# =============================================================================

print("\n" + "=" * 70)
print("3. The Volatility Smile")
print("=" * 70)

print("""
The volatility smile shows that IV typically:
- Is higher for OTM puts (low strikes) - "skew"
- Is higher for OTM calls (high strikes)  
- Is lowest around ATM strikes
- Creates a "smile" or "smirk" shape

This reflects:
- Fat tails in real return distributions
- Crash risk premium (especially for equities)
- Supply/demand imbalances
""")

# Create a vol surface with smile
# We'll use a simple parametric model for illustration
spot = 100.0
expiries = np.array([0.25, 0.5, 1.0, 2.0])
strikes = np.linspace(80, 120, 21)

# Generate smile surface (simplified model)
def generate_smile_vol(expiry, strike, spot=100, atm_vol=0.15, skew=-0.1, smile=0.05):
    """Generate implied vol with smile/skew."""
    moneyness = np.log(strike / spot)
    # Skew effect (linear in moneyness)
    vol = atm_vol + skew * moneyness
    # Smile effect (quadratic in moneyness)
    vol += smile * moneyness**2
    # Term structure (vol decreases with time)
    vol *= (1.0 - 0.1 * np.log(expiry + 0.1))
    return max(0.05, vol)  # Floor at 5%

# Build vol grid
vol_grid = np.zeros((len(expiries), len(strikes)))
for i, exp in enumerate(expiries):
    for j, k in enumerate(strikes):
        vol_grid[i, j] = generate_smile_vol(exp, k, spot)

# Create GridVolSurface
smile_surface = GridVolSurface(
    expiries=expiries,
    strikes=strikes,
    vols=vol_grid,
)

print(f"\nSmile surface at various points:")
print(f"{'Expiry':<10} {'Strike':<10} {'Moneyness':<12} {'Implied Vol':<12}")
print("-" * 44)

test_strikes = [85, 95, 100, 105, 115]
for strike in test_strikes:
    iv = smile_surface.implied_vol(1.0, strike)
    moneyness = np.log(strike / spot) * 100  # In percent
    print(f"{1.0:<10.2f} {strike:<10.0f} {moneyness:<12.1f}% {iv:<12.2%}")

# =============================================================================
# 4. Visualizing the Volatility Surface
# =============================================================================

print("\n" + "=" * 70)
print("4. Visualizing the Volatility Surface")
print("=" * 70)

fig = plt.figure(figsize=(15, 5))

# Plot 1: Smile at different expiries
ax1 = fig.add_subplot(131)
colors = plt.cm.viridis(np.linspace(0, 0.8, len(expiries)))

for i, (exp, color) in enumerate(zip(expiries, colors)):
    vols = [smile_surface.implied_vol(exp, k) * 100 for k in strikes]
    ax1.plot(strikes, vols, color=color, linewidth=2, label=f'T = {exp}Y')

ax1.axvline(spot, color='gray', linestyle='--', alpha=0.5, label='ATM')
ax1.set_xlabel('Strike')
ax1.set_ylabel('Implied Volatility (%)')
ax1.set_title('Volatility Smile by Expiry')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Term structure at different strikes
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

# Plot 3: 3D surface
ax3 = fig.add_subplot(133, projection='3d')
K_grid, T_grid = np.meshgrid(strikes, expiries)
V_grid = np.array([[smile_surface.implied_vol(t, k) * 100 
                    for k in strikes] for t in expiries])

surf = ax3.plot_surface(K_grid, T_grid, V_grid, cmap='viridis', alpha=0.8)
ax3.set_xlabel('Strike')
ax3.set_ylabel('Expiry')
ax3.set_zlabel('IV (%)')
ax3.set_title('3D Volatility Surface')
ax3.view_init(elev=25, azim=45)

plt.tight_layout()
plt.savefig('volatility_surface.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nPlot saved to volatility_surface.png")

# =============================================================================
# 5. Using Vol Surfaces in a Market Object
# =============================================================================

print("\n" + "=" * 70)
print("5. Using Vol Surfaces in a Market Object")
print("=" * 70)

# Create market IDs
eurusd_spot_id = MarketId(asset_class="FX", data_type="SPOT", name="EURUSD")
eurusd_vol_id = MarketId(asset_class="FX", data_type="VOL", name="EURUSD")

# Create market with vol surface
# Scale strikes to FX level (spot around 1.08)
fx_spot = 1.0850
fx_strikes = np.linspace(0.95, 1.20, 21)
fx_vol_grid = np.zeros((len(expiries), len(fx_strikes)))

for i, exp in enumerate(expiries):
    for j, k in enumerate(fx_strikes):
        fx_vol_grid[i, j] = generate_smile_vol(exp, k, fx_spot, 
                                                atm_vol=0.08, skew=-0.15, smile=0.10)

fx_vol_surface = GridVolSurface(
    expiries=expiries,
    strikes=fx_strikes,
    vols=fx_vol_grid,
)

market = Market(
    asof="2026-01-28",
    quotes={eurusd_spot_id: Quote(value=fx_spot)},
    curves={},
    vols={eurusd_vol_id: fx_vol_surface},
)

print(f"\nMarket created with EUR/USD vol surface:")
print(f"  As-of: {market.asof}")
print(f"  Spot: {market.quote(eurusd_spot_id)}")

# Query vol surface from market
vol_surface = market.vol_surface(eurusd_vol_id)

print(f"\nEUR/USD Implied Volatility:")
print(f"{'Expiry':<10} {'Strike':<12} {'IV':<10}")
print("-" * 32)

for exp in [0.25, 0.5, 1.0]:
    for strike in [1.00, 1.08, 1.15]:
        iv = vol_surface.implied_vol(exp, strike)
        print(f"{exp:<10.2f} {strike:<12.4f} {iv:<10.2%}")

# =============================================================================
# 6. Delta-Based Quoting Convention
# =============================================================================

print("\n" + "=" * 70)
print("6. Delta-Based Quoting Convention (FX Markets)")
print("=" * 70)

print("""
FX options are often quoted in terms of delta rather than strike:

  25D Put   - 25 delta put (OTM put, ~1 std dev below ATM)
  ATM       - At-the-money (50 delta, or DNS/ATMF)
  25D Call  - 25 delta call (OTM call, ~1 std dev above ATM)

Market quotes typically include:
  - ATM volatility
  - 25D Risk Reversal: σ(25D Call) - σ(25D Put)
  - 25D Butterfly: [σ(25D Call) + σ(25D Put)]/2 - σ(ATM)

Risk Reversal captures the skew (puts vs calls premium)
Butterfly captures the smile (wings vs center premium)
""")

# Illustrative delta-based quotes
atm_vol = 0.08
rr_25d = -0.012  # Negative = puts more expensive (typical FX skew)
bf_25d = 0.005   # Positive = wings more expensive (smile)

# Convert to strike vols
vol_25d_put = atm_vol + bf_25d - rr_25d / 2
vol_25d_call = atm_vol + bf_25d + rr_25d / 2

print(f"\nDelta-based EUR/USD Vol Quotes (1Y):")
print(f"  ATM:        {atm_vol:.2%}")
print(f"  25D RR:     {rr_25d*100:+.2f} vol points")
print(f"  25D BF:     {bf_25d*100:.2f} vol points")
print(f"\nImplied strike vols:")
print(f"  25D Put:    {vol_25d_put:.2%}")
print(f"  ATM:        {atm_vol:.2%}")
print(f"  25D Call:   {vol_25d_call:.2%}")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)

print("""
Key Takeaways:

1. Implied Volatility (IV):
   - Market's expectation of future volatility
   - Backed out from option prices via Black-Scholes
   - Not the same as historical volatility

2. Volatility Smile:
   - IV varies with strike (smile/skew)
   - IV varies with expiry (term structure)
   - Reflects fat tails and crash risk premium

3. Vol Surface Types:
   - FlatVolSurface: Constant vol (for testing)
   - GridVolSurface: Interpolated from grid of points

4. FX Quoting Convention:
   - Delta-based quotes (25D, ATM, 10D)
   - Risk Reversal = skew
   - Butterfly = curvature

5. Market Integration:
   - Vol surfaces stored in Market object
   - Accessed via vol_surface(market_id)

Next: See 04_timeseries_datasets.py for time series data.
""")

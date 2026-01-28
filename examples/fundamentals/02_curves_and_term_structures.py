#!/usr/bin/env python3
"""
Curves and Term Structures: Discount Factors and Interest Rates

This example covers the construction and use of discount curves:
- Discount factors and their meaning
- Zero rates and forward rates
- Interpolation methods
- Building curves from market data

Discount curves are fundamental to derivatives pricing as they
determine the time value of money.

Author: QuantStrata Team
"""

import sys
sys.path.insert(0, '../..')

import numpy as np
import matplotlib.pyplot as plt

from src.marketdata.core.ids import MarketId
from src.marketdata.core.interfaces import Quote
from src.marketdata.core.market import Market
from src.marketdata.curves.term_structure import FlatCurve, ZeroRateCurve

# Plot configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (12, 5),
    'font.size': 11,
    'axes.titlesize': 13,
    'lines.linewidth': 2,
})

# =============================================================================
# 1. Discount Factors: The Time Value of Money
# =============================================================================

print("=" * 70)
print("1. Discount Factors: The Time Value of Money")
print("=" * 70)

print("""
A discount factor df(t) represents the present value of $1 received at time t.

Key relationships:
  - df(0) = 1 (no discounting today)
  - df(t) < 1 for t > 0 (future money is worth less today)
  - df(t) = exp(-r * t) for continuous compounding at rate r

Example: If df(1) = 0.95, then $1 in 1 year is worth $0.95 today.
""")

# Create a simple flat curve (constant rate)
flat_rate = 0.05  # 5% annual rate
flat_curve = FlatCurve(rate=flat_rate)

# Examine discount factors at various tenors
tenors = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]

print(f"\nFlat curve at {flat_rate:.1%} continuous rate:")
print(f"{'Tenor (years)':<15} {'Discount Factor':<18} {'Zero Rate':<12}")
print("-" * 45)
for t in tenors:
    df = flat_curve.df(t)
    zr = flat_curve.zero_rate(t) if t > 0 else 0.0
    print(f"{t:<15.2f} {df:<18.6f} {zr:<12.4%}")

# =============================================================================
# 2. Zero Rates vs Forward Rates
# =============================================================================

print("\n" + "=" * 70)
print("2. Zero Rates vs Forward Rates")
print("=" * 70)

print("""
Zero Rate r(t):
  The annualized rate for borrowing/lending from today to time t.
  df(t) = exp(-r(t) * t)

Forward Rate f(t1, t2):
  The rate locked in today for borrowing/lending from t1 to t2.
  df(t1)/df(t2) = exp(f(t1,t2) * (t2 - t1))
""")

# Create a non-flat curve with upward sloping term structure
tenors_input = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
zero_rates_input = np.array([0.045, 0.047, 0.050, 0.052, 0.055, 0.057])

# Build zero rate curve
zero_curve = ZeroRateCurve(tenors=tenors_input, zero_rates=zero_rates_input)

print(f"\nUpward-sloping zero curve:")
print(f"{'Tenor':<10} {'Zero Rate':<12} {'DF':<15} {'Fwd Rate (to 10Y)':<18}")
print("-" * 55)

for t in [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]:
    zr = zero_curve.zero_rate(t)
    df = zero_curve.df(t)
    # Forward rate from t to 10Y
    if t < 10.0:
        fwd = zero_curve.forward_rate(t, 10.0)
    else:
        fwd = zr
    print(f"{t:<10.2f} {zr:<12.4%} {df:<15.6f} {fwd:<18.4%}")

# =============================================================================
# 3. Visualizing the Term Structure
# =============================================================================

print("\n" + "=" * 70)
print("3. Visualizing the Term Structure")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Dense grid for smooth plots
t_grid = np.linspace(0.01, 10.0, 100)

# Plot 1: Zero Rates
ax = axes[0]
zero_rates = [zero_curve.zero_rate(t) for t in t_grid]
ax.plot(t_grid, np.array(zero_rates) * 100, color='#2E86AB', linewidth=2)
ax.scatter(tenors_input, zero_rates_input * 100, color='#E94F37', s=60, zorder=5, 
           label='Input points')
ax.set_xlabel('Tenor (years)')
ax.set_ylabel('Zero Rate (%)')
ax.set_title('Zero Rate Curve')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Discount Factors
ax = axes[1]
dfs = [zero_curve.df(t) for t in t_grid]
ax.plot(t_grid, dfs, color='#8B5CF6', linewidth=2)
ax.set_xlabel('Tenor (years)')
ax.set_ylabel('Discount Factor')
ax.set_title('Discount Factor Curve')
ax.grid(True, alpha=0.3)

# Plot 3: Forward Rates
ax = axes[2]
# Instantaneous forward rates (approximated)
dt = 0.01
fwd_rates = [(zero_curve.df(t) / zero_curve.df(t + dt) - 1) / dt for t in t_grid[:-1]]
ax.plot(t_grid[:-1], np.array(fwd_rates) * 100, color='#10B981', linewidth=2)
ax.set_xlabel('Tenor (years)')
ax.set_ylabel('Instantaneous Forward Rate (%)')
ax.set_title('Forward Rate Curve')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('term_structure.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nPlot saved to term_structure.png")

# =============================================================================
# 4. Using Curves in a Market Object
# =============================================================================

print("\n" + "=" * 70)
print("4. Using Curves in a Market Object")
print("=" * 70)

# Create market IDs for curves
usd_curve_id = MarketId(asset_class="IR", data_type="CURVE", name="USD_OIS")
eur_curve_id = MarketId(asset_class="IR", data_type="CURVE", name="EUR_OIS")

# Create two different curves
usd_tenors = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
usd_rates = np.array([0.050, 0.051, 0.052, 0.053, 0.055, 0.057])
usd_curve = ZeroRateCurve(tenors=usd_tenors, zero_rates=usd_rates)

eur_tenors = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
eur_rates = np.array([0.035, 0.036, 0.038, 0.040, 0.042, 0.044])
eur_curve = ZeroRateCurve(tenors=eur_tenors, zero_rates=eur_rates)

# Also add spot quote
eurusd_spot_id = MarketId(asset_class="FX", data_type="SPOT", name="EURUSD")

# Create market with curves
market = Market(
    asof="2026-01-28",
    quotes={eurusd_spot_id: Quote(value=1.0850)},
    curves={
        usd_curve_id: usd_curve,
        eur_curve_id: eur_curve,
    },
    vols={},
)

print(f"\nMarket created with curves:")
print(f"  As-of: {market.asof}")
print(f"  Spot EUR/USD: {market.quote(eurusd_spot_id)}")

print(f"\nUSD Curve (5Y discount factor): {market.curve(usd_curve_id).df(5.0):.6f}")
print(f"EUR Curve (5Y discount factor): {market.curve(eur_curve_id).df(5.0):.6f}")

# Rate differential
usd_5y_rate = market.curve(usd_curve_id).zero_rate(5.0)
eur_5y_rate = market.curve(eur_curve_id).zero_rate(5.0)
print(f"\n5Y rate differential (USD - EUR): {(usd_5y_rate - eur_5y_rate)*100:.1f} bps")

# =============================================================================
# 5. FX Forward Pricing with Curves
# =============================================================================

print("\n" + "=" * 70)
print("5. FX Forward Pricing with Curves")
print("=" * 70)

print("""
The FX forward rate is determined by interest rate parity:

  F(T) = S * exp((r_d - r_f) * T)

Where:
  S   = Spot rate
  r_d = Domestic (USD) rate
  r_f = Foreign (EUR) rate
  T   = Time to maturity
""")

spot = market.quote(eurusd_spot_id)
usd_curve = market.curve(usd_curve_id)
eur_curve = market.curve(eur_curve_id)

print(f"\nEUR/USD Forward Rates:")
print(f"{'Tenor':<10} {'USD df':<12} {'EUR df':<12} {'Forward':<12} {'Fwd Points':<12}")
print("-" * 58)

for T in [0.25, 0.5, 1.0, 2.0, 5.0]:
    df_usd = usd_curve.df(T)
    df_eur = eur_curve.df(T)
    
    # Forward = Spot * (df_foreign / df_domestic)
    forward = spot * (df_eur / df_usd)
    fwd_points = (forward - spot) * 10000  # In pips
    
    print(f"{T:<10.2f} {df_usd:<12.6f} {df_eur:<12.6f} {forward:<12.4f} {fwd_points:<12.1f}")

# =============================================================================
# 6. Curve Comparison Plot
# =============================================================================

print("\n" + "=" * 70)
print("6. Curve Comparison: USD vs EUR")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

t_grid = np.linspace(0.1, 10.0, 100)

# Zero rates comparison
ax = axes[0]
usd_zeros = [usd_curve.zero_rate(t) * 100 for t in t_grid]
eur_zeros = [eur_curve.zero_rate(t) * 100 for t in t_grid]

ax.plot(t_grid, usd_zeros, color='#2E86AB', linewidth=2, label='USD OIS')
ax.plot(t_grid, eur_zeros, color='#E94F37', linewidth=2, label='EUR OIS')
ax.fill_between(t_grid, eur_zeros, usd_zeros, alpha=0.2, color='gray')
ax.set_xlabel('Tenor (years)')
ax.set_ylabel('Zero Rate (%)')
ax.set_title('Zero Rate Curves: USD vs EUR')
ax.legend()
ax.grid(True, alpha=0.3)

# Forward curve
ax = axes[1]
forwards = [spot * (eur_curve.df(t) / usd_curve.df(t)) for t in t_grid]

ax.plot(t_grid, forwards, color='#8B5CF6', linewidth=2)
ax.axhline(spot, color='gray', linestyle='--', alpha=0.7, label=f'Spot = {spot}')
ax.set_xlabel('Tenor (years)')
ax.set_ylabel('EUR/USD Forward Rate')
ax.set_title('EUR/USD Forward Curve')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('curve_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nPlot saved to curve_comparison.png")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)

print("""
Key Takeaways:

1. Discount Factor df(t):
   - Present value of $1 received at time t
   - df(t) = exp(-r(t) * t) for continuous compounding

2. Zero Rate r(t):
   - Annualized rate from today to time t
   - Inverted from: r(t) = -ln(df(t)) / t

3. Forward Rate f(t1, t2):
   - Rate locked today for period [t1, t2]
   - f(t1,t2) = ln(df(t1)/df(t2)) / (t2-t1)

4. Curve Types:
   - FlatCurve: Constant rate
   - ZeroRateCurve: Interpolated from tenor points

5. FX Forwards via Interest Rate Parity:
   - F(T) = S * df_foreign(T) / df_domestic(T)

Next: See 03_volatility_surfaces.py for volatility surfaces.
""")

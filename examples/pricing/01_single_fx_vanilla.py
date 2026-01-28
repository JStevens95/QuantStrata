#!/usr/bin/env python3
"""
Single FX Vanilla Option Pricing

This example demonstrates pricing a single European FX vanilla option
using three different methods:
- Black-Scholes-Merton (analytical)
- Monte Carlo simulation
- Finite Difference (PDE)

We compare results, analyze convergence, and discuss trade-offs.

Author: QuantStrata Team
"""

import sys
sys.path.insert(0, '../..')

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

from src.marketdata.core.ids import MarketId
from src.marketdata.core.interfaces import Quote
from src.marketdata.core.market import Market
from src.marketdata.curves.term_structure import FlatCurve
from src.marketdata.surfaces.vol_surface import FlatVolSurface

from src.instruments.fx.options.vanilla import EuropeanFxVanillaOption
from src.pricers.fx.european_bsm import EuropeanFxVanillaBsmPricer
from src.pricers.fx.european_mc import EuropeanFxVanillaMcPricer
from src.pricers.fx.european_fde import EuropeanFxVanillaFdePricer

# Plot configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (12, 5),
    'font.size': 11,
    'axes.titlesize': 13,
    'lines.linewidth': 2,
})

COLORS = {
    'bsm': '#2E86AB',
    'mc': '#E94F37',
    'fd': '#8B5CF6',
}

# =============================================================================
# 1. Setup: Define the Option and Market
# =============================================================================

print("=" * 70)
print("1. Setup: Option and Market")
print("=" * 70)

# Market IDs
EURUSD_SPOT = MarketId(asset_class="FX", data_type="SPOT", name="EURUSD")
USD_CURVE = MarketId(asset_class="IR", data_type="CURVE", name="USD_OIS")
EUR_CURVE = MarketId(asset_class="IR", data_type="CURVE", name="EUR_OIS")
EURUSD_VOL = MarketId(asset_class="FX", data_type="VOL", name="EURUSD")

# Market data
spot = 1.0850
r_usd = 0.05  # 5% USD rate
r_eur = 0.02  # 2% EUR rate
vol = 0.10    # 10% volatility

# Create market
market = Market(
    asof="2026-01-28",
    quotes={EURUSD_SPOT: Quote(value=spot)},
    curves={
        USD_CURVE: FlatCurve(rate=r_usd),
        EUR_CURVE: FlatCurve(rate=r_eur),
    },
    vols={EURUSD_VOL: FlatVolSurface(vol=vol)},
)

# Define the option
strike = 1.1000  # OTM call
expiry_years = 1.0
notional = 1_000_000  # 1M EUR notional

option = EuropeanFxVanillaOption(
    ccy_pair="EURUSD",
    strike=strike,
    expiry_years=expiry_years,
    is_call=True,
    notional=notional,
    spot_id=EURUSD_SPOT,
    domestic_curve_id=USD_CURVE,
    foreign_curve_id=EUR_CURVE,
    vol_id=EURUSD_VOL,
)

print(f"\nOption details:")
print(f"  Currency pair: EUR/USD")
print(f"  Type: {'Call' if option.is_call else 'Put'}")
print(f"  Strike: {strike}")
print(f"  Expiry: {expiry_years} year")
print(f"  Notional: {notional:,.0f} EUR")

print(f"\nMarket data:")
print(f"  Spot: {spot}")
print(f"  USD rate: {r_usd:.2%}")
print(f"  EUR rate: {r_eur:.2%}")
print(f"  Volatility: {vol:.2%}")

# Forward price
fwd = spot * np.exp((r_usd - r_eur) * expiry_years)
print(f"\n  Forward (1Y): {fwd:.4f}")
print(f"  Moneyness (K/F): {strike/fwd:.2%}")

# =============================================================================
# 2. Price with BSM (Analytical)
# =============================================================================

print("\n" + "=" * 70)
print("2. Pricing with Black-Scholes-Merton (Analytical)")
print("=" * 70)

bsm_pricer = EuropeanFxVanillaBsmPricer()
bsm_price = bsm_pricer.price(option, market)
bsm_greeks = bsm_pricer.greeks(option, market)

print(f"\nBSM Results:")
print(f"  Price: {bsm_price:,.2f} USD")
print(f"  Unit price: {bsm_price/notional:.6f}")

print(f"\nGreeks:")
print(f"  Delta: {bsm_greeks['delta']:.4f}")
print(f"  Gamma: {bsm_greeks['gamma']:.4f}")
print(f"  Vega:  {bsm_greeks['vega']:.2f}")
print(f"  Theta: {bsm_greeks['theta']:.2f}")
print(f"  Rho (dom): {bsm_greeks.get('rho_domestic', 0):.2f}")
print(f"  Rho (for): {bsm_greeks.get('rho_foreign', 0):.2f}")

# =============================================================================
# 3. Price with Monte Carlo
# =============================================================================

print("\n" + "=" * 70)
print("3. Pricing with Monte Carlo")
print("=" * 70)

# Standard MC pricer
mc_pricer = EuropeanFxVanillaMcPricer(n_paths=100_000, seed=42)
mc_price = mc_pricer.price(option, market)

print(f"\nMonte Carlo Results (100,000 paths):")
print(f"  Price: {mc_price:,.2f} USD")
print(f"  Unit price: {mc_price/notional:.6f}")
print(f"  Error vs BSM: {abs(mc_price - bsm_price):,.2f} ({abs(mc_price/bsm_price - 1)*100:.3f}%)")

# MC with different path counts
print(f"\nMC convergence:")
print(f"{'Paths':<12} {'Price':<15} {'Error':<15} {'Error %':<10}")
print("-" * 52)

path_counts = [1_000, 10_000, 50_000, 100_000, 500_000]
mc_prices = []

for n_paths in path_counts:
    pricer = EuropeanFxVanillaMcPricer(n_paths=n_paths, seed=42)
    price = pricer.price(option, market)
    mc_prices.append(price)
    error = abs(price - bsm_price)
    print(f"{n_paths:<12,} {price:<15,.2f} {error:<15,.2f} {error/bsm_price*100:<10.3f}%")

# =============================================================================
# 4. Price with Finite Difference
# =============================================================================

print("\n" + "=" * 70)
print("4. Pricing with Finite Difference (PDE)")
print("=" * 70)

fd_pricer = EuropeanFxVanillaFdePricer(n_spot=200, n_time=100)
fd_price = fd_pricer.price(option, market)

print(f"\nFinite Difference Results (200×100 grid):")
print(f"  Price: {fd_price:,.2f} USD")
print(f"  Unit price: {fd_price/notional:.6f}")
print(f"  Error vs BSM: {abs(fd_price - bsm_price):,.2f} ({abs(fd_price/bsm_price - 1)*100:.3f}%)")

# FD with different grid sizes
print(f"\nFD convergence:")
print(f"{'Grid':<15} {'Price':<15} {'Error':<15} {'Error %':<10}")
print("-" * 55)

grid_sizes = [(50, 25), (100, 50), (200, 100), (400, 200)]
fd_prices = []

for n_spot, n_time in grid_sizes:
    pricer = EuropeanFxVanillaFdePricer(n_spot=n_spot, n_time=n_time)
    price = pricer.price(option, market)
    fd_prices.append(price)
    error = abs(price - bsm_price)
    print(f"{n_spot}×{n_time:<9} {price:<15,.2f} {error:<15,.2f} {error/bsm_price*100:<10.4f}%")

# =============================================================================
# 5. Method Comparison
# =============================================================================

print("\n" + "=" * 70)
print("5. Method Comparison Summary")
print("=" * 70)

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                      PRICING METHOD COMPARISON                        ║
╠══════════════════════════════════════════════════════════════════════╣
║  Method              Price (USD)      Error vs BSM     Relative       ║
╠══════════════════════════════════════════════════════════════════════╣
║  BSM (Analytical)    {bsm_price:>12,.2f}       —                —             ║
║  Monte Carlo         {mc_price:>12,.2f}       {abs(mc_price-bsm_price):>8,.2f}         {abs(mc_price/bsm_price-1)*100:>6.3f}%       ║
║  Finite Difference   {fd_price:>12,.2f}       {abs(fd_price-bsm_price):>8,.2f}         {abs(fd_price/bsm_price-1)*100:>6.4f}%       ║
╚══════════════════════════════════════════════════════════════════════╝
""")

print("""
Trade-offs:

BSM (Analytical):
  ✓ Exact (no discretization error)
  ✓ Fastest computation
  ✓ Greeks computed analytically
  ✗ Only for European vanilla with flat vol

Monte Carlo:
  ✓ Handles any payoff structure
  ✓ Handles path-dependent options
  ✓ Handles stochastic volatility
  ✗ Slow convergence O(1/√N)
  ✗ Noisy Greeks

Finite Difference:
  ✓ Greeks directly from solution
  ✓ Handles American options (PSOR)
  ✓ Deterministic (no sampling noise)
  ✗ Limited to low dimensions
  ✗ Boundary condition sensitivity
""")

# =============================================================================
# 6. Visualization
# =============================================================================

print("=" * 70)
print("6. Visualization")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Method comparison across strikes
ax = axes[0]
strikes_plot = np.linspace(0.95, 1.25, 20)
bsm_prices_plot = []
mc_prices_plot = []

for k in strikes_plot:
    opt = EuropeanFxVanillaOption(
        ccy_pair="EURUSD", strike=k, expiry_years=1.0, is_call=True,
        notional=notional, spot_id=EURUSD_SPOT, domestic_curve_id=USD_CURVE,
        foreign_curve_id=EUR_CURVE, vol_id=EURUSD_VOL,
    )
    bsm_prices_plot.append(bsm_pricer.price(opt, market) / notional)

ax.plot(strikes_plot, bsm_prices_plot, '-', color=COLORS['bsm'], linewidth=2, label='BSM')
ax.axvline(spot, color='gray', linestyle='--', alpha=0.5, label=f'Spot = {spot}')
ax.axvline(fwd, color='gray', linestyle=':', alpha=0.5, label=f'Forward = {fwd:.4f}')
ax.set_xlabel('Strike')
ax.set_ylabel('Unit Price')
ax.set_title('Call Price vs Strike')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: MC Convergence
ax = axes[1]
errors = [abs(p - bsm_price) for p in mc_prices]
ax.loglog(path_counts, errors, 'o-', color=COLORS['mc'], linewidth=2, markersize=8)
# Reference line for O(1/√N) convergence
ref_x = np.array(path_counts)
ref_y = errors[0] * np.sqrt(path_counts[0]) / np.sqrt(ref_x)
ax.loglog(ref_x, ref_y, '--', color='gray', alpha=0.7, label=r'$O(1/\sqrt{N})$')
ax.set_xlabel('Number of Paths')
ax.set_ylabel('Absolute Error (USD)')
ax.set_title('MC Convergence')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: FD Convergence
ax = axes[2]
grid_labels = [f'{s}×{t}' for s, t in grid_sizes]
fd_errors = [abs(p - bsm_price) for p in fd_prices]
ax.semilogy(range(len(grid_sizes)), fd_errors, 'o-', color=COLORS['fd'], linewidth=2, markersize=8)
ax.set_xticks(range(len(grid_sizes)))
ax.set_xticklabels(grid_labels)
ax.set_xlabel('Grid Size (Spot × Time)')
ax.set_ylabel('Absolute Error (USD)')
ax.set_title('FD Convergence')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('single_fx_vanilla_pricing.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nPlot saved to single_fx_vanilla_pricing.png")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)

print("""
Key Takeaways:

1. Three pricing methods available:
   - BSM: Fast, exact for European vanilla
   - MC: Flexible, handles complex payoffs
   - FD: Deterministic, handles American options

2. For European vanilla, BSM is the benchmark.

3. MC converges at O(1/√N) — need 4× paths for 2× accuracy.

4. FD converges at O(h²) — doubling grid improves accuracy ~4×.

5. Greeks available from all methods:
   - BSM: Analytical (exact)
   - MC: Pathwise or bump-and-reprice
   - FD: Direct from solution grid

Next: See 02_exotic_options.py for barrier, Asian, lookback options.
""")

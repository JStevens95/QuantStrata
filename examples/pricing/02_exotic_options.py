#!/usr/bin/env python3
"""
Exotic Options Pricing: Barriers, Asians, Lookbacks, and Touch Options

This example demonstrates pricing path-dependent FX options using
Monte Carlo simulation:
- Barrier options (knock-in/knock-out)
- Asian options (arithmetic/geometric averaging)
- Lookback options (floating/fixed strike)
- Touch options (one-touch/no-touch)

Author: QuantStrata Team
"""

import sys
sys.path.insert(0, '../..')

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from src.marketdata.core.ids import MarketId
from src.marketdata.core.interfaces import Quote
from src.marketdata.core.market import Market
from src.marketdata.curves.term_structure import FlatZeroRateCurve
from src.marketdata.surfaces.vol_surface import FlatVolSurface

# Plot configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (14, 8),
    'font.size': 11,
    'axes.titlesize': 13,
    'lines.linewidth': 2,
})

COLORS = {
    'vanilla': '#2E86AB',
    'barrier': '#E94F37',
    'asian': '#8B5CF6',
    'lookback': '#10B981',
    'touch': '#F59E0B',
}

# =============================================================================
# Market Setup
# =============================================================================

print("=" * 70)
print("Market Setup")
print("=" * 70)

# Market parameters
spot = 100.0
r_dom = 0.05
r_for = 0.02
vol = 0.20
T = 1.0

# Market IDs
SPOT_ID = MarketId(asset_class="FX", mkt_type="SPOT", name="TEST")
DOM_CURVE = MarketId(asset_class="IR", mkt_type="CURVE", name="DOM")
FOR_CURVE = MarketId(asset_class="IR", mkt_type="CURVE", name="FOR")
VOL_ID = MarketId(asset_class="FX", mkt_type="VOL", name="TEST")

market = Market(
    asof="2026-01-28",
    quotes={SPOT_ID: Quote(value=spot)},
    curves={
        DOM_CURVE: FlatZeroRateCurve(continuously_compounded_rate=r_dom),
        FOR_CURVE: FlatZeroRateCurve(continuously_compounded_rate=r_for)
    },
    vols={VOL_ID: FlatVolSurface(sigma=vol)},
)

print(f"\nMarket parameters:")
print(f"  Spot: {spot}")
print(f"  Domestic rate: {r_dom:.2%}")
print(f"  Foreign rate: {r_for:.2%}")
print(f"  Volatility: {vol:.2%}")
print(f"  Time to expiry: {T} year")

# Forward price
forward = spot * np.exp((r_dom - r_for) * T)
print(f"  Forward: {forward:.4f}")

# =============================================================================
# Path Simulation
# =============================================================================

def simulate_paths(S0: float, r: float, q: float, sigma: float, T: float,
                   n_paths: int = 50000, n_steps: int = 252, seed: int = 42) -> np.ndarray:
    """Simulate GBM paths."""
    np.random.seed(seed)
    dt = T / n_steps
    
    drift = (r - q - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    
    Z = np.random.randn(n_steps, n_paths)
    log_returns = drift + diffusion * Z
    
    paths = np.zeros((n_steps + 1, n_paths))
    paths[0, :] = S0
    paths[1:, :] = S0 * np.exp(np.cumsum(log_returns, axis=0))
    
    return paths

print("\nSimulating 50,000 paths with 252 steps...")
paths = simulate_paths(spot, r_dom, r_for, vol, T)
print(f"  Paths shape: {paths.shape}")

# =============================================================================
# 1. Vanilla Option (Benchmark)
# =============================================================================

print("\n" + "=" * 70)
print("1. Vanilla Option (Benchmark)")
print("=" * 70)

K = 100.0  # ATM strike

def price_vanilla(paths: np.ndarray, K: float, r: float, T: float, 
                  option_type: str = 'call') -> Tuple[float, float]:
    terminal = paths[-1, :]
    if option_type == 'call':
        payoffs = np.maximum(terminal - K, 0)
    else:
        payoffs = np.maximum(K - terminal, 0)
    
    disc_payoffs = np.exp(-r * T) * payoffs
    price = np.mean(disc_payoffs)
    stderr = np.std(disc_payoffs) / np.sqrt(len(disc_payoffs))
    return price, stderr

vanilla_call, vanilla_se = price_vanilla(paths, K, r_dom, T, 'call')
vanilla_put, _ = price_vanilla(paths, K, r_dom, T, 'put')

print(f"\nVanilla option prices (K={K}):")
print(f"  Call: {vanilla_call:.4f} ± {vanilla_se*1.96:.4f}")
print(f"  Put:  {vanilla_put:.4f}")

# =============================================================================
# 2. Barrier Options
# =============================================================================

print("\n" + "=" * 70)
print("2. Barrier Options")
print("=" * 70)

def price_barrier(paths: np.ndarray, K: float, B: float, r: float, T: float,
                  option_type: str = 'call', barrier_type: str = 'up_and_out',
                  rebate: float = 0.0) -> Tuple[float, float]:
    terminal = paths[-1, :]
    
    if barrier_type.startswith('up'):
        breached = np.any(paths >= B, axis=0)
    else:
        breached = np.any(paths <= B, axis=0)
    
    if option_type == 'call':
        vanilla = np.maximum(terminal - K, 0)
    else:
        vanilla = np.maximum(K - terminal, 0)
    
    if barrier_type.endswith('out'):
        payoffs = np.where(breached, rebate, vanilla)
    else:
        payoffs = np.where(breached, vanilla, 0)
    
    disc_payoffs = np.exp(-r * T) * payoffs
    price = np.mean(disc_payoffs)
    stderr = np.std(disc_payoffs) / np.sqrt(len(disc_payoffs))
    return price, stderr

# Up-and-out call
B_up = 115.0
uo_call, uo_se = price_barrier(paths, K, B_up, r_dom, T, 'call', 'up_and_out')
ui_call, ui_se = price_barrier(paths, K, B_up, r_dom, T, 'call', 'up_and_in')

# Down-and-out put
B_down = 85.0
do_put, do_se = price_barrier(paths, K, B_down, r_dom, T, 'put', 'down_and_out')
di_put, di_se = price_barrier(paths, K, B_down, r_dom, T, 'put', 'down_and_in')

print(f"\nBarrier option prices:")
print(f"  Up-and-Out Call (K={K}, B={B_up}):")
print(f"    Price: {uo_call:.4f} ± {uo_se*1.96:.4f}")
print(f"    Discount vs Vanilla: {(1-uo_call/vanilla_call)*100:.1f}%")
print(f"\n  Up-and-In Call (K={K}, B={B_up}):")
print(f"    Price: {ui_call:.4f} ± {ui_se*1.96:.4f}")
print(f"\n  In-Out Parity Check: KO + KI = {uo_call + ui_call:.4f} vs Vanilla = {vanilla_call:.4f}")

print(f"\n  Down-and-Out Put (K={K}, B={B_down}):")
print(f"    Price: {do_put:.4f} ± {do_se*1.96:.4f}")
print(f"    Discount vs Vanilla: {(1-do_put/vanilla_put)*100:.1f}%")

# =============================================================================
# 3. Asian Options
# =============================================================================

print("\n" + "=" * 70)
print("3. Asian Options")
print("=" * 70)

def price_asian(paths: np.ndarray, K: float, r: float, T: float,
                option_type: str = 'call', 
                avg_type: str = 'arithmetic') -> Tuple[float, float]:
    if avg_type == 'arithmetic':
        avg = np.mean(paths, axis=0)
    else:
        avg = np.exp(np.mean(np.log(paths), axis=0))
    
    if option_type == 'call':
        payoffs = np.maximum(avg - K, 0)
    else:
        payoffs = np.maximum(K - avg, 0)
    
    disc_payoffs = np.exp(-r * T) * payoffs
    price = np.mean(disc_payoffs)
    stderr = np.std(disc_payoffs) / np.sqrt(len(disc_payoffs))
    return price, stderr

asian_arith_call, asian_se = price_asian(paths, K, r_dom, T, 'call', 'arithmetic')
asian_geom_call, _ = price_asian(paths, K, r_dom, T, 'call', 'geometric')
asian_arith_put, _ = price_asian(paths, K, r_dom, T, 'put', 'arithmetic')

print(f"\nAsian option prices (K={K}):")
print(f"  Arithmetic Average Call: {asian_arith_call:.4f} ± {asian_se*1.96:.4f}")
print(f"  Geometric Average Call:  {asian_geom_call:.4f}")
print(f"  Arithmetic Average Put:  {asian_arith_put:.4f}")
print(f"\n  Discount vs Vanilla Call: {(1-asian_arith_call/vanilla_call)*100:.1f}%")
print(f"  (Asian cheaper due to averaging → lower volatility)")

# Show volatility reduction
terminal = paths[-1, :]
arith_avg = np.mean(paths, axis=0)
print(f"\n  Std(Terminal): {np.std(terminal):.4f}")
print(f"  Std(Average):  {np.std(arith_avg):.4f}")
print(f"  Reduction:     {(1-np.std(arith_avg)/np.std(terminal))*100:.1f}%")

# =============================================================================
# 4. Lookback Options
# =============================================================================

print("\n" + "=" * 70)
print("4. Lookback Options")
print("=" * 70)

def price_lookback(paths: np.ndarray, k: float, r: float, t: float,
                   option_type: str = 'call',
                   lookback_type: str = 'floating') -> Tuple[float, float]:
    terminal = paths[-1, :]
    
    if lookback_type == 'floating':
        if option_type == 'call':
            min_S = np.min(paths, axis=0)
            payoffs = terminal - min_S
        else:
            max_S = np.max(paths, axis=0)
            payoffs = max_S - terminal
    else:
        if option_type == 'call':
            max_S = np.max(paths, axis=0)
            payoffs = np.maximum(max_S - k, 0)
        else:
            min_S = np.min(paths, axis=0)
            payoffs = np.maximum(k - min_S, 0)
    
    disc_payoffs = np.exp(-r * t) * payoffs
    price = np.mean(disc_payoffs)
    stderr = np.std(disc_payoffs) / np.sqrt(len(disc_payoffs))
    return price, stderr

float_call, float_se = price_lookback(paths, K, r_dom, T, 'call', 'floating')
float_put, _ = price_lookback(paths, K, r_dom, T, 'put', 'floating')
fixed_call, _ = price_lookback(paths, K, r_dom, T, 'call', 'fixed')
fixed_put, _ = price_lookback(paths, K, r_dom, T, 'put', 'fixed')

print(f"\nLookback option prices:")
print(f"  Floating Strike Call: {float_call:.4f} ± {float_se*1.96:.4f}")
print(f"  Floating Strike Put:  {float_put:.4f}")
print(f"  Fixed Strike Call (K={K}): {fixed_call:.4f}")
print(f"  Fixed Strike Put (K={K}):  {fixed_put:.4f}")
print(f"\n  Premium vs Vanilla Call: {float_call/vanilla_call:.1f}x")
print(f"  (Lookback is significantly more expensive)")

# =============================================================================
# 5. Touch Options
# =============================================================================

print("\n" + "=" * 70)
print("5. Touch Options")
print("=" * 70)

def price_touch(paths: np.ndarray, B: float, r: float, T: float,
                direction: str = 'up', touch_type: str = 'one_touch',
                payout: float = 1.0) -> Tuple[float, float]:
    if direction == 'up':
        touched = np.any(paths >= B, axis=0)
    else:
        touched = np.any(paths <= B, axis=0)
    
    if touch_type == 'one_touch':
        payoffs = np.where(touched, payout, 0)
    else:
        payoffs = np.where(touched, 0, payout)
    
    disc_payoffs = np.exp(-r * T) * payoffs
    price = np.mean(disc_payoffs)
    stderr = np.std(disc_payoffs) / np.sqrt(len(disc_payoffs))
    return price, stderr

B_touch = 115.0
one_touch_up, ot_se = price_touch(paths, B_touch, r_dom, T, 'up', 'one_touch')
no_touch_up, _ = price_touch(paths, B_touch, r_dom, T, 'up', 'no_touch')

B_touch_down = 85.0
one_touch_down, _ = price_touch(paths, B_touch_down, r_dom, T, 'down', 'one_touch')
no_touch_down, _ = price_touch(paths, B_touch_down, r_dom, T, 'down', 'no_touch')

print(f"\nTouch option prices (payout = 1.0):")
print(f"  One-Touch Up (B={B_touch}):")
print(f"    Price: {one_touch_up:.4f} ± {ot_se*1.96:.4f}")
print(f"    Probability of touch: {one_touch_up * np.exp(r_dom * T) * 100:.1f}%")
print(f"\n  No-Touch Up (B={B_touch}):")
print(f"    Price: {no_touch_up:.4f}")
print(f"\n  Touch Parity Check: OT + NT = {one_touch_up + no_touch_up:.4f} vs df = {np.exp(-r_dom*T):.4f}")

print(f"\n  One-Touch Down (B={B_touch_down}): {one_touch_down:.4f}")
print(f"  No-Touch Down (B={B_touch_down}):  {no_touch_down:.4f}")

# =============================================================================
# 6. Summary Comparison
# =============================================================================

print("\n" + "=" * 70)
print("6. Summary Comparison")
print("=" * 70)

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    EXOTIC OPTIONS PRICE COMPARISON                    ║
╠══════════════════════════════════════════════════════════════════════╣
║  Option Type                    Price        vs Vanilla               ║
╠══════════════════════════════════════════════════════════════════════╣
║  Vanilla Call (K=100)           {vanilla_call:>8.4f}      (benchmark)             ║
║  Up-Out Call (K=100, B=115)     {uo_call:>8.4f}      {(uo_call/vanilla_call)*100:>6.1f}% (cheaper)       ║
║  Asian Call (Arithmetic)        {asian_arith_call:>8.4f}      {(asian_arith_call/vanilla_call)*100:>6.1f}% (cheaper)       ║
║  Lookback Call (Floating)       {float_call:>8.4f}      {(float_call/vanilla_call)*100:>6.1f}% (expensive)     ║
╠══════════════════════════════════════════════════════════════════════╣
║  One-Touch Up (B=115)           {one_touch_up:>8.4f}      Prob: {one_touch_up*np.exp(r_dom*T)*100:>5.1f}%           ║
╚══════════════════════════════════════════════════════════════════════╝
""")

# =============================================================================
# 7. Visualization
# =============================================================================

print("=" * 70)
print("7. Visualization")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Sample paths with barrier
ax = axes[0, 0]
time_grid = np.linspace(0, T, paths.shape[0])
B = 115.0

for i in range(min(50, paths.shape[1])):
    path = paths[:, i]
    hit_idx = np.where(path >= B)[0]
    if len(hit_idx) > 0:
        ax.plot(time_grid[:hit_idx[0]+1], path[:hit_idx[0]+1], 
               color=COLORS['barrier'], alpha=0.3, linewidth=0.8)
    else:
        ax.plot(time_grid, path, color=COLORS['vanilla'], alpha=0.3, linewidth=0.8)

ax.axhline(B, color='red', linestyle='--', linewidth=2, label=f'Barrier = {B}')
ax.axhline(K, color='gray', linestyle=':', alpha=0.7, label=f'Strike = {K}')
ax.set_xlabel('Time (years)')
ax.set_ylabel('Spot Price')
ax.set_title('Barrier Option: Path Visualization')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Price comparison
ax = axes[0, 1]
products = ['Vanilla\nCall', 'Up-Out\nCall', 'Asian\nCall', 'Lookback\nCall']
prices = [vanilla_call, uo_call, asian_arith_call, float_call]
colors = [COLORS['vanilla'], COLORS['barrier'], COLORS['asian'], COLORS['lookback']]

bars = ax.bar(products, prices, color=colors)
ax.axhline(vanilla_call, color='gray', linestyle='--', alpha=0.5)
ax.set_ylabel('Option Price')
ax.set_title('Call Option Price Comparison')
ax.grid(True, alpha=0.3, axis='y')

for bar, price in zip(bars, prices):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
           f'{price:.2f}', ha='center', fontsize=10)

# Plot 3: Asian averaging effect
ax = axes[1, 0]
ax.hist(terminal, bins=50, alpha=0.5, density=True, color=COLORS['vanilla'], label='Terminal')
ax.hist(arith_avg, bins=50, alpha=0.5, density=True, color=COLORS['asian'], label='Average')
ax.axvline(K, color='gray', linestyle='--', alpha=0.7)
ax.set_xlabel('Price')
ax.set_ylabel('Density')
ax.set_title('Asian Option: Averaging Reduces Volatility')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Lookback payoff distribution
ax = axes[1, 1]
lb_payoffs = terminal - np.min(paths, axis=0)
vanilla_payoffs = np.maximum(terminal - K, 0)

ax.hist(vanilla_payoffs, bins=50, alpha=0.5, density=True, color=COLORS['vanilla'], label='Vanilla Call')
ax.hist(lb_payoffs, bins=50, alpha=0.5, density=True, color=COLORS['lookback'], label='Lookback Call')
ax.set_xlabel('Payoff')
ax.set_ylabel('Density')
ax.set_title('Lookback vs Vanilla: Payoff Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
# plt.savefig('exotic_options_pricing.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nPlot saved to exotic_options_pricing.png")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)

print("""
Key Takeaways:

1. Barrier Options:
   - Knock-out: Cheaper than vanilla (barrier risk)
   - Knock-in: Remaining value after knock-out
   - In-Out Parity: KI + KO = Vanilla

2. Asian Options:
   - Cheaper than vanilla (averaging reduces effective vol)
   - Arithmetic > Geometric (Jensen's inequality)
   - Popular for hedging average exposures

3. Lookback Options:
   - Most expensive (guaranteed best entry/exit)
   - Floating: Buy at min, sell at max
   - Fixed: Call on path maximum

4. Touch Options:
   - Binary payoff (all or nothing)
   - One-Touch + No-Touch = Discount factor
   - Used for range bets

5. All path-dependent options require MC simulation.

Next: See 03_american_options.py for early exercise.
""")

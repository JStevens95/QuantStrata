#!/usr/bin/env python3
"""
Exotic Options Gallery: Barriers, Asians, Lookbacks, and Touch Options

This example showcases the exotic option products implemented in QuantStrata,
demonstrating their unique characteristics, pricing methods, and risk profiles.

Topics Covered:
- Barrier options (knock-in/knock-out)
- Asian options (arithmetic/geometric averaging)
- Lookback options (floating/fixed strike)
- Touch options (one-touch, no-touch)
- Path-dependent payoff visualization
- Risk profile comparison

Author: QuantStrata Team
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from dataclasses import dataclass

# Path setup
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# QuantStrata imports - use library dynamics
from src.models.dynamics.gbm_dynamics import GbmDynamicsSimulator, GbmScheme

# =============================================================================
# Configuration
# =============================================================================

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (14, 8),
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'lines.linewidth': 2,
})

COLORS = {
    'vanilla': '#2E86AB',
    'barrier': '#E94F37',
    'asian': '#8B5CF6',
    'lookback': '#10B981',
    'touch': '#F59E0B',
    'path': '#64748B',
}

@dataclass
class MarketParams:
    """Common market parameters."""
    S0: float = 100.0        # Initial spot
    r: float = 0.05          # Risk-free rate
    q: float = 0.02          # Dividend yield
    sigma: float = 0.20      # Volatility
    T: float = 1.0           # Time to expiry

# =============================================================================
# Path Simulation
# =============================================================================

def simulate_gbm_paths(params: MarketParams, n_paths: int = 10000, 
                       n_steps: int = 252, seed: int = 42) -> np.ndarray:
    """
    Simulate GBM paths using QuantStrata's library dynamics.
    
    Uses GbmDynamicsSimulator with LOG_EULER scheme for numerical stability.
    """
    # Use library GBM simulator
    simulator = GbmDynamicsSimulator(scheme=GbmScheme.LOG_EULER)
    
    # Drift for risk-neutral measure: r - q
    drift = params.r - params.q
    
    # Simulate paths (returns shape (n_paths, n_steps + 1))
    paths_raw = simulator.simulate(
        S0=params.S0,
        drift=drift,
        sigma=params.sigma,
        T=params.T,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
        antithetic=True,
    )
    
    # Transpose to (n_steps + 1, n_paths) for compatibility
    return paths_raw.T

# =============================================================================
# Barrier Option Pricing
# =============================================================================

def price_barrier_option(paths: np.ndarray, K: float, B: float, r: float, T: float,
                         option_type: str = 'call', barrier_type: str = 'up_and_out',
                         rebate: float = 0.0) -> Tuple[float, float]:
    """
    Price barrier option via Monte Carlo.
    
    Parameters
    ----------
    barrier_type : str
        'up_and_out', 'up_and_in', 'down_and_out', 'down_and_in'
    rebate : float
        Rebate paid if barrier is breached (for knock-out)
    """
    terminal = paths[-1, :]
    
    # Check barrier breach
    if barrier_type.startswith('up'):
        breached = np.any(paths >= B, axis=0)
    else:
        breached = np.any(paths <= B, axis=0)
    
    # Vanilla payoff
    if option_type == 'call':
        vanilla = np.maximum(terminal - K, 0)
    else:
        vanilla = np.maximum(K - terminal, 0)
    
    # Apply barrier logic
    if barrier_type.endswith('out'):
        payoffs = np.where(breached, rebate, vanilla)
    else:  # knock-in
        payoffs = np.where(breached, vanilla, 0)
    
    # Discount and compute
    disc_payoffs = np.exp(-r*T) * payoffs
    price = np.mean(disc_payoffs)
    stderr = np.std(disc_payoffs) / np.sqrt(len(disc_payoffs))
    
    return price, stderr

# =============================================================================
# Asian Option Pricing
# =============================================================================

def price_asian_option(paths: np.ndarray, K: float, r: float, T: float,
                       option_type: str = 'call', 
                       avg_type: str = 'arithmetic') -> Tuple[float, float]:
    """Price Asian option via Monte Carlo."""
    
    # Compute average
    if avg_type == 'arithmetic':
        avg = np.mean(paths, axis=0)
    else:  # geometric
        avg = np.exp(np.mean(np.log(paths), axis=0))
    
    # Payoff
    if option_type == 'call':
        payoffs = np.maximum(avg - K, 0)
    else:
        payoffs = np.maximum(K - avg, 0)
    
    disc_payoffs = np.exp(-r*T) * payoffs
    price = np.mean(disc_payoffs)
    stderr = np.std(disc_payoffs) / np.sqrt(len(disc_payoffs))
    
    return price, stderr

# =============================================================================
# Lookback Option Pricing
# =============================================================================

def price_lookback_option(paths: np.ndarray, K: float, r: float, T: float,
                          option_type: str = 'call',
                          lookback_type: str = 'floating') -> Tuple[float, float]:
    """Price lookback option via Monte Carlo."""
    terminal = paths[-1, :]
    
    if lookback_type == 'floating':
        if option_type == 'call':
            # Buy at minimum
            min_S = np.min(paths, axis=0)
            payoffs = terminal - min_S
        else:
            # Sell at maximum
            max_S = np.max(paths, axis=0)
            payoffs = max_S - terminal
    else:  # fixed strike
        if option_type == 'call':
            max_S = np.max(paths, axis=0)
            payoffs = np.maximum(max_S - K, 0)
        else:
            min_S = np.min(paths, axis=0)
            payoffs = np.maximum(K - min_S, 0)
    
    disc_payoffs = np.exp(-r*T) * payoffs
    price = np.mean(disc_payoffs)
    stderr = np.std(disc_payoffs) / np.sqrt(len(disc_payoffs))
    
    return price, stderr

# =============================================================================
# Touch Option Pricing
# =============================================================================

def price_touch_option(paths: np.ndarray, B: float, r: float, T: float,
                       direction: str = 'up', touch_type: str = 'one_touch',
                       payout: float = 1.0) -> Tuple[float, float]:
    """Price touch option via Monte Carlo."""
    
    # Check if barrier is touched
    if direction == 'up':
        touched = np.any(paths >= B, axis=0)
    else:
        touched = np.any(paths <= B, axis=0)
    
    # Payoff
    if touch_type == 'one_touch':
        payoffs = np.where(touched, payout, 0)
    else:  # no_touch
        payoffs = np.where(touched, 0, payout)
    
    disc_payoffs = np.exp(-r*T) * payoffs
    price = np.mean(disc_payoffs)
    stderr = np.std(disc_payoffs) / np.sqrt(len(disc_payoffs))
    
    return price, stderr

# =============================================================================
# Visualization Functions
# =============================================================================

def plot_barrier_analysis(params: MarketParams, paths: np.ndarray):
    """Visualize barrier option behavior."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    K = 100
    barriers = np.linspace(105, 130, 20)
    
    # Sample paths with barrier
    ax = axes[0, 0]
    time_grid = np.linspace(0, params.T, paths.shape[0])
    B = 115
    
    for i in range(min(50, paths.shape[1])):
        path = paths[:, i]
        hit_idx = np.where(path >= B)[0]
        
        if len(hit_idx) > 0:
            ax.plot(time_grid[:hit_idx[0]+1], path[:hit_idx[0]+1], 
                   color=COLORS['barrier'], alpha=0.3, linewidth=0.8)
            ax.scatter(time_grid[hit_idx[0]], path[hit_idx[0]], 
                      color=COLORS['barrier'], s=20, alpha=0.5)
        else:
            ax.plot(time_grid, path, color=COLORS['vanilla'], alpha=0.3, linewidth=0.8)
    
    ax.axhline(B, color='red', linestyle='--', linewidth=2, label=f'Barrier = {B}')
    ax.axhline(K, color='gray', linestyle=':', alpha=0.7, label=f'Strike = {K}')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Spot Price')
    ax.set_title('Up-and-Out Barrier: Path Visualization')
    ax.legend()
    
    # Barrier level impact
    ax = axes[0, 1]
    uo_prices = [price_barrier_option(paths, K, B, params.r, params.T, 
                                       'call', 'up_and_out')[0] for B in barriers]
    ui_prices = [price_barrier_option(paths, K, B, params.r, params.T,
                                       'call', 'up_and_in')[0] for B in barriers]
    
    ax.plot(barriers, uo_prices, '-o', color=COLORS['barrier'], 
           markersize=6, label='Up-and-Out')
    ax.plot(barriers, ui_prices, '-s', color=COLORS['asian'],
           markersize=6, label='Up-and-In')
    ax.axhline(price_barrier_option(paths, K, 1000, params.r, params.T,
                                    'call', 'up_and_out')[0],
              color='gray', linestyle='--', alpha=0.7, label='Vanilla')
    
    ax.set_xlabel('Barrier Level')
    ax.set_ylabel('Option Price')
    ax.set_title('Barrier Level Impact on Price')
    ax.legend()
    
    # In-Out parity check
    ax = axes[1, 0]
    vanilla_price = price_barrier_option(paths, K, 1000, params.r, params.T,
                                         'call', 'up_and_out')[0]
    
    sum_prices = [uo + ui for uo, ui in zip(uo_prices, ui_prices)]
    
    ax.plot(barriers, sum_prices, '-o', color=COLORS['lookback'],
           markersize=6, label='KO + KI')
    ax.axhline(vanilla_price, color=COLORS['vanilla'], linestyle='--',
              linewidth=2, label='Vanilla')
    
    ax.set_xlabel('Barrier Level')
    ax.set_ylabel('Sum of Prices')
    ax.set_title('In-Out Parity: KO + KI = Vanilla')
    ax.legend()
    
    # Payoff comparison
    ax = axes[1, 1]
    spots = np.linspace(80, 130, 100)
    B = 115
    
    vanilla_payoff = np.maximum(spots - K, 0)
    uo_payoff = np.where(spots < B, vanilla_payoff, 0)
    
    ax.fill_between(spots, 0, vanilla_payoff, alpha=0.3, 
                   color=COLORS['vanilla'], label='Vanilla Call')
    ax.plot(spots, uo_payoff, color=COLORS['barrier'], linewidth=2.5,
           label='Up-and-Out Call')
    ax.axvline(B, color='red', linestyle='--', label=f'Barrier = {B}')
    ax.axvline(K, color='gray', linestyle=':', alpha=0.7)
    
    ax.set_xlabel('Terminal Spot')
    ax.set_ylabel('Payoff')
    ax.set_title('Barrier Option Payoff Profile')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('barrier_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_asian_analysis(params: MarketParams, paths: np.ndarray):
    """Visualize Asian option behavior."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    K = 100
    
    # Sample paths with average
    ax = axes[0, 0]
    time_grid = np.linspace(0, params.T, paths.shape[0])
    
    for i in range(min(20, paths.shape[1])):
        path = paths[:, i]
        ax.plot(time_grid, path, color=COLORS['path'], alpha=0.3, linewidth=0.8)
        
        # Running average
        avg = np.cumsum(path) / np.arange(1, len(path)+1)
        ax.plot(time_grid, avg, color=COLORS['asian'], alpha=0.4, linewidth=1)
    
    ax.axhline(K, color='gray', linestyle='--', alpha=0.7, label=f'Strike = {K}')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Price')
    ax.set_title('Asian Option: Spot and Running Average')
    ax.legend(['Spot paths', 'Running averages', 'Strike'])
    
    # Arithmetic vs Geometric
    ax = axes[0, 1]
    strikes = np.linspace(90, 110, 20)
    
    arith_prices = [price_asian_option(paths, k, params.r, params.T, 
                                       'call', 'arithmetic')[0] for k in strikes]
    geom_prices = [price_asian_option(paths, k, params.r, params.T,
                                      'call', 'geometric')[0] for k in strikes]
    
    ax.plot(strikes, arith_prices, '-o', color=COLORS['asian'],
           markersize=6, label='Arithmetic Asian')
    ax.plot(strikes, geom_prices, '-s', color=COLORS['lookback'],
           markersize=6, label='Geometric Asian')
    
    ax.set_xlabel('Strike')
    ax.set_ylabel('Option Price')
    ax.set_title('Arithmetic vs Geometric Average')
    ax.legend()
    
    # Price distribution
    ax = axes[1, 0]
    arith_avg = np.mean(paths, axis=0)
    geom_avg = np.exp(np.mean(np.log(paths), axis=0))
    terminal = paths[-1, :]
    
    ax.hist(terminal, bins=50, alpha=0.5, density=True, 
           color=COLORS['vanilla'], label='Terminal Spot')
    ax.hist(arith_avg, bins=50, alpha=0.5, density=True,
           color=COLORS['asian'], label='Arithmetic Avg')
    ax.hist(geom_avg, bins=50, alpha=0.5, density=True,
           color=COLORS['lookback'], label='Geometric Avg')
    
    ax.axvline(K, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel('Price')
    ax.set_ylabel('Density')
    ax.set_title('Distribution: Terminal vs Average')
    ax.legend()
    
    # Volatility reduction
    ax = axes[1, 1]
    avg_types = ['Terminal', 'Arith Avg', 'Geom Avg']
    stds = [np.std(terminal), np.std(arith_avg), np.std(geom_avg)]
    
    bars = ax.bar(avg_types, stds, color=[COLORS['vanilla'], 
                                          COLORS['asian'], COLORS['lookback']])
    
    ax.set_ylabel('Standard Deviation')
    ax.set_title('Averaging Reduces Volatility')
    
    for bar, std in zip(bars, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{std:.2f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('asian_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_lookback_analysis(params: MarketParams, paths: np.ndarray):
    """Visualize lookback option behavior."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    K = 100
    
    # Sample paths with extrema
    ax = axes[0, 0]
    time_grid = np.linspace(0, params.T, paths.shape[0])
    
    for i in range(min(10, paths.shape[1])):
        path = paths[:, i]
        ax.plot(time_grid, path, color=COLORS['path'], alpha=0.5, linewidth=1)
        
        # Mark maximum and minimum
        max_idx = np.argmax(path)
        min_idx = np.argmin(path)
        
        ax.scatter(time_grid[max_idx], path[max_idx], color=COLORS['barrier'],
                  s=60, zorder=5, marker='^')
        ax.scatter(time_grid[min_idx], path[min_idx], color=COLORS['lookback'],
                  s=60, zorder=5, marker='v')
    
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Spot Price')
    ax.set_title('Lookback Options: Path Extrema')
    ax.legend(['Paths', 'Maximum', 'Minimum'], loc='upper right')
    
    # Floating vs Fixed
    ax = axes[0, 1]
    
    products = ['Float Call', 'Float Put', 'Fixed Call', 'Fixed Put']
    
    float_call, _ = price_lookback_option(paths, K, params.r, params.T, 'call', 'floating')
    float_put, _ = price_lookback_option(paths, K, params.r, params.T, 'put', 'floating')
    fixed_call, _ = price_lookback_option(paths, K, params.r, params.T, 'call', 'fixed')
    fixed_put, _ = price_lookback_option(paths, K, params.r, params.T, 'put', 'fixed')
    
    prices = [float_call, float_put, fixed_call, fixed_put]
    colors_list = [COLORS['lookback'], COLORS['barrier'], 
                   COLORS['asian'], COLORS['touch']]
    
    bars = ax.bar(products, prices, color=colors_list)
    ax.set_ylabel('Option Price')
    ax.set_title('Lookback Option Prices')
    
    for bar, price in zip(bars, prices):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{price:.2f}', ha='center', fontsize=10)
    
    # Payoff comparison
    ax = axes[1, 0]
    terminal = paths[-1, :]
    min_S = np.min(paths, axis=0)
    max_S = np.max(paths, axis=0)
    
    # Scatter plot of payoffs
    float_call_payoffs = terminal - min_S
    vanilla_call_payoffs = np.maximum(terminal - K, 0)
    
    ax.scatter(terminal, vanilla_call_payoffs, alpha=0.3, s=10,
              color=COLORS['vanilla'], label='Vanilla Call')
    ax.scatter(terminal, float_call_payoffs, alpha=0.3, s=10,
              color=COLORS['lookback'], label='Floating Call')
    
    ax.set_xlabel('Terminal Spot')
    ax.set_ylabel('Payoff')
    ax.set_title('Lookback vs Vanilla: Payoff Scatter')
    ax.legend()
    
    # Premium analysis
    ax = axes[1, 1]
    
    # Vanilla prices for reference using library BSM
    from src.models.analytic.black_scholes_merton.base import vanilla_price
    carry = params.r - params.q
    vanilla_call_bsm = vanilla_price(
        option_type="call", spot=params.S0, strike=K, expiry=params.T,
        discount_rate=params.r, carry=carry, vol=params.sigma
    )
    vanilla_put_bsm = vanilla_price(
        option_type="put", spot=params.S0, strike=K, expiry=params.T,
        discount_rate=params.r, carry=carry, vol=params.sigma
    )
    
    categories = ['Call (Float)', 'Call (Fixed)', 'Put (Float)', 'Put (Fixed)']
    lookback_prices = [float_call, fixed_call, float_put, fixed_put]
    vanilla_ref = [vanilla_call_bsm, vanilla_call_bsm, vanilla_put_bsm, vanilla_put_bsm]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax.bar(x - width/2, lookback_prices, width, label='Lookback', color=COLORS['lookback'])
    ax.bar(x + width/2, vanilla_ref, width, label='Vanilla', color=COLORS['vanilla'])
    
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=15)
    ax.set_ylabel('Price')
    ax.set_title('Lookback Premium over Vanilla')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('lookback_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_exotic_comparison(params: MarketParams, paths: np.ndarray):
    """Compare all exotic option types."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    K = params.S0  # ATM
    B_up = params.S0 * 1.15
    B_down = params.S0 * 0.85
    
    # Compute all prices
    products = {
        'Vanilla Call': price_barrier_option(paths, K, 1000, params.r, params.T, 'call', 'up_and_out'),
        'Up-Out Call': price_barrier_option(paths, K, B_up, params.r, params.T, 'call', 'up_and_out'),
        'Down-Out Put': price_barrier_option(paths, K, B_down, params.r, params.T, 'put', 'down_and_out'),
        'Asian Call': price_asian_option(paths, K, params.r, params.T, 'call', 'arithmetic'),
        'Lookback Call': price_lookback_option(paths, K, params.r, params.T, 'call', 'floating'),
        'One-Touch Up': price_touch_option(paths, B_up, params.r, params.T, 'up', 'one_touch'),
    }
    
    # Price comparison
    ax = axes[0]
    names = list(products.keys())
    prices = [products[n][0] for n in names]
    errors = [products[n][1] * 1.96 for n in names]
    colors_list = [COLORS['vanilla'], COLORS['barrier'], COLORS['barrier'],
                   COLORS['asian'], COLORS['lookback'], COLORS['touch']]
    
    bars = ax.barh(names, prices, xerr=errors, capsize=5, color=colors_list)
    ax.set_xlabel('Option Price')
    ax.set_title('Exotic Options: Price Comparison (ATM)')
    
    # Risk profiles
    ax = axes[1]
    spots_range = np.linspace(0.8, 1.2, 50) * params.S0
    
    # Compute prices across spot range (simplified - using terminal payoff)
    vanilla_profile = np.maximum(spots_range - K, 0) * np.exp(-params.r * params.T)
    barrier_profile = np.where(spots_range < B_up, 
                               np.maximum(spots_range - K, 0), 0) * np.exp(-params.r * params.T)
    
    ax.plot(spots_range, vanilla_profile, color=COLORS['vanilla'], 
           linewidth=2, label='Vanilla Call')
    ax.plot(spots_range, barrier_profile, color=COLORS['barrier'],
           linewidth=2, label='Up-Out Call')
    
    ax.axvline(K, color='gray', linestyle=':', alpha=0.5, label='Strike')
    ax.axvline(B_up, color='red', linestyle='--', alpha=0.7, label='Barrier')
    
    ax.set_xlabel('Terminal Spot')
    ax.set_ylabel('Discounted Payoff')
    ax.set_title('Payoff Profiles')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('exotic_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Run exotic options showcase."""
    print("=" * 70)
    print("Exotic Options Gallery")
    print("=" * 70)
    
    params = MarketParams()
    
    print(f"\nMarket Parameters:")
    print(f"  Spot (S0):      {params.S0}")
    print(f"  Risk-free rate: {params.r:.2%}")
    print(f"  Dividend yield: {params.q:.2%}")
    print(f"  Volatility:     {params.sigma:.2%}")
    print(f"  Time to expiry: {params.T} year")
    
    # Simulate paths
    print("\nSimulating 50,000 paths with 252 steps...")
    paths = simulate_gbm_paths(params, n_paths=50000, n_steps=252)
    print(f"  Path shape: {paths.shape}")
    
    K = params.S0  # ATM
    B_up = params.S0 * 1.15
    B_down = params.S0 * 0.85
    
    # Price all products
    print("\n" + "-" * 50)
    print("Pricing Results:")
    print("-" * 50)
    
    vanilla_call, vanilla_se = price_barrier_option(paths, K, 1000, params.r, 
                                                    params.T, 'call', 'up_and_out')
    print(f"\nVanilla Call:       {vanilla_call:.4f} ± {vanilla_se*1.96:.4f}")
    
    uo_call, uo_se = price_barrier_option(paths, K, B_up, params.r, 
                                          params.T, 'call', 'up_and_out')
    print(f"Up-Out Call (B={B_up}): {uo_call:.4f} ± {uo_se*1.96:.4f}")
    
    ui_call, ui_se = price_barrier_option(paths, K, B_up, params.r,
                                          params.T, 'call', 'up_and_in')
    print(f"Up-In Call (B={B_up}):  {ui_call:.4f} ± {ui_se*1.96:.4f}")
    print(f"  In-Out Parity Check: {uo_call + ui_call:.4f} ≈ {vanilla_call:.4f}")
    
    asian_arith, asian_se = price_asian_option(paths, K, params.r, params.T, 
                                               'call', 'arithmetic')
    print(f"\nAsian Call (Arith): {asian_arith:.4f} ± {asian_se*1.96:.4f}")
    
    asian_geom, _ = price_asian_option(paths, K, params.r, params.T, 'call', 'geometric')
    print(f"Asian Call (Geom):  {asian_geom:.4f}")
    
    lookback_float, lb_se = price_lookback_option(paths, K, params.r, params.T,
                                                   'call', 'floating')
    print(f"\nLookback Float Call: {lookback_float:.4f} ± {lb_se*1.96:.4f}")
    
    lookback_fixed, _ = price_lookback_option(paths, K, params.r, params.T,
                                              'call', 'fixed')
    print(f"Lookback Fixed Call: {lookback_fixed:.4f}")
    
    one_touch, ot_se = price_touch_option(paths, B_up, params.r, params.T,
                                          'up', 'one_touch')
    print(f"\nOne-Touch Up (B={B_up}): {one_touch:.4f} ± {ot_se*1.96:.4f}")
    
    no_touch, _ = price_touch_option(paths, B_up, params.r, params.T,
                                     'up', 'no_touch')
    print(f"No-Touch Up (B={B_up}):  {no_touch:.4f}")
    print(f"  Touch Parity Check: {one_touch + no_touch:.4f} ≈ {np.exp(-params.r*params.T):.4f}")
    
    # Generate plots
    print("\n" + "-" * 50)
    print("Generating Visualizations...")
    print("-" * 50)
    
    plot_barrier_analysis(params, paths)
    plot_asian_analysis(params, paths)
    plot_lookback_analysis(params, paths)
    plot_exotic_comparison(params, paths)
    
    print("\nPlots saved to current directory.")
    print("=" * 70)

if __name__ == "__main__":
    main()

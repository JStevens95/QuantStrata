#!/usr/bin/env python3
"""
European Vanilla Option Pricing: BSM, Monte Carlo, and Finite Difference

This example demonstrates pricing European vanilla FX options using three
different methods and compares their results and performance.

Topics Covered:
- Black-Scholes-Merton analytical pricing
- Monte Carlo simulation with variance reduction
- Finite Difference PDE solving
- Greeks computation and visualization
- Method comparison and convergence analysis

Author: QuantStrata Team
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple

# QuantStrata imports
import sys
sys.path.insert(0, '../..')

from src.instruments.fx.options.vanilla import EuropeanFxVanillaOption
from src.marketdata.core.market import Market

# =============================================================================
# Configuration and Setup
# =============================================================================

# Plot style configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (14, 8),
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'lines.linewidth': 2,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Color palette
COLORS = {
    'bsm': '#2E86AB',      # Blue
    'mc': '#A23B72',       # Magenta
    'fd': '#F18F01',       # Orange
    'call': '#2E86AB',
    'put': '#E94F37',
    'gamma': '#8B5CF6',
    'vega': '#10B981',
}

@dataclass
class OptionParams:
    """Container for option parameters."""
    spot: float = 1.3000          # EUR/USD spot
    strike: float = 1.3000        # ATM strike
    expiry: float = 1.0           # 1 year
    vol: float = 0.10             # 10% volatility
    r_dom: float = 0.05           # USD rate
    r_for: float = 0.02           # EUR rate
    notional: float = 1_000_000   # 1M EUR notional

# =============================================================================
# Analytical Black-Scholes-Merton
# =============================================================================

def bsm_price(S: float, K: float, T: float, r: float, q: float, sigma: float, 
              option_type: str = 'call') -> float:
    """
    Black-Scholes-Merton price for European option.
    
    Parameters
    ----------
    S : Spot price
    K : Strike price
    T : Time to expiry (years)
    r : Domestic risk-free rate
    q : Foreign/dividend rate
    sigma : Volatility
    option_type : 'call' or 'put'
    """
    from scipy.stats import norm
    
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_type == 'call':
        price = S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        price = K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)
    
    return price

def bsm_greeks(S: float, K: float, T: float, r: float, q: float, sigma: float,
               option_type: str = 'call') -> dict:
    """Compute all Greeks analytically."""
    from scipy.stats import norm
    
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    # Common terms
    Nd1 = norm.cdf(d1) if option_type == 'call' else norm.cdf(-d1)
    nd1 = norm.pdf(d1)
    
    # Greeks
    delta = np.exp(-q*T) * Nd1 if option_type == 'call' else -np.exp(-q*T) * norm.cdf(-d1)
    gamma = np.exp(-q*T) * nd1 / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q*T) * np.sqrt(T) * nd1 / 100  # Per 1% vol
    theta = -(S * sigma * np.exp(-q*T) * nd1) / (2*np.sqrt(T))
    if option_type == 'call':
        theta += q*S*np.exp(-q*T)*norm.cdf(d1) - r*K*np.exp(-r*T)*norm.cdf(d2)
    else:
        theta += -q*S*np.exp(-q*T)*norm.cdf(-d1) + r*K*np.exp(-r*T)*norm.cdf(-d2)
    theta /= 365  # Per day
    
    return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta}

# =============================================================================
# Monte Carlo Pricing
# =============================================================================

def mc_price(S: float, K: float, T: float, r: float, q: float, sigma: float,
             option_type: str = 'call', n_paths: int = 100000, 
             antithetic: bool = True) -> Tuple[float, float]:
    """
    Monte Carlo price with confidence interval.
    
    Returns
    -------
    price : float
    std_error : float
    """
    np.random.seed(42)
    
    if antithetic:
        n_half = n_paths // 2
        Z = np.random.randn(n_half)
        Z = np.concatenate([Z, -Z])  # Antithetic pairs
    else:
        Z = np.random.randn(n_paths)
    
    # Terminal spot under Q
    drift = (r - q - 0.5*sigma**2) * T
    diffusion = sigma * np.sqrt(T) * Z
    ST = S * np.exp(drift + diffusion)
    
    # Payoffs
    if option_type == 'call':
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)
    
    # Discounted price
    disc_payoffs = np.exp(-r*T) * payoffs
    price = np.mean(disc_payoffs)
    std_error = np.std(disc_payoffs) / np.sqrt(len(disc_payoffs))
    
    return price, std_error

# =============================================================================
# Finite Difference Pricing
# =============================================================================

def fd_price(S0: float, K: float, T: float, r: float, q: float, sigma: float,
             option_type: str = 'call', N: int = 200, M: int = 100) -> float:
    """
    Crank-Nicolson finite difference price.
    
    Parameters
    ----------
    N : Number of spot grid points
    M : Number of time steps
    """
    # Grid setup
    S_max = 4 * K
    dS = S_max / N
    dt = T / M
    
    S = np.linspace(0, S_max, N+1)
    
    # Terminal condition
    if option_type == 'call':
        V = np.maximum(S - K, 0)
    else:
        V = np.maximum(K - S, 0)
    
    # Coefficient vectors (for interior points)
    j = np.arange(1, N)
    a = 0.25 * dt * (sigma**2 * j**2 - (r-q) * j)
    b = -0.5 * dt * (sigma**2 * j**2 + r)
    c = 0.25 * dt * (sigma**2 * j**2 + (r-q) * j)
    
    # Build tridiagonal matrices
    A = np.diag(1 - b) + np.diag(-a[1:], -1) + np.diag(-c[:-1], 1)
    B = np.diag(1 + b) + np.diag(a[1:], -1) + np.diag(c[:-1], 1)
    
    # Time stepping
    for m in range(M):
        # Boundary values
        if option_type == 'call':
            V_lower = 0
            V_upper = S_max - K * np.exp(-r * (T - m*dt))
        else:
            V_lower = K * np.exp(-r * (T - m*dt))
            V_upper = 0
        
        # RHS
        rhs = B @ V[1:N]
        rhs[0] += a[0] * V_lower
        rhs[-1] += c[-1] * V_upper
        
        # Solve
        V[1:N] = np.linalg.solve(A, rhs)
        V[0] = V_lower
        V[N] = V_upper
    
    # Interpolate to spot
    return np.interp(S0, S, V)

# =============================================================================
# Visualization Functions
# =============================================================================

def plot_method_comparison(params: OptionParams):
    """Compare pricing methods across spot range."""
    spots = np.linspace(params.spot * 0.7, params.spot * 1.3, 50)
    
    bsm_calls = [bsm_price(s, params.strike, params.expiry, params.r_dom, 
                           params.r_for, params.vol, 'call') for s in spots]
    bsm_puts = [bsm_price(s, params.strike, params.expiry, params.r_dom,
                          params.r_for, params.vol, 'put') for s in spots]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Call prices
    ax = axes[0]
    ax.plot(spots, bsm_calls, color=COLORS['bsm'], linewidth=2.5, label='BSM Analytical')
    
    # MC prices at sample points
    mc_spots = spots[::5]
    mc_calls = [mc_price(s, params.strike, params.expiry, params.r_dom,
                         params.r_for, params.vol, 'call')[0] for s in mc_spots]
    ax.scatter(mc_spots, mc_calls, color=COLORS['mc'], s=60, zorder=5, 
               label='Monte Carlo', marker='o')
    
    # FD prices at sample points
    fd_calls = [fd_price(s, params.strike, params.expiry, params.r_dom,
                         params.r_for, params.vol, 'call') for s in mc_spots]
    ax.scatter(mc_spots, fd_calls, color=COLORS['fd'], s=60, zorder=5,
               label='Finite Difference', marker='s')
    
    ax.axvline(params.strike, color='gray', linestyle='--', alpha=0.5, label='Strike')
    ax.set_xlabel('Spot Price')
    ax.set_ylabel('Option Price')
    ax.set_title('European Call Option: Method Comparison')
    ax.legend()
    
    # Put prices
    ax = axes[1]
    ax.plot(spots, bsm_puts, color=COLORS['bsm'], linewidth=2.5, label='BSM Analytical')
    
    mc_puts = [mc_price(s, params.strike, params.expiry, params.r_dom,
                        params.r_for, params.vol, 'put')[0] for s in mc_spots]
    ax.scatter(mc_spots, mc_puts, color=COLORS['mc'], s=60, zorder=5,
               label='Monte Carlo', marker='o')
    
    fd_puts = [fd_price(s, params.strike, params.expiry, params.r_dom,
                        params.r_for, params.vol, 'put') for s in mc_spots]
    ax.scatter(mc_spots, fd_puts, color=COLORS['fd'], s=60, zorder=5,
               label='Finite Difference', marker='s')
    
    ax.axvline(params.strike, color='gray', linestyle='--', alpha=0.5, label='Strike')
    ax.set_xlabel('Spot Price')
    ax.set_ylabel('Option Price')
    ax.set_title('European Put Option: Method Comparison')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('european_vanilla_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_greeks_surface(params: OptionParams):
    """Plot Greeks as function of spot and time."""
    spots = np.linspace(params.spot * 0.7, params.spot * 1.3, 40)
    times = np.linspace(0.01, params.expiry, 30)
    
    S_grid, T_grid = np.meshgrid(spots, times)
    
    # Compute delta surface
    delta_grid = np.zeros_like(S_grid)
    gamma_grid = np.zeros_like(S_grid)
    
    for i, t in enumerate(times):
        for j, s in enumerate(spots):
            greeks = bsm_greeks(s, params.strike, t, params.r_dom, 
                               params.r_for, params.vol, 'call')
            delta_grid[i, j] = greeks['delta']
            gamma_grid[i, j] = greeks['gamma']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), subplot_kw={'projection': '3d'})
    
    # Delta surface
    ax = axes[0]
    surf = ax.plot_surface(S_grid, T_grid, delta_grid, cmap='viridis', alpha=0.8)
    ax.set_xlabel('Spot')
    ax.set_ylabel('Time to Expiry')
    ax.set_zlabel('Delta')
    ax.set_title('Call Option Delta Surface')
    ax.view_init(elev=25, azim=45)
    
    # Gamma surface
    ax = axes[1]
    surf = ax.plot_surface(S_grid, T_grid, gamma_grid, cmap='plasma', alpha=0.8)
    ax.set_xlabel('Spot')
    ax.set_ylabel('Time to Expiry')
    ax.set_zlabel('Gamma')
    ax.set_title('Call Option Gamma Surface')
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    plt.savefig('greeks_surface.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_convergence_analysis(params: OptionParams):
    """Analyze MC and FD convergence."""
    # BSM benchmark
    bsm_call = bsm_price(params.spot, params.strike, params.expiry,
                         params.r_dom, params.r_for, params.vol, 'call')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # MC convergence
    ax = axes[0]
    path_counts = [1000, 5000, 10000, 50000, 100000, 500000]
    mc_prices = []
    mc_errors = []
    
    for n in path_counts:
        price, stderr = mc_price(params.spot, params.strike, params.expiry,
                                 params.r_dom, params.r_for, params.vol, 'call',
                                 n_paths=n)
        mc_prices.append(price)
        mc_errors.append(stderr)
    
    ax.errorbar(path_counts, mc_prices, yerr=np.array(mc_errors)*1.96,
                fmt='o-', color=COLORS['mc'], capsize=5, label='MC Price ± 95% CI')
    ax.axhline(bsm_call, color=COLORS['bsm'], linestyle='--', linewidth=2,
               label=f'BSM = {bsm_call:.6f}')
    ax.set_xscale('log')
    ax.set_xlabel('Number of Paths')
    ax.set_ylabel('Option Price')
    ax.set_title('Monte Carlo Convergence')
    ax.legend()
    
    # FD convergence
    ax = axes[1]
    grid_sizes = [50, 100, 200, 400, 800]
    fd_prices = []
    
    for n in grid_sizes:
        price = fd_price(params.spot, params.strike, params.expiry,
                        params.r_dom, params.r_for, params.vol, 'call',
                        N=n, M=n//2)
        fd_prices.append(price)
    
    fd_errors = np.abs(np.array(fd_prices) - bsm_call)
    
    ax.semilogy(grid_sizes, fd_errors, 'o-', color=COLORS['fd'], 
                linewidth=2, markersize=8, label='|FD - BSM|')
    
    # Reference line for O(h²) convergence
    h_ref = np.array(grid_sizes)
    err_ref = fd_errors[0] * (grid_sizes[0]/h_ref)**2
    ax.semilogy(grid_sizes, err_ref, '--', color='gray', alpha=0.7,
                label=r'$O(h^2)$ reference')
    
    ax.set_xlabel('Grid Points (N)')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Finite Difference Convergence')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('convergence_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_volatility_smile_impact(params: OptionParams):
    """Show how volatility affects option prices."""
    vols = np.linspace(0.05, 0.40, 8)
    strikes = np.linspace(params.spot * 0.8, params.spot * 1.2, 50)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Call prices for different vols
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(vols)))
    
    for vol, color in zip(vols, colors):
        prices = [bsm_price(params.spot, k, params.expiry, params.r_dom,
                           params.r_for, vol, 'call') for k in strikes]
        ax.plot(strikes, prices, color=color, label=f'σ = {vol:.0%}')
    
    ax.axvline(params.spot, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Strike')
    ax.set_ylabel('Call Price')
    ax.set_title('Call Price vs Strike (Different Volatilities)')
    ax.legend(loc='upper right')
    
    # Implied volatility smile illustration
    ax = axes[1]
    
    # Simulate a volatility smile (in reality would be from market)
    moneyness = strikes / params.spot
    smile = params.vol * (1 + 0.3 * (moneyness - 1)**2 + 0.1 * (1 - moneyness))
    
    ax.plot(moneyness, smile * 100, color=COLORS['vega'], linewidth=2.5)
    ax.axvline(1.0, color='gray', linestyle='--', alpha=0.5, label='ATM')
    ax.axhline(params.vol * 100, color='gray', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Moneyness (K/S)')
    ax.set_ylabel('Implied Volatility (%)')
    ax.set_title('Typical Volatility Smile Shape')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('volatility_impact.png', dpi=150, bbox_inches='tight')
    plt.show()

# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Run all demonstrations."""
    print("=" * 70)
    print("European Vanilla Option Pricing Showcase")
    print("=" * 70)
    
    params = OptionParams()
    
    # Print parameters
    print(f"\nOption Parameters:")
    print(f"  Spot (EUR/USD):     {params.spot}")
    print(f"  Strike:             {params.strike}")
    print(f"  Expiry:             {params.expiry} year")
    print(f"  Volatility:         {params.vol:.1%}")
    print(f"  Domestic rate (USD):{params.r_dom:.2%}")
    print(f"  Foreign rate (EUR): {params.r_for:.2%}")
    
    # Compute prices
    print("\n" + "-" * 50)
    print("Pricing Results:")
    print("-" * 50)
    
    bsm_call = bsm_price(params.spot, params.strike, params.expiry,
                         params.r_dom, params.r_for, params.vol, 'call')
    bsm_put = bsm_price(params.spot, params.strike, params.expiry,
                        params.r_dom, params.r_for, params.vol, 'put')
    
    mc_call, mc_std = mc_price(params.spot, params.strike, params.expiry,
                               params.r_dom, params.r_for, params.vol, 'call')
    mc_put, _ = mc_price(params.spot, params.strike, params.expiry,
                         params.r_dom, params.r_for, params.vol, 'put')
    
    fd_call = fd_price(params.spot, params.strike, params.expiry,
                       params.r_dom, params.r_for, params.vol, 'call')
    fd_put = fd_price(params.spot, params.strike, params.expiry,
                      params.r_dom, params.r_for, params.vol, 'put')
    
    print(f"\n{'Method':<15} {'Call Price':<12} {'Put Price':<12}")
    print("-" * 40)
    print(f"{'BSM Analytical':<15} {bsm_call:<12.6f} {bsm_put:<12.6f}")
    print(f"{'Monte Carlo':<15} {mc_call:<12.6f} {mc_put:<12.6f}")
    print(f"{'Finite Diff':<15} {fd_call:<12.6f} {fd_put:<12.6f}")
    
    # Greeks
    greeks = bsm_greeks(params.spot, params.strike, params.expiry,
                        params.r_dom, params.r_for, params.vol, 'call')
    
    print("\n" + "-" * 50)
    print("Call Option Greeks (BSM):")
    print("-" * 50)
    print(f"  Delta:  {greeks['delta']:.4f}")
    print(f"  Gamma:  {greeks['gamma']:.4f}")
    print(f"  Vega:   {greeks['vega']:.4f} (per 1% vol)")
    print(f"  Theta:  {greeks['theta']:.4f} (per day)")
    
    # Put-Call Parity Check
    parity_lhs = bsm_call - bsm_put
    parity_rhs = params.spot * np.exp(-params.r_for * params.expiry) - \
                 params.strike * np.exp(-params.r_dom * params.expiry)
    
    print("\n" + "-" * 50)
    print("Put-Call Parity Check:")
    print("-" * 50)
    print(f"  C - P       = {parity_lhs:.6f}")
    print(f"  Se^(-qT) - Ke^(-rT) = {parity_rhs:.6f}")
    print(f"  Difference  = {abs(parity_lhs - parity_rhs):.2e}")
    
    # Generate plots
    print("\n" + "-" * 50)
    print("Generating Visualizations...")
    print("-" * 50)
    
    plot_method_comparison(params)
    plot_greeks_surface(params)
    plot_convergence_analysis(params)
    plot_volatility_smile_impact(params)
    
    print("\nPlots saved to current directory.")
    print("=" * 70)

if __name__ == "__main__":
    main()

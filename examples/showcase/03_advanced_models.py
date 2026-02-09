#!/usr/bin/env python3
"""
Advanced Volatility Models: Local Volatility and Heston Stochastic Volatility

This example demonstrates advanced volatility modeling beyond constant BSM:
- Local Volatility (Dupire): σ(S, t) deterministic function
- Heston Stochastic Volatility: σ follows its own diffusion

Topics Covered:
- Model dynamics and simulation
- Volatility smile generation
- Impact on exotic option pricing
- Model calibration concepts

Author: QuantStrata Team
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from dataclasses import dataclass
from typing import Tuple
from scipy.stats import norm

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    HAS_MATPLOTLIB = False

# Path setup
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# QuantStrata imports - use library dynamics
from src.models.dynamics.gbm_dynamics import GbmDynamicsSimulator, GbmScheme
from src.models.stochastic_volatility.heston import (
    HestonParameters,
    HestonDynamics,
    HestonSimulation,
)

# =============================================================================
# Configuration
# =============================================================================

if HAS_MATPLOTLIB:
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({
        'figure.figsize': (14, 8),
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'legend.fontsize': 10,
        'lines.linewidth': 2,
    })

COLORS = {
    'bsm': '#2E86AB',
    'local_vol': '#E94F37',
    'heston': '#8B5CF6',
    'market': '#10B981',
    'vol': '#F59E0B',
}

@dataclass
class BSMParams:
    """Black-Scholes-Merton parameters."""
    S0: float = 100.0
    r: float = 0.05
    q: float = 0.02
    sigma: float = 0.20
    T: float = 1.0

@dataclass
class HestonParams:
    """Heston stochastic volatility parameters."""
    S0: float = 100.0
    V0: float = 0.04         # Initial variance (σ₀² = 0.2²)
    r: float = 0.05
    q: float = 0.02
    kappa: float = 2.0       # Mean reversion speed
    theta: float = 0.04      # Long-term variance
    xi: float = 0.30         # Vol-of-vol
    rho: float = -0.70       # Spot-vol correlation
    T: float = 1.0

# =============================================================================
# Local Volatility Model
# =============================================================================

def local_vol_surface(S: np.ndarray, t: np.ndarray, S0: float = 100, 
                      sigma_atm: float = 0.20, skew: float = -0.1, 
                      smile: float = 0.2) -> np.ndarray:
    """
    Parametric local volatility surface.
    
    σ(S, t) = σ_ATM × [1 + skew × log(S/S0) + smile × log²(S/S0)] × decay(t)
    
    This is a stylized surface for demonstration; in practice would be
    calibrated via Dupire's formula from market implied vols.
    """
    S = np.atleast_1d(S)
    t = np.atleast_1d(t)
    
    # Pointwise (S[i], t[i]) when both 1d and same shape; otherwise meshgrid for 2d output
    if S.ndim == 1 and t.ndim == 1 and S.shape == t.shape:
        S_grid, t_grid = S, t
    elif S.ndim == 1 and t.ndim == 1:
        S_grid, t_grid = np.meshgrid(S, t, indexing='ij')
    else:
        S_grid, t_grid = S, t
    
    log_moneyness = np.log(S_grid / S0)
    time_decay = 1.0 / np.sqrt(np.maximum(t_grid, 0.001))  # Vol increases for short expiry
    time_decay = np.minimum(time_decay, 3.0)  # Cap the decay
    
    # Smile shape
    vol = sigma_atm * (1 + skew * log_moneyness + smile * log_moneyness**2)
    
    # Time component (term structure)
    vol = vol * (0.8 + 0.2 * np.sqrt(t_grid))
    
    return np.clip(vol, 0.05, 1.0)

def simulate_local_vol_paths(params: BSMParams, n_paths: int = 10000,
                             n_steps: int = 252, seed: int = 42) -> np.ndarray:
    """Simulate paths under local volatility model."""
    np.random.seed(seed)
    
    dt = params.T / n_steps
    sqrt_dt = np.sqrt(dt)
    
    paths = np.zeros((n_steps + 1, n_paths))
    paths[0, :] = params.S0
    
    for i in range(n_steps):
        t = i * dt
        S = paths[i, :]
        
        # Local vol at current (S, t)
        sigma = local_vol_surface(S, np.full_like(S, t), params.S0)
        
        # Euler step
        Z = np.random.randn(n_paths)
        drift = (params.r - params.q - 0.5 * sigma**2) * dt
        diffusion = sigma * sqrt_dt * Z
        paths[i+1, :] = S * np.exp(drift + diffusion)
    
    return paths

# =============================================================================
# Heston Stochastic Volatility Model
# =============================================================================

def simulate_heston_paths(params: HestonParams, n_paths: int = 10000,
                          n_steps: int = 252, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate Heston model paths using QuantStrata's library HestonDynamics.
    
    This function uses the library's production Heston simulator which provides
    multiple discretization schemes (euler, full_truncation, reflection, qe).
    
    Model:
        dS_t = (r - q) S_t dt + √V_t S_t dW_S
        dV_t = κ(θ - V_t) dt + ξ √V_t dW_V
        dW_S · dW_V = ρ dt
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Spot paths (n_steps+1, n_paths), Variance paths (n_steps+1, n_paths)
    """
    # Convert local HestonParams to library HestonParameters
    lib_params = HestonParameters(
        kappa=params.kappa,
        theta=params.theta,
        xi=params.xi,
        v0=params.V0,
        rho=params.rho,
    )
    
    # Create HestonDynamics with drift = r - q
    dynamics = HestonDynamics(
        params=lib_params,
        drift=params.r - params.q,
    )
    
    # Simulate using library (full_truncation scheme for robustness)
    sim: HestonSimulation = dynamics.simulate(
        spot0=params.S0,
        maturity=params.T,
        n_paths=n_paths,
        n_steps=n_steps,
        scheme="full_truncation",
        seed=seed,
        antithetic=True,
    )
    
    # Transpose to (n_steps+1, n_paths) for compatibility
    S_paths = sim.spot_paths.T
    V_paths = sim.variance_paths.T
    
    return S_paths, V_paths

def heston_characteristic_function(u: complex, params: HestonParams, T: float) -> complex:
    """
    Heston characteristic function for pricing via FFT.
    
    φ(u) = E[exp(iu × log(S_T))]
    """
    kappa, theta, xi, rho = params.kappa, params.theta, params.xi, params.rho
    V0 = params.V0
    
    # Complex parameters
    d = np.sqrt((rho * xi * 1j * u - kappa)**2 + xi**2 * (1j * u + u**2))
    g = (kappa - rho * xi * 1j * u - d) / (kappa - rho * xi * 1j * u + d)
    
    # Characteristic function components
    C = (params.r - params.q) * 1j * u * T + \
        (kappa * theta / xi**2) * ((kappa - rho * xi * 1j * u - d) * T - \
        2 * np.log((1 - g * np.exp(-d * T)) / (1 - g)))
    
    D = ((kappa - rho * xi * 1j * u - d) / xi**2) * \
        ((1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T)))
    
    return np.exp(C + D * V0 + 1j * u * np.log(params.S0))

# =============================================================================
# Implied Volatility Computation
# =============================================================================

def bsm_call_price(S, K, T, r, q, sigma):
    """Black-Scholes call price using library function."""
    from src.models.analytic.black_scholes_merton.base import vanilla_price
    carry = r - q
    return vanilla_price(
        option_type="call", spot=S, strike=K, expiry=T,
        discount_rate=r, carry=carry, vol=sigma
    )

def implied_vol_newton(price, S, K, T, r, q, max_iter=100, tol=1e-8):
    """Compute implied volatility using Newton-Raphson with library functions."""
    from src.models.analytic.black_scholes_merton.base import vanilla_price, vanilla_vega
    
    sigma = 0.20  # Initial guess
    carry = r - q
    
    for _ in range(max_iter):
        # Price and vega using library
        bsm_price_val = vanilla_price(
            option_type="call", spot=S, strike=K, expiry=T,
            discount_rate=r, carry=carry, vol=sigma
        )
        vega = vanilla_vega(
            option_type="call", spot=S, strike=K, expiry=T,
            discount_rate=r, carry=carry, vol=sigma
        )
        
        if vega < 1e-10:
            break
            
        diff = bsm_price_val - price
        if abs(diff) < tol:
            break
            
        sigma = sigma - diff / vega
        sigma = max(0.01, min(sigma, 2.0))  # Bounds
    
    return sigma

def compute_implied_vol_surface(paths: np.ndarray, S0: float, r: float, q: float,
                                T: float, strikes: np.ndarray) -> np.ndarray:
    """Compute implied vols from MC prices."""
    implied_vols = []
    
    terminal = paths[-1, :]
    
    for K in strikes:
        # MC price
        payoffs = np.maximum(terminal - K, 0)
        mc_price = np.exp(-r*T) * np.mean(payoffs)
        
        # Implied vol
        try:
            iv = implied_vol_newton(mc_price, S0, K, T, r, q)
        except:
            iv = np.nan
        implied_vols.append(iv)
    
    return np.array(implied_vols)

# =============================================================================
# Visualization Functions
# =============================================================================

def plot_local_vol_surface(params: BSMParams):
    """Visualize local volatility surface."""
    fig = plt.figure(figsize=(14, 5))
    
    # 3D surface
    ax1 = fig.add_subplot(121, projection='3d')
    
    S_range = np.linspace(params.S0 * 0.6, params.S0 * 1.4, 40)
    t_range = np.linspace(0.01, params.T, 30)
    S_grid, t_grid = np.meshgrid(S_range, t_range, indexing='ij')
    
    vol_surface = local_vol_surface(S_grid, t_grid, params.S0)
    
    surf = ax1.plot_surface(S_grid, t_grid, vol_surface * 100, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('Spot')
    ax1.set_ylabel('Time')
    ax1.set_zlabel('Local Vol (%)')
    ax1.set_title('Local Volatility Surface σ(S, t)')
    ax1.view_init(elev=25, azim=45)
    
    # Smile slices
    ax2 = fig.add_subplot(122)
    
    times = [0.1, 0.25, 0.5, 1.0]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(times)))
    
    for t, color in zip(times, colors):
        vols = local_vol_surface(S_range, np.full_like(S_range, t), params.S0)
        ax2.plot(S_range / params.S0, vols * 100, color=color, label=f'T = {t}')
    
    ax2.axvline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Moneyness (S/S₀)')
    ax2.set_ylabel('Local Volatility (%)')
    ax2.set_title('Local Vol Smile at Different Times')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('local_vol_surface.png', dpi=150, bbox_inches='tight')
    plt.show(block=True)

def plot_heston_dynamics(params: HestonParams):
    """Visualize Heston model dynamics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Simulate
    S_paths, V_paths = simulate_heston_paths(params, n_paths=10000, n_steps=252)
    time_grid = np.linspace(0, params.T, S_paths.shape[0])
    
    # Sample paths
    ax = axes[0, 0]
    for i in range(min(30, S_paths.shape[1])):
        ax.plot(time_grid, S_paths[:, i], alpha=0.4, linewidth=0.8, color=COLORS['heston'])
    ax.axhline(params.S0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Spot Price')
    ax.set_title('Heston Model: Sample Spot Paths')
    
    # Variance paths
    ax = axes[0, 1]
    for i in range(min(30, V_paths.shape[1])):
        ax.plot(time_grid, np.sqrt(V_paths[:, i]) * 100, alpha=0.4, linewidth=0.8, color=COLORS['vol'])
    ax.axhline(np.sqrt(params.theta) * 100, color='gray', linestyle='--', 
              label=f'√θ = {np.sqrt(params.theta)*100:.1f}%')
    ax.axhline(np.sqrt(params.V0) * 100, color='black', linestyle=':',
              label=f'√V₀ = {np.sqrt(params.V0)*100:.1f}%')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Volatility (%)')
    ax.set_title('Heston Model: Sample Volatility Paths')
    ax.legend()
    
    # Terminal distribution comparison
    ax = axes[1, 0]
    
    # BSM terminal
    bsm_params = BSMParams(S0=params.S0, r=params.r, q=params.q, 
                           sigma=np.sqrt(params.V0), T=params.T)
    bsm_paths = simulate_local_vol_paths(bsm_params, n_paths=10000, n_steps=252, seed=123)
    
    ax.hist(bsm_paths[-1, :], bins=50, alpha=0.5, density=True,
           color=COLORS['bsm'], label='BSM')
    ax.hist(S_paths[-1, :], bins=50, alpha=0.5, density=True,
           color=COLORS['heston'], label='Heston')
    
    ax.set_xlabel('Terminal Spot')
    ax.set_ylabel('Density')
    ax.set_title('Terminal Distribution: BSM vs Heston')
    ax.legend()
    
    # Spot-Vol correlation
    ax = axes[1, 1]
    
    final_S = S_paths[-1, :]
    final_V = V_paths[-1, :]
    
    ax.scatter(final_S, np.sqrt(final_V) * 100, alpha=0.2, s=10, color=COLORS['heston'])
    ax.axhline(np.sqrt(params.theta) * 100, color='gray', linestyle='--', alpha=0.7)
    ax.axvline(params.S0, color='gray', linestyle='--', alpha=0.7)
    
    # Correlation text
    corr = np.corrcoef(final_S, final_V)[0, 1]
    ax.text(0.05, 0.95, f'ρ(S,V) = {corr:.2f}\n(param ρ = {params.rho})',
           transform=ax.transAxes, fontsize=11,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Terminal Spot')
    ax.set_ylabel('Terminal Volatility (%)')
    ax.set_title('Spot-Volatility Correlation')
    
    plt.tight_layout()
    plt.savefig('heston_dynamics.png', dpi=150, bbox_inches='tight')
    plt.show(block=True)

def plot_implied_vol_comparison(bsm_params: BSMParams, heston_params: HestonParams):
    """Compare implied volatility smiles from different models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Simulate paths
    bsm_paths = simulate_local_vol_paths(bsm_params, n_paths=50000, seed=42)
    lv_paths = simulate_local_vol_paths(bsm_params, n_paths=50000, seed=42)  # LV simulation
    heston_paths, _ = simulate_heston_paths(heston_params, n_paths=50000, seed=42)
    
    strikes = np.linspace(bsm_params.S0 * 0.8, bsm_params.S0 * 1.2, 15)
    
    # Compute implied vols
    iv_bsm = compute_implied_vol_surface(bsm_paths, bsm_params.S0, bsm_params.r,
                                         bsm_params.q, bsm_params.T, strikes)
    iv_heston = compute_implied_vol_surface(heston_paths, heston_params.S0, heston_params.r,
                                            heston_params.q, heston_params.T, strikes)
    
    # Implied vol smile
    ax = axes[0]
    moneyness = strikes / bsm_params.S0
    
    ax.plot(moneyness, iv_bsm * 100, 'o-', color=COLORS['bsm'], 
           markersize=6, label='BSM (flat)')
    ax.plot(moneyness, iv_heston * 100, 's-', color=COLORS['heston'],
           markersize=6, label='Heston')
    
    ax.axvline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(bsm_params.sigma * 100, color=COLORS['bsm'], linestyle=':', alpha=0.7)
    
    ax.set_xlabel('Moneyness (K/S₀)')
    ax.set_ylabel('Implied Volatility (%)')
    ax.set_title('Implied Volatility Smile')
    ax.legend()
    
    # Skew comparison
    ax = axes[1]
    
    # Heston skew for different correlations
    rhos = [-0.9, -0.5, 0.0, 0.5]
    colors = plt.cm.RdYlBu(np.linspace(0.1, 0.9, len(rhos)))
    
    for rho, color in zip(rhos, colors):
        params_temp = HestonParams(S0=heston_params.S0, V0=heston_params.V0,
                                   r=heston_params.r, q=heston_params.q,
                                   kappa=heston_params.kappa, theta=heston_params.theta,
                                   xi=heston_params.xi, rho=rho, T=heston_params.T)
        paths_temp, _ = simulate_heston_paths(params_temp, n_paths=30000, seed=42)
        iv_temp = compute_implied_vol_surface(paths_temp, params_temp.S0, params_temp.r,
                                              params_temp.q, params_temp.T, strikes)
        ax.plot(moneyness, iv_temp * 100, 'o-', color=color, markersize=5, 
               label=f'ρ = {rho}')
    
    ax.axvline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Moneyness (K/S₀)')
    ax.set_ylabel('Implied Volatility (%)')
    ax.set_title('Heston Smile: Effect of Correlation ρ')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('implied_vol_comparison.png', dpi=150, bbox_inches='tight')
    plt.show(block=True)

def plot_model_comparison_exotic(bsm_params: BSMParams, heston_params: HestonParams):
    """Compare models on exotic option pricing."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Simulate
    bsm_paths = simulate_local_vol_paths(bsm_params, n_paths=50000, seed=42)
    heston_paths, _ = simulate_heston_paths(heston_params, n_paths=50000, seed=42)
    
    K = bsm_params.S0
    B_up = bsm_params.S0 * 1.20
    r, q, T = bsm_params.r, bsm_params.q, bsm_params.T
    
    # Price various products
    products = ['Vanilla Call', 'Up-Out Call', 'Asian Call', 'Lookback Call']
    
    def price_product(paths, product):
        terminal = paths[-1, :]
        if product == 'Vanilla Call':
            payoffs = np.maximum(terminal - K, 0)
        elif product == 'Up-Out Call':
            breached = np.any(paths >= B_up, axis=0)
            payoffs = np.where(breached, 0, np.maximum(terminal - K, 0))
        elif product == 'Asian Call':
            avg = np.mean(paths, axis=0)
            payoffs = np.maximum(avg - K, 0)
        elif product == 'Lookback Call':
            min_S = np.min(paths, axis=0)
            payoffs = terminal - min_S
        return np.exp(-r*T) * np.mean(payoffs)
    
    bsm_prices = [price_product(bsm_paths, p) for p in products]
    heston_prices = [price_product(heston_paths, p) for p in products]
    
    # Price comparison
    ax = axes[0]
    x = np.arange(len(products))
    width = 0.35
    
    ax.bar(x - width/2, bsm_prices, width, label='BSM', color=COLORS['bsm'])
    ax.bar(x + width/2, heston_prices, width, label='Heston', color=COLORS['heston'])
    
    ax.set_xticks(x)
    ax.set_xticklabels(products, rotation=15)
    ax.set_ylabel('Option Price')
    ax.set_title('Model Impact on Exotic Prices')
    ax.legend()
    
    # Relative difference
    ax = axes[1]
    rel_diff = [(h - b) / b * 100 for b, h in zip(bsm_prices, heston_prices)]
    
    colors_bars = [COLORS['heston'] if d > 0 else COLORS['bsm'] for d in rel_diff]
    ax.bar(products, rel_diff, color=colors_bars)
    ax.axhline(0, color='black', linewidth=0.5)
    
    ax.set_ylabel('Relative Difference (%)')
    ax.set_title('Heston vs BSM: Price Difference')
    ax.set_xticklabels(products, rotation=15)
    
    for i, (d, p) in enumerate(zip(rel_diff, products)):
        ax.text(i, d + (1 if d > 0 else -2), f'{d:.1f}%', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('model_comparison_exotic.png', dpi=150, bbox_inches='tight')
    plt.show(block=True)

# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Run advanced models showcase."""
    print("=" * 70)
    print("Advanced Volatility Models Showcase")
    print("=" * 70)
    
    bsm_params = BSMParams()
    heston_params = HestonParams()
    
    print("\nBlack-Scholes Parameters:")
    print(f"  S₀ = {bsm_params.S0}, σ = {bsm_params.sigma:.1%}")
    print(f"  r = {bsm_params.r:.2%}, q = {bsm_params.q:.2%}, T = {bsm_params.T}")
    
    print("\nHeston Parameters:")
    print(f"  S₀ = {heston_params.S0}, V₀ = {heston_params.V0} (σ₀ = {np.sqrt(heston_params.V0):.1%})")
    print(f"  κ = {heston_params.kappa} (mean reversion)")
    print(f"  θ = {heston_params.theta} (long-term variance, σ_∞ = {np.sqrt(heston_params.theta):.1%})")
    print(f"  ξ = {heston_params.xi} (vol-of-vol)")
    print(f"  ρ = {heston_params.rho} (spot-vol correlation)")
    
    # Feller condition check
    feller = 2 * heston_params.kappa * heston_params.theta / heston_params.xi**2
    print(f"\n  Feller condition: 2κθ/ξ² = {feller:.2f} {'> 1 ✓' if feller > 1 else '< 1 (may touch zero)'}")
    
    # Generate plots (optional; skip if matplotlib not available)
    if HAS_MATPLOTLIB:
        print("\n" + "-" * 50)
        print("Generating Visualizations...")
        print("-" * 50)
        plot_local_vol_surface(bsm_params)
        plot_heston_dynamics(heston_params)
        plot_implied_vol_comparison(bsm_params, heston_params)
        plot_model_comparison_exotic(bsm_params, heston_params)
        print("\nPlots saved to current directory.")
    else:
        print("\nSkipping plots (matplotlib not available).")
    print("=" * 70)

if __name__ == "__main__":
    main()

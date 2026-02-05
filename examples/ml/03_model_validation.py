#!/usr/bin/env python3
"""
===============================================================================
Model Validation: BSM vs Monte Carlo vs Finite Difference
===============================================================================

This example demonstrates systematic model validation - comparing analytical
(BSM), Monte Carlo (MC), and Finite Difference (FD) pricing methods.

Learning Objectives
-------------------
1. **Model Comparison**: Understand when each method excels
2. **Convergence Analysis**: Study how numerical methods converge
3. **Error Quantification**: Measure and report pricing errors
4. **Production Validation**: Model approval process for hedge funds

Mathematical Framework
----------------------
Black-Scholes-Merton (Analytical):
    C = S·e^(-qT)·N(d1) - K·e^(-rT)·N(d2)
    
    Exact for European options under GBM assumption.

Monte Carlo:
    C ≈ e^(-rT) × (1/N) × Σ max(S_T^(i) - K, 0)
    
    Error: O(1/√N) - slow convergence but flexible.

Finite Difference:
    Solve PDE: ∂V/∂t + ½σ²S²·∂²V/∂S² + (r-q)S·∂V/∂S - rV = 0
    
    Error: O(Δt) + O(Δx²) for explicit, better for implicit/CN.

Validation Metrics
------------------
- Absolute Error: |Price_numerical - Price_analytical|
- Relative Error: |Error| / |Price_analytical|
- Convergence Rate: How error decreases with more compute
- Runtime: Computational cost

Production Context
------------------
At a hedge fund:
- All pricing models must be validated before production
- Benchmark against analytical where available
- Document convergence properties
- Establish error tolerances for sign-off

Prerequisites
-------------
- Understanding of pricing methods (examples/pricing/)
- Basic numerical analysis concepts

Run This Example
----------------
    cd /path/to/QuantStrata
    PYTHONPATH=. python examples/ml/03_model_validation.py

Author: QuantStrata Team
===============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np

# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

ENABLE_PLOTTING = True

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available - plotting disabled")


# =============================================================================
# BLACK-SCHOLES FUNCTIONS (Using Library)
# =============================================================================

# Import library BSM functions - production implementations
from src.models.analytic.black_scholes_merton.base import (
    vanilla_price as _lib_vanilla_price,
    vanilla_delta as _lib_vanilla_delta,
    vanilla_gamma as _lib_vanilla_gamma,
    vanilla_vega as _lib_vanilla_vega,
    d1_d2 as _lib_d1_d2,
)

# GBM dynamics for Monte Carlo
from src.models.dynamics.gbm_dynamics import GbmDynamicsSimulator, GbmScheme


def bs_d1d2(S: float, K: float, T: float, r: float, q: float, sigma: float) -> Tuple[float, float]:
    """Compute d1 and d2 using library function."""
    if T <= 1e-10:
        return 0.0, 0.0
    carry = r - q
    return _lib_d1_d2(spot=S, strike=K, expiry=T, discount_rate=r, carry=carry, vol=sigma)


def bs_call_price(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Black-Scholes call price using library function."""
    if T <= 1e-10:
        return max(S - K, 0)
    carry = r - q
    return _lib_vanilla_price(
        option_type="call", spot=S, strike=K, expiry=T, 
        discount_rate=r, carry=carry, vol=sigma
    )


def bs_put_price(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Black-Scholes put price using library function."""
    if T <= 1e-10:
        return max(K - S, 0)
    carry = r - q
    return _lib_vanilla_price(
        option_type="put", spot=S, strike=K, expiry=T,
        discount_rate=r, carry=carry, vol=sigma
    )


def bs_delta(S: float, K: float, T: float, r: float, q: float, sigma: float, is_call: bool = True) -> float:
    """Black-Scholes delta using library function."""
    if T <= 1e-10:
        return 1.0 if (is_call and S > K) else (-1.0 if (not is_call and S < K) else 0.0)
    carry = r - q
    option_type = "call" if is_call else "put"
    return _lib_vanilla_delta(
        option_type=option_type, spot=S, strike=K, expiry=T,
        discount_rate=r, carry=carry, vol=sigma
    )


def bs_gamma(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Black-Scholes gamma using library function."""
    if T <= 1e-10:
        return 0.0
    carry = r - q
    return _lib_vanilla_gamma(
        spot=S, strike=K, expiry=T,
        discount_rate=r, carry=carry, vol=sigma
    )


def bs_vega(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Black-Scholes vega (per 1% vol move) using library function."""
    if T <= 1e-10:
        return 0.0
    carry = r - q
    # Library returns vega per 1 vol point, we divide by 100 for per 1% move
    return _lib_vanilla_vega(
        spot=S, strike=K, expiry=T,
        discount_rate=r, carry=carry, vol=sigma
    ) / 100


# =============================================================================
# MONTE CARLO PRICER
# =============================================================================

def mc_price(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    n_paths: int,
    is_call: bool = True,
    seed: int = 42,
    antithetic: bool = True,
) -> Tuple[float, float, float]:
    """
    Monte Carlo option price using library GbmDynamicsSimulator.
    
    Parameters
    ----------
    n_paths : int
        Number of simulation paths.
    antithetic : bool
        Use antithetic variates for variance reduction.
    
    Returns
    -------
    Tuple[float, float, float]
        Price, standard error, runtime (seconds).
    """
    start = time.time()
    
    # Use library GBM simulator for path generation
    simulator = GbmDynamicsSimulator(scheme=GbmScheme.LOG_EULER)
    
    # Simulate paths (only need terminal value, so 1 step is sufficient)
    # But for more realistic MC, use multiple steps
    drift = r - q  # Risk-neutral drift
    
    paths = simulator.simulate(
        S0=S,
        drift=drift,
        sigma=sigma,
        T=T,
        n_steps=1,  # Single step for European option
        n_paths=n_paths,
        seed=seed,
        antithetic=antithetic,
    )
    
    # Terminal prices
    S_T = paths[:, -1]
    
    # Compute payoffs
    if is_call:
        payoffs = np.maximum(S_T - K, 0)
    else:
        payoffs = np.maximum(K - S_T, 0)
    
    # Discount
    df = np.exp(-r * T)
    disc_payoffs = df * payoffs
    
    # Estimate price and standard error
    price = np.mean(disc_payoffs)
    stderr = np.std(disc_payoffs) / np.sqrt(len(disc_payoffs))
    
    runtime = time.time() - start
    
    return price, stderr, runtime


# =============================================================================
# FINITE DIFFERENCE PRICER
# =============================================================================

def fd_price(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    n_space: int,
    n_time: int,
    is_call: bool = True,
    scheme: str = "cn",  # "explicit", "implicit", "cn"
) -> Tuple[float, float]:
    """
    Finite difference option price.
    
    Parameters
    ----------
    n_space : int
        Number of space (S) grid points.
    n_time : int
        Number of time steps.
    scheme : str
        Differencing scheme: "explicit", "implicit", or "cn" (Crank-Nicolson).
    
    Returns
    -------
    Tuple[float, float]
        Price and runtime (seconds).
    """
    start = time.time()
    
    # Grid parameters
    S_max = 3 * S  # Far boundary
    dS = S_max / n_space
    dt = T / n_time
    
    # Grid
    S_grid = np.linspace(0, S_max, n_space + 1)
    
    # Terminal condition (payoff)
    if is_call:
        V = np.maximum(S_grid - K, 0)
    else:
        V = np.maximum(K - S_grid, 0)
    
    # Coefficients for tridiagonal system
    j = np.arange(1, n_space)
    a = 0.5 * dt * (sigma ** 2 * j ** 2 - (r - q) * j)
    b = 1 - dt * (sigma ** 2 * j ** 2 + r)
    c = 0.5 * dt * (sigma ** 2 * j ** 2 + (r - q) * j)
    
    if scheme == "implicit" or scheme == "cn":
        # For implicit/CN, we need to solve a tridiagonal system
        a_imp = -0.5 * dt * (sigma ** 2 * j ** 2 - (r - q) * j)
        b_imp = 1 + dt * (sigma ** 2 * j ** 2 + r)
        c_imp = -0.5 * dt * (sigma ** 2 * j ** 2 + (r - q) * j)
    
    # Time stepping
    for _ in range(n_time):
        V_old = V.copy()
        
        if scheme == "explicit":
            # Explicit Euler
            V[1:-1] = a * V_old[:-2] + b * V_old[1:-1] + c * V_old[2:]
        
        elif scheme == "implicit":
            # Implicit Euler (solve tridiagonal)
            V[1:-1] = _solve_tridiagonal(a_imp, b_imp, c_imp, V_old[1:-1])
        
        elif scheme == "cn":
            # Crank-Nicolson
            # RHS: explicit part
            rhs = 0.5 * a * V_old[:-2] + (1 - 0.5 * (b_imp - 1)) * V_old[1:-1] + 0.5 * c * V_old[2:]
            # Solve implicit part
            V[1:-1] = _solve_tridiagonal(0.5 * a_imp, 0.5 * b_imp + 0.5, 0.5 * c_imp, rhs)
        
        # Boundary conditions
        if is_call:
            V[0] = 0
            V[-1] = S_max - K * np.exp(-r * (T - (_ + 1) * dt))
        else:
            V[0] = K * np.exp(-r * (T - (_ + 1) * dt))
            V[-1] = 0
    
    # Interpolate to get price at S
    idx = int(S / dS)
    if idx >= n_space:
        price = V[-1]
    else:
        w = (S - S_grid[idx]) / dS
        price = (1 - w) * V[idx] + w * V[idx + 1]
    
    runtime = time.time() - start
    
    return price, runtime


def _solve_tridiagonal(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Solve tridiagonal system using Thomas algorithm."""
    n = len(d)
    c_prime = np.zeros(n)
    d_prime = np.zeros(n)
    x = np.zeros(n)
    
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]
    
    for i in range(1, n):
        denom = b[i] - a[i] * c_prime[i - 1]
        c_prime[i] = c[i] / denom if i < n - 1 else 0
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / denom
    
    x[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]
    
    return x


# =============================================================================
# SECTION 1: Single Point Comparison
# =============================================================================

def run_single_comparison() -> Dict[str, Any]:
    """
    Compare methods at a single point.
    
    Returns
    -------
    Dict[str, Any]
        Comparison results.
    """
    logger.info("=" * 70)
    logger.info("SECTION 1: Single Point Comparison")
    logger.info("=" * 70)
    
    # Parameters
    S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.0, 0.20
    
    logger.info("")
    logger.info("Option Parameters:")
    logger.info(f"  Spot:       ${S:.2f}")
    logger.info(f"  Strike:     ${K:.2f}")
    logger.info(f"  Expiry:     {T:.2f} years")
    logger.info(f"  Rate:       {r:.2%}")
    logger.info(f"  Vol:        {sigma:.2%}")
    
    # Analytical (benchmark)
    start = time.time()
    bsm_price = bs_call_price(S, K, T, r, q, sigma)
    bsm_time = time.time() - start
    bsm_delta = bs_delta(S, K, T, r, q, sigma)
    bsm_gamma = bs_gamma(S, K, T, r, q, sigma)
    
    # Monte Carlo
    mc_price_val, mc_stderr, mc_time = mc_price(S, K, T, r, q, sigma, n_paths=100_000)
    mc_error = mc_price_val - bsm_price
    mc_rel_error = abs(mc_error) / bsm_price
    
    # Finite Difference
    fd_price_val, fd_time = fd_price(S, K, T, r, q, sigma, n_space=200, n_time=200, scheme="cn")
    fd_error = fd_price_val - bsm_price
    fd_rel_error = abs(fd_error) / bsm_price
    
    logger.info("")
    logger.info("Pricing Results:")
    logger.info("-" * 70)
    logger.info(f"{'Method':<20} {'Price':>12} {'Error':>12} {'Rel Error':>12} {'Time (ms)':>12}")
    logger.info("-" * 70)
    logger.info(f"{'BSM (Analytical)':<20} ${bsm_price:>11.6f} {'---':>12} {'---':>12} {bsm_time*1000:>11.3f}")
    logger.info(f"{'Monte Carlo':<20} ${mc_price_val:>11.6f} ${mc_error:>+11.6f} {mc_rel_error:>11.4%} {mc_time*1000:>11.3f}")
    logger.info(f"{'Finite Difference':<20} ${fd_price_val:>11.6f} ${fd_error:>+11.6f} {fd_rel_error:>11.4%} {fd_time*1000:>11.3f}")
    logger.info("-" * 70)
    
    logger.info("")
    logger.info(f"MC 95% CI: [{mc_price_val - 1.96*mc_stderr:.6f}, {mc_price_val + 1.96*mc_stderr:.6f}]")
    
    return {
        "bsm_price": bsm_price,
        "mc_price": mc_price_val,
        "mc_stderr": mc_stderr,
        "fd_price": fd_price_val,
        "bsm_delta": bsm_delta,
        "bsm_gamma": bsm_gamma,
    }


# =============================================================================
# SECTION 2: Monte Carlo Convergence
# =============================================================================

def run_mc_convergence(bsm_price: float) -> Tuple[List[int], List[float], List[float]]:
    """
    Analyze Monte Carlo convergence.
    
    Returns
    -------
    Tuple[List[int], List[float], List[float]]
        Number of paths, errors, standard errors.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 2: Monte Carlo Convergence")
    logger.info("=" * 70)
    
    S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.0, 0.20
    
    path_counts = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000]
    
    errors = []
    stderrs = []
    
    logger.info("")
    logger.info("MC Convergence Analysis:")
    logger.info("-" * 60)
    logger.info(f"{'N Paths':>12} {'Price':>12} {'Error':>12} {'Std Err':>12}")
    logger.info("-" * 60)
    
    for n_paths in path_counts:
        price, stderr, _ = mc_price(S, K, T, r, q, sigma, n_paths)
        error = abs(price - bsm_price)
        errors.append(error)
        stderrs.append(stderr)
        
        logger.info(f"{n_paths:>12,} ${price:>11.6f} ${error:>11.6f} ${stderr:>11.6f}")
    
    logger.info("-" * 60)
    
    # Verify O(1/√N) convergence
    expected_ratio = np.sqrt(path_counts[-1] / path_counts[0])
    actual_ratio = stderrs[0] / stderrs[-1]
    
    logger.info("")
    logger.info(f"Convergence Rate Check:")
    logger.info(f"  Expected stderr ratio (√N): {expected_ratio:.2f}")
    logger.info(f"  Actual stderr ratio:        {actual_ratio:.2f}")
    
    return path_counts, errors, stderrs


# =============================================================================
# SECTION 3: Finite Difference Convergence
# =============================================================================

def run_fd_convergence(bsm_price: float) -> Tuple[List[int], List[float]]:
    """
    Analyze Finite Difference convergence.
    
    Returns
    -------
    Tuple[List[int], List[float]]
        Grid sizes, errors.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 3: Finite Difference Convergence")
    logger.info("=" * 70)
    
    S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.0, 0.20
    
    grid_sizes = [25, 50, 100, 200, 400, 800]
    
    errors = []
    
    logger.info("")
    logger.info("FD Convergence Analysis (Crank-Nicolson):")
    logger.info("-" * 60)
    logger.info(f"{'Grid Size':>12} {'Price':>12} {'Error':>12} {'Time (ms)':>12}")
    logger.info("-" * 60)
    
    for n in grid_sizes:
        price, runtime = fd_price(S, K, T, r, q, sigma, n_space=n, n_time=n, scheme="cn")
        error = abs(price - bsm_price)
        errors.append(error)
        
        logger.info(f"{n:>12} ${price:>11.6f} ${error:>11.6f} {runtime*1000:>11.3f}")
    
    logger.info("-" * 60)
    
    # Check convergence rate
    if len(errors) >= 2:
        ratio = errors[0] / errors[-1]
        grid_ratio = grid_sizes[-1] / grid_sizes[0]
        expected_ratio = grid_ratio ** 2  # O(h²) for CN
        
        logger.info("")
        logger.info(f"Convergence Rate Check:")
        logger.info(f"  Grid refined by: {grid_ratio:.0f}x")
        logger.info(f"  Expected error reduction (O(h²)): {expected_ratio:.0f}x")
        logger.info(f"  Actual error reduction: {ratio:.1f}x")
    
    return grid_sizes, errors


# =============================================================================
# SECTION 4: Greek Validation
# =============================================================================

def run_greek_validation() -> Dict[str, Dict[str, float]]:
    """
    Validate Greeks computed via bump-and-reprice.
    
    Returns
    -------
    Dict[str, Dict[str, float]]
        Greek validation results.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 4: Greek Validation")
    logger.info("=" * 70)
    
    S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.0, 0.20
    
    # Analytical Greeks
    bsm_d = bs_delta(S, K, T, r, q, sigma)
    bsm_g = bs_gamma(S, K, T, r, q, sigma)
    bsm_v = bs_vega(S, K, T, r, q, sigma)
    
    # Bump-and-reprice for MC
    bump = 0.01  # 1%
    mc_base, _, _ = mc_price(S, K, T, r, q, sigma, n_paths=100000, seed=42)
    mc_up, _, _ = mc_price(S * (1 + bump), K, T, r, q, sigma, n_paths=100000, seed=42)
    mc_down, _, _ = mc_price(S * (1 - bump), K, T, r, q, sigma, n_paths=100000, seed=42)
    
    mc_delta = (mc_up - mc_down) / (2 * S * bump)
    mc_gamma = (mc_up - 2 * mc_base + mc_down) / (S * bump) ** 2
    
    mc_vol_up, _, _ = mc_price(S, K, T, r, q, sigma + 0.01, n_paths=100000, seed=42)
    mc_vega = (mc_vol_up - mc_base)  # Per 1% vol move
    
    # Bump-and-reprice for FD
    fd_base, _ = fd_price(S, K, T, r, q, sigma, 200, 200)
    fd_up, _ = fd_price(S * (1 + bump), K, T, r, q, sigma, 200, 200)
    fd_down, _ = fd_price(S * (1 - bump), K, T, r, q, sigma, 200, 200)
    
    fd_delta = (fd_up - fd_down) / (2 * S * bump)
    fd_gamma = (fd_up - 2 * fd_base + fd_down) / (S * bump) ** 2
    
    fd_vol_up, _ = fd_price(S, K, T, r, q, sigma + 0.01, 200, 200)
    fd_vega = (fd_vol_up - fd_base)
    
    logger.info("")
    logger.info("Greek Validation (Bump-and-Reprice vs Analytical):")
    logger.info("-" * 70)
    logger.info(f"{'Greek':<10} {'BSM':>12} {'MC':>12} {'FD':>12} {'MC Err%':>12} {'FD Err%':>12}")
    logger.info("-" * 70)
    
    mc_d_err = (mc_delta - bsm_d) / bsm_d * 100 if bsm_d != 0 else 0
    fd_d_err = (fd_delta - bsm_d) / bsm_d * 100 if bsm_d != 0 else 0
    logger.info(f"{'Delta':<10} {bsm_d:>12.6f} {mc_delta:>12.6f} {fd_delta:>12.6f} {mc_d_err:>11.2f}% {fd_d_err:>11.2f}%")
    
    mc_g_err = (mc_gamma - bsm_g) / bsm_g * 100 if bsm_g != 0 else 0
    fd_g_err = (fd_gamma - bsm_g) / bsm_g * 100 if bsm_g != 0 else 0
    logger.info(f"{'Gamma':<10} {bsm_g:>12.6f} {mc_gamma:>12.6f} {fd_gamma:>12.6f} {mc_g_err:>11.2f}% {fd_g_err:>11.2f}%")
    
    mc_v_err = (mc_vega - bsm_v) / bsm_v * 100 if bsm_v != 0 else 0
    fd_v_err = (fd_vega - bsm_v) / bsm_v * 100 if bsm_v != 0 else 0
    logger.info(f"{'Vega':<10} {bsm_v:>12.6f} {mc_vega:>12.6f} {fd_vega:>12.6f} {mc_v_err:>11.2f}% {fd_v_err:>11.2f}%")
    
    logger.info("-" * 70)
    
    return {
        "bsm": {"delta": bsm_d, "gamma": bsm_g, "vega": bsm_v},
        "mc": {"delta": mc_delta, "gamma": mc_gamma, "vega": mc_vega},
        "fd": {"delta": fd_delta, "gamma": fd_gamma, "vega": fd_vega},
    }


# =============================================================================
# SECTION 5: Visualization
# =============================================================================

def visualize_results(
    mc_convergence: Tuple[List[int], List[float], List[float]],
    fd_convergence: Tuple[List[int], List[float]],
) -> None:
    """Create validation visualizations."""
    if not MATPLOTLIB_AVAILABLE or not ENABLE_PLOTTING:
        logger.info("Skipping plots (matplotlib not available or disabled)")
        return
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 5: Visualization")
    logger.info("=" * 70)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    path_counts, mc_errors, mc_stderrs = mc_convergence
    grid_sizes, fd_errors = fd_convergence
    
    # -------------------------------------------------------------------------
    # Plot 1: MC Error Convergence
    # -------------------------------------------------------------------------
    ax1 = axes[0, 0]
    
    ax1.loglog(path_counts, mc_errors, 'b-o', linewidth=2, markersize=8, label='Actual Error')
    ax1.loglog(path_counts, mc_stderrs, 'g--s', linewidth=2, markersize=8, label='Std Error')
    
    # Theoretical O(1/√N) line
    ref_line = mc_stderrs[0] * np.sqrt(path_counts[0]) / np.sqrt(path_counts)
    ax1.loglog(path_counts, ref_line, 'r:', linewidth=2, label='O(1/√N) reference')
    
    ax1.set_xlabel('Number of Paths')
    ax1.set_ylabel('Error')
    ax1.set_title('Monte Carlo Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3, which='both')
    
    # -------------------------------------------------------------------------
    # Plot 2: FD Error Convergence
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]
    
    ax2.loglog(grid_sizes, fd_errors, 'b-o', linewidth=2, markersize=8, label='FD Error')
    
    # Theoretical O(h²) line
    ref_line = fd_errors[0] * (grid_sizes[0] / np.array(grid_sizes)) ** 2
    ax2.loglog(grid_sizes, ref_line, 'r:', linewidth=2, label='O(h²) reference')
    
    ax2.set_xlabel('Grid Size (N)')
    ax2.set_ylabel('Error')
    ax2.set_title('Finite Difference Convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    
    # -------------------------------------------------------------------------
    # Plot 3: Error vs Runtime Trade-off
    # -------------------------------------------------------------------------
    ax3 = axes[1, 0]
    
    S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.0, 0.20
    bsm_price = bs_call_price(S, K, T, r, q, sigma)
    
    mc_runtimes = []
    mc_errors_rt = []
    for n in path_counts:
        price, _, runtime = mc_price(S, K, T, r, q, sigma, n)
        mc_runtimes.append(runtime * 1000)
        mc_errors_rt.append(abs(price - bsm_price))
    
    fd_runtimes = []
    fd_errors_rt = []
    for n in grid_sizes:
        price, runtime = fd_price(S, K, T, r, q, sigma, n, n)
        fd_runtimes.append(runtime * 1000)
        fd_errors_rt.append(abs(price - bsm_price))
    
    ax3.loglog(mc_runtimes, mc_errors_rt, 'b-o', linewidth=2, markersize=8, label='Monte Carlo')
    ax3.loglog(fd_runtimes, fd_errors_rt, 'g-s', linewidth=2, markersize=8, label='Finite Difference')
    
    ax3.set_xlabel('Runtime (ms)')
    ax3.set_ylabel('Error')
    ax3.set_title('Error vs Computational Cost')
    ax3.legend()
    ax3.grid(True, alpha=0.3, which='both')
    
    # -------------------------------------------------------------------------
    # Plot 4: Method comparison bar chart
    # -------------------------------------------------------------------------
    ax4 = axes[1, 1]
    
    # Fixed compute budget comparison
    methods = ['BSM\n(Analytical)', 'MC\n(100K paths)', 'FD\n(200×200)']
    
    bsm = bs_call_price(S, K, T, r, q, sigma)
    mc, mc_se, _ = mc_price(S, K, T, r, q, sigma, 100000)
    fd, _ = fd_price(S, K, T, r, q, sigma, 200, 200)
    
    prices = [bsm, mc, fd]
    errors = [0, abs(mc - bsm), abs(fd - bsm)]
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, prices, width, label='Price', color='#2E86AB')
    bars2 = ax4.bar(x + width/2, errors, width, label='Error', color='#E94F37')
    
    ax4.set_xticks(x)
    ax4.set_xticklabels(methods)
    ax4.set_ylabel('Value ($)')
    ax4.set_title('Method Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add price labels
    for bar, price in zip(bars1, prices):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'${price:.4f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.show(block=True)
    
    logger.info("Visualization complete")


# =============================================================================
# SUMMARY
# =============================================================================

def print_summary() -> None:
    """Print summary of key concepts."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    
    summary = """
    ┌─────────────────────────────────────────────────────────────────────┐
    │                         KEY TAKEAWAYS                                │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                      │
    │  1. Analytical (BSM):                                               │
    │     - Exact for European options under GBM                          │
    │     - Fast, use as benchmark                                        │
    │     - Limited to simple payoffs                                     │
    │                                                                      │
    │  2. Monte Carlo:                                                    │
    │     - Flexible for any payoff (path-dependent, exotic)              │
    │     - Convergence: O(1/√N) - slow                                   │
    │     - Variance reduction: antithetic, control variates              │
    │                                                                      │
    │  3. Finite Difference:                                              │
    │     - Solves PDE directly on a grid                                 │
    │     - Convergence: O(Δt) + O(Δx²) for Crank-Nicolson                │
    │     - Good for American options (free boundary)                     │
    │                                                                      │
    │  4. Production Validation:                                          │
    │     - Always benchmark against analytical if available              │
    │     - Document convergence properties                               │
    │     - Establish error tolerance thresholds                          │
    │     - Include Greek validation                                      │
    │                                                                      │
    │  RECOMMENDATION: Use BSM for European vanilla, MC for exotics,      │
    │  FD for American options                                            │
    │                                                                      │
    └─────────────────────────────────────────────────────────────────────┘
    """
    logger.info(summary)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main(args: argparse.Namespace) -> None:
    """
    Main entry point.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    """
    global ENABLE_PLOTTING
    ENABLE_PLOTTING = args.plot
    
    logger.info("=" * 70)
    logger.info("Model Validation Example")
    logger.info("=" * 70)
    
    try:
        # Section 1: Single point comparison
        results = run_single_comparison()
        bsm_price = results["bsm_price"]
        
        # Section 2: MC convergence
        mc_convergence = run_mc_convergence(bsm_price)
        
        # Section 3: FD convergence
        fd_convergence = run_fd_convergence(bsm_price)
        
        # Section 4: Greek validation
        greek_results = run_greek_validation()
        
        # Section 5: Visualization
        visualize_results(mc_convergence, fd_convergence)
        
        # Summary
        print_summary()
        
        logger.info("Example completed successfully!")
        
    except Exception as e:
        logger.exception(f"Example failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Model Validation Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=True,
        help="Enable plotting (default: True)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_false",
        dest="plot",
        help="Disable plotting",
    )
    
    args = parser.parse_args()
    main(args)

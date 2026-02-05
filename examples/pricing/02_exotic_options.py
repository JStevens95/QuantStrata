#!/usr/bin/env python3
"""
===============================================================================
Exotic Options Pricing: Barriers, Asians, Lookbacks, and Touch Options
===============================================================================

This example demonstrates pricing path-dependent FX options using Monte Carlo
simulation. These exotic payoffs require simulation because their value depends
on the entire price path, not just the terminal value.

Learning Objectives
-------------------
1. **Path Dependence**: Understand why some options require Monte Carlo
2. **Barrier Options**: Knock-in/knock-out mechanics and in-out parity
3. **Asian Options**: Averaging effect and volatility reduction
4. **Lookback Options**: Optimal hindsight pricing
5. **Touch Options**: Binary payoffs and touch probabilities

Mathematical Framework
----------------------
Barrier Options:
    - Knock-Out: pays vanilla if barrier never breached
    - Knock-In: pays vanilla only if barrier breached
    - In-Out Parity: C_KI + C_KO = C_vanilla

Asian Options (arithmetic):
    Payoff = max(0, A - K)
    where A = (1/n) Σ S_ti (arithmetic average)
    
    Key insight: Var(A) < Var(S_T) → cheaper than vanilla

Lookback Options (floating strike call):
    Payoff = S_T - min(S_t)  (buy at the low)
    
    Always ITM → significantly more expensive

Touch Options:
    One-Touch: pays 1 if S touches barrier B at any time
    No-Touch: pays 1 if S never touches barrier B
    Parity: OT + NT = df (discount factor)

Production Context
------------------
At a hedge fund:
- Barriers are popular for FX hedging (cheaper than vanilla)
- Asians are used for commodity hedging (average price exposure)
- Lookbacks are exotic and rarely traded (expensive)
- Touch options are used for range bets and digital structures

Prerequisites
-------------
- Examples in fundamentals/ and pricing/01_fx_vanilla_pricing.py
- Understanding of Monte Carlo simulation

Run This Example
----------------
    cd /path/to/QuantStrata
    PYTHONPATH=. python examples/pricing/02_exotic_options.py

Author: QuantStrata Team
===============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# -----------------------------------------------------------------------------
# QuantStrata imports
# -----------------------------------------------------------------------------
from src.marketdata.core.ids import MarketId
from src.marketdata.core.interfaces import Quote
from src.marketdata.core.market import Market
from src.marketdata.curves.term_structure import FlatZeroRateCurve
from src.marketdata.surfaces.vol_surface import FlatVolSurface

# GBM dynamics for path simulation (use library instead of standalone code)
from src.models.dynamics.gbm_dynamics import GbmDynamicsSimulator, GbmScheme

# Library exotic pricers (production alternatives to manual implementations)
from src.pricers.fx.european_bsm_mc import (
    FxVanillaEuropeanOptionMcPricer,
    FxBarrierEuropeanOptionMcPricer,
    FxAsianEuropeanOptionMcPricer,
    FxLookbackEuropeanOptionMcPricer,
    FxTouchEuropeanOptionMcPricer,
)


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
# CONSTANTS
# =============================================================================

# Market IDs - use mkt_type (not data_type)
SPOT_ID = MarketId(asset_class="FX", mkt_type="SPOT", name="TEST")
DOM_CURVE = MarketId(asset_class="IR", mkt_type="CURVE", name="DOM")
FOR_CURVE = MarketId(asset_class="IR", mkt_type="CURVE", name="FOR")
VOL_ID = MarketId(asset_class="FX", mkt_type="VOL", name="TEST")

# Color scheme for plots
COLORS = {
    'vanilla': '#2E86AB',
    'barrier': '#E94F37',
    'asian': '#8B5CF6',
    'lookback': '#10B981',
    'touch': '#F59E0B',
}


# =============================================================================
# SECTION 1: Market Setup
# =============================================================================

def setup_market() -> Tuple[Market, dict]:
    """
    Create market snapshot for exotic option pricing.
    
    Returns
    -------
    Tuple[Market, dict]
        Market and parameters dictionary.
    """
    logger.info("=" * 70)
    logger.info("SECTION 1: Market Setup")
    logger.info("=" * 70)
    
    # Market parameters
    spot = 100.0
    r_dom = 0.05
    r_for = 0.02
    vol = 0.20
    T = 1.0
    
    # Create market with correct API (sigma, not vol)
    market = Market(
        asof="2026-01-28",
        quotes={SPOT_ID: Quote(value=spot)},
        curves={
            DOM_CURVE: FlatZeroRateCurve(continuously_compounded_rate=r_dom),
            FOR_CURVE: FlatZeroRateCurve(continuously_compounded_rate=r_for),
        },
        vols={VOL_ID: FlatVolSurface(sigma=vol)},
    )
    
    params = {
        "spot": spot,
        "r_dom": r_dom,
        "r_for": r_for,
        "vol": vol,
        "T": T,
        "K": 100.0,  # ATM strike
    }
    
    # Forward price under risk-neutral measure
    forward = spot * np.exp((r_dom - r_for) * T)
    
    logger.info("")
    logger.info("Market Parameters:")
    logger.info(f"  Spot:           {spot}")
    logger.info(f"  Domestic rate:  {r_dom:.2%}")
    logger.info(f"  Foreign rate:   {r_for:.2%}")
    logger.info(f"  Volatility:     {vol:.2%}")
    logger.info(f"  Time to expiry: {T} year")
    logger.info(f"  Forward:        {forward:.4f}")
    
    return market, params


# =============================================================================
# SECTION 2: Path Simulation
# =============================================================================

def simulate_paths(
    S0: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    n_paths: int = 50000,
    n_steps: int = 252,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate GBM paths using QuantStrata's library dynamics.
    
    This function uses the library's GbmDynamicsSimulator for production-grade
    path simulation with proper variance reduction and numerical stability.
    
    Parameters
    ----------
    S0 : float
        Initial spot price.
    r : float
        Domestic risk-free rate.
    q : float
        Foreign risk-free rate (cost of carry).
    sigma : float
        Volatility.
    T : float
        Time to expiry.
    n_paths : int
        Number of simulation paths.
    n_steps : int
        Number of time steps per path.
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    np.ndarray
        Shape (n_steps + 1, n_paths) with simulated paths.
    
    Mathematical Details
    --------------------
    Under risk-neutral measure:
        dS/S = (r - q) dt + σ dW
    
    Exact solution (log-Euler scheme):
        S_t = S_0 · exp((r - q - σ²/2)t + σ W_t)
    
    Library Implementation
    ----------------------
    Uses GbmDynamicsSimulator with LOG_EULER scheme for numerical stability.
    Antithetic variates are used for variance reduction.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 2: Path Simulation (Library GbmDynamicsSimulator)")
    logger.info("=" * 70)
    
    # Use library GBM simulator
    simulator = GbmDynamicsSimulator(scheme=GbmScheme.LOG_EULER)
    
    # Drift for FX options: r_domestic - r_foreign
    drift = r - q
    
    # Simulate paths using library
    # Returns shape (n_paths, n_steps + 1), we transpose for compatibility
    paths_raw = simulator.simulate(
        S0=S0,
        drift=drift,
        sigma=sigma,
        T=T,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
        antithetic=True,  # Variance reduction
    )
    
    # Transpose to (n_steps + 1, n_paths) for backward compatibility with payoff functions
    paths = paths_raw.T
    
    logger.info("")
    logger.info(f"Simulated {n_paths:,} paths with {n_steps} steps")
    logger.info(f"  Using: GbmDynamicsSimulator (LOG_EULER scheme)")
    logger.info(f"  Paths shape: {paths.shape}")
    logger.info(f"  Mean terminal: {np.mean(paths[-1, :]):.4f}")
    logger.info(f"  Std terminal:  {np.std(paths[-1, :]):.4f}")
    
    return paths


# =============================================================================
# SECTION 3: Vanilla Option (Benchmark)
# =============================================================================

def price_vanilla(
    paths: np.ndarray,
    K: float,
    r: float,
    T: float,
    option_type: str = 'call',
) -> Tuple[float, float]:
    """
    Price vanilla option using Monte Carlo.
    
    Returns
    -------
    Tuple[float, float]
        Price and standard error.
    """
    # Get terminal values
    terminal = paths[-1, :]
    
    # Compute payoffs
    if option_type == 'call':
        payoffs = np.maximum(terminal - K, 0)
    else:
        payoffs = np.maximum(K - terminal, 0)
    
    # Discount to present value
    disc_payoffs = np.exp(-r * T) * payoffs
    
    # Monte Carlo estimator
    price = np.mean(disc_payoffs)
    stderr = np.std(disc_payoffs) / np.sqrt(len(disc_payoffs))
    
    return price, stderr


def run_vanilla_benchmark(paths: np.ndarray, params: dict) -> Tuple[float, float]:
    """
    Run vanilla option pricing as benchmark.
    
    Returns
    -------
    Tuple[float, float]
        Call and put prices.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 3: Vanilla Option (Benchmark)")
    logger.info("=" * 70)
    
    K = params["K"]
    r_dom = params["r_dom"]
    T = params["T"]
    
    vanilla_call, vanilla_se = price_vanilla(paths, K, r_dom, T, 'call')
    vanilla_put, put_se = price_vanilla(paths, K, r_dom, T, 'put')
    
    logger.info("")
    logger.info(f"Vanilla option prices (K={K}):")
    logger.info(f"  Call: {vanilla_call:.4f} ± {vanilla_se * 1.96:.4f} (95% CI)")
    logger.info(f"  Put:  {vanilla_put:.4f} ± {put_se * 1.96:.4f}")
    
    return vanilla_call, vanilla_put


# =============================================================================
# SECTION 4: Barrier Options
# =============================================================================

def price_barrier(
    paths: np.ndarray,
    K: float,
    B: float,
    r: float,
    T: float,
    option_type: str = 'call',
    barrier_type: str = 'up_and_out',
    rebate: float = 0.0,
) -> Tuple[float, float]:
    """
    Price barrier option using Monte Carlo.
    
    Parameters
    ----------
    B : float
        Barrier level.
    barrier_type : str
        One of: 'up_and_out', 'up_and_in', 'down_and_out', 'down_and_in'.
    rebate : float
        Rebate paid if barrier is breached (for knock-out).
    
    Returns
    -------
    Tuple[float, float]
        Price and standard error.
    
    In-Out Parity
    -------------
    For any barrier B:
        C_KI + C_KO = C_vanilla
    """
    terminal = paths[-1, :]
    
    # Determine if barrier was breached
    if barrier_type.startswith('up'):
        breached = np.any(paths >= B, axis=0)
    else:
        breached = np.any(paths <= B, axis=0)
    
    # Compute vanilla payoff
    if option_type == 'call':
        vanilla = np.maximum(terminal - K, 0)
    else:
        vanilla = np.maximum(K - terminal, 0)
    
    # Apply barrier condition
    if barrier_type.endswith('out'):
        # Knock-out: pay rebate if breached, else vanilla
        payoffs = np.where(breached, rebate, vanilla)
    else:
        # Knock-in: pay vanilla only if breached
        payoffs = np.where(breached, vanilla, 0)
    
    # Discount to present value
    disc_payoffs = np.exp(-r * T) * payoffs
    
    # Monte Carlo estimator
    price = np.mean(disc_payoffs)
    stderr = np.std(disc_payoffs) / np.sqrt(len(disc_payoffs))
    
    return price, stderr


def run_barrier_analysis(paths: np.ndarray, params: dict, vanilla_call: float, vanilla_put: float) -> dict:
    """
    Run barrier option analysis.
    
    Returns
    -------
    dict
        Barrier option prices.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 4: Barrier Options")
    logger.info("=" * 70)
    
    K = params["K"]
    r_dom = params["r_dom"]
    T = params["T"]
    
    # Barrier levels
    B_up = 115.0
    B_down = 85.0
    
    # Up barriers (call options)
    uo_call, uo_se = price_barrier(paths, K, B_up, r_dom, T, 'call', 'up_and_out')
    ui_call, ui_se = price_barrier(paths, K, B_up, r_dom, T, 'call', 'up_and_in')
    
    # Down barriers (put options)
    do_put, do_se = price_barrier(paths, K, B_down, r_dom, T, 'put', 'down_and_out')
    di_put, di_se = price_barrier(paths, K, B_down, r_dom, T, 'put', 'down_and_in')
    
    logger.info("")
    logger.info(f"Barrier option prices:")
    logger.info("")
    logger.info(f"  Up-and-Out Call (K={K}, B={B_up}):")
    logger.info(f"    Price: {uo_call:.4f} ± {uo_se * 1.96:.4f}")
    logger.info(f"    Discount vs Vanilla: {(1 - uo_call / vanilla_call) * 100:.1f}%")
    
    logger.info("")
    logger.info(f"  Up-and-In Call (K={K}, B={B_up}):")
    logger.info(f"    Price: {ui_call:.4f} ± {ui_se * 1.96:.4f}")
    
    logger.info("")
    logger.info(f"  In-Out Parity Check:")
    logger.info(f"    KO + KI = {uo_call + ui_call:.4f}")
    logger.info(f"    Vanilla = {vanilla_call:.4f}")
    logger.info(f"    Error:    {abs(uo_call + ui_call - vanilla_call):.6f}")
    
    logger.info("")
    logger.info(f"  Down-and-Out Put (K={K}, B={B_down}):")
    logger.info(f"    Price: {do_put:.4f} ± {do_se * 1.96:.4f}")
    logger.info(f"    Discount vs Vanilla: {(1 - do_put / vanilla_put) * 100:.1f}%")
    
    return {
        "uo_call": uo_call, "ui_call": ui_call,
        "do_put": do_put, "di_put": di_put,
        "B_up": B_up, "B_down": B_down,
    }


# =============================================================================
# SECTION 5: Asian Options
# =============================================================================

def price_asian(
    paths: np.ndarray,
    K: float,
    r: float,
    T: float,
    option_type: str = 'call',
    avg_type: str = 'arithmetic',
) -> Tuple[float, float]:
    """
    Price Asian option using Monte Carlo.
    
    Parameters
    ----------
    avg_type : str
        'arithmetic' or 'geometric'.
    
    Returns
    -------
    Tuple[float, float]
        Price and standard error.
    
    Averaging Effect
    ----------------
    The averaging reduces the effective volatility, making
    Asian options cheaper than vanilla options.
    
    For arithmetic average: no closed-form solution
    For geometric average: closed-form exists (GBM property)
    """
    # Compute average along path
    if avg_type == 'arithmetic':
        avg = np.mean(paths, axis=0)
    else:
        # Geometric average
        avg = np.exp(np.mean(np.log(paths), axis=0))
    
    # Compute payoffs
    if option_type == 'call':
        payoffs = np.maximum(avg - K, 0)
    else:
        payoffs = np.maximum(K - avg, 0)
    
    # Discount to present value
    disc_payoffs = np.exp(-r * T) * payoffs
    
    # Monte Carlo estimator
    price = np.mean(disc_payoffs)
    stderr = np.std(disc_payoffs) / np.sqrt(len(disc_payoffs))
    
    return price, stderr


def run_asian_analysis(paths: np.ndarray, params: dict, vanilla_call: float) -> dict:
    """
    Run Asian option analysis.
    
    Returns
    -------
    dict
        Asian option prices.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 5: Asian Options")
    logger.info("=" * 70)
    
    K = params["K"]
    r_dom = params["r_dom"]
    T = params["T"]
    
    # Price Asian options
    asian_arith_call, asian_se = price_asian(paths, K, r_dom, T, 'call', 'arithmetic')
    asian_geom_call, _ = price_asian(paths, K, r_dom, T, 'call', 'geometric')
    asian_arith_put, _ = price_asian(paths, K, r_dom, T, 'put', 'arithmetic')
    
    logger.info("")
    logger.info(f"Asian option prices (K={K}):")
    logger.info(f"  Arithmetic Average Call: {asian_arith_call:.4f} ± {asian_se * 1.96:.4f}")
    logger.info(f"  Geometric Average Call:  {asian_geom_call:.4f}")
    logger.info(f"  Arithmetic Average Put:  {asian_arith_put:.4f}")
    
    logger.info("")
    logger.info(f"  Discount vs Vanilla Call: {(1 - asian_arith_call / vanilla_call) * 100:.1f}%")
    logger.info(f"  (Asian cheaper due to averaging → lower volatility)")
    
    # Demonstrate volatility reduction
    terminal = paths[-1, :]
    arith_avg = np.mean(paths, axis=0)
    
    logger.info("")
    logger.info("Volatility Reduction Effect:")
    logger.info(f"  Std(Terminal): {np.std(terminal):.4f}")
    logger.info(f"  Std(Average):  {np.std(arith_avg):.4f}")
    logger.info(f"  Reduction:     {(1 - np.std(arith_avg) / np.std(terminal)) * 100:.1f}%")
    
    return {
        "asian_arith_call": asian_arith_call,
        "asian_geom_call": asian_geom_call,
        "arith_avg": arith_avg,
        "terminal": terminal,
    }


# =============================================================================
# SECTION 6: Lookback Options
# =============================================================================

def price_lookback(
    paths: np.ndarray,
    K: float,
    r: float,
    T: float,
    option_type: str = 'call',
    lookback_type: str = 'floating',
) -> Tuple[float, float]:
    """
    Price lookback option using Monte Carlo.
    
    Parameters
    ----------
    lookback_type : str
        'floating' or 'fixed'.
    
    Returns
    -------
    Tuple[float, float]
        Price and standard error.
    
    Lookback Types
    --------------
    Floating Strike Call: payoff = S_T - min(S_t)  (buy at the low)
    Floating Strike Put:  payoff = max(S_t) - S_T  (sell at the high)
    Fixed Strike Call:    payoff = max(max(S_t) - K, 0)
    Fixed Strike Put:     payoff = max(K - min(S_t), 0)
    """
    terminal = paths[-1, :]
    
    if lookback_type == 'floating':
        if option_type == 'call':
            min_S = np.min(paths, axis=0)
            payoffs = terminal - min_S  # Always positive
        else:
            max_S = np.max(paths, axis=0)
            payoffs = max_S - terminal  # Always positive
    else:
        # Fixed strike
        if option_type == 'call':
            max_S = np.max(paths, axis=0)
            payoffs = np.maximum(max_S - K, 0)
        else:
            min_S = np.min(paths, axis=0)
            payoffs = np.maximum(K - min_S, 0)
    
    # Discount to present value
    disc_payoffs = np.exp(-r * T) * payoffs
    
    # Monte Carlo estimator
    price = np.mean(disc_payoffs)
    stderr = np.std(disc_payoffs) / np.sqrt(len(disc_payoffs))
    
    return price, stderr


def run_lookback_analysis(paths: np.ndarray, params: dict, vanilla_call: float) -> dict:
    """
    Run lookback option analysis.
    
    Returns
    -------
    dict
        Lookback option prices.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 6: Lookback Options")
    logger.info("=" * 70)
    
    K = params["K"]
    r_dom = params["r_dom"]
    T = params["T"]
    
    # Price lookback options
    float_call, float_se = price_lookback(paths, K, r_dom, T, 'call', 'floating')
    float_put, _ = price_lookback(paths, K, r_dom, T, 'put', 'floating')
    fixed_call, _ = price_lookback(paths, K, r_dom, T, 'call', 'fixed')
    fixed_put, _ = price_lookback(paths, K, r_dom, T, 'put', 'fixed')
    
    logger.info("")
    logger.info("Lookback option prices:")
    logger.info(f"  Floating Strike Call: {float_call:.4f} ± {float_se * 1.96:.4f}")
    logger.info(f"  Floating Strike Put:  {float_put:.4f}")
    logger.info(f"  Fixed Strike Call (K={K}): {fixed_call:.4f}")
    logger.info(f"  Fixed Strike Put (K={K}):  {fixed_put:.4f}")
    
    logger.info("")
    logger.info(f"  Premium vs Vanilla Call: {float_call / vanilla_call:.1f}x")
    logger.info(f"  (Lookback is significantly more expensive - guaranteed hindsight)")
    
    return {"float_call": float_call, "float_put": float_put}


# =============================================================================
# SECTION 7: Touch Options
# =============================================================================

def price_touch(
    paths: np.ndarray,
    B: float,
    r: float,
    T: float,
    direction: str = 'up',
    touch_type: str = 'one_touch',
    payout: float = 1.0,
) -> Tuple[float, float]:
    """
    Price touch option using Monte Carlo.
    
    Parameters
    ----------
    B : float
        Barrier level.
    direction : str
        'up' or 'down'.
    touch_type : str
        'one_touch' or 'no_touch'.
    payout : float
        Payout amount if condition is satisfied.
    
    Returns
    -------
    Tuple[float, float]
        Price and standard error.
    
    Touch Parity
    ------------
    One-Touch + No-Touch = df (discount factor)
    """
    # Check if barrier was touched
    if direction == 'up':
        touched = np.any(paths >= B, axis=0)
    else:
        touched = np.any(paths <= B, axis=0)
    
    # Compute payoffs
    if touch_type == 'one_touch':
        payoffs = np.where(touched, payout, 0)
    else:
        payoffs = np.where(touched, 0, payout)
    
    # Discount to present value
    disc_payoffs = np.exp(-r * T) * payoffs
    
    # Monte Carlo estimator
    price = np.mean(disc_payoffs)
    stderr = np.std(disc_payoffs) / np.sqrt(len(disc_payoffs))
    
    return price, stderr


def run_touch_analysis(paths: np.ndarray, params: dict) -> dict:
    """
    Run touch option analysis.
    
    Returns
    -------
    dict
        Touch option prices.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 7: Touch Options")
    logger.info("=" * 70)
    
    r_dom = params["r_dom"]
    T = params["T"]
    
    B_up = 115.0
    B_down = 85.0
    
    # Up touch options
    one_touch_up, ot_se = price_touch(paths, B_up, r_dom, T, 'up', 'one_touch')
    no_touch_up, _ = price_touch(paths, B_up, r_dom, T, 'up', 'no_touch')
    
    # Down touch options
    one_touch_down, _ = price_touch(paths, B_down, r_dom, T, 'down', 'one_touch')
    no_touch_down, _ = price_touch(paths, B_down, r_dom, T, 'down', 'no_touch')
    
    df = np.exp(-r_dom * T)
    
    logger.info("")
    logger.info(f"Touch option prices (payout = 1.0):")
    logger.info("")
    logger.info(f"  One-Touch Up (B={B_up}):")
    logger.info(f"    Price: {one_touch_up:.4f} ± {ot_se * 1.96:.4f}")
    logger.info(f"    Touch probability: {one_touch_up / df * 100:.1f}%")
    
    logger.info("")
    logger.info(f"  No-Touch Up (B={B_up}):")
    logger.info(f"    Price: {no_touch_up:.4f}")
    
    logger.info("")
    logger.info(f"  Touch Parity Check:")
    logger.info(f"    OT + NT = {one_touch_up + no_touch_up:.4f}")
    logger.info(f"    df = {df:.4f}")
    logger.info(f"    Error: {abs(one_touch_up + no_touch_up - df):.6f}")
    
    logger.info("")
    logger.info(f"  One-Touch Down (B={B_down}): {one_touch_down:.4f}")
    logger.info(f"  No-Touch Down (B={B_down}):  {no_touch_down:.4f}")
    
    return {
        "one_touch_up": one_touch_up,
        "no_touch_up": no_touch_up,
        "B_up": B_up,
    }


# =============================================================================
# SECTION 8: Summary Comparison
# =============================================================================

def print_comparison(
    vanilla_call: float,
    barrier_results: dict,
    asian_results: dict,
    lookback_results: dict,
    touch_results: dict,
    params: dict,
) -> None:
    """Print summary comparison table."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 8: Summary Comparison")
    logger.info("=" * 70)
    
    uo_call = barrier_results["uo_call"]
    asian_arith_call = asian_results["asian_arith_call"]
    float_call = lookback_results["float_call"]
    one_touch_up = touch_results["one_touch_up"]
    r_dom = params["r_dom"]
    T = params["T"]
    
    summary = f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    EXOTIC OPTIONS PRICE COMPARISON                    ║
╠══════════════════════════════════════════════════════════════════════╣
║  Option Type                    Price        vs Vanilla               ║
╠══════════════════════════════════════════════════════════════════════╣
║  Vanilla Call (K=100)           {vanilla_call:>8.4f}      (benchmark)             ║
║  Up-Out Call (K=100, B=115)     {uo_call:>8.4f}      {(uo_call / vanilla_call) * 100:>6.1f}% (cheaper)       ║
║  Asian Call (Arithmetic)        {asian_arith_call:>8.4f}      {(asian_arith_call / vanilla_call) * 100:>6.1f}% (cheaper)       ║
║  Lookback Call (Floating)       {float_call:>8.4f}      {(float_call / vanilla_call) * 100:>6.1f}% (expensive)     ║
╠══════════════════════════════════════════════════════════════════════╣
║  One-Touch Up (B=115)           {one_touch_up:>8.4f}      Prob: {one_touch_up * np.exp(r_dom * T) * 100:>5.1f}%           ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    logger.info(summary)


# =============================================================================
# SECTION 9: Visualization
# =============================================================================

def visualize_results(
    paths: np.ndarray,
    vanilla_call: float,
    barrier_results: dict,
    asian_results: dict,
    lookback_results: dict,
    params: dict,
) -> None:
    """Create comprehensive visualizations."""
    if not MATPLOTLIB_AVAILABLE or not ENABLE_PLOTTING:
        logger.info("Skipping plots (matplotlib not available or disabled)")
        return
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 9: Visualization")
    logger.info("=" * 70)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    K = params["K"]
    T = params["T"]
    B_up = barrier_results["B_up"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # -------------------------------------------------------------------------
    # Plot 1: Sample paths with barrier
    # -------------------------------------------------------------------------
    ax = axes[0, 0]
    time_grid = np.linspace(0, T, paths.shape[0])
    
    for i in range(min(50, paths.shape[1])):
        path = paths[:, i]
        hit_idx = np.where(path >= B_up)[0]
        if len(hit_idx) > 0:
            ax.plot(
                time_grid[:hit_idx[0] + 1], path[:hit_idx[0] + 1],
                color=COLORS['barrier'], alpha=0.3, linewidth=0.8,
            )
        else:
            ax.plot(time_grid, path, color=COLORS['vanilla'], alpha=0.3, linewidth=0.8)
    
    ax.axhline(B_up, color='red', linestyle='--', linewidth=2, label=f'Barrier = {B_up}')
    ax.axhline(K, color='gray', linestyle=':', alpha=0.7, label=f'Strike = {K}')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Spot Price')
    ax.set_title('Barrier Option: Path Visualization')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 2: Price comparison
    # -------------------------------------------------------------------------
    ax = axes[0, 1]
    products = ['Vanilla\nCall', 'Up-Out\nCall', 'Asian\nCall', 'Lookback\nCall']
    prices = [
        vanilla_call,
        barrier_results["uo_call"],
        asian_results["asian_arith_call"],
        lookback_results["float_call"],
    ]
    colors = [COLORS['vanilla'], COLORS['barrier'], COLORS['asian'], COLORS['lookback']]
    
    bars = ax.bar(products, prices, color=colors)
    ax.axhline(vanilla_call, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Option Price')
    ax.set_title('Call Option Price Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, price in zip(bars, prices):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f'{price:.2f}', ha='center', fontsize=10,
        )
    
    # -------------------------------------------------------------------------
    # Plot 3: Asian averaging effect
    # -------------------------------------------------------------------------
    ax = axes[1, 0]
    terminal = asian_results["terminal"]
    arith_avg = asian_results["arith_avg"]
    
    ax.hist(terminal, bins=50, alpha=0.5, density=True, color=COLORS['vanilla'], label='Terminal')
    ax.hist(arith_avg, bins=50, alpha=0.5, density=True, color=COLORS['asian'], label='Average')
    ax.axvline(K, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel('Price')
    ax.set_ylabel('Density')
    ax.set_title('Asian Option: Averaging Reduces Volatility')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 4: Lookback payoff distribution
    # -------------------------------------------------------------------------
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
    │  1. Barrier Options:                                                │
    │     - Knock-out: Cheaper than vanilla (barrier risk)                │
    │     - Knock-in: Remaining value after knock-out                     │
    │     - In-Out Parity: KI + KO = Vanilla                              │
    │                                                                      │
    │  2. Asian Options:                                                  │
    │     - Cheaper than vanilla (averaging reduces effective vol)        │
    │     - Arithmetic > Geometric (Jensen's inequality)                  │
    │     - Popular for hedging average exposures                         │
    │                                                                      │
    │  3. Lookback Options:                                               │
    │     - Most expensive (guaranteed best entry/exit)                   │
    │     - Floating: Buy at min, sell at max                             │
    │     - Fixed: Call on path maximum                                   │
    │                                                                      │
    │  4. Touch Options:                                                  │
    │     - Binary payoff (all or nothing)                                │
    │     - One-Touch + No-Touch = Discount factor                        │
    │     - Used for range bets                                           │
    │                                                                      │
    │  5. All path-dependent options require MC simulation                │
    │                                                                      │
    │  NEXT: See 03_portfolio_pricing.py for aggregating positions        │
    │                                                                      │
    └─────────────────────────────────────────────────────────────────────┘
    """
    logger.info(summary)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main(args: argparse.Namespace) -> None:
    """
    Main entry point for the example.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    """
    global ENABLE_PLOTTING
    ENABLE_PLOTTING = args.plot
    
    try:
        # Section 1: Market setup
        market, params = setup_market()
        
        # Section 2: Path simulation
        paths = simulate_paths(
            S0=params["spot"],
            r=params["r_dom"],
            q=params["r_for"],
            sigma=params["vol"],
            T=params["T"],
        )
        
        # Section 3: Vanilla benchmark
        vanilla_call, vanilla_put = run_vanilla_benchmark(paths, params)
        
        # Section 4: Barrier options
        barrier_results = run_barrier_analysis(paths, params, vanilla_call, vanilla_put)
        
        # Section 5: Asian options
        asian_results = run_asian_analysis(paths, params, vanilla_call)
        
        # Section 6: Lookback options
        lookback_results = run_lookback_analysis(paths, params, vanilla_call)
        
        # Section 7: Touch options
        touch_results = run_touch_analysis(paths, params)
        
        # Section 8: Summary comparison
        print_comparison(
            vanilla_call, barrier_results, asian_results,
            lookback_results, touch_results, params,
        )
        
        # Section 9: Visualization
        visualize_results(
            paths, vanilla_call, barrier_results,
            asian_results, lookback_results, params,
        )
        
        # Summary
        print_summary()
        
        logger.info("Example completed successfully!")
        
    except Exception as e:
        logger.exception(f"Example failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exotic Options Pricing Example",
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

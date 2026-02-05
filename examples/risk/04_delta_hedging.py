#!/usr/bin/env python3
"""
===============================================================================
Delta Hedging: Dynamic Risk Management
===============================================================================

This example demonstrates delta hedging - the practice of dynamically adjusting
a hedge position to neutralize directional risk from option positions.

Learning Objectives
-------------------
1. **Delta Hedging Theory**: Understand delta-neutral portfolios
2. **Discrete Hedging**: Impact of rebalancing frequency
3. **Transaction Costs**: Trade-off between hedge quality and costs
4. **Hedging P&L**: Gamma/theta relationship and hedging costs

Mathematical Framework
----------------------
Delta-Neutral Portfolio:
    Δ_portfolio + Δ_hedge = 0
    
For a short call option with delta Δ, we hold Δ shares of underlying.

Discrete Hedging P&L:
    Over interval dt with hedge position h = Δ(t-):
    
    Hedge P&L = h × ΔS
    Option P&L = -ΔC (we're short the option)
    
    Net P&L ≈ ½Γ(ΔS)² - Θ·dt - transaction costs

The hedge P&L matches first-order option movement, leaving:
    - Gamma P&L: Positive when large moves (long gamma)
    - Theta decay: Negative (option loses time value)
    - Transaction costs: From rebalancing

Production Context
------------------
At a hedge fund:
- Delta hedging is the most common risk management technique
- Rebalancing frequency depends on gamma exposure and costs
- Real-time delta monitoring for large portfolios
- Hedge slippage tracking and analysis

Prerequisites
-------------
- Understanding of Greeks (examples/risk/02_sensitivities_computation.py)
- Understanding of P&L attribution (examples/risk/03_pnl_attribution.py)

Run This Example
----------------
    cd /path/to/QuantStrata
    PYTHONPATH=. python examples/risk/04_delta_hedging.py

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

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
# BLACK-SCHOLES FUNCTIONS
# =============================================================================

def norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)


def bs_d1d2(S: float, K: float, T: float, r: float, q: float, sigma: float) -> Tuple[float, float]:
    """Compute d1 and d2."""
    if T <= 0:
        return 0.0, 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def bs_call_price(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Black-Scholes call price."""
    if T <= 0:
        return max(S - K, 0)
    d1, d2 = bs_d1d2(S, K, T, r, q, sigma)
    return S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)


def bs_delta(S: float, K: float, T: float, r: float, q: float, sigma: float, is_call: bool = True) -> float:
    """Black-Scholes delta."""
    if T <= 0:
        if is_call:
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    d1, _ = bs_d1d2(S, K, T, r, q, sigma)
    if is_call:
        return math.exp(-q * T) * norm_cdf(d1)
    else:
        return math.exp(-q * T) * (norm_cdf(d1) - 1)


def bs_gamma(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Black-Scholes gamma."""
    if T <= 0:
        return 0.0
    d1, _ = bs_d1d2(S, K, T, r, q, sigma)
    return math.exp(-q * T) * norm_pdf(d1) / (S * sigma * math.sqrt(T))


def bs_theta(S: float, K: float, T: float, r: float, q: float, sigma: float, is_call: bool = True) -> float:
    """Black-Scholes theta (per year)."""
    if T <= 0:
        return 0.0
    d1, d2 = bs_d1d2(S, K, T, r, q, sigma)
    sqrt_T = math.sqrt(T)
    
    term1 = -S * math.exp(-q * T) * norm_pdf(d1) * sigma / (2 * sqrt_T)
    
    if is_call:
        term2 = q * S * math.exp(-q * T) * norm_cdf(d1)
        term3 = -r * K * math.exp(-r * T) * norm_cdf(d2)
    else:
        term2 = -q * S * math.exp(-q * T) * norm_cdf(-d1)
        term3 = r * K * math.exp(-r * T) * norm_cdf(-d2)
    
    return term1 + term2 + term3


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class HedgingParams:
    """Hedging simulation parameters."""
    S0: float = 100.0           # Initial spot
    K: float = 100.0            # Strike (ATM)
    T: float = 0.25             # Time to expiry (3 months)
    r: float = 0.05             # Risk-free rate
    q: float = 0.0              # Dividend yield
    sigma: float = 0.20         # Volatility
    notional: float = 1_000_000 # Option notional
    
    # Hedging parameters
    rebalance_freq: int = 1     # Rebalance every N days
    transaction_cost: float = 0.001  # 10 bps per trade


@dataclass
class HedgingResult:
    """Result of hedging simulation."""
    times: np.ndarray
    spots: np.ndarray
    option_values: np.ndarray
    hedge_positions: np.ndarray
    cash: np.ndarray
    portfolio_values: np.ndarray
    deltas: np.ndarray
    gammas: np.ndarray
    total_trades: int
    total_cost: float
    final_pnl: float
    pnl_std: float


# =============================================================================
# SECTION 1: GBM Simulation
# =============================================================================

def simulate_gbm_path(
    S0: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    n_steps: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a single GBM path.
    
    Parameters
    ----------
    S0 : float
        Initial spot.
    r : float
        Risk-free rate.
    q : float
        Dividend yield.
    sigma : float
        Volatility.
    T : float
        Time horizon.
    n_steps : int
        Number of time steps.
    seed : int
        Random seed.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Time grid and spot path.
    """
    np.random.seed(seed)
    
    dt = T / n_steps
    times = np.linspace(0, T, n_steps + 1)
    
    # Generate returns
    drift = (r - q - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt) * np.random.randn(n_steps)
    
    # Build path
    log_returns = drift + diffusion
    spots = np.zeros(n_steps + 1)
    spots[0] = S0
    spots[1:] = S0 * np.exp(np.cumsum(log_returns))
    
    return times, spots


# =============================================================================
# SECTION 2: Delta Hedging Simulation
# =============================================================================

def run_delta_hedge(params: HedgingParams, seed: int = 42) -> HedgingResult:
    """
    Run delta hedging simulation.
    
    Parameters
    ----------
    params : HedgingParams
        Hedging parameters.
    seed : int
        Random seed.
    
    Returns
    -------
    HedgingResult
        Hedging simulation result.
    
    Algorithm
    ---------
    1. Start short 1 call option
    2. At each rebalance date:
       - Compute delta
       - Adjust hedge to be delta-neutral
       - Pay transaction costs
    3. At expiry:
       - Settle option payoff
       - Liquidate hedge
       - Compute final P&L
    """
    # Trading days to expiry
    n_days = int(params.T * 252)
    
    # Simulate path
    times, spots = simulate_gbm_path(
        params.S0, params.r, params.q, params.sigma, params.T, n_days, seed
    )
    
    # Initialize tracking arrays
    option_values = np.zeros(n_days + 1)
    hedge_positions = np.zeros(n_days + 1)
    cash = np.zeros(n_days + 1)
    deltas = np.zeros(n_days + 1)
    gammas = np.zeros(n_days + 1)
    
    # Initial setup
    time_to_expiry = params.T
    S = spots[0]
    
    # Option value (we are SHORT the option)
    option_values[0] = bs_call_price(S, params.K, time_to_expiry, params.r, params.q, params.sigma)
    
    # Initial delta and hedge
    delta = bs_delta(S, params.K, time_to_expiry, params.r, params.q, params.sigma)
    deltas[0] = delta
    gammas[0] = bs_gamma(S, params.K, time_to_expiry, params.r, params.q, params.sigma)
    
    # Initial hedge: buy delta shares to offset short option
    hedge_positions[0] = delta * params.notional
    
    # Cash: receive option premium, pay for initial hedge
    initial_cost = hedge_positions[0] * S * (1 + params.transaction_cost)
    cash[0] = option_values[0] * params.notional - initial_cost
    
    total_trades = 1
    total_cost = hedge_positions[0] * S * params.transaction_cost
    
    # Simulation loop
    dt = params.T / n_days
    
    for i in range(1, n_days + 1):
        S = spots[i]
        time_to_expiry = params.T - i * dt
        
        # Compute option value
        if time_to_expiry > 0:
            option_values[i] = bs_call_price(S, params.K, time_to_expiry, params.r, params.q, params.sigma)
            delta = bs_delta(S, params.K, time_to_expiry, params.r, params.q, params.sigma)
            gamma = bs_gamma(S, params.K, time_to_expiry, params.r, params.q, params.sigma)
        else:
            option_values[i] = max(S - params.K, 0)
            delta = 1.0 if S > params.K else 0.0
            gamma = 0.0
        
        deltas[i] = delta
        gammas[i] = gamma
        
        # Rebalance check
        if i % params.rebalance_freq == 0 and time_to_expiry > 0:
            # Target hedge position
            target_hedge = delta * params.notional
            trade_size = target_hedge - hedge_positions[i - 1]
            
            if abs(trade_size) > 1e-6:
                trade_cost = abs(trade_size) * S * params.transaction_cost
                cash[i] = cash[i - 1] - trade_size * S - trade_cost
                hedge_positions[i] = target_hedge
                total_trades += 1
                total_cost += trade_cost
            else:
                cash[i] = cash[i - 1]
                hedge_positions[i] = hedge_positions[i - 1]
        else:
            cash[i] = cash[i - 1]
            hedge_positions[i] = hedge_positions[i - 1]
    
    # Final settlement
    final_S = spots[-1]
    option_payoff = max(final_S - params.K, 0) * params.notional
    
    # We're short the option, so we pay the payoff
    final_cash = cash[-1] - option_payoff
    
    # Liquidate hedge
    final_cash += hedge_positions[-1] * final_S * (1 - params.transaction_cost)
    total_cost += hedge_positions[-1] * final_S * params.transaction_cost
    
    # Portfolio value at each step
    portfolio_values = cash - option_values * params.notional + hedge_positions * spots
    
    # Final P&L
    initial_premium = option_values[0] * params.notional
    final_pnl = final_cash
    
    # P&L volatility (daily changes)
    daily_pnl = np.diff(portfolio_values)
    pnl_std = np.std(daily_pnl)
    
    return HedgingResult(
        times=times,
        spots=spots,
        option_values=option_values,
        hedge_positions=hedge_positions,
        cash=cash,
        portfolio_values=portfolio_values,
        deltas=deltas,
        gammas=gammas,
        total_trades=total_trades,
        total_cost=total_cost,
        final_pnl=final_pnl,
        pnl_std=pnl_std,
    )


# =============================================================================
# SECTION 3: Display Results
# =============================================================================

def display_hedging_results(result: HedgingResult, params: HedgingParams) -> None:
    """Display hedging simulation results."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 3: Hedging Simulation Results")
    logger.info("=" * 70)
    
    logger.info("")
    logger.info("Simulation Parameters:")
    logger.info(f"  Initial Spot:     ${params.S0:.2f}")
    logger.info(f"  Strike:           ${params.K:.2f}")
    logger.info(f"  Time to Expiry:   {params.T:.2f} years ({int(params.T * 252)} days)")
    logger.info(f"  Volatility:       {params.sigma:.1%}")
    logger.info(f"  Notional:         ${params.notional:,.0f}")
    logger.info(f"  Rebalance Freq:   Every {params.rebalance_freq} day(s)")
    logger.info(f"  Transaction Cost: {params.transaction_cost:.2%}")
    
    logger.info("")
    logger.info("Simulation Results:")
    logger.info(f"  Final Spot:       ${result.spots[-1]:.2f}")
    logger.info(f"  Spot Return:      {(result.spots[-1] / result.spots[0] - 1) * 100:+.2f}%")
    logger.info(f"  Option Payoff:    ${max(result.spots[-1] - params.K, 0) * params.notional:,.0f}")
    
    logger.info("")
    logger.info("Hedging Statistics:")
    logger.info(f"  Total Trades:     {result.total_trades}")
    logger.info(f"  Total Cost:       ${result.total_cost:,.2f}")
    logger.info(f"  Final P&L:        ${result.final_pnl:,.2f}")
    logger.info(f"  Daily P&L Std:    ${result.pnl_std:,.2f}")
    
    # Theoretical hedge cost (theta)
    initial_theta = bs_theta(params.S0, params.K, params.T, params.r, params.q, params.sigma)
    expected_theta_cost = -initial_theta * params.T * params.notional
    
    logger.info("")
    logger.info("Theoretical Comparison:")
    logger.info(f"  Initial Theta:    ${initial_theta * params.notional / 252:.2f}/day")
    logger.info(f"  Expected Cost:    ${expected_theta_cost:,.2f} (approximate)")


# =============================================================================
# SECTION 4: Rebalancing Frequency Analysis
# =============================================================================

def analyze_rebalance_frequency(params: HedgingParams) -> List[Tuple[int, HedgingResult]]:
    """
    Analyze impact of rebalancing frequency on hedging quality.
    
    Returns
    -------
    List[Tuple[int, HedgingResult]]
        List of (frequency, result) tuples.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 4: Rebalancing Frequency Analysis")
    logger.info("=" * 70)
    
    frequencies = [1, 2, 5, 10, 21]  # Daily, 2-day, weekly, 2-week, monthly
    results = []
    
    logger.info("")
    logger.info("Impact of Rebalancing Frequency:")
    logger.info("-" * 70)
    logger.info(f"{'Frequency':<15} {'Trades':>10} {'Cost':>15} {'P&L Std':>15} {'Final P&L':>15}")
    logger.info("-" * 70)
    
    for freq in frequencies:
        params_copy = HedgingParams(
            S0=params.S0, K=params.K, T=params.T,
            r=params.r, q=params.q, sigma=params.sigma,
            notional=params.notional,
            rebalance_freq=freq,
            transaction_cost=params.transaction_cost,
        )
        
        result = run_delta_hedge(params_copy, seed=42)
        results.append((freq, result))
        
        freq_label = f"Every {freq} day(s)"
        logger.info(
            f"{freq_label:<15} {result.total_trades:>10} ${result.total_cost:>13,.0f} "
            f"${result.pnl_std:>13,.0f} ${result.final_pnl:>13,.0f}"
        )
    
    logger.info("-" * 70)
    
    logger.info("")
    logger.info("Key Insight:")
    logger.info("  More frequent rebalancing → Lower P&L volatility but higher costs")
    logger.info("  Optimal frequency balances hedge quality vs transaction costs")
    
    return results


# =============================================================================
# SECTION 5: Monte Carlo Analysis
# =============================================================================

def run_monte_carlo_hedging(params: HedgingParams, n_paths: int = 1000) -> np.ndarray:
    """
    Run Monte Carlo simulation of hedging outcomes.
    
    Returns
    -------
    np.ndarray
        Final P&L for each path.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 5: Monte Carlo Hedging Analysis")
    logger.info("=" * 70)
    
    logger.info(f"Running {n_paths} hedging simulations...")
    
    final_pnls = np.zeros(n_paths)
    
    for i in range(n_paths):
        result = run_delta_hedge(params, seed=i)
        final_pnls[i] = result.final_pnl
    
    logger.info("")
    logger.info("Monte Carlo Results:")
    logger.info(f"  Mean P&L:         ${np.mean(final_pnls):,.2f}")
    logger.info(f"  Std P&L:          ${np.std(final_pnls):,.2f}")
    logger.info(f"  5th Percentile:   ${np.percentile(final_pnls, 5):,.2f}")
    logger.info(f"  95th Percentile:  ${np.percentile(final_pnls, 95):,.2f}")
    logger.info(f"  % Profitable:     {(final_pnls > 0).mean() * 100:.1f}%")
    
    return final_pnls


# =============================================================================
# SECTION 6: Visualization
# =============================================================================

def visualize_hedging(
    result: HedgingResult,
    freq_results: List[Tuple[int, HedgingResult]],
    mc_pnls: np.ndarray,
    params: HedgingParams,
) -> None:
    """Create hedging visualizations."""
    if not MATPLOTLIB_AVAILABLE or not ENABLE_PLOTTING:
        logger.info("Skipping plots (matplotlib not available or disabled)")
        return
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 6: Visualization")
    logger.info("=" * 70)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # -------------------------------------------------------------------------
    # Plot 1: Spot path and hedge position
    # -------------------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()
    
    ax1.plot(result.times * 252, result.spots, 'b-', linewidth=1.5, label='Spot')
    ax1.axhline(params.K, color='gray', linestyle='--', alpha=0.7, label=f'Strike = {params.K}')
    
    ax1_twin.plot(result.times * 252, result.hedge_positions / 1000, 'g-', linewidth=1, alpha=0.7, label='Hedge (000s)')
    
    ax1.set_xlabel('Trading Days')
    ax1.set_ylabel('Spot Price ($)', color='blue')
    ax1_twin.set_ylabel('Hedge Position (000s)', color='green')
    ax1.set_title('Spot Price and Hedge Position')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    
    # -------------------------------------------------------------------------
    # Plot 2: Delta and Gamma over time
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]
    ax2_twin = ax2.twinx()
    
    ax2.plot(result.times * 252, result.deltas, 'b-', linewidth=1.5, label='Delta')
    ax2_twin.plot(result.times * 252, result.gammas * result.spots, 'r-', linewidth=1, alpha=0.7, label='Dollar Gamma')
    
    ax2.set_xlabel('Trading Days')
    ax2.set_ylabel('Delta', color='blue')
    ax2_twin.set_ylabel('Dollar Gamma', color='red')
    ax2.set_title('Greeks Evolution')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    
    # -------------------------------------------------------------------------
    # Plot 3: Rebalancing frequency comparison
    # -------------------------------------------------------------------------
    ax3 = axes[1, 0]
    
    frequencies = [r[0] for r in freq_results]
    costs = [r[1].total_cost for r in freq_results]
    pnl_stds = [r[1].pnl_std for r in freq_results]
    
    ax3_twin = ax3.twinx()
    
    bars = ax3.bar([f"Every {f}d" for f in frequencies], costs, color='#E94F37', alpha=0.7, label='Transaction Costs')
    ax3_twin.plot([f"Every {f}d" for f in frequencies], pnl_stds, 'b-o', linewidth=2, markersize=8, label='P&L Std')
    
    ax3.set_xlabel('Rebalancing Frequency')
    ax3.set_ylabel('Transaction Costs ($)', color='#E94F37')
    ax3_twin.set_ylabel('Daily P&L Std ($)', color='blue')
    ax3.set_title('Rebalancing Frequency Trade-off')
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
    
    # -------------------------------------------------------------------------
    # Plot 4: Monte Carlo P&L distribution
    # -------------------------------------------------------------------------
    ax4 = axes[1, 1]
    
    ax4.hist(mc_pnls, bins=50, density=True, color='#2E86AB', alpha=0.7, edgecolor='white')
    ax4.axvline(0, color='black', linestyle='-', linewidth=1)
    ax4.axvline(np.mean(mc_pnls), color='#E94F37', linestyle='--', linewidth=2, label=f'Mean: ${np.mean(mc_pnls):,.0f}')
    ax4.axvline(np.percentile(mc_pnls, 5), color='gray', linestyle=':', linewidth=1.5, label=f'5th pct: ${np.percentile(mc_pnls, 5):,.0f}')
    
    ax4.set_xlabel('Final P&L ($)')
    ax4.set_ylabel('Density')
    ax4.set_title('Monte Carlo Hedging P&L Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
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
    │  1. Delta Hedging:                                                  │
    │     - Hold Δ shares to offset option delta                          │
    │     - Rebalance as delta changes (spot/time moves)                  │
    │                                                                      │
    │  2. Hedging P&L:                                                    │
    │     - Gamma P&L: ½Γ(ΔS)² (positive for large moves)                 │
    │     - Theta cost: Θ·Δt (pay time decay)                             │
    │     - Transaction costs from rebalancing                            │
    │                                                                      │
    │  3. Rebalancing Trade-off:                                          │
    │     - More frequent → better hedge, higher costs                    │
    │     - Less frequent → worse hedge, lower costs                      │
    │     - Optimal depends on gamma exposure and cost structure          │
    │                                                                      │
    │  4. Production Considerations:                                      │
    │     - Real-time delta monitoring                                    │
    │     - Threshold-based rebalancing (not just time-based)             │
    │     - Hedge slippage analysis                                       │
    │                                                                      │
    │  NEXT: See examples/ml/ for RL-based hedging                        │
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
    logger.info("Delta Hedging Example")
    logger.info("=" * 70)
    
    try:
        # Setup parameters
        params = HedgingParams(
            S0=100.0,
            K=100.0,
            T=0.25,  # 3 months
            r=0.05,
            q=0.0,
            sigma=0.20,
            notional=1_000_000,
            rebalance_freq=1,  # Daily
            transaction_cost=0.001,  # 10 bps
        )
        
        # Section 2: Run single simulation
        logger.info("")
        logger.info("=" * 70)
        logger.info("SECTION 2: Single Path Hedging Simulation")
        logger.info("=" * 70)
        
        result = run_delta_hedge(params, seed=42)
        
        # Section 3: Display results
        display_hedging_results(result, params)
        
        # Section 4: Frequency analysis
        freq_results = analyze_rebalance_frequency(params)
        
        # Section 5: Monte Carlo
        mc_pnls = run_monte_carlo_hedging(params, n_paths=500)
        
        # Section 6: Visualization
        visualize_hedging(result, freq_results, mc_pnls, params)
        
        # Summary
        print_summary()
        
        logger.info("Example completed successfully!")
        
    except Exception as e:
        logger.exception(f"Example failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Delta Hedging Example",
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

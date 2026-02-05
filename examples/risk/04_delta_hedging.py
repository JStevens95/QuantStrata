#!/usr/bin/env python3
"""
===============================================================================
Delta Hedging: Dynamic Hedging Workflow
===============================================================================

This example demonstrates dynamic delta hedging - maintaining a delta-neutral
position through continuous rebalancing as the market moves.

Learning Objectives
-------------------
1. **Delta Hedging**: Understand the goal of eliminating directional risk
2. **Rebalancing**: See how hedging positions change with spot and time
3. **Transaction Costs**: Understand the trade-off between hedge accuracy and costs
4. **Hedge Effectiveness**: Measure P&L volatility reduction from hedging

Mathematical Framework
----------------------
Delta hedge position for a short option:
    Hedge Position = +Δ (buy delta shares to offset option delta)

For a call with Δ = 0.5:
    - Short 1 call (Δ = -0.5)
    - Buy 0.5 shares (Δ = +0.5)
    - Net delta = 0 (delta neutral)

As spot moves, delta changes (gamma effect):
    ΔΔ ≈ Γ × ΔS

Requiring rebalancing to maintain neutrality.

Hedging P&L decomposition:
    Hedge P&L = Gamma P&L - Transaction Costs - Theta Decay

Production Context
------------------
At a hedge fund:
- Options desks hedge to isolate vol/gamma exposure
- Rebalancing frequency is a key decision (cost vs accuracy)
- Perfect hedging is impossible (discrete rebalancing, transaction costs)
- Gamma P&L should offset theta over time (hedging earns vol carry)

Prerequisites
-------------
- Greeks computation (examples/risk/02_sensitivities_computation.py)
- P&L attribution (examples/risk/03_pnl_attribution.py)

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

# -----------------------------------------------------------------------------
# QuantStrata imports
# -----------------------------------------------------------------------------
from src.marketdata.core.ids import MarketId
from src.marketdata.core.interfaces import Quote
from src.marketdata.core.market import Market
from src.marketdata.curves.term_structure import FlatZeroRateCurve
from src.marketdata.surfaces.vol_surface import FlatVolSurface

from src.instruments.fx.options.vanilla import EuropeanFxVanillaOption
from src.pricers.fx.european_bsm import FxEuropeanVanillaBsmPricer


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

EURUSD_SPOT = MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD")
USD_CURVE = MarketId(asset_class="IR", mkt_type="CURVE", name="USD_OIS")
EUR_CURVE = MarketId(asset_class="IR", mkt_type="CURVE", name="EUR_OIS")
EURUSD_VOL = MarketId(asset_class="FX", mkt_type="VOL", name="EURUSD")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class HedgingConfig:
    """Configuration for hedging simulation."""
    # Option parameters
    spot: float = 1.0850
    strike: float = 1.0850  # ATM
    expiry: float = 0.25  # 3 months
    vol: float = 0.10
    r_dom: float = 0.05
    r_for: float = 0.04
    notional: float = 10_000_000
    
    # Simulation parameters
    n_steps: int = 63  # Daily steps (~3 months)
    n_paths: int = 1000  # MC paths
    
    # Transaction costs
    proportional_cost: float = 0.0001  # 1 bp
    
    # Rebalancing
    rebalance_frequency: int = 1  # Every n steps


@dataclass
class HedgeState:
    """State of the hedge at a point in time."""
    time: float
    spot: float
    option_value: float
    delta: float
    gamma: float
    hedge_position: float
    cash: float
    portfolio_value: float
    cumulative_costs: float


@dataclass
class HedgingResult:
    """Result of a hedging simulation."""
    states: List[HedgeState] = field(default_factory=list)
    final_pnl: float = 0.0
    pnl_std: float = 0.0
    total_costs: float = 0.0
    n_rebalances: int = 0


# =============================================================================
# SECTION 1: Black-Scholes Utilities
# =============================================================================

def bs_d1d2(S: float, K: float, T: float, sigma: float, r: float, q: float) -> Tuple[float, float]:
    """Compute d1 and d2."""
    if T <= 0:
        return 0.0, 0.0
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def bs_call_price(S: float, K: float, T: float, sigma: float, r: float, q: float) -> float:
    """Black-Scholes call price."""
    if T <= 0:
        return max(S - K, 0)
    d1, d2 = bs_d1d2(S, K, T, sigma, r, q)
    from scipy.stats import norm
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_call_delta(S: float, K: float, T: float, sigma: float, r: float, q: float) -> float:
    """Black-Scholes call delta."""
    if T <= 0:
        return 1.0 if S > K else 0.0
    d1, _ = bs_d1d2(S, K, T, sigma, r, q)
    from scipy.stats import norm
    return np.exp(-q * T) * norm.cdf(d1)


def bs_call_gamma(S: float, K: float, T: float, sigma: float, r: float, q: float) -> float:
    """Black-Scholes call gamma."""
    if T <= 0:
        return 0.0
    d1, _ = bs_d1d2(S, K, T, sigma, r, q)
    from scipy.stats import norm
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


# =============================================================================
# SECTION 2: GBM Simulation
# =============================================================================

def simulate_gbm_paths(
    S0: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate GBM paths.
    
    Returns
    -------
    np.ndarray
        Shape (n_steps + 1, n_paths) with spot paths.
    """
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


# =============================================================================
# SECTION 3: Hedging Simulation
# =============================================================================

def run_single_hedge_path(
    path: np.ndarray,
    config: HedgingConfig,
) -> List[HedgeState]:
    """
    Run delta hedging along a single price path.
    
    Parameters
    ----------
    path : np.ndarray
        Spot price path.
    config : HedgingConfig
        Hedging configuration.
    
    Returns
    -------
    List[HedgeState]
        Hedge states at each time step.
    """
    dt = config.expiry / config.n_steps
    states: List[HedgeState] = []
    
    # Initial state
    S = path[0]
    T = config.expiry
    
    option_value = bs_call_price(S, config.strike, T, config.vol, config.r_dom, config.r_for) * config.notional
    delta = bs_call_delta(S, config.strike, T, config.vol, config.r_dom, config.r_for) * config.notional
    gamma = bs_call_gamma(S, config.strike, T, config.vol, config.r_dom, config.r_for) * config.notional
    
    # We sold the option, so we receive the premium
    # To delta hedge, we buy delta shares
    hedge_position = delta  # Shares to buy (in notional terms: delta * S)
    cash = option_value - hedge_position * S  # Received premium, paid for hedge
    cumulative_costs = 0.0
    
    states.append(HedgeState(
        time=0.0,
        spot=S,
        option_value=option_value,
        delta=delta,
        gamma=gamma,
        hedge_position=hedge_position,
        cash=cash,
        portfolio_value=cash + hedge_position * S - option_value,
        cumulative_costs=cumulative_costs,
    ))
    
    # Simulate through time
    for step in range(1, config.n_steps + 1):
        S = path[step]
        T = config.expiry - step * dt
        T = max(T, 0)
        
        # New option value and Greeks
        new_option_value = bs_call_price(S, config.strike, T, config.vol, config.r_dom, config.r_for) * config.notional
        new_delta = bs_call_delta(S, config.strike, T, config.vol, config.r_dom, config.r_for) * config.notional
        new_gamma = bs_call_gamma(S, config.strike, T, config.vol, config.r_dom, config.r_for) * config.notional
        
        # Update cash with interest and hedge P&L
        cash = cash * np.exp(config.r_dom * dt)  # Cash earns interest
        cash += hedge_position * S * (np.exp((config.r_dom - config.r_for) * dt) - 1)  # Carry
        
        # Rebalance hedge
        if step % config.rebalance_frequency == 0 or step == config.n_steps:
            trade_size = new_delta - hedge_position
            trade_cost = abs(trade_size * S) * config.proportional_cost
            
            cash -= trade_size * S + trade_cost
            hedge_position = new_delta
            cumulative_costs += trade_cost
        
        # Portfolio value: cash + hedge value - option liability
        portfolio_value = cash + hedge_position * S - new_option_value
        
        states.append(HedgeState(
            time=step * dt,
            spot=S,
            option_value=new_option_value,
            delta=new_delta,
            gamma=new_gamma,
            hedge_position=hedge_position,
            cash=cash,
            portfolio_value=portfolio_value,
            cumulative_costs=cumulative_costs,
        ))
        
        option_value = new_option_value
    
    return states


def run_hedging_simulation(config: HedgingConfig) -> Tuple[HedgingResult, np.ndarray]:
    """
    Run full hedging simulation across multiple paths.
    
    Returns
    -------
    Tuple[HedgingResult, np.ndarray]
        Hedging result and final P&Ls.
    """
    logger.info("=" * 70)
    logger.info("SECTION 1: Hedging Simulation Setup")
    logger.info("=" * 70)
    
    logger.info("")
    logger.info("Option Parameters:")
    logger.info(f"  Spot:      {config.spot}")
    logger.info(f"  Strike:    {config.strike}")
    logger.info(f"  Expiry:    {config.expiry:.2f} years ({int(config.expiry*252)} days)")
    logger.info(f"  Vol:       {config.vol:.1%}")
    logger.info(f"  Notional:  ${config.notional:,.0f}")
    
    logger.info("")
    logger.info("Simulation Parameters:")
    logger.info(f"  Steps:     {config.n_steps}")
    logger.info(f"  Paths:     {config.n_paths}")
    logger.info(f"  Rebalance: Every {config.rebalance_frequency} step(s)")
    logger.info(f"  Cost:      {config.proportional_cost*10000:.1f} bps")
    
    # Simulate paths
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 2: Running Simulation")
    logger.info("=" * 70)
    
    paths = simulate_gbm_paths(
        S0=config.spot,
        r=config.r_dom,
        q=config.r_for,
        sigma=config.vol,
        T=config.expiry,
        n_steps=config.n_steps,
        n_paths=config.n_paths,
    )
    
    logger.info("")
    logger.info(f"Simulated {config.n_paths} GBM paths")
    
    # Run hedging on each path
    final_pnls = []
    total_costs = []
    
    for i in range(config.n_paths):
        states = run_single_hedge_path(paths[:, i], config)
        final_pnls.append(states[-1].portfolio_value)
        total_costs.append(states[-1].cumulative_costs)
    
    final_pnls = np.array(final_pnls)
    total_costs = np.array(total_costs)
    
    # Store one sample path for visualization
    sample_states = run_single_hedge_path(paths[:, 0], config)
    
    result = HedgingResult(
        states=sample_states,
        final_pnl=np.mean(final_pnls),
        pnl_std=np.std(final_pnls),
        total_costs=np.mean(total_costs),
        n_rebalances=config.n_steps // config.rebalance_frequency,
    )
    
    logger.info("")
    logger.info("Simulation complete")
    
    return result, final_pnls


# =============================================================================
# SECTION 4: Compare Rebalancing Frequencies
# =============================================================================

def compare_rebalancing_frequencies(config: HedgingConfig) -> List[Tuple[int, float, float, float]]:
    """
    Compare different rebalancing frequencies.
    
    Returns
    -------
    List[Tuple[int, float, float, float]]
        List of (frequency, mean_pnl, std_pnl, mean_costs).
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 3: Rebalancing Frequency Analysis")
    logger.info("=" * 70)
    
    frequencies = [1, 5, 10, 21, 63]  # Daily, weekly, bi-weekly, monthly, quarterly
    results = []
    
    logger.info("")
    logger.info(f"{'Frequency':<15} {'Mean P&L':>12} {'Std P&L':>12} {'Mean Costs':>12} {'Sharpe':>10}")
    logger.info("-" * 61)
    
    for freq in frequencies:
        cfg = HedgingConfig(
            spot=config.spot,
            strike=config.strike,
            expiry=config.expiry,
            vol=config.vol,
            r_dom=config.r_dom,
            r_for=config.r_for,
            notional=config.notional,
            n_steps=config.n_steps,
            n_paths=500,  # Fewer paths for speed
            proportional_cost=config.proportional_cost,
            rebalance_frequency=freq,
        )
        
        result, pnls = run_hedging_simulation.__wrapped__(cfg) if hasattr(run_hedging_simulation, '__wrapped__') else run_hedging_simulation_silent(cfg)
        
        sharpe = result.final_pnl / result.pnl_std if result.pnl_std > 0 else 0
        
        freq_label = {1: 'Daily', 5: 'Weekly', 10: 'Bi-weekly', 21: 'Monthly', 63: 'Quarterly'}.get(freq, f'{freq} steps')
        
        logger.info(
            f"{freq_label:<15} ${result.final_pnl:>10,.0f} ${result.pnl_std:>10,.0f} "
            f"${result.total_costs:>10,.0f} {sharpe:>9.2f}"
        )
        
        results.append((freq, result.final_pnl, result.pnl_std, result.total_costs))
    
    return results


def run_hedging_simulation_silent(config: HedgingConfig) -> Tuple[HedgingResult, np.ndarray]:
    """Run simulation without logging."""
    paths = simulate_gbm_paths(
        S0=config.spot,
        r=config.r_dom,
        q=config.r_for,
        sigma=config.vol,
        T=config.expiry,
        n_steps=config.n_steps,
        n_paths=config.n_paths,
    )
    
    final_pnls = []
    total_costs = []
    
    for i in range(config.n_paths):
        states = run_single_hedge_path(paths[:, i], config)
        final_pnls.append(states[-1].portfolio_value)
        total_costs.append(states[-1].cumulative_costs)
    
    final_pnls = np.array(final_pnls)
    total_costs = np.array(total_costs)
    
    sample_states = run_single_hedge_path(paths[:, 0], config)
    
    result = HedgingResult(
        states=sample_states,
        final_pnl=np.mean(final_pnls),
        pnl_std=np.std(final_pnls),
        total_costs=np.mean(total_costs),
        n_rebalances=config.n_steps // config.rebalance_frequency,
    )
    
    return result, final_pnls


# =============================================================================
# SECTION 5: Visualization
# =============================================================================

def visualize_hedging(
    result: HedgingResult,
    final_pnls: np.ndarray,
    freq_comparison: List[Tuple[int, float, float, float]],
) -> None:
    """Create hedging visualizations."""
    if not MATPLOTLIB_AVAILABLE or not ENABLE_PLOTTING:
        logger.info("Skipping plots (matplotlib not available or disabled)")
        return
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 4: Visualization")
    logger.info("=" * 70)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    states = result.states
    times = [s.time for s in states]
    
    # -------------------------------------------------------------------------
    # Plot 1: Spot and Delta path
    # -------------------------------------------------------------------------
    ax = axes[0, 0]
    spots = [s.spot for s in states]
    deltas = [s.delta / 10_000_000 for s in states]  # Normalize
    
    ax2 = ax.twinx()
    ax.plot(times, spots, 'b-', linewidth=2, label='Spot')
    ax2.plot(times, deltas, 'g--', linewidth=2, label='Delta')
    
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Spot', color='blue')
    ax2.set_ylabel('Delta (normalized)', color='green')
    ax.set_title('Spot Path and Delta Evolution')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 2: Portfolio value and hedge position
    # -------------------------------------------------------------------------
    ax = axes[0, 1]
    portfolio_values = [s.portfolio_value for s in states]
    
    ax.plot(times, [v / 1000 for v in portfolio_values], 'b-', linewidth=2)
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.fill_between(
        times, [v / 1000 for v in portfolio_values], 0,
        where=[v > 0 for v in portfolio_values],
        alpha=0.3, color='green',
    )
    ax.fill_between(
        times, [v / 1000 for v in portfolio_values], 0,
        where=[v <= 0 for v in portfolio_values],
        alpha=0.3, color='red',
    )
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Portfolio Value ($000s)')
    ax.set_title('Hedged Portfolio Value Over Time')
    ax.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 3: Final P&L distribution
    # -------------------------------------------------------------------------
    ax = axes[1, 0]
    ax.hist(final_pnls / 1000, bins=50, density=True, alpha=0.7, color='#2E86AB')
    ax.axvline(np.mean(final_pnls) / 1000, color='red', linestyle='--', linewidth=2, label=f'Mean: ${np.mean(final_pnls):,.0f}')
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Final P&L ($000s)')
    ax.set_ylabel('Density')
    ax.set_title(f'Final P&L Distribution (Std: ${np.std(final_pnls):,.0f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 4: Rebalancing frequency comparison
    # -------------------------------------------------------------------------
    ax = axes[1, 1]
    freqs = [r[0] for r in freq_comparison]
    stds = [r[2] for r in freq_comparison]
    costs = [r[3] for r in freq_comparison]
    
    x = np.arange(len(freqs))
    width = 0.35
    
    ax2 = ax.twinx()
    bars1 = ax.bar(x - width/2, [s/1000 for s in stds], width, label='P&L Std', color='#2E86AB')
    bars2 = ax2.bar(x + width/2, [c/1000 for c in costs], width, label='Total Costs', color='#E94F37')
    
    ax.set_xlabel('Rebalancing Frequency')
    ax.set_ylabel('P&L Std ($000s)', color='#2E86AB')
    ax2.set_ylabel('Costs ($000s)', color='#E94F37')
    ax.set_title('Trade-off: Hedge Accuracy vs Transaction Costs')
    ax.set_xticks(x)
    ax.set_xticklabels(['Daily', 'Weekly', 'Bi-weekly', 'Monthly', 'Quarterly'])
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
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
    │  1. Delta Hedging Goal:                                             │
    │     - Eliminate directional (delta) risk                            │
    │     - Isolate gamma/vol exposure                                    │
    │                                                                      │
    │  2. Rebalancing Trade-off:                                          │
    │     - More frequent: Lower P&L variance, higher costs               │
    │     - Less frequent: Higher P&L variance, lower costs               │
    │     - Optimal frequency depends on vol, gamma, and costs            │
    │                                                                      │
    │  3. Hedging P&L Sources:                                            │
    │     - Gamma P&L: ½Γ × (ΔS)² (positive for long gamma)              │
    │     - Theta decay: Θ × Δt (negative for long options)               │
    │     - Transaction costs: Proportional to rebalancing                │
    │                                                                      │
    │  4. Production Considerations:                                      │
    │     - Optimal hedge ratio may differ from delta                     │
    │     - Consider hedging bands (don't rebalance for small moves)      │
    │     - RL can learn better hedging strategies than delta hedge       │
    │                                                                      │
    │  NEXT: See q_learning/01_hedging_agent.py for RL hedging            │
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
        # Configuration
        config = HedgingConfig()
        
        # Run main simulation
        result, final_pnls = run_hedging_simulation(config)
        
        # Display results
        logger.info("")
        logger.info("Hedging Results:")
        logger.info("-" * 50)
        logger.info(f"  Mean Final P&L:  ${result.final_pnl:>12,.2f}")
        logger.info(f"  P&L Std Dev:     ${result.pnl_std:>12,.2f}")
        logger.info(f"  Total Costs:     ${result.total_costs:>12,.2f}")
        logger.info(f"  # Rebalances:    {result.n_rebalances:>12}")
        
        # Compare rebalancing frequencies
        freq_comparison = compare_rebalancing_frequencies(config)
        
        # Visualization
        visualize_hedging(result, final_pnls, freq_comparison)
        
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

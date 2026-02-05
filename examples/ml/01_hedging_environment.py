#!/usr/bin/env python3
"""
===============================================================================
RL Hedging Environment: Gymnasium Interface
===============================================================================

This example demonstrates the RL hedging environment - a Gymnasium-compatible
interface for training reinforcement learning agents to hedge options.

Learning Objectives
-------------------
1. **RL Environment Design**: Understand state, action, reward structure
2. **Hedging as RL Problem**: Framing option hedging for RL agents
3. **Benchmark Comparison**: Compare RL agent vs delta hedging
4. **Environment Configuration**: Tune environment for different use cases

Mathematical Framework
----------------------
State Space (observation):
    s_t = [S_t/S_0 - 1, τ/T, Δ_t, Γ_t·S_t, h_t, PnL_t/V_0]
    
    where:
    - S_t/S_0 - 1: Normalized spot return
    - τ/T: Normalized time to expiry
    - Δ_t: Option delta
    - Γ_t·S_t: Dollar gamma
    - h_t: Current hedge position
    - PnL_t/V_0: Normalized cumulative P&L

Action Space:
    a_t ∈ [-2, 2]: Hedge ratio (multiple of delta)
    
    Target position = a_t × |Δ_t| × Notional

Reward Function (risk-adjusted):
    r_t = ΔPnL_t/S_0 - λ × σ(PnL)/S_0
    
    where λ is risk aversion parameter

Production Context
------------------
At a hedge fund:
- RL hedging is an active research area
- Can potentially learn to hedge better than delta
- Especially useful for path-dependent options
- Model-free approach avoids model specification error

Prerequisites
-------------
- Understanding of delta hedging (examples/risk/04_delta_hedging.py)
- Basic RL concepts (states, actions, rewards)

Run This Example
----------------
    cd /path/to/QuantStrata
    PYTHONPATH=. python examples/ml/01_hedging_environment.py

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
from typing import Dict, List, Tuple, Any

import numpy as np

# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# -----------------------------------------------------------------------------
# QuantStrata imports
# -----------------------------------------------------------------------------
from src.q_learning.environments.hedging import HedgingEnvironment, HedgingEnvConfig


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
# SECTION 1: Environment Setup
# =============================================================================

def create_hedging_environment() -> Tuple[HedgingEnvironment, HedgingEnvConfig]:
    """
    Create and configure the hedging environment.
    
    Returns
    -------
    Tuple[HedgingEnvironment, HedgingEnvConfig]
        Environment and its configuration.
    
    Configuration Details
    ---------------------
    - ATM call option (S0 = K = 100)
    - 3-month expiry
    - 50 hedging intervals (about weekly)
    - 10 bps transaction costs
    - Risk-adjusted reward
    """
    logger.info("=" * 70)
    logger.info("SECTION 1: Environment Setup")
    logger.info("=" * 70)
    
    config = HedgingEnvConfig(
        # Option parameters
        spot=100.0,
        strike=100.0,
        maturity=0.25,  # 3 months
        volatility=0.20,
        risk_free_rate=0.05,
        dividend_yield=0.0,
        option_type="call",
        
        # Simulation parameters
        n_steps=50,  # ~weekly rebalancing
        n_paths=1,
        
        # Transaction costs
        proportional_cost=0.001,  # 10 bps
        fixed_cost=0.0,
        
        # State features
        include_delta=True,
        include_gamma=True,
        include_vega=False,
        include_time=True,
        include_position=True,
        include_pnl=True,
        
        # Action space
        action_type="continuous",
        max_hedge_ratio=2.0,
        
        # Reward
        reward_type="risk_adjusted",
        risk_aversion=0.1,
        
        # Real-world drift (can differ from risk-free)
        drift=0.0,
    )
    
    env = HedgingEnvironment(config=config, seed=42)
    
    logger.info("")
    logger.info("Environment Configuration:")
    logger.info(f"  Spot:           ${config.spot:.2f}")
    logger.info(f"  Strike:         ${config.strike:.2f}")
    logger.info(f"  Maturity:       {config.maturity:.2f} years")
    logger.info(f"  Volatility:     {config.volatility:.1%}")
    logger.info(f"  Hedging steps:  {config.n_steps}")
    logger.info(f"  Trans. cost:    {config.proportional_cost:.2%}")
    
    logger.info("")
    logger.info("State Space:")
    logger.info(f"  Dimension:      {env.observation_space_dim}")
    logger.info(f"  Features:       [spot_return, time, delta, gamma, position, pnl]")
    
    logger.info("")
    logger.info("Action Space:")
    logger.info(f"  Type:           {config.action_type}")
    logger.info(f"  Range:          [-{config.max_hedge_ratio}, {config.max_hedge_ratio}]")
    
    return env, config


# =============================================================================
# SECTION 2: Basic Environment Interaction
# =============================================================================

def demonstrate_environment_api(env: HedgingEnvironment) -> None:
    """
    Demonstrate basic environment API (reset, step).
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 2: Environment API Demonstration")
    logger.info("=" * 70)
    
    # Reset environment
    state, info = env.reset(seed=42)
    
    logger.info("")
    logger.info("After reset:")
    logger.info(f"  State shape:    {state.shape}")
    logger.info(f"  State values:   {np.array2string(state, precision=4)}")
    logger.info(f"  Info keys:      {list(info.keys())}")
    logger.info(f"  Initial spot:   ${info['spot']:.2f}")
    logger.info(f"  Initial delta:  {info['delta']:.4f}")
    logger.info(f"  Option value:   ${info['option_value']:.4f}")
    
    # Take a step with delta hedge action
    action = 1.0  # Hedge ratio = 1.0 (perfect delta hedge)
    next_state, reward, terminated, truncated, info = env.step(action)
    
    logger.info("")
    logger.info("After one step (action=1.0, delta hedge):")
    logger.info(f"  New spot:       ${info['spot']:.2f}")
    logger.info(f"  New delta:      {info['delta']:.4f}")
    logger.info(f"  Position:       {info['position']:.4f}")
    logger.info(f"  Reward:         {reward:.6f}")
    logger.info(f"  PnL so far:     ${info['pnl']:.4f}")
    logger.info(f"  Trade cost:     ${info.get('trade_cost', 0):.4f}")
    logger.info(f"  Terminated:     {terminated}")
    logger.info(f"  Truncated:      {truncated}")


# =============================================================================
# SECTION 3: Delta Hedging Benchmark
# =============================================================================

def run_delta_hedge_episode(env: HedgingEnvironment, seed: int = 42) -> Dict[str, Any]:
    """
    Run an episode using pure delta hedging strategy.
    
    Parameters
    ----------
    env : HedgingEnvironment
        The hedging environment.
    seed : int
        Random seed.
    
    Returns
    -------
    Dict[str, Any]
        Episode statistics.
    """
    state, info = env.reset(seed=seed)
    
    total_reward = 0.0
    total_cost = 0.0
    spots = [info['spot']]
    positions = [0.0]
    pnls = [0.0]
    
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        # Delta hedge: action = 1.0 means hedge at exactly delta
        action = env.get_delta_hedge_action()
        
        state, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        total_cost += info.get('trade_cost', 0)
        spots.append(info['spot'])
        positions.append(info['position'])
        pnls.append(info['pnl'])
    
    return {
        'final_pnl': info['pnl'],
        'total_reward': total_reward,
        'total_cost': info['cumulative_cost'],
        'spots': np.array(spots),
        'positions': np.array(positions),
        'pnls': np.array(pnls),
    }


def run_benchmark_comparison(env: HedgingEnvironment, n_episodes: int = 500) -> Dict[str, np.ndarray]:
    """
    Run multiple episodes with delta hedging and alternative strategies.
    
    Returns
    -------
    Dict[str, np.ndarray]
        Results for each strategy.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 3: Strategy Benchmark")
    logger.info("=" * 70)
    
    logger.info(f"Running {n_episodes} episodes for each strategy...")
    
    # Strategy 1: Delta hedge (action = 1.0)
    delta_pnls = []
    for i in range(n_episodes):
        result = run_delta_hedge_episode(env, seed=i)
        delta_pnls.append(result['final_pnl'])
    delta_pnls = np.array(delta_pnls)
    
    # Strategy 2: No hedge (action = 0.0)
    no_hedge_pnls = []
    for i in range(n_episodes):
        state, info = env.reset(seed=i)
        terminated = truncated = False
        while not (terminated or truncated):
            state, reward, terminated, truncated, info = env.step(0.0)
        no_hedge_pnls.append(info['pnl'])
    no_hedge_pnls = np.array(no_hedge_pnls)
    
    # Strategy 3: Over-hedge (action = 1.5)
    over_hedge_pnls = []
    for i in range(n_episodes):
        state, info = env.reset(seed=i)
        terminated = truncated = False
        while not (terminated or truncated):
            state, reward, terminated, truncated, info = env.step(1.5)
        over_hedge_pnls.append(info['pnl'])
    over_hedge_pnls = np.array(over_hedge_pnls)
    
    # Display results
    logger.info("")
    logger.info("Strategy Comparison:")
    logger.info("-" * 70)
    logger.info(f"{'Strategy':<20} {'Mean P&L':>15} {'Std P&L':>15} {'Sharpe':>15}")
    logger.info("-" * 70)
    
    for name, pnls in [
        ('Delta Hedge (1.0)', delta_pnls),
        ('No Hedge (0.0)', no_hedge_pnls),
        ('Over-Hedge (1.5)', over_hedge_pnls),
    ]:
        mean = np.mean(pnls)
        std = np.std(pnls)
        sharpe = mean / std if std > 0 else 0
        logger.info(f"{name:<20} ${mean:>14,.2f} ${std:>14,.2f} {sharpe:>14.3f}")
    
    logger.info("-" * 70)
    
    return {
        'delta_hedge': delta_pnls,
        'no_hedge': no_hedge_pnls,
        'over_hedge': over_hedge_pnls,
    }


# =============================================================================
# SECTION 4: Single Episode Visualization
# =============================================================================

def run_detailed_episode(env: HedgingEnvironment) -> Dict[str, Any]:
    """
    Run a single episode with detailed tracking.
    
    Returns
    -------
    Dict[str, Any]
        Detailed episode data.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 4: Detailed Episode Analysis")
    logger.info("=" * 70)
    
    state, info = env.reset(seed=42)
    
    # Track everything
    steps = [0]
    spots = [info['spot']]
    deltas = [info['delta']]
    positions = [info['position']]
    pnls = [info['pnl']]
    rewards = []
    actions = []
    
    terminated = truncated = False
    step = 0
    
    while not (terminated or truncated):
        step += 1
        
        # Use delta hedge action
        action = 1.0
        state, reward, terminated, truncated, info = env.step(action)
        
        steps.append(step)
        spots.append(info['spot'])
        deltas.append(info['delta'])
        positions.append(info['position'])
        pnls.append(info['pnl'])
        rewards.append(reward)
        actions.append(action)
    
    logger.info("")
    logger.info("Episode Summary:")
    logger.info(f"  Steps:          {step}")
    logger.info(f"  Final spot:     ${spots[-1]:.2f}")
    logger.info(f"  Final P&L:      ${pnls[-1]:.4f}")
    logger.info(f"  Total reward:   {sum(rewards):.6f}")
    logger.info(f"  Cumulative cost: ${info['cumulative_cost']:.4f}")
    
    return {
        'steps': np.array(steps),
        'spots': np.array(spots),
        'deltas': np.array(deltas),
        'positions': np.array(positions),
        'pnls': np.array(pnls),
        'rewards': np.array(rewards),
    }


# =============================================================================
# SECTION 5: Visualization
# =============================================================================

def visualize_results(
    episode_data: Dict[str, Any],
    benchmark_results: Dict[str, np.ndarray],
    config: HedgingEnvConfig,
) -> None:
    """Create visualizations."""
    if not MATPLOTLIB_AVAILABLE or not ENABLE_PLOTTING:
        logger.info("Skipping plots (matplotlib not available or disabled)")
        return
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 5: Visualization")
    logger.info("=" * 70)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # -------------------------------------------------------------------------
    # Plot 1: Spot and delta evolution
    # -------------------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()
    
    ax1.plot(episode_data['steps'], episode_data['spots'], 'b-', linewidth=1.5, label='Spot')
    ax1.axhline(config.strike, color='gray', linestyle='--', alpha=0.7, label=f'Strike={config.strike}')
    
    ax1_twin.plot(episode_data['steps'], episode_data['deltas'], 'g-', linewidth=1, alpha=0.7, label='Delta')
    
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Spot ($)', color='blue')
    ax1_twin.set_ylabel('Delta', color='green')
    ax1.set_title('Spot Price and Delta Evolution')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    
    # -------------------------------------------------------------------------
    # Plot 2: P&L evolution
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]
    
    ax2.plot(episode_data['steps'], episode_data['pnls'], 'b-', linewidth=1.5)
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax2.fill_between(
        episode_data['steps'], episode_data['pnls'], 0,
        where=(episode_data['pnls'] > 0), alpha=0.3, color='green',
    )
    ax2.fill_between(
        episode_data['steps'], episode_data['pnls'], 0,
        where=(episode_data['pnls'] <= 0), alpha=0.3, color='red',
    )
    
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Cumulative P&L ($)')
    ax2.set_title('P&L Evolution')
    ax2.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 3: Strategy comparison histogram
    # -------------------------------------------------------------------------
    ax3 = axes[1, 0]
    
    ax3.hist(benchmark_results['delta_hedge'], bins=30, alpha=0.7, label='Delta Hedge', color='#2E86AB')
    ax3.hist(benchmark_results['no_hedge'], bins=30, alpha=0.5, label='No Hedge', color='#E94F37')
    ax3.hist(benchmark_results['over_hedge'], bins=30, alpha=0.5, label='Over Hedge', color='#10B981')
    
    ax3.axvline(0, color='black', linestyle='-', linewidth=1)
    
    ax3.set_xlabel('Final P&L ($)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('P&L Distribution by Strategy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 4: Strategy risk-return
    # -------------------------------------------------------------------------
    ax4 = axes[1, 1]
    
    strategies = ['Delta\nHedge', 'No\nHedge', 'Over\nHedge']
    means = [
        np.mean(benchmark_results['delta_hedge']),
        np.mean(benchmark_results['no_hedge']),
        np.mean(benchmark_results['over_hedge']),
    ]
    stds = [
        np.std(benchmark_results['delta_hedge']),
        np.std(benchmark_results['no_hedge']),
        np.std(benchmark_results['over_hedge']),
    ]
    
    colors = ['#2E86AB', '#E94F37', '#10B981']
    
    ax4.scatter(stds, means, c=colors, s=200, zorder=3)
    for i, (x, y, name) in enumerate(zip(stds, means, strategies)):
        ax4.annotate(name, (x, y), textcoords="offset points", xytext=(10, 5), fontsize=10)
    
    ax4.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    ax4.set_xlabel('P&L Std Deviation ($)')
    ax4.set_ylabel('Mean P&L ($)')
    ax4.set_title('Risk-Return Trade-off')
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
    │  1. Environment Structure:                                          │
    │     - State: [spot_return, time, delta, gamma, position, pnl]       │
    │     - Action: hedge ratio ∈ [-2, 2]                                 │
    │     - Reward: risk-adjusted P&L change                              │
    │                                                                      │
    │  2. Episode Flow:                                                   │
    │     - reset() → initial state, info                                 │
    │     - step(action) → next_state, reward, terminated, truncated      │
    │     - Runs until option expiry                                      │
    │                                                                      │
    │  3. Benchmark Strategies:                                           │
    │     - Delta hedge (action=1.0): balanced risk/return                │
    │     - No hedge (action=0.0): high variance                          │
    │     - Over-hedge (action=1.5): can increase costs                   │
    │                                                                      │
    │  4. RL Opportunity:                                                 │
    │     - Learn to adapt hedge ratio based on market state              │
    │     - Can potentially outperform fixed delta hedge                  │
    │     - Especially valuable for path-dependent options                │
    │                                                                      │
    │  NEXT: See 02_rl_hedging_agent.py for training an RL agent          │
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
    
    try:
        # Section 1: Setup
        env, config = create_hedging_environment()
        
        # Section 2: Basic API
        demonstrate_environment_api(env)
        
        # Section 3: Benchmark
        benchmark_results = run_benchmark_comparison(env, n_episodes=500)
        
        # Section 4: Detailed episode
        episode_data = run_detailed_episode(env)
        
        # Section 5: Visualization
        visualize_results(episode_data, benchmark_results, config)
        
        # Summary
        print_summary()
        
        logger.info("Example completed successfully!")
        
    except Exception as e:
        logger.exception(f"Example failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RL Hedging Environment Example",
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

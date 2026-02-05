#!/usr/bin/env python3
"""
===============================================================================
Q-Learning: RL Trading Agent
===============================================================================

This example demonstrates training a reinforcement learning agent to trade
a single asset. The agent learns when to buy, hold, or sell based on price
patterns and portfolio state.

Learning Objectives
-------------------
1. **RL for Trading**: Understand RL formulation for systematic trading
2. **TradingEnvironment**: State, action, and reward design
3. **Agent Training**: Train and evaluate a simple trading agent
4. **Performance Metrics**: Sharpe ratio, max drawdown, win rate

Mathematical Framework
----------------------
RL Formulation:
    State s_t = (returns_{t-k:t}, position_t, PnL_t, cash_t)
    Action a_t ∈ {-1, -0.5, 0, 0.5, 1} (position targets)
    Reward r_t = log_return_t or sharpe_t

Value Function:
    V(s) = E[Σ γ^t r_t | s_0 = s]

The agent learns to maximize cumulative risk-adjusted returns.

Production Context
------------------
At a hedge fund:
- RL trading is used for market making, execution, and alpha generation
- Requires careful feature engineering (technical, fundamental)
- Extensive walk-forward validation essential
- Risk management overlays are critical

Prerequisites
-------------
- Q-learning basics (01_hedging_agent.py)
- Trading environment (src/q_learning/environments/trading.py)

Run This Example
----------------
    cd /path/to/QuantStrata
    PYTHONPATH=. python examples/q_learning/02_trading_agent.py

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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# -----------------------------------------------------------------------------
# QuantStrata Q-Learning imports
# -----------------------------------------------------------------------------
from src.q_learning.environments.trading import (
    TradingEnvironment,
    TradingEnvConfig,
    SimpleDataProvider,
    create_trading_env_from_prices,
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
# SYNTHETIC DATA GENERATION
# =============================================================================

def generate_synthetic_prices(
    n_steps: int = 1000,
    initial_price: float = 100.0,
    annual_return: float = 0.10,
    annual_vol: float = 0.20,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate synthetic price series using GBM.
    
    Parameters
    ----------
    n_steps : int
        Number of time steps.
    initial_price : float
        Starting price.
    annual_return : float
        Expected annual return.
    annual_vol : float
        Annual volatility.
    seed : int
        Random seed.
    
    Returns
    -------
    np.ndarray
        Price series.
    """
    np.random.seed(seed)
    
    # Daily parameters
    dt = 1 / 252
    daily_drift = (annual_return - 0.5 * annual_vol**2) * dt
    daily_vol = annual_vol * np.sqrt(dt)
    
    # Generate log returns
    log_returns = np.random.normal(daily_drift, daily_vol, n_steps)
    
    # Convert to prices
    prices = initial_price * np.exp(np.cumsum(log_returns))
    prices = np.insert(prices, 0, initial_price)
    
    return prices


def generate_mean_reverting_prices(
    n_steps: int = 1000,
    initial_price: float = 100.0,
    mean_price: float = 100.0,
    kappa: float = 0.1,
    vol: float = 0.02,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate mean-reverting price series (OU process).
    
    Parameters
    ----------
    n_steps : int
        Number of time steps.
    initial_price : float
        Starting price.
    mean_price : float
        Long-term mean.
    kappa : float
        Mean reversion speed.
    vol : float
        Volatility.
    seed : int
        Random seed.
    
    Returns
    -------
    np.ndarray
        Price series.
    """
    np.random.seed(seed)
    
    prices = [initial_price]
    for _ in range(n_steps):
        dp = kappa * (mean_price - prices[-1]) + vol * prices[-1] * np.random.randn()
        prices.append(prices[-1] + dp)
    
    return np.array(prices)


# =============================================================================
# SIMPLE TRADING AGENT
# =============================================================================

@dataclass
class TradingAgentConfig:
    """Configuration for trading agent."""
    learning_rate: float = 0.05
    discount_factor: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.998
    n_episodes: int = 300
    eval_frequency: int = 50


class SimpleTradingAgent:
    """
    Simple tabular Q-learning trading agent.
    
    Discretizes the state space and uses Q-learning to learn
    a trading policy.
    
    Example:
        agent = SimpleTradingAgent(
            state_bins=[10, 10, 5, 5],
            n_actions=5,
        )
        
        for episode in range(n_episodes):
            state, info = env.reset()
            done = False
            while not done:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                agent.update(state, action, reward, next_state, terminated)
                state = next_state
                done = terminated or truncated
    """
    
    def __init__(
        self,
        state_bins: List[int],
        n_actions: int,
        config: Optional[TradingAgentConfig] = None,
    ) -> None:
        """Initialize trading agent."""
        self.state_bins = state_bins
        self.n_actions = n_actions
        self.config = config or TradingAgentConfig()
        
        # Q-table
        q_shape = tuple(state_bins) + (n_actions,)
        self.q_table = np.zeros(q_shape)
        
        # Exploration
        self.epsilon = self.config.epsilon_start
    
    def discretize_state(self, state: np.ndarray) -> Tuple[int, ...]:
        """Discretize continuous state to indices."""
        # State typically has returns (multiple), position, pnl, cash
        # Normalize roughly to [-1, 1] and bin
        normalized = np.clip(state, -1, 1)
        normalized = (normalized + 1) / 2  # [0, 1]
        
        indices = []
        for i, (val, n_bins) in enumerate(zip(normalized[:len(self.state_bins)], self.state_bins)):
            idx = int(val * (n_bins - 1))
            idx = max(0, min(n_bins - 1, idx))
            indices.append(idx)
        
        return tuple(indices)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            state_idx = self.discretize_state(state)
            return int(np.argmax(self.q_table[state_idx]))
    
    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Update Q-value using TD learning."""
        state_idx = self.discretize_state(state)
        next_state_idx = self.discretize_state(next_state)
        
        current_q = self.q_table[state_idx + (action,)]
        
        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[next_state_idx])
            target_q = reward + self.config.discount_factor * max_next_q
        
        self.q_table[state_idx + (action,)] += self.config.learning_rate * (target_q - current_q)
    
    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay
        )


class BuyAndHoldAgent:
    """Baseline buy-and-hold agent."""
    
    def __init__(self, n_actions: int) -> None:
        self.n_actions = n_actions
        # Action to go fully long (typically last action)
        self.long_action = n_actions - 1
    
    def select_action(self, state: np.ndarray, training: bool = False) -> int:
        return self.long_action


# =============================================================================
# TRAINING
# =============================================================================

def train_trading_agent(
    env: TradingEnvironment,
    agent: SimpleTradingAgent,
    config: TradingAgentConfig,
) -> List[float]:
    """
    Train the trading agent.
    
    Returns
    -------
    List[float]
        Episode returns during training.
    """
    logger.info("=" * 70)
    logger.info("SECTION 2: Training Trading Agent")
    logger.info("=" * 70)
    
    episode_returns: List[float] = []
    
    for episode in range(config.n_episodes):
        state, info = env.reset()
        done = False
        
        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.update(state, action, reward, next_state, terminated or truncated)
            state = next_state
            done = terminated or truncated
        
        # Get portfolio metrics
        metrics = env.get_portfolio_metrics()
        episode_returns.append(metrics['total_return'])
        agent.decay_epsilon()
        
        if (episode + 1) % config.eval_frequency == 0:
            avg_return = np.mean(episode_returns[-config.eval_frequency:])
            logger.info(
                f"Episode {episode + 1:>4}/{config.n_episodes}: "
                f"Avg Return = {avg_return*100:>+6.2f}%, "
                f"Epsilon = {agent.epsilon:.3f}"
            )
    
    return episode_returns


# =============================================================================
# EVALUATION
# =============================================================================

@dataclass
class TradingEvalResult:
    """Evaluation results for trading agent."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    n_trades: int
    pnl_series: List[float]


def evaluate_trading_agent(
    env: TradingEnvironment,
    agent,
    n_episodes: int = 50,
) -> TradingEvalResult:
    """
    Evaluate trading agent.
    
    Returns
    -------
    TradingEvalResult
        Evaluation metrics.
    """
    returns: List[float] = []
    sharpes: List[float] = []
    drawdowns: List[float] = []
    
    for i in range(n_episodes):
        state, info = env.reset(seed=10000 + i)
        done = False
        
        while not done:
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        metrics = env.get_portfolio_metrics()
        returns.append(metrics['total_return'])
        sharpes.append(metrics['sharpe_ratio'])
        drawdowns.append(metrics['max_drawdown'])
    
    return TradingEvalResult(
        total_return=np.mean(returns),
        sharpe_ratio=np.mean(sharpes),
        max_drawdown=np.mean(drawdowns),
        n_trades=0,
        pnl_series=returns,
    )


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def run_trading_rl() -> Tuple[SimpleTradingAgent, TradingEvalResult, TradingEvalResult]:
    """
    Run the full RL trading workflow.
    
    Returns
    -------
    Tuple
        Trained agent, RL results, and baseline results.
    """
    logger.info("=" * 70)
    logger.info("SECTION 1: Environment Setup")
    logger.info("=" * 70)
    
    # Generate synthetic price data
    logger.info("")
    logger.info("Generating synthetic price data...")
    prices = generate_synthetic_prices(
        n_steps=2000,
        initial_price=100.0,
        annual_return=0.08,
        annual_vol=0.15,
        seed=42,
    )
    
    logger.info(f"  Price range: {prices.min():.2f} - {prices.max():.2f}")
    logger.info(f"  Total return: {(prices[-1]/prices[0] - 1)*100:.1f}%")
    
    # Environment configuration
    env_config = TradingEnvConfig(
        initial_capital=100_000.0,
        transaction_cost=0.001,  # 10 bps
        max_steps=252,  # 1 year episodes
        lookback_window=10,
        action_type="discrete",
        n_discrete_actions=5,  # -1, -0.5, 0, 0.5, 1
        max_position=1.0,
        reward_type="pnl",
        reward_scale=100.0,
    )
    
    env = create_trading_env_from_prices(prices, config=env_config)
    
    logger.info("")
    logger.info("Environment Configuration:")
    logger.info(f"  Initial capital: ${env_config.initial_capital:,.0f}")
    logger.info(f"  Transaction cost: {env_config.transaction_cost*10000:.0f} bps")
    logger.info(f"  Episode length:  {env_config.max_steps} steps")
    logger.info(f"  Lookback window: {env_config.lookback_window}")
    logger.info(f"  State dim:       {env.observation_space_dim}")
    logger.info(f"  Action dim:      {env.n_actions}")
    
    # Agent configuration
    agent_config = TradingAgentConfig(
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        n_episodes=300,
        eval_frequency=50,
    )
    
    # Create agents
    state_bins = [5] * env_config.lookback_window + [5, 5, 5]  # returns + pos/pnl/cash
    rl_agent = SimpleTradingAgent(
        state_bins=state_bins,
        n_actions=env.n_actions,
        config=agent_config,
    )
    
    baseline_agent = BuyAndHoldAgent(n_actions=env.n_actions)
    
    # Train RL agent
    logger.info("")
    episode_returns = train_trading_agent(env, rl_agent, agent_config)
    
    # Evaluate agents
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 3: Evaluation")
    logger.info("=" * 70)
    
    logger.info("")
    logger.info("Evaluating RL agent...")
    rl_results = evaluate_trading_agent(env, rl_agent, n_episodes=100)
    
    logger.info("Evaluating buy-and-hold baseline...")
    baseline_results = evaluate_trading_agent(env, baseline_agent, n_episodes=100)
    
    # Display results
    logger.info("")
    logger.info("Comparison:")
    logger.info("-" * 60)
    logger.info(f"{'Metric':<20} {'RL Agent':>15} {'Buy & Hold':>15}")
    logger.info("-" * 60)
    logger.info(f"{'Total Return':<20} {rl_results.total_return*100:>+14.2f}% {baseline_results.total_return*100:>+14.2f}%")
    logger.info(f"{'Sharpe Ratio':<20} {rl_results.sharpe_ratio:>15.2f} {baseline_results.sharpe_ratio:>15.2f}")
    logger.info(f"{'Max Drawdown':<20} {rl_results.max_drawdown*100:>14.2f}% {baseline_results.max_drawdown*100:>14.2f}%")
    logger.info("-" * 60)
    
    return rl_agent, rl_results, baseline_results


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_trading_results(
    rl_results: TradingEvalResult,
    baseline_results: TradingEvalResult,
) -> None:
    """Visualize trading results."""
    if not MATPLOTLIB_AVAILABLE or not ENABLE_PLOTTING:
        logger.info("Skipping plots (matplotlib not available or disabled)")
        return
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 4: Visualization")
    logger.info("=" * 70)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # -------------------------------------------------------------------------
    # Plot 1: Return distributions
    # -------------------------------------------------------------------------
    ax = axes[0, 0]
    ax.hist(
        [r * 100 for r in rl_results.pnl_series], bins=20, alpha=0.6,
        label=f'RL (μ={rl_results.total_return*100:.1f}%)', color='#2E86AB'
    )
    ax.hist(
        [r * 100 for r in baseline_results.pnl_series], bins=20, alpha=0.6,
        label=f'B&H (μ={baseline_results.total_return*100:.1f}%)', color='#E94F37'
    )
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Episode Return (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Return Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 2: Performance metrics
    # -------------------------------------------------------------------------
    ax = axes[0, 1]
    metrics = ['Return (%)', 'Sharpe', 'MaxDD (%)']
    rl_values = [
        rl_results.total_return * 100,
        rl_results.sharpe_ratio,
        rl_results.max_drawdown * 100,
    ]
    baseline_values = [
        baseline_results.total_return * 100,
        baseline_results.sharpe_ratio,
        baseline_results.max_drawdown * 100,
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax.bar(x - width/2, rl_values, width, label='RL Agent', color='#2E86AB')
    ax.bar(x + width/2, baseline_values, width, label='Buy & Hold', color='#E94F37')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Value')
    ax.set_title('Performance Metrics')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # -------------------------------------------------------------------------
    # Plot 3: Risk-return scatter
    # -------------------------------------------------------------------------
    ax = axes[1, 0]
    ax.scatter(
        np.std(rl_results.pnl_series) * 100, rl_results.total_return * 100,
        s=200, marker='o', color='#2E86AB', label='RL Agent'
    )
    ax.scatter(
        np.std(baseline_results.pnl_series) * 100, baseline_results.total_return * 100,
        s=200, marker='s', color='#E94F37', label='Buy & Hold'
    )
    ax.set_xlabel('Return Volatility (%)')
    ax.set_ylabel('Mean Return (%)')
    ax.set_title('Risk-Return Trade-off')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 4: Cumulative returns comparison (simulated)
    # -------------------------------------------------------------------------
    ax = axes[1, 1]
    # Simulate cumulative returns
    n_episodes = len(rl_results.pnl_series)
    rl_cum = np.cumprod(1 + np.array(rl_results.pnl_series[:min(50, n_episodes)]))
    bh_cum = np.cumprod(1 + np.array(baseline_results.pnl_series[:min(50, n_episodes)]))
    
    ax.plot(rl_cum, label='RL Agent', color='#2E86AB', linewidth=2)
    ax.plot(bh_cum, label='Buy & Hold', color='#E94F37', linewidth=2)
    ax.axhline(1, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Wealth')
    ax.set_title('Cumulative Performance')
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
    │  1. RL Trading Formulation:                                         │
    │     - State: price returns, position, PnL, cash                     │
    │     - Action: position target (long/short/flat)                     │
    │     - Reward: P&L or Sharpe-based                                   │
    │                                                                      │
    │  2. Training Considerations:                                        │
    │     - Non-stationary markets make training difficult                │
    │     - Overfitting to training data is a major risk                  │
    │     - Walk-forward validation is essential                          │
    │                                                                      │
    │  3. Agent Design:                                                   │
    │     - Tabular Q-learning for simple, interpretable agents           │
    │     - DQN for larger state spaces                                   │
    │     - Policy gradient for continuous actions                        │
    │                                                                      │
    │  4. Production Deployment:                                          │
    │     - Risk limits and position constraints                          │
    │     - Transaction cost modeling                                     │
    │     - Real-time inference pipeline                                  │
    │     - Performance monitoring and model retraining                   │
    │                                                                      │
    │  NEXT: See machine_learning/ for neural network pricers             │
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
        # Run trading RL workflow
        agent, rl_results, baseline_results = run_trading_rl()
        
        # Visualization
        visualize_trading_results(rl_results, baseline_results)
        
        # Summary
        print_summary()
        
        logger.info("Example completed successfully!")
        
    except Exception as e:
        logger.exception(f"Example failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Q-Learning Trading Agent Example",
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

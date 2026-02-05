#!/usr/bin/env python3
"""
===============================================================================
Q-Learning: Training a Trading Agent with QuantStrata Infrastructure
===============================================================================

This example demonstrates training an RL agent for systematic trading
using QuantStrata's production q_learning module.

Learning Objectives
-------------------
1. **Trading Environment**: Use TradingEnvironment with TradingEnvConfig
2. **Data Providers**: Use SimpleDataProvider for price data
3. **Training Pipeline**: Use run_training() with multiple seeds
4. **Performance Metrics**: Sharpe, max drawdown, win rate

Mathematical Framework
----------------------
The agent maximizes risk-adjusted returns:
    max_π E[Σ_t γ^t r_t]
    
Where r_t depends on reward_type:
    - "pnl": r_t = (portfolio_value_t - portfolio_value_{t-1}) / capital
    - "sharpe": running Sharpe ratio
    - "log_return": log(1 + return_t)

State: recent returns, position, P&L, cash
Action: target position as fraction of capital

Production Context
------------------
At a hedge fund:
- RL trading agents can learn complex market patterns
- Must be validated against buy-and-hold and other baselines
- Transaction costs significantly impact performance
- Regime changes require ongoing monitoring and retraining

Prerequisites
-------------
- Basic RL concepts
- Market microstructure understanding
- Q-learning hedging example

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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# -----------------------------------------------------------------------------
# QuantStrata imports - using existing q_learning infrastructure
# -----------------------------------------------------------------------------
from src.q_learning.environments.trading import (
    TradingEnvironment,
    TradingEnvConfig,
    SimpleDataProvider,
)
from src.q_learning.core.types import RLTrainingConfig, RLTrainingResult, Transition
from src.q_learning.core.protocols import RLAgent
from src.q_learning.pipelines.training import run_training
from src.q_learning.evaluation.metrics import sharpe_ratio, max_drawdown, win_rate


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

def generate_trending_prices(
    n_steps: int = 1000,
    initial_price: float = 100.0,
    trend: float = 0.0001,
    volatility: float = 0.02,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate synthetic price series with trend and volatility.
    
    Parameters
    ----------
    n_steps : int
        Number of time steps.
    initial_price : float
        Starting price.
    trend : float
        Daily drift.
    volatility : float
        Daily volatility.
    seed : int, optional
        Random seed.
    
    Returns
    -------
    ndarray
        Price series.
    """
    rng = np.random.default_rng(seed)
    
    returns = trend + volatility * rng.standard_normal(n_steps)
    prices = initial_price * np.cumprod(1 + returns)
    prices = np.insert(prices, 0, initial_price)
    
    return prices


def generate_mean_reverting_prices(
    n_steps: int = 1000,
    initial_price: float = 100.0,
    mean_price: float = 100.0,
    reversion_speed: float = 0.1,
    volatility: float = 0.02,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate mean-reverting price series.
    
    Parameters
    ----------
    n_steps : int
        Number of time steps.
    initial_price : float
        Starting price.
    mean_price : float
        Long-term mean price.
    reversion_speed : float
        Speed of mean reversion.
    volatility : float
        Daily volatility.
    seed : int, optional
        Random seed.
    
    Returns
    -------
    ndarray
        Price series.
    """
    rng = np.random.default_rng(seed)
    
    prices = np.zeros(n_steps + 1)
    prices[0] = initial_price
    
    for t in range(n_steps):
        drift = reversion_speed * (mean_price - prices[t])
        noise = volatility * prices[t] * rng.standard_normal()
        prices[t + 1] = prices[t] + drift + noise
        prices[t + 1] = max(prices[t + 1], 1.0)  # Prevent negative prices
    
    return prices


# =============================================================================
# TRADING AGENT (implements RLAgent protocol)
# =============================================================================

@dataclass
class TradingAgentConfig:
    """Configuration for trading Q-learning agent."""
    n_state_bins: int = 10
    learning_rate: float = 0.05
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.02
    epsilon_decay: float = 0.998


class TradingQLearningAgent:
    """
    Q-learning agent for trading, conforming to RLAgent protocol.
    
    Uses state discretization with focus on:
    - Recent price momentum
    - Current position
    - P&L status
    
    Example:
        agent = TradingQLearningAgent(
            state_dim=23,  # 20 returns + position + pnl + cash
            n_actions=5,
            config=TradingAgentConfig(),
        )
    """
    
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        config: Optional[TradingAgentConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize trading agent."""
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.config = config or TradingAgentConfig()
        self._rng = np.random.default_rng(seed)
        
        self.n_bins = self.config.n_state_bins
        self._q_table: Dict[tuple, np.ndarray] = {}
        
        self.epsilon = self.config.epsilon_start
        
        # State bounds - will be updated during training
        self._state_min = np.full(state_dim, -1.0)
        self._state_max = np.full(state_dim, 1.0)
        
        self.n_updates = 0
    
    def _discretize_state(self, state: np.ndarray) -> tuple:
        """Convert continuous state to discrete bins."""
        state = np.clip(state, self._state_min, self._state_max)
        normalized = (state - self._state_min) / (self._state_max - self._state_min + 1e-8)
        bin_indices = (normalized * (self.n_bins - 1)).astype(int)
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
        return tuple(bin_indices)
    
    def _get_q_values(self, state_key: tuple) -> np.ndarray:
        """Get or initialize Q-values for state."""
        if state_key not in self._q_table:
            self._q_table[state_key] = np.zeros(self.n_actions)
        return self._q_table[state_key]
    
    def select_action(
        self,
        state: Any,
        *,
        training: bool = False,
        explore: bool = True,
    ) -> int:
        """Select action using epsilon-greedy policy."""
        state = np.atleast_1d(state)
        
        # Update bounds adaptively
        self._state_min = np.minimum(self._state_min, state)
        self._state_max = np.maximum(self._state_max, state)
        
        state_key = self._discretize_state(state)
        q_values = self._get_q_values(state_key)
        
        if explore and training and self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_actions))
        
        return int(np.argmax(q_values))
    
    def update(
        self,
        transitions: Optional[List[Any]] = None,
        batch: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, float]]:
        """Update Q-values from transitions."""
        if not transitions:
            return None
        
        total_td_error = 0.0
        
        for t in transitions:
            state = np.atleast_1d(t.state)
            next_state = np.atleast_1d(t.next_state)
            action = int(t.action)
            reward = float(t.reward)
            done = t.terminated or t.truncated
            
            state_key = self._discretize_state(state)
            next_state_key = self._discretize_state(next_state)
            
            q_values = self._get_q_values(state_key)
            next_q_values = self._get_q_values(next_state_key)
            
            td_target = reward if done else reward + self.config.gamma * np.max(next_q_values)
            td_error = td_target - q_values[action]
            total_td_error += abs(td_error)
            
            q_values[action] += self.config.learning_rate * td_error
        
        self.n_updates += 1
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)
        
        return {
            "td_error": total_td_error / len(transitions),
            "epsilon": self.epsilon,
            "q_table_size": len(self._q_table),
        }
    
    def get_parameters(self) -> Dict[str, Any]:
        """Return parameters for checkpointing."""
        return {
            "q_table": {str(k): v.tolist() for k, v in self._q_table.items()},
            "epsilon": self.epsilon,
            "state_min": self._state_min.tolist(),
            "state_max": self._state_max.tolist(),
            "n_updates": self.n_updates,
        }
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Load parameters from checkpoint."""
        self._q_table = {eval(k): np.array(v) for k, v in params["q_table"].items()}
        self.epsilon = params["epsilon"]
        self._state_min = np.array(params["state_min"])
        self._state_max = np.array(params["state_max"])
        self.n_updates = params["n_updates"]


# =============================================================================
# BASELINE AGENTS
# =============================================================================

class BuyAndHoldAgent:
    """Buy-and-hold baseline: always full long position."""
    
    def __init__(self, n_actions: int) -> None:
        self.n_actions = n_actions
        self.long_action = n_actions - 1  # Maximum long position
    
    def select_action(self, state: Any, *, training: bool = False, explore: bool = True) -> int:
        return self.long_action
    
    def update(self, transitions: Optional[List[Any]] = None, batch: Optional[Dict[str, Any]] = None):
        return None
    
    def get_parameters(self) -> Dict[str, Any]:
        return {"long_action": self.long_action}
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        self.long_action = params["long_action"]


class MomentumAgent:
    """Simple momentum baseline: position based on recent returns."""
    
    def __init__(self, n_actions: int, lookback: int = 5) -> None:
        self.n_actions = n_actions
        self.lookback = lookback
    
    def select_action(self, state: Any, *, training: bool = False, explore: bool = True) -> int:
        state = np.atleast_1d(state)
        # First lookback elements are returns
        if len(state) >= self.lookback:
            recent_return = np.mean(state[:self.lookback])
            if recent_return > 0.001:
                return self.n_actions - 1  # Long
            elif recent_return < -0.001:
                return 0  # Short
        return self.n_actions // 2  # Neutral
    
    def update(self, transitions: Optional[List[Any]] = None, batch: Optional[Dict[str, Any]] = None):
        return None
    
    def get_parameters(self) -> Dict[str, Any]:
        return {"lookback": self.lookback}
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        self.lookback = params["lookback"]


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_trading_agent(
    agent: RLAgent,
    env: TradingEnvironment,
    n_episodes: int = 50,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate trading agent performance.
    
    Returns
    -------
    dict
        Performance metrics.
    """
    returns = []
    final_values = []
    all_step_returns = []
    
    for ep in range(n_episodes):
        state, info = env.reset(seed=seed + ep if seed else None)
        episode_return = 0.0
        step_returns = []
        
        while True:
            action = agent.select_action(state, training=False, explore=False)
            state, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            step_returns.append(info.get("return", 0.0))
            
            if terminated or truncated:
                break
        
        returns.append(episode_return)
        final_values.append(info.get("portfolio_value", 0.0))
        all_step_returns.extend(step_returns)
    
    returns_arr = np.array(returns)
    step_returns_arr = np.array(all_step_returns)
    
    # Annualized Sharpe (assuming daily steps, 252 trading days)
    if np.std(step_returns_arr) > 0:
        ann_sharpe = np.mean(step_returns_arr) / np.std(step_returns_arr) * np.sqrt(252)
    else:
        ann_sharpe = 0.0
    
    return {
        "mean_return": float(np.mean(returns_arr)),
        "std_return": float(np.std(returns_arr)),
        "total_return": float(np.mean([v / 1_000_000 - 1 for v in final_values])),
        "sharpe": ann_sharpe,
        "max_drawdown": max_drawdown(list(np.cumsum(returns))),
        "win_rate": win_rate(returns),
    }


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def run_trading_rl() -> Tuple[RLTrainingResult, Dict[str, Dict[str, float]]]:
    """
    Run the complete trading RL workflow.
    
    Returns
    -------
    Tuple
        Training result and evaluation metrics.
    """
    logger.info("=" * 70)
    logger.info("SECTION 1: Data Generation")
    logger.info("=" * 70)
    
    # Generate synthetic price data
    n_train_steps = 2000
    n_test_steps = 500
    
    # Mix of trending and mean-reverting data
    train_prices = generate_mean_reverting_prices(
        n_steps=n_train_steps,
        initial_price=100.0,
        mean_price=100.0,
        reversion_speed=0.05,
        volatility=0.015,
        seed=42,
    )
    
    test_prices = generate_mean_reverting_prices(
        n_steps=n_test_steps,
        initial_price=train_prices[-1],
        mean_price=100.0,
        reversion_speed=0.05,
        volatility=0.015,
        seed=123,
    )
    
    logger.info("")
    logger.info(f"  Train samples: {len(train_prices)}")
    logger.info(f"  Test samples:  {len(test_prices)}")
    logger.info(f"  Price range:   [{train_prices.min():.2f}, {train_prices.max():.2f}]")
    
    # Setup environment
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 2: Environment Setup")
    logger.info("=" * 70)
    
    env_config = TradingEnvConfig(
        initial_capital=1_000_000.0,
        transaction_cost=0.001,  # 10 bps
        max_steps=200,
        lookback_window=20,
        include_position=True,
        include_pnl=True,
        include_cash=True,
        action_type="discrete",
        n_discrete_actions=5,  # [-1, -0.5, 0, 0.5, 1] positions
        max_position=1.0,
        reward_type="pnl",
    )
    
    train_provider = SimpleDataProvider(train_prices)
    train_env = TradingEnvironment(train_provider, config=env_config, seed=42)
    
    logger.info("")
    logger.info(f"  Initial capital: ${env_config.initial_capital:,.0f}")
    logger.info(f"  Transaction cost: {env_config.transaction_cost:.1%}")
    logger.info(f"  Max steps/episode: {env_config.max_steps}")
    logger.info(f"  Lookback window: {env_config.lookback_window}")
    logger.info(f"  Actions: {env_config.n_discrete_actions}")
    logger.info(f"  State dim: {train_env.observation_space_dim}")
    
    # Create agents
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 3: Agent Creation")
    logger.info("=" * 70)
    
    agent_config = TradingAgentConfig(
        n_state_bins=8,
        learning_rate=0.05,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.02,
        epsilon_decay=0.998,
    )
    
    rl_agent = TradingQLearningAgent(
        state_dim=train_env.observation_space_dim,
        n_actions=env_config.n_discrete_actions,
        config=agent_config,
        seed=42,
    )
    
    buy_hold_agent = BuyAndHoldAgent(n_actions=env_config.n_discrete_actions)
    momentum_agent = MomentumAgent(n_actions=env_config.n_discrete_actions)
    
    logger.info("")
    logger.info("  RL Agent:       TradingQLearningAgent")
    logger.info(f"    State bins:   {agent_config.n_state_bins}")
    logger.info(f"    Learning rate: {agent_config.learning_rate}")
    logger.info("  Baselines:      BuyAndHold, Momentum")
    
    # Training
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 4: Training")
    logger.info("=" * 70)
    
    training_config = RLTrainingConfig(
        n_episodes=300,
        max_steps_per_episode=env_config.max_steps,
        gamma=agent_config.gamma,
        log_every=100,
        eval_episodes=5,
        verbose=1,
    )
    
    logger.info("")
    logger.info(f"  Episodes: {training_config.n_episodes}")
    logger.info("")
    
    training_result = run_training(
        agent=rl_agent,
        env=train_env,
        config=training_config,
    )
    
    logger.info("")
    logger.info(f"  Training time: {training_result.training_time_seconds:.1f}s")
    logger.info(f"  Best episode:  {training_result.best_episode}")
    logger.info(f"  Best return:   {training_result.best_episode_return:.4f}")
    
    # Evaluation on test data
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 5: Evaluation (Out-of-Sample)")
    logger.info("=" * 70)
    
    test_provider = SimpleDataProvider(test_prices)
    test_env = TradingEnvironment(test_provider, config=env_config, seed=123)
    
    logger.info("")
    logger.info("Evaluating agents on test data...")
    
    rl_metrics = evaluate_trading_agent(rl_agent, test_env, n_episodes=50, seed=1000)
    bh_metrics = evaluate_trading_agent(buy_hold_agent, test_env, n_episodes=50, seed=1000)
    mom_metrics = evaluate_trading_agent(momentum_agent, test_env, n_episodes=50, seed=1000)
    
    logger.info("")
    logger.info(f"{'Metric':<20} {'RL Agent':>12} {'Buy&Hold':>12} {'Momentum':>12}")
    logger.info("-" * 60)
    logger.info(f"{'Mean Return':<20} {rl_metrics['mean_return']:>12.4f} {bh_metrics['mean_return']:>12.4f} {mom_metrics['mean_return']:>12.4f}")
    logger.info(f"{'Std Return':<20} {rl_metrics['std_return']:>12.4f} {bh_metrics['std_return']:>12.4f} {mom_metrics['std_return']:>12.4f}")
    logger.info(f"{'Total Return':<20} {rl_metrics['total_return']:>12.2%} {bh_metrics['total_return']:>12.2%} {mom_metrics['total_return']:>12.2%}")
    logger.info(f"{'Sharpe (ann.)':<20} {rl_metrics['sharpe']:>12.2f} {bh_metrics['sharpe']:>12.2f} {mom_metrics['sharpe']:>12.2f}")
    logger.info(f"{'Max Drawdown':<20} {rl_metrics['max_drawdown']:>12.4f} {bh_metrics['max_drawdown']:>12.4f} {mom_metrics['max_drawdown']:>12.4f}")
    logger.info(f"{'Win Rate':<20} {rl_metrics['win_rate']:>12.1%} {bh_metrics['win_rate']:>12.1%} {mom_metrics['win_rate']:>12.1%}")
    logger.info("-" * 60)
    
    evaluation_results = {
        "rl_agent": rl_metrics,
        "buy_hold": bh_metrics,
        "momentum": mom_metrics,
    }
    
    return training_result, evaluation_results


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_results(
    training_result: RLTrainingResult,
    evaluation_results: Dict[str, Dict[str, float]],
) -> None:
    """Visualize training and evaluation results."""
    if not MATPLOTLIB_AVAILABLE or not ENABLE_PLOTTING:
        logger.info("Skipping plots")
        return
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 6: Visualization")
    logger.info("=" * 70)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Training returns
    ax = axes[0, 0]
    returns = training_result.episode_returns
    window = 30
    smoothed = np.convolve(returns, np.ones(window) / window, mode='valid')
    
    ax.plot(returns, alpha=0.3, color='#2E86AB', label='Episode Return')
    ax.plot(range(window - 1, len(returns)), smoothed, color='#E94F37', 
            linewidth=2, label=f'MA({window})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Return')
    ax.set_title('Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Epsilon decay
    ax = axes[0, 1]
    if 'epsilon' in training_result.history:
        ax.plot(training_result.history['epsilon'], color='#4CAF50', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Epsilon')
        ax.set_title('Exploration Rate')
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Performance comparison
    ax = axes[1, 0]
    
    agents = ['RL Agent', 'Buy & Hold', 'Momentum']
    agent_keys = ['rl_agent', 'buy_hold', 'momentum']
    metrics_to_plot = ['mean_return', 'sharpe']
    
    x = np.arange(len(agents))
    width = 0.35
    
    values1 = [evaluation_results[k]['mean_return'] for k in agent_keys]
    values2 = [evaluation_results[k]['sharpe'] / 10 for k in agent_keys]  # Scale Sharpe
    
    ax.bar(x - width/2, values1, width, label='Mean Return', color='#2E86AB', alpha=0.8)
    ax.bar(x + width/2, values2, width, label='Sharpe / 10', color='#E94F37', alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(agents)
    ax.set_ylabel('Value')
    ax.set_title('Agent Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Risk-Return
    ax = axes[1, 1]
    
    colors = ['#2E86AB', '#E94F37', '#4CAF50']
    markers = ['o', 's', '^']
    
    for i, (name, key) in enumerate(zip(agents, agent_keys)):
        m = evaluation_results[key]
        ax.scatter(m['std_return'], m['mean_return'], 
                   s=200, c=colors[i], marker=markers[i], label=name, zorder=5)
    
    ax.set_xlabel('Return Std (Risk)')
    ax.set_ylabel('Mean Return')
    ax.set_title('Risk-Return Trade-off')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show(block=True)
    
    logger.info("Visualization complete")


# =============================================================================
# SUMMARY
# =============================================================================

def print_summary() -> None:
    """Print key takeaways."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    
    summary = """
    ┌─────────────────────────────────────────────────────────────────────┐
    │                         KEY TAKEAWAYS                                │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                      │
    │  1. QuantStrata Trading Infrastructure:                             │
    │     - TradingEnvironment: Gymnasium-compatible trading sim          │
    │     - SimpleDataProvider: Wraps price arrays                        │
    │     - TradingEnvConfig: Capital, costs, actions, rewards            │
    │                                                                      │
    │  2. Agent Design:                                                   │
    │     - State: returns + position + P&L + cash                        │
    │     - Action: discrete position targets [-1, 1]                     │
    │     - Reward: P&L, Sharpe, or log returns                           │
    │                                                                      │
    │  3. Training Considerations:                                        │
    │     - Exploration vs exploitation (epsilon decay)                   │
    │     - Transaction costs impact performance                          │
    │     - Compare to simple baselines (buy-hold, momentum)              │
    │                                                                      │
    │  4. Production Deployment:                                          │
    │     - Validate on out-of-sample data                                │
    │     - Monitor for regime changes                                    │
    │     - Use ensemble of agents                                        │
    │     - Implement risk limits                                         │
    │                                                                      │
    └─────────────────────────────────────────────────────────────────────┘
    """
    logger.info(summary)


# =============================================================================
# MAIN
# =============================================================================

def main(args: argparse.Namespace) -> None:
    """Main entry point."""
    global ENABLE_PLOTTING
    ENABLE_PLOTTING = args.plot
    
    try:
        training_result, evaluation_results = run_trading_rl()
        visualize_results(training_result, evaluation_results)
        print_summary()
        logger.info("Example completed successfully!")
        
    except Exception as e:
        logger.exception(f"Example failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q-Learning Trading Agent Example")
    parser.add_argument("--plot", action="store_true", default=True)
    parser.add_argument("--no-plot", action="store_false", dest="plot")
    
    args = parser.parse_args()
    main(args)

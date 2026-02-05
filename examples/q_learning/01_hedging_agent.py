#!/usr/bin/env python3
"""
===============================================================================
Q-Learning: RL Hedging Agent
===============================================================================

This example demonstrates training a reinforcement learning agent to hedge
an options portfolio. The agent learns to minimize P&L variance while
controlling transaction costs - potentially outperforming naive delta hedging.

Learning Objectives
-------------------
1. **RL for Hedging**: Understand RL formulation for option hedging
2. **Environment Design**: State, action, and reward specification
3. **Agent Training**: Train a simple Q-learning or policy gradient agent
4. **Benchmark Comparison**: Compare RL agent vs delta hedge

Mathematical Framework
----------------------
RL Formulation:
    State s_t = (S_t/S_0, τ, Δ_t, Γ_t, position_t, PnL_t)
    Action a_t = hedge_ratio ∈ [-2, 2] (multiple of delta)
    Reward r_t = -risk_aversion × Var(PnL) - transaction_costs

Optimal Policy:
    π*(s) = argmax_a E[Σ γ^t r_t | s_0 = s, a_0 = a]

Deep Hedging Insight:
    RL can discover that:
    - Hedging less in low-gamma regions saves costs
    - Hedging more in high-gamma regions reduces risk
    - Optimal frequency depends on realized volatility

Production Context
------------------
At a hedge fund:
- RL hedging is cutting-edge research (JP Morgan, etc.)
- Can reduce transaction costs by 30-50% vs delta hedge
- Requires careful backtesting and risk limits
- Often combined with classical delta hedge as baseline

Prerequisites
-------------
- Delta hedging (examples/risk/04_delta_hedging.py)
- Q-learning environments (src/q_learning/)

Run This Example
----------------
    cd /path/to/QuantStrata
    PYTHONPATH=. python examples/q_learning/01_hedging_agent.py

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
# SIMPLE Q-LEARNING AGENT
# =============================================================================

@dataclass
class QLearningConfig:
    """Configuration for Q-learning agent."""
    # Learning parameters
    learning_rate: float = 0.1
    discount_factor: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    # Training parameters
    n_episodes: int = 500
    eval_frequency: int = 50


class SimpleQLearningAgent:
    """
    Simple tabular Q-learning agent for hedging.
    
    Uses discretized state space and action space for simplicity.
    For production, use DQN or policy gradient methods.
    
    Example:
        agent = SimpleQLearningAgent(
            state_bins=[10, 10, 5, 5],  # Discretization per state dim
            n_actions=11,
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
        config: Optional[QLearningConfig] = None,
    ) -> None:
        """
        Initialize Q-learning agent.
        
        Parameters
        ----------
        state_bins : List[int]
            Number of bins for each state dimension.
        n_actions : int
            Number of discrete actions.
        config : QLearningConfig, optional
            Learning configuration.
        """
        self.state_bins = state_bins
        self.n_actions = n_actions
        self.config = config or QLearningConfig()
        
        # Initialize Q-table
        q_shape = tuple(state_bins) + (n_actions,)
        self.q_table = np.zeros(q_shape)
        
        # Exploration parameter
        self.epsilon = self.config.epsilon_start
        
        # Training stats
        self.episode_rewards: List[float] = []
        
    def discretize_state(self, state: np.ndarray) -> Tuple[int, ...]:
        """
        Discretize continuous state to table indices.
        
        Assumes state values are roughly in [-2, 2] range.
        """
        # Clip and normalize to [0, 1]
        normalized = (np.clip(state, -2, 2) + 2) / 4
        
        # Convert to bin indices
        indices = []
        for i, (val, n_bins) in enumerate(zip(normalized, self.state_bins)):
            idx = int(val * (n_bins - 1))
            idx = max(0, min(n_bins - 1, idx))
            indices.append(idx)
        
        return tuple(indices)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Parameters
        ----------
        state : np.ndarray
            Current state.
        training : bool
            If True, use epsilon-greedy; if False, use greedy.
        
        Returns
        -------
        int
            Selected action index.
        """
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(self.n_actions)
        else:
            # Exploit: best Q-value action
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
        """
        Update Q-value using TD learning.
        
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        """
        state_idx = self.discretize_state(state)
        next_state_idx = self.discretize_state(next_state)
        
        # Current Q-value
        current_q = self.q_table[state_idx + (action,)]
        
        # Target Q-value
        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[next_state_idx])
            target_q = reward + self.config.discount_factor * max_next_q
        
        # TD update
        self.q_table[state_idx + (action,)] += self.config.learning_rate * (target_q - current_q)
    
    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay
        )


class DeltaHedgeAgent:
    """
    Baseline delta hedge agent for comparison.
    
    Always hedges at the current delta (action = 1.0 in our formulation).
    """
    
    def __init__(self, n_actions: int) -> None:
        """Initialize delta hedge agent."""
        self.n_actions = n_actions
        # Find action closest to 1.0 (full delta hedge)
        self.delta_action = n_actions // 2 + (n_actions // 4)  # Roughly 1.0
    
    def select_action(self, state: np.ndarray, training: bool = False) -> int:
        """Always return delta hedge action."""
        return self.delta_action


# =============================================================================
# TRAINING
# =============================================================================

def train_agent(
    env: HedgingEnvironment,
    agent: SimpleQLearningAgent,
    config: QLearningConfig,
) -> List[float]:
    """
    Train the Q-learning agent.
    
    Returns
    -------
    List[float]
        Episode rewards during training.
    """
    logger.info("=" * 70)
    logger.info("SECTION 2: Training Q-Learning Agent")
    logger.info("=" * 70)
    
    episode_rewards: List[float] = []
    
    for episode in range(config.n_episodes):
        state, info = env.reset(seed=episode)
        total_reward = 0.0
        done = False
        
        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            agent.update(state, action, reward, next_state, terminated or truncated)
            
            state = next_state
            total_reward += reward
            done = terminated or truncated
        
        episode_rewards.append(total_reward)
        agent.decay_epsilon()
        
        # Logging
        if (episode + 1) % config.eval_frequency == 0:
            avg_reward = np.mean(episode_rewards[-config.eval_frequency:])
            logger.info(
                f"Episode {episode + 1:>4}/{config.n_episodes}: "
                f"Avg Reward = {avg_reward:>8.4f}, "
                f"Epsilon = {agent.epsilon:.3f}"
            )
    
    return episode_rewards


# =============================================================================
# EVALUATION
# =============================================================================

@dataclass
class EvaluationResult:
    """Evaluation results for an agent."""
    mean_pnl: float
    std_pnl: float
    mean_costs: float
    sharpe: float
    pnl_distribution: np.ndarray


def evaluate_agent(
    env: HedgingEnvironment,
    agent,
    n_episodes: int = 100,
    seed_offset: int = 10000,
) -> EvaluationResult:
    """
    Evaluate agent performance.
    
    Returns
    -------
    EvaluationResult
        Evaluation metrics.
    """
    pnls: List[float] = []
    costs: List[float] = []
    
    for i in range(n_episodes):
        state, info = env.reset(seed=seed_offset + i)
        done = False
        
        while not done:
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        pnls.append(info['pnl'])
        costs.append(info['cumulative_cost'])
    
    pnls = np.array(pnls)
    costs = np.array(costs)
    
    return EvaluationResult(
        mean_pnl=np.mean(pnls),
        std_pnl=np.std(pnls),
        mean_costs=np.mean(costs),
        sharpe=np.mean(pnls) / (np.std(pnls) + 1e-8),
        pnl_distribution=pnls,
    )


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def run_hedging_rl() -> Tuple[SimpleQLearningAgent, EvaluationResult, EvaluationResult]:
    """
    Run the full RL hedging workflow.
    
    Returns
    -------
    Tuple
        Trained agent, RL results, and baseline results.
    """
    logger.info("=" * 70)
    logger.info("SECTION 1: Environment Setup")
    logger.info("=" * 70)
    
    # Environment configuration
    env_config = HedgingEnvConfig(
        spot=100.0,
        strike=100.0,  # ATM
        maturity=0.25,  # 3 months
        volatility=0.2,
        risk_free_rate=0.05,
        dividend_yield=0.0,
        option_type="call",
        n_steps=50,  # Hedging intervals
        proportional_cost=0.001,  # 10 bps
        action_type="discrete",
        n_discrete_actions=11,  # Hedge ratios from -1 to +2
        reward_type="risk_adjusted",
        risk_aversion=0.1,
    )
    
    env = HedgingEnvironment(config=env_config)
    
    logger.info("")
    logger.info("Environment Configuration:")
    logger.info(f"  Spot/Strike:    {env_config.spot}/{env_config.strike}")
    logger.info(f"  Maturity:       {env_config.maturity:.2f} years")
    logger.info(f"  Volatility:     {env_config.volatility:.1%}")
    logger.info(f"  Hedging steps:  {env_config.n_steps}")
    logger.info(f"  Transaction cost: {env_config.proportional_cost*10000:.0f} bps")
    logger.info(f"  State dim:      {env.observation_space_dim}")
    logger.info(f"  Action dim:     {env.n_actions}")
    
    # Q-learning configuration
    q_config = QLearningConfig(
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        n_episodes=500,
        eval_frequency=100,
    )
    
    # Create agents
    rl_agent = SimpleQLearningAgent(
        state_bins=[10, 10, 5, 5, 5],  # Discretization
        n_actions=env.n_actions,
        config=q_config,
    )
    
    baseline_agent = DeltaHedgeAgent(n_actions=env.n_actions)
    
    # Train RL agent
    logger.info("")
    episode_rewards = train_agent(env, rl_agent, q_config)
    
    # Evaluate agents
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 3: Evaluation")
    logger.info("=" * 70)
    
    logger.info("")
    logger.info("Evaluating RL agent...")
    rl_results = evaluate_agent(env, rl_agent, n_episodes=200)
    
    logger.info("Evaluating delta hedge baseline...")
    baseline_results = evaluate_agent(env, baseline_agent, n_episodes=200)
    
    # Display results
    logger.info("")
    logger.info("Comparison:")
    logger.info("-" * 60)
    logger.info(f"{'Metric':<20} {'RL Agent':>15} {'Delta Hedge':>15}")
    logger.info("-" * 60)
    logger.info(f"{'Mean P&L':<20} {rl_results.mean_pnl:>15.4f} {baseline_results.mean_pnl:>15.4f}")
    logger.info(f"{'Std P&L':<20} {rl_results.std_pnl:>15.4f} {baseline_results.std_pnl:>15.4f}")
    logger.info(f"{'Mean Costs':<20} {rl_results.mean_costs:>15.4f} {baseline_results.mean_costs:>15.4f}")
    logger.info(f"{'Sharpe':<20} {rl_results.sharpe:>15.4f} {baseline_results.sharpe:>15.4f}")
    logger.info("-" * 60)
    
    # Improvement
    cost_reduction = (baseline_results.mean_costs - rl_results.mean_costs) / baseline_results.mean_costs * 100
    std_change = (baseline_results.std_pnl - rl_results.std_pnl) / baseline_results.std_pnl * 100
    
    logger.info("")
    logger.info("RL Agent Improvement:")
    logger.info(f"  Cost reduction:     {cost_reduction:>+.1f}%")
    logger.info(f"  Volatility change:  {std_change:>+.1f}%")
    
    return rl_agent, rl_results, baseline_results


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_results(
    rl_results: EvaluationResult,
    baseline_results: EvaluationResult,
) -> None:
    """Visualize agent comparison."""
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
    # Plot 1: P&L distributions
    # -------------------------------------------------------------------------
    ax = axes[0, 0]
    ax.hist(
        rl_results.pnl_distribution, bins=30, alpha=0.6,
        label=f'RL Agent (μ={rl_results.mean_pnl:.3f})', color='#2E86AB'
    )
    ax.hist(
        baseline_results.pnl_distribution, bins=30, alpha=0.6,
        label=f'Delta Hedge (μ={baseline_results.mean_pnl:.3f})', color='#E94F37'
    )
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Final P&L')
    ax.set_ylabel('Frequency')
    ax.set_title('P&L Distribution Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 2: Metrics comparison
    # -------------------------------------------------------------------------
    ax = axes[0, 1]
    metrics = ['Mean P&L', 'Std P&L', 'Mean Costs']
    rl_values = [rl_results.mean_pnl, rl_results.std_pnl, rl_results.mean_costs]
    baseline_values = [baseline_results.mean_pnl, baseline_results.std_pnl, baseline_results.mean_costs]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax.bar(x - width/2, rl_values, width, label='RL Agent', color='#2E86AB')
    ax.bar(x + width/2, baseline_values, width, label='Delta Hedge', color='#E94F37')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Value')
    ax.set_title('Performance Metrics')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # -------------------------------------------------------------------------
    # Plot 3: Sharpe ratio comparison
    # -------------------------------------------------------------------------
    ax = axes[1, 0]
    sharpes = [rl_results.sharpe, baseline_results.sharpe]
    colors = ['#2E86AB', '#E94F37']
    bars = ax.bar(['RL Agent', 'Delta Hedge'], sharpes, color=colors)
    
    for bar, val in zip(bars, sharpes):
        ax.text(
            bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', fontsize=12
        )
    
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Risk-Adjusted Performance')
    ax.grid(True, alpha=0.3, axis='y')
    
    # -------------------------------------------------------------------------
    # Plot 4: Cost-risk trade-off
    # -------------------------------------------------------------------------
    ax = axes[1, 1]
    ax.scatter(
        rl_results.mean_costs, rl_results.std_pnl,
        s=200, marker='o', color='#2E86AB', label='RL Agent'
    )
    ax.scatter(
        baseline_results.mean_costs, baseline_results.std_pnl,
        s=200, marker='s', color='#E94F37', label='Delta Hedge'
    )
    
    # Arrow showing improvement direction
    ax.annotate(
        '', xy=(rl_results.mean_costs, rl_results.std_pnl),
        xytext=(baseline_results.mean_costs, baseline_results.std_pnl),
        arrowprops=dict(arrowstyle='->', color='green', lw=2)
    )
    
    ax.set_xlabel('Mean Transaction Costs')
    ax.set_ylabel('P&L Volatility')
    ax.set_title('Cost-Risk Trade-off (arrow = improvement)')
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
    │  1. RL Hedging Formulation:                                         │
    │     - State: spot, time, delta, gamma, position, PnL                │
    │     - Action: hedge ratio (fraction of delta to hold)               │
    │     - Reward: risk-adjusted return minus costs                      │
    │                                                                      │
    │  2. Q-Learning Approach:                                            │
    │     - Discretize state space for tabular Q-learning                 │
    │     - Epsilon-greedy exploration during training                    │
    │     - TD learning to update Q-values                                │
    │                                                                      │
    │  3. RL vs Delta Hedge:                                              │
    │     - RL can learn to hedge less when gamma is low                  │
    │     - RL can learn optimal rebalancing frequency                    │
    │     - RL typically reduces transaction costs                        │
    │                                                                      │
    │  4. Production Considerations:                                      │
    │     - Use DQN or policy gradient for continuous actions             │
    │     - Extensive backtesting required                                │
    │     - Risk limits and fallback to delta hedge                       │
    │                                                                      │
    │  NEXT: See 02_trading_agent.py for trading RL                       │
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
        # Run RL hedging workflow
        agent, rl_results, baseline_results = run_hedging_rl()
        
        # Visualization
        visualize_results(rl_results, baseline_results)
        
        # Summary
        print_summary()
        
        logger.info("Example completed successfully!")
        
    except Exception as e:
        logger.exception(f"Example failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Q-Learning Hedging Agent Example",
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

#!/usr/bin/env python3
"""
===============================================================================
Q-Learning: Training a Hedging Agent with QuantStrata Infrastructure
===============================================================================

This example demonstrates training an RL agent to hedge an option position
using QuantStrata's production q_learning module.

Learning Objectives
-------------------
1. **Environment Setup**: Use HedgingEnvironment with HedgingEnvConfig
2. **Agent Protocol**: Implement RLAgent interface for custom agents
3. **Training Pipeline**: Use run_training() with RLTrainingConfig
4. **Evaluation**: Compare RL agent vs delta hedging baseline

Mathematical Framework
----------------------
The agent learns to minimize hedging P&L variance:
    min_π Var[Σ_t (r_t)]
    
Where r_t is the reward at step t, derived from:
    - Option P&L: change in option value
    - Hedge P&L: position × price change
    - Transaction costs

State features: normalized spot, time, delta, gamma, position, P&L
Action: hedge ratio (fraction of delta to hedge)

Production Context
------------------
At a hedge fund:
- RL hedging can reduce transaction costs vs naive delta hedging
- Can learn optimal rebalancing frequency
- Adapts to different market regimes
- Requires careful validation against benchmarks

Prerequisites
-------------
- Basic RL concepts (state, action, reward)
- Option Greeks understanding
- Previous hedging examples

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
from src.q_learning.environments.hedging import HedgingEnvironment, HedgingEnvConfig
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
# Q-LEARNING AGENT (implements RLAgent protocol)
# =============================================================================

@dataclass
class QLearningAgentConfig:
    """Configuration for tabular Q-learning agent."""
    n_state_bins: int = 10  # Discretization bins per state dimension
    learning_rate: float = 0.1
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995


class TabularQLearningAgent:
    """
    Tabular Q-learning agent conforming to RLAgent protocol.
    
    Uses state discretization for tabular Q-values with epsilon-greedy
    exploration and TD(0) learning.
    
    Example:
        agent = TabularQLearningAgent(
            state_dim=6,
            n_actions=11,
            config=QLearningAgentConfig(),
        )
        
        action = agent.select_action(state, training=True, explore=True)
        metrics = agent.update(transitions=[transition])
    """
    
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        config: Optional[QLearningAgentConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize Q-learning agent.
        
        Parameters
        ----------
        state_dim : int
            Dimension of continuous state space.
        n_actions : int
            Number of discrete actions.
        config : QLearningAgentConfig, optional
            Agent configuration.
        seed : int, optional
            Random seed.
        """
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.config = config or QLearningAgentConfig()
        self._rng = np.random.default_rng(seed)
        
        # Discretization bins
        self.n_bins = self.config.n_state_bins
        
        # Q-table: (bins^state_dim) x n_actions
        # Use dictionary for sparse storage
        self._q_table: Dict[tuple, np.ndarray] = {}
        
        # Exploration rate
        self.epsilon = self.config.epsilon_start
        
        # State bounds (learned from data)
        self._state_min = np.full(state_dim, -2.0)
        self._state_max = np.full(state_dim, 2.0)
        
        # Training statistics
        self.n_updates = 0
    
    def _discretize_state(self, state: np.ndarray) -> tuple:
        """Convert continuous state to discrete bin indices."""
        # Clip to known bounds
        state = np.clip(state, self._state_min, self._state_max)
        
        # Normalize to [0, 1]
        normalized = (state - self._state_min) / (self._state_max - self._state_min + 1e-8)
        
        # Convert to bin indices
        bin_indices = (normalized * (self.n_bins - 1)).astype(int)
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
        
        return tuple(bin_indices)
    
    def _get_q_values(self, state_key: tuple) -> np.ndarray:
        """Get Q-values for discretized state (initialize if needed)."""
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
        """
        Select action using epsilon-greedy policy.
        
        Parameters
        ----------
        state : ndarray
            Current observation.
        training : bool
            Whether in training mode.
        explore : bool
            Whether to use exploration.
        
        Returns
        -------
        int
            Selected action index.
        """
        state = np.atleast_1d(state)
        
        # Update state bounds
        self._state_min = np.minimum(self._state_min, state)
        self._state_max = np.maximum(self._state_max, state)
        
        state_key = self._discretize_state(state)
        q_values = self._get_q_values(state_key)
        
        # Epsilon-greedy
        if explore and training and self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_actions))
        
        return int(np.argmax(q_values))
    
    def update(
        self,
        transitions: Optional[List[Any]] = None,
        batch: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, float]]:
        """
        Update Q-values from transitions.
        
        Parameters
        ----------
        transitions : list of Transition
            Experience tuples.
        batch : dict, optional
            Not used for tabular Q-learning.
        
        Returns
        -------
        dict
            Training metrics (td_error, epsilon).
        """
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
            
            # TD target
            if done:
                td_target = reward
            else:
                td_target = reward + self.config.gamma * np.max(next_q_values)
            
            # TD error
            td_error = td_target - q_values[action]
            total_td_error += abs(td_error)
            
            # Q-value update
            q_values[action] += self.config.learning_rate * td_error
        
        self.n_updates += 1
        
        # Decay epsilon
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay
        )
        
        return {
            "td_error": total_td_error / len(transitions),
            "epsilon": self.epsilon,
            "q_table_size": len(self._q_table),
        }
    
    def get_parameters(self) -> Dict[str, Any]:
        """Return agent parameters for checkpointing."""
        return {
            "q_table": {str(k): v.tolist() for k, v in self._q_table.items()},
            "epsilon": self.epsilon,
            "state_min": self._state_min.tolist(),
            "state_max": self._state_max.tolist(),
            "n_updates": self.n_updates,
        }
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Load agent parameters from checkpoint."""
        self._q_table = {
            eval(k): np.array(v) for k, v in params["q_table"].items()
        }
        self.epsilon = params["epsilon"]
        self._state_min = np.array(params["state_min"])
        self._state_max = np.array(params["state_max"])
        self.n_updates = params["n_updates"]


# =============================================================================
# DELTA HEDGE BASELINE AGENT
# =============================================================================

class DeltaHedgeAgent:
    """
    Baseline agent that always delta hedges (action = 1.0 hedge ratio).
    
    Conforms to RLAgent protocol for fair comparison.
    """
    
    def __init__(self, n_actions: int) -> None:
        """Initialize delta hedge agent."""
        self.n_actions = n_actions
        # Action that corresponds to hedge ratio = 1.0
        self.delta_action = n_actions // 2  # Middle action (assuming symmetric)
    
    def select_action(
        self,
        state: Any,
        *,
        training: bool = False,
        explore: bool = True,
    ) -> int:
        """Always return delta hedge action."""
        return self.delta_action
    
    def update(
        self,
        transitions: Optional[List[Any]] = None,
        batch: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, float]]:
        """No learning needed."""
        return None
    
    def get_parameters(self) -> Dict[str, Any]:
        return {"delta_action": self.delta_action}
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        self.delta_action = params["delta_action"]


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_agent(
    agent: RLAgent,
    env: HedgingEnvironment,
    n_episodes: int = 100,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate agent performance over multiple episodes.
    
    Parameters
    ----------
    agent : RLAgent
        Agent to evaluate.
    env : HedgingEnvironment
        Environment.
    n_episodes : int
        Number of evaluation episodes.
    seed : int, optional
        Random seed.
    
    Returns
    -------
    dict
        Evaluation metrics.
    """
    returns = []
    final_pnls = []
    total_costs = []
    
    for ep in range(n_episodes):
        state, info = env.reset(seed=seed + ep if seed else None)
        episode_return = 0.0
        
        while True:
            action = agent.select_action(state, training=False, explore=False)
            state, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            
            if terminated or truncated:
                break
        
        returns.append(episode_return)
        final_pnls.append(info["pnl"])
        total_costs.append(info["cumulative_cost"])
    
    returns_arr = np.array(returns)
    pnls_arr = np.array(final_pnls)
    costs_arr = np.array(total_costs)
    
    return {
        "mean_return": float(np.mean(returns_arr)),
        "std_return": float(np.std(returns_arr)),
        "mean_pnl": float(np.mean(pnls_arr)),
        "std_pnl": float(np.std(pnls_arr)),
        "mean_cost": float(np.mean(costs_arr)),
        "sharpe": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(list(np.cumsum(returns))),
        "win_rate": win_rate(returns),
    }


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def run_hedging_rl() -> Tuple[RLTrainingResult, Dict[str, Dict[str, float]]]:
    """
    Run the complete hedging RL workflow.
    
    Returns
    -------
    Tuple
        Training result and evaluation metrics.
    """
    logger.info("=" * 70)
    logger.info("SECTION 1: Environment Setup")
    logger.info("=" * 70)
    
    # Configure environment
    env_config = HedgingEnvConfig(
        spot=100.0,
        strike=100.0,
        maturity=0.25,  # 3 months
        volatility=0.2,
        risk_free_rate=0.05,
        n_steps=50,
        proportional_cost=0.001,  # 10 bps
        action_type="discrete",
        n_discrete_actions=11,  # -1 to +1 in 0.2 increments
        max_hedge_ratio=1.5,
        reward_type="risk_adjusted",
        risk_aversion=0.1,
        include_delta=True,
        include_gamma=True,
        include_position=True,
        include_pnl=True,
    )
    
    env = HedgingEnvironment(config=env_config, seed=42)
    
    logger.info("")
    logger.info(f"  Spot:        {env_config.spot}")
    logger.info(f"  Strike:      {env_config.strike}")
    logger.info(f"  Maturity:    {env_config.maturity}y")
    logger.info(f"  Vol:         {env_config.volatility:.0%}")
    logger.info(f"  Steps:       {env_config.n_steps}")
    logger.info(f"  Actions:     {env_config.n_discrete_actions}")
    logger.info(f"  Txn cost:    {env_config.proportional_cost:.1%}")
    logger.info(f"  State dim:   {env.observation_space_dim}")
    
    # Create agents
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 2: Agent Creation")
    logger.info("=" * 70)
    
    agent_config = QLearningAgentConfig(
        n_state_bins=8,
        learning_rate=0.1,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
    )
    
    rl_agent = TabularQLearningAgent(
        state_dim=env.observation_space_dim,
        n_actions=env_config.n_discrete_actions,
        config=agent_config,
        seed=42,
    )
    
    baseline_agent = DeltaHedgeAgent(n_actions=env_config.n_discrete_actions)
    
    logger.info("")
    logger.info("  RL Agent:       TabularQLearningAgent")
    logger.info(f"    State bins:   {agent_config.n_state_bins}")
    logger.info(f"    Learning rate: {agent_config.learning_rate}")
    logger.info(f"    Gamma:        {agent_config.gamma}")
    logger.info(f"    Epsilon:      {agent_config.epsilon_start} → {agent_config.epsilon_end}")
    logger.info("  Baseline:       DeltaHedgeAgent")
    
    # Training
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 3: Training")
    logger.info("=" * 70)
    
    training_config = RLTrainingConfig(
        n_episodes=500,
        max_steps_per_episode=env_config.n_steps,
        gamma=agent_config.gamma,
        log_every=100,
        eval_episodes=10,
        verbose=1,
    )
    
    logger.info("")
    logger.info(f"  Episodes:    {training_config.n_episodes}")
    logger.info(f"  Eval every:  {training_config.log_every}")
    logger.info("")
    
    # Run training using QuantStrata pipeline
    training_result = run_training(
        agent=rl_agent,
        env=env,
        config=training_config,
    )
    
    logger.info("")
    logger.info(f"  Training time: {training_result.training_time_seconds:.1f}s")
    logger.info(f"  Best episode:  {training_result.best_episode}")
    logger.info(f"  Best return:   {training_result.best_episode_return:.4f}")
    
    # Evaluation
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 4: Evaluation")
    logger.info("=" * 70)
    
    eval_env = HedgingEnvironment(config=env_config, seed=123)
    
    logger.info("")
    logger.info("Evaluating agents over 200 episodes...")
    
    rl_metrics = evaluate_agent(rl_agent, eval_env, n_episodes=200, seed=1000)
    baseline_metrics = evaluate_agent(baseline_agent, eval_env, n_episodes=200, seed=1000)
    
    logger.info("")
    logger.info(f"{'Metric':<20} {'RL Agent':>12} {'Delta Hedge':>12}")
    logger.info("-" * 50)
    logger.info(f"{'Mean Return':<20} {rl_metrics['mean_return']:>12.4f} {baseline_metrics['mean_return']:>12.4f}")
    logger.info(f"{'Std Return':<20} {rl_metrics['std_return']:>12.4f} {baseline_metrics['std_return']:>12.4f}")
    logger.info(f"{'Mean P&L':<20} {rl_metrics['mean_pnl']:>12.2f} {baseline_metrics['mean_pnl']:>12.2f}")
    logger.info(f"{'Std P&L':<20} {rl_metrics['std_pnl']:>12.2f} {baseline_metrics['std_pnl']:>12.2f}")
    logger.info(f"{'Mean Cost':<20} {rl_metrics['mean_cost']:>12.2f} {baseline_metrics['mean_cost']:>12.2f}")
    logger.info(f"{'Sharpe':<20} {rl_metrics['sharpe']:>12.4f} {baseline_metrics['sharpe']:>12.4f}")
    logger.info(f"{'Win Rate':<20} {rl_metrics['win_rate']:>12.1%} {baseline_metrics['win_rate']:>12.1%}")
    logger.info("-" * 50)
    
    # Calculate improvement
    pnl_std_improvement = (baseline_metrics['std_pnl'] - rl_metrics['std_pnl']) / baseline_metrics['std_pnl'] * 100
    cost_reduction = (baseline_metrics['mean_cost'] - rl_metrics['mean_cost']) / baseline_metrics['mean_cost'] * 100
    
    logger.info("")
    logger.info(f"  P&L Std Improvement: {pnl_std_improvement:+.1f}%")
    logger.info(f"  Cost Reduction:      {cost_reduction:+.1f}%")
    
    evaluation_results = {
        "rl_agent": rl_metrics,
        "delta_hedge": baseline_metrics,
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
        logger.info("Skipping plots (matplotlib not available or disabled)")
        return
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 5: Visualization")
    logger.info("=" * 70)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # -------------------------------------------------------------------------
    # Plot 1: Training returns
    # -------------------------------------------------------------------------
    ax = axes[0, 0]
    
    returns = training_result.episode_returns
    window = 50
    smoothed = np.convolve(returns, np.ones(window) / window, mode='valid')
    
    ax.plot(returns, alpha=0.3, color='#2E86AB', label='Episode Return')
    ax.plot(range(window - 1, len(returns)), smoothed, color='#E94F37', 
            linewidth=2, label=f'Moving Avg ({window})')
    ax.axvline(training_result.best_episode, color='green', linestyle='--',
               alpha=0.7, label=f'Best ({training_result.best_episode})')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Return')
    ax.set_title('Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 2: Epsilon decay
    # -------------------------------------------------------------------------
    ax = axes[0, 1]
    
    if 'epsilon' in training_result.history:
        ax.plot(training_result.history['epsilon'], color='#4CAF50', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Epsilon')
        ax.set_title('Exploration Rate Decay')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Epsilon history not available', 
                ha='center', va='center', transform=ax.transAxes)
    
    # -------------------------------------------------------------------------
    # Plot 3: Metric comparison
    # -------------------------------------------------------------------------
    ax = axes[1, 0]
    
    metrics = ['mean_pnl', 'std_pnl', 'mean_cost']
    labels = ['Mean P&L', 'Std P&L', 'Mean Cost']
    
    x = np.arange(len(metrics))
    width = 0.35
    
    rl_vals = [evaluation_results['rl_agent'][m] for m in metrics]
    baseline_vals = [evaluation_results['delta_hedge'][m] for m in metrics]
    
    ax.bar(x - width/2, rl_vals, width, label='RL Agent', color='#2E86AB', alpha=0.8)
    ax.bar(x + width/2, baseline_vals, width, label='Delta Hedge', color='#E94F37', alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Value')
    ax.set_title('Performance Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # -------------------------------------------------------------------------
    # Plot 4: Risk-return trade-off
    # -------------------------------------------------------------------------
    ax = axes[1, 1]
    
    rl = evaluation_results['rl_agent']
    bl = evaluation_results['delta_hedge']
    
    ax.scatter(rl['std_pnl'], rl['mean_pnl'], s=200, c='#2E86AB', 
               label='RL Agent', marker='o', zorder=5)
    ax.scatter(bl['std_pnl'], bl['mean_pnl'], s=200, c='#E94F37',
               label='Delta Hedge', marker='s', zorder=5)
    
    ax.set_xlabel('P&L Std (Risk)')
    ax.set_ylabel('Mean P&L (Return)')
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
    │  1. QuantStrata Q-Learning Infrastructure:                          │
    │     - HedgingEnvironment: Gymnasium-compatible hedging sim          │
    │     - RLAgent protocol: Standard interface for agents               │
    │     - run_training(): Generic training loop                         │
    │     - Evaluation metrics: Sharpe, drawdown, win rate                │
    │                                                                      │
    │  2. Tabular Q-Learning Agent:                                       │
    │     - State discretization for tabular Q-values                     │
    │     - Epsilon-greedy exploration                                    │
    │     - TD(0) learning updates                                        │
    │     - Checkpoint support via get/set_parameters()                   │
    │                                                                      │
    │  3. Training Pipeline:                                              │
    │     - RLTrainingConfig: episodes, eval frequency, etc.              │
    │     - RLTrainingResult: returns, history, best episode              │
    │     - Automatic logging and evaluation                              │
    │                                                                      │
    │  4. Production Considerations:                                      │
    │     - Compare to delta hedge baseline                               │
    │     - Monitor P&L variance reduction                                │
    │     - Track transaction cost savings                                │
    │     - Validate on out-of-sample data                                │
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
        # Run hedging RL workflow
        training_result, evaluation_results = run_hedging_rl()
        
        # Visualization
        visualize_results(training_result, evaluation_results)
        
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

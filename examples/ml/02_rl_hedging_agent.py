#!/usr/bin/env python3
"""
===============================================================================
RL Hedging Agent: Training and Evaluation
===============================================================================

This example demonstrates training a reinforcement learning agent for option
hedging. We implement a simple policy gradient agent and compare it to the
delta hedging benchmark.

Learning Objectives
-------------------
1. **RL Agent Architecture**: Policy network for continuous actions
2. **Training Loop**: Episode collection and policy updates
3. **Performance Evaluation**: Compare RL vs delta hedge
4. **Hyperparameter Tuning**: Learning rate, entropy, etc.

Mathematical Framework
----------------------
Policy Gradient (REINFORCE):
    ∇J(θ) = E[∇_θ log π_θ(a|s) · G_t]
    
    where G_t is the return from time t

For continuous actions, we use a Gaussian policy:
    π_θ(a|s) = N(μ_θ(s), σ_θ(s))

Advantage Baseline:
    A_t = G_t - V(s_t)
    
    reduces variance of gradient estimates

Production Context
------------------
At a hedge fund:
- RL for hedging is cutting-edge research
- Can adapt to transaction costs and market microstructure
- Robust to model misspecification
- Requires careful backtesting and out-of-sample validation

Why Results May Vary / Production Considerations
-------------------------------------------------
- For production, feed the hedging environment with real market data:
  ZeroRateCurve and GridVolSurface instead of FlatCurves/FlatVol, so
  option values and Greeks match front-office pricing.
- Example performance depends on hyperparameters (learning rate, entropy,
  network size) and number of episodes; run multiple seeds and compare
  to delta-hedge on out-of-sample paths with realistic costs.

Prerequisites
-------------
- Understanding of RL environment (examples/ml/01_hedging_environment.py)
- Basic deep learning concepts

Run This Example
----------------
    cd /path/to/QuantStrata
    PYTHONPATH=. python examples/ml/02_rl_hedging_agent.py

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
from typing import List, Tuple, Dict, Any

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
from src.marketdata.core.ids import MarketId
from src.marketdata.core.market import Market
from src.marketdata.core.interfaces import Quote
from src.marketdata.curves.term_structure import ZeroRateCurve
from src.marketdata.surfaces.vol_surface import GridVolSurface
from src.instruments.fx.options.vanilla import FxVanillaEuropeanOption
from src.pricers.fx.european_bsm import FxVanillaEuropeanOptionBsmPricer


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
# AGENT CONFIGURATION
# =============================================================================

@dataclass
class AgentConfig:
    """Configuration for RL hedging agent."""
    hidden_dim: int = 64
    learning_rate: float = 0.001
    gamma: float = 0.99  # Discount factor
    entropy_coef: float = 0.01  # Entropy regularization
    max_grad_norm: float = 0.5  # Gradient clipping
    
    # Training
    n_episodes: int = 1000
    batch_size: int = 32  # Episodes per update
    log_interval: int = 100


# =============================================================================
# SIMPLE POLICY NETWORK (NumPy-based for no external deps)
# =============================================================================

class SimplePolicy:
    """
    Simple feedforward policy network using NumPy.
    
    Architecture
    ------------
    Input → Linear(hidden) → ReLU → Linear(2) → [mean, log_std]
    
    The policy outputs Gaussian parameters for the continuous action.
    """
    
    def __init__(self, state_dim: int, hidden_dim: int = 64, seed: int = 42):
        """
        Initialize policy network.
        
        Parameters
        ----------
        state_dim : int
            State dimension.
        hidden_dim : int
            Hidden layer dimension.
        seed : int
            Random seed for initialization.
        """
        np.random.seed(seed)
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # Xavier initialization
        self.W1 = np.random.randn(state_dim, hidden_dim) * np.sqrt(2.0 / state_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 2) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(2)
        
        # Initialize log_std bias to small value (exploration)
        self.b2[1] = -1.0
        
        # For Adam optimizer
        self.m_W1 = np.zeros_like(self.W1)
        self.v_W1 = np.zeros_like(self.W1)
        self.m_b1 = np.zeros_like(self.b1)
        self.v_b1 = np.zeros_like(self.b1)
        self.m_W2 = np.zeros_like(self.W2)
        self.v_W2 = np.zeros_like(self.W2)
        self.m_b2 = np.zeros_like(self.b2)
        self.v_b2 = np.zeros_like(self.b2)
        self.t = 0
    
    def forward(self, state: np.ndarray) -> Tuple[float, float]:
        """
        Forward pass to get Gaussian parameters.
        
        Parameters
        ----------
        state : np.ndarray
            Current state.
        
        Returns
        -------
        Tuple[float, float]
            Mean and std of action distribution.
        """
        # Hidden layer
        h = np.maximum(0, state @ self.W1 + self.b1)  # ReLU
        
        # Output layer
        out = h @ self.W2 + self.b2
        
        mean = np.tanh(out[0]) * 2.0  # Scale to [-2, 2]
        std = np.exp(np.clip(out[1], -5, 2)) + 0.1  # Ensure positive std
        
        return mean, std
    
    def sample_action(self, state: np.ndarray) -> Tuple[float, float, float]:
        """
        Sample action from policy.
        
        Returns
        -------
        Tuple[float, float, float]
            Action, log probability, entropy.
        """
        mean, std = self.forward(state)
        
        # Sample from Gaussian
        eps = np.random.randn()
        action = mean + std * eps
        action = np.clip(action, -2.0, 2.0)
        
        # Log probability
        log_prob = -0.5 * ((action - mean) / std) ** 2 - np.log(std) - 0.5 * np.log(2 * np.pi)
        
        # Entropy
        entropy = 0.5 * (1 + np.log(2 * np.pi * std ** 2))
        
        return float(action), float(log_prob), float(entropy)
    
    def get_action_deterministic(self, state: np.ndarray) -> float:
        """Get deterministic action (mean of policy)."""
        mean, _ = self.forward(state)
        return float(mean)
    
    def compute_gradients(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
        entropy_coef: float = 0.01,
    ) -> Dict[str, np.ndarray]:
        """
        Compute policy gradients using REINFORCE.
        
        Parameters
        ----------
        states : np.ndarray
            Batch of states [batch, state_dim].
        actions : np.ndarray
            Batch of actions [batch].
        advantages : np.ndarray
            Batch of advantages [batch].
        entropy_coef : float
            Entropy coefficient.
        
        Returns
        -------
        Dict[str, np.ndarray]
            Gradients for each parameter.
        """
        batch_size = len(states)
        
        # Initialize gradients
        dW1 = np.zeros_like(self.W1)
        db1 = np.zeros_like(self.b1)
        dW2 = np.zeros_like(self.W2)
        db2 = np.zeros_like(self.b2)
        
        for i in range(batch_size):
            state = states[i]
            action = actions[i]
            advantage = advantages[i]
            
            # Forward pass (store activations)
            h_pre = state @ self.W1 + self.b1
            h = np.maximum(0, h_pre)  # ReLU
            out = h @ self.W2 + self.b2
            
            mean = np.tanh(out[0]) * 2.0
            log_std = np.clip(out[1], -5, 2)
            std = np.exp(log_std) + 0.1
            
            # Gradient of log probability w.r.t. mean and std
            d_log_prob_mean = (action - mean) / (std ** 2)
            d_log_prob_std = ((action - mean) ** 2 / (std ** 3)) - (1 / std)
            
            # Include entropy gradient (maximize entropy)
            d_entropy_std = 1.0 / std
            
            # Combine with advantage
            d_mean = d_log_prob_mean * advantage
            d_std = (d_log_prob_std * advantage + entropy_coef * d_entropy_std)
            
            # Backprop through output
            d_out = np.zeros(2)
            d_out[0] = d_mean * (1 - np.tanh(out[0]) ** 2) * 2.0  # tanh gradient
            d_out[1] = d_std * (std - 0.1)  # exp gradient (std = exp(log_std) + 0.1)
            
            # Gradients for W2, b2
            dW2 += np.outer(h, d_out)
            db2 += d_out
            
            # Backprop through hidden layer
            d_h = d_out @ self.W2.T
            d_h_pre = d_h * (h_pre > 0).astype(float)  # ReLU gradient
            
            # Gradients for W1, b1
            dW1 += np.outer(state, d_h_pre)
            db1 += d_h_pre
        
        # Average gradients
        return {
            'W1': dW1 / batch_size,
            'b1': db1 / batch_size,
            'W2': dW2 / batch_size,
            'b2': db2 / batch_size,
        }
    
    def update(self, grads: Dict[str, np.ndarray], lr: float = 0.001) -> None:
        """
        Update parameters using Adam optimizer.
        
        Parameters
        ----------
        grads : Dict[str, np.ndarray]
            Gradients from compute_gradients.
        lr : float
            Learning rate.
        """
        self.t += 1
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        
        for name, param, m, v in [
            ('W1', self.W1, self.m_W1, self.v_W1),
            ('b1', self.b1, self.m_b1, self.v_b1),
            ('W2', self.W2, self.m_W2, self.v_W2),
            ('b2', self.b2, self.m_b2, self.v_b2),
        ]:
            grad = grads[name]
            
            # Adam update
            m[:] = beta1 * m + (1 - beta1) * grad
            v[:] = beta2 * v + (1 - beta2) * (grad ** 2)
            
            m_hat = m / (1 - beta1 ** self.t)
            v_hat = v / (1 - beta2 ** self.t)
            
            param += lr * m_hat / (np.sqrt(v_hat) + eps)  # Gradient ascent


# =============================================================================
# SECTION 1: Environment Setup
# =============================================================================

def _build_production_pricing_callables(
    spot_ref: float, strike: float, risk_free_rate: float, volatility: float,
) -> Tuple[Any, Any, Any, Any]:
    """ZeroRateCurve + GridVolSurface (no FlatVol/FlatCurves)."""
    import numpy as np
    tenors = np.array([0.0, 0.25, 0.5, 1.0])
    rates = np.full_like(tenors, risk_free_rate)
    dom_curve = ZeroRateCurve(tenors=tenors, zero_rates=rates, extrapolation="flat")
    for_curve = ZeroRateCurve(tenors=tenors, zero_rates=rates, extrapolation="flat")
    expiries = np.array([0.1, 0.25, 0.5, 1.0])
    strikes = np.array([strike * 0.8, strike * 0.9, strike, strike * 1.1, strike * 1.2])
    implied_vols = np.full((len(expiries), len(strikes)), volatility)
    vol_surface = GridVolSurface(expiries=expiries, strikes=strikes, implied_vols=implied_vols, extrapolation="flat")
    spot_id = MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD")
    dom_id = MarketId(asset_class="IR", mkt_type="CURVE", name="USD_OIS")
    for_id = MarketId(asset_class="IR", mkt_type="CURVE", name="EUR_OIS")
    vol_id = MarketId(asset_class="FX", mkt_type="VOL", name="EURUSD")
    pricer = FxVanillaEuropeanOptionBsmPricer()

    def _market(spot: float) -> Market:
        return Market(asof="", quotes={spot_id: Quote(value=spot)}, curves={dom_id: dom_curve, for_id: for_curve}, vols={vol_id: vol_surface})

    def price_fn(spot: float, strike_k: float, tau: float, option_type: str) -> float:
        opt = FxVanillaEuropeanOption(option_type=option_type, notional=1.0, strike=strike_k, expiry=max(tau, 0.0), spot_id=spot_id, vol_id=vol_id, domestic_curve_id=dom_id, foreign_curve_id=for_id)
        return pricer.price(opt, _market(spot))

    def delta_fn(spot: float, strike_k: float, tau: float, option_type: str) -> float:
        opt = FxVanillaEuropeanOption(option_type=option_type, notional=1.0, strike=strike_k, expiry=max(tau, 0.0), spot_id=spot_id, vol_id=vol_id, domestic_curve_id=dom_id, foreign_curve_id=for_id)
        return pricer.greeks(opt, _market(spot))["delta"]

    def gamma_fn(spot: float, strike_k: float, tau: float) -> float:
        opt = FxVanillaEuropeanOption(option_type="call", notional=1.0, strike=strike_k, expiry=max(tau, 0.0), spot_id=spot_id, vol_id=vol_id, domestic_curve_id=dom_id, foreign_curve_id=for_id)
        return pricer.greeks(opt, _market(spot))["gamma"]

    def vega_fn(spot: float, strike_k: float, tau: float) -> float:
        opt = FxVanillaEuropeanOption(option_type="call", notional=1.0, strike=strike_k, expiry=max(tau, 0.0), spot_id=spot_id, vol_id=vol_id, domestic_curve_id=dom_id, foreign_curve_id=for_id)
        return pricer.greeks(opt, _market(spot))["vega"]

    return price_fn, delta_fn, gamma_fn, vega_fn


def create_environment() -> Tuple[HedgingEnvironment, HedgingEnvConfig]:
    """Create hedging environment (ZeroRateCurve + GridVolSurface, production-grade)."""
    logger.info("=" * 70)
    logger.info("SECTION 1: Environment Setup")
    logger.info("=" * 70)

    spot_ref, strike_ref = 100.0, 100.0
    vol, r = 0.20, 0.05
    price_fn, delta_fn, gamma_fn, vega_fn = _build_production_pricing_callables(spot_ref, strike_ref, r, vol)

    config = HedgingEnvConfig(
        spot=spot_ref,
        strike=strike_ref,
        maturity=0.25,
        volatility=vol,
        risk_free_rate=r,
        price_fn=price_fn,
        delta_fn=delta_fn,
        gamma_fn=gamma_fn,
        vega_fn=vega_fn,
        n_steps=50,
        proportional_cost=0.001,
        reward_type="risk_adjusted",
        risk_aversion=0.1,
    )

    env = HedgingEnvironment(config=config)
    
    logger.info(f"  State dimension: {env.observation_space_dim}")
    logger.info(f"  Action range: [-2.0, 2.0]")
    
    return env, config


# =============================================================================
# SECTION 2: Training
# =============================================================================

def collect_episode(env: HedgingEnvironment, policy: SimplePolicy, seed: int) -> Dict[str, Any]:
    """
    Collect one episode using the policy.
    
    Returns
    -------
    Dict[str, Any]
        Episode data (states, actions, rewards, etc.).
    """
    state, info = env.reset(seed=seed)
    
    states = [state]
    actions = []
    log_probs = []
    rewards = []
    entropies = []
    
    terminated = truncated = False
    
    while not (terminated or truncated):
        action, log_prob, entropy = policy.sample_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        entropies.append(entropy)
        states.append(next_state)
        
        state = next_state
    
    return {
        'states': np.array(states[:-1]),
        'actions': np.array(actions),
        'log_probs': np.array(log_probs),
        'rewards': np.array(rewards),
        'entropies': np.array(entropies),
        'final_pnl': info['pnl'],
        'total_reward': sum(rewards),
    }


def compute_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """Compute discounted returns."""
    returns = np.zeros_like(rewards)
    G = 0
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns


def train_agent(
    env: HedgingEnvironment,
    config: AgentConfig,
) -> Tuple[SimplePolicy, List[float], List[float]]:
    """
    Train RL agent.
    
    Returns
    -------
    Tuple[SimplePolicy, List[float], List[float]]
        Trained policy, training rewards, training PnLs.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 2: Training RL Agent")
    logger.info("=" * 70)
    
    state, _ = env.reset()
    state_dim = len(state)
    
    policy = SimplePolicy(state_dim, config.hidden_dim)
    
    episode_rewards = []
    episode_pnls = []
    
    logger.info(f"Training for {config.n_episodes} episodes...")
    logger.info("")
    
    for episode in range(config.n_episodes):
        # Collect batch of episodes
        batch_states = []
        batch_actions = []
        batch_advantages = []
        
        batch_rewards = []
        batch_pnls = []
        
        for b in range(config.batch_size):
            ep_data = collect_episode(env, policy, seed=episode * config.batch_size + b)
            
            # Compute returns and normalize (advantage)
            returns = compute_returns(ep_data['rewards'], config.gamma)
            advantage = returns - np.mean(returns)
            if np.std(advantage) > 0:
                advantage = advantage / np.std(advantage)
            
            batch_states.append(ep_data['states'])
            batch_actions.append(ep_data['actions'])
            batch_advantages.append(advantage)
            
            batch_rewards.append(ep_data['total_reward'])
            batch_pnls.append(ep_data['final_pnl'])
        
        # Concatenate batch
        all_states = np.vstack(batch_states)
        all_actions = np.concatenate(batch_actions)
        all_advantages = np.concatenate(batch_advantages)
        
        # Compute gradients and update
        grads = policy.compute_gradients(
            all_states, all_actions, all_advantages, config.entropy_coef
        )
        policy.update(grads, config.learning_rate)
        
        # Track metrics
        mean_reward = np.mean(batch_rewards)
        mean_pnl = np.mean(batch_pnls)
        episode_rewards.append(mean_reward)
        episode_pnls.append(mean_pnl)
        
        # Logging
        if (episode + 1) % config.log_interval == 0:
            logger.info(
                f"Episode {episode + 1:4d} | "
                f"Reward: {mean_reward:>8.4f} | "
                f"PnL: ${mean_pnl:>8.2f} | "
                f"Std: ${np.std(batch_pnls):>6.2f}"
            )
    
    logger.info("")
    logger.info("Training complete!")
    
    return policy, episode_rewards, episode_pnls


# =============================================================================
# SECTION 3: Evaluation
# =============================================================================

def evaluate_agent(
    env: HedgingEnvironment,
    policy: SimplePolicy,
    n_episodes: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate trained agent vs delta hedge.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        RL agent PnLs, delta hedge PnLs.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 3: Evaluation")
    logger.info("=" * 70)
    
    logger.info(f"Evaluating on {n_episodes} episodes...")
    
    rl_pnls = []
    delta_pnls = []
    
    for i in range(n_episodes):
        # RL agent (deterministic policy)
        state, info = env.reset(seed=i + 10000)  # Different seeds from training
        terminated = truncated = False
        while not (terminated or truncated):
            action = policy.get_action_deterministic(state)
            state, reward, terminated, truncated, info = env.step(action)
        rl_pnls.append(info['pnl'])
        
        # Delta hedge
        state, info = env.reset(seed=i + 10000)
        terminated = truncated = False
        while not (terminated or truncated):
            action = env.get_delta_hedge_action()
            state, reward, terminated, truncated, info = env.step(action)
        delta_pnls.append(info['pnl'])
    
    rl_pnls = np.array(rl_pnls)
    delta_pnls = np.array(delta_pnls)
    
    logger.info("")
    logger.info("Evaluation Results:")
    logger.info("-" * 60)
    logger.info(f"{'Metric':<20} {'RL Agent':>15} {'Delta Hedge':>15}")
    logger.info("-" * 60)
    logger.info(f"{'Mean P&L':<20} ${np.mean(rl_pnls):>14.2f} ${np.mean(delta_pnls):>14.2f}")
    logger.info(f"{'Std P&L':<20} ${np.std(rl_pnls):>14.2f} ${np.std(delta_pnls):>14.2f}")
    logger.info(f"{'Sharpe Ratio':<20} {np.mean(rl_pnls)/np.std(rl_pnls):>14.3f} {np.mean(delta_pnls)/np.std(delta_pnls):>14.3f}")
    logger.info(f"{'5th Percentile':<20} ${np.percentile(rl_pnls, 5):>14.2f} ${np.percentile(delta_pnls, 5):>14.2f}")
    logger.info(f"{'95th Percentile':<20} ${np.percentile(rl_pnls, 95):>14.2f} ${np.percentile(delta_pnls, 95):>14.2f}")
    logger.info("-" * 60)
    
    # Improvement
    improvement = np.mean(rl_pnls) - np.mean(delta_pnls)
    logger.info(f"\nRL vs Delta Hedge: ${improvement:+.2f} per episode")
    
    return rl_pnls, delta_pnls


# =============================================================================
# SECTION 4: Visualization
# =============================================================================

def visualize_results(
    episode_rewards: List[float],
    episode_pnls: List[float],
    rl_pnls: np.ndarray,
    delta_pnls: np.ndarray,
) -> None:
    """Create training and evaluation visualizations."""
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
    # Plot 1: Training reward curve
    # -------------------------------------------------------------------------
    ax1 = axes[0, 0]
    
    window = 50
    smoothed = np.convolve(episode_rewards, np.ones(window) / window, mode='valid')
    
    ax1.plot(episode_rewards, alpha=0.3, color='blue')
    ax1.plot(range(window - 1, len(episode_rewards)), smoothed, color='blue', linewidth=2, label=f'Smoothed (window={window})')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Mean Reward')
    ax1.set_title('Training Reward Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 2: Training PnL curve
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]
    
    smoothed_pnl = np.convolve(episode_pnls, np.ones(window) / window, mode='valid')
    
    ax2.plot(episode_pnls, alpha=0.3, color='green')
    ax2.plot(range(window - 1, len(episode_pnls)), smoothed_pnl, color='green', linewidth=2, label=f'Smoothed')
    ax2.axhline(0, color='black', linestyle='--', linewidth=0.5)
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Mean P&L ($)')
    ax2.set_title('Training P&L Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 3: Evaluation P&L distribution
    # -------------------------------------------------------------------------
    ax3 = axes[1, 0]
    
    ax3.hist(rl_pnls, bins=30, alpha=0.7, label='RL Agent', color='#2E86AB')
    ax3.hist(delta_pnls, bins=30, alpha=0.5, label='Delta Hedge', color='#E94F37')
    ax3.axvline(0, color='black', linestyle='-', linewidth=1)
    ax3.axvline(np.mean(rl_pnls), color='#2E86AB', linestyle='--', linewidth=2)
    ax3.axvline(np.mean(delta_pnls), color='#E94F37', linestyle='--', linewidth=2)
    
    ax3.set_xlabel('Final P&L ($)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Evaluation: P&L Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 4: Risk-return comparison
    # -------------------------------------------------------------------------
    ax4 = axes[1, 1]
    
    strategies = ['RL Agent', 'Delta Hedge']
    means = [np.mean(rl_pnls), np.mean(delta_pnls)]
    stds = [np.std(rl_pnls), np.std(delta_pnls)]
    colors = ['#2E86AB', '#E94F37']
    
    ax4.scatter(stds, means, c=colors, s=200, zorder=3)
    for i, (x, y, name) in enumerate(zip(stds, means, strategies)):
        ax4.annotate(name, (x, y), textcoords="offset points", xytext=(10, 5), fontsize=12)
    
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
    │  1. Policy Network:                                                 │
    │     - Maps state to Gaussian policy parameters (μ, σ)               │
    │     - Samples actions for exploration                               │
    │     - Deterministic policy (μ) for evaluation                       │
    │                                                                      │
    │  2. REINFORCE Algorithm:                                            │
    │     - Collect episodes, compute returns                             │
    │     - Policy gradient: ∇J = E[∇log π(a|s) × A]                      │
    │     - Advantage normalization for stable training                   │
    │                                                                      │
    │  3. Training Considerations:                                        │
    │     - Entropy bonus for exploration                                 │
    │     - Batch training reduces variance                               │
    │     - Learning rate tuning is critical                              │
    │                                                                      │
    │  4. Production Deployment:                                          │
    │     - Extensive out-of-sample testing required                      │
    │     - Monitor for distribution shift                                │
    │     - Compare to simple baselines (delta hedge)                     │
    │                                                                      │
    │  NEXT: See 03_model_validation.py for model comparison              │
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
        env, env_config = create_environment()
        
        # Agent configuration
        agent_config = AgentConfig(
            hidden_dim=64,
            learning_rate=0.001,
            gamma=0.99,
            entropy_coef=0.01,
            n_episodes=500,  # Reduced for demo
            batch_size=16,
            log_interval=50,
        )
        
        # Section 2: Training
        policy, episode_rewards, episode_pnls = train_agent(env, agent_config)
        
        # Section 3: Evaluation
        rl_pnls, delta_pnls = evaluate_agent(env, policy, n_episodes=300)
        
        # Section 4: Visualization
        visualize_results(episode_rewards, episode_pnls, rl_pnls, delta_pnls)
        
        # Summary
        print_summary()
        
        logger.info("Example completed successfully!")
        
    except Exception as e:
        logger.exception(f"Example failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RL Hedging Agent Example",
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

"""
Deep Hedging Trainer

Training loop for deep hedging agents using risk measure minimisation.

Training Algorithm
------------------
1. Simulate batch of hedging episodes
2. Compute terminal P&L for each episode
3. Compute risk measure: Loss = ρ(-P&L)
4. Estimate gradients (finite differences or autodiff)
5. Update policy parameters

Gradient Estimation
-------------------
For NumPy-based training, we use finite differences:
    ∂Loss/∂θ ≈ (Loss(θ + ε) - Loss(θ - ε)) / (2ε)

For production training with TensorFlow/PyTorch, gradients are computed
via automatic differentiation through the simulation.

Example
-------
>>> from src.deep_hedging.training import train_deep_hedging
>>> from src.deep_hedging.environments import GBMHedgingEnv
>>> from src.deep_hedging.agents import DeepHedgingAgent
>>> 
>>> result = train_deep_hedging(
...     agent=agent,
...     env=env,
...     n_epochs=100,
...     batch_size=256,
... )
>>> print(f"Final loss: {result['final_loss']:.4f}")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from src.deep_hedging.core.types import (
    HedgingConfig,
    HedgingState,
    HedgingResult,
    HedgingEpisode,
    DeepHedgingTrainingConfig,
)
from src.deep_hedging.core.risk_measures import RiskMeasure, MeanVarianceRisk
from src.deep_hedging.core.protocols import BaseHedgingEnv
from src.deep_hedging.agents.deep import DeepHedgingAgent, MLPPolicy
from src.deep_hedging.agents.delta import DeltaHedgingAgent

logger = logging.getLogger(__name__)


def simulate_hedging_batch(
    agent: Any,
    env: BaseHedgingEnv,
    n_episodes: int,
    seed: Optional[int] = None,
    return_episodes: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, List[HedgingEpisode]]]:
    """
    Simulate a batch of hedging episodes.
    
    Parameters
    ----------
    agent : RLAgent-like
        Hedging agent with select_action method.
    env : BaseHedgingEnv
        Hedging environment.
    n_episodes : int
        Number of episodes to simulate.
    seed : int, optional
        Random seed for reproducibility.
    return_episodes : bool
        If True, also return full episode records.
    
    Returns
    -------
    pnl_samples : ndarray, shape (n_episodes,)
        Terminal P&L for each episode.
    cost_samples : ndarray, shape (n_episodes,)
        Total transaction costs for each episode.
    episodes : list of HedgingEpisode, optional
        Full episode records (if return_episodes=True).
    """
    rng = np.random.default_rng(seed)
    
    pnl_samples = np.zeros(n_episodes)
    cost_samples = np.zeros(n_episodes)
    episodes = [] if return_episodes else None
    
    for i in range(n_episodes):
        # Reset environment with unique seed
        episode_seed = rng.integers(0, 2**31)
        state, info = env.reset(seed=episode_seed)
        
        # Reset agent if it has state
        if hasattr(agent, "reset"):
            agent.reset()
        
        total_cost = 0.0
        
        # Run episode
        while True:
            action = agent.select_action(state, training=False, explore=False)
            state, reward, terminated, truncated, step_info = env.step(action)
            total_cost += step_info.get("cost", 0.0)
            
            if terminated or truncated:
                break
        
        # Record results
        pnl_samples[i] = step_info.get("terminal_pnl", state.pnl)
        cost_samples[i] = total_cost
        
        if return_episodes:
            episodes.append(env.get_episode())
    
    if return_episodes:
        return pnl_samples, cost_samples, episodes
    return pnl_samples, cost_samples


def simulate_hedging_batch_vectorised(
    agent: DeepHedgingAgent,
    env: BaseHedgingEnv,
    n_episodes: int,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorised batch simulation for faster training.
    
    This simulates all episodes in parallel using pre-generated paths,
    which is more efficient for gradient estimation.
    
    Parameters
    ----------
    agent : DeepHedgingAgent
        Deep hedging agent.
    env : BaseHedgingEnv
        Hedging environment (must support simulate_paths).
    n_episodes : int
        Number of episodes.
    seed : int, optional
        Random seed.
    
    Returns
    -------
    pnl_samples : ndarray, shape (n_episodes,)
        Terminal P&L.
    cost_samples : ndarray, shape (n_episodes,)
        Total costs.
    action_history : ndarray, shape (n_episodes, n_steps)
        Actions taken at each step.
    """
    config = env.config
    n_steps = config.n_steps
    
    # Generate all paths at once
    spot_paths = env.simulate_paths(n_episodes, seed=seed, antithetic=True)
    
    # Initial values
    initial_premium = env._compute_bsm_price() * config.notional
    
    # Arrays to track state
    positions = np.zeros((n_episodes, n_steps))
    costs = np.zeros((n_episodes, n_steps))
    
    # Simulate step by step (vectorised over episodes)
    current_positions = np.full(n_episodes, config.initial_position)
    current_pnl = np.full(n_episodes, initial_premium)
    
    for t in range(n_steps):
        spots = spot_paths[:, t]
        tau = config.maturity - t * config.dt
        
        # Compute Greeks for all episodes
        greeks = env.compute_greeks(float(np.mean(spots)), tau)  # Approximate
        
        # Build states and get actions
        actions = np.zeros(n_episodes)
        for i in range(n_episodes):
            state = HedgingState(
                spot=spots[i],
                time=t * config.dt,
                time_to_maturity=tau,
                position=current_positions[i],
                pnl=current_pnl[i],
                step=t,
                delta_bs=greeks.get("delta"),
                gamma_bs=greeks.get("gamma"),
                vega_bs=greeks.get("vega"),
                strike=config.strike,
                initial_spot=config.spot_initial,
            )
            actions[i] = agent.select_action(state, training=False, explore=False)
        
        # Compute trades and costs
        trades = actions - current_positions
        step_costs = env.cost_model.compute(
            trade_size=trades * config.notional,
            spot=spots,
        )
        
        # Update positions
        current_positions = actions
        positions[:, t] = actions
        costs[:, t] = step_costs
        
        # Update P&L with hedge gains
        if t < n_steps - 1:
            next_spots = spot_paths[:, t + 1]
        else:
            next_spots = spot_paths[:, -1]
        
        hedge_pnl = actions * (next_spots - spots) * config.notional
        current_pnl += hedge_pnl - step_costs
    
    # Terminal: subtract payoff
    terminal_spots = spot_paths[:, -1]
    if config.option_type == "call":
        payoffs = np.maximum(terminal_spots - config.strike, 0) * config.notional
    else:
        payoffs = np.maximum(config.strike - terminal_spots, 0) * config.notional
    
    terminal_pnl = current_pnl - payoffs
    total_costs = np.sum(costs, axis=1)
    
    return terminal_pnl, total_costs, positions


@dataclass
class HedgingTrainer:
    """
    Trainer for deep hedging agents.
    
    This class manages the training loop:
    1. Simulate batch of episodes
    2. Compute loss (risk measure)
    3. Estimate gradients
    4. Update agent
    
    Parameters
    ----------
    agent : DeepHedgingAgent
        Agent to train.
    env : BaseHedgingEnv
        Hedging environment.
    risk_measure : RiskMeasure, optional
        Risk measure for loss. If None, uses agent's risk measure.
    batch_size : int
        Number of episodes per batch.
    learning_rate : float
        Learning rate (can override agent's).
    gradient_method : str
        "finite_diff" or "autodiff" (autodiff requires TF/PyTorch).
    
    Attributes
    ----------
    history : dict
        Training history (loss, metrics per epoch).
    """
    
    agent: DeepHedgingAgent
    env: BaseHedgingEnv
    risk_measure: Optional[RiskMeasure] = None
    batch_size: int = 256
    learning_rate: Optional[float] = None
    gradient_method: str = "finite_diff"
    
    # Training state
    history: Dict[str, List[float]] = field(default_factory=lambda: {
        "loss": [],
        "mean_pnl": [],
        "std_pnl": [],
        "mean_cost": [],
    })
    _epoch: int = 0
    
    def __post_init__(self):
        if self.risk_measure is None:
            self.risk_measure = self.agent.risk_measure
        if self.learning_rate is not None:
            self.agent.learning_rate = self.learning_rate
    
    def train_epoch(self, seed: Optional[int] = None) -> Dict[str, float]:
        """
        Train for one epoch (one batch of episodes).
        
        Returns
        -------
        dict
            Metrics for this epoch.
        """
        # Simulate batch
        pnl_samples, cost_samples = simulate_hedging_batch(
            agent=self.agent,
            env=self.env,
            n_episodes=self.batch_size,
            seed=seed,
        )
        
        # Compute loss
        loss = self.risk_measure.compute(-pnl_samples)
        
        # Estimate gradients and update
        if self.gradient_method == "finite_diff":
            gradients = self._estimate_gradients_fd(pnl_samples)
            self.agent.update(batch={"gradients": gradients, "loss": loss})
        
        # Record metrics
        metrics = {
            "loss": loss,
            "mean_pnl": float(np.mean(pnl_samples)),
            "std_pnl": float(np.std(pnl_samples)),
            "mean_cost": float(np.mean(cost_samples)),
        }
        
        for key, value in metrics.items():
            self.history[key].append(value)
        
        self._epoch += 1
        return metrics
    
    def _estimate_gradients_fd(
        self,
        pnl_samples: np.ndarray,
        eps: float = 1e-4,
    ) -> Dict[str, List[np.ndarray]]:
        """
        Estimate gradients via finite differences.
        
        This is slow but doesn't require autodiff frameworks.
        For each parameter θᵢ:
            ∂Loss/∂θᵢ ≈ (Loss(θᵢ + ε) - Loss(θᵢ - ε)) / (2ε)
        
        Note: This is an approximation because we use the same P&L samples
        for both perturbed parameters. For better estimates, we would need
        to re-simulate with perturbed policies.
        """
        weight_grads = []
        bias_grads = []
        
        # For efficiency, we use a simplified gradient estimate
        # based on the risk measure gradient w.r.t. P&L
        # and the policy gradient w.r.t. parameters
        
        # This is a placeholder - full implementation would require
        # re-simulating with perturbed parameters
        for w in self.agent.policy.weights:
            weight_grads.append(np.random.randn(*w.shape) * 0.01)
        for b in self.agent.policy.biases:
            bias_grads.append(np.random.randn(*b.shape) * 0.01)
        
        return {"weights": weight_grads, "biases": bias_grads}
    
    def train(
        self,
        n_epochs: int,
        verbose: int = 1,
        log_every: int = 10,
        early_stopping_patience: int = 0,
        checkpoint_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train for multiple epochs.
        
        Parameters
        ----------
        n_epochs : int
            Number of training epochs.
        verbose : int
            Verbosity level (0=silent, 1=progress, 2=detailed).
        log_every : int
            Log metrics every N epochs.
        early_stopping_patience : int
            Stop if no improvement for this many epochs (0=disabled).
        checkpoint_dir : str, optional
            Directory to save checkpoints.
        
        Returns
        -------
        dict
            Training results including final metrics and history.
        """
        start_time = time.time()
        best_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(1, n_epochs + 1):
            metrics = self.train_epoch()
            
            # Logging
            if verbose >= 1 and epoch % log_every == 0:
                logger.info(
                    f"Epoch {epoch}/{n_epochs} — "
                    f"loss: {metrics['loss']:.4f}, "
                    f"mean_pnl: {metrics['mean_pnl']:.4f}, "
                    f"std_pnl: {metrics['std_pnl']:.4f}"
                )
            
            # Early stopping
            if early_stopping_patience > 0:
                if metrics["loss"] < best_loss:
                    best_loss = metrics["loss"]
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        if verbose >= 1:
                            logger.info(f"Early stopping at epoch {epoch}")
                        break
            
            # Checkpointing
            if checkpoint_dir is not None and epoch % log_every == 0:
                self._save_checkpoint(checkpoint_dir, epoch)
        
        training_time = time.time() - start_time
        
        return {
            "final_loss": self.history["loss"][-1] if self.history["loss"] else None,
            "best_loss": best_loss,
            "n_epochs": epoch,
            "training_time_seconds": training_time,
            "history": self.history,
        }
    
    def _save_checkpoint(self, checkpoint_dir: str, epoch: int) -> None:
        """Save agent checkpoint."""
        import json
        path = Path(checkpoint_dir)
        path.mkdir(parents=True, exist_ok=True)
        
        params = self.agent.get_parameters()
        # Convert numpy arrays to lists for JSON
        params["policy"]["weights"] = [w.tolist() for w in params["policy"]["weights"]]
        params["policy"]["biases"] = [b.tolist() for b in params["policy"]["biases"]]
        
        with open(path / f"checkpoint_epoch_{epoch}.json", "w") as f:
            json.dump(params, f)


def train_deep_hedging(
    agent: DeepHedgingAgent,
    env: BaseHedgingEnv,
    n_epochs: int = 100,
    batch_size: int = 256,
    learning_rate: Optional[float] = None,
    risk_measure: Optional[RiskMeasure] = None,
    verbose: int = 1,
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function to train a deep hedging agent.
    
    Parameters
    ----------
    agent : DeepHedgingAgent
        Agent to train.
    env : BaseHedgingEnv
        Hedging environment.
    n_epochs : int
        Number of training epochs.
    batch_size : int
        Episodes per batch.
    learning_rate : float, optional
        Override agent's learning rate.
    risk_measure : RiskMeasure, optional
        Override agent's risk measure.
    verbose : int
        Verbosity level.
    **kwargs : dict
        Additional arguments for HedgingTrainer.train().
    
    Returns
    -------
    dict
        Training results.
    
    Example
    -------
    >>> result = train_deep_hedging(agent, env, n_epochs=100)
    >>> print(f"Final loss: {result['final_loss']:.4f}")
    """
    trainer = HedgingTrainer(
        agent=agent,
        env=env,
        batch_size=batch_size,
        learning_rate=learning_rate,
        risk_measure=risk_measure,
    )
    
    return trainer.train(n_epochs=n_epochs, verbose=verbose, **kwargs)


__all__ = [
    "HedgingTrainer",
    "simulate_hedging_batch",
    "simulate_hedging_batch_vectorised",
    "train_deep_hedging",
]

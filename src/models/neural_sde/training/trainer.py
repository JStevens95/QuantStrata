"""
Training pipelines for Neural SDEs.

Provides gradient-based training for learning drift and diffusion
functions from historical data.

Example:
    from src.models.neural_sde import NeuralSDEDynamics
    from src.models.neural_sde.training import NeuralSDETrainer, TrainingConfig
    
    sde = NeuralSDEDynamics()
    trainer = NeuralSDETrainer(config=TrainingConfig(n_epochs=100))
    
    result = trainer.fit(sde, historical_paths)
    print(f"Final loss: {result.final_loss:.6f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.models.neural_sde.training.losses import (
    MomentMatchingLoss,
    PathwiseLoss,
)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class TrainingConfig:
    """Configuration for Neural SDE training."""
    
    # Optimization
    n_epochs: int = 100
    learning_rate: float = 1e-3
    batch_size: int = 32
    
    # Loss weights
    moment_weight: float = 1.0
    pathwise_weight: float = 0.5
    
    # Regularization
    l2_reg: float = 1e-5
    
    # Simulation for training
    n_sim_paths: int = 1000
    n_sim_steps: int = 50
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-6
    
    # Logging
    verbose: bool = True
    log_interval: int = 10


@dataclass
class TrainingResult:
    """Result from Neural SDE training."""
    
    final_loss: float
    loss_history: List[float]
    epoch: int
    converged: bool
    
    moment_losses: List[float] = field(default_factory=list)
    pathwise_losses: List[float] = field(default_factory=list)
    
    def summary(self) -> Dict[str, Any]:
        """Get summary dictionary."""
        return {
            "final_loss": self.final_loss,
            "epochs": self.epoch,
            "converged": self.converged,
            "loss_reduction": (
                self.loss_history[0] - self.final_loss 
                if self.loss_history else 0.0
            ),
        }


# =============================================================================
# Neural SDE Trainer
# =============================================================================


class NeuralSDETrainer:
    """
    Trainer for Neural SDE models.
    
    Uses gradient-free optimization (finite differences) for
    portability. For GPU acceleration, use TensorFlow/JAX backends.
    
    Training approach:
    1. Simulate paths from current model
    2. Compare to historical paths using loss functions
    3. Update network weights to minimize loss
    
    Example:
        trainer = NeuralSDETrainer(
            config=TrainingConfig(n_epochs=100, learning_rate=1e-3)
        )
        
        result = trainer.fit(sde, historical_paths)
    """
    
    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize trainer.
        
        Parameters
        ----------
        config : TrainingConfig, optional
            Training configuration.
        seed : int, optional
            Random seed.
        """
        self.config = config or TrainingConfig()
        self._rng = np.random.default_rng(seed)
        
        # Loss functions
        self.moment_loss = MomentMatchingLoss()
        self.pathwise_loss = PathwiseLoss()
    
    def fit(
        self,
        model: Any,  # NeuralSDEDynamics
        historical_paths: np.ndarray,
        validation_paths: Optional[np.ndarray] = None,
    ) -> TrainingResult:
        """
        Train Neural SDE on historical data.
        
        Parameters
        ----------
        model : NeuralSDEDynamics
            Model to train.
        historical_paths : ndarray
            Historical paths of shape (n_paths, n_steps).
        validation_paths : ndarray, optional
            Validation paths for early stopping.
        
        Returns
        -------
        TrainingResult
            Training result with loss history.
        """
        cfg = self.config
        
        # Extract initial parameters
        drift_params = model.drift_network.get_parameters()
        diffusion_params = model.diffusion_network.get_parameters()
        
        # Training history
        loss_history: List[float] = []
        moment_losses: List[float] = []
        pathwise_losses: List[float] = []
        
        # Get simulation parameters from data
        S0 = float(historical_paths[0, 0])
        n_steps = historical_paths.shape[1] - 1
        T = 1.0  # Assume unit time (can be parameterized)
        
        # Early stopping
        best_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(cfg.n_epochs):
            # Sample batch from historical data
            batch_indices = self._rng.choice(
                len(historical_paths),
                size=min(cfg.batch_size, len(historical_paths)),
                replace=False,
            )
            batch_paths = historical_paths[batch_indices]
            
            # Simulate paths from current model
            sim_paths = model.simulate(
                S0=S0,
                T=T,
                n_steps=min(n_steps, cfg.n_sim_steps),
                n_paths=cfg.n_sim_paths,
            )
            
            # Compute losses
            m_loss = self.moment_loss.compute(sim_paths, batch_paths)
            p_loss = self.pathwise_loss.compute(sim_paths, batch_paths)
            
            total_loss = (
                cfg.moment_weight * m_loss +
                cfg.pathwise_weight * p_loss
            )
            
            # L2 regularization
            if cfg.l2_reg > 0:
                reg_loss = self._compute_l2_reg(drift_params, diffusion_params)
                total_loss += cfg.l2_reg * reg_loss
            
            loss_history.append(total_loss)
            moment_losses.append(m_loss)
            pathwise_losses.append(p_loss)
            
            # Update parameters (gradient-free optimization)
            self._update_parameters(
                model=model,
                drift_params=drift_params,
                diffusion_params=diffusion_params,
                historical_paths=batch_paths,
                S0=S0,
                T=T,
                n_steps=min(n_steps, cfg.n_sim_steps),
            )
            
            # Get updated parameters
            drift_params = model.drift_network.get_parameters()
            diffusion_params = model.diffusion_network.get_parameters()
            
            # Early stopping check
            if total_loss < best_loss - cfg.min_delta:
                best_loss = total_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= cfg.patience:
                if cfg.verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break
            
            # Logging
            if cfg.verbose and (epoch + 1) % cfg.log_interval == 0:
                print(
                    f"Epoch {epoch + 1}/{cfg.n_epochs} - "
                    f"Loss: {total_loss:.6f} "
                    f"(moment: {m_loss:.6f}, pathwise: {p_loss:.6f})"
                )
        
        return TrainingResult(
            final_loss=loss_history[-1] if loss_history else 0.0,
            loss_history=loss_history,
            epoch=epoch + 1,
            converged=patience_counter >= cfg.patience,
            moment_losses=moment_losses,
            pathwise_losses=pathwise_losses,
        )
    
    def _update_parameters(
        self,
        model: Any,
        drift_params: dict,
        diffusion_params: dict,
        historical_paths: np.ndarray,
        S0: float,
        T: float,
        n_steps: int,
        bump_size: float = 1e-4,
    ) -> None:
        """
        Update model parameters using finite-difference gradients.
        
        This is a gradient-free approach for portability.
        For efficiency, use TensorFlow/PyTorch with autodiff.
        """
        lr = self.config.learning_rate
        
        # Update drift network weights
        for i, (W, b) in enumerate(zip(
            drift_params["network"]["weights"],
            drift_params["network"]["biases"]
        )):
            # Gradient for weights (sample a few elements)
            for idx in self._sample_indices(W.shape, n_samples=5):
                grad = self._finite_diff_gradient(
                    model=model,
                    param_type="drift_weight",
                    layer_idx=i,
                    param_idx=idx,
                    historical_paths=historical_paths,
                    S0=S0,
                    T=T,
                    n_steps=n_steps,
                    bump_size=bump_size,
                )
                W[idx] -= lr * grad
            
            # Gradient for biases
            for j in range(min(3, len(b))):
                grad = self._finite_diff_gradient(
                    model=model,
                    param_type="drift_bias",
                    layer_idx=i,
                    param_idx=j,
                    historical_paths=historical_paths,
                    S0=S0,
                    T=T,
                    n_steps=n_steps,
                    bump_size=bump_size,
                )
                b[j] -= lr * grad
        
        # Update diffusion network (similar)
        for i, (W, b) in enumerate(zip(
            diffusion_params["network"]["weights"],
            diffusion_params["network"]["biases"]
        )):
            for idx in self._sample_indices(W.shape, n_samples=5):
                grad = self._finite_diff_gradient(
                    model=model,
                    param_type="diffusion_weight",
                    layer_idx=i,
                    param_idx=idx,
                    historical_paths=historical_paths,
                    S0=S0,
                    T=T,
                    n_steps=n_steps,
                    bump_size=bump_size,
                )
                W[idx] -= lr * grad
        
        # Set updated parameters
        model.drift_network.set_parameters(drift_params)
        model.diffusion_network.set_parameters(diffusion_params)
    
    def _finite_diff_gradient(
        self,
        model: Any,
        param_type: str,
        layer_idx: int,
        param_idx: Any,
        historical_paths: np.ndarray,
        S0: float,
        T: float,
        n_steps: int,
        bump_size: float,
    ) -> float:
        """Compute gradient using central finite differences."""
        cfg = self.config
        
        # Get current parameters
        if "drift" in param_type:
            params = model.drift_network.get_parameters()
            if "weight" in param_type:
                params_arr = params["network"]["weights"][layer_idx]
            else:
                params_arr = params["network"]["biases"][layer_idx]
        else:
            params = model.diffusion_network.get_parameters()
            if "weight" in param_type:
                params_arr = params["network"]["weights"][layer_idx]
            else:
                params_arr = params["network"]["biases"][layer_idx]
        
        original_value = params_arr[param_idx]
        
        # Forward bump
        params_arr[param_idx] = original_value + bump_size
        if "drift" in param_type:
            model.drift_network.set_parameters(params)
        else:
            model.diffusion_network.set_parameters(params)
        
        sim_paths_up = model.simulate(
            S0=S0, T=T, n_steps=n_steps, n_paths=cfg.n_sim_paths // 4
        )
        loss_up = (
            cfg.moment_weight * self.moment_loss.compute(sim_paths_up, historical_paths) +
            cfg.pathwise_weight * self.pathwise_loss.compute(sim_paths_up, historical_paths)
        )
        
        # Backward bump
        params_arr[param_idx] = original_value - bump_size
        if "drift" in param_type:
            model.drift_network.set_parameters(params)
        else:
            model.diffusion_network.set_parameters(params)
        
        sim_paths_down = model.simulate(
            S0=S0, T=T, n_steps=n_steps, n_paths=cfg.n_sim_paths // 4
        )
        loss_down = (
            cfg.moment_weight * self.moment_loss.compute(sim_paths_down, historical_paths) +
            cfg.pathwise_weight * self.pathwise_loss.compute(sim_paths_down, historical_paths)
        )
        
        # Restore original
        params_arr[param_idx] = original_value
        if "drift" in param_type:
            model.drift_network.set_parameters(params)
        else:
            model.diffusion_network.set_parameters(params)
        
        # Central difference
        gradient = (loss_up - loss_down) / (2 * bump_size)
        
        return gradient
    
    def _sample_indices(
        self,
        shape: tuple,
        n_samples: int,
    ) -> List[tuple]:
        """Sample random indices from array."""
        indices = []
        for _ in range(n_samples):
            idx = tuple(self._rng.integers(0, s) for s in shape)
            indices.append(idx)
        return indices
    
    def _compute_l2_reg(
        self,
        drift_params: dict,
        diffusion_params: dict,
    ) -> float:
        """Compute L2 regularization loss."""
        reg = 0.0
        
        for W in drift_params["network"]["weights"]:
            reg += np.sum(W ** 2)
        
        for W in diffusion_params["network"]["weights"]:
            reg += np.sum(W ** 2)
        
        return float(reg)


__all__ = [
    "NeuralSDETrainer",
    "TrainingConfig",
    "TrainingResult",
]

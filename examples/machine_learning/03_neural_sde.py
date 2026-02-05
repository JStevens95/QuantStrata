#!/usr/bin/env python3
"""
===============================================================================
Machine Learning: Neural Stochastic Differential Equations
===============================================================================

This example demonstrates Neural SDEs - combining deep learning with stochastic
processes for flexible, learnable dynamics models in quantitative finance.

Learning Objectives
-------------------
1. **Neural SDE Concept**: Learn dynamics from data
2. **Architecture Design**: Drift and diffusion networks
3. **Training Process**: Backprop through SDE solutions
4. **Applications**: Volatility modeling, path simulation

Mathematical Framework
----------------------
Standard SDE:
    dX_t = μ(X_t, t) dt + σ(X_t, t) dW_t

Neural SDE:
    dX_t = μ_θ(X_t, t) dt + σ_φ(X_t, t) dW_t

Where μ_θ and σ_φ are neural networks with learnable parameters.

Training objective (simplified):
    L = Σ_t ||X_t^data - X_t^model||² + regularization

Production Context
------------------
At a hedge fund:
- Neural SDEs can learn complex volatility dynamics
- More flexible than parametric models (Heston, SABR)
- Used for simulation, pricing, risk
- Requires careful regularization and validation

Prerequisites
-------------
- Basic SDE knowledge
- Neural network fundamentals
- Previous ML examples

Run This Example
----------------
    cd /path/to/QuantStrata
    PYTHONPATH=. python examples/machine_learning/03_neural_sde.py

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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

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
# NEURAL NETWORK COMPONENTS
# =============================================================================

class DenseLayer:
    """
    Dense layer for neural networks.
    
    Implements: y = activation(Wx + b)
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: str = 'tanh',
    ) -> None:
        """Initialize dense layer."""
        # Xavier initialization
        scale = np.sqrt(2.0 / (input_dim + output_dim))
        self.W = np.random.randn(input_dim, output_dim) * scale
        self.b = np.zeros((1, output_dim))
        
        self.activation = activation
        
        # Gradients
        self.dW = None
        self.db = None
        
        # Cache for backprop
        self._input = None
        self._pre_activation = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        self._input = x
        self._pre_activation = x @ self.W + self.b
        
        if self.activation == 'tanh':
            return np.tanh(self._pre_activation)
        elif self.activation == 'relu':
            return np.maximum(0, self._pre_activation)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-self._pre_activation))
        elif self.activation == 'softplus':
            return np.log(1 + np.exp(self._pre_activation))
        else:  # linear
            return self._pre_activation
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass."""
        # Activation gradient
        if self.activation == 'tanh':
            a = np.tanh(self._pre_activation)
            grad_act = grad_output * (1 - a**2)
        elif self.activation == 'relu':
            grad_act = grad_output * (self._pre_activation > 0).astype(float)
        elif self.activation == 'sigmoid':
            s = 1 / (1 + np.exp(-self._pre_activation))
            grad_act = grad_output * s * (1 - s)
        elif self.activation == 'softplus':
            grad_act = grad_output * (1 / (1 + np.exp(-self._pre_activation)))
        else:  # linear
            grad_act = grad_output
        
        # Parameter gradients
        self.dW = self._input.T @ grad_act
        self.db = np.sum(grad_act, axis=0, keepdims=True)
        
        # Input gradient
        return grad_act @ self.W.T


class NeuralNetwork:
    """
    Simple multi-layer neural network.
    """
    
    def __init__(
        self,
        layer_dims: List[int],
        activations: List[str],
    ) -> None:
        """
        Initialize neural network.
        
        Parameters
        ----------
        layer_dims : List[int]
            Dimensions including input and output.
        activations : List[str]
            Activation for each layer (len = len(layer_dims) - 1).
        """
        self.layers: List[DenseLayer] = []
        
        for i in range(len(layer_dims) - 1):
            self.layers.append(
                DenseLayer(layer_dims[i], layer_dims[i + 1], activations[i])
            )
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through all layers."""
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass through all layers."""
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output
    
    def get_params(self) -> List[np.ndarray]:
        """Get all parameters."""
        params = []
        for layer in self.layers:
            params.extend([layer.W, layer.b])
        return params
    
    def get_grads(self) -> List[np.ndarray]:
        """Get all gradients."""
        grads = []
        for layer in self.layers:
            grads.extend([layer.dW, layer.db])
        return grads
    
    def update_params(self, lr: float) -> None:
        """Update parameters with gradients."""
        for layer in self.layers:
            layer.W -= lr * layer.dW
            layer.b -= lr * layer.db


# =============================================================================
# NEURAL SDE
# =============================================================================

class NeuralSDE:
    """
    Neural Stochastic Differential Equation.
    
    Models: dX_t = μ_θ(X_t, t) dt + σ_φ(X_t, t) dW_t
    
    Where μ_θ and σ_φ are neural networks.
    
    Example:
        sde = NeuralSDE(state_dim=1, hidden_dim=32)
        paths = sde.simulate(X0, T, n_steps, n_paths)
        sde.fit(observed_paths, T, epochs=100)
    """
    
    def __init__(
        self,
        state_dim: int = 1,
        hidden_dim: int = 32,
        learning_rate: float = 0.001,
    ) -> None:
        """
        Initialize Neural SDE.
        
        Parameters
        ----------
        state_dim : int
            Dimension of state X.
        hidden_dim : int
            Hidden layer dimension.
        learning_rate : float
            Learning rate for training.
        """
        self.state_dim = state_dim
        self.learning_rate = learning_rate
        
        # Drift network: μ_θ(X, t) -> R^state_dim
        # Input: [X, t]
        self.drift_net = NeuralNetwork(
            layer_dims=[state_dim + 1, hidden_dim, hidden_dim, state_dim],
            activations=['tanh', 'tanh', 'linear'],
        )
        
        # Diffusion network: σ_φ(X, t) -> R^state_dim (diagonal)
        # Output through softplus to ensure positivity
        self.diffusion_net = NeuralNetwork(
            layer_dims=[state_dim + 1, hidden_dim, hidden_dim, state_dim],
            activations=['tanh', 'tanh', 'softplus'],
        )
        
        self.train_losses: List[float] = []
    
    def drift(self, X: np.ndarray, t: float) -> np.ndarray:
        """
        Compute drift μ_θ(X, t).
        
        Parameters
        ----------
        X : np.ndarray
            State, shape (batch, state_dim).
        t : float
            Time.
        
        Returns
        -------
        np.ndarray
            Drift values, shape (batch, state_dim).
        """
        t_vec = np.full((X.shape[0], 1), t)
        inputs = np.hstack([X, t_vec])
        return self.drift_net.forward(inputs)
    
    def diffusion(self, X: np.ndarray, t: float) -> np.ndarray:
        """
        Compute diffusion σ_φ(X, t).
        
        Parameters
        ----------
        X : np.ndarray
            State, shape (batch, state_dim).
        t : float
            Time.
        
        Returns
        -------
        np.ndarray
            Diffusion values (positive), shape (batch, state_dim).
        """
        t_vec = np.full((X.shape[0], 1), t)
        inputs = np.hstack([X, t_vec])
        # Ensure minimum diffusion for numerical stability
        return self.diffusion_net.forward(inputs) + 1e-4
    
    def simulate(
        self,
        X0: np.ndarray,
        T: float,
        n_steps: int,
        n_paths: int,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Simulate paths using Euler-Maruyama scheme.
        
        Parameters
        ----------
        X0 : np.ndarray
            Initial state, shape (state_dim,).
        T : float
            Time horizon.
        n_steps : int
            Number of time steps.
        n_paths : int
            Number of paths to simulate.
        seed : Optional[int]
            Random seed.
        
        Returns
        -------
        np.ndarray
            Paths, shape (n_paths, n_steps + 1, state_dim).
        """
        if seed is not None:
            np.random.seed(seed)
        
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        
        # Initialize paths
        paths = np.zeros((n_paths, n_steps + 1, self.state_dim))
        paths[:, 0, :] = X0
        
        # Euler-Maruyama
        for i in range(n_steps):
            t = i * dt
            X = paths[:, i, :]
            
            mu = self.drift(X, t)
            sigma = self.diffusion(X, t)
            
            dW = np.random.randn(n_paths, self.state_dim) * sqrt_dt
            
            paths[:, i + 1, :] = X + mu * dt + sigma * dW
        
        return paths
    
    def fit(
        self,
        observed_paths: np.ndarray,
        T: float,
        epochs: int = 100,
        batch_size: int = 64,
        verbose: bool = True,
    ) -> None:
        """
        Fit Neural SDE to observed paths.
        
        Uses MSE loss on drift prediction + variance matching for diffusion.
        This provides interpretable, always-positive loss values.
        
        Parameters
        ----------
        observed_paths : np.ndarray
            Observed paths, shape (n_paths, n_steps + 1, state_dim).
        T : float
            Time horizon.
        epochs : int
            Training epochs.
        batch_size : int
            Batch size.
        verbose : bool
            Print progress.
        """
        n_paths, n_steps_plus_1, _ = observed_paths.shape
        n_steps = n_steps_plus_1 - 1
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        
        # Compute empirical increments for diffusion target
        # dX = X_{t+1} - X_t, empirical volatility from data
        increments = np.diff(observed_paths, axis=1)  # (n_paths, n_steps, state_dim)
        
        for epoch in range(epochs):
            epoch_drift_loss = 0.0
            epoch_diff_loss = 0.0
            n_batches = 0
            
            # Shuffle paths
            perm = np.random.permutation(n_paths)
            
            for batch_start in range(0, n_paths, batch_size):
                batch_idx = perm[batch_start:batch_start + batch_size]
                batch_paths = observed_paths[batch_idx]
                batch_increments = increments[batch_idx]
                
                # Process each time step
                for i in range(n_steps):
                    t = i * dt
                    X = batch_paths[:, i, :]
                    dX_actual = batch_increments[:, i, :]  # Actual increment
                    
                    # Forward pass
                    mu = self.drift(X, t)
                    sigma = self.diffusion(X, t)
                    
                    # Drift prediction: dX ≈ μ·dt, so μ ≈ dX/dt
                    # MSE loss on drift
                    drift_target = dX_actual / dt
                    drift_loss = np.mean((mu - drift_target) ** 2)
                    
                    # Diffusion: match variance of residuals
                    # Var(dX - μ·dt) = σ²·dt, so σ ≈ |dX - μ·dt| / sqrt(dt)
                    residual = dX_actual - mu * dt
                    empirical_vol = np.abs(residual) / sqrt_dt
                    diff_loss = np.mean((sigma - empirical_vol) ** 2)
                    
                    epoch_drift_loss += drift_loss
                    epoch_diff_loss += diff_loss
                    
                    # Backward pass for drift
                    grad_mu = 2 * (mu - drift_target) / len(batch_idx)
                    t_vec = np.full((len(batch_idx), 1), t)
                    inputs = np.hstack([X, t_vec])
                    _ = self.drift_net.forward(inputs)
                    self.drift_net.backward(grad_mu)
                    self.drift_net.update_params(self.learning_rate)
                    
                    # Backward pass for diffusion
                    grad_sigma = 2 * (sigma - empirical_vol) / len(batch_idx)
                    _ = self.diffusion_net.forward(inputs)
                    self.diffusion_net.backward(grad_sigma)
                    self.diffusion_net.update_params(self.learning_rate * 0.1)  # Slower for stability
                
                n_batches += 1
            
            # Combined loss (weighted sum)
            avg_drift_loss = epoch_drift_loss / (n_batches * n_steps)
            avg_diff_loss = epoch_diff_loss / (n_batches * n_steps)
            total_loss = avg_drift_loss + 0.1 * avg_diff_loss  # Weight diffusion less
            
            self.train_losses.append(total_loss)
            
            if verbose and (epoch + 1) % 20 == 0:
                logger.info(
                    f"Epoch {epoch + 1:>3}/{epochs}: "
                    f"Drift Loss = {avg_drift_loss:.6f}, "
                    f"Diff Loss = {avg_diff_loss:.6f}"
                )


# =============================================================================
# TARGET SDE MODELS
# =============================================================================

def simulate_gbm_paths(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate Geometric Brownian Motion paths.
    
    dS_t = μ S_t dt + σ S_t dW_t
    """
    if seed is not None:
        np.random.seed(seed)
    
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    
    paths = np.zeros((n_paths, n_steps + 1, 1))
    paths[:, 0, 0] = S0
    
    for i in range(n_steps):
        S = paths[:, i, 0]
        dW = np.random.randn(n_paths) * sqrt_dt
        paths[:, i + 1, 0] = S + mu * S * dt + sigma * S * dW
    
    return paths


def simulate_cev_paths(
    S0: float,
    mu: float,
    sigma: float,
    gamma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate Constant Elasticity of Variance (CEV) paths.
    
    dS_t = μ S_t dt + σ S_t^γ dW_t
    """
    if seed is not None:
        np.random.seed(seed)
    
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    
    paths = np.zeros((n_paths, n_steps + 1, 1))
    paths[:, 0, 0] = S0
    
    for i in range(n_steps):
        S = np.maximum(paths[:, i, 0], 1e-6)  # Prevent negative
        dW = np.random.randn(n_paths) * sqrt_dt
        paths[:, i + 1, 0] = S + mu * S * dt + sigma * (S ** gamma) * dW
        paths[:, i + 1, 0] = np.maximum(paths[:, i + 1, 0], 1e-6)
    
    return paths


def simulate_ou_paths(
    X0: float,
    theta: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate Ornstein-Uhlenbeck paths.
    
    dX_t = θ(μ - X_t) dt + σ dW_t
    """
    if seed is not None:
        np.random.seed(seed)
    
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    
    paths = np.zeros((n_paths, n_steps + 1, 1))
    paths[:, 0, 0] = X0
    
    for i in range(n_steps):
        X = paths[:, i, 0]
        dW = np.random.randn(n_paths) * sqrt_dt
        paths[:, i + 1, 0] = X + theta * (mu - X) * dt + sigma * dW
    
    return paths


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for Neural SDE experiment."""
    T: float = 1.0
    n_steps: int = 50
    n_train_paths: int = 1000
    n_test_paths: int = 200
    hidden_dim: int = 32
    epochs: int = 100
    learning_rate: float = 0.01


def run_neural_sde() -> Tuple[dict, dict]:
    """
    Run Neural SDE experiments.
    
    Returns
    -------
    Tuple
        Experiment results and trained models.
    """
    logger.info("=" * 70)
    logger.info("SECTION 1: Neural SDE for GBM")
    logger.info("=" * 70)
    
    config = ExperimentConfig()
    results = {}
    models = {}
    
    # -------------------------------------------------------------------------
    # Experiment 1: Learn GBM dynamics
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("Generating GBM training data...")
    
    S0 = 100.0
    mu_true = 0.05
    sigma_true = 0.2
    
    gbm_train = simulate_gbm_paths(
        S0, mu_true, sigma_true, config.T, config.n_steps,
        config.n_train_paths, seed=42
    )
    
    gbm_test = simulate_gbm_paths(
        S0, mu_true, sigma_true, config.T, config.n_steps,
        config.n_test_paths, seed=123
    )
    
    logger.info(f"  Train paths: {config.n_train_paths}")
    logger.info(f"  Test paths:  {config.n_test_paths}")
    logger.info(f"  Time steps:  {config.n_steps}")
    
    # Normalize for training
    gbm_train_norm = gbm_train / S0
    gbm_test_norm = gbm_test / S0
    
    # Train Neural SDE
    logger.info("")
    logger.info("Training Neural SDE on GBM data...")
    
    nsde_gbm = NeuralSDE(
        state_dim=1,
        hidden_dim=config.hidden_dim,
        learning_rate=config.learning_rate,
    )
    
    nsde_gbm.fit(gbm_train_norm, config.T, epochs=config.epochs)
    
    # Evaluate
    logger.info("")
    logger.info("Evaluating on test data...")
    
    nsde_gbm_paths = nsde_gbm.simulate(
        X0=np.array([1.0]),
        T=config.T,
        n_steps=config.n_steps,
        n_paths=config.n_test_paths,
        seed=123,
    ) * S0
    
    # Compare terminal distributions
    true_terminal = gbm_test[:, -1, 0]
    pred_terminal = nsde_gbm_paths[:, -1, 0]
    
    results['gbm'] = {
        'true_mean': np.mean(true_terminal),
        'pred_mean': np.mean(pred_terminal),
        'true_std': np.std(true_terminal),
        'pred_std': np.std(pred_terminal),
        'true_paths': gbm_test,
        'pred_paths': nsde_gbm_paths,
        'train_loss': nsde_gbm.train_losses,
    }
    
    models['gbm'] = nsde_gbm
    
    logger.info(f"  True terminal: mean={results['gbm']['true_mean']:.2f}, std={results['gbm']['true_std']:.2f}")
    logger.info(f"  Pred terminal: mean={results['gbm']['pred_mean']:.2f}, std={results['gbm']['pred_std']:.2f}")
    
    # -------------------------------------------------------------------------
    # Experiment 2: Learn OU dynamics
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 2: Neural SDE for Ornstein-Uhlenbeck")
    logger.info("=" * 70)
    
    X0 = 0.0
    theta_true = 2.0
    mu_true_ou = 0.5
    sigma_true_ou = 0.3
    
    logger.info("")
    logger.info("Generating OU training data...")
    
    ou_train = simulate_ou_paths(
        X0, theta_true, mu_true_ou, sigma_true_ou, config.T, config.n_steps,
        config.n_train_paths, seed=42
    )
    
    ou_test = simulate_ou_paths(
        X0, theta_true, mu_true_ou, sigma_true_ou, config.T, config.n_steps,
        config.n_test_paths, seed=123
    )
    
    # Train
    logger.info("Training Neural SDE on OU data...")
    
    nsde_ou = NeuralSDE(
        state_dim=1,
        hidden_dim=config.hidden_dim,
        learning_rate=config.learning_rate,
    )
    
    nsde_ou.fit(ou_train, config.T, epochs=config.epochs)
    
    # Evaluate
    nsde_ou_paths = nsde_ou.simulate(
        X0=np.array([X0]),
        T=config.T,
        n_steps=config.n_steps,
        n_paths=config.n_test_paths,
        seed=123,
    )
    
    true_terminal_ou = ou_test[:, -1, 0]
    pred_terminal_ou = nsde_ou_paths[:, -1, 0]
    
    results['ou'] = {
        'true_mean': np.mean(true_terminal_ou),
        'pred_mean': np.mean(pred_terminal_ou),
        'true_std': np.std(true_terminal_ou),
        'pred_std': np.std(pred_terminal_ou),
        'true_paths': ou_test,
        'pred_paths': nsde_ou_paths,
        'train_loss': nsde_ou.train_losses,
    }
    
    models['ou'] = nsde_ou
    
    logger.info(f"  True terminal: mean={results['ou']['true_mean']:.3f}, std={results['ou']['true_std']:.3f}")
    logger.info(f"  Pred terminal: mean={results['ou']['pred_mean']:.3f}, std={results['ou']['pred_std']:.3f}")
    
    # -------------------------------------------------------------------------
    # Summary metrics
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 3: Summary Metrics")
    logger.info("=" * 70)
    
    logger.info("")
    logger.info(f"{'Model':<10} {'True Mean':>12} {'Pred Mean':>12} {'Mean Err':>10} {'Std Err':>10}")
    logger.info("-" * 60)
    
    for name in ['gbm', 'ou']:
        r = results[name]
        mean_err = abs(r['true_mean'] - r['pred_mean']) / abs(r['true_mean'] + 1e-6) * 100
        std_err = abs(r['true_std'] - r['pred_std']) / r['true_std'] * 100
        logger.info(
            f"{name.upper():<10} {r['true_mean']:>12.4f} {r['pred_mean']:>12.4f} "
            f"{mean_err:>9.1f}% {std_err:>9.1f}%"
        )
    
    return results, models


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_neural_sde(results: dict, models: dict) -> None:
    """Visualize Neural SDE results."""
    if not MATPLOTLIB_AVAILABLE or not ENABLE_PLOTTING:
        logger.info("Skipping plots (matplotlib not available or disabled)")
        return
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 4: Visualization")
    logger.info("=" * 70)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # -------------------------------------------------------------------------
    # Row 1: GBM
    # -------------------------------------------------------------------------
    
    # Sample paths
    ax = axes[0, 0]
    for i in range(min(20, results['gbm']['true_paths'].shape[0])):
        ax.plot(results['gbm']['true_paths'][i, :, 0], alpha=0.3, color='#2E86AB')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Price')
    ax.set_title('GBM: True Paths')
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    for i in range(min(20, results['gbm']['pred_paths'].shape[0])):
        ax.plot(results['gbm']['pred_paths'][i, :, 0], alpha=0.3, color='#E94F37')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Price')
    ax.set_title('GBM: Neural SDE Paths')
    ax.grid(True, alpha=0.3)
    
    # Terminal distribution
    ax = axes[0, 2]
    ax.hist(results['gbm']['true_paths'][:, -1, 0], bins=30, alpha=0.5, label='True', color='#2E86AB', density=True)
    ax.hist(results['gbm']['pred_paths'][:, -1, 0], bins=30, alpha=0.5, label='Neural SDE', color='#E94F37', density=True)
    ax.set_xlabel('Terminal Value')
    ax.set_ylabel('Density')
    ax.set_title('GBM: Terminal Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Row 2: OU
    # -------------------------------------------------------------------------
    
    ax = axes[1, 0]
    for i in range(min(20, results['ou']['true_paths'].shape[0])):
        ax.plot(results['ou']['true_paths'][i, :, 0], alpha=0.3, color='#2E86AB')
    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=2, label='μ=0.5')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.set_title('OU: True Paths (mean-reverting)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    for i in range(min(20, results['ou']['pred_paths'].shape[0])):
        ax.plot(results['ou']['pred_paths'][i, :, 0], alpha=0.3, color='#E94F37')
    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=2, label='μ=0.5')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.set_title('OU: Neural SDE Paths')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Training loss
    ax = axes[1, 2]
    ax.plot(results['gbm']['train_loss'], label='GBM', color='#2E86AB')
    ax.plot(results['ou']['train_loss'], label='OU', color='#E94F37')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
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
    │  1. Neural SDE Formulation:                                         │
    │     - dX = μ_θ(X,t) dt + σ_φ(X,t) dW                               │
    │     - μ_θ, σ_φ are neural networks                                  │
    │     - Learns dynamics from observed paths                           │
    │                                                                      │
    │  2. Training Approach:                                              │
    │     - Maximum likelihood via score matching                         │
    │     - Euler-Maruyama discretization                                 │
    │     - Backprop through SDE solution                                 │
    │                                                                      │
    │  3. Architecture Design:                                            │
    │     - Separate drift and diffusion networks                         │
    │     - Softplus for positive diffusion                               │
    │     - Input: (X, t) for time-varying dynamics                       │
    │                                                                      │
    │  4. Production Considerations:                                      │
    │     - Regularization to prevent overfitting                         │
    │     - Validation on out-of-sample paths                             │
    │     - Compare to parametric models                                  │
    │     - Use for pricing, simulation, risk                             │
    │                                                                      │
    │  Neural SDEs offer flexible, data-driven dynamics modeling          │
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
        # Run Neural SDE experiments
        results, models = run_neural_sde()
        
        # Visualization
        visualize_neural_sde(results, models)
        
        # Summary
        print_summary()
        
        logger.info("Example completed successfully!")
        
    except Exception as e:
        logger.exception(f"Example failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Neural SDE Example",
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

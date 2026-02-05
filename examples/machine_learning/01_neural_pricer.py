#!/usr/bin/env python3
"""
===============================================================================
Machine Learning: Neural Network Option Pricer
===============================================================================

This example demonstrates training a neural network to price options faster
than traditional methods while maintaining accuracy.

Learning Objectives
-------------------
1. **ML for Pricing**: Understand when and why to use neural network pricers
2. **Data Generation**: Create training data from analytical/MC pricers
3. **Model Architecture**: Design networks for financial applications
4. **Inference**: Use trained models for real-time pricing

Mathematical Framework
----------------------
The goal is to learn a function f_θ such that:
    f_θ(S, K, T, σ, r) ≈ V_BSM(S, K, T, σ, r)

Loss function:
    L(θ) = E[(f_θ(x) - V_true(x))²]

For exotics where no closed-form exists, train on MC prices:
    f_θ(x) ≈ E[payoff | x]

Input features typically include:
    - Moneyness: K/S or log(K/S)
    - Time to expiry: T
    - Volatility: σ
    - Interest rate: r
    - Option parameters (barrier, etc.)

Production Context
------------------
At a hedge fund:
- Neural pricers enable real-time pricing of large portfolios
- 10-1000x speedup vs MC for exotic options
- Used in risk calculation, scenario analysis, and optimization
- Requires careful validation against benchmark pricers

Prerequisites
-------------
- Basic pricing examples (examples/pricing/)
- Understanding of neural networks (PyTorch/TensorFlow)

Run This Example
----------------
    cd /path/to/QuantStrata
    PYTHONPATH=. python examples/machine_learning/01_neural_pricer.py

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
from typing import List, Optional, Tuple

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
# BLACK-SCHOLES BENCHMARK
# =============================================================================

def bs_call_price(S: float, K: float, T: float, sigma: float, r: float) -> float:
    """Black-Scholes call price."""
    if T <= 0:
        return max(S - K, 0)
    
    from scipy.stats import norm
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S: float, K: float, T: float, sigma: float, r: float) -> float:
    """Black-Scholes put price."""
    if T <= 0:
        return max(K - S, 0)
    
    from scipy.stats import norm
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# =============================================================================
# DATA GENERATION
# =============================================================================

@dataclass
class PricingDataConfig:
    """Configuration for training data generation."""
    # Parameter ranges
    spot_range: Tuple[float, float] = (80.0, 120.0)
    strike_range: Tuple[float, float] = (80.0, 120.0)
    expiry_range: Tuple[float, float] = (0.1, 2.0)
    vol_range: Tuple[float, float] = (0.05, 0.50)
    rate_range: Tuple[float, float] = (0.0, 0.10)
    
    # Dataset sizes
    n_train: int = 50000
    n_val: int = 10000
    n_test: int = 10000


def generate_training_data(config: PricingDataConfig, seed: int = 42) -> dict:
    """
    Generate training data for neural pricer.
    
    Returns
    -------
    dict
        Dictionary with X_train, y_train, X_val, y_val, X_test, y_test.
    """
    np.random.seed(seed)
    
    def sample_batch(n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample a batch of inputs and compute prices."""
        # Sample parameters
        S = np.random.uniform(*config.spot_range, n)
        K = np.random.uniform(*config.strike_range, n)
        T = np.random.uniform(*config.expiry_range, n)
        sigma = np.random.uniform(*config.vol_range, n)
        r = np.random.uniform(*config.rate_range, n)
        
        # Compute prices
        prices = np.array([
            bs_call_price(s, k, t, sig, rate)
            for s, k, t, sig, rate in zip(S, K, T, sigma, r)
        ])
        
        # Create feature matrix
        # Use normalized features for better training
        moneyness = np.log(K / S)  # Log-moneyness
        X = np.column_stack([moneyness, T, sigma, r])
        
        # Normalize price by spot
        y = prices / S
        
        return X, y
    
    X_train, y_train = sample_batch(config.n_train)
    X_val, y_val = sample_batch(config.n_val)
    X_test, y_test = sample_batch(config.n_test)
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
    }


# =============================================================================
# SIMPLE NEURAL NETWORK (NumPy-based)
# =============================================================================

class SimpleNeuralPricer:
    """
    Simple feedforward neural network for option pricing.
    
    Uses NumPy for portability (no PyTorch/TensorFlow dependency).
    For production, use src/machine_learning/models/pricing/.
    
    Architecture:
        Input (4) -> Dense(64) -> ReLU -> Dense(64) -> ReLU -> Dense(1)
    
    Example:
        model = SimpleNeuralPricer(input_dim=4, hidden_dims=[64, 64])
        model.train(X_train, y_train, X_val, y_val, epochs=100)
        predictions = model.predict(X_test)
    """
    
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dims: List[int] = None,
        learning_rate: float = 0.001,
    ) -> None:
        """Initialize neural network."""
        if hidden_dims is None:
            hidden_dims = [64, 64]
        
        self.learning_rate = learning_rate
        self.layers = []
        
        # Initialize weights
        dims = [input_dim] + hidden_dims + [1]
        for i in range(len(dims) - 1):
            # Xavier initialization
            scale = np.sqrt(2.0 / (dims[i] + dims[i + 1]))
            W = np.random.randn(dims[i], dims[i + 1]) * scale
            b = np.zeros((1, dims[i + 1]))
            self.layers.append({'W': W, 'b': b})
        
        # Training history
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation."""
        return np.maximum(0, x)
    
    def _relu_grad(self, x: np.ndarray) -> np.ndarray:
        """ReLU gradient."""
        return (x > 0).astype(float)
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, List]:
        """Forward pass."""
        activations = [X]
        a = X
        
        for i, layer in enumerate(self.layers):
            z = a @ layer['W'] + layer['b']
            if i < len(self.layers) - 1:
                a = self._relu(z)
            else:
                a = z  # Linear output
            activations.append(a)
        
        return a, activations
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        output, _ = self.forward(X)
        return output.flatten()
    
    def _backward(
        self,
        X: np.ndarray,
        y: np.ndarray,
        activations: List,
    ) -> List[dict]:
        """Backward pass."""
        m = X.shape[0]
        gradients = []
        
        # Output layer gradient
        y = y.reshape(-1, 1)
        dz = activations[-1] - y  # MSE gradient
        
        for i in range(len(self.layers) - 1, -1, -1):
            a_prev = activations[i]
            
            dW = a_prev.T @ dz / m
            db = np.sum(dz, axis=0, keepdims=True) / m
            
            gradients.insert(0, {'dW': dW, 'db': db})
            
            if i > 0:
                da_prev = dz @ self.layers[i]['W'].T
                # Pre-ReLU gradient
                z_prev = a_prev
                dz = da_prev * self._relu_grad(z_prev)
        
        return gradients
    
    def _update_weights(self, gradients: List[dict]) -> None:
        """Update weights using gradients."""
        for layer, grad in zip(self.layers, gradients):
            layer['W'] -= self.learning_rate * grad['dW']
            layer['b'] -= self.learning_rate * grad['db']
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 256,
        verbose: bool = True,
    ) -> None:
        """
        Train the neural network.
        
        Parameters
        ----------
        X_train, y_train : np.ndarray
            Training data.
        X_val, y_val : np.ndarray
            Validation data.
        epochs : int
            Number of training epochs.
        batch_size : int
            Mini-batch size.
        verbose : bool
            Print progress.
        """
        n_samples = X_train.shape[0]
        n_batches = n_samples // batch_size
        
        for epoch in range(epochs):
            # Shuffle data
            idx = np.random.permutation(n_samples)
            X_shuffled = X_train[idx]
            y_shuffled = y_train[idx]
            
            epoch_loss = 0.0
            
            for i in range(n_batches):
                start = i * batch_size
                end = start + batch_size
                
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                # Forward pass
                output, activations = self.forward(X_batch)
                
                # Compute loss
                loss = np.mean((output.flatten() - y_batch) ** 2)
                epoch_loss += loss
                
                # Backward pass
                gradients = self._backward(X_batch, y_batch, activations)
                
                # Update weights
                self._update_weights(gradients)
            
            # Record losses
            train_loss = epoch_loss / n_batches
            val_pred = self.predict(X_val)
            val_loss = np.mean((val_pred - y_val) ** 2)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            if verbose and (epoch + 1) % 20 == 0:
                logger.info(
                    f"Epoch {epoch + 1:>3}/{epochs}: "
                    f"Train Loss = {train_loss:.6f}, "
                    f"Val Loss = {val_loss:.6f}"
                )


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def run_neural_pricer() -> Tuple[SimpleNeuralPricer, dict, dict]:
    """
    Run the neural pricer workflow.
    
    Returns
    -------
    Tuple
        Trained model, data, and evaluation metrics.
    """
    logger.info("=" * 70)
    logger.info("SECTION 1: Data Generation")
    logger.info("=" * 70)
    
    config = PricingDataConfig(
        n_train=30000,
        n_val=5000,
        n_test=5000,
    )
    
    logger.info("")
    logger.info("Generating training data from BSM prices...")
    data = generate_training_data(config)
    
    logger.info(f"  Train samples: {config.n_train:,}")
    logger.info(f"  Val samples:   {config.n_val:,}")
    logger.info(f"  Test samples:  {config.n_test:,}")
    logger.info(f"  Features:      moneyness, T, σ, r")
    
    # Train model
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 2: Training Neural Pricer")
    logger.info("=" * 70)
    
    model = SimpleNeuralPricer(
        input_dim=4,
        hidden_dims=[64, 64],
        learning_rate=0.01,
    )
    
    logger.info("")
    logger.info("Model architecture: 4 -> 64 -> 64 -> 1")
    logger.info("Training...")
    logger.info("")
    
    model.train(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val'],
        epochs=100,
        batch_size=256,
    )
    
    # Evaluate
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 3: Evaluation")
    logger.info("=" * 70)
    
    # Test set evaluation
    y_pred = model.predict(data['X_test'])
    y_true = data['y_test']
    
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred - y_true))
    mape = np.mean(np.abs((y_pred - y_true) / (y_true + 1e-8))) * 100
    r2 = 1 - np.sum((y_pred - y_true) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2,
    }
    
    logger.info("")
    logger.info("Test Set Metrics:")
    logger.info(f"  RMSE:  {rmse:.6f}")
    logger.info(f"  MAE:   {mae:.6f}")
    logger.info(f"  MAPE:  {mape:.2f}%")
    logger.info(f"  R²:    {r2:.4f}")
    
    # Speed comparison
    logger.info("")
    logger.info("Speed Comparison:")
    
    # BSM timing
    start = time.time()
    for _ in range(1000):
        bs_call_price(100, 100, 1.0, 0.2, 0.05)
    bsm_time = (time.time() - start) / 1000 * 1000  # ms
    
    # Neural timing
    test_input = data['X_test'][:1000]
    start = time.time()
    _ = model.predict(test_input)
    neural_time = (time.time() - start) / 1000 * 1000  # ms per sample
    
    logger.info(f"  BSM (single):    {bsm_time:.4f} ms")
    logger.info(f"  Neural (batch):  {neural_time:.4f} ms")
    logger.info(f"  Speedup:         {bsm_time/neural_time:.1f}x")
    
    return model, data, metrics


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_pricer(model: SimpleNeuralPricer, data: dict, metrics: dict) -> None:
    """Visualize neural pricer results."""
    if not MATPLOTLIB_AVAILABLE or not ENABLE_PLOTTING:
        logger.info("Skipping plots (matplotlib not available or disabled)")
        return
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 4: Visualization")
    logger.info("=" * 70)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    y_pred = model.predict(data['X_test'])
    y_true = data['y_test']
    
    # -------------------------------------------------------------------------
    # Plot 1: Training history
    # -------------------------------------------------------------------------
    ax = axes[0, 0]
    ax.plot(model.train_losses, label='Train', color='#2E86AB')
    ax.plot(model.val_losses, label='Validation', color='#E94F37')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Training History')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 2: Predicted vs Actual
    # -------------------------------------------------------------------------
    ax = axes[0, 1]
    ax.scatter(y_true * 100, y_pred * 100, alpha=0.3, s=10, color='#2E86AB')
    ax.plot([0, 30], [0, 30], 'r--', linewidth=2, label='Perfect fit')
    ax.set_xlabel('BSM Price (% of spot)')
    ax.set_ylabel('Neural Price (% of spot)')
    ax.set_title(f'Predicted vs Actual (R² = {metrics["r2"]:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 3: Error distribution
    # -------------------------------------------------------------------------
    ax = axes[1, 0]
    errors = (y_pred - y_true) * 100
    ax.hist(errors, bins=50, density=True, alpha=0.7, color='#2E86AB')
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Pricing Error (% of spot)')
    ax.set_ylabel('Density')
    ax.set_title(f'Error Distribution (MAE = {metrics["mae"]*100:.3f}%)')
    ax.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 4: Error vs moneyness
    # -------------------------------------------------------------------------
    ax = axes[1, 1]
    moneyness = data['X_test'][:, 0]  # Log-moneyness
    ax.scatter(moneyness, np.abs(errors), alpha=0.3, s=10, color='#2E86AB')
    ax.set_xlabel('Log-Moneyness (ln(K/S))')
    ax.set_ylabel('Absolute Error (% of spot)')
    ax.set_title('Error vs Moneyness')
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
    │  1. Neural Pricer Design:                                           │
    │     - Input: normalized features (moneyness, T, σ, r)               │
    │     - Output: price / spot (normalized)                             │
    │     - Architecture: feedforward network with ReLU                   │
    │                                                                      │
    │  2. Training Data:                                                  │
    │     - Generate from analytical pricer (BSM) or MC                   │
    │     - Cover full parameter space uniformly                          │
    │     - Normalize inputs and outputs for training                     │
    │                                                                      │
    │  3. Performance:                                                    │
    │     - Accuracy: <0.1% error achievable with enough data             │
    │     - Speed: 10-100x faster than MC for exotics                     │
    │     - Scalability: batch inference for large portfolios             │
    │                                                                      │
    │  4. Production Use:                                                 │
    │     - Real-time pricing for risk calculations                       │
    │     - Greeks via automatic differentiation                          │
    │     - Regular retraining and validation                             │
    │                                                                      │
    │  NEXT: See 02_calibration_ml.py for model calibration               │
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
        # Run neural pricer workflow
        model, data, metrics = run_neural_pricer()
        
        # Visualization
        visualize_pricer(model, data, metrics)
        
        # Summary
        print_summary()
        
        logger.info("Example completed successfully!")
        
    except Exception as e:
        logger.exception(f"Example failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Neural Network Option Pricer Example",
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

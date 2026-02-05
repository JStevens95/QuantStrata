#!/usr/bin/env python3
"""
===============================================================================
Machine Learning: Model Calibration with Neural Networks
===============================================================================

This example demonstrates using machine learning to accelerate model calibration
- the inverse problem of finding model parameters from market prices.

Learning Objectives
-------------------
1. **Calibration Problem**: Understand the inverse problem formulation
2. **ML for Calibration**: Train networks to map prices → parameters
3. **Speed vs Accuracy**: Trade-offs in ML-based calibration
4. **Hybrid Approaches**: ML initialization + optimization refinement

Mathematical Framework
----------------------
Traditional calibration solves:
    θ* = argmin_θ Σᵢ (V_model(θ; xᵢ) - V_market(xᵢ))²

This requires iterative optimization calling the pricer many times.

ML approach learns the inverse mapping directly:
    θ̂ = f_NN(V_market)

Where f_NN is trained on synthetic data:
    Training pairs: (V_model(θ; x), θ) for random θ

Production Context
------------------
At a hedge fund:
- Calibration is needed for marking, risk, and Greeks
- Heston, SABR, local vol all require calibration
- Speed: minutes → milliseconds with ML
- Often combined with optimization refinement

Prerequisites
-------------
- Neural pricing (01_neural_pricer.py)
- Understanding of stochastic vol models

Run This Example
----------------
    cd /path/to/QuantStrata
    PYTHONPATH=. python examples/machine_learning/02_calibration_ml.py

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
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

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
# HESTON MODEL (SIMPLIFIED)
# =============================================================================

def heston_call_price_approx(
    S: float,
    K: float,
    T: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
) -> float:
    """
    Simplified Heston call price using moment matching.
    
    For production, use proper characteristic function approach.
    This approximation gives qualitatively correct behavior.
    
    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike.
    T : float
        Time to expiry.
    r : float
        Risk-free rate.
    v0 : float
        Initial variance.
    kappa : float
        Mean reversion speed.
    theta : float
        Long-term variance.
    sigma : float
        Vol of vol.
    rho : float
        Spot-vol correlation.
    """
    from scipy.stats import norm
    
    # Effective variance (first moment approximation)
    var_T = v0 * np.exp(-kappa * T) + theta * (1 - np.exp(-kappa * T))
    sigma_eff = np.sqrt(var_T)
    
    # Skew adjustment (from rho)
    skew = rho * sigma * np.sqrt(T) / (2 * sigma_eff)
    
    # BSM-like pricing with adjustments
    if T <= 0 or sigma_eff <= 0:
        return max(S - K, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma_eff**2) * T) / (sigma_eff * np.sqrt(T))
    d2 = d1 - sigma_eff * np.sqrt(T)
    
    # Base price
    price = S * norm.cdf(d1 + skew) - K * np.exp(-r * T) * norm.cdf(d2 + skew)
    
    return max(price, 0)


def heston_implied_vol_smile(
    T: float,
    moneyness_range: np.ndarray,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    r: float = 0.05,
    S: float = 100.0,
) -> np.ndarray:
    """Generate implied vol smile from Heston parameters."""
    from scipy.optimize import brentq
    from scipy.stats import norm
    
    def bs_call(S, K, T, vol, r):
        if vol <= 0:
            return max(S - K, 0)
        d1 = (np.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
        d2 = d1 - vol * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    ivs = []
    for m in moneyness_range:
        K = S * np.exp(m)
        heston_price = heston_call_price_approx(S, K, T, r, v0, kappa, theta, sigma, rho)
        
        # Implied vol via bisection
        try:
            iv = brentq(
                lambda vol: bs_call(S, K, T, vol, r) - heston_price,
                0.01, 2.0
            )
        except:
            iv = np.sqrt(v0)
        
        ivs.append(iv)
    
    return np.array(ivs)


# =============================================================================
# CALIBRATION DATA GENERATION
# =============================================================================

@dataclass
class CalibrationDataConfig:
    """Configuration for calibration data generation."""
    # Heston parameter ranges
    v0_range: Tuple[float, float] = (0.01, 0.16)  # 10-40% vol
    kappa_range: Tuple[float, float] = (0.5, 5.0)
    theta_range: Tuple[float, float] = (0.01, 0.16)
    sigma_range: Tuple[float, float] = (0.1, 1.0)
    rho_range: Tuple[float, float] = (-0.9, -0.1)
    
    # Smile grid
    moneyness_grid: np.ndarray = None
    expiry: float = 0.5
    
    # Dataset sizes
    n_train: int = 20000
    n_val: int = 5000
    n_test: int = 5000
    
    def __post_init__(self):
        if self.moneyness_grid is None:
            self.moneyness_grid = np.linspace(-0.3, 0.3, 7)  # -30% to +30%


def generate_calibration_data(config: CalibrationDataConfig, seed: int = 42) -> dict:
    """
    Generate training data for calibration network.
    
    Input: implied vol smile
    Output: Heston parameters (v0, kappa, theta, sigma, rho)
    """
    np.random.seed(seed)
    
    def sample_batch(n: int) -> Tuple[np.ndarray, np.ndarray]:
        X_list = []
        y_list = []
        
        for _ in range(n):
            # Sample Heston parameters
            v0 = np.random.uniform(*config.v0_range)
            kappa = np.random.uniform(*config.kappa_range)
            theta = np.random.uniform(*config.theta_range)
            sigma = np.random.uniform(*config.sigma_range)
            rho = np.random.uniform(*config.rho_range)
            
            # Generate smile
            ivs = heston_implied_vol_smile(
                T=config.expiry,
                moneyness_range=config.moneyness_grid,
                v0=v0,
                kappa=kappa,
                theta=theta,
                sigma=sigma,
                rho=rho,
            )
            
            X_list.append(ivs)
            y_list.append([v0, kappa, theta, sigma, rho])
        
        return np.array(X_list), np.array(y_list)
    
    logger.info("Generating calibration training data...")
    X_train, y_train = sample_batch(config.n_train)
    
    logger.info("Generating validation data...")
    X_val, y_val = sample_batch(config.n_val)
    
    logger.info("Generating test data...")
    X_test, y_test = sample_batch(config.n_test)
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'moneyness_grid': config.moneyness_grid,
    }


# =============================================================================
# CALIBRATION NEURAL NETWORK
# =============================================================================

class CalibrationNetwork:
    """
    Neural network for Heston model calibration.
    
    Maps implied vol smile → model parameters.
    
    Architecture:
        Input (n_strikes) -> Dense(128) -> ReLU -> Dense(64) -> ReLU -> Dense(5)
    
    Example:
        model = CalibrationNetwork(input_dim=7, output_dim=5)
        model.train(X_train, y_train, X_val, y_val, epochs=100)
        params = model.predict(iv_smile)
    """
    
    def __init__(
        self,
        input_dim: int = 7,
        output_dim: int = 5,
        hidden_dims: List[int] = None,
        learning_rate: float = 0.001,
    ) -> None:
        """Initialize calibration network."""
        if hidden_dims is None:
            hidden_dims = [128, 64]
        
        self.learning_rate = learning_rate
        self.layers = []
        
        # Initialize weights
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / (dims[i] + dims[i + 1]))
            W = np.random.randn(dims[i], dims[i + 1]) * scale
            b = np.zeros((1, dims[i + 1]))
            self.layers.append({'W': W, 'b': b})
        
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        
        # Output normalization (to scale parameters to similar ranges)
        self.y_mean = None
        self.y_std = None
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def _relu_grad(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, List]:
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
        output, _ = self.forward(X)
        # Denormalize output
        if self.y_mean is not None:
            output = output * self.y_std + self.y_mean
        return output
    
    def _backward(self, X: np.ndarray, y: np.ndarray, activations: List) -> List[dict]:
        m = X.shape[0]
        gradients = []
        
        dz = activations[-1] - y  # MSE gradient
        
        for i in range(len(self.layers) - 1, -1, -1):
            a_prev = activations[i]
            
            dW = a_prev.T @ dz / m
            db = np.sum(dz, axis=0, keepdims=True) / m
            
            gradients.insert(0, {'dW': dW, 'db': db})
            
            if i > 0:
                da_prev = dz @ self.layers[i]['W'].T
                z_prev = a_prev
                dz = da_prev * self._relu_grad(z_prev)
        
        return gradients
    
    def _update_weights(self, gradients: List[dict]) -> None:
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
        batch_size: int = 128,
        verbose: bool = True,
    ) -> None:
        """Train the calibration network."""
        # Normalize targets
        self.y_mean = y_train.mean(axis=0)
        self.y_std = y_train.std(axis=0) + 1e-8
        y_train_norm = (y_train - self.y_mean) / self.y_std
        y_val_norm = (y_val - self.y_mean) / self.y_std
        
        n_samples = X_train.shape[0]
        n_batches = n_samples // batch_size
        
        for epoch in range(epochs):
            idx = np.random.permutation(n_samples)
            X_shuffled = X_train[idx]
            y_shuffled = y_train_norm[idx]
            
            epoch_loss = 0.0
            
            for i in range(n_batches):
                start = i * batch_size
                end = start + batch_size
                
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                output, activations = self.forward(X_batch)
                loss = np.mean((output - y_batch) ** 2)
                epoch_loss += loss
                
                gradients = self._backward(X_batch, y_batch, activations)
                self._update_weights(gradients)
            
            train_loss = epoch_loss / n_batches
            val_pred, _ = self.forward(X_val)
            val_loss = np.mean((val_pred - y_val_norm) ** 2)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            if verbose and (epoch + 1) % 20 == 0:
                logger.info(
                    f"Epoch {epoch + 1:>3}/{epochs}: "
                    f"Train Loss = {train_loss:.6f}, "
                    f"Val Loss = {val_loss:.6f}"
                )


# =============================================================================
# TRADITIONAL CALIBRATION
# =============================================================================

def calibrate_heston_traditional(
    target_ivs: np.ndarray,
    moneyness_grid: np.ndarray,
    T: float = 0.5,
    max_iter: int = 100,
) -> Tuple[np.ndarray, float]:
    """
    Traditional optimization-based Heston calibration.
    
    Returns
    -------
    Tuple
        (fitted parameters, calibration time in seconds)
    """
    def objective(params):
        v0, kappa, theta, sigma, rho = params
        if v0 < 0 or kappa < 0 or theta < 0 or sigma < 0:
            return 1e10
        if rho < -1 or rho > 1:
            return 1e10
        
        model_ivs = heston_implied_vol_smile(
            T=T,
            moneyness_range=moneyness_grid,
            v0=v0,
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            rho=rho,
        )
        
        return np.sum((model_ivs - target_ivs) ** 2)
    
    # Initial guess
    x0 = [0.04, 2.0, 0.04, 0.5, -0.5]
    
    start = time.time()
    result = minimize(
        objective,
        x0,
        method='Nelder-Mead',
        options={'maxiter': max_iter}
    )
    elapsed = time.time() - start
    
    return result.x, elapsed


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def run_calibration_ml() -> Tuple[CalibrationNetwork, dict, dict]:
    """
    Run the ML calibration workflow.
    
    Returns
    -------
    Tuple
        Trained model, data, and evaluation metrics.
    """
    logger.info("=" * 70)
    logger.info("SECTION 1: Data Generation")
    logger.info("=" * 70)
    
    config = CalibrationDataConfig(
        n_train=15000,
        n_val=3000,
        n_test=3000,
    )
    
    logger.info("")
    data = generate_calibration_data(config)
    
    logger.info("")
    logger.info(f"  Train samples: {config.n_train:,}")
    logger.info(f"  Val samples:   {config.n_val:,}")
    logger.info(f"  Test samples:  {config.n_test:,}")
    logger.info(f"  IV strikes:    {len(config.moneyness_grid)}")
    
    # Train model
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 2: Training Calibration Network")
    logger.info("=" * 70)
    
    model = CalibrationNetwork(
        input_dim=len(config.moneyness_grid),
        output_dim=5,
        hidden_dims=[128, 64],
        learning_rate=0.01,
    )
    
    logger.info("")
    logger.info(f"Model architecture: {len(config.moneyness_grid)} -> 128 -> 64 -> 5")
    logger.info("Training...")
    logger.info("")
    
    model.train(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val'],
        epochs=100,
        batch_size=128,
    )
    
    # Evaluate
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 3: Evaluation")
    logger.info("=" * 70)
    
    # Test set evaluation
    y_pred = model.predict(data['X_test'])
    y_true = data['y_test']
    
    param_names = ['v0', 'kappa', 'theta', 'sigma', 'rho']
    
    logger.info("")
    logger.info("Per-Parameter Errors:")
    logger.info("-" * 50)
    logger.info(f"{'Parameter':<10} {'MAE':>10} {'RMSE':>10} {'MAPE':>10}")
    logger.info("-" * 50)
    
    metrics = {}
    for i, name in enumerate(param_names):
        mae = np.mean(np.abs(y_pred[:, i] - y_true[:, i]))
        rmse = np.sqrt(np.mean((y_pred[:, i] - y_true[:, i]) ** 2))
        mape = np.mean(np.abs((y_pred[:, i] - y_true[:, i]) / (y_true[:, i] + 1e-8))) * 100
        
        metrics[f'{name}_mae'] = mae
        metrics[f'{name}_rmse'] = rmse
        metrics[f'{name}_mape'] = mape
        
        logger.info(f"{name:<10} {mae:>10.4f} {rmse:>10.4f} {mape:>9.1f}%")
    
    logger.info("-" * 50)
    
    # Speed comparison
    logger.info("")
    logger.info("Speed Comparison:")
    
    # ML calibration timing
    start = time.time()
    _ = model.predict(data['X_test'][:100])
    ml_time = (time.time() - start) / 100
    
    # Traditional calibration timing (single sample)
    sample_ivs = data['X_test'][0]
    _, trad_time = calibrate_heston_traditional(
        sample_ivs,
        config.moneyness_grid,
        T=config.expiry,
        max_iter=100,
    )
    
    logger.info(f"  ML (per sample):         {ml_time*1000:.3f} ms")
    logger.info(f"  Traditional (per sample): {trad_time*1000:.1f} ms")
    logger.info(f"  Speedup:                  {trad_time/ml_time:.0f}x")
    
    metrics['ml_time'] = ml_time
    metrics['trad_time'] = trad_time
    
    return model, data, metrics


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_calibration(model: CalibrationNetwork, data: dict, metrics: dict) -> None:
    """Visualize calibration results."""
    if not MATPLOTLIB_AVAILABLE or not ENABLE_PLOTTING:
        logger.info("Skipping plots (matplotlib not available or disabled)")
        return
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 4: Visualization")
    logger.info("=" * 70)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    y_pred = model.predict(data['X_test'])
    y_true = data['y_test']
    param_names = ['v₀', 'κ', 'θ', 'σ_v', 'ρ']
    
    # -------------------------------------------------------------------------
    # Plots 1-5: Parameter predictions
    # -------------------------------------------------------------------------
    for i, (ax, name) in enumerate(zip(axes.flat[:5], param_names)):
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.3, s=10, color='#2E86AB')
        
        # Perfect fit line
        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        ax.set_xlabel(f'True {name}')
        ax.set_ylabel(f'Predicted {name}')
        ax.set_title(f'{name} (MAPE={metrics[f"{["v0", "kappa", "theta", "sigma", "rho"][i]}_mape"]:.1f}%)')
        ax.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 6: Training history
    # -------------------------------------------------------------------------
    ax = axes[1, 2]
    ax.plot(model.train_losses, label='Train', color='#2E86AB')
    ax.plot(model.val_losses, label='Validation', color='#E94F37')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Training History')
    ax.legend()
    ax.set_yscale('log')
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
    │  1. Calibration as Inverse Problem:                                 │
    │     - Forward: parameters → prices                                  │
    │     - Inverse: prices → parameters (harder!)                        │
    │     - ML learns the inverse mapping directly                        │
    │                                                                      │
    │  2. Network Design:                                                 │
    │     - Input: vol smile (or surface)                                 │
    │     - Output: model parameters                                      │
    │     - Normalize both for training stability                         │
    │                                                                      │
    │  3. Training Data:                                                  │
    │     - Generate synthetic smiles from random parameters              │
    │     - Cover full parameter space                                    │
    │     - Include realistic market conditions                           │
    │                                                                      │
    │  4. Production Pipeline:                                            │
    │     - ML for fast initial guess                                     │
    │     - Optional: refine with 1-2 optimization iterations             │
    │     - Validate: check model prices match market                     │
    │                                                                      │
    │  Speed improvement: 100-1000x vs traditional optimization           │
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
        # Run calibration ML workflow
        model, data, metrics = run_calibration_ml()
        
        # Visualization
        visualize_calibration(model, data, metrics)
        
        # Summary
        print_summary()
        
        logger.info("Example completed successfully!")
        
    except Exception as e:
        logger.exception(f"Example failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ML Calibration Example",
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

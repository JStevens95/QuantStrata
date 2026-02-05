#!/usr/bin/env python3
"""
===============================================================================
Machine Learning: Neural Stochastic Differential Equations
===============================================================================

This example demonstrates QuantStrata's Neural SDE module for learning
asset dynamics from historical data.

Learning Objectives
-------------------
1. **Neural SDE Architecture**: Use NeuralSDEDynamics for learnable dynamics
2. **Network Design**: NeuralDriftNetwork and NeuralDiffusionNetwork
3. **Training Pipeline**: Use NeuralSDETrainer with TrainingConfig
4. **Validation**: Compare learned dynamics to known parametric models

Mathematical Framework
----------------------
Standard SDE:
    dS_t = μ(S_t, t) dt + σ(S_t, t) dW_t

Neural SDE:
    dS_t = μ_θ(S_t, t) dt + σ_φ(S_t, t) dW_t

Where μ_θ and σ_φ are neural networks:
- μ_θ: Drift network (any real output)
- σ_φ: Diffusion network (positive output via softplus)

Training minimizes:
    L = moment_weight × MomentMatchingLoss + pathwise_weight × PathwiseLoss

Production Context
------------------
At a hedge fund:
- Neural SDEs can capture complex volatility dynamics
- More flexible than parametric models (Heston, SABR)
- Used for pricing, risk simulation, scenario generation
- Requires careful validation against known benchmarks

Prerequisites
-------------
- SDE fundamentals
- Neural network basics
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
from typing import Dict, List, Optional, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# -----------------------------------------------------------------------------
# QuantStrata imports - using existing neural_sde module
# -----------------------------------------------------------------------------
from src.models.neural_sde.dynamics import NeuralSDEDynamics, NeuralSDEConfig
from src.models.neural_sde.networks import NeuralDriftNetwork, NeuralDiffusionNetwork
from src.models.neural_sde.training.trainer import (
    NeuralSDETrainer,
    TrainingConfig,
    TrainingResult,
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
# SYNTHETIC DATA GENERATION (TARGET MODELS)
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
    
    dS = μ·S·dt + σ·S·dW
    
    Parameters
    ----------
    S0 : float
        Initial spot price.
    mu : float
        Drift (annualized).
    sigma : float
        Volatility (annualized).
    T : float
        Time horizon in years.
    n_steps : int
        Number of time steps.
    n_paths : int
        Number of paths to simulate.
    seed : int, optional
        Random seed.
    
    Returns
    -------
    ndarray
        Paths of shape (n_paths, n_steps + 1).
    """
    rng = np.random.default_rng(seed)
    
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    
    for i in range(n_steps):
        dW = rng.standard_normal(n_paths) * sqrt_dt
        paths[:, i + 1] = paths[:, i] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * dW
        )
    
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
    
    dS = μ·S·dt + σ·S^γ·dW
    
    Parameters
    ----------
    S0 : float
        Initial spot price.
    mu : float
        Drift.
    sigma : float
        Volatility scale.
    gamma : float
        CEV exponent (γ=1 is GBM, γ<1 gives leverage effect).
    T : float
        Time horizon.
    n_steps : int
        Number of steps.
    n_paths : int
        Number of paths.
    seed : int, optional
        Random seed.
    
    Returns
    -------
    ndarray
        Paths of shape (n_paths, n_steps + 1).
    """
    rng = np.random.default_rng(seed)
    
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    
    for i in range(n_steps):
        S = np.maximum(paths[:, i], 1e-6)
        dW = rng.standard_normal(n_paths) * sqrt_dt
        paths[:, i + 1] = S + mu * S * dt + sigma * (S ** gamma) * dW
        paths[:, i + 1] = np.maximum(paths[:, i + 1], 1e-6)
    
    return paths


def simulate_mean_reverting_vol_paths(
    S0: float,
    kappa: float,
    theta: float,
    sigma_vol: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate paths with mean-reverting stochastic volatility (simplified).
    
    dS = μ·S·dt + σ_t·S·dW
    dσ = κ(θ - σ)·dt + σ_v·dZ
    
    Parameters
    ----------
    S0 : float
        Initial spot.
    kappa : float
        Mean reversion speed.
    theta : float
        Long-term volatility.
    sigma_vol : float
        Vol of vol.
    T : float
        Time horizon.
    n_steps : int
        Number of steps.
    n_paths : int
        Number of paths.
    seed : int, optional
        Random seed.
    
    Returns
    -------
    ndarray
        Paths of shape (n_paths, n_steps + 1).
    """
    rng = np.random.default_rng(seed)
    
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    
    vol = np.full(n_paths, theta)
    
    for i in range(n_steps):
        dW_s = rng.standard_normal(n_paths) * sqrt_dt
        dW_v = rng.standard_normal(n_paths) * sqrt_dt
        
        # Update volatility
        vol = vol + kappa * (theta - vol) * dt + sigma_vol * dW_v
        vol = np.maximum(vol, 0.01)
        
        # Update spot
        S = paths[:, i]
        paths[:, i + 1] = S * np.exp(-0.5 * vol**2 * dt + vol * dW_s)
    
    return paths


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def compute_path_statistics(paths: np.ndarray) -> Dict[str, float]:
    """
    Compute summary statistics from paths.
    
    Parameters
    ----------
    paths : ndarray
        Paths of shape (n_paths, n_steps + 1).
    
    Returns
    -------
    dict
        Statistics dictionary.
    """
    final_prices = paths[:, -1]
    returns = np.log(paths[:, -1] / paths[:, 0])
    
    # Daily returns for higher moments
    daily_returns = np.diff(np.log(paths), axis=1).flatten()
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    # Skewness
    if std_return > 1e-8:
        skewness = np.mean(((returns - mean_return) / std_return) ** 3)
    else:
        skewness = 0.0
    
    # Kurtosis (excess)
    if std_return > 1e-8:
        kurtosis = np.mean(((returns - mean_return) / std_return) ** 4) - 3
    else:
        kurtosis = 0.0
    
    return {
        "mean_final": float(np.mean(final_prices)),
        "std_final": float(np.std(final_prices)),
        "mean_return": float(mean_return),
        "std_return": float(std_return),
        "skewness": float(skewness),
        "kurtosis": float(kurtosis),
        "min_price": float(np.min(final_prices)),
        "max_price": float(np.max(final_prices)),
    }


def compare_distributions(
    true_paths: np.ndarray,
    model_paths: np.ndarray,
) -> Dict[str, float]:
    """
    Compare terminal distributions.
    
    Parameters
    ----------
    true_paths : ndarray
        True paths.
    model_paths : ndarray
        Model-generated paths.
    
    Returns
    -------
    dict
        Comparison metrics.
    """
    true_final = true_paths[:, -1]
    model_final = model_paths[:, -1]
    
    # Mean and std errors
    mean_error = abs(np.mean(model_final) - np.mean(true_final)) / np.mean(true_final)
    std_error = abs(np.std(model_final) - np.std(true_final)) / np.std(true_final)
    
    # Kolmogorov-Smirnov style max difference
    true_sorted = np.sort(true_final)
    model_sorted = np.sort(model_final)
    
    # Resample to same size for comparison
    n = min(len(true_sorted), len(model_sorted))
    true_q = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(true_sorted)), true_sorted)
    model_q = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(model_sorted)), model_sorted)
    
    ks_stat = np.max(np.abs(true_q - model_q)) / np.mean(true_final)
    
    return {
        "mean_error_pct": float(mean_error * 100),
        "std_error_pct": float(std_error * 100),
        "ks_stat_pct": float(ks_stat * 100),
    }


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def run_neural_sde() -> Tuple[Dict[str, TrainingResult], Dict[str, Dict]]:
    """
    Run Neural SDE experiments.
    
    Returns
    -------
    Tuple
        Training results and comparison metrics.
    """
    logger.info("=" * 70)
    logger.info("SECTION 1: Data Generation")
    logger.info("=" * 70)
    
    # Generate training data from GBM
    S0 = 100.0
    mu_true = 0.05
    sigma_true = 0.2
    T = 1.0
    n_steps = 50
    n_train_paths = 2000
    n_test_paths = 500
    
    logger.info("")
    logger.info("Generating GBM training data...")
    logger.info(f"  S0={S0}, μ={mu_true}, σ={sigma_true}")
    logger.info(f"  T={T}y, steps={n_steps}")
    logger.info(f"  Train paths: {n_train_paths}")
    logger.info(f"  Test paths:  {n_test_paths}")
    
    gbm_train = simulate_gbm_paths(S0, mu_true, sigma_true, T, n_steps, n_train_paths, seed=42)
    gbm_test = simulate_gbm_paths(S0, mu_true, sigma_true, T, n_steps, n_test_paths, seed=123)
    
    # Statistics of true data
    train_stats = compute_path_statistics(gbm_train)
    logger.info("")
    logger.info(f"  Training data stats:")
    logger.info(f"    Mean final: {train_stats['mean_final']:.2f}")
    logger.info(f"    Std final:  {train_stats['std_final']:.2f}")
    logger.info(f"    Mean return: {train_stats['mean_return']:.4f}")
    
    # Create Neural SDE
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 2: Neural SDE Setup")
    logger.info("=" * 70)
    
    sde_config = NeuralSDEConfig(
        drift_hidden_dims=[64, 32],
        diffusion_hidden_dims=[64, 32],
        activation="tanh",
        min_vol=0.01,
        max_vol=1.0,
        solver_type="euler",
        S_mean=S0,
        S_std=20.0,
    )
    
    neural_sde = NeuralSDEDynamics(config=sde_config, seed=42)
    
    logger.info("")
    logger.info("  Neural SDE Configuration:")
    logger.info(f"    Drift network:     {sde_config.drift_hidden_dims}")
    logger.info(f"    Diffusion network: {sde_config.diffusion_hidden_dims}")
    logger.info(f"    Activation:        {sde_config.activation}")
    logger.info(f"    Vol range:         [{sde_config.min_vol}, {sde_config.max_vol}]")
    logger.info(f"    Solver:            {sde_config.solver_type}")
    
    # Training
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 3: Training")
    logger.info("=" * 70)
    
    training_config = TrainingConfig(
        n_epochs=60,
        learning_rate=0.001,
        batch_size=64,
        moment_weight=1.0,
        pathwise_weight=0.5,
        l2_reg=1e-5,
        n_sim_paths=500,
        n_sim_steps=n_steps,
        patience=15,
        verbose=True,
        log_interval=20,
    )
    
    logger.info("")
    logger.info(f"  Training Configuration:")
    logger.info(f"    Epochs:         {training_config.n_epochs}")
    logger.info(f"    Learning rate:  {training_config.learning_rate}")
    logger.info(f"    Batch size:     {training_config.batch_size}")
    logger.info(f"    Moment weight:  {training_config.moment_weight}")
    logger.info(f"    Pathwise weight: {training_config.pathwise_weight}")
    logger.info("")
    
    trainer = NeuralSDETrainer(config=training_config, seed=42)
    
    start_time = time.time()
    training_result = trainer.fit(
        model=neural_sde,
        historical_paths=gbm_train,
    )
    training_time = time.time() - start_time
    
    logger.info("")
    logger.info(f"  Training completed in {training_time:.1f}s")
    logger.info(f"  Final loss: {training_result.final_loss:.6f}")
    logger.info(f"  Converged:  {training_result.converged}")
    
    # Evaluation
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 4: Evaluation")
    logger.info("=" * 70)
    
    # Simulate from trained model
    logger.info("")
    logger.info("Simulating from trained Neural SDE...")
    
    model_paths = neural_sde.simulate(
        S0=S0,
        T=T,
        n_steps=n_steps,
        n_paths=n_test_paths,
    )
    
    # Compare statistics
    model_stats = compute_path_statistics(model_paths)
    true_stats = compute_path_statistics(gbm_test)
    comparison = compare_distributions(gbm_test, model_paths)
    
    logger.info("")
    logger.info(f"{'Statistic':<20} {'True':>12} {'Neural SDE':>12} {'Error':>10}")
    logger.info("-" * 60)
    logger.info(f"{'Mean Final':<20} {true_stats['mean_final']:>12.2f} {model_stats['mean_final']:>12.2f} {comparison['mean_error_pct']:>9.1f}%")
    logger.info(f"{'Std Final':<20} {true_stats['std_final']:>12.2f} {model_stats['std_final']:>12.2f} {comparison['std_error_pct']:>9.1f}%")
    logger.info(f"{'Mean Return':<20} {true_stats['mean_return']:>12.4f} {model_stats['mean_return']:>12.4f}")
    logger.info(f"{'Std Return':<20} {true_stats['std_return']:>12.4f} {model_stats['std_return']:>12.4f}")
    logger.info(f"{'Skewness':<20} {true_stats['skewness']:>12.4f} {model_stats['skewness']:>12.4f}")
    logger.info(f"{'Kurtosis':<20} {true_stats['kurtosis']:>12.4f} {model_stats['kurtosis']:>12.4f}")
    logger.info("-" * 60)
    logger.info(f"  KS-like statistic: {comparison['ks_stat_pct']:.2f}%")
    
    results = {
        "training_result": training_result,
        "comparison": comparison,
        "true_stats": true_stats,
        "model_stats": model_stats,
        "true_paths": gbm_test,
        "model_paths": model_paths,
    }
    
    return {"gbm": training_result}, results


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_results(results: Dict) -> None:
    """Visualize Neural SDE results."""
    if not MATPLOTLIB_AVAILABLE or not ENABLE_PLOTTING:
        logger.info("Skipping plots")
        return
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 5: Visualization")
    logger.info("=" * 70)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    true_paths = results["true_paths"]
    model_paths = results["model_paths"]
    training_result = results["training_result"]
    
    # Plot 1: Sample true paths
    ax = axes[0, 0]
    for i in range(min(30, true_paths.shape[0])):
        ax.plot(true_paths[i], alpha=0.4, color='#2E86AB', linewidth=0.8)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Price')
    ax.set_title('True GBM Paths')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Sample model paths
    ax = axes[0, 1]
    for i in range(min(30, model_paths.shape[0])):
        ax.plot(model_paths[i], alpha=0.4, color='#E94F37', linewidth=0.8)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Price')
    ax.set_title('Neural SDE Paths')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Terminal distribution comparison
    ax = axes[1, 0]
    ax.hist(true_paths[:, -1], bins=40, alpha=0.5, density=True, 
            label='True', color='#2E86AB')
    ax.hist(model_paths[:, -1], bins=40, alpha=0.5, density=True,
            label='Neural SDE', color='#E94F37')
    ax.set_xlabel('Terminal Price')
    ax.set_ylabel('Density')
    ax.set_title('Terminal Distribution Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Training loss
    ax = axes[1, 1]
    if training_result.loss_history:
        ax.plot(training_result.loss_history, color='#2E86AB', linewidth=2, label='Total Loss')
        if training_result.moment_losses:
            ax.plot(training_result.moment_losses, color='#4CAF50', 
                    linewidth=1.5, alpha=0.7, label='Moment Loss')
        if training_result.pathwise_losses:
            ax.plot(training_result.pathwise_losses, color='#E94F37',
                    linewidth=1.5, alpha=0.7, label='Pathwise Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training History')
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
    │  1. QuantStrata Neural SDE Module:                                  │
    │     - NeuralSDEDynamics: Main class for learnable dynamics          │
    │     - NeuralDriftNetwork: Neural network for μ_θ(S, t)              │
    │     - NeuralDiffusionNetwork: Neural network for σ_φ(S, t)          │
    │     - EulerMaruyama/Milstein solvers for simulation                 │
    │                                                                      │
    │  2. Network Design:                                                 │
    │     - Drift: Any real output (linear activation)                    │
    │     - Diffusion: Positive output (softplus + clipping)              │
    │     - Input normalization for training stability                    │
    │                                                                      │
    │  3. Training Pipeline:                                              │
    │     - NeuralSDETrainer with TrainingConfig                          │
    │     - Moment matching loss (mean, variance)                         │
    │     - Pathwise loss (trajectory similarity)                         │
    │     - Early stopping and L2 regularization                          │
    │                                                                      │
    │  4. Production Considerations:                                      │
    │     - Validate against known parametric models                      │
    │     - Compare terminal distributions                                │
    │     - Check moment matching (mean, std, skew, kurtosis)             │
    │     - Use for pricing, risk simulation, stress testing              │
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
        training_results, results = run_neural_sde()
        visualize_results(results)
        print_summary()
        logger.info("Example completed successfully!")
        
    except Exception as e:
        logger.exception(f"Example failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural SDE Example")
    parser.add_argument("--plot", action="store_true", default=True)
    parser.add_argument("--no-plot", action="store_false", dest="plot")
    
    args = parser.parse_args()
    main(args)

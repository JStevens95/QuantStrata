#!/usr/bin/env python3
"""
===============================================================================
Machine Learning: Neural Network Option Pricer with QuantStrata MLPPricer
===============================================================================

This example demonstrates training a neural network to price options using
QuantStrata's production machine_learning module.

Learning Objectives
-------------------
1. **MLPPricer Architecture**: Use the library's production MLP model
2. **Data Generation**: Generate training data from analytical BSM pricer
3. **Training Pipeline**: Use Trainer with TrainingConfig
4. **Model Evaluation**: Compare speed and accuracy vs analytical pricing

Mathematical Framework
----------------------
The goal is to learn a function f_θ such that:
    f_θ(S, K, T, σ, r) ≈ V_BSM(S, K, T, σ, r)

Loss function:
    L(θ) = E[(f_θ(x) - V_true(x))²]

Input features (normalized for training stability):
    - Moneyness: log(S/K)
    - Time to expiry: T
    - Volatility: σ
    - Interest rate: r
    - Option type indicator

Production Context
------------------
At a hedge fund:
- Neural pricers enable real-time pricing of large portfolios
- 10-1000x speedup vs MC for exotic options
- Used in risk calculation, scenario analysis, and optimization
- Requires careful validation against benchmark pricers

Why Results May Vary / Production Considerations
-------------------------------------------------
- This example uses constant (flat) parameters for data generation; production
  would use library Market with GridVolSurface and ZeroRateCurve from real
  market data or calibrated models (no FlatVolSurface/FlatZeroRateCurve in
  production pricing).
- Accuracy depends on training set size, architecture, and early stopping;
  increase n_train or tune hidden_units for better fit.
- For exotics or smile-sensitive products, train on full vol surface inputs
  and use production vol/curve types throughout.

Production checklist (hedge fund)
---------------------------------
- Use --seed for reproducibility; use --output-dir to save config and test metrics for audit.
- In production use GridVolSurface and ZeroRateCurve from market data (no flat vol/curves).
- Validate on held-out test set; document MAE/R² and inference latency.

Prerequisites
-------------
- TensorFlow 2.x installed
- Basic pricing examples (examples/pricing/)

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
from typing import Dict, List, Optional, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# -----------------------------------------------------------------------------
# QuantStrata imports
# -----------------------------------------------------------------------------
# BSM model for generating training data
from src.models.analytic.black_scholes_merton.base import (
    vanilla_price,
    vanilla_delta,
    vanilla_gamma,
    vanilla_vega,
    vanilla_theta,
)

# Check if TensorFlow is available
try:
    import tensorflow as tf
    TF_AVAILABLE = True
    
    # ML infrastructure
    from src.machine_learning.models.pricing.model import (
        MLPPricer,
        create_mlp_pricer,
    )
    from src.machine_learning.training.trainer import (
        Trainer,
        TrainingResult,
    )
    from src.machine_learning.core.config import (
        TrainingConfig,
        OptimizerConfig,
        EarlyStoppingConfig,
    )
except ImportError as e:
    TF_AVAILABLE = False
    tf = None
    logger_init = logging.getLogger(__name__)
    logger_init.warning(f"TensorFlow not available: {e}")


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
# DATA GENERATION USING QUANTSTRATA BSM
# =============================================================================

@dataclass
class PricingDataConfig:
    """Configuration for pricing data generation."""
    n_train: int = 50000
    n_val: int = 10000
    n_test: int = 10000
    
    # Parameter ranges
    spot_range: Tuple[float, float] = (50.0, 150.0)
    strike_range: Tuple[float, float] = (50.0, 150.0)
    expiry_range: Tuple[float, float] = (0.05, 2.0)
    vol_range: Tuple[float, float] = (0.05, 0.6)
    rate_range: Tuple[float, float] = (0.0, 0.1)
    
    seed: int = 42


def generate_pricing_data(config: PricingDataConfig, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate option pricing data using QuantStrata's BSM module.
    
    Parameters
    ----------
    config : PricingDataConfig
        Data generation configuration.
    n_samples : int
        Number of samples to generate.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Features (X) and targets (y).
    """
    rng = np.random.default_rng(config.seed)
    
    # Sample parameters uniformly
    spots = rng.uniform(*config.spot_range, n_samples)
    strikes = rng.uniform(*config.strike_range, n_samples)
    expiries = rng.uniform(*config.expiry_range, n_samples)
    vols = rng.uniform(*config.vol_range, n_samples)
    rates = rng.uniform(*config.rate_range, n_samples)
    is_call = rng.integers(0, 2, n_samples)  # 0 = put, 1 = call
    
    # Compute prices using QuantStrata BSM
    prices = np.zeros(n_samples)
    
    for i in range(n_samples):
        option_type = "call" if is_call[i] == 1 else "put"
        
        # Use library BSM with cost-of-carry = rate (no dividends)
        prices[i] = vanilla_price(
            option_type=option_type,
            spot=float(spots[i]),
            strike=float(strikes[i]),
            expiry=float(expiries[i]),
            discount_rate=float(rates[i]),
            carry=float(rates[i]),  # b = r for non-dividend equity
            vol=float(vols[i]),
        )
    
    # Build feature matrix (normalized)
    # Features: [moneyness, expiry, vol, rate, is_call]
    moneyness = np.log(spots / strikes)
    
    X = np.column_stack([
        moneyness,
        expiries,
        vols,
        rates,
        is_call.astype(float),
    ])
    
    # Normalize price by spot for better training
    y = prices / spots
    
    return X.astype(np.float32), y.astype(np.float32).reshape(-1, 1)


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_pricer(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    spots_test: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate neural pricer performance.
    
    Returns
    -------
    dict
        Evaluation metrics.
    """
    # Get predictions
    y_pred = model.predict(X_test, verbose=0)
    
    # Denormalize (multiply by spot)
    prices_pred = y_pred.flatten() * spots_test
    prices_true = y_test.flatten() * spots_test
    
    # Compute metrics
    errors = prices_pred - prices_true
    abs_errors = np.abs(errors)
    rel_errors = np.abs(errors) / (prices_true + 1e-8)
    
    return {
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(np.mean(abs_errors)),
        "mape": float(np.mean(rel_errors) * 100),
        "max_error": float(np.max(abs_errors)),
        "r2": float(1 - np.var(errors) / np.var(prices_true)),
    }


def speed_comparison(
    model,
    n_samples: int = 10000,
    config: PricingDataConfig = None,
) -> Dict[str, float]:
    """
    Compare neural pricer speed vs analytical BSM.
    
    Returns
    -------
    dict
        Timing results.
    """
    config = config or PricingDataConfig(seed=999)
    rng = np.random.default_rng(config.seed)
    
    # Generate test data
    spots = rng.uniform(*config.spot_range, n_samples)
    strikes = rng.uniform(*config.strike_range, n_samples)
    expiries = rng.uniform(*config.expiry_range, n_samples)
    vols = rng.uniform(*config.vol_range, n_samples)
    rates = rng.uniform(*config.rate_range, n_samples)
    
    # Neural pricer timing
    X_test = np.column_stack([
        np.log(spots / strikes),
        expiries,
        vols,
        rates,
        np.ones(n_samples),  # All calls
    ]).astype(np.float32)
    
    # Warm-up
    _ = model.predict(X_test[:100], verbose=0)
    
    start = time.time()
    _ = model.predict(X_test, verbose=0)
    neural_time = time.time() - start
    
    # BSM timing
    start = time.time()
    for i in range(n_samples):
        _ = vanilla_price(
            option_type="call",
            spot=float(spots[i]),
            strike=float(strikes[i]),
            expiry=float(expiries[i]),
            discount_rate=float(rates[i]),
            carry=float(rates[i]),
            vol=float(vols[i]),
        )
    bsm_time = time.time() - start
    
    return {
        "neural_time_ms": neural_time * 1000,
        "bsm_time_ms": bsm_time * 1000,
        "speedup": bsm_time / neural_time,
        "neural_per_option_us": neural_time / n_samples * 1e6,
        "bsm_per_option_us": bsm_time / n_samples * 1e6,
    }


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def run_neural_pricer(seed: int = 42) -> Tuple[TrainingResult, Dict[str, float], Dict[str, float], "tf.keras.Model", np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the complete neural pricer workflow.
    
    Parameters
    ----------
    seed : int
        Random seed for reproducibility (training and test data).
    
    Returns
    -------
    Tuple
        Training result, evaluation metrics, speed comparison, model, X_test, y_test, spots_test.
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is required for this example. Please install: pip install tensorflow")
    
    logger.info("=" * 70)
    logger.info("SECTION 1: Data Generation (using QuantStrata BSM)")
    logger.info("=" * 70)
    
    config = PricingDataConfig(
        n_train=50000,
        n_val=10000,
        n_test=10000,
        seed=42,
    )
    
    logger.info("")
    logger.info("Generating training data from analytical BSM prices...")
    
    X_train, y_train = generate_pricing_data(config, config.n_train)
    config.seed = 43  # Different seed for validation
    X_val, y_val = generate_pricing_data(config, config.n_val)
    config.seed = 44  # Different seed for test
    X_test, y_test = generate_pricing_data(config, config.n_test)
    
    logger.info(f"  Train samples: {len(X_train):,}")
    logger.info(f"  Val samples:   {len(X_val):,}")
    logger.info(f"  Test samples:  {len(X_test):,}")
    logger.info(f"  Features:      {X_train.shape[1]} (moneyness, T, σ, r, is_call)")
    
    # Create model using QuantStrata MLPPricer
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 2: Model Creation (QuantStrata MLPPricer)")
    logger.info("=" * 70)
    
    model = create_mlp_pricer(
        n_features=5,
        hidden_units=[128, 64, 32],
        activation="relu",
        dropout_rate=0.1,
        use_batch_norm=True,
        output_activation="softplus",  # Ensure positive prices
    )
    
    logger.info("")
    logger.info(f"  Model: {model.name}")
    logger.info(f"  Architecture: 5 -> 128 -> 64 -> 32 -> 1")
    logger.info(f"  Activation: ReLU (softplus output)")
    logger.info(f"  Batch normalization: Yes")
    logger.info(f"  Dropout: 10%")
    
    # Training using QuantStrata Trainer
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 3: Training (QuantStrata Trainer)")
    logger.info("=" * 70)
    
    training_config = TrainingConfig(
        epochs=100,
        batch_size=256,
        optimizer=OptimizerConfig(name="adam", learning_rate=0.001),
        loss="mae",
        metrics=["mse", "mae"],
        early_stopping=EarlyStoppingConfig(patience=10, min_delta=1e-6),
        seed=seed,
        verbose=1,
    )
    
    trainer = Trainer(model, training_config)
    trainer.compile()
    
    logger.info("")
    logger.info(f"  Epochs: {training_config.epochs}")
    logger.info(f"  Batch size: {training_config.batch_size}")
    logger.info(f"  Optimizer: Adam (lr={training_config.optimizer.learning_rate})")
    logger.info(f"  Early stopping: patience={training_config.early_stopping.patience}")
    logger.info("")
    
    training_result = trainer.fit(
        train_data=(X_train, y_train),
        val_data=(X_val, y_val),
    )
    
    logger.info("")
    logger.info(f"  Training completed in {training_result.total_time_seconds:.1f}s")
    logger.info(f"  Best epoch: {training_result.best_epoch}")
    logger.info(f"  Best val loss: {training_result.best_val_loss:.6f}")
    logger.info(f"  Early stopped: {training_result.stopped_early}")
    
    # Evaluation
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 4: Evaluation")
    logger.info("=" * 70)
    
    # Generate spots for denormalization
    rng = np.random.default_rng(seed + 1)
    spots_test = rng.uniform(50.0, 150.0, config.n_test)
    
    eval_metrics = evaluate_pricer(model, X_test, y_test, spots_test)
    
    logger.info("")
    logger.info("  Test Set Metrics:")
    logger.info(f"    RMSE:       ${eval_metrics['rmse']:.4f}")
    logger.info(f"    MAE:        ${eval_metrics['mae']:.4f}")
    logger.info(f"    MAPE:       {eval_metrics['mape']:.2f}%")
    logger.info(f"    Max Error:  ${eval_metrics['max_error']:.4f}")
    logger.info(f"    R²:         {eval_metrics['r2']:.6f}")
    
    # Speed comparison
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 5: Speed Comparison")
    logger.info("=" * 70)
    
    speed_metrics = speed_comparison(model, n_samples=10000)
    
    logger.info("")
    logger.info(f"  Pricing 10,000 options:")
    logger.info(f"    Neural Pricer: {speed_metrics['neural_time_ms']:.2f} ms ({speed_metrics['neural_per_option_us']:.2f} μs/option)")
    logger.info(f"    BSM Analytical: {speed_metrics['bsm_time_ms']:.2f} ms ({speed_metrics['bsm_per_option_us']:.2f} μs/option)")
    logger.info(f"    Speedup:       {speed_metrics['speedup']:.1f}x")
    
    return training_result, eval_metrics, speed_metrics, model, X_test, y_test, spots_test


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_results(
    training_result: TrainingResult,
    eval_metrics: Dict[str, float],
    speed_metrics: Dict[str, float],
    model=None,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    spots_test: Optional[np.ndarray] = None,
) -> None:
    """Visualize training and evaluation results in a single figure (all panels visible)."""
    if not MATPLOTLIB_AVAILABLE or not ENABLE_PLOTTING:
        logger.info("Skipping plots")
        return
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 6: Visualization")
    logger.info("=" * 70)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Single figure with 2x2 grid so all plots are visible at once
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Neural Pricer: Training and Evaluation', fontsize=14, fontweight='bold', y=1.02)
    
    # 1. Training history
    ax = axes[0, 0]
    epochs = range(1, len(training_result.history['loss']) + 1)
    ax.plot(epochs, training_result.history['loss'], label='Train Loss', color='#2E86AB', linewidth=2)
    if 'val_loss' in training_result.history:
        ax.plot(epochs, training_result.history['val_loss'], label='Val Loss', color='#E94F37', linewidth=2)
    ax.axvline(training_result.best_epoch, color='green', linestyle='--', alpha=0.8, label=f'Best ({training_result.best_epoch})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training History')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # 2. Prediction vs actual (if test data and model provided)
    ax = axes[0, 1]
    if model is not None and X_test is not None and y_test is not None and spots_test is not None:
        n_show = min(2000, len(X_test))  # Subsample for clear scatter
        idx = np.random.RandomState(42).choice(len(X_test), n_show, replace=False)
        y_pred = model.predict(X_test[idx], verbose=0).flatten()
        prices_pred = y_pred * spots_test[idx]
        prices_true = y_test[idx].flatten() * spots_test[idx]
        ax.scatter(prices_true, prices_pred, alpha=0.4, s=8, c='#2E86AB', edgecolors='none')
        lims = [min(prices_true.min(), prices_pred.min()), max(prices_true.max(), prices_pred.max())]
        ax.plot(lims, lims, 'k--', linewidth=2, label='Perfect fit')
        ax.set_xlabel('BSM Price')
        ax.set_ylabel('Neural Pricer Price')
        ax.set_title('Prediction vs Actual')
        ax.legend()
        ax.set_aspect('equal', adjustable='box')
    else:
        ax.text(0.5, 0.5, 'Prediction vs Actual\n(no test data)', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Prediction vs Actual')
    ax.grid(True, alpha=0.3)
    
    # 3. Error distribution (if test data and model provided)
    ax = axes[1, 0]
    if model is not None and X_test is not None and y_test is not None and spots_test is not None:
        y_pred = model.predict(X_test, verbose=0).flatten()
        errors = (y_pred * spots_test) - (y_test.flatten() * spots_test)
        ax.hist(errors, bins=50, color='#2E86AB', alpha=0.7, edgecolor='white', density=True)
        ax.axvline(0, color='black', linestyle='--', linewidth=2)
        ax.axvline(np.mean(errors), color='#E94F37', linestyle='-', linewidth=2, label=f'Mean: ${np.mean(errors):.4f}')
        ax.set_xlabel('Pricing Error ($)')
        ax.set_ylabel('Density')
        ax.set_title('Error Distribution')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'Error Distribution\n(no test data)', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Error Distribution')
    ax.grid(True, alpha=0.3)
    
    # 4. Speed comparison
    ax = axes[1, 1]
    methods = ['Neural Pricer', 'BSM Analytical']
    times = [speed_metrics['neural_time_ms'], speed_metrics['bsm_time_ms']]
    colors = ['#2E86AB', '#E94F37']
    bars = ax.bar(methods, times, color=colors, alpha=0.8, edgecolor='white')
    ax.set_ylabel('Time (ms) for 10k options')
    ax.set_title(f'Speed Comparison ({speed_metrics["speedup"]:.1f}x speedup)')
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f'{t:.1f}ms', ha='center', va='bottom', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
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
    │  1. QuantStrata ML Infrastructure:                                  │
    │     - MLPPricer: Production neural network for option pricing       │
    │     - Trainer: High-level training interface with best practices    │
    │     - TrainingConfig: Comprehensive configuration management        │
    │                                                                      │
    │  2. Data Generation:                                                │
    │     - Use analytical BSM (vanilla_price) as ground truth            │
    │     - Normalize features (moneyness, not raw spot/strike)           │
    │     - Normalize targets (price/spot ratio)                          │
    │                                                                      │
    │  3. Model Architecture:                                             │
    │     - Hidden layers: [128, 64, 32] works well for vanilla options   │
    │     - Batch normalization for training stability                    │
    │     - Softplus output ensures positive prices                       │
    │                                                                      │
    │  4. Production Deployment:                                          │
    │     - Significant speedup for batch pricing                         │
    │     - Validate RMSE/MAPE against business tolerance                 │
    │     - Monitor for distribution shift in production                  │
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

    if not TF_AVAILABLE:
        logger.warning("TensorFlow is not installed. Skipping neural pricer example (no error). Install with: pip install tensorflow")
        logger.info("Example skipped successfully (exit 0).")
        return

    seed = getattr(args, "seed", 42)
    output_dir = getattr(args, "output_dir", None)
    if output_dir is not None:
        from pathlib import Path
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)
    if TF_AVAILABLE:
        tf.random.set_seed(seed)

    try:
        training_result, eval_metrics, speed_metrics, model, X_test, y_test, spots_test = run_neural_pricer(seed=seed)
        visualize_results(
            training_result, eval_metrics, speed_metrics,
            model=model, X_test=X_test, y_test=y_test, spots_test=spots_test,
        )
        if output_dir is not None:
            import json
            with open(output_dir / "run_config.json", "w") as f:
                json.dump({"script": "01_neural_pricer", "seed": seed}, f, indent=2)
            with open(output_dir / "eval_metrics.json", "w") as f:
                json.dump({k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in eval_metrics.items()}, f, indent=2)
            with open(output_dir / "speed_metrics.json", "w") as f:
                json.dump({k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in speed_metrics.items()}, f, indent=2)
            logger.info("Saved artifacts to %s", output_dir)
        print_summary()
        logger.info("Example completed successfully!")

    except Exception as e:
        logger.exception(f"Example failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Pricer Example")
    parser.add_argument("--plot", action="store_true", default=True)
    parser.add_argument("--no-plot", action="store_false", dest="plot")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=str, default=None, metavar="DIR", help="Save run_config and metrics to DIR")
    
    args = parser.parse_args()
    main(args)

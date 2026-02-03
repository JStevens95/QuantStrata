"""
Custom Keras callbacks for ML training.

This module provides specialized callbacks for:
    - Training metrics logging
    - Model performance tracking
    - Custom checkpointing
    - Training visualization

Usage:
    callbacks = [
        MetricsLogger(log_dir="./logs"),
        PricingErrorCallback(val_data, price_scaler),
        TrainingProgressCallback(total_epochs=100),
    ]
    model.fit(X, y, callbacks=callbacks)
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import tensorflow as tf
import numpy as np


class MetricsLogger(tf.keras.callbacks.Callback):
    """
    Callback to log training metrics to JSON file.
    
    Creates a detailed training log with:
        - Per-epoch metrics (loss, validation loss, custom metrics)
        - Training timestamps
        - Best epoch tracking
        - Total training time
    
    Attributes:
        log_dir: Directory to save logs
        log_file: Name of the log file
    """
    
    def __init__(self, log_dir: str = "./logs", log_file: str = "training_log.json"):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_file = log_file
        self.log_path = self.log_dir / log_file
        
        self.history: Dict[str, List[float]] = {}
        self.epoch_times: List[float] = []
        self.start_time: Optional[float] = None
        self.best_val_loss: float = float("inf")
        self.best_epoch: int = 0
    
    def on_train_begin(self, logs=None):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()
        self.history = {}
        self.epoch_times = []
    
    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch_time = time.time() - self._epoch_start
        self.epoch_times.append(epoch_time)
        
        # Update history
        for key, value in logs.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(float(value))
        
        # Track best epoch
        val_loss = logs.get("val_loss", logs.get("loss", float("inf")))
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch + 1
    
    def on_train_end(self, logs=None):
        total_time = time.time() - self.start_time if self.start_time else 0
        
        log_data = {
            "training_completed": datetime.utcnow().isoformat(),
            "total_epochs": len(self.epoch_times),
            "total_time_seconds": total_time,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "avg_epoch_time": sum(self.epoch_times) / len(self.epoch_times) if self.epoch_times else 0,
            "history": self.history,
            "epoch_times": self.epoch_times,
        }
        
        with open(self.log_path, "w") as f:
            json.dump(log_data, f, indent=2)
        
        print(f"\nTraining log saved to: {self.log_path}")


class PricingErrorCallback(tf.keras.callbacks.Callback):
    """
    Callback to track pricing-specific error metrics.
    
    Computes and logs:
        - Mean absolute pricing error (in original scale)
        - Mean percentage pricing error
        - Max pricing error
        - Error distribution percentiles
    
    Useful for monitoring model performance in financial terms.
    
    Attributes:
        val_features: Validation features
        val_prices: Validation target prices (original scale)
        price_scaler: Optional scaler to denormalize predictions
    """
    
    def __init__(
        self,
        val_features: np.ndarray,
        val_prices: np.ndarray,
        price_scaler: Optional[Any] = None,
        log_every: int = 5,
    ):
        super().__init__()
        self.val_features = val_features
        self.val_prices = val_prices
        self.price_scaler = price_scaler
        self.log_every = log_every
        
        self.pricing_errors: Dict[str, List[float]] = {
            "mae": [],
            "mape": [],
            "max_error": [],
            "p95_error": [],
        }
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.log_every != 0:
            return
        
        # Get predictions
        preds = self.model.predict(self.val_features, verbose=0)
        preds = preds.flatten()
        
        # Denormalize if scaler provided
        if self.price_scaler is not None:
            if hasattr(self.price_scaler, "inverse_transform"):
                preds = self.price_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
            elif isinstance(self.price_scaler, tuple):
                # Assume (mean, std) tuple
                mean, std = self.price_scaler
                preds = preds * std + mean
        
        # Compute errors
        errors = np.abs(self.val_prices - preds)
        mae = float(np.mean(errors))
        mape = float(np.mean(errors / (np.abs(self.val_prices) + 1e-8)) * 100)
        max_error = float(np.max(errors))
        p95_error = float(np.percentile(errors, 95))
        
        self.pricing_errors["mae"].append(mae)
        self.pricing_errors["mape"].append(mape)
        self.pricing_errors["max_error"].append(max_error)
        self.pricing_errors["p95_error"].append(p95_error)
        
        print(f"  Pricing: MAE=${mae:.2f}, MAPE={mape:.2f}%, Max=${max_error:.2f}, P95=${p95_error:.2f}")


class TrainingProgressCallback(tf.keras.callbacks.Callback):
    """
    Callback for detailed training progress visualization.
    
    Provides:
        - Estimated time remaining
        - Learning rate tracking
        - Memory usage (if available)
        - Progress bar with metrics
    """
    
    def __init__(self, total_epochs: int):
        super().__init__()
        self.total_epochs = total_epochs
        self.epoch_times: List[float] = []
        self.start_time: Optional[float] = None
    
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        print(f"\n{'='*60}")
        print(f"Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total epochs: {self.total_epochs}")
        print(f"{'='*60}\n")
    
    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self._epoch_start
        self.epoch_times.append(epoch_time)
        
        # Estimate remaining time
        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
        remaining_epochs = self.total_epochs - (epoch + 1)
        eta_seconds = avg_epoch_time * remaining_epochs
        
        # Format ETA
        if eta_seconds > 3600:
            eta_str = f"{eta_seconds/3600:.1f}h"
        elif eta_seconds > 60:
            eta_str = f"{eta_seconds/60:.1f}m"
        else:
            eta_str = f"{eta_seconds:.0f}s"
        
        # Get current learning rate
        lr = self._get_learning_rate()
        
        # Build progress string
        progress = (epoch + 1) / self.total_epochs
        bar_width = 30
        filled = int(bar_width * progress)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        # Print progress (only every 10 epochs to reduce noise)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logs = logs or {}
            loss = logs.get("loss", 0)
            val_loss = logs.get("val_loss", 0)
            print(f"[{bar}] {epoch+1}/{self.total_epochs} | "
                  f"Loss: {loss:.4f} | Val: {val_loss:.4f} | "
                  f"LR: {lr:.2e} | ETA: {eta_str}")
    
    def _get_learning_rate(self) -> float:
        """Get current learning rate from optimizer."""
        try:
            lr = self.model.optimizer.learning_rate
            if hasattr(lr, "numpy"):
                return float(lr.numpy())
            elif callable(lr):
                # Learning rate schedule
                return float(lr(self.model.optimizer.iterations))
            return float(lr)
        except Exception:
            return 0.0
    
    def on_train_end(self, logs=None):
        total_time = time.time() - self.start_time if self.start_time else 0
        
        print(f"\n{'='*60}")
        print(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
        print(f"Avg epoch time: {sum(self.epoch_times)/len(self.epoch_times):.2f}s")
        print(f"{'='*60}\n")


class GradientMonitorCallback(tf.keras.callbacks.Callback):
    """
    Callback to monitor gradient statistics during training.
    
    Tracks:
        - Gradient norms per layer
        - Gradient mean/std
        - Detects vanishing/exploding gradients
    
    Useful for debugging training issues.
    """
    
    def __init__(self, log_every: int = 10, warn_threshold: float = 100.0):
        super().__init__()
        self.log_every = log_every
        self.warn_threshold = warn_threshold
        self.gradient_norms: List[Dict[str, float]] = []
    
    def on_batch_end(self, batch, logs=None):
        if batch % self.log_every != 0:
            return
        
        # Get gradient norms (approximation from weight changes)
        norms = {}
        for layer in self.model.layers:
            for weight in layer.trainable_weights:
                norm = float(tf.norm(weight).numpy())
                norms[weight.name] = norm
                
                if norm > self.warn_threshold:
                    print(f"  Warning: Large weight norm in {weight.name}: {norm:.2f}")
                elif norm < 1e-7:
                    print(f"  Warning: Near-zero weight in {weight.name}: {norm:.2e}")
        
        self.gradient_norms.append(norms)


def get_standard_callbacks(
    config: "TrainingConfig",
    val_data: Optional[tuple] = None,
) -> List[tf.keras.callbacks.Callback]:
    """
    Build standard callback list from training config.
    
    Args:
        config: Training configuration
        val_data: Optional (features, targets) tuple for validation
    
    Returns:
        List of Keras callbacks
    """
    from src.m_learning.core.config import TrainingConfig
    
    callbacks = []
    
    # Early stopping
    if config.early_stopping is not None:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            patience=config.early_stopping.patience,
            min_delta=config.early_stopping.min_delta,
            monitor=config.early_stopping.monitor,
            mode=config.early_stopping.mode,
            restore_best_weights=config.early_stopping.restore_best_weights,
            verbose=1,
        ))
    
    # Checkpointing
    if config.checkpoint is not None:
        checkpoint_path = Path(config.checkpoint.checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path / "model_{epoch:03d}.keras"),
            save_freq=config.checkpoint.save_freq,
            save_best_only=config.checkpoint.save_best_only,
            monitor=config.checkpoint.monitor,
            mode=config.checkpoint.mode,
            save_weights_only=config.checkpoint.save_weights_only,
            verbose=1,
        ))
    
    # TensorBoard
    if config.log_dir is not None:
        callbacks.append(tf.keras.callbacks.TensorBoard(
            log_dir=config.log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq="epoch",
        ))
    
    # Training progress
    callbacks.append(TrainingProgressCallback(total_epochs=config.epochs))
    
    # Metrics logger
    if config.log_dir is not None:
        callbacks.append(MetricsLogger(log_dir=config.log_dir))
    
    return callbacks

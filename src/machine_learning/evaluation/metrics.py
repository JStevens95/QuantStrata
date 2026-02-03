"""
Domain-specific TensorFlow metrics.

This module provides custom Keras metrics for:
    - Option pricing evaluation
    - Model calibration evaluation
    - Portfolio P&L prediction

Usage:
    model.compile(
        loss='mse',
        metrics=[PricingMetrics.mape(), PricingMetrics.max_error()]
    )
"""
from __future__ import annotations

from typing import List, Optional

import tensorflow as tf
import numpy as np


class PricingMetrics:
    """
    Collection of pricing-specific metrics.
    
    These metrics are designed for evaluating option pricing models
    where financial accuracy is paramount.
    """
    
    @staticmethod
    def mape() -> tf.keras.metrics.Metric:
        """Mean Absolute Percentage Error metric."""
        return MeanAbsolutePercentageError(name="mape")
    
    @staticmethod
    def max_absolute_error() -> tf.keras.metrics.Metric:
        """Maximum Absolute Error metric."""
        return MaxAbsoluteError(name="max_abs_error")
    
    @staticmethod
    def relative_error() -> tf.keras.metrics.Metric:
        """Mean Relative Error metric."""
        return MeanRelativeError(name="rel_error")
    
    @staticmethod
    def r2_score() -> tf.keras.metrics.Metric:
        """R² (coefficient of determination) metric."""
        return R2Score(name="r2")
    
    @staticmethod
    def pricing_accuracy(threshold: float = 0.01) -> tf.keras.metrics.Metric:
        """
        Fraction of predictions within threshold of actual.
        
        Args:
            threshold: Acceptable relative error (default 1%)
        """
        return PricingAccuracy(threshold=threshold, name=f"acc_{int(threshold*100)}pct")


class CalibrationMetrics:
    """
    Collection of calibration-specific metrics.
    
    These metrics are designed for evaluating model calibration networks
    that predict model parameters from market data.
    """
    
    @staticmethod
    def parameter_mae() -> tf.keras.metrics.Metric:
        """Mean Absolute Error for parameters."""
        return tf.keras.metrics.MeanAbsoluteError(name="param_mae")
    
    @staticmethod
    def parameter_relative_error() -> tf.keras.metrics.Metric:
        """Mean Relative Error for parameters."""
        return MeanRelativeError(name="param_rel_error")
    
    @staticmethod
    def parameter_accuracy(threshold: float = 0.05) -> tf.keras.metrics.Metric:
        """
        Fraction of parameters within threshold of true value.
        
        Args:
            threshold: Acceptable relative error (default 5%)
        """
        return PricingAccuracy(threshold=threshold, name=f"param_acc_{int(threshold*100)}pct")


class MeanAbsolutePercentageError(tf.keras.metrics.Metric):
    """
    Mean Absolute Percentage Error (MAPE).
    
    MAPE = mean(|y_true - y_pred| / |y_true|) * 100
    
    Handles zero values by adding small epsilon.
    """
    
    def __init__(self, name: str = "mape", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        
        # Avoid division by zero
        abs_true = tf.abs(y_true) + 1e-8
        ape = tf.abs(y_true - y_pred) / abs_true * 100
        
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            ape = ape * sample_weight
        
        self.total.assign_add(tf.reduce_sum(ape))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))
    
    def result(self):
        return self.total / (self.count + 1e-8)
    
    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


class MaxAbsoluteError(tf.keras.metrics.Metric):
    """
    Maximum Absolute Error.
    
    Tracks the worst prediction error in the batch.
    """
    
    def __init__(self, name: str = "max_abs_error", **kwargs):
        super().__init__(name=name, **kwargs)
        self.max_error = self.add_weight(name="max_error", initializer="zeros")
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        
        errors = tf.abs(y_true - y_pred)
        batch_max = tf.reduce_max(errors)
        
        self.max_error.assign(tf.maximum(self.max_error, batch_max))
    
    def result(self):
        return self.max_error
    
    def reset_state(self):
        self.max_error.assign(0.0)


class MeanRelativeError(tf.keras.metrics.Metric):
    """
    Mean Relative Error.
    
    MRE = mean((y_true - y_pred) / y_true)
    
    Unlike MAPE, this preserves sign (bias detection).
    """
    
    def __init__(self, name: str = "rel_error", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        
        # Avoid division by zero
        abs_true = tf.abs(y_true) + 1e-8
        rel_error = (y_true - y_pred) / abs_true
        
        self.total.assign_add(tf.reduce_sum(rel_error))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))
    
    def result(self):
        return self.total / (self.count + 1e-8)
    
    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


class R2Score(tf.keras.metrics.Metric):
    """
    R² (Coefficient of Determination).
    
    R² = 1 - SS_res / SS_tot
    
    where:
        SS_res = sum((y_true - y_pred)²)
        SS_tot = sum((y_true - mean(y_true))²)
    """
    
    def __init__(self, name: str = "r2", **kwargs):
        super().__init__(name=name, **kwargs)
        self.ss_res = self.add_weight(name="ss_res", initializer="zeros")
        self.ss_tot = self.add_weight(name="ss_tot", initializer="zeros")
        self.sum_y = self.add_weight(name="sum_y", initializer="zeros")
        self.sum_y2 = self.add_weight(name="sum_y2", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        
        n = tf.cast(tf.size(y_true), tf.float32)
        
        self.ss_res.assign_add(tf.reduce_sum(tf.square(y_true - y_pred)))
        self.sum_y.assign_add(tf.reduce_sum(y_true))
        self.sum_y2.assign_add(tf.reduce_sum(tf.square(y_true)))
        self.count.assign_add(n)
    
    def result(self):
        # Compute variance using parallel algorithm
        mean_y = self.sum_y / (self.count + 1e-8)
        ss_tot = self.sum_y2 - 2 * mean_y * self.sum_y + self.count * tf.square(mean_y)
        
        return 1.0 - self.ss_res / (ss_tot + 1e-8)
    
    def reset_state(self):
        self.ss_res.assign(0.0)
        self.ss_tot.assign(0.0)
        self.sum_y.assign(0.0)
        self.sum_y2.assign(0.0)
        self.count.assign(0.0)


class PricingAccuracy(tf.keras.metrics.Metric):
    """
    Pricing Accuracy.
    
    Fraction of predictions within a relative threshold of actual values.
    
    Useful for measuring "good enough" predictions in trading contexts.
    
    Attributes:
        threshold: Acceptable relative error (e.g., 0.01 = 1%)
    """
    
    def __init__(self, threshold: float = 0.01, name: str = "pricing_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        
        # Relative error
        abs_true = tf.abs(y_true) + 1e-8
        rel_error = tf.abs(y_true - y_pred) / abs_true
        
        # Count predictions within threshold
        within_threshold = tf.cast(rel_error <= self.threshold, tf.float32)
        
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            within_threshold = within_threshold * sample_weight
            self.total.assign_add(tf.reduce_sum(sample_weight))
        else:
            self.total.assign_add(tf.cast(tf.size(y_true), tf.float32))
        
        self.correct.assign_add(tf.reduce_sum(within_threshold))
    
    def result(self):
        return self.correct / (self.total + 1e-8)
    
    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)
    
    def get_config(self):
        config = super().get_config()
        config["threshold"] = self.threshold
        return config


class QuantileError(tf.keras.metrics.Metric):
    """
    Quantile Error Metric.
    
    Tracks error at a specific quantile (e.g., 95th percentile).
    
    Note: This is an approximation using reservoir sampling
    for memory efficiency.
    """
    
    def __init__(self, quantile: float = 0.95, name: str = "quantile_error", **kwargs):
        super().__init__(name=name, **kwargs)
        self.quantile = quantile
        self.reservoir_size = 10000
        self.reservoir = self.add_weight(
            name="reservoir",
            shape=(self.reservoir_size,),
            initializer="zeros"
        )
        self.count = self.add_weight(name="count", initializer="zeros")
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        
        errors = tf.abs(y_true - y_pred)
        
        # Simple reservoir sampling approximation
        # For exact quantiles, collect all errors (memory intensive)
        current_count = self.count.numpy()
        new_errors = errors.numpy()
        
        for i, err in enumerate(new_errors):
            idx = int(current_count + i)
            if idx < self.reservoir_size:
                self.reservoir[idx].assign(err)
            else:
                # Reservoir sampling
                j = np.random.randint(0, idx + 1)
                if j < self.reservoir_size:
                    self.reservoir[j].assign(err)
        
        self.count.assign_add(tf.cast(tf.size(errors), tf.float32))
    
    def result(self):
        n = min(int(self.count.numpy()), self.reservoir_size)
        if n == 0:
            return 0.0
        
        errors = self.reservoir[:n].numpy()
        return tf.constant(np.percentile(errors, self.quantile * 100), dtype=tf.float32)
    
    def reset_state(self):
        self.reservoir.assign(tf.zeros((self.reservoir_size,)))
        self.count.assign(0.0)

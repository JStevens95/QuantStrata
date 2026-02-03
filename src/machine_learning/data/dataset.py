"""
TensorFlow-native dataset utilities.

This module provides classes and functions for creating efficient
tf.data.Dataset pipelines for ML training and inference.

Key Features:
    - Automatic batching, shuffling, and prefetching
    - Feature normalization with saved statistics
    - Train/validation/test splitting
    - Memory-efficient data loading

Usage:
    # Create dataset from arrays
    dataset = TFDataset.from_arrays(features, targets)
    
    # Get tf.data.Dataset for training
    train_ds = dataset.to_tf_dataset(batch_size=32, shuffle=True)
    
    # Split into train/val/test
    train_ds, val_ds, test_ds = dataset.split(train=0.7, val=0.15, test=0.15)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf


@dataclass
class NormalizationStats:
    """
    Statistics for feature/target normalization.
    
    Attributes:
        mean: Mean values for each feature
        std: Standard deviation for each feature
        min_val: Minimum values (for min-max scaling)
        max_val: Maximum values (for min-max scaling)
        method: Normalization method ('zscore', 'minmax', 'none')
    """
    mean: np.ndarray
    std: np.ndarray
    min_val: Optional[np.ndarray] = None
    max_val: Optional[np.ndarray] = None
    method: str = "zscore"
    
    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Apply normalization to data."""
        if self.method == "zscore":
            return (data - self.mean) / (self.std + 1e-8)
        elif self.method == "minmax":
            return (data - self.min_val) / (self.max_val - self.min_val + 1e-8)
        return data
    
    def denormalize(self, data: np.ndarray) -> np.ndarray:
        """Reverse normalization."""
        if self.method == "zscore":
            return data * self.std + self.mean
        elif self.method == "minmax":
            return data * (self.max_val - self.min_val) + self.min_val
        return data
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "min_val": self.min_val.tolist() if self.min_val is not None else None,
            "max_val": self.max_val.tolist() if self.max_val is not None else None,
            "method": self.method,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NormalizationStats":
        """Create from dictionary."""
        return cls(
            mean=np.array(d["mean"]),
            std=np.array(d["std"]),
            min_val=np.array(d["min_val"]) if d.get("min_val") else None,
            max_val=np.array(d["max_val"]) if d.get("max_val") else None,
            method=d.get("method", "zscore"),
        )
    
    @classmethod
    def compute(
        cls,
        data: np.ndarray,
        method: str = "zscore",
    ) -> "NormalizationStats":
        """Compute normalization statistics from data."""
        return cls(
            mean=data.mean(axis=0),
            std=data.std(axis=0),
            min_val=data.min(axis=0) if method == "minmax" else None,
            max_val=data.max(axis=0) if method == "minmax" else None,
            method=method,
        )


@dataclass
class TFDataset:
    """
    TensorFlow-native dataset wrapper.
    
    Provides a consistent interface for creating tf.data.Dataset pipelines
    with automatic normalization, batching, and preprocessing.
    
    Attributes:
        features: Feature array of shape [n_samples, n_features]
        targets: Target array of shape [n_samples] or [n_samples, n_outputs]
        feature_names: Optional list of feature column names
        target_names: Optional list of target column names
        feature_stats: Normalization statistics for features
        target_stats: Normalization statistics for targets
        metadata: Additional metadata dictionary
    
    Example:
        # Create from arrays
        dataset = TFDataset.from_arrays(X, y, feature_names=['spot', 'strike', ...])
        
        # Normalize features
        dataset.normalize_features(method='zscore')
        
        # Create tf.data.Dataset
        train_ds = dataset.to_tf_dataset(batch_size=32, shuffle=True)
        
        # Use in training
        model.fit(train_ds, epochs=100)
    """
    features: np.ndarray
    targets: np.ndarray
    feature_names: Optional[List[str]] = None
    target_names: Optional[List[str]] = None
    feature_stats: Optional[NormalizationStats] = None
    target_stats: Optional[NormalizationStats] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __repr__(self) -> str:
        return (
            f"TFDataset(n_samples={len(self)}, "
            f"n_features={self.features.shape[1]}, "
            f"n_targets={self.targets.shape[-1] if self.targets.ndim > 1 else 1})"
        )
    
    @classmethod
    def from_arrays(
        cls,
        features: np.ndarray,
        targets: np.ndarray,
        feature_names: Optional[List[str]] = None,
        target_names: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "TFDataset":
        """
        Create TFDataset from NumPy arrays.
        
        Args:
            features: Feature array [n_samples, n_features]
            targets: Target array [n_samples] or [n_samples, n_outputs]
            feature_names: Optional feature column names
            target_names: Optional target column names
            metadata: Optional metadata dictionary
        
        Returns:
            TFDataset instance
        """
        features = np.asarray(features, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)
        
        return cls(
            features=features,
            targets=targets,
            feature_names=feature_names,
            target_names=target_names,
            metadata=metadata or {},
        )
    
    def normalize_features(
        self,
        method: str = "zscore",
        stats: Optional[NormalizationStats] = None,
    ) -> "TFDataset":
        """
        Normalize features in-place.
        
        Args:
            method: 'zscore' (mean=0, std=1) or 'minmax' (0-1 range)
            stats: Pre-computed statistics (for applying same normalization to test set)
        
        Returns:
            Self for chaining
        """
        if stats is not None:
            self.feature_stats = stats
        else:
            self.feature_stats = NormalizationStats.compute(self.features, method)
        
        self.features = self.feature_stats.normalize(self.features).astype(np.float32)
        return self
    
    def normalize_targets(
        self,
        method: str = "zscore",
        stats: Optional[NormalizationStats] = None,
    ) -> "TFDataset":
        """
        Normalize targets in-place.
        
        Args:
            method: 'zscore' or 'minmax'
            stats: Pre-computed statistics
        
        Returns:
            Self for chaining
        """
        if stats is not None:
            self.target_stats = stats
        else:
            self.target_stats = NormalizationStats.compute(self.targets, method)
        
        self.targets = self.target_stats.normalize(self.targets).astype(np.float32)
        return self
    
    def denormalize_targets(self, predictions: np.ndarray) -> np.ndarray:
        """
        Denormalize predictions back to original scale.
        
        Args:
            predictions: Normalized predictions
        
        Returns:
            Denormalized predictions
        """
        if self.target_stats is None:
            return predictions
        return self.target_stats.denormalize(predictions)
    
    def split(
        self,
        train: float = 0.8,
        val: float = 0.1,
        test: float = 0.1,
        seed: Optional[int] = None,
        stratify: bool = False,
    ) -> Tuple["TFDataset", "TFDataset", "TFDataset"]:
        """
        Split dataset into train/validation/test sets.
        
        Args:
            train: Fraction for training (default 0.8)
            val: Fraction for validation (default 0.1)
            test: Fraction for testing (default 0.1)
            seed: Random seed for reproducibility
            stratify: Whether to stratify by target (for classification)
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        assert abs(train + val + test - 1.0) < 1e-6, "Fractions must sum to 1.0"
        
        n = len(self)
        indices = np.arange(n)
        
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(indices)
        
        n_train = int(n * train)
        n_val = int(n * val)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        def make_split(idx: np.ndarray) -> "TFDataset":
            return TFDataset(
                features=self.features[idx].copy(),
                targets=self.targets[idx].copy(),
                feature_names=self.feature_names,
                target_names=self.target_names,
                feature_stats=self.feature_stats,
                target_stats=self.target_stats,
                metadata=self.metadata.copy(),
            )
        
        return make_split(train_idx), make_split(val_idx), make_split(test_idx)
    
    def to_tf_dataset(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        shuffle_buffer: Optional[int] = None,
        prefetch: int = tf.data.AUTOTUNE,
        cache: bool = True,
        repeat: bool = False,
    ) -> tf.data.Dataset:
        """
        Convert to tf.data.Dataset for efficient training.
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle data
            shuffle_buffer: Buffer size for shuffling (default: dataset size)
            prefetch: Number of batches to prefetch (AUTOTUNE recommended)
            cache: Whether to cache dataset in memory
            repeat: Whether to repeat dataset indefinitely
        
        Returns:
            tf.data.Dataset ready for model.fit()
        """
        ds = tf.data.Dataset.from_tensor_slices((self.features, self.targets))
        
        if cache:
            ds = ds.cache()
        
        if shuffle:
            buffer = shuffle_buffer or len(self)
            ds = ds.shuffle(buffer_size=buffer)
        
        if repeat:
            ds = ds.repeat()
        
        ds = ds.batch(batch_size)
        ds = ds.prefetch(prefetch)
        
        return ds
    
    def to_dict_dataset(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> tf.data.Dataset:
        """
        Convert to tf.data.Dataset with named feature dict.
        
        Useful for models that expect named inputs.
        
        Returns:
            tf.data.Dataset with (feature_dict, targets) structure
        """
        if self.feature_names is None:
            raise ValueError("feature_names required for dict dataset")
        
        feature_dict = {
            name: self.features[:, i:i+1]
            for i, name in enumerate(self.feature_names)
        }
        
        ds = tf.data.Dataset.from_tensor_slices((feature_dict, self.targets))
        
        if shuffle:
            ds = ds.shuffle(buffer_size=len(self))
        
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        
        return ds
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save dataset to disk.
        
        Saves features, targets, and metadata to a directory.
        
        Args:
            path: Directory path to save to
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        np.save(path / "features.npy", self.features)
        np.save(path / "targets.npy", self.targets)
        
        meta = {
            "feature_names": self.feature_names,
            "target_names": self.target_names,
            "feature_stats": self.feature_stats.to_dict() if self.feature_stats else None,
            "target_stats": self.target_stats.to_dict() if self.target_stats else None,
            "metadata": self.metadata,
        }
        
        with open(path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "TFDataset":
        """
        Load dataset from disk.
        
        Args:
            path: Directory path to load from
        
        Returns:
            TFDataset instance
        """
        path = Path(path)
        
        features = np.load(path / "features.npy")
        targets = np.load(path / "targets.npy")
        
        with open(path / "metadata.json", "r") as f:
            meta = json.load(f)
        
        return cls(
            features=features,
            targets=targets,
            feature_names=meta.get("feature_names"),
            target_names=meta.get("target_names"),
            feature_stats=NormalizationStats.from_dict(meta["feature_stats"]) if meta.get("feature_stats") else None,
            target_stats=NormalizationStats.from_dict(meta["target_stats"]) if meta.get("target_stats") else None,
            metadata=meta.get("metadata", {}),
        )


def create_pricing_dataset(
    n_samples: int = 10000,
    spot_range: Tuple[float, float] = (80.0, 120.0),
    strike_range: Tuple[float, float] = (80.0, 120.0),
    vol_range: Tuple[float, float] = (0.1, 0.5),
    rate_range: Tuple[float, float] = (0.01, 0.10),
    expiry_range: Tuple[float, float] = (0.1, 2.0),
    seed: Optional[int] = None,
    pricing_fn: Optional[callable] = None,
) -> TFDataset:
    """
    Generate synthetic option pricing dataset.
    
    Creates random option parameters and computes prices using either
    a provided pricing function or Black-Scholes approximation.
    
    Args:
        n_samples: Number of samples to generate
        spot_range: (min, max) for spot price
        strike_range: (min, max) for strike price
        vol_range: (min, max) for volatility
        rate_range: (min, max) for risk-free rate
        expiry_range: (min, max) for time to expiry
        seed: Random seed
        pricing_fn: Custom pricing function(spot, strike, vol, rate, expiry, is_call) -> price
    
    Returns:
        TFDataset with features and target prices
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random parameters
    spot = np.random.uniform(*spot_range, n_samples)
    strike = np.random.uniform(*strike_range, n_samples)
    vol = np.random.uniform(*vol_range, n_samples)
    rate = np.random.uniform(*rate_range, n_samples)
    expiry = np.random.uniform(*expiry_range, n_samples)
    is_call = np.random.choice([0.0, 1.0], n_samples)  # 0=put, 1=call
    
    # Stack features
    features = np.stack([spot, strike, vol, rate, expiry, is_call], axis=1).astype(np.float32)
    
    # Compute prices
    if pricing_fn is not None:
        prices = np.array([
            pricing_fn(s, k, v, r, t, c)
            for s, k, v, r, t, c in features
        ], dtype=np.float32)
    else:
        # Simple Black-Scholes approximation
        prices = _black_scholes_price(spot, strike, vol, rate, expiry, is_call)
    
    return TFDataset.from_arrays(
        features=features,
        targets=prices,
        feature_names=["spot", "strike", "volatility", "rate", "time_to_expiry", "is_call"],
        target_names=["price"],
        metadata={
            "n_samples": n_samples,
            "spot_range": spot_range,
            "strike_range": strike_range,
            "vol_range": vol_range,
            "rate_range": rate_range,
            "expiry_range": expiry_range,
            "seed": seed,
            "pricing_method": "custom" if pricing_fn else "black_scholes",
        },
    )


def _black_scholes_price(
    spot: np.ndarray,
    strike: np.ndarray,
    vol: np.ndarray,
    rate: np.ndarray,
    expiry: np.ndarray,
    is_call: np.ndarray,
) -> np.ndarray:
    """Simple Black-Scholes price calculation."""
    from scipy.stats import norm
    
    sqrt_t = np.sqrt(expiry)
    d1 = (np.log(spot / strike) + (rate + 0.5 * vol**2) * expiry) / (vol * sqrt_t + 1e-8)
    d2 = d1 - vol * sqrt_t
    
    call_price = spot * norm.cdf(d1) - strike * np.exp(-rate * expiry) * norm.cdf(d2)
    put_price = strike * np.exp(-rate * expiry) * norm.cdf(-d2) - spot * norm.cdf(-d1)
    
    # Select call or put based on is_call flag
    prices = np.where(is_call > 0.5, call_price, put_price)
    
    return prices.astype(np.float32)


def create_calibration_dataset(
    n_samples: int = 5000,
    n_strikes: int = 10,
    n_expiries: int = 5,
    model: str = "heston",
    seed: Optional[int] = None,
) -> TFDataset:
    """
    Generate synthetic calibration dataset.
    
    Creates random model parameters, generates implied volatility surfaces,
    and returns (IV surface, model parameters) pairs for calibration training.
    
    Args:
        n_samples: Number of parameter sets to generate
        n_strikes: Number of strike points in IV surface
        n_expiries: Number of expiry points in IV surface
        model: Target model ('heston', 'sabr')
        seed: Random seed
    
    Returns:
        TFDataset with IV surface features and model parameters as targets
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random model parameters
    if model == "heston":
        # Heston parameters: [v0, kappa, theta, sigma, rho]
        v0 = np.random.uniform(0.01, 0.1, n_samples)       # Initial variance
        kappa = np.random.uniform(0.5, 5.0, n_samples)     # Mean reversion speed
        theta = np.random.uniform(0.01, 0.1, n_samples)    # Long-term variance
        sigma = np.random.uniform(0.1, 0.8, n_samples)     # Vol of vol
        rho = np.random.uniform(-0.9, -0.1, n_samples)     # Correlation
        
        params = np.stack([v0, kappa, theta, sigma, rho], axis=1)
        param_names = ["v0", "kappa", "theta", "sigma", "rho"]
        
    elif model == "sabr":
        # SABR parameters: [alpha, beta, rho, nu]
        alpha = np.random.uniform(0.1, 0.5, n_samples)
        beta = np.random.uniform(0.3, 1.0, n_samples)
        rho = np.random.uniform(-0.8, 0.0, n_samples)
        nu = np.random.uniform(0.1, 0.8, n_samples)
        
        params = np.stack([alpha, beta, rho, nu], axis=1)
        param_names = ["alpha", "beta", "rho", "nu"]
    else:
        raise ValueError(f"Unknown model: {model}")
    
    # Generate synthetic IV surfaces for each parameter set
    # (In practice, you'd use actual model pricing here)
    moneyness = np.linspace(0.8, 1.2, n_strikes)
    expiries = np.linspace(0.1, 2.0, n_expiries)
    
    features_list = []
    for i in range(n_samples):
        # Generate a synthetic IV surface based on parameters
        # This is a simplified approximation
        base_vol = np.sqrt(params[i, 0] if model == "heston" else params[i, 0])
        skew = params[i, -1] if model == "heston" else params[i, 2]  # rho
        
        iv_surface = np.zeros((n_strikes, n_expiries))
        for j, m in enumerate(moneyness):
            for k, t in enumerate(expiries):
                # Simplified IV approximation with skew and term structure
                iv = base_vol * (1 + 0.1 * skew * (m - 1.0) + 0.05 * np.sqrt(t))
                iv_surface[j, k] = max(iv, 0.01)
        
        features_list.append(iv_surface.flatten())
    
    features = np.array(features_list, dtype=np.float32)
    
    # Create feature names
    feature_names = [
        f"iv_m{m:.2f}_t{t:.2f}"
        for m in moneyness
        for t in expiries
    ]
    
    return TFDataset.from_arrays(
        features=features,
        targets=params.astype(np.float32),
        feature_names=feature_names,
        target_names=param_names,
        metadata={
            "n_samples": n_samples,
            "n_strikes": n_strikes,
            "n_expiries": n_expiries,
            "model": model,
            "moneyness": moneyness.tolist(),
            "expiries": expiries.tolist(),
            "seed": seed,
        },
    )

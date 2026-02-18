"""
Pricing model data builder.

Produces train / val / test ``tf.data.Dataset`` splits for pricing models.
Normalisation uses ``sklearn.preprocessing.StandardScaler``; splitting uses
``sklearn.model_selection.train_test_split``.

Usage:
    result = build_pricing_data(n_samples=10_000, seed=42)
    trainer.fit(result.train_ds, result.val_ds)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.machine_learning.data.dataset import (
    build_tf_dataset,
    create_pricing_dataset,
    SyntheticData,
)


@dataclass
class PricingDataResult:
    """
    Result of ``build_pricing_data``.

    Attributes
    ----------
    train_ds : tf.data.Dataset
        Training dataset (batched, shuffled, prefetched).
    val_ds : tf.data.Dataset
        Validation dataset (batched, no shuffle).
    test_ds : tf.data.Dataset
        Test dataset (batched, no shuffle).
    feature_scaler : StandardScaler
        Fitted sklearn scaler for features (for inference denormalisation).
    target_scaler : StandardScaler
        Fitted sklearn scaler for targets (for prediction denormalisation).
    feature_names : list of str
        Feature column names.
    target_names : list of str
        Target column names.
    metadata : dict
        Build metadata (n_samples, splits, seed, etc.).
    """

    train_ds: Any  # tf.data.Dataset
    val_ds: Any    # tf.data.Dataset
    test_ds: Any   # tf.data.Dataset
    feature_scaler: StandardScaler = field(default_factory=StandardScaler)
    target_scaler: StandardScaler = field(default_factory=StandardScaler)
    feature_names: List[str] = field(default_factory=list)
    target_names: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def build_pricing_data(
    n_samples: int = 10_000,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    batch_size: int = 256,
    seed: Optional[int] = None,
    normalize: bool = True,
) -> PricingDataResult:
    """
    Build pricing data and return train / val / test ``tf.data.Dataset`` splits.

    Steps:
        1. Generate synthetic pricing data via ``create_pricing_dataset``.
        2. Split with ``sklearn.model_selection.train_test_split``.
        3. Normalise with ``sklearn.preprocessing.StandardScaler`` (fit on train only).
        4. Wrap each split in a ``tf.data.Dataset`` via ``build_tf_dataset``.

    Parameters
    ----------
    n_samples : int
        Total number of samples to generate.
    train_ratio : float
        Fraction for training (default 0.7).
    val_ratio : float
        Fraction for validation (default 0.15).
    test_ratio : float
        Fraction for test (default 0.15).
    batch_size : int
        Batch size for all datasets.
    seed : int, optional
        Random seed for reproducibility.
    normalize : bool
        Whether to z-score normalise features and targets.

    Returns
    -------
    PricingDataResult
        ``train_ds``, ``val_ds``, ``test_ds``, fitted scalers, and metadata.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    # 1. Generate synthetic data
    data: SyntheticData = create_pricing_dataset(n_samples=n_samples, seed=seed)
    features = data.features
    targets = data.targets.reshape(-1, 1) if data.targets.ndim == 1 else data.targets

    # 2. Split: train | temp → val | test
    temp_size = val_ratio + test_ratio
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        features, targets, test_size=temp_size, random_state=seed,
    )
    relative_test = test_ratio / temp_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=relative_test, random_state=seed,
    )

    # 3. Normalise (fit on training data only)
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    if normalize:
        X_train = feature_scaler.fit_transform(X_train)
        X_val = feature_scaler.transform(X_val)
        X_test = feature_scaler.transform(X_test)
        y_train = target_scaler.fit_transform(y_train)
        y_val = target_scaler.transform(y_val)
        y_test = target_scaler.transform(y_test)

    # 4. Build tf.data.Dataset pipelines
    train_ds = build_tf_dataset(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_ds = build_tf_dataset(X_val, y_val, batch_size=batch_size, shuffle=False)
    test_ds = build_tf_dataset(X_test, y_test, batch_size=batch_size, shuffle=False)

    return PricingDataResult(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        feature_names=data.feature_names,
        target_names=data.target_names,
        metadata={
            "n_samples": n_samples,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "batch_size": batch_size,
            "seed": seed,
            "normalize": normalize,
        },
    )


def build_pricing_dataset_from_fn(
    n_samples: int = 10_000,
    pricing_fn: Optional[Callable[..., np.ndarray]] = None,
    seed: Optional[int] = None,
    spot_range: Tuple[float, float] = (80.0, 120.0),
    strike_range: Tuple[float, float] = (80.0, 120.0),
    vol_range: Tuple[float, float] = (0.1, 0.5),
    rate_range: Tuple[float, float] = (0.01, 0.10),
    expiry_range: Tuple[float, float] = (0.1, 2.0),
) -> SyntheticData:
    """
    Build pricing dataset using a custom pricing callable.

    If ``pricing_fn`` is None, falls back to Black-Scholes via
    ``create_pricing_dataset``.  This is a convenience wrapper for MC
    or analytic pricers.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    pricing_fn : callable, optional
        ``(spot, strike, vol, rate, expiry, is_call) -> price``.
    seed : int, optional
        Random seed.
    spot_range, strike_range, vol_range, rate_range, expiry_range : tuple
        Uniform sampling ranges for each parameter.

    Returns
    -------
    SyntheticData
        Raw features, targets, names, and metadata.  Apply sklearn scaler
        and ``build_tf_dataset`` yourself.
    """
    return create_pricing_dataset(
        n_samples=n_samples,
        seed=seed,
        spot_range=spot_range,
        strike_range=strike_range,
        vol_range=vol_range,
        rate_range=rate_range,
        expiry_range=expiry_range,
        pricing_fn=pricing_fn,
    )


__all__ = [
    "PricingDataResult",
    "build_pricing_data",
    "build_pricing_dataset_from_fn",
]

"""
Pricing model data builder: produces tf.data.Dataset(s) for models/pricing.

Contract: build_pricing_data() returns train_ds, val_ds, test_ds and optional
normalisation stats so the training pipeline has a single, repeatable interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from src.machine_learning.data.dataset import (
    TFDataset,
    NormalizationStats,
    create_pricing_dataset,
)
from src.machine_learning.data.types import MLDataset


@dataclass
class PricingDataResult:
    """
    Result of build_pricing_data(): tf.data.Dataset splits and stats.

    Attributes
    ----------
    train_ds : tf.data.Dataset
        Training dataset (batched, shuffled, prefetched).
    val_ds : tf.data.Dataset
        Validation dataset (batched, no shuffle).
    test_ds : tf.data.Dataset
        Test dataset (batched, no shuffle).
    feature_stats : NormalizationStats or None
        Stats used to normalise features (for inference).
    target_stats : NormalizationStats or None
        Stats used to normalise targets (for denormalising predictions).
    metadata : dict
        Data build metadata (n_samples, splits, seed).
    """

    train_ds: tf.data.Dataset
    val_ds: tf.data.Dataset
    test_ds: tf.data.Dataset
    feature_stats: Optional[NormalizationStats] = None
    target_stats: Optional[NormalizationStats] = None
    metadata: dict = field(default_factory=dict)


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
    Build pricing data and return train/val/test tf.data.Dataset(s).

    Uses the generic TFDataset and create_pricing_dataset; split and
    batching are applied so pipelines receive a consistent interface.

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
        train_ds, val_ds, test_ds, feature_stats, target_stats, metadata.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    dataset = create_pricing_dataset(n_samples=n_samples, seed=seed)
    if normalize:
        dataset.normalize_features(method="zscore").normalize_targets(method="zscore")

    train_ds_split, val_ds_split, test_ds_split = dataset.split(
        train=train_ratio,
        val=val_ratio,
        test=test_ratio,
        seed=seed,
    )

    train_ds = train_ds_split.to_tf_dataset(
        batch_size=batch_size,
        shuffle=True,
        prefetch=tf.data.AUTOTUNE,
    )
    val_ds = val_ds_split.to_tf_dataset(
        batch_size=batch_size,
        shuffle=False,
        prefetch=tf.data.AUTOTUNE,
    )
    test_ds = test_ds_split.to_tf_dataset(
        batch_size=batch_size,
        shuffle=False,
        prefetch=tf.data.AUTOTUNE,
    )

    return PricingDataResult(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        feature_stats=dataset.feature_stats,
        target_stats=dataset.target_stats,
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


def build_pricing_dataset_from_mc(
    n_samples: int = 10_000,
    paths_or_fn: Optional[Union[Tuple[np.ndarray, np.ndarray], Callable[..., np.ndarray]]] = None,
    seed: Optional[int] = None,
    spot_range: Tuple[float, float] = (80.0, 120.0),
    strike_range: Tuple[float, float] = (80.0, 120.0),
    vol_range: Tuple[float, float] = (0.1, 0.5),
    rate_range: Tuple[float, float] = (0.01, 0.10),
    expiry_range: Tuple[float, float] = (0.1, 2.0),
) -> MLDataset:
    """
    Build pricing dataset from MC paths or a pricing callable (MC-style).

    If paths_or_fn is a callable, it should have signature (spot, strike, vol, rate, expiry, is_call) -> prices
    (e.g. wrapping an MC pricer). If it is a tuple (features, prices), those arrays are used directly.
    Otherwise uses create_pricing_dataset (synthetic Black-Scholes).

    Returns
    -------
    MLDataset
        features, targets, feature_names, target_names.
    """
    if paths_or_fn is None:
        ds = create_pricing_dataset(
            n_samples=n_samples,
            seed=seed,
            spot_range=spot_range,
            strike_range=strike_range,
            vol_range=vol_range,
            rate_range=rate_range,
            expiry_range=expiry_range,
        )
        return MLDataset(
            features=ds.features,
            targets=ds.targets,
            feature_names=ds.feature_names,
            target_names=ds.target_names,
            metadata=getattr(ds, "metadata", {}),
        )
    if callable(paths_or_fn):
        ds = create_pricing_dataset(
            n_samples=n_samples,
            seed=seed,
            spot_range=spot_range,
            strike_range=strike_range,
            vol_range=vol_range,
            rate_range=rate_range,
            expiry_range=expiry_range,
            pricing_fn=paths_or_fn,
        )
        return MLDataset(
            features=ds.features,
            targets=ds.targets,
            feature_names=ds.feature_names,
            target_names=ds.target_names,
            metadata=getattr(ds, "metadata", {}),
        )
    features, prices = paths_or_fn
    return MLDataset(
        features=np.asarray(features),
        targets=np.asarray(prices),
        feature_names=["spot", "strike", "volatility", "rate", "time_to_expiry", "is_call"],
        target_names=["price"],
        metadata={"n_samples": len(features), "source": "mc_paths"},
    )


def build_pricing_dataset_from_analytic(
    n_samples: int = 10_000,
    pricer_fn: Optional[Callable[..., np.ndarray]] = None,
    seed: Optional[int] = None,
    spot_range: Tuple[float, float] = (80.0, 120.0),
    strike_range: Tuple[float, float] = (80.0, 120.0),
    vol_range: Tuple[float, float] = (0.1, 0.5),
    rate_range: Tuple[float, float] = (0.01, 0.10),
    expiry_range: Tuple[float, float] = (0.1, 2.0),
) -> MLDataset:
    """
    Build pricing dataset using an analytic pricer.

    pricer_fn(spot, strike, vol, rate, expiry, is_call) -> prices. If None, uses Black-Scholes
    via create_pricing_dataset.

    Returns
    -------
    MLDataset
        features, targets, feature_names, target_names.
    """
    ds = create_pricing_dataset(
        n_samples=n_samples,
        seed=seed,
        spot_range=spot_range,
        strike_range=strike_range,
        vol_range=vol_range,
        rate_range=rate_range,
        expiry_range=expiry_range,
        pricing_fn=pricer_fn,
    )
    return MLDataset(
        features=ds.features,
        targets=ds.targets,
        feature_names=ds.feature_names,
        target_names=ds.target_names,
        metadata=getattr(ds, "metadata", {}),
    )


__all__ = [
    "PricingDataResult",
    "build_pricing_data",
    "build_pricing_dataset_from_mc",
    "build_pricing_dataset_from_analytic",
]

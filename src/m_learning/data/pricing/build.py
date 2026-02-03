"""
Pricing model data builder: produces tf.data.Dataset(s) for models/pricing.

Contract: build_pricing_data() returns train_ds, val_ds, test_ds and optional
normalisation stats so the training pipeline has a single, repeatable interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import tensorflow as tf

from src.m_learning.data.dataset import (
    TFDataset,
    NormalizationStats,
    create_pricing_dataset,
)


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


__all__ = ["PricingDataResult", "build_pricing_data"]

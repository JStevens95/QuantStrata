"""
Calibration model data builder.

Produces train / val / test ``tf.data.Dataset`` splits for calibration models.
Normalisation uses ``sklearn.preprocessing.StandardScaler``; splitting uses
``sklearn.model_selection.train_test_split``.

Usage:
    result = build_calibration_data(n_samples=5000, model="heston", seed=42)
    trainer.fit(result.train_ds, result.val_ds)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.machine_learning.data.dataset import (
    build_tf_dataset,
    create_calibration_dataset,
    SyntheticData,
)


@dataclass
class CalibrationDataResult:
    """
    Result of ``build_calibration_data``.

    Attributes
    ----------
    train_ds : tf.data.Dataset
        Training dataset (batched, shuffled, prefetched).
    val_ds : tf.data.Dataset
        Validation dataset (batched, no shuffle).
    test_ds : tf.data.Dataset
        Test dataset (batched, no shuffle).
    feature_scaler : StandardScaler
        Fitted sklearn scaler for features (IV surfaces).
    target_scaler : StandardScaler
        Fitted sklearn scaler for targets (model parameters).
    feature_names : list of str
        Feature column names.
    target_names : list of str
        Target (parameter) column names.
    metadata : dict
        Build metadata.
    """

    train_ds: Any  # tf.data.Dataset
    val_ds: Any    # tf.data.Dataset
    test_ds: Any   # tf.data.Dataset
    feature_scaler: StandardScaler = field(default_factory=StandardScaler)
    target_scaler: StandardScaler = field(default_factory=StandardScaler)
    feature_names: list = field(default_factory=list)
    target_names: list = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def build_calibration_data(
    n_samples: int = 5000,
    n_strikes: int = 10,
    n_expiries: int = 5,
    model: str = "heston",
    test_size: float = 0.2,
    batch_size: int = 256,
    normalize: bool = True,
    seed: Optional[int] = None,
) -> CalibrationDataResult:
    """
    Build calibration dataset: (IV surface) -> (model parameters).

    Steps:
        1. Generate synthetic IV / parameter pairs via ``create_calibration_dataset``.
        2. Split with ``sklearn.model_selection.train_test_split``.
        3. Normalise with ``sklearn.preprocessing.StandardScaler`` (fit on train only).
        4. Wrap each split in a ``tf.data.Dataset`` via ``build_tf_dataset``.

    Parameters
    ----------
    n_samples : int
        Number of calibration samples.
    n_strikes : int
        Strike points in IV surface.
    n_expiries : int
        Expiry points in IV surface.
    model : str
        Target model (``"heston"`` or ``"sabr"``).
    test_size : float
        Fraction for test split (validation is half of test).
    batch_size : int
        Batch size for all datasets.
    normalize : bool
        Whether to z-score normalise features and targets.
    seed : int, optional
        Random seed.

    Returns
    -------
    CalibrationDataResult
        ``train_ds``, ``val_ds``, ``test_ds``, fitted scalers, and metadata.
    """
    # 1. Generate synthetic data
    data: SyntheticData = create_calibration_dataset(
        n_samples=n_samples,
        n_strikes=n_strikes,
        n_expiries=n_expiries,
        model=model,
        seed=seed,
    )
    features = data.features
    targets = data.targets

    # 2. Split: train | val | test
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        features, targets, test_size=test_size, random_state=seed,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=seed,
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

    return CalibrationDataResult(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        feature_names=data.feature_names,
        target_names=data.target_names,
        metadata={
            "n_samples": n_samples,
            "n_strikes": n_strikes,
            "n_expiries": n_expiries,
            "model": model,
            "test_size": test_size,
            "batch_size": batch_size,
            "normalize": normalize,
            "seed": seed,
        },
    )


__all__ = ["CalibrationDataResult", "build_calibration_data"]

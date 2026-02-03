"""
Data types for QuantStrata ML data preparation.

- MLDataset: Container for features, targets, and optional metadata.
- PricingFeatures: Standard feature schema for option pricing.
- CalibrationFeatures: Standard feature schema for model calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class MLDataset:
    """
    Generic container for ML data.

    Parameters
    ----------
    features : ndarray
        Input features, shape (n_samples, n_features) or dict of arrays.
    targets : ndarray
        Target values, shape (n_samples,) or (n_samples, n_outputs).
    feature_names : list of str, optional
        Names of feature columns.
    target_names : list of str, optional
        Names of target columns.
    metadata : dict, optional
        Additional metadata (e.g. instrument types, dates).

    Example
    -------
    >>> dataset = MLDataset(features=X, targets=y, feature_names=["spot", "strike", "vol"])
    >>> print(dataset.features.shape)
    (1000, 3)
    """

    features: np.ndarray
    targets: np.ndarray
    feature_names: Optional[List[str]] = None
    target_names: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.features = np.asarray(self.features)
        self.targets = np.asarray(self.targets)

    def __len__(self) -> int:
        return len(self.features)

    def split(
        self, train_ratio: float = 0.8, shuffle: bool = True, seed: Optional[int] = None
    ) -> tuple["MLDataset", "MLDataset"]:
        """
        Split into train and test datasets.

        Parameters
        ----------
        train_ratio : float
            Fraction of data for training.
        shuffle : bool
            Whether to shuffle before splitting.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        train_dataset, test_dataset : MLDataset, MLDataset
        """
        n = len(self)
        indices = np.arange(n)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)
        split_idx = int(n * train_ratio)
        train_idx, test_idx = indices[:split_idx], indices[split_idx:]
        return (
            MLDataset(
                features=self.features[train_idx],
                targets=self.targets[train_idx],
                feature_names=self.feature_names,
                target_names=self.target_names,
                metadata=self.metadata,
            ),
            MLDataset(
                features=self.features[test_idx],
                targets=self.targets[test_idx],
                feature_names=self.feature_names,
                target_names=self.target_names,
                metadata=self.metadata,
            ),
        )


@dataclass
class PricingFeatures:
    """
    Standard feature schema for option pricing.

    All values are scalars or arrays of shape (n_samples,).

    Parameters
    ----------
    spot : float or ndarray
        Spot price.
    strike : float or ndarray
        Strike price.
    vol : float or ndarray
        Volatility (e.g. implied or flat vol).
    rate : float or ndarray
        Risk-free rate (domestic).
    rate_foreign : float or ndarray, optional
        Foreign rate (for FX options).
    expiry : float or ndarray
        Time to expiry (years).
    option_type : int or ndarray
        Option type: 1 for call, -1 for put.
    """

    spot: np.ndarray
    strike: np.ndarray
    vol: np.ndarray
    rate: np.ndarray
    expiry: np.ndarray
    option_type: np.ndarray
    rate_foreign: Optional[np.ndarray] = None

    def to_array(self) -> np.ndarray:
        """Convert to feature matrix (n_samples, n_features)."""
        arrays = [
            np.asarray(self.spot).reshape(-1, 1),
            np.asarray(self.strike).reshape(-1, 1),
            np.asarray(self.vol).reshape(-1, 1),
            np.asarray(self.rate).reshape(-1, 1),
            np.asarray(self.expiry).reshape(-1, 1),
            np.asarray(self.option_type).reshape(-1, 1),
        ]
        if self.rate_foreign is not None:
            arrays.append(np.asarray(self.rate_foreign).reshape(-1, 1))
        return np.hstack(arrays)

    @staticmethod
    def feature_names(include_rate_foreign: bool = False) -> List[str]:
        names = ["spot", "strike", "vol", "rate", "expiry", "option_type"]
        if include_rate_foreign:
            names.append("rate_foreign")
        return names


@dataclass
class CalibrationFeatures:
    """
    Standard feature schema for model calibration.

    Used for training ML models that predict calibrated parameters from market data.

    Parameters
    ----------
    market_quotes : ndarray
        Market observables (e.g. option prices or implied vols), shape (n_samples, n_quotes).
    strikes : ndarray, optional
        Strike grid, shape (n_samples, n_strikes) or (n_strikes,).
    expiries : ndarray, optional
        Expiry grid, shape (n_samples, n_expiries) or (n_expiries,).
    spot : ndarray, optional
        Spot price, shape (n_samples,).
    forward : ndarray, optional
        Forward price, shape (n_samples,).
    """

    market_quotes: np.ndarray
    strikes: Optional[np.ndarray] = None
    expiries: Optional[np.ndarray] = None
    spot: Optional[np.ndarray] = None
    forward: Optional[np.ndarray] = None

    def to_array(self) -> np.ndarray:
        """Convert to feature matrix (n_samples, n_features)."""
        features = [np.asarray(self.market_quotes)]
        if self.strikes is not None:
            s = np.asarray(self.strikes)
            if s.ndim == 1:
                s = np.tile(s, (len(self.market_quotes), 1))
            features.append(s)
        if self.expiries is not None:
            e = np.asarray(self.expiries)
            if e.ndim == 1:
                e = np.tile(e, (len(self.market_quotes), 1))
            features.append(e)
        if self.spot is not None:
            features.append(np.asarray(self.spot).reshape(-1, 1))
        if self.forward is not None:
            features.append(np.asarray(self.forward).reshape(-1, 1))
        return np.hstack(features)


__all__ = ["MLDataset", "PricingFeatures", "CalibrationFeatures"]

"""
Calibration model data builder: produces datasets for ML-based calibration.

Contract: build_calibration_dataset() returns an MLDataset (or TFDataset) with
features = market quotes / IV surface, targets = model parameters, for use by
the generic training pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np

from src.machine_learning.data.dataset import (
    TFDataset,
    create_calibration_dataset,
)
from src.machine_learning.data.types import MLDataset


@dataclass
class CalibrationDataResult:
    """
    Result of build_calibration_data(): dataset and metadata.

    Attributes
    ----------
    features : np.ndarray
        Market/IV features (e.g. flattened IV surface).
    targets : np.ndarray
        Model parameters (e.g. Heston/SABR).
    feature_names : list
        Names of feature dimensions.
    target_names : list
        Names of parameter dimensions.
    metadata : dict
        Build metadata (model, n_strikes, n_expiries, etc.).
    """

    features: np.ndarray
    targets: np.ndarray
    feature_names: list
    target_names: list
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_ml_dataset(self) -> MLDataset:
        """Convert to MLDataset for generic pipeline."""
        return MLDataset(
            features=self.features,
            targets=self.targets,
            feature_names=self.feature_names,
            target_names=self.target_names,
            metadata=self.metadata,
        )


def build_calibration_dataset(
    n_samples: int = 5000,
    n_strikes: int = 10,
    n_expiries: int = 5,
    model: str = "heston",
    seed: Optional[int] = None,
    forward_model_fn: Optional[Callable[..., np.ndarray]] = None,
) -> MLDataset:
    """
    Build calibration dataset: (market observables / IV surface) -> model parameters.

    Uses create_calibration_dataset from dataset.py (synthetic IV surface from
    random Heston/SABR params). If forward_model_fn is provided, it can be used
    to generate observables from parameters (signature left to caller; this
    builder still uses the synthetic path unless integrated elsewhere).

    Parameters
    ----------
    n_samples : int
        Number of calibration samples.
    n_strikes : int
        Strike points in IV surface.
    n_expiries : int
        Expiry points in IV surface.
    model : str
        Target model ('heston', 'sabr').
    seed : int, optional
        Random seed.
    forward_model_fn : callable, optional
        Optional forward model (params -> market quotes) for real integration.

    Returns
    -------
    MLDataset
        features (IV/market), targets (model parameters), names, metadata.
    """
    ds: TFDataset = create_calibration_dataset(
        n_samples=n_samples,
        n_strikes=n_strikes,
        n_expiries=n_expiries,
        model=model,
        seed=seed,
    )
    return MLDataset(
        features=ds.features,
        targets=ds.targets,
        feature_names=ds.feature_names,
        target_names=ds.target_names,
        metadata=getattr(ds, "metadata", {}),
    )


__all__ = ["build_calibration_dataset", "CalibrationDataResult"]

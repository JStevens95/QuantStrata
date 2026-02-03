"""
Data adapters for ML-based calibration.

Provides functions to build MLDataset for training calibration networks.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np

from src.m_learning.data.types import MLDataset, CalibrationFeatures


def build_calibration_dataset(
    n_samples: int,
    param_ranges: List[Tuple[float, float]],
    forward_model_fn: Callable[..., np.ndarray],
    strikes: np.ndarray,
    expiries: np.ndarray,
    spot_range: Tuple[float, float] = (90.0, 110.0),
    noise_std: float = 0.0,
    seed: Optional[int] = None,
) -> MLDataset:
    """
    Build a calibration dataset by sampling model parameters and generating market quotes.

    This is the inverse problem: given market quotes (implied vols or prices), predict
    the underlying model parameters.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    param_ranges : list of tuple
        (min, max) for each model parameter (e.g. [(0.01, 0.5), (-0.9, 0.9), ...]).
    forward_model_fn : callable
        Function that takes (params, strikes, expiries, spot) and returns
        market quotes (e.g. implied vols), shape (n_strikes * n_expiries,).
    strikes : ndarray
        Strike grid.
    expiries : ndarray
        Expiry grid.
    spot_range : tuple
        (min, max) for spot price.
    noise_std : float
        Standard deviation of Gaussian noise to add to quotes (for robustness).
    seed : int, optional
        Random seed.

    Returns
    -------
    MLDataset
        Features = market quotes (+ optional strikes, expiries, spot).
        Targets = model parameters.

    Example
    -------
    >>> def sabr_implied_vol(params, strikes, expiries, spot):
    ...     # Return implied vol surface (flattened)
    ...     alpha, rho, nu = params
    ...     ...
    ...     return ivs.flatten()
    >>> dataset = build_calibration_dataset(
    ...     n_samples=5000,
    ...     param_ranges=[(0.1, 0.5), (-0.9, 0.9), (0.1, 1.0)],
    ...     forward_model_fn=sabr_implied_vol,
    ...     strikes=np.array([90, 100, 110]),
    ...     expiries=np.array([0.25, 0.5, 1.0]),
    ... )
    """
    rng = np.random.default_rng(seed)
    n_params = len(param_ranges)

    # Sample parameters
    params_array = np.zeros((n_samples, n_params))
    for i, (lo, hi) in enumerate(param_ranges):
        params_array[:, i] = rng.uniform(lo, hi, n_samples)

    # Sample spots
    spots = rng.uniform(*spot_range, n_samples)

    # Generate market quotes
    n_quotes = len(strikes) * len(expiries)
    quotes_array = np.zeros((n_samples, n_quotes))
    for i in range(n_samples):
        quotes = forward_model_fn(params_array[i], strikes, expiries, spots[i])
        if noise_std > 0:
            quotes = quotes + rng.normal(0, noise_std, len(quotes))
        quotes_array[i] = quotes

    # Build features
    features_obj = CalibrationFeatures(
        market_quotes=quotes_array,
        strikes=strikes,
        expiries=expiries,
        spot=spots,
    )

    return MLDataset(
        features=features_obj.to_array(),
        targets=params_array,
        feature_names=[f"quote_{i}" for i in range(n_quotes)] + [f"strike_{i}" for i in range(len(strikes))] + [f"expiry_{i}" for i in range(len(expiries))] + ["spot"],
        target_names=[f"param_{i}" for i in range(n_params)],
        metadata={"method": "calibration", "n_samples": n_samples, "n_params": n_params},
    )


__all__ = ["build_calibration_dataset"]

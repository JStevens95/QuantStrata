"""
GNN-RNN hybrid model data builder: produces tf.data.Dataset(s) for models/gnn_rnn_hybrid.

Contract: build_gnn_data() returns train_ds, val_ds, proj_ds so the training
pipeline has a single, repeatable interface. Uses synthetic or FX portfolio data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf

from src.m_learning.data.gnn_synthetic import SyntheticGnnData, generate_synthetic_gnn_data
from src.m_learning.data.portfolio_builder import (
    GnnPortfolioData,
    build_fx_gnn_data,
    train_val_projection_split,
)
from src.m_learning.data.portfolio import gnn_inputs_to_tf_dataset


@dataclass
class GnnDataResult:
    """
    Result of build_gnn_data(): tf.data.Dataset splits for GNN-RNN hybrid.

    Attributes
    ----------
    train_ds : tf.data.Dataset
        Training dataset (batched, shuffled).
    val_ds : tf.data.Dataset
        Validation dataset (batched, no shuffle).
    proj_ds : tf.data.Dataset
        Projection/holdout dataset (batched, no shuffle).
    metadata : dict
        Data build metadata (n_trades, n_samples, splits, etc.).
    """

    train_ds: tf.data.Dataset
    val_ds: tf.data.Dataset
    proj_ds: tf.data.Dataset
    metadata: dict = field(default_factory=dict)


def _split_gnn_data(
    data: Any,
    train_ratio: float,
    val_ratio: float,
    projection_ratio: float,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, np.ndarray], np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """
    Split GNN data by sample index. Works with GnnPortfolioData or SyntheticGnnData.
    """
    n = data.pnl_history.shape[0]
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    def slice_data(start: int, length: int) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        pnl = data.pnl_history[start : start + length]
        tgt = data.targets[start : start + length]
        inputs = {
            "trade_features": np.tile(data.trade_features, (length, 1, 1)),
            "adjacency_matrix": np.tile(data.adjacency_matrix, (length, 1, 1)),
            "pnl_history": pnl,
            "target_indices": np.tile(data.target_indices, (length, 1)),
            "elementary_indices": np.tile(data.elementary_indices, (length, 1)),
        }
        return inputs, tgt.astype(np.float32)

    train_inputs, train_targets = slice_data(0, n_train)
    val_inputs, val_targets = slice_data(n_train, n_val)
    proj_inputs, proj_targets = slice_data(n_train + n_val, n - n_train - n_val)
    return train_inputs, train_targets, val_inputs, val_targets, proj_inputs, proj_targets


def build_gnn_data(
    use_synthetic: bool = True,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    projection_ratio: float = 0.2,
    batch_size: int = 32,
    seed: Optional[int] = None,
    # Synthetic params
    n_trades: int = 50,
    n_elementary: int = 30,
    n_targets: int = 10,
    n_samples: int = 500,
    n_timesteps: int = 20,
    k_neighbours: int = 5,
    noise_std: float = 0.5,
    # FX params (used when use_synthetic=False)
    n_vanilla: int = 100,
    n_digital: int = 100,
    n_barrier: int = 5,
    n_double_barrier: int = 5,
    n_asian: int = 5,
    n_touch: int = 5,
    spot: float = 1.10,
    sigma: float = 0.15,
) -> GnnDataResult:
    """
    Build GNN-RNN data and return train/val/projection tf.data.Dataset(s).

    Parameters
    ----------
    use_synthetic : bool
        If True, use generate_synthetic_gnn_data; else use build_fx_gnn_data.
    train_ratio, val_ratio, projection_ratio : float
        Splits (must sum to 1.0).
    batch_size : int
        Batch size for all datasets.
    seed : int, optional
        Random seed.
    n_trades, n_elementary, n_targets, n_samples, n_timesteps, k_neighbours, noise_std : int/float
        Used when use_synthetic=True.
    n_vanilla, n_digital, n_barrier, ... : int/float
        Used when use_synthetic=False for FX portfolio.

    Returns
    -------
    GnnDataResult
        train_ds, val_ds, proj_ds, metadata.
    """
    if abs(train_ratio + val_ratio + projection_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + projection_ratio must equal 1.0")

    if use_synthetic:
        data: Any = generate_synthetic_gnn_data(
            n_trades=n_trades,
            n_elementary=n_elementary,
            n_targets=n_targets,
            n_samples=n_samples,
            n_timesteps=n_timesteps,
            k_neighbours=k_neighbours,
            noise_std=noise_std,
            seed=seed,
        )
        train_inputs, train_targets, val_inputs, val_targets, proj_inputs, proj_targets = _split_gnn_data(
            data, train_ratio, val_ratio, projection_ratio
        )
        metadata = {
            "use_synthetic": True,
            "n_trades": n_trades,
            "n_elementary": n_elementary,
            "n_targets": n_targets,
            "n_samples": n_samples,
            "n_timesteps": n_timesteps,
            "k_neighbours": k_neighbours,
            "seed": seed,
        }
    else:
        data = build_fx_gnn_data(
            n_vanilla=n_vanilla,
            n_digital=n_digital,
            n_barrier=n_barrier,
            n_double_barrier=n_double_barrier,
            n_asian=n_asian,
            n_touch=n_touch,
            n_samples=n_samples,
            n_timesteps=n_timesteps,
            k_neighbours=k_neighbours,
            spot=spot,
            sigma=sigma,
            noise_std=noise_std,
            seed=seed,
        )
        train_inputs, train_targets, val_inputs, val_targets, proj_inputs, proj_targets = train_val_projection_split(
            data, train_ratio=train_ratio, val_ratio=val_ratio, projection_ratio=projection_ratio
        )
        metadata = {
            "use_synthetic": False,
            "n_samples": n_samples,
            "n_timesteps": n_timesteps,
            "seed": seed,
        }

    train_ds = gnn_inputs_to_tf_dataset(train_inputs, train_targets, batch_size=batch_size, shuffle=True)
    val_ds = gnn_inputs_to_tf_dataset(val_inputs, val_targets, batch_size=batch_size, shuffle=False)
    proj_ds = gnn_inputs_to_tf_dataset(proj_inputs, proj_targets, batch_size=batch_size, shuffle=False)

    return GnnDataResult(
        train_ds=train_ds,
        val_ds=val_ds,
        proj_ds=proj_ds,
        metadata={**metadata, "train_ratio": train_ratio, "val_ratio": val_ratio, "projection_ratio": projection_ratio, "batch_size": batch_size},
    )


__all__ = ["GnnDataResult", "build_gnn_data", "_split_gnn_data"]

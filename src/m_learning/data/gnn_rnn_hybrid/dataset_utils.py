"""
Dataset utilities for GNN-RNN hybrid: build GNN input dicts and tf.data.Dataset.

Used by data/gnn_rnn_hybrid/build.py to produce batched tf.data.Dataset(s)
for the HybridGnnRnn model.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

try:
    import tensorflow as tf
except ImportError as e:
    tf = None  # type: ignore


def build_gnn_dataset_from_portfolio(
    trade_features: np.ndarray,
    adjacency_matrix: np.ndarray,
    pnl_history: np.ndarray,
    target_indices: np.ndarray,
    elementary_indices: Optional[np.ndarray] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, np.ndarray]:
    """
    Build GNN inputs from portfolio data (dict expected by HybridGnnRnn).

    Parameters
    ----------
    trade_features : ndarray
        Shape (n_trades, n_features).
    adjacency_matrix : ndarray
        Shape (n_trades, n_trades).
    pnl_history : ndarray
        Shape (n_samples, n_timesteps, n_elementary).
    target_indices : ndarray
        Shape (n_targets,).
    elementary_indices : ndarray, optional
        Shape (n_elementary,).
    metadata : dict, optional
        Additional metadata.

    Returns
    -------
    dict
        trade_features, adjacency_matrix, pnl_history, target_indices, optional elementary_indices.
    """
    inputs = {
        "trade_features": np.asarray(trade_features),
        "adjacency_matrix": np.asarray(adjacency_matrix),
        "pnl_history": np.asarray(pnl_history),
        "target_indices": np.asarray(target_indices),
    }
    if elementary_indices is not None:
        inputs["elementary_indices"] = np.asarray(elementary_indices)
    if metadata is not None:
        inputs["metadata"] = metadata
    return inputs


def gnn_inputs_to_tf_dataset(
    gnn_inputs: Dict[str, np.ndarray],
    targets: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
) -> Any:
    """
    Convert GNN inputs to tf.data.Dataset yielding (inputs_dict, targets) batches.

    Batched dict has trade_features (B, n_trades, n_features), adjacency_matrix (B, n_trades, n_trades),
    pnl_history (B, n_timesteps, n_elementary), target_indices (B, n_targets). The model (HybridGnnRnn)
    accepts these and squeezes the batch dimension for the static graph inputs internally.
    """
    if tf is None:
        raise ImportError("TensorFlow required for gnn_inputs_to_tf_dataset") from None

    pnl_history = gnn_inputs["pnl_history"]
    n_samples = pnl_history.shape[0]

    trade_features = np.tile(gnn_inputs["trade_features"], (n_samples, 1, 1))
    adjacency_matrix = np.tile(gnn_inputs["adjacency_matrix"], (n_samples, 1, 1))
    target_indices = np.tile(gnn_inputs["target_indices"], (n_samples, 1))
    elementary_indices = gnn_inputs.get("elementary_indices")
    if elementary_indices is not None:
        elementary_indices = np.tile(elementary_indices, (n_samples, 1))

    input_dict = {
        "trade_features": trade_features,
        "adjacency_matrix": adjacency_matrix,
        "pnl_history": pnl_history,
        "target_indices": target_indices,
    }
    if elementary_indices is not None:
        input_dict["elementary_indices"] = elementary_indices

    ds = tf.data.Dataset.from_tensor_slices((input_dict, targets))
    if shuffle:
        ds = ds.shuffle(buffer_size=n_samples)
    ds = ds.batch(batch_size)
    return ds


__all__ = ["build_gnn_dataset_from_portfolio", "gnn_inputs_to_tf_dataset"]

"""
Data adapters for portfolio/GNN inputs.

Provides functions to build GNN inputs from portfolio representation,
integrating with TradeGraphBuilder and trade_attribute_encoder.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.m_learning.data.types import MLDataset


def build_gnn_dataset_from_portfolio(
    trade_features: np.ndarray,
    adjacency_matrix: np.ndarray,
    pnl_history: np.ndarray,
    target_indices: np.ndarray,
    elementary_indices: Optional[np.ndarray] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, np.ndarray]:
    """
    Build GNN inputs from portfolio data.

    This is a thin wrapper that packages the inputs expected by HybridGnnRnn.
    For full portfolio → GNN data flow, use TradeGraphBuilder and
    trade_attribute_encoder to construct trade_features and adjacency_matrix.

    Parameters
    ----------
    trade_features : ndarray
        Trade attribute features, shape (n_trades, n_features).
    adjacency_matrix : ndarray
        Trade relationship graph, shape (n_trades, n_trades).
    pnl_history : ndarray
        Historical PnL for elementary trades, shape (n_samples, n_timesteps, n_elementary).
    target_indices : ndarray
        Indices of target trades, shape (n_targets,).
    elementary_indices : ndarray, optional
        Indices of elementary trades, shape (n_elementary,).
    metadata : dict, optional
        Additional metadata.

    Returns
    -------
    dict
        Dictionary of inputs for HybridGnnRnn:
        - trade_features: (n_trades, n_features)
        - adjacency_matrix: (n_trades, n_trades)
        - pnl_history: (n_samples, n_timesteps, n_elementary)
        - target_indices: (n_targets,)
        - elementary_indices: (n_elementary,) if provided

    Example
    -------
    >>> from src.m_learning.utilities.trade_graph_builder import TradeGraphBuilder
    >>> builder = TradeGraphBuilder(k=10)
    >>> adj = builder.build(trade_features)
    >>> gnn_inputs = build_gnn_dataset_from_portfolio(
    ...     trade_features, adj, pnl_history, target_indices
    ... )
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
    Convert GNN inputs to a TensorFlow Dataset.

    Parameters
    ----------
    gnn_inputs : dict
        Output from build_gnn_dataset_from_portfolio.
    targets : ndarray
        Target PnL values, shape (n_samples, n_targets).
    batch_size : int
        Batch size.
    shuffle : bool
        Whether to shuffle.

    Returns
    -------
    tf.data.Dataset
        Dataset yielding (inputs_dict, targets) batches.
    """
    try:
        import tensorflow as tf
    except ImportError as e:
        raise ImportError("TensorFlow required for gnn_inputs_to_tf_dataset") from e

    # Build dataset from dict of arrays
    # For batching, we need to handle the static (per-portfolio) vs dynamic (per-sample) arrays
    # trade_features and adjacency_matrix are typically static; pnl_history is per-sample
    pnl_history = gnn_inputs["pnl_history"]
    n_samples = pnl_history.shape[0]

    # Tile static features
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

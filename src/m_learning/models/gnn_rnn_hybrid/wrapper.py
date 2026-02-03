"""
Wrapper for HybridGnnRnn to handle batched inputs.

The HybridGnnRnn model expects rank-2 trade_features and adjacency_matrix
(one graph shared across the batch). When using tf.data.Dataset with batching,
these arrays become rank-3 (batch, n_trades, ...). This wrapper squeezes
the batch dimension for the static graph inputs before calling the model.
"""

from __future__ import annotations

from typing import Any, Dict, Union

import tensorflow as tf

from src.m_learning.models.gnn_rnn_hybrid.hybrid_model import HybridGnnRnn


class BatchedHybridGnnRnn(tf.keras.Model):
    """
    Wrapper around HybridGnnRnn that handles batched graph inputs.

    When using tf.data.Dataset, trade_features and adjacency_matrix
    are tiled to (batch, n_trades, ...). This wrapper takes the first
    element [0] to get the rank-2 arrays expected by HybridGnnRnn.

    The wrapper assumes the graph is identical for all samples in a batch.

    Example
    -------
    >>> model_config = default_hybrid_model_config(n_targets=10)
    >>> wrapper = BatchedHybridGnnRnn(model_config, name="hybrid")
    >>> # inputs: trade_features (B, T, F), adjacency (B, T, T), pnl_history (B, S, E), target_indices (B, N)
    >>> outputs = wrapper(inputs, training=True)  # (B, N)
    """

    def __init__(self, model_config: Dict[str, Any], **kwargs) -> None:
        """
        Initialise the wrapper.

        Parameters
        ----------
        model_config : dict
            Config dict for HybridGnnRnn.
        **kwargs
            Passed to tf.keras.Model.
        """
        super().__init__(**kwargs)
        self.model_config = model_config
        self.hybrid = HybridGnnRnn(model_config=model_config, name=f"{self.name}_hybrid")

    def call(
        self,
        inputs: Dict[str, Union[tf.Tensor, tf.SparseTensor]],
        training: bool = False,
    ) -> tf.Tensor:
        """
        Forward pass with batch dimension handling.

        Parameters
        ----------
        inputs : dict
            - trade_features: (batch, n_trades, n_features) or (n_trades, n_features)
            - adjacency_matrix: (batch, n_trades, n_trades) or (n_trades, n_trades)
            - pnl_history: (batch, n_timesteps, n_elementary)
            - target_indices: (batch, n_targets) or (n_targets,)
            - elementary_indices: optional (batch, n_elementary) or (n_elementary,)
        training : bool
            Whether in training mode.

        Returns
        -------
        tf.Tensor
            Predictions of shape (batch, n_targets).
        """
        trade_features = inputs["trade_features"]
        adjacency_matrix = inputs["adjacency_matrix"]
        pnl_history = inputs["pnl_history"]
        target_indices = inputs["target_indices"]

        # Squeeze batch dimension from static graph inputs if needed
        if len(trade_features.shape) == 3:
            # Take first element (all batch elements have same graph)
            trade_features = trade_features[0]
        if len(adjacency_matrix.shape) == 3:
            adjacency_matrix = adjacency_matrix[0]
        if len(target_indices.shape) == 2:
            target_indices = target_indices[0]

        # Build inputs dict for the underlying model
        hybrid_inputs = {
            "trade_features": trade_features,
            "adjacency_matrix": adjacency_matrix,
            "pnl_history": pnl_history,
            "target_indices": target_indices,
        }

        # Handle elementary_indices if present
        if "elementary_indices" in inputs:
            elem_idx = inputs["elementary_indices"]
            if len(elem_idx.shape) == 2:
                elem_idx = elem_idx[0]
            hybrid_inputs["elementary_indices"] = elem_idx

        return self.hybrid(hybrid_inputs, training=training)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config["model_config"] = self.model_config
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BatchedHybridGnnRnn":
        return cls(**config)


def create_gnn_tf_dataset(
    gnn_inputs: Dict[str, Any],
    targets: Any,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int = None,
) -> tf.data.Dataset:
    """
    Create tf.data.Dataset for GNN-LSTM training with static graph tiling.

    Parameters
    ----------
    gnn_inputs : dict
        - trade_features: (n_trades, n_features)
        - adjacency_matrix: (n_trades, n_trades)
        - pnl_history: (n_samples, n_timesteps, n_elementary)
        - target_indices: (n_targets,)
        - elementary_indices: optional (n_elementary,)
    targets : np.ndarray
        Shape (n_samples, n_targets).
    batch_size : int
        Batch size.
    shuffle : bool
        Whether to shuffle.
    seed : int, optional
        Shuffle seed.

    Returns
    -------
    tf.data.Dataset
        Yields (inputs_dict, targets) where inputs_dict has:
        - trade_features: (batch, n_trades, n_features)
        - adjacency_matrix: (batch, n_trades, n_trades)
        - pnl_history: (batch, n_timesteps, n_elementary)
        - target_indices: (batch, n_targets)
    """
    import numpy as np

    pnl_history = gnn_inputs["pnl_history"]
    n_samples = pnl_history.shape[0]

    trade_features = gnn_inputs["trade_features"]
    adjacency_matrix = gnn_inputs["adjacency_matrix"]
    target_indices = gnn_inputs["target_indices"]

    # Tile static arrays to (n_samples, ...)
    trade_features_tiled = np.tile(trade_features[None, :, :], (n_samples, 1, 1))
    adjacency_tiled = np.tile(adjacency_matrix[None, :, :], (n_samples, 1, 1))
    target_indices_tiled = np.tile(target_indices[None, :], (n_samples, 1))

    input_dict = {
        "trade_features": trade_features_tiled.astype(np.float32),
        "adjacency_matrix": adjacency_tiled.astype(np.float32),
        "pnl_history": pnl_history.astype(np.float32),
        "target_indices": target_indices_tiled.astype(np.int32),
    }

    if "elementary_indices" in gnn_inputs:
        elem_idx = gnn_inputs["elementary_indices"]
        input_dict["elementary_indices"] = np.tile(elem_idx[None, :], (n_samples, 1)).astype(np.int32)

    ds = tf.data.Dataset.from_tensor_slices((input_dict, targets.astype(np.float32)))
    if shuffle:
        ds = ds.shuffle(buffer_size=n_samples, seed=seed)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def train_val_split_gnn(
    gnn_inputs: Dict[str, Any],
    targets: Any,
    val_fraction: float = 0.2,
    seed: int = None,
) -> tuple:
    """
    Split GNN data into train and validation sets.

    Parameters
    ----------
    gnn_inputs : dict
        GNN inputs dict with pnl_history (n_samples, ...).
    targets : np.ndarray
        Shape (n_samples, n_targets).
    val_fraction : float
        Fraction for validation.
    seed : int, optional
        Random seed.

    Returns
    -------
    (train_inputs, train_targets, val_inputs, val_targets)
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    n_samples = gnn_inputs["pnl_history"].shape[0]
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    n_val = int(n_samples * val_fraction)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    def split_dict(d, idx):
        result = {}
        for k, v in d.items():
            if k in ("trade_features", "adjacency_matrix", "target_indices", "elementary_indices"):
                # Static arrays: keep as-is
                result[k] = v
            else:
                # Per-sample arrays
                result[k] = v[idx]
        return result

    train_inputs = split_dict(gnn_inputs, train_idx)
    val_inputs = split_dict(gnn_inputs, val_idx)
    train_targets = targets[train_idx]
    val_targets = targets[val_idx]

    return train_inputs, train_targets, val_inputs, val_targets

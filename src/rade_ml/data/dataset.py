"""
Dataset utilities for rade ML framework.

This module provides:
    - build_td_dataset: thin helper to wrap arrays / dicts into a batched, shuffled, prefetched tf.data.Dataset pipeline.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from typing import Any, Dict, List, Union, Tuple, Optional

from src.rade_ml.core.config import DataPipelineConfig


def _build_static_tensors(
        static_inputs: Dict[str, Any], ensure_float32: bool,
) -> Dict[str, Union["tf.Tensor", "tf.SparseTensor"]]:
    """
    Convert static inputs to TF tensors for the map closure.

    Static inputs have no sample dimension and are shared across all batches. Converting once to tf.constant avoids
    repeated conversion inside the map and keeps them in the graph.

    tf.SparseTensor and tf.Tensor values are passed through untouched — they
    already live in the TF graph and should not be wrapped in tf.constant().

    :param static_inputs: arrays (or TF tensors) with no sample dimension. Injected to every batch via map.
    :param ensure_float32: whether to convert static numpy array to float32 if possible.
    :return:
    """
    results: Dict[str, Any] = {}
    for key, value in static_inputs.items():
        if isinstance(value, (tf.SparseTensor, tf.Tensor)):
            results[key] = value
            continue
        arr = np.asarray(value)
        if ensure_float32 and np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float32)
        results[key] = tf.constant(arr)
    return results


def build_tf_dataset(
        variable_inputs: Union[np.ndarray, Dict[str, np.ndarray]], targets: np.ndarray, config: DataPipelineConfig,
        static_inputs: Optional[Dict[str, Any]] = None,
) -> tf.data.Dataset:
    """
    Build a batched, refetched tf.data.Dataset for Keras training.

    Supports two pattersn:
        1. **Simple**: only variables inputs and targets. Each sample is a row; no static data.
        2. **Static + variables**: variables inputs are per-sample (batched); static_inputs are shared across all
        samples and injected into every batch via a map.

    No explicit tensor conversion is needed - 'from_tensor_slices' accepts and converts lazily. Static inputs
    or converted to tensors once for the map closure.

    :param variable_inputs: per-sample data with first dimension 'n_samples'.
    :param targets: target array
    :param config: data pipeline configuration for preprocessing and building tf datasets.
    :param static_inputs: arrays with no sample dimension. Injected to every both via map.

    Examples:
        Simple:
        train_ds = build_tf_dataset(x_train, y_train, batch_size=32)

        GNN-RNN ds:
            train_ds = build_tf_dataset(
                variable_inputs={'pnl_history': pnl_history},
                targets=targets_pnl,
                static_inputs={
                    'trade_features': trade_features,
                    'adjacency_indices': adj_indices,
                    'adjacency_values': adj_values,
                    'adjacency_dense_shape': adj_dense_shape,
                    'elementary_indices': elementary_idx,
                    'target_indices': target_idx,
                }
            )
    """
    # 1. validate and prepare targets.
    targets = np.asarray(targets)
    n_samples = len(targets)

    def _ensure_dtype(arr: np.ndarray) -> np.ndarray:
        """Cast float array to float32 when required."""
        arr = np.asarray(arr)
        if config.ensure_float32 and np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float32)
        return arr

    # ensure targets are of correct data type.
    targets = _ensure_dtype(targets)

    # 2. validate and prepare variable inputs.
    if isinstance(variable_inputs, dict):
        variable_inputs = {k: _ensure_dtype(v) for k, v in variable_inputs.items()}
        for k, v in variable_inputs.items():
            if len(v) != n_samples:
                raise ValueError(f"variable_inputs[{k}] first dim {len(v)} != targets {n_samples}")
    else:
        variable_inputs = _ensure_dtype(variable_inputs)
        if len(variable_inputs) != n_samples:
            raise ValueError(f"variable_inputs first dim {len(variable_inputs)} != targets {n_samples}")

    # 3. create base dataset.
    ds = tf.data.Dataset.from_tensor_slices((variable_inputs, targets))

    # 4. cache (optional): stores dataset in memory; speeds up later epochs.
    if config.cache:
        ds = ds.cache()

    # 5. shuffle (optional): shuffle before batching.
    if config.shuffle:
        ds = ds.shuffle(
            buffer_size=max(1, min(n_samples, 50000)),
            seed=config.seed,
            reshuffle_each_iteration=True,
        )

    # 6. batch: group samples into mini batches for efficient GPU transfer.
    ds = ds.batch(config.batch_size, drop_remainder=config.drop_remainder)

    # 7. inject static inputs: convert to tf.constant, map adds them to every.
    if static_inputs:
        static_inputs = _build_static_tensors(static_inputs, config.ensure_float32)

        def merge_statics(var_batch: Any, tgt_batch: tf.Tensor) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
            if isinstance(var_batch, dict):
                merged = {**static_inputs, **var_batch}
            else:
                merged = {**static_inputs, 'features': var_batch}
            return merged, tgt_batch

        ds = ds.map(merge_statics)

    # 8. prefetch: overlaps data loading with training, improves GPU utilisation.
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

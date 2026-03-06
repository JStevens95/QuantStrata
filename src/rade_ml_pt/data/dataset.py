"""
Dataset utilities for rade_ml_pt framework.

This module provides:
    - RadeDataset: a torch.utils.data.Dataset that holds per-sample variable inputs and targets,
      plus optional shared static inputs that are broadcast into every sample.
    - build_dataloader: thin helper to wrap arrays / dicts into a batched, shuffled DataLoader.
"""
from __future__ import annotations

import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from typing import Any, Dict, Optional, Tuple, Union

from src.rade_ml_pt.core.config import DataPipelineConfig


class RadeDataset(Dataset):
    """
    PyTorch Dataset wrapping per-sample variable inputs, targets, and optional shared static data.

    Static inputs (e.g. trade features, adjacency components) have no sample dimension and are
    returned identically for every __getitem__ call, merged into the variable input dict.
    """

    def __init__(
        self,
        variable_inputs: Union[np.ndarray, Dict[str, np.ndarray]],
        targets: np.ndarray,
        static_inputs: Optional[Dict[str, Any]] = None,
        ensure_float32: bool = True,
    ):
        """
        Initialise the dataset.

        :param variable_inputs: per-sample data with first dimension n_samples.
        :param targets: target array with first dimension n_samples.
        :param static_inputs: arrays with no sample dimension, injected into every sample.
        :param ensure_float32: cast float arrays to float32 for GPU efficiency.
        """
        # convert targets to numpy and validate
        self.targets = np.asarray(targets)
        self.n_samples = len(self.targets)

        # cast float targets to float32 when requested
        if ensure_float32 and np.issubdtype(self.targets.dtype, np.floating):
            self.targets = self.targets.astype(np.float32)

        # store variable inputs (either a single array or a dict of arrays)
        if isinstance(variable_inputs, dict):
            self.variable_inputs = {}
            for k, v in variable_inputs.items():
                arr = np.asarray(v)
                if len(arr) != self.n_samples:
                    raise ValueError(
                        f"variable_inputs[{k}] first dim {len(arr)} != targets {self.n_samples}"
                    )
                if ensure_float32 and np.issubdtype(arr.dtype, np.floating):
                    arr = arr.astype(np.float32)
                self.variable_inputs[k] = arr
            self._dict_mode = True
        else:
            arr = np.asarray(variable_inputs)
            if len(arr) != self.n_samples:
                raise ValueError(
                    f"variable_inputs first dim {len(arr)} != targets {self.n_samples}"
                )
            if ensure_float32 and np.issubdtype(arr.dtype, np.floating):
                arr = arr.astype(np.float32)
            self.variable_inputs = arr
            self._dict_mode = False

        # pre-convert static inputs to tensors once (shared across all samples)
        self.static_tensors: Optional[Dict[str, torch.Tensor]] = None
        if static_inputs:
            self.static_tensors = {}
            for key, value in static_inputs.items():
                if isinstance(value, torch.Tensor):
                    self.static_tensors[key] = value
                else:
                    arr = np.asarray(value)
                    if ensure_float32 and np.issubdtype(arr.dtype, np.floating):
                        arr = arr.astype(np.float32)
                    self.static_tensors[key] = torch.from_numpy(arr)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[Any, torch.Tensor]:
        """
        Return (inputs, target) for a single sample.

        If static_inputs were provided, the inputs dict merges variable + static data.
        Otherwise, returns just the variable input (tensor or dict of tensors).
        """
        target = torch.from_numpy(self.targets[idx])

        # build per-sample variable tensors
        if self._dict_mode:
            var_sample = {k: torch.from_numpy(v[idx]) for k, v in self.variable_inputs.items()}
        else:
            var_sample = torch.from_numpy(self.variable_inputs[idx])

        # merge static inputs into every sample when present
        if self.static_tensors is not None:
            if isinstance(var_sample, dict):
                merged = {**self.static_tensors, **var_sample}
            else:
                merged = {**self.static_tensors, "features": var_sample}
            return merged, target

        return var_sample, target


def _collate_dict_batch(batch):
    """
    Custom collate function for dict-based samples.

    Stacks per-sample variable tensors along dim 0 while preserving shared static tensors
    (which have the same shape for every sample and should not be stacked).
    """
    inputs_list, targets_list = zip(*batch)
    targets = torch.stack(targets_list)

    # if inputs are dicts, collate each key
    if isinstance(inputs_list[0], dict):
        collated = {}
        for key in inputs_list[0]:
            values = [inp[key] for inp in inputs_list]
            # static tensors have identical shapes across samples; keep as-is
            if all(v.shape == values[0].shape and torch.equal(v, values[0]) for v in values[1:]):
                collated[key] = values[0]
            else:
                collated[key] = torch.stack(values)
        return collated, targets

    # simple tensor inputs
    return torch.stack(inputs_list), targets


def build_dataloader(
    variable_inputs: Union[np.ndarray, Dict[str, np.ndarray]],
    targets: np.ndarray,
    config: DataPipelineConfig,
    static_inputs: Optional[Dict[str, Any]] = None,
) -> DataLoader:
    """
    Build a batched DataLoader for PyTorch training.

    Supports two patterns:
        1. **Simple**: only variable inputs and targets. Each sample is a row; no static data.
        2. **Static + variable**: variable inputs are per-sample (batched); static_inputs are
           shared across all samples and injected into every batch.

    :param variable_inputs: per-sample data with first dimension n_samples.
    :param targets: target array.
    :param config: data pipeline configuration for preprocessing and DataLoader settings.
    :param static_inputs: arrays with no sample dimension, injected into every batch.
    :returns: configured PyTorch DataLoader.

    Examples:
        Simple:
            train_dl = build_dataloader(x_train, y_train, config)

        GNN-RNN:
            train_dl = build_dataloader(
                variable_inputs={'pnl_history': pnl_history},
                targets=targets_pnl,
                config=config,
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
    # build the underlying dataset
    dataset = RadeDataset(
        variable_inputs=variable_inputs,
        targets=targets,
        static_inputs=static_inputs,
        ensure_float32=config.ensure_float32,
    )

    # use custom collate for dict-mode datasets, default for simple tensors
    use_custom_collate = dataset._dict_mode or dataset.static_tensors is not None

    # reproducible shuffle via a seeded generator
    generator = torch.Generator()
    generator.manual_seed(config.seed)

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        drop_last=config.drop_remainder,
        collate_fn=_collate_dict_batch if use_custom_collate else None,
        generator=generator if config.shuffle else None,
        pin_memory=torch.cuda.is_available(),
    )

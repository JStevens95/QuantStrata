"""
Model-independent inference runner for PyTorch.

Loads a trained PyTorch model (from the registry or a direct path), runs the
forward pass on caller-supplied inputs, and returns a structured
``InferenceResult`` with full provenance.

The runner does **not** perform data preparation -- that responsibility stays
with the caller (typically a model-specific inference pipeline).  This keeps
the runner generic and usable with any ``torch.nn.Module``.

Usage::

    runner = InferenceRunner.from_registry(registry, "best")
    result = runner.predict(inputs, sample_ids=target_ids)
"""
from __future__ import annotations

import time
import hashlib
import logging

import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

from src.rade_ml_pt.core.types import InferenceResult

if TYPE_CHECKING:
    from src.rade_ml_pt.registry.entry import RegistryEntry
    from src.rade_ml_pt.registry.store import ModelRegistry

logger = logging.getLogger(__name__)


class InferenceRunner:
    """
    Stateful runner that holds a loaded model and produces ``InferenceResult``.

    Construct via the class methods :meth:`from_registry` or :meth:`from_path`
    rather than calling ``__init__`` directly.

    Parameters
    ----------
    model : nn.Module
        Pre-loaded PyTorch model.
    model_path : str
        Filesystem path from which the model was loaded.
    model_version : str or None
        Registry version string, if the model came from a registry.
    metadata : dict
        Extra provenance info (entry tags, description, etc.).
    device : torch.device, optional
        Device to run inference on. Defaults to model's current device.
    """

    def __init__(
        self,
        model: nn.Module,
        model_path: str,
        model_version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.model_path = model_path
        self.model_version = model_version
        self.metadata = metadata or {}

        # resolve device from model parameters if not explicitly specified
        if device is not None:
            self.device = device
        else:
            params = list(model.parameters())
            self.device = params[0].device if params else torch.device("cpu")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_registry(
        cls,
        registry: "ModelRegistry",
        version_or_tag: str = "latest",
    ) -> "InferenceRunner":
        """
        Load a model from the registry and wrap it in a runner.

        Parameters
        ----------
        registry : ModelRegistry
            Active registry instance.
        version_or_tag : str
            Version string or tag to resolve.
        """
        model, entry = registry.load(version_or_tag)
        return cls(
            model=model,
            model_path=entry.model_dir,
            model_version=entry.version,
            metadata={
                "tags": entry.tags,
                "description": entry.description,
                "best_epoch": entry.best_epoch,
            },
        )

    @classmethod
    def from_path(
        cls,
        model_path: Union[str, Path],
        model_version: Optional[str] = None,
        map_location: Optional[Union[str, torch.device]] = None,
    ) -> "InferenceRunner":
        """
        Load a model directly from a filesystem path.

        Expects the path to a file saved via ``torch.save(model, path)``.

        Parameters
        ----------
        model_path : str or Path
            Path to a ``.pt`` file containing the full pickled model.
        model_version : str, optional
            Version label to embed in inference results.
        map_location : str or torch.device, optional
            Device mapping for ``torch.load`` (e.g. 'cpu').
        """
        loc = map_location or torch.device("cpu")
        model = torch.load(str(model_path), map_location=loc, weights_only=False)
        model.eval()
        return cls(
            model=model,
            model_path=str(model_path),
            model_version=model_version,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        inputs: Any,
        sample_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        hash_inputs: bool = True,
        result_cls: Optional[type] = None,
    ) -> InferenceResult:
        """
        Run the forward pass and return a provenance-enriched result.

        Parameters
        ----------
        inputs : Any
            Model inputs -- dict of tensors, numpy arrays, or single tensor.
        sample_ids : list of str, optional
            Identifiers for each predicted entity.
        metadata : dict, optional
            Additional key-value pairs to attach to the result.
        hash_inputs : bool
            If True, compute a deterministic hash of the inputs for audit.
        result_cls : type, optional
            Subclass of InferenceResult to instantiate.

        Returns
        -------
        InferenceResult or subclass
        """
        input_hash = self._hash_inputs(inputs) if hash_inputs else None

        # ensure the model is in eval mode for deterministic inference
        self.model.eval()

        # convert numpy arrays to tensors if needed, then move to device
        prepared = self._prepare_inputs(inputs)

        t0 = time.perf_counter()
        with torch.no_grad():
            raw_output = self.model(prepared)
        latency = time.perf_counter() - t0

        # convert output to numpy for the result object
        if isinstance(raw_output, torch.Tensor):
            raw_predictions = raw_output.cpu().numpy()
        elif isinstance(raw_output, dict):
            raw_predictions = {
                k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                for k, v in raw_output.items()
            }
        else:
            raw_predictions = raw_output

        # determine sample count from prediction shape
        if isinstance(raw_predictions, np.ndarray):
            n_samples = raw_predictions.shape[0]
        elif isinstance(raw_predictions, dict):
            first = next(iter(raw_predictions.values()))
            n_samples = first.shape[0] if hasattr(first, "shape") else 0
        else:
            n_samples = 0

        merged_meta = {**self.metadata, **(metadata or {})}

        result_type = result_cls or InferenceResult
        result = result_type(
            predictions=raw_predictions,
            n_samples=n_samples,
            sample_ids=sample_ids,
            model_path=self.model_path,
            model_version=self.model_version,
            latency_seconds=latency,
            input_hash=input_hash,
            metadata=merged_meta,
        )

        logger.info(
            f"Inference complete: {n_samples} samples in {latency:.3f}s "
            f"(version={self.model_version})"
        )
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _prepare_inputs(self, inputs: Any) -> Any:
        """Convert numpy arrays to tensors and move to the inference device."""
        if isinstance(inputs, np.ndarray):
            return torch.from_numpy(inputs).to(self.device)
        if isinstance(inputs, torch.Tensor):
            return inputs.to(self.device)
        if isinstance(inputs, dict):
            return {
                k: (
                    torch.from_numpy(v).to(self.device) if isinstance(v, np.ndarray)
                    else v.to(self.device) if isinstance(v, torch.Tensor)
                    else v
                )
                for k, v in inputs.items()
            }
        return inputs

    @staticmethod
    def _hash_inputs(inputs: Any) -> str:
        """
        Compute a deterministic MD5 hash of the input data for audit.

        Handles numpy arrays, dicts of arrays, and torch Tensors.
        """
        hasher = hashlib.md5()

        def _update(arr: Any) -> None:
            if isinstance(arr, torch.Tensor):
                arr = arr.cpu().numpy()
            if isinstance(arr, np.ndarray):
                hasher.update(arr.tobytes())

        if isinstance(inputs, dict):
            for key in sorted(inputs.keys()):
                hasher.update(key.encode())
                _update(inputs[key])
        elif isinstance(inputs, (list, tuple)):
            for item in inputs:
                _update(item)
        else:
            _update(inputs)

        return hasher.hexdigest()

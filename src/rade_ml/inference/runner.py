"""
Model-independent inference runner.

Loads a trained Keras model (from the registry or a direct path), runs the
forward pass on caller-supplied inputs, and returns a structured
``InferenceResult`` with full provenance.

The runner does **not** perform data preparation -- that responsibility stays
with the caller (typically a model-specific inference pipeline).  This keeps
the runner generic and usable with any ``tf.keras.Model``.

Usage::

    runner = InferenceRunner.from_registry(registry, "best")
    result = runner.predict(inputs, sample_ids=target_ids)
"""
from __future__ import annotations

import time
import hashlib
import logging

import numpy as np
import tensorflow as tf

from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

from src.rade_ml.core.types import InferenceResult

if TYPE_CHECKING:
    from src.rade_ml.registry.entry import RegistryEntry
    from src.rade_ml.registry.store import ModelRegistry

logger = logging.getLogger(__name__)


class InferenceRunner:
    """
    Stateful runner that holds a loaded model and produces ``InferenceResult``.

    Construct via the class methods :meth:`from_registry` or :meth:`from_path`
    rather than calling ``__init__`` directly.

    Parameters
    ----------
    model : tf.keras.Model
        Pre-loaded Keras model.
    model_path : str
        Filesystem path from which the model was loaded.
    model_version : str or None
        Registry version string, if the model came from a registry.
    metadata : dict
        Extra provenance info (entry tags, description, etc.).
    """

    def __init__(
        self,
        model: tf.keras.Model,
        model_path: str,
        model_version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.model_path = model_path
        self.model_version = model_version
        self.metadata = metadata or {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_registry(
        cls,
        registry: "ModelRegistry",
        version_or_tag: str = "latest",
        compile_model: bool = False,
    ) -> "InferenceRunner":
        """
        Load a model from the registry and wrap it in a runner.

        Parameters
        ----------
        registry : ModelRegistry
            Active registry instance.
        version_or_tag : str
            Version string or tag to resolve.
        compile_model : bool
            Whether to compile the loaded model.
        """
        model, entry = registry.load(version_or_tag, compile_model=compile_model)
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
        compile_model: bool = False,
    ) -> "InferenceRunner":
        """
        Load a model directly from a filesystem path.

        Parameters
        ----------
        model_path : str or Path
            Path to a ``.keras`` file or SavedModel directory.
        model_version : str, optional
            Version label to embed in inference results.
        compile_model : bool
            Whether to compile the loaded model.
        """
        model = tf.keras.models.load_model(str(model_path), compile=compile_model)
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
            Model inputs -- dict of tensors, numpy arrays, or tf.data.Dataset.
        sample_ids : list of str, optional
            Identifiers for each predicted entity.
        metadata : dict, optional
            Additional key-value pairs to attach to the result.
        hash_inputs : bool
            If True, compute a deterministic hash of the inputs for audit.
        result_cls : type, optional
            Subclass of InferenceResult to instantiate (e.g. DeepHedgingInferenceResult).

        Returns
        -------
        InferenceResult or subclass
        """
        input_hash = self._hash_inputs(inputs) if hash_inputs else None

        t0 = time.perf_counter()
        raw_predictions = self.model.predict(inputs, verbose=0)
        latency = time.perf_counter() - t0

        if isinstance(raw_predictions, np.ndarray):
            n_samples = raw_predictions.shape[0]
        elif isinstance(raw_predictions, dict):
            first = next(iter(raw_predictions.values()))
            n_samples = first.shape[0] if hasattr(first, "shape") else 0
        else:
            n_samples = 0

        merged_meta = {**self.metadata, **(metadata or {})}

        cls = result_cls or InferenceResult
        result = cls(
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

    @staticmethod
    def _hash_inputs(inputs: Any) -> str:
        """
        Compute a deterministic MD5 hash of the input data for audit.

        Handles numpy arrays, dicts of arrays, and tf.Tensors.
        """
        hasher = hashlib.md5()

        def _update(arr: Any) -> None:
            if isinstance(arr, tf.Tensor):
                arr = arr.numpy()
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

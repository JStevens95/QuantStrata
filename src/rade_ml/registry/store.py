"""
Local filesystem model registry.

Stores trained model checkpoints alongside structured metadata so that
every prediction can be traced back to a specific model version, config,
and training run.

Storage layout::

    root_dir/
      {version}/
        model.keras          # Keras SavedModel
        metadata.json        # RegistryEntry serialised
      index.json             # tag -> version mapping for fast lookup

Usage:
    registry = ModelRegistry("./model_store")
    entry = registry.register(model, training_result, tags=["best"])
    model, entry = registry.load("best")
"""
from __future__ import annotations

import json
import hashlib
import logging
import shutil

import tensorflow as tf

from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

from src.rade_ml.registry.entry import RegistryEntry

if TYPE_CHECKING:
    from src.rade_ml.core.types import TrainingResult

logger = logging.getLogger(__name__)

_MODEL_FILENAME = "model.keras"
_META_FILENAME = "metadata.json"
_INDEX_FILENAME = "index.json"


class ModelRegistry:
    """
    Local filesystem registry for trained TensorFlow models.

    Parameters
    ----------
    root_dir : str or Path
        Root directory for all registered model versions.
    """

    def __init__(self, root_dir: Union[str, Path]) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.root_dir / _INDEX_FILENAME
        self._index: Dict[str, str] = self._load_index()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        model: tf.keras.Model,
        training_result: "TrainingResult",
        tags: Optional[List[str]] = None,
        description: str = "",
    ) -> RegistryEntry:
        """
        Save a trained model and its metadata to the registry.

        Parameters
        ----------
        model : tf.keras.Model
            Trained Keras model.
        training_result : TrainingResult
            Output from Trainer.fit().
        tags : list of str, optional
            Labels for easy retrieval (e.g. "best", "prod").
        description : str
            Free-text note about this version.

        Returns
        -------
        RegistryEntry
        """
        tags = tags or []
        version = self._generate_version()
        version_dir = self.root_dir / version
        version_dir.mkdir(parents=True, exist_ok=False)

        model_path = version_dir / _MODEL_FILENAME
        model.save(str(model_path))

        entry = RegistryEntry(
            version=version,
            model_dir=str(version_dir),
            tags=list(tags),
            description=description,
            metrics={
                "best_val_loss": training_result.best_val_loss,
                "best_train_loss": training_result.best_train_loss,
                "final_epoch": training_result.final_epoch,
            },
            config=training_result.config,
            model_summary=training_result.model_summary,
            training_time_seconds=training_result.training_time_seconds,
            best_epoch=training_result.best_epoch,
        )

        entry.to_json(version_dir / _META_FILENAME)

        for t in tags:
            self._index[t] = version
        self._index["latest"] = version
        self._save_index()

        logger.info(f"Registered model version '{version}' with tags {tags}")
        return entry

    def load(
        self,
        version_or_tag: str = "latest",
        compile_model: bool = False,
    ) -> Tuple[tf.keras.Model, RegistryEntry]:
        """
        Load a model and its metadata by version ID or tag.

        Parameters
        ----------
        version_or_tag : str
            Exact version string or a tag name (e.g. "best", "latest").
        compile_model : bool
            If True, compile the model after loading.

        Returns
        -------
        tuple of (tf.keras.Model, RegistryEntry)
        """
        version = self._resolve_version(version_or_tag)
        version_dir = self.root_dir / version

        model = tf.keras.models.load_model(
            str(version_dir / _MODEL_FILENAME), compile=compile_model
        )
        entry = RegistryEntry.from_json(version_dir / _META_FILENAME)

        logger.info(f"Loaded model version '{version}' (resolved from '{version_or_tag}')")
        return model, entry

    def list_versions(
        self,
        tag_filter: Optional[str] = None,
    ) -> List[RegistryEntry]:
        """
        List all registered model versions, optionally filtered by tag.

        Parameters
        ----------
        tag_filter : str, optional
            Only return entries that carry this tag.

        Returns
        -------
        list of RegistryEntry
        """
        entries = []
        for version_dir in sorted(self.root_dir.iterdir()):
            meta_path = version_dir / _META_FILENAME
            if not meta_path.exists():
                continue
            entry = RegistryEntry.from_json(meta_path)
            if tag_filter is None or tag_filter in entry.tags:
                entries.append(entry)
        return entries

    def get_best(
        self,
        metric: str = "best_val_loss",
        mode: str = "min",
    ) -> RegistryEntry:
        """
        Return the registry entry with the best value for a given metric.

        Parameters
        ----------
        metric : str
            Key in RegistryEntry.metrics to optimise.
        mode : str
            "min" or "max".

        Returns
        -------
        RegistryEntry
        """
        entries = self.list_versions()
        if not entries:
            raise ValueError("Registry is empty.")

        def _score(e: RegistryEntry) -> float:
            val = e.metrics.get(metric)
            if val is None:
                return float("inf") if mode == "min" else float("-inf")
            return float(val)

        best = min(entries, key=_score) if mode == "min" else max(entries, key=_score)
        return best

    def tag(self, version: str, tag: str) -> None:
        """Add a tag to an existing version (also updates the fast-lookup index)."""
        version_dir = self.root_dir / version
        meta_path = version_dir / _META_FILENAME
        if not meta_path.exists():
            raise FileNotFoundError(f"Version '{version}' not found in registry.")

        entry = RegistryEntry.from_json(meta_path)
        if tag not in entry.tags:
            entry.tags.append(tag)
            entry.to_json(meta_path)

        self._index[tag] = version
        self._save_index()
        logger.info(f"Tagged version '{version}' with '{tag}'")

    def untag(self, version: str, tag: str) -> None:
        """Remove a tag from a version."""
        version_dir = self.root_dir / version
        meta_path = version_dir / _META_FILENAME
        if not meta_path.exists():
            raise FileNotFoundError(f"Version '{version}' not found in registry.")

        entry = RegistryEntry.from_json(meta_path)
        if tag in entry.tags:
            entry.tags.remove(tag)
            entry.to_json(meta_path)

        if self._index.get(tag) == version:
            del self._index[tag]
            self._save_index()
        logger.info(f"Removed tag '{tag}' from version '{version}'")

    def delete(self, version: str) -> None:
        """Delete a version and its artifacts from the registry."""
        version_dir = self.root_dir / version
        if not version_dir.exists():
            raise FileNotFoundError(f"Version '{version}' not found in registry.")

        tags_to_remove = [t for t, v in self._index.items() if v == version]
        for t in tags_to_remove:
            del self._index[t]
        self._save_index()

        shutil.rmtree(version_dir)
        logger.info(f"Deleted version '{version}' from registry")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_version(self, version_or_tag: str) -> str:
        """Resolve a tag to a version, or return the version directly."""
        if version_or_tag in self._index:
            return self._index[version_or_tag]

        version_dir = self.root_dir / version_or_tag
        if version_dir.exists() and (version_dir / _META_FILENAME).exists():
            return version_or_tag

        raise KeyError(
            f"'{version_or_tag}' is not a known tag or version. "
            f"Available tags: {list(self._index.keys())}"
        )

    def _load_index(self) -> Dict[str, str]:
        """Load the tag -> version index from disk."""
        if self._index_path.exists():
            with open(self._index_path, "r") as f:
                return json.load(f)
        return {}

    def _save_index(self) -> None:
        """Persist the tag -> version index to disk."""
        with open(self._index_path, "w") as f:
            json.dump(self._index, f, indent=2)

    @staticmethod
    def _generate_version() -> str:
        """Generate a unique version string: timestamp + short hash."""
        now = datetime.now()
        ts = now.strftime("%Y%m%d_%H%M%S")
        short_hash = hashlib.md5(now.isoformat().encode()).hexdigest()[:6]
        return f"{ts}_{short_hash}"

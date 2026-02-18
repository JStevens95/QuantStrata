"""
Dataset manifest for data lineage tracking.

Every trained model should be paired with a ``DatasetManifest`` that records
exactly which data was used — enabling full reproducibility, audit trails,
and regulatory compliance.  This is a fundamental requirement for
production-grade ML systems in hedge funds and investment banks.

Key fields:
    - **name / version** — human-readable dataset identifier
    - **split_hash** — deterministic SHA-256 hash of the train/val/test split
    - **feature_columns / target_column** — schema of the data
    - **row_count / feature_count** — shape metadata
    - **normalization** — how features and targets were scaled
    - **pipeline_version** — which pipeline code produced the data
    - **created_at** — ISO timestamp
    - **tags** — freeform metadata (e.g. ``{"asset_class": "fx"}``)

Usage:
    manifest = DatasetManifest.from_arrays(
        name="fx_vanilla_pricing_v3",
        features=train_features,
        targets=train_targets,
        feature_columns=["spot", "strike", "vol", "rate", "ttm", "cp"],
        target_column="price",
        pipeline_version="2.1.0",
    )
    manifest.to_json("artifacts/dataset_manifest.json")

    # Later — reload and verify
    loaded = DatasetManifest.from_json("artifacts/dataset_manifest.json")
    assert loaded.split_hash == manifest.split_hash
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


@dataclass
class DatasetManifest:
    """
    Immutable metadata record describing the dataset used for a model run.

    This manifest is persisted alongside model artefacts so that any
    deployment or audit can trace back to the exact data slice.

    Parameters
    ----------
    name : str
        Human-readable dataset name (e.g. ``"fx_vanilla_pricing_v3"``).
    version : str
        Semantic version or date string for the dataset.
    split_hash : str
        SHA-256 hash of the concatenated train+val+test arrays,
        guaranteeing reproducibility.
    feature_columns : list of str
        Ordered list of feature column names.
    target_column : str
        Name of the target column.
    row_count : int
        Total number of samples (train + val + test).
    feature_count : int
        Number of features per sample.
    train_rows : int
        Number of training samples.
    val_rows : int
        Number of validation samples.
    test_rows : int
        Number of test samples.
    normalization : dict
        Description of normalisation applied
        (e.g. ``{"method": "zscore", "feature_mean": [...], ...}``).
    pipeline_version : str
        Version of the data pipeline code that produced this dataset.
    source_path : str, optional
        Path or URI to the raw data source.
    created_at : str
        ISO-8601 timestamp when the manifest was created.
    tags : dict
        Freeform metadata (asset class, model type, environment, …).
    """

    name: str
    version: str = "1.0.0"
    split_hash: str = ""
    feature_columns: List[str] = field(default_factory=list)
    target_column: str = "target"
    row_count: int = 0
    feature_count: int = 0
    train_rows: int = 0
    val_rows: int = 0
    test_rows: int = 0
    normalization: Dict[str, Any] = field(default_factory=dict)
    pipeline_version: str = "unknown"
    source_path: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    tags: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_arrays(
        cls,
        name: str,
        features: np.ndarray,
        targets: np.ndarray,
        feature_columns: Optional[List[str]] = None,
        target_column: str = "target",
        val_features: Optional[np.ndarray] = None,
        val_targets: Optional[np.ndarray] = None,
        test_features: Optional[np.ndarray] = None,
        test_targets: Optional[np.ndarray] = None,
        normalization: Optional[Dict[str, Any]] = None,
        pipeline_version: str = "unknown",
        source_path: Optional[str] = None,
        version: str = "1.0.0",
        tags: Optional[Dict[str, Any]] = None,
    ) -> "DatasetManifest":
        """
        Build a manifest from numpy arrays.

        Computes a deterministic SHA-256 hash over the concatenated data
        arrays, providing an integrity fingerprint for the dataset.

        Parameters
        ----------
        name : str
            Dataset name.
        features : np.ndarray
            Training features array.
        targets : np.ndarray
            Training targets array.
        feature_columns : list of str, optional
            Feature column names (auto-generated if not provided).
        target_column : str
            Target column name.
        val_features, val_targets : np.ndarray, optional
            Validation arrays.
        test_features, test_targets : np.ndarray, optional
            Test arrays.
        normalization : dict, optional
            Normalisation metadata.
        pipeline_version : str
            Code version of the data pipeline.
        source_path : str, optional
            Path to raw data source.
        version : str
            Dataset version string.
        tags : dict, optional
            Freeform tags.

        Returns
        -------
        DatasetManifest
        """
        # Auto-generate column names if not provided
        n_features = features.shape[1] if features.ndim >= 2 else 1
        if feature_columns is None:
            feature_columns = [f"feature_{i}" for i in range(n_features)]

        # Compute row counts
        train_rows = len(features)
        val_rows = len(val_features) if val_features is not None else 0
        test_rows = len(test_features) if test_features is not None else 0

        # Compute deterministic hash over all data arrays
        hasher = hashlib.sha256()
        for arr in [features, targets, val_features, val_targets, test_features, test_targets]:
            if arr is not None:
                hasher.update(np.ascontiguousarray(arr).tobytes())
        split_hash = hasher.hexdigest()

        return cls(
            name=name,
            version=version,
            split_hash=split_hash,
            feature_columns=feature_columns,
            target_column=target_column,
            row_count=train_rows + val_rows + test_rows,
            feature_count=n_features,
            train_rows=train_rows,
            val_rows=val_rows,
            test_rows=test_rows,
            normalization=normalization or {},
            pipeline_version=pipeline_version,
            source_path=source_path,
            tags=tags or {},
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serialisable dictionary."""
        return asdict(self)

    def to_json(self, path: Union[str, Path]) -> None:
        """Save manifest to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "DatasetManifest":
        """Load manifest from JSON file."""
        with open(path) as f:
            return cls(**json.load(f))

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def verify_hash(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        val_features: Optional[np.ndarray] = None,
        val_targets: Optional[np.ndarray] = None,
        test_features: Optional[np.ndarray] = None,
        test_targets: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Verify that the provided arrays match the recorded split hash.

        Returns
        -------
        bool
            ``True`` if the hash matches, ``False`` otherwise.
        """
        hasher = hashlib.sha256()
        for arr in [features, targets, val_features, val_targets, test_features, test_targets]:
            if arr is not None:
                hasher.update(np.ascontiguousarray(arr).tobytes())
        return hasher.hexdigest() == self.split_hash

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Formatted summary string for logging."""
        lines = [
            "=" * 55,
            "DATASET MANIFEST",
            "=" * 55,
            f"Name            : {self.name}",
            f"Version         : {self.version}",
            f"Pipeline        : {self.pipeline_version}",
            f"Created         : {self.created_at}",
            f"Source          : {self.source_path or 'N/A'}",
            "",
            f"Features        : {self.feature_count}",
            f"Target          : {self.target_column}",
            f"Total rows      : {self.row_count:,}",
            f"  Train         : {self.train_rows:,}",
            f"  Validation    : {self.val_rows:,}",
            f"  Test          : {self.test_rows:,}",
            "",
            f"Split hash      : {self.split_hash[:16]}…",
            f"Normalisation   : {self.normalization.get('method', 'none')}",
        ]
        if self.tags:
            lines.append(f"Tags            : {self.tags}")
        lines.append("=" * 55)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"DatasetManifest(name={self.name!r}, version={self.version!r}, "
            f"rows={self.row_count}, features={self.feature_count})"
        )


__all__ = ["DatasetManifest"]

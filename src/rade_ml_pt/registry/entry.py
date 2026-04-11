"""
Registry entry dataclass representing a single versioned model snapshot.

Each entry captures the model's filesystem location, training provenance,
performance metrics, and user-supplied tags for retrieval.
"""
from __future__ import annotations

import json

from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union

from src.rade_ml_pt.core import json_safe


@dataclass
class RegistryEntry:
    """
    Immutable record for one registered model version.

    Attributes
    ----------
    version : str
        Unique version identifier (timestamp + short hash).
    model_dir : str
        Absolute path to the directory containing model.keras + metadata.json.
    tags : list of str
        User-supplied labels for retrieval (e.g. "best", "gnn-rnn-v1").
    description : str
        Free-text description of the model / experiment.
    metrics : dict
        Key training/evaluation metrics snapshot (best_val_loss, etc.).
    config : dict
        Training configuration used to produce this model.
    model_summary : dict
        Architecture summary (param counts, layer count).
    training_time_seconds : float
        Wall-clock training time.
    best_epoch : int
        Epoch that produced the best validation loss.
    timestamp : str
        ISO-8601 timestamp of when the model was registered.
    """

    version: str = ""
    model_dir: str = ""
    tags: List[str] = field(default_factory=list)
    description: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    config: Optional[Dict[str, Any]] = None
    model_summary: Optional[Dict[str, Any]] = None
    training_time_seconds: float = 0.0
    best_epoch: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serialisable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RegistryEntry":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_json(self, path: Union[str, Path]) -> None:
        """Persist entry metadata to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=json_safe)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "RegistryEntry":
        """Load entry metadata from a JSON file."""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        tag_str = ", ".join(self.tags) if self.tags else "none"
        loss = self.metrics.get("best_val_loss", "N/A")
        return f"RegistryEntry(version={self.version}, tags=[{tag_str}], val_loss={loss})"

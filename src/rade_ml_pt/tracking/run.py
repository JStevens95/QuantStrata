"""
Experiment run record.

Each ``Run`` captures the full context of a single training experiment:
configuration, resulting metrics, associated model version, and any
user-supplied notes or parameters.  Runs are persisted as human-readable
JSON files for audit and comparison.
"""
from __future__ import annotations

import json
import uuid

from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union

_STATUS_RUNNING = "running"
_STATUS_COMPLETED = "completed"
_STATUS_FAILED = "failed"


@dataclass
class Run:
    """
    Immutable record for a single experiment run.

    Attributes
    ----------
    run_id : str
        Unique identifier (auto-generated UUID4 prefix).
    name : str
        Human-readable name for the run.
    status : str
        Current status: "running", "completed", or "failed".
    tags : list of str
        User-supplied labels for filtering.
    params : dict
        Arbitrary hyperparameters / custom key-value pairs logged by the user.
    metrics : dict
        Scalar metrics logged during or after training.
    config : dict or None
        Snapshot of the full training config (serialised TrainingConfig).
    model_version : str or None
        Registry version produced by this run (set after registration).
    start_time : str
        ISO-8601 timestamp when the run started.
    end_time : str or None
        ISO-8601 timestamp when the run ended.
    error : str or None
        Error message if the run failed.
    notes : str
        Free-text notes.
    """

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    status: str = _STATUS_RUNNING
    tags: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    config: Optional[Dict[str, Any]] = None
    model_version: Optional[str] = None
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None
    error: Optional[str] = None
    notes: str = ""

    # ------------------------------------------------------------------
    # Logging helpers (mutate in-place while the run is active)
    # ------------------------------------------------------------------

    def log_config(self, config: Any) -> None:
        """
        Snapshot a training / data config.

        Accepts a dataclass, dict, or anything with a ``to_dict()`` method.
        """
        if hasattr(config, "to_dict"):
            self.config = config.to_dict()
        elif hasattr(config, "__dataclass_fields__"):
            self.config = asdict(config)
        elif isinstance(config, dict):
            self.config = config
        else:
            self.config = {"raw": str(config)}

    def log_params(self, params: Dict[str, Any]) -> None:
        """Merge additional key-value parameters into this run."""
        self.params.update(params)

    def log_metric(self, key: str, value: float) -> None:
        """Log a single named scalar metric."""
        self.metrics[key] = value

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log multiple scalar metrics at once."""
        self.metrics.update(metrics)

    def log_result(self, training_result: Any) -> None:
        """
        Extract metrics from a ``TrainingResult`` dataclass and log them.
        """
        if hasattr(training_result, "best_val_loss") and training_result.best_val_loss is not None:
            self.metrics["best_val_loss"] = training_result.best_val_loss
        if hasattr(training_result, "best_train_loss"):
            self.metrics["best_train_loss"] = training_result.best_train_loss
        if hasattr(training_result, "final_epoch"):
            self.metrics["final_epoch"] = float(training_result.final_epoch)
        if hasattr(training_result, "training_time_seconds"):
            self.metrics["training_time_seconds"] = training_result.training_time_seconds
        if hasattr(training_result, "stopped_early"):
            self.params["stopped_early"] = training_result.stopped_early

    def set_model_version(self, version: str) -> None:
        """Link this run to a registry model version."""
        self.model_version = version

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def end(self, status: str = _STATUS_COMPLETED) -> None:
        """Mark the run as finished."""
        self.status = status
        self.end_time = datetime.now().isoformat()

    def fail(self, error: str) -> None:
        """Mark the run as failed with an error message."""
        self.error = error
        self.end(status=_STATUS_FAILED)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serialisable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Run":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_json(self, path: Union[str, Path]) -> None:
        """Persist run to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "Run":
        """Load run from a JSON file."""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        tag_str = ", ".join(self.tags) if self.tags else "none"
        return (
            f"Run(id={self.run_id}, name={self.name!r}, "
            f"status={self.status}, tags=[{tag_str}])"
        )

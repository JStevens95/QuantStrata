"""
Canonical result and metadata types for rade ML framework.

All result dataclasses live here as the single source of truth. Other modules (training, evaluation, pipelines) import
from this file rather than defining own copies.

Exported types:
    - CheckpointInfo: metadata about a save checkpoint.
    - TrainingResult: output of a training run (history, metadata)
    - EvaluationResult: output of a model evaluation run.
    - InferenceResult: output of a calibrated model inference.
    - TuningResult: output of a hyperparameter tuning run.
"""
from __future__ import annotations

import json
import matplotlib.pyplot as plt

from src.rade_ml_pt.core import json_safe

from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union


@dataclass
class CheckpointInfo:
    """Metadata about a save checkpoint for rade ML framework."""

    path: str
    epoch: int
    train_loss: float
    val_loss: Optional[float]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    is_best: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert a dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CheckpointInfo":
        """Create from dictionary for serialization."""
        return cls(**d)


@dataclass
class TrainingResult:
    """
    Canonical output of the training run.

    This dataclass is the *single* TrainingResult used across the entire rade ML framework.
    """

    history: Dict[str, List[float]] = field(default_factory=dict)
    final_epoch: int = 0
    best_epoch: int = 0
    best_train_loss: float = float("inf")
    best_val_loss: Optional[float] = None
    checkpoints: List[CheckpointInfo] = field(default_factory=list)
    config: Optional[Dict[str, Any]] = None
    training_time_seconds: float = 0.0
    stopped_early: bool = False
    model_summary: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        d = asdict(self)

        # ensure CheckpointInfo objects are converted to dict.
        d["checkpoints"] = [c.to_dict() if isinstance(c, CheckpointInfo) else c for c in self.checkpoints]
        return d

    def to_json(self, path: Union[str, Path]) -> None:
        """Save to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=json_safe)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "TrainingResult":
        """Load from JSON file."""
        with open(path, "r") as f:
            d = json.load(f)
        d["checkpoints"] = [CheckpointInfo.from_dict(c) for c in d.get("checkpoints", [])]
        return cls(**d)

    def plot_history(self, metrics: Optional[List[str]] = None, figsize: tuple = (12, 4)) -> None:
        """Plot training history and validation loss curves."""
        if metrics is None:
            metrics = ['loss']
            if "val_loss" in self.history:
                metrics.append("val_loss")

        n_metrics = len(metrics)
        fig, axs = plt.subplots(1, n_metrics, figsize=figsize)
        if n_metrics == 1:
            axs = [axs]

        for ax, metric in zip(axs, metrics):
            if metric in self.history:
                epochs = range(1, len(self.history[metric]) + 1)
                ax.plot(epochs, self.history[metric], label=metric)

                val_metric = (
                    f"val_{metric}" if not metric.startswith("val_") else metric
                )
                if val_metric in self.history and val_metric != metric:
                    ax.plot(epochs, self.history[val_metric], label=val_metric)

                ax.axvline(
                    self.best_epoch, color="green", linestyle="--", alpha=0.7, label=f"Best epoch ({self.best_epoch})"
                )
                ax.set_xlabel("Epoch")
                ax.set_ylabel(metric)
                ax.set_title(metric.replace("_", " ").title())
                ax.legend()
                ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


@dataclass
class EvaluationResult:
    """
    Canonical output of model evaluation run.

    This dataclass is the *single* EvaluationResult used across the entire rade ML framework.
    """

    metrics: Dict[str, float] = field(default_factory=dict)
    loss: Optional[float] = None
    loss_curves: Optional[Dict[str, List[float]]] = None
    predictions: Optional[Any] = None
    targets: Optional[Any] = None
    residuals: Optional[Any] = None
    dataset_info: Dict[str, Any] = field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def __repr__(self) -> str:
        metrics_str = ", ".join(
            f"{k}={v:.4f}" for k, v in self.metrics.items()
        )
        return f"EvaluationResult({metrics_str})"

    def to_dict(self) -> Dict[str, Any]:
        """Conver to dict (excludes large arrays for serialization)."""
        d = asdict(self)

        # drop potentially large numpy arrays
        for k in ("predictions", "targets", "residuals"):
            d.pop(k, None)
        return d

    def summary(self) -> str:
        """formatted summary string for logging / reporting."""
        lines = [
            "=" * 50, "EVALUATION RESULTS", "=" * 50, f"Timestamp: {self.timestamp}",
            f"Samples: {self.dataset_info.get('samples', 'N/A')}",
        ]
        if self.loss is not None:
            lines.append(f"Loss: {self.loss:.6f}")
        lines += ["", "Metrics:", "-" * 30]
        for name, value in sorted(self.metrics.items()):
            lines.append(f"{name:20s}: {value:12.6f}")
        lines.append("=" * 50)
        return "\n".join(lines)

    def to_json(self, path: Union[str, Path]) -> None:
        """Save to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "EvaluationResult":
        """Load from JSON file."""
        with open(path, "r") as f:
            return cls(**json.load(f))


@dataclass
class InferenceResult:
    """
    Base output of a model inference run.

    Captures raw predictions alongside provenance metadata (which model,
    which checkpoint, which inputs) so that downstream consumers can trace
    every prediction back to its source.

    Model-specific subclasses (e.g. DeepHedgingInferenceResult) extend this
    with domain fields and semantics.

    Fields:
        predictions: model output array, shape depends on model.
        n_samples: number of prediction rows (first dimension of predictions).
        sample_ids: optional identifiers for each predicted entity (trades, customers, etc.).
        model_path: filesystem path to the checkpoint used for inference.
        model_version: semantic version or experiment tag of the model.
        latency_seconds: wall-clock time for the forward pass (excludes data loading).
        input_hash: optional deterministic hash of the input tensor(s) for reproducibility auditing.
        metadata: arbitrary key-value bag for pipeline-specific information.
        timestamp: ISO-8601 timestamp of when inference was executed.
    """

    predictions: Any = None
    n_samples: int = 0
    sample_ids: Optional[List[str]] = None
    model_path: Optional[str] = None
    model_version: Optional[str] = None
    latency_seconds: float = 0.0
    input_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def __repr__(self) -> str:
        n_ids = len(self.sample_ids) if self.sample_ids else 0
        return (
            f"InferenceResult(samples={self.n_samples}, ids={n_ids}, "
            f"latency={self.latency_seconds:.3f}s)"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict (excludes large prediction arrays for serialization)."""
        d = asdict(self)
        d.pop("predictions", None)
        return d

    def to_json(self, path: Union[str, Path]) -> None:
        """Save metadata to JSON file (predictions excluded)."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "InferenceResult":
        """Load metadata from JSON file (predictions will be None)."""
        with open(path, "r") as f:
            return cls(**json.load(f))


@dataclass
class DeepHedgingInferenceResult(InferenceResult):
    """
    Inference result for Deep Hedging models.

    Extends base InferenceResult with domain-specific semantics for derivatives
    hedging: scenario_count (MC paths) and trade_ids (option/trade identifiers).
    """

    @property
    def scenario_count(self) -> int:
        """Number of Monte Carlo scenarios (alias for n_samples)."""
        return self.n_samples

    @property
    def trade_ids(self) -> Optional[List[str]]:
        """Trade/option identifiers (alias for sample_ids)."""
        return self.sample_ids


@dataclass
class TuningResult:
    """
    Aggregated output of a hyperparameter search.

    Attributes
    ----------
    study_name : str
        Name of the Optuna study.
    direction : str
        Optimisation direction ("minimize" or "maximize").
    n_trials : int
        Total number of trials executed.
    n_completed : int
        Number of trials that completed successfully.
    n_pruned : int
        Number of trials pruned early.
    best_trial_number : int
        Index of the best trial.
    best_value : float
        Objective value of the best trial.
    best_params : dict
        Parameters of the best trial.
    all_trials : list of dict
        Per-trial summary (number, value, params, state, duration).
    elapsed_seconds : float
        Total wall-clock time for the tuning run.
    """

    study_name: str = ""
    direction: str = "minimize"
    n_trials: int = 0
    n_completed: int = 0
    n_pruned: int = 0
    best_trial_number: int = 0
    best_value: float = float("inf")
    best_params: Dict[str, Any] = field(default_factory=dict)
    all_trials: List[Dict[str, Any]] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, path: Union[str, Path]) -> None:
        """Persist the tuning result to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "TuningResult":
        """Load a TuningResult from JSON."""
        with open(path, "r") as f:
            return cls(**{k: v for k, v in json.load(f).items() if k in cls.__dataclass_fields__})


__all__ = [
    "TrainingResult", "EvaluationResult", "InferenceResult", "DeepHedgingInferenceResult",
    "TuningResult", "CheckpointInfo"
]

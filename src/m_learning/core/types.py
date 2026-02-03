"""
Data types for the QuantStrata ML framework.

- TrainingConfig: configuration for a training run
- TrainingResult: output of a training run (history, metadata)
- EvaluationResult: output of model evaluation
- CheckpointInfo: metadata about a saved checkpoint
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


@dataclass
class TrainingConfig:
    """
    Configuration for a training run.

    Parameters
    ----------
    epochs : int
        Number of training epochs.
    learning_rate : float
        Optimizer learning rate.
    batch_size : int, optional
        Batch size (if batching is handled by the pipeline).
    checkpoint_dir : str, optional
        Directory to save checkpoints. If None, no checkpointing.
    checkpoint_frequency : int
        Save checkpoint every N epochs (0 = only save best/last).
    save_best_only : bool
        If True, only save the best model (by validation loss).
    early_stopping_patience : int
        Stop if validation loss doesn't improve for N epochs (0 = disabled).
    log_every : int
        Log metrics every N epochs.
    validation_split : float
        Fraction of training data to use for validation (0 = no split).
    loss_fn : callable, optional
        Loss function (y_true, y_pred) -> scalar. If None, uses model default.
    optimizer : str
        Optimizer name (e.g. "adam", "sgd").
    optimizer_kwargs : dict
        Additional optimizer arguments.
    metrics : list of str
        Metric names to compute during training (e.g. ["mae", "mse"]).
    verbose : int
        Verbosity level (0 = silent, 1 = progress, 2 = detailed).
    """

    epochs: int = 100
    learning_rate: float = 0.001
    batch_size: int = 32
    checkpoint_dir: Optional[str] = None
    checkpoint_frequency: int = 0
    save_best_only: bool = True
    early_stopping_patience: int = 0
    log_every: int = 1
    validation_split: float = 0.0
    loss_fn: Optional[Callable[[Any, Any], float]] = None
    optimizer: str = "adam"
    optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    metrics: List[str] = field(default_factory=list)
    verbose: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict (excluding non-serialisable fields)."""
        d = asdict(self)
        d.pop("loss_fn", None)
        return d


@dataclass
class CheckpointInfo:
    """
    Metadata about a saved checkpoint.

    Parameters
    ----------
    path : str
        Path to the checkpoint file or directory.
    epoch : int
        Epoch at which the checkpoint was saved.
    train_loss : float
        Training loss at checkpoint.
    val_loss : float, optional
        Validation loss at checkpoint.
    timestamp : str
        ISO timestamp when saved.
    is_best : bool
        Whether this is the best checkpoint so far.
    """

    path: str
    epoch: int
    train_loss: float
    val_loss: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    is_best: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CheckpointInfo":
        return cls(**d)


@dataclass
class TrainingResult:
    """
    Output of a training run.

    Parameters
    ----------
    history : dict
        Training history: {"loss": [...], "val_loss": [...], ...}.
    final_epoch : int
        Last completed epoch (may be < config.epochs if early stopped).
    best_epoch : int
        Epoch with best validation (or training) loss.
    best_train_loss : float
        Best training loss achieved.
    best_val_loss : float, optional
        Best validation loss achieved.
    checkpoints : list of CheckpointInfo
        List of saved checkpoints.
    config : TrainingConfig
        Configuration used for training.
    training_time_seconds : float
        Total training time in seconds.
    metadata : dict
        Additional metadata (e.g. model name, git hash).
    """

    history: Dict[str, List[float]] = field(default_factory=dict)
    final_epoch: int = 0
    best_epoch: int = 0
    best_train_loss: float = float("inf")
    best_val_loss: Optional[float] = None
    checkpoints: List[CheckpointInfo] = field(default_factory=list)
    config: Optional[TrainingConfig] = None
    training_time_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["checkpoints"] = [c.to_dict() if isinstance(c, CheckpointInfo) else c for c in self.checkpoints]
        if self.config is not None:
            d["config"] = self.config.to_dict()
        return d

    def to_json(self, path: str) -> None:
        """Save to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "TrainingResult":
        """Load from JSON file."""
        with open(path) as f:
            d = json.load(f)
        d["checkpoints"] = [CheckpointInfo.from_dict(c) for c in d.get("checkpoints", [])]
        d["config"] = TrainingConfig(**d["config"]) if d.get("config") else None
        return cls(**d)


@dataclass
class EvaluationResult:
    """
    Output of model evaluation.

    Parameters
    ----------
    loss : float
        Evaluation loss.
    metrics : dict
        Computed metrics (e.g. {"mae": 0.05, "mse": 0.01}).
    loss_curves : dict, optional
        Training/validation loss curves (from TrainingResult.history).
    pricing_error : float, optional
        Pricing error vs. benchmark (e.g. analytic or MC pricer).
    metadata : dict
        Additional metadata (e.g. model name, dataset info).
    """

    loss: float
    metrics: Dict[str, float] = field(default_factory=dict)
    loss_curves: Optional[Dict[str, List[float]]] = None
    pricing_error: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str) -> None:
        """Save to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "EvaluationResult":
        """Load from JSON file."""
        with open(path) as f:
            return cls(**json.load(f))


__all__ = [
    "TrainingConfig",
    "TrainingResult",
    "EvaluationResult",
    "CheckpointInfo",
]

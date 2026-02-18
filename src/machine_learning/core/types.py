"""
Canonical result and metadata types for the QuantStrata ML framework.

All result dataclasses live here as the single source of truth.  Other
modules (training, evaluation, pipelines) import from this file rather
than defining their own copies.

Exported types
--------------
- CheckpointInfo  : metadata about a saved checkpoint
- TrainingResult  : output of a training run (history, metadata)
- EvaluationResult: output of model evaluation
- TuningResult    : output of hyperparameter tuning

Note
----
``TrainingConfig`` is defined in ``core.config`` — *not* here — to keep
framework-specific configuration (optimizer builders, LR schedules, etc.)
separate from plain result containers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# ---------------------------------------------------------------------------
# CheckpointInfo
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# TrainingResult  (merged from core/types + training/trainer definitions)
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:
    """
    Canonical output of a training run.

    This dataclass is the **single** ``TrainingResult`` used across the
    entire framework — both the Keras ``Trainer`` and the generic
    ``TrainingLoop`` return instances of this class.

    Parameters
    ----------
    history : dict
        Training history keyed by metric name, e.g.
        ``{"loss": [...], "val_loss": [...], "mae": [...]}``.
    final_epoch : int
        Last completed epoch (may be < configured epochs if early-stopped).
    best_epoch : int
        Epoch with the best validation (or training) loss.
    best_train_loss : float
        Best training loss achieved.
    best_val_loss : float, optional
        Best validation loss achieved (``None`` when no validation set).
    checkpoints : list of CheckpointInfo
        Saved checkpoints produced during training.
    config : dict, optional
        Serialised training configuration (the ``to_dict()`` output of
        whichever ``TrainingConfig`` was used).
    training_time_seconds : float
        Wall-clock training time in seconds.
    stopped_early : bool
        ``True`` if training was terminated by early-stopping.
    model_summary : dict, optional
        Model architecture summary (name, param counts, layer list).
    metadata : dict
        Arbitrary metadata (model name, git hash, run tags, …).
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
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serialisable dictionary."""
        d = asdict(self)
        # Ensure CheckpointInfo objects are converted to dicts
        d["checkpoints"] = [
            c.to_dict() if isinstance(c, CheckpointInfo) else c
            for c in self.checkpoints
        ]
        return d

    def to_json(self, path: Union[str, Path]) -> None:
        """Save to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "TrainingResult":
        """Load from JSON file."""
        with open(path) as f:
            d = json.load(f)
        d["checkpoints"] = [
            CheckpointInfo.from_dict(c) for c in d.get("checkpoints", [])
        ]
        return cls(**d)

    # -- visualisation -------------------------------------------------------

    def plot_history(
        self,
        metrics: Optional[List[str]] = None,
        figsize: tuple = (12, 4),
    ) -> None:
        """
        Plot training history curves.

        Parameters
        ----------
        metrics : list of str, optional
            Metric keys to plot (default: ``["loss"]`` plus ``val_loss`` if
            present).
        figsize : tuple
            Matplotlib figure size.
        """
        import matplotlib.pyplot as plt

        if metrics is None:
            metrics = ["loss"]
            if "val_loss" in self.history:
                metrics.append("val_loss")

        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
        if n_metrics == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            if metric in self.history:
                epochs = range(1, len(self.history[metric]) + 1)
                ax.plot(epochs, self.history[metric], label=metric)

                # Plot validation counterpart if it exists
                val_metric = (
                    f"val_{metric}" if not metric.startswith("val_") else metric
                )
                if val_metric in self.history and val_metric != metric:
                    ax.plot(epochs, self.history[val_metric], label=val_metric)

                ax.axvline(
                    self.best_epoch,
                    color="green",
                    linestyle="--",
                    alpha=0.7,
                    label=f"Best ({self.best_epoch})",
                )
                ax.set_xlabel("Epoch")
                ax.set_ylabel(metric)
                ax.set_title(metric.replace("_", " ").title())
                ax.legend()
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# EvaluationResult  (merged from core/types + evaluation/evaluator defs)
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    """
    Canonical output of model evaluation.

    This is the **single** ``EvaluationResult`` used by both the Keras
    ``Evaluator`` and the generic ``evaluate_model()`` pipeline function.

    Parameters
    ----------
    metrics : dict
        Computed metrics, e.g. ``{"mae": 0.05, "mse": 0.01, "r2": 0.99}``.
    loss : float, optional
        Standalone evaluation loss (may be ``None`` when the loss is already
        included in *metrics*).
    loss_curves : dict, optional
        Training/validation loss curves copied from ``TrainingResult.history``.
    pricing_error : float, optional
        Pricing error vs. a benchmark (analytic pricer, MC, …).
    predictions : array-like, optional
        Model predictions (excluded from serialisation by default).
    targets : array-like, optional
        Ground-truth targets.
    residuals : array-like, optional
        ``targets − predictions``.
    dataset_info : dict
        Dataset metadata (``n_samples``, feature names, …).
    metadata : dict
        Additional metadata (model name, run tags, …).
    timestamp : str
        ISO timestamp of the evaluation.
    """

    metrics: Dict[str, float] = field(default_factory=dict)
    loss: Optional[float] = None
    loss_curves: Optional[Dict[str, List[float]]] = None
    pricing_error: Optional[float] = None
    predictions: Optional[Any] = None
    targets: Optional[Any] = None
    residuals: Optional[Any] = None
    dataset_info: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )

    def __repr__(self) -> str:
        metrics_str = ", ".join(
            f"{k}={v:.4f}" for k, v in self.metrics.items()
        )
        return f"EvaluationResult({metrics_str})"

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict (excludes large arrays for serialisation)."""
        d = asdict(self)
        # Drop potentially large numpy arrays
        for key in ("predictions", "targets", "residuals"):
            d.pop(key, None)
        return d

    def summary(self) -> str:
        """Formatted summary string for logging / reporting."""
        lines = [
            "=" * 50,
            "EVALUATION RESULTS",
            "=" * 50,
            f"Timestamp : {self.timestamp}",
            f"Samples   : {self.dataset_info.get('n_samples', 'N/A')}",
        ]
        if self.loss is not None:
            lines.append(f"Loss      : {self.loss:.6f}")
        lines += ["", "Metrics:", "-" * 30]
        for name, value in sorted(self.metrics.items()):
            lines.append(f"  {name:20s}: {value:12.6f}")
        if self.pricing_error is not None:
            lines.append(f"  {'pricing_error':20s}: {self.pricing_error:12.6f}")
        lines.append("=" * 50)
        return "\n".join(lines)

    def to_json(self, path: Union[str, Path]) -> None:
        """Save to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "EvaluationResult":
        """Load from JSON file."""
        with open(path) as f:
            return cls(**json.load(f))


@dataclass
class TuningResult:
    """
    Output of hyperparameter tuning.

    Parameters
    ----------
    best_config : dict
        Best hyperparameter configuration found.
    best_score : float
        Best metric value (e.g. validation loss or negative MAE).
    best_checkpoint_path : str, optional
        Path to best model checkpoint if saved.
    trials : list of dict
        Each entry: {"config": dict, "score": float, "metadata": dict}.
    metadata : dict
        Tuning run metadata (e.g. search strategy, n_trials).
    """

    best_config: Dict[str, Any]
    best_score: float
    best_checkpoint_path: Optional[str] = None
    trials: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str) -> None:
        """Save to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "TuningResult":
        """Load from JSON file."""
        with open(path) as f:
            return cls(**json.load(f))


__all__ = [
    "TrainingResult",
    "EvaluationResult",
    "CheckpointInfo",
    "TuningResult",
]

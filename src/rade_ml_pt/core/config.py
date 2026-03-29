"""
Configuration dataclasses for ML model training.

This module provides pure dataclasses for configuring training, model architecture and data processing.
All configurations are serializable to JSON for reproducibility and experiment tracking.

Usage:
    config = TrainingConfig(...)

    # save config
    config.to_json("config.json")

    # load config
    config = TrainingConfig.from_json("config.json")

    # build PyTorch optimizer from config
    optimizer = config.optimizer.build(model.parameters())
"""
from __future__ import annotations

import json
import torch

from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterator, List, Optional, Union


# ---------------------------------------------------------------------------
# YAML / config sanitisation helpers
# ---------------------------------------------------------------------------

_YAML_NONE_STRINGS = frozenset({"None", "none", "NONE"})


def sanitize_yaml_values(obj: Any) -> Any:
    """Recursively fix common ``yaml.safe_load`` artefacts.

    ``yaml.safe_load`` only recognises ``null``, ``Null``, ``NULL``, ``~``
    and empty values as Python ``None``.  The unquoted token ``None`` is
    returned as the **string** ``'None'`` — a frequent source of downstream
    ``TypeError`` when the value is expected to be ``NoneType`` or ``int``.

    This function walks a nested dict/list structure and:

    * Converts the strings ``'None'``, ``'none'``, ``'NONE'`` → ``None``.
    * Converts numeric strings (``'42'``, ``'1e-3'``) → ``int`` / ``float``
      only when the *entire* string is a valid literal.

    Call this on the dict returned by ``yaml.safe_load`` before passing it
    into any ``from_dict`` constructor.
    """
    if isinstance(obj, dict):
        return {k: sanitize_yaml_values(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_yaml_values(v) for v in obj]
    if isinstance(obj, str):
        if obj in _YAML_NONE_STRINGS:
            return None
        try:
            return int(obj)
        except ValueError:
            pass
        try:
            return float(obj)
        except ValueError:
            pass
    return obj


@dataclass
class DataPipelineConfig:
    """
    Configuration for DataLoader building and preprocessing.

    Controls how arrays are assembled into a PyTorch DataLoader,
    focusing on core data processing and pipeline components.
    """
    # input data standardisation
    transform_type: str = "standard"

    # train / validation / test splits
    seq_length: int = 1
    validation_split: float = 0.1
    test_split: float = 0.05

    # DataLoader settings
    batch_size: int = 32
    cache: bool = False
    shuffle: bool = False
    drop_remainder: bool = False

    # reproducibility
    seed: int = 42
    ensure_float32: bool = True

    def to_build_kwargs(self) -> Dict[str, Any]:
        """Return kwargs suitable for build_dataloader(**cfg.to_build_kwargs())."""
        return {
            "transform_type": self.transform_type,
            "seq_length": self.seq_length,
            "validation_split": self.validation_split,
            "test_split": self.test_split,
            "batch_size": self.batch_size,
            "cache": self.cache,
            "shuffle": self.shuffle,
            "drop_remainder": self.drop_remainder,
            "seed": self.seed,
            "ensure_float32": self.ensure_float32,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataPipelineConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_json(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "DataPipelineConfig":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "DataPipelineConfig":
        """Load configuration from YAML file (requires ``pyyaml``)."""
        import yaml
        with open(path, "r") as f:
            return cls.from_dict(sanitize_yaml_values(yaml.safe_load(f)))


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping during training."""

    patience: int = 10
    min_delta: float = 0.0
    monitor: str = "val_loss"
    mode: str = "min"
    restore_best_weights: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EarlyStoppingConfig":
        """Create from dictionary."""
        return cls(**d)


@dataclass
class CheckpointConfig:
    """Configuration for model checkpointing."""

    checkpoint_dir: str = "./checkpoints"
    save_freq: Union[str, int] = "epoch"
    save_best_only: bool = True
    monitor: str = "val_loss"
    mode: str = "min"
    save_weights_only: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CheckpointConfig":
        """Create from dictionary."""
        return cls(**d)


@dataclass
class OptimizerConfig:
    """Configuration for PyTorch optimizer."""

    name: str = "adam"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    momentum: float = 0.9
    beta_1: float = 0.9
    beta_2: float = 0.999
    clipnorm: Optional[float] = None
    clipvalue: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OptimizerConfig":
        """Create from dictionary."""
        return cls(**d)

    def build(self, params: Iterator[torch.nn.Parameter]) -> torch.optim.Optimizer:
        """Build and return a PyTorch optimizer from this configuration.

        :param params: iterable of model parameters (e.g. model.parameters()).
        :returns: configured PyTorch optimizer instance.
        """
        optimizer_name = self.name.lower()

        if optimizer_name == "adam":
            return torch.optim.Adam(
                params,
                lr=self.learning_rate,
                betas=(self.beta_1, self.beta_2),
                weight_decay=self.weight_decay,
            )
        elif optimizer_name == "adamw":
            return torch.optim.AdamW(
                params,
                lr=self.learning_rate,
                betas=(self.beta_1, self.beta_2),
                weight_decay=self.weight_decay,
            )
        elif optimizer_name == "sgd":
            return torch.optim.SGD(
                params,
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        elif optimizer_name == "rmsprop":
            return torch.optim.RMSprop(
                params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.name}")


@dataclass
class LrScheduleConfig:
    """Configuration for learning rate scheduling."""

    schedule: str = "constant"
    initial_lr: float = 1e-3
    decay_rate: float = 0.96
    decay_steps: int = 1000
    warmup_steps: int = 1
    min_lr: float = 1e-5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LrScheduleConfig":
        """Create from dictionary."""
        return cls(**d)

    def build(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: Optional[int] = None,
    ) -> Any:
        """Build and return a PyTorch learning rate scheduler.

        :param optimizer: the optimizer whose LR will be scheduled.
        :param total_steps: total training steps (required for cosine / warmup_cosine).
        :returns: a PyTorch LR scheduler, or None for constant schedule.
        """
        schedule_name = self.schedule.lower()

        if schedule_name == "constant":
            # no scheduler needed; optimizer already has the initial_lr
            return None

        elif schedule_name == "exponential":
            # gamma = decay_rate applied every decay_steps
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.decay_steps,
                gamma=self.decay_rate,
            )

        elif schedule_name == "cosine":
            if total_steps is None:
                raise ValueError("total_steps must be specified for cosine schedule")
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=self.min_lr,
            )

        elif schedule_name == "warmup_cosine":
            if total_steps is None:
                raise ValueError("total_steps must be specified for warmup_cosine schedule")
            from src.rade_ml_pt.training.schedules import WarmupCosineSchedule
            return WarmupCosineSchedule(
                optimizer,
                warmup_steps=self.warmup_steps,
                total_steps=total_steps,
                min_lr=self.min_lr,
            )

        else:
            raise ValueError(f"Unsupported schedule: {self.schedule}")


@dataclass
class ReduceLrConfig:
    """Configuration for reduce learning rate on plateau."""

    initial_lr: float = 1e-3
    min_lr: float = 1e-5
    patience: int = 5
    factor: float = 0.9
    monitor: str = "val_loss"
    mode: str = "min"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReduceLrConfig":
        """Create from dictionary."""
        return cls(**d)


@dataclass
class TrainingConfig:
    """
    Master configuration for model training.

    This is the main configuration class that encompasses all training settings. It can be serialized to JSON for
    experiment tracking and reproducibility.

    Example:
        config = TrainingConfig(
            epochs=100,
            optimizer=OptimizerConfig(name="adam", learning_rate=1e-3),
            early_stopping=EarlyStoppingConfig(patience=10),
        )
    """
    # basic training settings
    epochs: int = 100

    # optimizer and learning rate schedule
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    lr_schedule: Optional[LrScheduleConfig] = None

    # callbacks
    early_stopping: Optional[EarlyStoppingConfig] = field(default_factory=lambda: EarlyStoppingConfig(patience=10))
    checkpoint: Optional[CheckpointConfig] = None
    lr_reduction: Optional[ReduceLrConfig] = None

    # loss and metrics
    loss: str = "mae"
    metrics: List[str] = field(default_factory=lambda: ["mae", "mse"])

    # performance settings
    mixed_precision: bool = False
    compile_model: bool = False  # torch.compile() for graph-mode speedup
    strategy: Optional[str] = None  # "auto" | "ddp" | "cuda" | "mps" | "cpu"

    # logging
    verbose: bool = False
    log_dir: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "epochs": self.epochs,
            "optimizer": self.optimizer.to_dict() if self.optimizer else None,
            "lr_schedule": self.lr_schedule.to_dict() if self.lr_schedule else None,
            "lr_reduction": self.lr_reduction.to_dict() if self.lr_reduction else None,
            "early_stopping": self.early_stopping.to_dict() if self.early_stopping else None,
            "checkpoint": self.checkpoint.to_dict() if self.checkpoint else None,
            "loss": self.loss,
            "metrics": self.metrics,
            "mixed_precision": self.mixed_precision,
            "compile_model": self.compile_model,
            "strategy": self.strategy,
            "log_dir": self.log_dir,
            "verbose": self.verbose,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingConfig":
        """Create from dictionary."""
        return cls(
            epochs=d.get("epochs", 100),
            optimizer=OptimizerConfig.from_dict(d["optimizer"]) if d.get("optimizer") else OptimizerConfig(),
            lr_schedule=LrScheduleConfig.from_dict(d["lr_schedule"]) if d.get("lr_schedule") else None,
            lr_reduction=ReduceLrConfig.from_dict(d["lr_reduction"]) if d.get("lr_reduction") else None,
            early_stopping=EarlyStoppingConfig.from_dict(d["early_stopping"]) if d.get("early_stopping") else EarlyStoppingConfig(),
            checkpoint=CheckpointConfig.from_dict(d["checkpoint"]) if d.get("checkpoint") else None,
            loss=d.get("loss", "mae"),
            metrics=d.get("metrics") or ["mae", "mse"],
            mixed_precision=d.get("mixed_precision", False),
            compile_model=d.get("compile_model", False),
            strategy=d.get("strategy"),
            verbose=d.get("verbose", False),
            log_dir=d.get("log_dir"),
        )

    def to_json(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "TrainingConfig":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "TrainingConfig":
        """Load configuration from YAML file (requires ``pyyaml``)."""
        import yaml
        with open(path, "r") as f:
            return cls.from_dict(sanitize_yaml_values(yaml.safe_load(f)))

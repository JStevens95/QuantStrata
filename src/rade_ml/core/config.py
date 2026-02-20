"""
Configuration dataclasses for ML model training.

This module provides pure dataclasses for configuring training, model architecture and data processing.
All configurations are serializable to JSON for reproducibility and experiment tracking.

Usage:
    config = TrainingConfig(...)

    # save model
    config.to_json("config.json")

    # load model
    config = TrainingConfig.from_json("config.json")

    # build keras object.
    optimizer = config.optimizer.build()
"""
from __future__ import annotations

import json
import tensorflow as tf

from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union

from src.rade_ml.training import WarmupCosineSchedule


@dataclass
class DataPipelineConfig:
    """
    Configuration for tf.data.Dataset building / preprocessing.

    Controls how arrays are assembled into a tf.data.Dataset.
    Focusing on core data processing / pipeline components.
    """
    # input data standardisation.
    transform_type: str = "standard"

    # input data validation / test split
    seq_length: int = 1
    validation_split: float  = 0.1
    test_split: float = 0.05

    # base tf data processing.
    batch_size: int = 32
    cache: bool = False
    shuffle: bool = False
    drop_remainder: bool = False

    # reproducibility.
    seed: int = 42
    ensure_float32: bool = True

    def to_build_kwargs(self) -> Dict[str, Any]:
        """Return kwargs for suitable for build_tf_dataset(**cfg.to_build_kwargs())."""
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
        """Convert a dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataPipelineConfig":
        """Create from dictionary for serialization."""
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


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping during training."""

    patience: int = 10
    min_delta: float = 1e-4
    monitor: str = "val_loss"
    mode: str = "min"
    restore_best_weights: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert a dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EarlyStoppingConfig":
        """Create from dictionary for serialization."""
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
        """Convert a dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CheckpointConfig":
        """Create from dictionary for serialization."""
        return cls(**d)


@dataclass
class OptimizerConfig:
    """Configuration for optimizer."""

    name: str = "adam"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    momentum: float = 0.9
    beta_1: float = 0.0
    beta_2: float = 0.0
    clipnorm: Optional[float] = None
    clipvalue: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert a dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OptimizerConfig":
        """Create from dictionary for serialization."""
        return cls(**d)

    def build(self) -> Any:
        """Build and return a Keras optimizer from this configuration."""

        # define kwargs output.
        kwargs = {}

        # setting clipnorm and clipvalue if present.
        if self.clipnorm is not None:
            kwargs["clipnorm"] = self.clipnorm
        if self.clipvalue is not None:
            kwargs["clipvalue"] = self.clipvalue

        # building keras optimizers
        if self.name.lower() == "adam":
            return tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                beta_1=self.beta_1,
                beta_2=self.beta_2,
                **kwargs
            )
        elif self.name.lower() == "adamw":
            return tf.keras.optimizers.AdamW(
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
                beta_1=self.beta_1,
                beta_2=self.beta_2,
                **kwargs
            )
        elif self.name.lower() == "sgd":
            return tf.keras.optimizers.SGD(
                learning_rate=self.learning_rate,
                momentum=self.momentum,
                **kwargs
            )
        elif self.name.lower() == "rmsprop":
            return tf.keras.optimizers.RMSprop(
                learning_rate=self.learning_rate,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.name}")


@dataclass
class LrScheduleConfig:
    """Configuration for learning rate scheduling."""

    schedule: str = 'constant'
    initial_lr: float = 1e-3
    decay_rate: float = 0.96
    decay_steps: int = 1000
    warmup_steps: int = 1
    min_lr: float = 1e-5

    def to_dict(self) -> Dict[str, Any]:
        """Convert a dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LrScheduleConfig":
        """Create from dictionary for serialization."""
        return cls(**d)

    def build(self, total_steps: Optional[int] = None) -> Any:
        """
        Build and return a Keras learning rate schedule.

        :param total_steps: total training steps (required for cosine / warmup cosine)
        """
        # define learning rate schedules.
        if self.schedule.lower() == "constant":
            return self.initial_lr
        elif self.schedule.lower() == "exponential":
            return tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.initial_lr,
                decay_steps=self.decay_steps,
                decay_rate=self.decay_rate,
            )
        elif self.schedule.lower() == "cosine":
            if total_steps is None:
                raise ValueError("total_steps must be specified for cosine schedule")
            return tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=self.initial_lr,
                decay_steps=total_steps,
                alpha=self.min_lr / self.initial_lr,
            )
        elif self.schedule.lower() == "warmup_cosine":
            if total_steps is None:
                raise ValueError("total_steps must be specified for warmup_cosine schedule")
            return WarmupCosineSchedule(
                initial_lr=self.initial_lr,
                warmup_steps=self.warmup_steps,
                decay_steps=total_steps - self.warmup_steps,
                min_lr=self.min_lr
            )
        else:
            raise ValueError(f"Unsupported schedule: {self.schedule}")


@dataclass
class ReduceLrConfig:
    """Configuration for reduce learning rate."""

    initial_lr: float = 1e-3
    min_lr: float = 1e-5
    patience: int = 5
    factor: float = 0.9
    monitor: str = "val_loss"
    mode: str = "min"

    def to_dict(self) -> Dict[str, Any]:
        """Convert a dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReduceLrConfig":
        """Create from dictionary for serialization."""
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

    # optimizer and learning rate schedule.
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    lr_schedule: Optional[LrScheduleConfig] = None

    # callbacks.
    early_stopping: Optional[EarlyStoppingConfig] = field(default_factory=lambda: EarlyStoppingConfig(patience=10))
    checkpoint: Optional[CheckpointConfig] = None
    lr_reduction: Optional[ReduceLrConfig] = None

    # loss and metrics.
    loss: str = "mae"
    metrics: List[str] = field(default_factory=lambda: ["mae", "mse"])

    # performance settings.
    mixed_precision: bool = False
    xla_compile: bool = False

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
            "xla_compile": self.xla_compile,
            "log_dir": self.log_dir,
            "verbose": self.verbose,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingConfig":
        """Create from dictionary."""
        return cls(
            epochs=d.get("epochs"),
            optimizer=OptimizerConfig.from_dict(d["optimizer"]) if d.get("optimizer") else OptimizerConfig(),
            lr_schedule=LrScheduleConfig.from_dict(d["lr_schedule"]) if d.get("lr_schedule") else None,
            lr_reduction=ReduceLrConfig.from_dict(d["lr_reduction"]) if d.get("lr_reduction") else None,
            early_stopping=EarlyStoppingConfig.from_dict(d["early_stopping"]) if d.get("early_stopping") else None,
            checkpoint=CheckpointConfig.from_dict(d["checkpoint"]) if d.get("checkpoint") else None,
            loss=d.get("loss"),
            metrics=d.get("metrics"),
            mixed_precision=d.get("mixed_precision"),
            xla_compile=d.get("xla_compile"),
            verbose=d.get("verbose"),
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

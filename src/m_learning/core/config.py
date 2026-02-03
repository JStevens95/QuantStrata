"""
Configuration classes for TensorFlow-native ML training.

This module provides dataclasses for configuring training, model architecture,
and data processing. All configurations are serializable to JSON for
reproducibility and experiment tracking.

Usage:
    config = TrainingConfig(
        epochs=100,
        batch_size=256,
        learning_rate=1e-3,
        optimizer="adam",
        early_stopping=EarlyStoppingConfig(patience=10),
    )
    
    # Save config
    config.to_json("config.json")
    
    # Load config
    loaded = TrainingConfig.from_json("config.json")
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class EarlyStoppingConfig:
    """
    Configuration for early stopping during training.
    
    Attributes:
        patience: Number of epochs with no improvement before stopping
        min_delta: Minimum change to qualify as improvement
        monitor: Metric to monitor ('val_loss', 'loss', etc.)
        mode: 'min' for loss metrics, 'max' for accuracy metrics
        restore_best_weights: Whether to restore best weights after stopping
    """
    patience: int = 10
    min_delta: float = 1e-4
    monitor: str = "val_loss"
    mode: str = "min"
    restore_best_weights: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EarlyStoppingConfig":
        return cls(**d)


@dataclass
class CheckpointConfig:
    """
    Configuration for model checkpointing.
    
    Attributes:
        checkpoint_dir: Directory to save checkpoints
        save_freq: 'epoch' or integer (steps between saves)
        save_best_only: Only save when monitored metric improves
        monitor: Metric to monitor for best model
        mode: 'min' or 'max'
        save_weights_only: Save only weights (faster) vs full model
    """
    checkpoint_dir: str = "./checkpoints"
    save_freq: Union[str, int] = "epoch"
    save_best_only: bool = True
    monitor: str = "val_loss"
    mode: str = "min"
    save_weights_only: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CheckpointConfig":
        return cls(**d)


@dataclass
class OptimizerConfig:
    """
    Configuration for optimizer.
    
    Attributes:
        name: Optimizer name ('adam', 'sgd', 'adamw', 'rmsprop')
        learning_rate: Initial learning rate
        weight_decay: L2 regularization weight (for AdamW)
        momentum: Momentum (for SGD)
        beta_1: Adam beta_1 parameter
        beta_2: Adam beta_2 parameter
        clipnorm: Gradient clipping by norm
        clipvalue: Gradient clipping by value
    """
    name: str = "adam"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    momentum: float = 0.9
    beta_1: float = 0.9
    beta_2: float = 0.999
    clipnorm: Optional[float] = None
    clipvalue: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OptimizerConfig":
        return cls(**d)
    
    def build(self) -> "tf.keras.optimizers.Optimizer":
        """Build and return a Keras optimizer from this config."""
        import tensorflow as tf
        
        kwargs = {}
        if self.clipnorm is not None:
            kwargs["clipnorm"] = self.clipnorm
        if self.clipvalue is not None:
            kwargs["clipvalue"] = self.clipvalue
        
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
            raise ValueError(f"Unknown optimizer: {self.name}")


@dataclass
class LRScheduleConfig:
    """
    Configuration for learning rate scheduling.
    
    Attributes:
        schedule: Schedule type ('constant', 'cosine', 'exponential', 'step', 'warmup_cosine')
        initial_lr: Initial learning rate
        decay_rate: Decay rate (for exponential/step)
        decay_steps: Steps between decay (for step schedule)
        warmup_steps: Warmup steps (for warmup schedules)
        min_lr: Minimum learning rate
    """
    schedule: str = "constant"
    initial_lr: float = 1e-3
    decay_rate: float = 0.96
    decay_steps: int = 1000
    warmup_steps: int = 0
    min_lr: float = 1e-6
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LRScheduleConfig":
        return cls(**d)
    
    def build(self, total_steps: Optional[int] = None) -> "tf.keras.optimizers.schedules.LearningRateSchedule":
        """Build and return a Keras learning rate schedule."""
        import tensorflow as tf
        
        if self.schedule == "constant":
            return self.initial_lr
        elif self.schedule == "exponential":
            return tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.initial_lr,
                decay_steps=self.decay_steps,
                decay_rate=self.decay_rate,
            )
        elif self.schedule == "cosine":
            if total_steps is None:
                raise ValueError("total_steps required for cosine schedule")
            return tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=self.initial_lr,
                decay_steps=total_steps,
                alpha=self.min_lr / self.initial_lr,
            )
        elif self.schedule == "warmup_cosine":
            if total_steps is None:
                raise ValueError("total_steps required for warmup_cosine schedule")
            # Custom warmup + cosine decay
            return WarmupCosineSchedule(
                initial_lr=self.initial_lr,
                warmup_steps=self.warmup_steps,
                decay_steps=total_steps - self.warmup_steps,
                min_lr=self.min_lr,
            )
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")


class WarmupCosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule with linear warmup followed by cosine decay."""
    
    def __init__(
        self,
        initial_lr: float,
        warmup_steps: int,
        decay_steps: int,
        min_lr: float = 1e-6,
    ):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.min_lr = min_lr
    
    def __call__(self, step):
        import tensorflow as tf
        step = tf.cast(step, tf.float32)
        
        # Linear warmup
        warmup_lr = self.initial_lr * (step / tf.maximum(self.warmup_steps, 1.0))
        
        # Cosine decay
        progress = (step - self.warmup_steps) / tf.maximum(self.decay_steps, 1.0)
        progress = tf.clip_by_value(progress, 0.0, 1.0)
        cosine_lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1 + tf.cos(3.14159 * progress))
        
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)
    
    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "warmup_steps": self.warmup_steps,
            "decay_steps": self.decay_steps,
            "min_lr": self.min_lr,
        }


@dataclass
class TrainingConfig:
    """
    Master configuration for model training.
    
    This is the main configuration class that encompasses all training
    settings. It can be serialized to JSON for experiment tracking and
    reproducibility.
    
    Attributes:
        epochs: Maximum number of training epochs
        batch_size: Training batch size
        validation_split: Fraction of data to use for validation (if no val set provided)
        shuffle: Whether to shuffle training data each epoch
        seed: Random seed for reproducibility
        
        optimizer: Optimizer configuration
        lr_schedule: Learning rate schedule configuration
        early_stopping: Early stopping configuration (None to disable)
        checkpoint: Checkpoint configuration (None to disable)
        
        loss: Loss function name ('mse', 'mae', 'huber')
        metrics: List of metrics to track
        
        mixed_precision: Whether to use mixed precision (float16)
        xla_compile: Whether to use XLA compilation
        
        verbose: Verbosity level (0=silent, 1=progress bar, 2=one line per epoch)
        log_dir: Directory for TensorBoard logs (None to disable)
    
    Example:
        config = TrainingConfig(
            epochs=100,
            batch_size=256,
            optimizer=OptimizerConfig(name="adam", learning_rate=1e-3),
            early_stopping=EarlyStoppingConfig(patience=10),
        )
    """
    # Basic training settings
    epochs: int = 100
    batch_size: int = 32
    validation_split: float = 0.1
    shuffle: bool = True
    seed: Optional[int] = None
    
    # Optimizer and LR schedule
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    lr_schedule: Optional[LRScheduleConfig] = None
    
    # Callbacks
    early_stopping: Optional[EarlyStoppingConfig] = field(
        default_factory=lambda: EarlyStoppingConfig(patience=10)
    )
    checkpoint: Optional[CheckpointConfig] = None
    
    # Loss and metrics
    loss: str = "mse"
    metrics: List[str] = field(default_factory=lambda: ["mae"])
    
    # Performance settings
    mixed_precision: bool = False
    xla_compile: bool = False
    
    # Logging
    verbose: int = 1
    log_dir: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "validation_split": self.validation_split,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "optimizer": self.optimizer.to_dict() if self.optimizer else None,
            "lr_schedule": self.lr_schedule.to_dict() if self.lr_schedule else None,
            "early_stopping": self.early_stopping.to_dict() if self.early_stopping else None,
            "checkpoint": self.checkpoint.to_dict() if self.checkpoint else None,
            "loss": self.loss,
            "metrics": self.metrics,
            "mixed_precision": self.mixed_precision,
            "xla_compile": self.xla_compile,
            "verbose": self.verbose,
            "log_dir": self.log_dir,
        }
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingConfig":
        """Create from dictionary."""
        return cls(
            epochs=d.get("epochs", 100),
            batch_size=d.get("batch_size", 32),
            validation_split=d.get("validation_split", 0.1),
            shuffle=d.get("shuffle", True),
            seed=d.get("seed"),
            optimizer=OptimizerConfig.from_dict(d["optimizer"]) if d.get("optimizer") else OptimizerConfig(),
            lr_schedule=LRScheduleConfig.from_dict(d["lr_schedule"]) if d.get("lr_schedule") else None,
            early_stopping=EarlyStoppingConfig.from_dict(d["early_stopping"]) if d.get("early_stopping") else None,
            checkpoint=CheckpointConfig.from_dict(d["checkpoint"]) if d.get("checkpoint") else None,
            loss=d.get("loss", "mse"),
            metrics=d.get("metrics", ["mae"]),
            mixed_precision=d.get("mixed_precision", False),
            xla_compile=d.get("xla_compile", False),
            verbose=d.get("verbose", 1),
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


@dataclass
class DataConfig:
    """
    Configuration for data loading and preprocessing.
    
    Attributes:
        feature_columns: List of feature column names
        target_column: Target column name
        normalize_features: Whether to normalize features
        normalize_targets: Whether to normalize targets
        cache: Whether to cache dataset in memory
        prefetch: Number of batches to prefetch
    """
    feature_columns: Optional[List[str]] = None
    target_column: str = "price"
    normalize_features: bool = True
    normalize_targets: bool = True
    cache: bool = True
    prefetch: int = 2  # tf.data.AUTOTUNE equivalent
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataConfig":
        return cls(**d)


@dataclass 
class ModelConfig:
    """
    Configuration for model architecture.
    
    This is a generic config that can be extended for specific model types.
    
    Attributes:
        name: Model name
        hidden_units: List of hidden layer sizes
        activation: Activation function
        dropout_rate: Dropout rate (0 to disable)
        use_batch_norm: Whether to use batch normalization
        kernel_regularizer: L2 regularization weight (0 to disable)
    """
    name: str = "model"
    hidden_units: List[int] = field(default_factory=lambda: [64, 32])
    activation: str = "relu"
    dropout_rate: float = 0.0
    use_batch_norm: bool = False
    kernel_regularizer: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        return cls(**d)
    
    def to_json(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "ModelConfig":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


# Import tensorflow for WarmupCosineSchedule
try:
    import tensorflow as tf
except ImportError:
    tf = None  # Will fail at runtime if TF not installed

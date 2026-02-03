"""
Unit tests for src.machine_learning.core.config module.

Tests TrainingConfig, OptimizerConfig, EarlyStoppingConfig, and related classes.
"""

import pytest
import json
from pathlib import Path

# Skip entire module if TensorFlow is not available
tf = pytest.importorskip("tensorflow")

from src.machine_learning.core.config import (
    EarlyStoppingConfig,
    CheckpointConfig,
    OptimizerConfig,
    LRScheduleConfig,
    TrainingConfig,
    DataConfig,
    ModelConfig,
    WarmupCosineSchedule,
)


# =============================================================================
# EarlyStoppingConfig Tests
# =============================================================================


class TestEarlyStoppingConfig:
    """Tests for EarlyStoppingConfig dataclass."""

    def test_default_values(self):
        """Default values are set correctly."""
        config = EarlyStoppingConfig()
        assert config.patience == 10
        assert config.min_delta == 1e-4
        assert config.monitor == "val_loss"
        assert config.mode == "min"
        assert config.restore_best_weights is True

    def test_custom_values(self):
        """Custom values are accepted."""
        config = EarlyStoppingConfig(
            patience=20,
            min_delta=1e-3,
            monitor="val_mae",
            mode="min",
            restore_best_weights=False,
        )
        assert config.patience == 20
        assert config.min_delta == 1e-3
        assert config.monitor == "val_mae"

    def test_to_dict(self):
        """to_dict returns proper dictionary."""
        config = EarlyStoppingConfig(patience=15)
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["patience"] == 15

    def test_from_dict(self):
        """from_dict reconstructs config."""
        d = {"patience": 25, "min_delta": 0.001, "monitor": "loss", "mode": "min", "restore_best_weights": True}
        config = EarlyStoppingConfig.from_dict(d)
        assert config.patience == 25
        assert config.min_delta == 0.001


# =============================================================================
# CheckpointConfig Tests
# =============================================================================


class TestCheckpointConfig:
    """Tests for CheckpointConfig dataclass."""

    def test_default_values(self):
        """Default values are set correctly."""
        config = CheckpointConfig()
        assert config.checkpoint_dir == "./checkpoints"
        assert config.save_freq == "epoch"
        assert config.save_best_only is True

    def test_to_dict_from_dict_roundtrip(self):
        """Config survives dict roundtrip."""
        config = CheckpointConfig(
            checkpoint_dir="/tmp/ckpt",
            save_freq=100,
            save_best_only=False,
        )
        d = config.to_dict()
        restored = CheckpointConfig.from_dict(d)
        assert restored.checkpoint_dir == "/tmp/ckpt"
        assert restored.save_freq == 100
        assert restored.save_best_only is False


# =============================================================================
# OptimizerConfig Tests
# =============================================================================


class TestOptimizerConfig:
    """Tests for OptimizerConfig dataclass."""

    def test_default_values(self):
        """Default values are correct."""
        config = OptimizerConfig()
        assert config.name == "adam"
        assert config.learning_rate == 1e-3
        assert config.beta_1 == 0.9
        assert config.beta_2 == 0.999

    def test_build_adam(self):
        """build() creates Adam optimizer."""
        config = OptimizerConfig(name="adam", learning_rate=0.01)
        optimizer = config.build()
        assert isinstance(optimizer, tf.keras.optimizers.Adam)
        # Check learning rate
        lr = optimizer.learning_rate
        if hasattr(lr, 'numpy'):
            assert abs(lr.numpy() - 0.01) < 1e-6
        else:
            assert abs(float(lr) - 0.01) < 1e-6

    def test_build_adamw(self):
        """build() creates AdamW optimizer."""
        config = OptimizerConfig(name="adamw", learning_rate=0.001, weight_decay=0.01)
        optimizer = config.build()
        assert isinstance(optimizer, tf.keras.optimizers.AdamW)

    def test_build_sgd(self):
        """build() creates SGD optimizer."""
        config = OptimizerConfig(name="sgd", learning_rate=0.1, momentum=0.9)
        optimizer = config.build()
        assert isinstance(optimizer, tf.keras.optimizers.SGD)

    def test_build_rmsprop(self):
        """build() creates RMSprop optimizer."""
        config = OptimizerConfig(name="rmsprop", learning_rate=0.001)
        optimizer = config.build()
        assert isinstance(optimizer, tf.keras.optimizers.RMSprop)

    def test_build_unknown_raises(self):
        """build() raises for unknown optimizer."""
        config = OptimizerConfig(name="unknown_optimizer")
        with pytest.raises(ValueError, match="Unknown optimizer"):
            config.build()

    def test_build_with_gradient_clipping(self):
        """build() applies gradient clipping."""
        # Note: Keras 3.0 only allows one of clipnorm, clipvalue, or global_clipnorm
        config = OptimizerConfig(name="adam", clipnorm=1.0)
        optimizer = config.build()
        # Clipping should be set
        assert optimizer.clipnorm == 1.0


# =============================================================================
# LRScheduleConfig Tests
# =============================================================================


class TestLRScheduleConfig:
    """Tests for LRScheduleConfig dataclass."""

    def test_default_values(self):
        """Default is constant schedule."""
        config = LRScheduleConfig()
        assert config.schedule == "constant"
        assert config.initial_lr == 1e-3

    def test_build_constant(self):
        """Constant schedule returns float."""
        config = LRScheduleConfig(schedule="constant", initial_lr=0.01)
        lr = config.build()
        assert lr == 0.01

    def test_build_exponential(self):
        """Exponential decay schedule is built correctly."""
        config = LRScheduleConfig(
            schedule="exponential",
            initial_lr=0.01,
            decay_rate=0.96,
            decay_steps=1000,
        )
        schedule = config.build()
        assert isinstance(schedule, tf.keras.optimizers.schedules.ExponentialDecay)
        
        # Test schedule value at step 0
        lr_0 = schedule(0)
        assert abs(float(lr_0) - 0.01) < 1e-6

    def test_build_cosine(self):
        """Cosine decay schedule is built correctly."""
        config = LRScheduleConfig(
            schedule="cosine",
            initial_lr=0.01,
            min_lr=1e-6,
        )
        schedule = config.build(total_steps=10000)
        assert isinstance(schedule, tf.keras.optimizers.schedules.CosineDecay)

    def test_build_cosine_requires_total_steps(self):
        """Cosine schedule raises without total_steps."""
        config = LRScheduleConfig(schedule="cosine")
        with pytest.raises(ValueError, match="total_steps required"):
            config.build()

    def test_build_warmup_cosine(self):
        """Warmup + cosine schedule is built correctly."""
        config = LRScheduleConfig(
            schedule="warmup_cosine",
            initial_lr=0.01,
            warmup_steps=1000,
            min_lr=1e-6,
        )
        schedule = config.build(total_steps=10000)
        assert isinstance(schedule, WarmupCosineSchedule)


# =============================================================================
# WarmupCosineSchedule Tests
# =============================================================================


class TestWarmupCosineSchedule:
    """Tests for WarmupCosineSchedule class."""

    def test_warmup_phase(self):
        """LR increases linearly during warmup."""
        schedule = WarmupCosineSchedule(
            initial_lr=0.01,
            warmup_steps=100,
            decay_steps=900,
            min_lr=1e-6,
        )
        
        # At step 0, LR should be 0
        lr_0 = float(schedule(0))
        assert lr_0 == 0.0
        
        # At step 50 (middle of warmup), LR should be ~0.005
        lr_50 = float(schedule(50))
        assert abs(lr_50 - 0.005) < 1e-4
        
        # At step 100 (end of warmup), LR should be ~0.01
        lr_100 = float(schedule(100))
        assert abs(lr_100 - 0.01) < 1e-4

    def test_cosine_decay_phase(self):
        """LR decays following cosine after warmup."""
        schedule = WarmupCosineSchedule(
            initial_lr=0.01,
            warmup_steps=100,
            decay_steps=900,
            min_lr=0.001,
        )
        
        # At end of training, LR should approach min_lr
        lr_end = float(schedule(1000))
        assert lr_end < 0.002  # Should be close to min_lr

    def test_get_config(self):
        """get_config returns serializable config."""
        schedule = WarmupCosineSchedule(
            initial_lr=0.01,
            warmup_steps=100,
            decay_steps=900,
            min_lr=1e-6,
        )
        config = schedule.get_config()
        assert config["initial_lr"] == 0.01
        assert config["warmup_steps"] == 100


# =============================================================================
# TrainingConfig Tests
# =============================================================================


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_values(self):
        """Default values are reasonable."""
        config = TrainingConfig()
        assert config.epochs == 100
        assert config.batch_size == 32
        assert config.validation_split == 0.1
        assert config.shuffle is True
        assert config.loss == "mse"
        assert "mae" in config.metrics

    def test_early_stopping_default(self):
        """Early stopping is enabled by default."""
        config = TrainingConfig()
        assert config.early_stopping is not None
        assert config.early_stopping.patience == 10

    def test_to_dict(self):
        """to_dict returns nested dict."""
        config = TrainingConfig(
            epochs=50,
            batch_size=128,
            optimizer=OptimizerConfig(name="adam", learning_rate=0.001),
        )
        d = config.to_dict()
        assert d["epochs"] == 50
        assert d["batch_size"] == 128
        assert d["optimizer"]["name"] == "adam"

    def test_from_dict(self):
        """from_dict reconstructs config with nested objects."""
        d = {
            "epochs": 200,
            "batch_size": 64,
            "optimizer": {"name": "sgd", "learning_rate": 0.01, "momentum": 0.9,
                         "weight_decay": 0.0, "beta_1": 0.9, "beta_2": 0.999,
                         "clipnorm": None, "clipvalue": None},
            "early_stopping": {"patience": 5, "min_delta": 1e-4, "monitor": "val_loss",
                              "mode": "min", "restore_best_weights": True},
        }
        config = TrainingConfig.from_dict(d)
        assert config.epochs == 200
        assert config.optimizer.name == "sgd"
        assert config.early_stopping.patience == 5

    def test_to_json_from_json(self, tmp_path):
        """Config can be saved to and loaded from JSON."""
        config = TrainingConfig(
            epochs=75,
            batch_size=256,
            optimizer=OptimizerConfig(name="adamw", learning_rate=0.0005),
            early_stopping=EarlyStoppingConfig(patience=15),
        )
        
        json_path = tmp_path / "config.json"
        config.to_json(json_path)
        
        loaded = TrainingConfig.from_json(json_path)
        assert loaded.epochs == 75
        assert loaded.batch_size == 256
        assert loaded.optimizer.name == "adamw"
        assert loaded.early_stopping.patience == 15

    def test_none_early_stopping(self):
        """Early stopping can be disabled."""
        config = TrainingConfig(early_stopping=None)
        assert config.early_stopping is None

    def test_mixed_precision_flag(self):
        """Mixed precision flag is available."""
        config = TrainingConfig(mixed_precision=True)
        assert config.mixed_precision is True

    def test_xla_compile_flag(self):
        """XLA compilation flag is available."""
        config = TrainingConfig(xla_compile=True)
        assert config.xla_compile is True


# =============================================================================
# DataConfig Tests
# =============================================================================


class TestDataConfig:
    """Tests for DataConfig dataclass."""

    def test_default_values(self):
        """Default values are correct."""
        config = DataConfig()
        assert config.target_column == "price"
        assert config.normalize_features is True
        assert config.normalize_targets is True
        assert config.cache is True
        assert config.prefetch == 2

    def test_to_dict_from_dict(self):
        """Dict roundtrip works."""
        config = DataConfig(
            feature_columns=["spot", "strike", "vol"],
            target_column="price",
            normalize_features=False,
        )
        d = config.to_dict()
        restored = DataConfig.from_dict(d)
        assert restored.feature_columns == ["spot", "strike", "vol"]
        assert restored.normalize_features is False


# =============================================================================
# ModelConfig Tests
# =============================================================================


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_default_values(self):
        """Default values are correct."""
        config = ModelConfig()
        assert config.name == "model"
        assert config.hidden_units == [64, 32]
        assert config.activation == "relu"
        assert config.dropout_rate == 0.0
        assert config.use_batch_norm is False

    def test_custom_architecture(self):
        """Custom architecture can be specified."""
        config = ModelConfig(
            name="deep_net",
            hidden_units=[256, 128, 64, 32],
            activation="swish",
            dropout_rate=0.2,
            use_batch_norm=True,
            kernel_regularizer=0.01,
        )
        assert config.hidden_units == [256, 128, 64, 32]
        assert config.activation == "swish"
        assert config.dropout_rate == 0.2

    def test_to_json_from_json(self, tmp_path):
        """Config survives JSON roundtrip."""
        config = ModelConfig(
            name="test_model",
            hidden_units=[128, 64],
            dropout_rate=0.1,
        )
        
        json_path = tmp_path / "model_config.json"
        config.to_json(json_path)
        
        loaded = ModelConfig.from_json(json_path)
        assert loaded.name == "test_model"
        assert loaded.hidden_units == [128, 64]
        assert loaded.dropout_rate == 0.1

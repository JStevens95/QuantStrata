"""Tests for m_learning.core.types."""

import json
import tempfile
from pathlib import Path

import pytest

from src.m_learning.core.types import (
    TrainingConfig,
    TrainingResult,
    EvaluationResult,
    CheckpointInfo,
    TuningResult,
)


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TrainingConfig()
        assert config.epochs == 100
        assert config.learning_rate == 0.001
        assert config.batch_size == 32
        assert config.optimizer == "adam"
        assert config.verbose == 1

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TrainingConfig(
            epochs=50,
            learning_rate=0.01,
            batch_size=64,
            checkpoint_dir="/tmp/checkpoints",
            early_stopping_patience=10,
        )
        assert config.epochs == 50
        assert config.learning_rate == 0.01
        assert config.batch_size == 64
        assert config.checkpoint_dir == "/tmp/checkpoints"
        assert config.early_stopping_patience == 10

    def test_to_dict(self):
        """Test serialisation to dict."""
        config = TrainingConfig(epochs=50)
        d = config.to_dict()
        assert d["epochs"] == 50
        assert "loss_fn" not in d  # Non-serialisable field excluded


class TestCheckpointInfo:
    """Tests for CheckpointInfo."""

    def test_creation(self):
        """Test checkpoint info creation."""
        info = CheckpointInfo(
            path="/tmp/ckpt.json",
            epoch=10,
            train_loss=0.05,
            val_loss=0.06,
            is_best=True,
        )
        assert info.path == "/tmp/ckpt.json"
        assert info.epoch == 10
        assert info.is_best is True

    def test_serialisation(self):
        """Test round-trip serialisation."""
        info = CheckpointInfo(
            path="/tmp/ckpt.json",
            epoch=10,
            train_loss=0.05,
        )
        d = info.to_dict()
        restored = CheckpointInfo.from_dict(d)
        assert restored.path == info.path
        assert restored.epoch == info.epoch
        assert restored.train_loss == info.train_loss


class TestTrainingResult:
    """Tests for TrainingResult."""

    def test_creation(self):
        """Test result creation."""
        result = TrainingResult(
            history={"loss": [0.5, 0.3, 0.1], "val_loss": [0.6, 0.4, 0.2]},
            final_epoch=3,
            best_epoch=3,
            best_train_loss=0.1,
            best_val_loss=0.2,
        )
        assert result.final_epoch == 3
        assert result.best_train_loss == 0.1
        assert len(result.history["loss"]) == 3

    def test_json_round_trip(self):
        """Test JSON serialisation round-trip."""
        result = TrainingResult(
            history={"loss": [0.5, 0.3, 0.1]},
            final_epoch=3,
            best_epoch=3,
            best_train_loss=0.1,
            config=TrainingConfig(epochs=3),
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            result.to_json(f.name)
            restored = TrainingResult.from_json(f.name)
        assert restored.final_epoch == result.final_epoch
        assert restored.history["loss"] == result.history["loss"]
        assert restored.config.epochs == 3


class TestEvaluationResult:
    """Tests for EvaluationResult."""

    def test_creation(self):
        """Test evaluation result creation."""
        result = EvaluationResult(
            loss=0.05,
            metrics={"mse": 0.001, "mae": 0.02},
            pricing_error=0.01,
        )
        assert result.loss == 0.05
        assert result.metrics["mse"] == 0.001
        assert result.pricing_error == 0.01

    def test_json_round_trip(self):
        """Test JSON serialisation round-trip."""
        result = EvaluationResult(
            loss=0.05,
            metrics={"mse": 0.001},
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            result.to_json(f.name)
            restored = EvaluationResult.from_json(f.name)
        assert restored.loss == result.loss
        assert restored.metrics == result.metrics

    def test_summary(self):
        """Test summary string generation."""
        result = EvaluationResult(
            loss=0.05,
            metrics={"mse": 0.001, "mae": 0.02},
        )
        s = result.summary()
        assert "EVALUATION RESULTS" in s
        assert "0.050000" in s or "0.05" in s
        assert "mse" in s and "mae" in s

    def test_to_dict_excludes_arrays(self):
        """Test that to_dict excludes predictions/targets/residuals for serialisation."""
        result = EvaluationResult(
            loss=0.1,
            metrics={"mse": 0.01},
            predictions=[1.0, 2.0],
            targets=[1.1, 2.1],
        )
        d = result.to_dict()
        assert "predictions" not in d
        assert "targets" not in d
        assert d["loss"] == 0.1
        assert d["metrics"]["mse"] == 0.01


class TestTuningResult:
    """Tests for TuningResult."""

    def test_creation(self):
        """Test tuning result creation."""
        result = TuningResult(
            best_config={"lr": 0.01, "units": 64},
            best_score=0.05,
            trials=[
                {"config": {"lr": 0.001}, "score": 0.1, "metadata": {}},
                {"config": {"lr": 0.01}, "score": 0.05, "metadata": {}},
            ],
            metadata={"method": "grid", "n_trials": 2},
        )
        assert result.best_config["lr"] == 0.01
        assert result.best_score == 0.05
        assert len(result.trials) == 2

    def test_json_round_trip(self):
        """Test JSON serialisation round-trip."""
        result = TuningResult(
            best_config={"lr": 0.01},
            best_score=0.05,
            metadata={"method": "grid"},
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            result.to_json(f.name)
            restored = TuningResult.from_json(f.name)
        assert restored.best_config == result.best_config
        assert restored.best_score == result.best_score
        assert restored.metadata["method"] == "grid"

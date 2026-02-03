"""
Unit tests for src.machine_learning.core.types module.

Tests TrainingConfig, TrainingResult, EvaluationResult, CheckpointInfo, TuningResult.
"""

import pytest
import json
from datetime import datetime
from pathlib import Path

from src.machine_learning.core.types import (
    TrainingConfig,
    TrainingResult,
    EvaluationResult,
    CheckpointInfo,
    TuningResult,
)


# =============================================================================
# TrainingConfig Tests (from types.py)
# =============================================================================


class TestTrainingConfigTypes:
    """Tests for TrainingConfig from types module."""

    def test_default_values(self):
        """Default values are set correctly."""
        config = TrainingConfig()
        assert config.epochs == 100
        assert config.learning_rate == 0.001
        assert config.batch_size == 32
        assert config.optimizer == "adam"
        assert config.verbose == 1

    def test_custom_values(self):
        """Custom values are accepted."""
        config = TrainingConfig(
            epochs=50,
            learning_rate=0.01,
            batch_size=64,
            checkpoint_dir="/tmp/checkpoints",
            early_stopping_patience=5,
        )
        assert config.epochs == 50
        assert config.learning_rate == 0.01
        assert config.checkpoint_dir == "/tmp/checkpoints"

    def test_to_dict_excludes_loss_fn(self):
        """to_dict excludes non-serializable loss_fn."""
        def custom_loss(y_true, y_pred):
            return 0.0
        
        config = TrainingConfig(loss_fn=custom_loss)
        d = config.to_dict()
        assert "loss_fn" not in d

    def test_validation_split_range(self):
        """Validation split accepts valid range."""
        config = TrainingConfig(validation_split=0.2)
        assert config.validation_split == 0.2
        
        config = TrainingConfig(validation_split=0.0)
        assert config.validation_split == 0.0


# =============================================================================
# CheckpointInfo Tests
# =============================================================================


class TestCheckpointInfo:
    """Tests for CheckpointInfo dataclass."""

    def test_default_timestamp(self):
        """Timestamp is auto-generated."""
        info = CheckpointInfo(path="/tmp/ckpt", epoch=5, train_loss=0.1)
        assert info.timestamp is not None
        # Should be a valid ISO format
        datetime.fromisoformat(info.timestamp)

    def test_all_fields(self):
        """All fields can be set."""
        info = CheckpointInfo(
            path="/tmp/ckpt/best.h5",
            epoch=10,
            train_loss=0.05,
            val_loss=0.06,
            timestamp="2024-01-15T10:30:00",
            is_best=True,
        )
        assert info.path == "/tmp/ckpt/best.h5"
        assert info.epoch == 10
        assert info.is_best is True

    def test_to_dict(self):
        """to_dict returns all fields."""
        info = CheckpointInfo(
            path="/tmp/ckpt",
            epoch=5,
            train_loss=0.1,
            val_loss=0.12,
            is_best=False,
        )
        d = info.to_dict()
        assert d["path"] == "/tmp/ckpt"
        assert d["epoch"] == 5
        assert d["train_loss"] == 0.1

    def test_from_dict(self):
        """from_dict reconstructs CheckpointInfo."""
        d = {
            "path": "/tmp/model",
            "epoch": 20,
            "train_loss": 0.02,
            "val_loss": 0.03,
            "timestamp": "2024-01-01T00:00:00",
            "is_best": True,
        }
        info = CheckpointInfo.from_dict(d)
        assert info.epoch == 20
        assert info.is_best is True


# =============================================================================
# TrainingResult Tests
# =============================================================================


class TestTrainingResult:
    """Tests for TrainingResult dataclass."""

    def test_default_values(self):
        """Default values are set."""
        result = TrainingResult()
        assert result.history == {}
        assert result.final_epoch == 0
        assert result.best_epoch == 0
        assert result.best_train_loss == float("inf")
        assert result.checkpoints == []

    def test_with_history(self):
        """History can be populated."""
        result = TrainingResult(
            history={"loss": [1.0, 0.5, 0.2], "val_loss": [1.1, 0.6, 0.25]},
            final_epoch=3,
            best_epoch=3,
            best_train_loss=0.2,
            best_val_loss=0.25,
        )
        assert len(result.history["loss"]) == 3
        assert result.best_val_loss == 0.25

    def test_with_checkpoints(self):
        """Checkpoints can be added."""
        ckpt = CheckpointInfo(path="/tmp/ckpt", epoch=5, train_loss=0.1)
        result = TrainingResult(checkpoints=[ckpt])
        assert len(result.checkpoints) == 1

    def test_to_dict(self):
        """to_dict serializes properly."""
        ckpt = CheckpointInfo(path="/tmp/ckpt", epoch=5, train_loss=0.1)
        config = TrainingConfig(epochs=10)
        result = TrainingResult(
            history={"loss": [0.5, 0.3]},
            final_epoch=2,
            best_epoch=2,
            best_train_loss=0.3,
            checkpoints=[ckpt],
            config=config,
            training_time_seconds=60.5,
        )
        d = result.to_dict()
        assert d["final_epoch"] == 2
        assert d["training_time_seconds"] == 60.5
        assert isinstance(d["checkpoints"][0], dict)

    def test_to_json_from_json(self, tmp_path):
        """JSON roundtrip works."""
        config = TrainingConfig(epochs=50, learning_rate=0.01)
        result = TrainingResult(
            history={"loss": [1.0, 0.5, 0.2], "val_loss": [1.1, 0.6, 0.3]},
            final_epoch=3,
            best_epoch=3,
            best_train_loss=0.2,
            best_val_loss=0.3,
            config=config,
            training_time_seconds=120.0,
            metadata={"model_name": "test"},
        )
        
        json_path = tmp_path / "result.json"
        result.to_json(str(json_path))
        
        loaded = TrainingResult.from_json(str(json_path))
        assert loaded.final_epoch == 3
        assert loaded.best_val_loss == 0.3
        assert loaded.config.epochs == 50


# =============================================================================
# EvaluationResult Tests
# =============================================================================


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_required_loss(self):
        """Loss is required."""
        result = EvaluationResult(loss=0.05)
        assert result.loss == 0.05

    def test_with_metrics(self):
        """Metrics dict can be populated."""
        result = EvaluationResult(
            loss=0.05,
            metrics={"mae": 0.02, "mse": 0.001, "r2": 0.95},
        )
        assert result.metrics["r2"] == 0.95

    def test_with_predictions(self):
        """Predictions and targets can be stored."""
        import numpy as np
        
        preds = np.array([1.0, 2.0, 3.0])
        targets = np.array([1.1, 2.1, 2.9])
        
        result = EvaluationResult(
            loss=0.01,
            predictions=preds,
            targets=targets,
            residuals=targets - preds,
        )
        assert result.predictions is not None
        assert len(result.residuals) == 3

    def test_to_dict_excludes_arrays(self):
        """to_dict excludes large arrays."""
        import numpy as np
        
        result = EvaluationResult(
            loss=0.05,
            metrics={"mae": 0.02},
            predictions=np.zeros(1000),
            targets=np.zeros(1000),
            residuals=np.zeros(1000),
        )
        d = result.to_dict()
        assert "predictions" not in d
        assert "targets" not in d
        assert "residuals" not in d
        assert "loss" in d
        assert "metrics" in d

    def test_summary(self):
        """summary() returns formatted string."""
        result = EvaluationResult(
            loss=0.05,
            metrics={"mae": 0.02, "mse": 0.001},
            pricing_error=0.03,
        )
        summary = result.summary()
        assert "EVALUATION RESULTS" in summary
        assert "Loss: 0.050000" in summary
        assert "mae" in summary
        assert "pricing_error" in summary

    def test_to_json_from_json(self, tmp_path):
        """JSON roundtrip works."""
        result = EvaluationResult(
            loss=0.05,
            metrics={"mae": 0.02, "r2": 0.95},
            loss_curves={"loss": [1.0, 0.5, 0.1], "val_loss": [1.1, 0.6, 0.12]},
            pricing_error=0.01,
            metadata={"dataset": "test"},
        )
        
        json_path = tmp_path / "eval.json"
        result.to_json(str(json_path))
        
        loaded = EvaluationResult.from_json(str(json_path))
        assert loaded.loss == 0.05
        assert loaded.metrics["r2"] == 0.95
        assert loaded.pricing_error == 0.01


# =============================================================================
# TuningResult Tests
# =============================================================================


class TestTuningResult:
    """Tests for TuningResult dataclass."""

    def test_required_fields(self):
        """Best config and score are required."""
        result = TuningResult(
            best_config={"hidden_units": [128, 64], "learning_rate": 0.001},
            best_score=0.95,
        )
        assert result.best_config["hidden_units"] == [128, 64]
        assert result.best_score == 0.95

    def test_with_trials(self):
        """Trials list can be populated."""
        trials = [
            {"config": {"lr": 0.01}, "score": 0.90, "metadata": {}},
            {"config": {"lr": 0.001}, "score": 0.95, "metadata": {}},
            {"config": {"lr": 0.0001}, "score": 0.92, "metadata": {}},
        ]
        result = TuningResult(
            best_config={"lr": 0.001},
            best_score=0.95,
            trials=trials,
        )
        assert len(result.trials) == 3

    def test_with_checkpoint_path(self):
        """Best checkpoint path can be stored."""
        result = TuningResult(
            best_config={"lr": 0.001},
            best_score=0.95,
            best_checkpoint_path="/models/best_model",
        )
        assert result.best_checkpoint_path == "/models/best_model"

    def test_to_dict(self):
        """to_dict returns all fields."""
        result = TuningResult(
            best_config={"lr": 0.001},
            best_score=0.95,
            trials=[{"config": {"lr": 0.01}, "score": 0.90, "metadata": {}}],
            metadata={"n_trials": 10, "strategy": "random"},
        )
        d = result.to_dict()
        assert d["best_score"] == 0.95
        assert len(d["trials"]) == 1
        assert d["metadata"]["strategy"] == "random"

    def test_to_json_from_json(self, tmp_path):
        """JSON roundtrip works."""
        result = TuningResult(
            best_config={"hidden_units": [64, 32], "dropout": 0.1},
            best_score=0.98,
            best_checkpoint_path="/tmp/best",
            trials=[
                {"config": {"hidden_units": [128]}, "score": 0.95, "metadata": {}},
                {"config": {"hidden_units": [64, 32]}, "score": 0.98, "metadata": {}},
            ],
            metadata={"search_strategy": "grid"},
        )
        
        json_path = tmp_path / "tuning.json"
        result.to_json(str(json_path))
        
        loaded = TuningResult.from_json(str(json_path))
        assert loaded.best_score == 0.98
        assert len(loaded.trials) == 2
        assert loaded.metadata["search_strategy"] == "grid"

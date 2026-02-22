"""Unit tests for rade_ml.training.callbacks."""
import json
import pytest
import tensorflow as tf

from src.rade_ml.core.config import TrainingConfig, EarlyStoppingConfig, CheckpointConfig, ReduceLrConfig
from src.rade_ml.training.callbacks import MetricsLogger, get_standard_callbacks


class TestMetricsLogger:
    def test_writes_log_file(self, tmp_path):
        logger = MetricsLogger(log_dir=str(tmp_path), log_file="test_log.json")
        logger.on_train_begin()
        logger.on_epoch_begin(0)
        logger.on_epoch_end(0, logs={"loss": 0.5, "val_loss": 0.4})
        logger.on_train_end()

        log_path = tmp_path / "test_log.json"
        assert log_path.exists()
        data = json.loads(log_path.read_text())
        assert data["total_epochs"] == 1
        assert data["best_val_loss"] == 0.4
        assert "loss" in data["history"]

    def test_tracks_best_epoch(self, tmp_path):
        logger = MetricsLogger(log_dir=str(tmp_path))
        logger.on_train_begin()
        for i, vl in enumerate([0.5, 0.3, 0.4]):
            logger.on_epoch_begin(i)
            logger.on_epoch_end(i, logs={"loss": vl, "val_loss": vl})
        logger.on_train_end()
        assert logger.best_epoch == 2  # 0-indexed epoch 1 is best
        assert logger.best_val_loss == 0.3


class TestGetStandardCallbacks:
    def test_empty_config_returns_empty(self):
        cfg = TrainingConfig()
        cfg.early_stopping = None
        cfg.checkpoint = None
        cfg.lr_reduction = None
        cfg.log_dir = None
        callbacks = get_standard_callbacks(cfg)
        assert callbacks == []

    def test_early_stopping_included(self):
        cfg = TrainingConfig()
        cfg.early_stopping = EarlyStoppingConfig(patience=5)
        cfg.checkpoint = None
        cfg.lr_reduction = None
        cfg.log_dir = None
        callbacks = get_standard_callbacks(cfg)
        types = [type(c).__name__ for c in callbacks]
        assert "EarlyStopping" in types

    def test_all_callbacks_included(self, tmp_path):
        cfg = TrainingConfig(
            early_stopping=EarlyStoppingConfig(patience=3),
            checkpoint=CheckpointConfig(checkpoint_dir=str(tmp_path / "ckpt")),
            lr_reduction=ReduceLrConfig(patience=5),
            log_dir=str(tmp_path / "logs"),
        )
        callbacks = get_standard_callbacks(cfg)
        types = [type(c).__name__ for c in callbacks]
        assert "EarlyStopping" in types
        assert "ModelCheckpoint" in types
        assert "ReduceLROnPlateau" in types
        assert "TensorBoard" in types
        assert "MetricsLogger" in types

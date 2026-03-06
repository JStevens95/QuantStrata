import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.rade_ml_pt.core.config import DataPipelineConfig, TrainingConfig
from src.rade_ml_pt.core.types import TrainingResult
from src.rade_ml_pt.training.trainer import Trainer


def _make_simple_model():
    """Simple 5->1 linear model for testing."""
    return nn.Linear(5, 1)


def _make_datasets():
    """Create train and validation DataLoaders."""
    np.random.seed(0)
    X = np.random.randn(60, 5).astype(np.float32)
    y = np.random.randn(60, 1).astype(np.float32)

    train_ds = DataLoader(
        TensorDataset(torch.from_numpy(X[:40]), torch.from_numpy(y[:40])),
        batch_size=20,
    )
    val_ds = DataLoader(
        TensorDataset(torch.from_numpy(X[40:]), torch.from_numpy(y[40:])),
        batch_size=20,
    )
    return train_ds, val_ds


class TestTrainerCompile:
    def test_auto_compile_on_fit(self):
        model = _make_simple_model()
        config = TrainingConfig(epochs=1, loss="mse")
        trainer = Trainer(model, config)
        assert not trainer._is_compiled

        train_ds, val_ds = _make_datasets()
        trainer.fit(train_ds, val_ds)
        assert trainer._is_compiled

    def test_explicit_compile(self):
        model = _make_simple_model()
        config = TrainingConfig(loss="mae")
        trainer = Trainer(model, config)
        trainer.compile()
        assert trainer._is_compiled


class TestTrainerFit:
    def test_returns_training_result(self):
        model = _make_simple_model()
        config = TrainingConfig(epochs=2, loss="mse")
        config.early_stopping = None
        trainer = Trainer(model, config)

        train_ds, val_ds = _make_datasets()
        result = trainer.fit(train_ds, val_ds)

        assert isinstance(result, TrainingResult)
        assert result.final_epoch == 2
        assert "loss" in result.history
        assert result.training_time_seconds > 0

    def test_early_stopping_detected(self):
        model = _make_simple_model()
        config = TrainingConfig(epochs=1000, loss="mse")
        config.early_stopping.patience = 2
        trainer = Trainer(model, config)

        train_ds, val_ds = _make_datasets()
        result = trainer.fit(train_ds, val_ds)

        assert result.stopped_early or result.final_epoch < 1000

    def test_best_epoch_populated(self):
        model = _make_simple_model()
        config = TrainingConfig(epochs=3, loss="mse")
        config.early_stopping = None
        trainer = Trainer(model, config)

        train_ds, val_ds = _make_datasets()
        result = trainer.fit(train_ds, val_ds)

        assert 1 <= result.best_epoch <= 3


class TestTrainerEvaluate:
    def test_evaluate_returns_dict(self):
        model = _make_simple_model()
        config = TrainingConfig(epochs=1, loss="mse")
        config.early_stopping = None
        trainer = Trainer(model, config)

        train_ds, val_ds = _make_datasets()
        trainer.fit(train_ds, val_ds)

        metrics = trainer.evaluate(val_ds)
        assert isinstance(metrics, dict)
        assert "loss" in metrics

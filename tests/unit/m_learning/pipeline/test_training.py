"""Tests for m_learning.pipeline.training."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.m_learning.core.types import TrainingConfig, TrainingResult
from src.m_learning.pipeline.training import run_training, TrainingLoop


class DummyModel:
    """Simple model for testing the training loop."""

    def __init__(self):
        self.weights = np.array([0.5, 0.5])
        self.learning_rate = 0.1

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Linear prediction: y = X @ w."""
        return inputs @ self.weights

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """MSE loss."""
        return float(np.mean((y_true - y_pred) ** 2))

    def get_parameters(self):
        return {"weights": self.weights.copy()}

    def set_parameters(self, params):
        self.weights = np.array(params["weights"])

    def train_step(self, inputs: np.ndarray, targets: np.ndarray) -> float:
        """Simple gradient descent step."""
        y_pred = self.forward(inputs)
        loss = self.compute_loss(targets, y_pred)
        # Gradient of MSE w.r.t. weights
        grad = -2 * inputs.T @ (targets - y_pred) / len(inputs)
        self.weights -= self.learning_rate * grad
        return loss


class TestDummyModel:
    """Verify dummy model works correctly."""

    def test_forward(self):
        model = DummyModel()
        X = np.array([[1, 2], [3, 4]])
        y = model.forward(X)
        expected = X @ model.weights
        np.testing.assert_array_almost_equal(y, expected)

    def test_train_step(self):
        model = DummyModel()
        X = np.array([[1, 0], [0, 1]])
        y = np.array([1.0, 1.0])
        loss1 = model.train_step(X, y)
        loss2 = model.train_step(X, y)
        assert loss2 <= loss1  # Loss should decrease


class TestTrainingLoop:
    """Tests for TrainingLoop."""

    def test_basic_training(self):
        """Test basic training loop runs without error."""
        model = DummyModel()
        X = np.random.randn(100, 2)
        y = X @ np.array([1.0, 2.0]) + 0.1 * np.random.randn(100)

        config = TrainingConfig(epochs=10, batch_size=20, verbose=0)
        loop = TrainingLoop(model, config)
        result = loop.run(X, y)

        assert isinstance(result, TrainingResult)
        assert result.final_epoch == 10
        assert len(result.history["loss"]) == 10

    def test_training_with_validation(self):
        """Test training with validation split."""
        model = DummyModel()
        X = np.random.randn(100, 2)
        y = X @ np.array([1.0, 2.0])

        config = TrainingConfig(epochs=5, validation_split=0.2, verbose=0)
        result = run_training(model, X, y, config)

        assert len(result.history["val_loss"]) == 5

    def test_training_with_explicit_validation(self):
        """Test training with explicit validation data."""
        model = DummyModel()
        X_train = np.random.randn(80, 2)
        y_train = X_train @ np.array([1.0, 2.0])
        X_val = np.random.randn(20, 2)
        y_val = X_val @ np.array([1.0, 2.0])

        config = TrainingConfig(epochs=5, verbose=0)
        result = run_training(model, X_train, y_train, config, X_val, y_val)

        assert len(result.history["val_loss"]) == 5

    def test_checkpointing(self):
        """Test checkpoint saving."""
        model = DummyModel()
        X = np.random.randn(50, 2)
        y = X @ np.array([1.0, 2.0])

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                epochs=5,
                checkpoint_dir=tmpdir,
                save_best_only=True,
                verbose=0,
            )
            result = run_training(model, X, y, config)

            # Check that checkpoint was saved
            assert len(result.checkpoints) > 0
            ckpt_path = Path(result.checkpoints[0].path)
            assert ckpt_path.exists()

    def test_early_stopping(self):
        """Test early stopping triggers when loss plateaus."""
        model = DummyModel()
        model.weights = np.array([1.0, 2.0])  # Already optimal
        # Generate data that model already fits perfectly
        X = np.random.randn(50, 2)
        y = X @ np.array([1.0, 2.0])

        config = TrainingConfig(
            epochs=100,
            validation_split=0.2,
            early_stopping_patience=3,
            verbose=0,
        )
        result = run_training(model, X, y, config)

        # Model already fits perfectly so validation loss won't improve
        # Early stopping should trigger within a few epochs
        assert result.final_epoch <= 10  # Should stop early


class TestRunTraining:
    """Tests for run_training function."""

    def test_run_training_api(self):
        """Test the run_training API."""
        model = DummyModel()
        X = np.random.randn(100, 2)
        y = X @ np.array([1.0, 2.0])
        config = TrainingConfig(epochs=5, verbose=0)

        result = run_training(model, X, y, config)

        assert isinstance(result, TrainingResult)
        assert result.training_time_seconds > 0

    def test_loss_decreases(self):
        """Test that loss decreases during training."""
        model = DummyModel()
        X = np.random.randn(100, 2)
        y = X @ np.array([1.0, 2.0])
        config = TrainingConfig(epochs=20, batch_size=50, verbose=0)

        result = run_training(model, X, y, config)

        # Loss should generally decrease
        assert result.history["loss"][-1] < result.history["loss"][0]

"""Tests for m_learning.pipeline.inference."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.m_learning.core.types import TrainingConfig
from src.m_learning.pipeline.inference import save_model, load_model, predict


class DummyModel:
    """Simple model for testing inference."""

    def __init__(self, weights: np.ndarray = None):
        self.weights = weights if weights is not None else np.array([0.5, 0.5])

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return inputs @ self.weights

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_true - y_pred) ** 2))

    def get_parameters(self):
        return {"weights": self.weights.copy()}

    def set_parameters(self, params):
        self.weights = np.array(params["weights"])


class TestSaveModel:
    """Tests for save_model function."""

    def test_save_model_basic(self):
        """Test basic model saving."""
        model = DummyModel(np.array([1.0, 2.0]))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_model(model, tmpdir)

            assert Path(path).exists()
            assert (Path(path) / "parameters.json").exists()
            assert (Path(path) / "metadata.json").exists()

    def test_save_model_with_config(self):
        """Test saving model with config."""
        model = DummyModel(np.array([1.0, 2.0]))
        config = TrainingConfig(epochs=50, learning_rate=0.01)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_model(model, tmpdir, config=config)

            assert (Path(tmpdir) / "config.json").exists()

    def test_save_model_with_metadata(self):
        """Test saving model with metadata."""
        model = DummyModel(np.array([1.0, 2.0]))

        with tempfile.TemporaryDirectory() as tmpdir:
            save_model(model, tmpdir, metadata={"version": "1.0"})

            import json
            with open(Path(tmpdir) / "metadata.json") as f:
                meta = json.load(f)
            assert meta["version"] == "1.0"
            assert meta["model_class"] == "DummyModel"


class TestLoadModel:
    """Tests for load_model function."""

    def test_load_model_basic(self):
        """Test basic model loading."""
        original = DummyModel(np.array([1.0, 2.0]))

        with tempfile.TemporaryDirectory() as tmpdir:
            save_model(original, tmpdir)

            def factory():
                return DummyModel()

            loaded = load_model(tmpdir, factory)

            np.testing.assert_array_almost_equal(
                loaded.weights, original.weights
            )

    def test_load_model_not_found(self):
        """Test loading from non-existent directory."""
        with pytest.raises(FileNotFoundError):
            load_model("/nonexistent/path", DummyModel)

    def test_save_load_round_trip(self):
        """Test complete save/load round trip."""
        original = DummyModel(np.array([3.14, 2.71]))
        X = np.array([[1, 0], [0, 1], [1, 1]])

        with tempfile.TemporaryDirectory() as tmpdir:
            save_model(original, tmpdir)
            loaded = load_model(tmpdir, DummyModel)

            # Predictions should match
            np.testing.assert_array_almost_equal(
                original.forward(X),
                loaded.forward(X)
            )


class TestPredict:
    """Tests for predict function."""

    def test_predict_basic(self):
        """Test basic prediction."""
        model = DummyModel(np.array([1.0, 2.0]))
        X = np.array([[1, 0], [0, 1], [1, 1]])

        predictions = predict(model, X)

        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(predictions, expected)

    def test_predict_with_batching(self):
        """Test prediction with batching."""
        model = DummyModel(np.array([1.0, 2.0]))
        X = np.random.randn(100, 2)

        predictions_full = predict(model, X)
        predictions_batched = predict(model, X, batch_size=20)

        np.testing.assert_array_almost_equal(
            predictions_full, predictions_batched
        )

    def test_predict_single_sample(self):
        """Test prediction on single sample."""
        model = DummyModel(np.array([1.0, 2.0]))
        X = np.array([[1, 2]])

        predictions = predict(model, X)

        assert predictions.shape == (1,)
        assert abs(predictions[0] - 5.0) < 1e-8


class TestEndToEndInference:
    """End-to-end inference tests."""

    def test_train_save_load_predict(self):
        """Test full workflow: train -> save -> load -> predict."""
        # "Train" a model (just set weights)
        model = DummyModel(np.array([1.5, 2.5]))
        X_train = np.random.randn(50, 2)
        y_train = X_train @ model.weights

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            save_model(model, tmpdir, metadata={"trained": True})

            # Load
            loaded = load_model(tmpdir, DummyModel)

            # Predict
            X_test = np.random.randn(10, 2)
            predictions = predict(loaded, X_test)
            expected = X_test @ model.weights

            np.testing.assert_array_almost_equal(predictions, expected)

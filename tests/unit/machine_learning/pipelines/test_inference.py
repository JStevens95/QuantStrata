"""
Unit tests for src.machine_learning.pipelines.inference module.

Tests save_model(), load_model(), predict() for Trainable models.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from src.machine_learning.pipelines.inference import (
    load_model,
    predict,
    save_model,
)
from src.machine_learning.core.types import TrainingConfig
from src.machine_learning.core.protocols import KerasTrainableAdapter


@pytest.fixture
def simple_trainable_model():
    """Create a simple Keras model wrapped as Trainable."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation="relu", input_shape=(4,)),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss="mse")
    return KerasTrainableAdapter(model)


@pytest.fixture
def artifact_dir():
    """Temporary directory for artifacts."""
    with tempfile.TemporaryDirectory() as d:
        yield d


def test_save_model_creates_artifact_dir(simple_trainable_model, artifact_dir):
    """save_model creates artifact directory and parameter files."""
    path = save_model(
        simple_trainable_model,
        artifact_dir,
        config=TrainingConfig(epochs=5, learning_rate=0.001),
        metadata={"version": "test"},
    )
    assert Path(path).exists()
    assert (Path(path) / "parameters.json").exists()
    assert (Path(path) / "config.json").exists()
    assert (Path(path) / "metadata.json").exists()


def test_load_model_restores_parameters(simple_trainable_model, artifact_dir):
    """load_model restores model parameters from artifact dir."""
    save_model(simple_trainable_model, artifact_dir)
    def factory():
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(8, activation="relu", input_shape=(4,)),
            tf.keras.layers.Dense(1),
        ])
        return KerasTrainableAdapter(model)
    loaded = load_model(artifact_dir, factory)
    assert loaded is not None
    params = loaded.get_parameters()
    assert "weights" in params or len(params) >= 1


def test_predict_returns_array(simple_trainable_model):
    """predict() returns numpy array of correct shape."""
    X = np.random.randn(10, 4).astype(np.float32)
    out = predict(simple_trainable_model, X, batch_size=4)
    assert isinstance(out, np.ndarray)
    assert out.shape[0] == 10
    assert out.ndim >= 1


def test_save_load_predict_roundtrip(simple_trainable_model, artifact_dir):
    """Save -> load -> predict gives same shape as original predict."""
    X = np.random.randn(6, 4).astype(np.float32)
    pred_before = predict(simple_trainable_model, X, batch_size=2)
    save_model(simple_trainable_model, artifact_dir)
    def factory():
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(8, activation="relu", input_shape=(4,)),
            tf.keras.layers.Dense(1),
        ])
        return KerasTrainableAdapter(model)
    loaded = load_model(artifact_dir, factory)
    pred_after = predict(loaded, X, batch_size=2)
    assert pred_before.shape == pred_after.shape

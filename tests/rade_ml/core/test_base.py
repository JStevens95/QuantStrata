"""Unit tests for rade_ml.core.base -- BaseModel."""
import pytest
import tensorflow as tf

from src.rade_ml.core.base import BaseModel


class _DummyModel(BaseModel):
    """Minimal concrete subclass for testing."""

    def __init__(self, units=4, **kwargs):
        super().__init__(name="dummy", **kwargs)
        self.dense = tf.keras.layers.Dense(units)

    def call(self, inputs, training=False):
        return self.dense(inputs["features"])


class TestBaseModelInit:
    def test_metadata_has_required_keys(self):
        model = _DummyModel()
        meta = model.metadata
        assert "model_name" in meta
        assert "model_class" in meta
        assert "framework" in meta
        assert meta["model_name"] == "dummy"
        assert meta["model_class"] == "_DummyModel"
        assert meta["framework"] == "tensorflow"

    def test_custom_metadata_merged(self):
        model = _DummyModel(metadata={"experiment": "test-001"})
        assert model.metadata["experiment"] == "test-001"

    def test_update_metadata(self):
        model = _DummyModel()
        model.update_metadata(version="v1")
        assert model.metadata["version"] == "v1"

    def test_metadata_returns_copy(self):
        model = _DummyModel()
        meta = model.metadata
        meta["injected"] = True
        assert "injected" not in model.metadata


class TestBaseModelForward:
    def test_call_produces_output(self):
        model = _DummyModel(units=3)
        x = {"features": tf.constant([[1.0, 2.0]])}
        out = model(x, training=False)
        assert out.shape == (1, 3)

    def test_call_training_flag(self):
        model = _DummyModel(units=2)
        x = {"features": tf.constant([[1.0, 2.0]])}
        out_train = model(x, training=True)
        out_infer = model(x, training=False)
        assert out_train.shape == out_infer.shape


class TestBaseModelSerialisation:
    def test_get_config_contains_metadata(self):
        model = _DummyModel()
        config = model.get_config()
        assert "metadata" in config

    def test_summary_dict(self):
        model = _DummyModel(units=4)
        model({"features": tf.constant([[1.0, 2.0]])})
        summary = model.summary_dict()
        assert "trainable_params" in summary
        assert "name" in summary
        assert summary["class"] == "_DummyModel"

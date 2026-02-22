"""Unit tests for rade_ml.core.base -- BaseModel class."""
import pytest
import tensorflow as tf

from src.rade_ml.core.base import BaseModel


class ConcreteBaseModel(BaseModel):
    """Concrete subclass for testing BaseModel (implements abstract call())."""

    def __init__(self, name: str = "concrete_base", **kwargs):
        super().__init__(name=name, **kwargs)
        self.dense = tf.keras.layers.Dense(4)

    def call(self, inputs, training: bool = False):
        return self.dense(inputs)


class TestBaseModel:
    def test_metadata_defaults(self):
        model = ConcreteBaseModel()
        meta = model.metadata
        assert "model_name" in meta
        assert meta["model_name"] == "concrete_base"
        assert "model_class" in meta
        assert meta["model_class"] == "ConcreteBaseModel"
        assert "created_at" in meta
        assert "framework" in meta
        assert meta["framework"] == "tensorflow"
        assert "framework_version" in meta

    def test_metadata_custom_name(self):
        model = ConcreteBaseModel(name="my_model")
        assert model.metadata["model_name"] == "my_model"

    def test_metadata_from_kwargs(self):
        model = ConcreteBaseModel(metadata={"custom_key": "custom_value"})
        assert model.metadata["custom_key"] == "custom_value"

    def test_update_metadata(self):
        model = ConcreteBaseModel()
        model.update_metadata(extra_field=42)
        assert model.metadata["extra_field"] == 42

    def test_metadata_is_copy(self):
        model = ConcreteBaseModel()
        meta = model.metadata
        meta["mutated"] = True
        assert "mutated" not in model._model_metadata

    def test_get_config_includes_metadata(self):
        model = ConcreteBaseModel(name="config_test")
        config = model.get_config()
        assert "metadata" in config
        assert config["metadata"]["model_name"] == "config_test"

    def test_from_config_restores_metadata(self):
        # Use minimal config to test metadata roundtrip (avoid full keras config)
        config = {
            "name": "original",
            "metadata": {
                "model_name": "original",
                "model_class": "ConcreteBaseModel",
                "tag": "v1",
                "framework": "tensorflow",
            },
        }
        restored = ConcreteBaseModel.from_config(config.copy())
        assert restored.metadata["model_name"] == "original"
        assert restored.metadata["tag"] == "v1"

    def test_summary_dict_structure(self):
        model = ConcreteBaseModel(name="summary_test")
        # Build model by running a forward pass
        _ = model(tf.constant([[1.0, 2.0, 3.0]]))
        summary = model.summary_dict()

        assert "name" in summary
        assert summary["name"] == "summary_test"
        assert "class" in summary
        assert summary["class"] == "ConcreteBaseModel"
        assert "trainable_params" in summary
        assert "non_trainable_params" in summary
        assert "layers" in summary
        assert isinstance(summary["layers"], list)
        assert "metadata" in summary

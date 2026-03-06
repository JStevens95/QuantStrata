"""Unit tests for rade_ml_pt.core.base -- BaseModel class (PyTorch)."""
import pytest
import torch

from src.rade_ml_pt.core.base import BaseModel


class ConcreteBaseModel(BaseModel):
    """Concrete subclass for testing BaseModel (implements abstract forward())."""

    def __init__(self, name: str = "concrete_base", **kwargs):
        super().__init__(name=name, **kwargs)
        # simple linear layer for parameter counting tests
        self.linear = torch.nn.Linear(3, 4)

    def forward(self, inputs, **kwargs):
        return self.linear(inputs)


class TestBaseModel:
    def test_metadata_defaults(self):
        """Verify default metadata fields are populated correctly."""
        model = ConcreteBaseModel()
        meta = model.metadata
        assert "model_name" in meta
        assert meta["model_name"] == "concrete_base"
        assert "model_class" in meta
        assert meta["model_class"] == "ConcreteBaseModel"
        assert "created_at" in meta
        assert "framework" in meta
        assert meta["framework"] == "pytorch"
        assert "framework_version" in meta

    def test_metadata_custom_name(self):
        """Verify model name can be overridden via constructor."""
        model = ConcreteBaseModel(name="my_model")
        assert model.metadata["model_name"] == "my_model"
        assert model.model_name == "my_model"

    def test_metadata_from_kwargs(self):
        """Verify extra metadata dict is merged into model metadata."""
        model = ConcreteBaseModel(metadata={"custom_key": "custom_value"})
        assert model.metadata["custom_key"] == "custom_value"

    def test_update_metadata(self):
        """Verify update_metadata merges new key-value pairs."""
        model = ConcreteBaseModel()
        model.update_metadata(extra_field=42)
        assert model.metadata["extra_field"] == 42

    def test_metadata_is_copy(self):
        """Verify metadata property returns a copy (not a mutable reference)."""
        model = ConcreteBaseModel()
        meta = model.metadata
        meta["mutated"] = True
        assert "mutated" not in model._model_metadata

    def test_get_config_includes_metadata(self):
        """Verify get_config() includes model metadata for serialization."""
        model = ConcreteBaseModel(name="config_test")
        config = model.get_config()
        assert "metadata" in config
        assert config["metadata"]["model_name"] == "config_test"

    def test_from_config_restores_metadata(self):
        """Verify from_config() round-trips metadata correctly."""
        config = {
            "name": "original",
            "metadata": {
                "model_name": "original",
                "model_class": "ConcreteBaseModel",
                "tag": "v1",
                "framework": "pytorch",
            },
        }
        restored = ConcreteBaseModel.from_config(config.copy())
        assert restored.metadata["model_name"] == "original"
        assert restored.metadata["tag"] == "v1"

    def test_summary_dict_structure(self):
        """Verify summary_dict() returns expected keys and correct param counts."""
        model = ConcreteBaseModel(name="summary_test")
        summary = model.summary_dict()

        assert "name" in summary
        assert summary["name"] == "summary_test"
        assert "class" in summary
        assert summary["class"] == "ConcreteBaseModel"
        assert "trainable_params" in summary
        assert "non_trainable_params" in summary
        assert "modules" in summary
        assert isinstance(summary["modules"], list)
        assert "metadata" in summary

    def test_summary_dict_param_counts(self):
        """Verify trainable param count matches expected Linear(3,4) dimensions."""
        model = ConcreteBaseModel()
        summary = model.summary_dict()
        # Linear(3, 4) has 3*4 weights + 4 bias = 16 trainable params
        assert summary["trainable_params"] == 16
        assert summary["non_trainable_params"] == 0

    def test_forward_pass(self):
        """Verify forward pass produces correct output shape."""
        model = ConcreteBaseModel()
        x = torch.randn(2, 3)
        out = model(x)
        assert out.shape == (2, 4)

    def test_save_and_load_config(self, tmp_path):
        """Verify config can be saved to JSON and loaded back."""
        model = ConcreteBaseModel(name="save_test", metadata={"version": "1.0"})
        config_path = tmp_path / "model_config.json"
        model.save_config(config_path)

        loaded_config = BaseModel.load_config(config_path)
        assert loaded_config["name"] == "save_test"
        assert loaded_config["metadata"]["version"] == "1.0"

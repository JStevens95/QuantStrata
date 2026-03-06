"""Unit tests for rade_ml_pt.pipelines.config -- PipelineConfig."""
import pytest

from src.rade_ml_pt.pipelines.config import PipelineConfig


class TestPipelineConfig:
    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.training_config is None
        assert cfg.registry_dir is None
        assert cfg.version_or_tag == "latest"
        assert cfg.metadata == {}

    def test_custom_values(self):
        cfg = PipelineConfig(
            training_config={"epochs": 10},
            registry_dir="/tmp/registry",
            version_or_tag="best",
            metadata={"team": "quant"},
        )
        assert cfg.training_config["epochs"] == 10
        assert cfg.registry_dir == "/tmp/registry"
        assert cfg.version_or_tag == "best"
        assert cfg.metadata["team"] == "quant"

    def test_metadata_default_factory(self):
        cfg1 = PipelineConfig()
        cfg2 = PipelineConfig()
        cfg1.metadata["key"] = "val"
        assert "key" not in cfg2.metadata

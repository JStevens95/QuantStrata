"""Unit tests for rade_ml_pt.models.hybrid_gnn_rnn.config."""
import json
import tempfile
from pathlib import Path

import pytest

from src.rade_ml_pt.models.hybrid_gnn_rnn.config import (
    HybridGnnRnnModelConfig,
    default_model_config,
)


class TestHybridGnnRnnModelConfig:
    """Tests for dataclass-based model config."""

    def test_from_dict_returns_dataclass(self):
        cfg = HybridGnnRnnModelConfig.from_dict({})
        assert isinstance(cfg, HybridGnnRnnModelConfig)

    def test_from_dict_partial_override(self):
        d = {"gnn_layer": {"parameters": {"units": 256}}}
        cfg = HybridGnnRnnModelConfig.from_dict(d)
        assert cfg.gnn_layer.parameters.units == 256
        assert cfg.rnn_layer.parameters.units == 128  # default

    def test_to_dict_matches_layer_expectation(self):
        cfg = HybridGnnRnnModelConfig()
        d = cfg.to_dict()
        assert "general" in d
        assert "gnn_layer" in d and "general" in d["gnn_layer"] and "parameters" in d["gnn_layer"]
        assert d["gnn_layer"]["parameters"]["units"] == 128

    def test_roundtrip_dict(self):
        cfg = HybridGnnRnnModelConfig()
        cfg.gnn_layer.parameters.units = 64
        d = cfg.to_dict()
        restored = HybridGnnRnnModelConfig.from_dict(d)
        assert restored.gnn_layer.parameters.units == 64

    def test_from_json(self):
        d = default_model_config()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(d, f, indent=2)
            path = f.name
        try:
            cfg = HybridGnnRnnModelConfig.from_json(path)
            assert cfg.gnn_layer.parameters.units == 128
        finally:
            Path(path).unlink(missing_ok=True)

    def test_to_json(self):
        cfg = HybridGnnRnnModelConfig()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cfg.to_json(path)
            loaded = HybridGnnRnnModelConfig.from_json(path)
            assert loaded.to_dict() == cfg.to_dict()
        finally:
            Path(path).unlink(missing_ok=True)


class TestDefaultModelConfig:
    def test_returns_dict(self):
        cfg = default_model_config()
        assert isinstance(cfg, dict)

    def test_all_layer_keys_present(self):
        cfg = default_model_config()
        expected = {"general", "gnn_layer", "rnn_layer", "fusion_layer", "attention_layer", "projection_layer"}
        assert expected.issubset(cfg.keys())

    def test_gnn_layer_structure(self):
        cfg = default_model_config()
        gnn = cfg["gnn_layer"]
        assert "general" in gnn
        assert "parameters" in gnn
        assert gnn["general"]["layer_type"] == "mixed_graph_sage"
        assert gnn["parameters"]["units"] == 128

    def test_rnn_layer_structure(self):
        cfg = default_model_config()
        rnn = cfg["rnn_layer"]
        assert rnn["general"]["layer_type"] == "lstm"
        assert rnn["parameters"]["units"] == 128
        assert rnn["parameters"]["recurrent_activation"] == "sigmoid"

    def test_fusion_layer_structure(self):
        cfg = default_model_config()
        fusion = cfg["fusion_layer"]
        assert fusion["general"]["fusion_mode"] == "gate"
        assert fusion["parameters"]["units"] == 64

    def test_attention_layer_structure(self):
        cfg = default_model_config()
        attn = cfg["attention_layer"]
        assert attn["parameters"]["units"] == 32
        assert attn["parameters"]["activation"] == "tanh"

    def test_projection_layer_structure(self):
        cfg = default_model_config()
        proj = cfg["projection_layer"]
        assert proj["general"]["knn_mode"] == "cosine_softmax"
        assert proj["parameters"]["activation"] == "gelu"

    def test_units_divisible_by_heads(self):
        """Fusion and attention units must be divisible by num_heads."""
        cfg = default_model_config()
        fusion_units = cfg["fusion_layer"]["parameters"]["units"]
        fusion_heads = cfg["fusion_layer"]["general"]["num_heads"]
        assert fusion_units % fusion_heads == 0

        attn_units = cfg["attention_layer"]["parameters"]["units"]
        attn_heads = cfg["attention_layer"]["general"]["num_heads"]
        assert attn_units % attn_heads == 0

    def test_independent_copies(self):
        cfg1 = default_model_config()
        cfg2 = default_model_config()
        cfg1["gnn_layer"]["parameters"]["units"] = 999
        assert cfg2["gnn_layer"]["parameters"]["units"] == 128

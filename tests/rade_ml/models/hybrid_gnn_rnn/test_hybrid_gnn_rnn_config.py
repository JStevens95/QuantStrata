"""Unit tests for rade_ml.models.hybrid_gnn_rnn.config."""
import pytest

from src.rade_ml.models.hybrid_gnn_rnn.config import default_model_config


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

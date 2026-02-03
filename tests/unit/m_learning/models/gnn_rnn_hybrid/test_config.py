"""Tests for m_learning.models.gnn_rnn_hybrid.config."""

import pytest

from src.m_learning.models.gnn_rnn_hybrid.config import default_hybrid_model_config


class TestDefaultHybridModelConfig:
    """Tests for default_hybrid_model_config."""

    def test_returns_dict(self):
        """default_hybrid_model_config returns a dict."""
        config = default_hybrid_model_config(n_targets=5)
        assert isinstance(config, dict)

    def test_has_required_sections(self):
        """Config contains sections required by HybridGnnRnn."""
        config = default_hybrid_model_config(n_targets=10)
        assert "general" in config
        assert "gnn_model" in config
        assert "rnn_model" in config
        assert "fusion_model" in config
        assert "attention_model" in config
        assert "projection_model" in config

    def test_gnn_parameters(self):
        """GNN section has parameters and general."""
        config = default_hybrid_model_config(gnn_units=64)
        gnn = config["gnn_model"]
        assert "parameters" in gnn
        assert gnn["parameters"]["units"] == 64
        assert "general" in gnn
        assert "layers" in gnn["general"]

    def test_projection_baseline_trade_count(self):
        """Projection section baseline_trade_count matches n_targets."""
        config = default_hybrid_model_config(n_targets=7)
        proj = config["projection_model"]["general"]
        assert proj["baseline_trade_count"] == 7

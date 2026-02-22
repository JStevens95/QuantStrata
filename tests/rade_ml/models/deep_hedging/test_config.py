"""Unit tests for rade_ml.models.deep_hedging.config -- default_model_config."""
import pytest

from src.rade_ml.models.deep_hedging.config import default_model_config


class TestDefaultModelConfig:
    def test_returns_dict(self):
        config = default_model_config()
        assert isinstance(config, dict)

    def test_has_all_top_level_keys(self):
        config = default_model_config()
        for key in ("general", "encoder", "policy", "risk_measure"):
            assert key in config, f"Missing top-level key: {key}"

    def test_general_defaults(self):
        g = default_model_config()["general"]
        assert g["num_hedging_instruments"] == 1
        assert g["transaction_cost_rate"] == 0.001
        assert g["position_limit"] is None

    def test_encoder_defaults(self):
        enc = default_model_config()["encoder"]
        assert enc["units"] == 64
        assert enc["dropout_rate"] == 0.1
        assert enc["activation"] == "elu"

    def test_policy_defaults(self):
        pol = default_model_config()["policy"]
        assert pol["rnn_type"] == "gru"
        assert pol["rnn_units"] == 128
        assert pol["rnn_layers"] == 2
        assert pol["output_activation"] is None

    def test_risk_measure_defaults(self):
        rm = default_model_config()["risk_measure"]
        assert rm["type"] == "cvar"
        assert rm["alpha"] == 0.95

    def test_configs_are_independent_copies(self):
        c1 = default_model_config()
        c2 = default_model_config()
        c1["encoder"]["units"] = 999
        assert c2["encoder"]["units"] == 64

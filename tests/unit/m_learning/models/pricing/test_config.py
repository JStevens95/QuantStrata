"""Tests for m_learning.models.pricing.config."""

import pytest

from src.m_learning.models.pricing.config import (
    PricingModelConfig,
    default_pricing_config,
)


class TestPricingModelConfig:
    """Tests for PricingModelConfig."""

    def test_defaults(self):
        """Default config has expected values."""
        config = PricingModelConfig()
        assert config.n_features == 6
        assert config.hidden_units == [128, 64, 32]
        assert config.activation == "relu"
        assert config.dropout_rate == 0.1
        assert config.use_batch_norm is True

    def test_to_dict(self):
        """to_dict returns serialisable dict."""
        config = PricingModelConfig(n_features=8, hidden_units=[64, 32])
        d = config.to_dict()
        assert d["n_features"] == 8
        assert d["hidden_units"] == [64, 32]
        assert "dropout_rate" in d


class TestDefaultPricingConfig:
    """Tests for default_pricing_config factory."""

    def test_returns_config(self):
        """default_pricing_config returns PricingModelConfig."""
        config = default_pricing_config()
        assert isinstance(config, PricingModelConfig)
        assert config.n_features == 6

    def test_custom_args(self):
        """Custom n_features and hidden_units are applied."""
        config = default_pricing_config(
            n_features=10,
            hidden_units=[256, 128, 64],
            dropout_rate=0.2,
        )
        assert config.n_features == 10
        assert config.hidden_units == [256, 128, 64]
        assert config.dropout_rate == 0.2

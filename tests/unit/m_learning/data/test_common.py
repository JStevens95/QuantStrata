"""Tests for m_learning.data.common (shared building blocks)."""

import pytest

from src.m_learning.data.common import TradeAttributeEncoder, TradeGraphBuilder


class TestTradeAttributeEncoder:
    """Tests for TradeAttributeEncoder re-export."""

    def test_encoder_instantiation(self):
        """TradeAttributeEncoder can be instantiated."""
        encoder = TradeAttributeEncoder()
        assert encoder is not None

    def test_encoder_fit_transform_dict_api(self):
        """Encoder fit/transform accept dict of list attributes (default keys)."""
        encoder = TradeAttributeEncoder(
            numeric_keys=("moneyness", "delta", "vega", "yrs_to_maturity"),
            categorical_keys=("product_type", "product_subtype", "trade_type"),
            multi_label_keys=("underlying_risk_factors",),
            num_decay_terms=0,
        )
        elem_attrs = {
            "moneyness": [1.0, 1.1],
            "delta": [0.5, -0.2],
            "vega": [10.0, 20.0],
            "yrs_to_maturity": [0.5, 1.0],
            "product_type": ["vanilla", "digital"],
            "product_subtype": ["euro", "one_touch"],
            "trade_type": ["option", "option"],
            "underlying_risk_factors": [["fx"], ["fx"]],
        }
        target_attrs = {
            "moneyness": [0.95],
            "delta": [0.3],
            "vega": [15.0],
            "yrs_to_maturity": [0.25],
            "product_type": ["barrier"],
            "product_subtype": ["knock_in"],
            "trade_type": ["option"],
            "underlying_risk_factors": [["fx"]],
        }
        encoder.fit(elem_attrs, target_attrs)
        out = encoder.transform({
            "moneyness": [1.0],
            "delta": [0.5],
            "vega": [10.0],
            "yrs_to_maturity": [0.5],
            "product_type": ["vanilla"],
            "product_subtype": ["euro"],
            "trade_type": ["option"],
            "underlying_risk_factors": [["fx"]],
        })
        assert isinstance(out, dict)
        assert len(out) >= 1


class TestTradeGraphBuilder:
    """Tests for TradeGraphBuilder re-export."""

    def test_builder_instantiation(self):
        """TradeGraphBuilder can be instantiated with k."""
        builder = TradeGraphBuilder(k=5)
        assert builder.k == 5

    def test_build_adjacency(self):
        """Builder produces row-normalised adjacency from features."""
        import numpy as np
        builder = TradeGraphBuilder(k=2)
        features = np.random.randn(10, 4).astype(np.float32)
        adj = builder.build(features)
        assert adj.shape == (10, 10)
        # Row-normalised: rows sum to 1 (or 0 if no neighbours)
        row_sums = adj.sum(axis=1)
        assert np.allclose(row_sums, 1.0) or np.allclose(row_sums, 0.0)

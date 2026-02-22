"""Unit tests for rade_ml.models.hybrid_gnn_rnn.layers.attention_layer."""
import numpy as np
import pytest
import tensorflow as tf

from src.rade_ml.models.hybrid_gnn_rnn.layers.attention_layer import TargetAttentionLayer


BATCH = 4
NUM_TRADES = 20
NUM_TARGETS = 8
FUSED_DIM = 64


def _make_attn_config(units=32, num_heads=1, dropout=0.0):
    return {
        "general": {
            "layer_type": "standard",
            "use_residual": True,
            "use_layer_norm": True,
            "attention_mode": True,
            "num_heads": num_heads,
            "dropout_rate": dropout,
            "k_nbrs": 10,
        },
        "parameters": {
            "units": units,
            "activation": "tanh",
            "kernel_initializer": "he_uniform",
            "bias_initializer": "zeros",
        }
    }


@pytest.fixture
def fused_features():
    np.random.seed(42)
    return tf.constant(np.random.randn(BATCH, NUM_TRADES, FUSED_DIM).astype(np.float32))


@pytest.fixture
def adjacency():
    np.random.seed(42)
    raw = np.random.rand(NUM_TRADES, NUM_TRADES).astype(np.float32)
    raw = (raw > 0.5).astype(np.float32)
    np.fill_diagonal(raw, 1.0)
    row_sums = raw.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return tf.constant(raw / row_sums)


@pytest.fixture
def target_idx():
    return tf.constant(list(range(NUM_TARGETS)), dtype=tf.int32)


class TestTargetAttentionLayerForward:
    def test_output_shape(self, fused_features, adjacency, target_idx):
        layer = TargetAttentionLayer(layer_config=_make_attn_config(), name="attn_test")
        out = layer((fused_features, adjacency, target_idx), training=False)
        assert out.shape == (BATCH, NUM_TARGETS, 32)

    def test_no_nan_output(self, fused_features, adjacency, target_idx):
        layer = TargetAttentionLayer(layer_config=_make_attn_config(), name="attn_nan")
        out = layer((fused_features, adjacency, target_idx), training=False)
        assert not tf.reduce_any(tf.math.is_nan(out)).numpy()

    def test_training_mode(self, fused_features, adjacency, target_idx):
        cfg = _make_attn_config(dropout=0.3)
        layer = TargetAttentionLayer(layer_config=cfg, name="attn_train")
        out = layer((fused_features, adjacency, target_idx), training=True)
        assert out.shape == (BATCH, NUM_TARGETS, 32)


class TestTargetAttentionLayerMultiHead:
    def test_multi_head_output(self, fused_features, adjacency, target_idx):
        layer = TargetAttentionLayer(
            layer_config=_make_attn_config(units=32, num_heads=4),
            name="attn_mh"
        )
        out = layer((fused_features, adjacency, target_idx), training=False)
        assert out.shape == (BATCH, NUM_TARGETS, 32)


class TestTargetAttentionLayerSparse:
    def test_sparse_adjacency(self, fused_features, adjacency, target_idx):
        sparse_adj = tf.sparse.from_dense(adjacency)
        layer = TargetAttentionLayer(layer_config=_make_attn_config(), name="attn_sparse")
        out = layer((fused_features, sparse_adj, target_idx), training=False)
        assert out.shape == (BATCH, NUM_TARGETS, 32)


class TestTargetAttentionLayerConfig:
    def test_get_config(self):
        layer = TargetAttentionLayer(layer_config=_make_attn_config(), name="attn_cfg")
        config = layer.get_config()
        assert "layer_config" in config

    def test_units_not_divisible_by_heads_raises(self):
        with pytest.raises(AssertionError, match="divisible"):
            TargetAttentionLayer(
                layer_config=_make_attn_config(units=33, num_heads=4),
                name="attn_bad_heads"
            )

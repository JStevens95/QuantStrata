"""Unit tests for rade_ml.models.hybrid_gnn_rnn.layers.fusion_layer."""
import numpy as np
import pytest
import tensorflow as tf

from src.rade_ml.models.hybrid_gnn_rnn.layers.fusion_layer import FusionLayer


NUM_TRADES = 20
BATCH = 4
GNN_DIM = 64
RNN_DIM = 64


def _make_fusion_config(fusion_mode="gate", units=64, num_heads=1):
    return {
        "general": {
            "fusion_mode": fusion_mode,
            "dropout_rate": 0.0,
            "num_heads": num_heads,
            "k_nbrs": 10,
        },
        "parameters": {
            "units": units,
            "activation": "sigmoid",
            "kernel_initializer": "he_uniform",
            "bias_initializer": "zeros",
        }
    }


@pytest.fixture
def gnn_features():
    np.random.seed(42)
    return tf.constant(np.random.randn(NUM_TRADES, GNN_DIM).astype(np.float32))


@pytest.fixture
def rnn_features():
    np.random.seed(43)
    return tf.constant(np.random.randn(BATCH, RNN_DIM).astype(np.float32))


@pytest.fixture
def adjacency():
    np.random.seed(44)
    raw = np.random.rand(NUM_TRADES, NUM_TRADES).astype(np.float32)
    raw = (raw > 0.7).astype(np.float32)
    np.fill_diagonal(raw, 1.0)
    row_sums = raw.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return tf.constant(raw / row_sums)


class TestFusionLayerGate:
    def test_output_shape(self, gnn_features, rnn_features, adjacency):
        layer = FusionLayer(layer_config=_make_fusion_config("gate"), name="fusion_gate")
        out = layer((gnn_features, rnn_features, adjacency), training=False)
        assert out.shape == (BATCH, NUM_TRADES, 64)

    def test_no_nan_output(self, gnn_features, rnn_features, adjacency):
        layer = FusionLayer(layer_config=_make_fusion_config("gate"), name="fusion_gate_nan")
        out = layer((gnn_features, rnn_features, adjacency), training=False)
        assert not tf.reduce_any(tf.math.is_nan(out)).numpy()

    def test_training_mode(self, gnn_features, rnn_features, adjacency):
        cfg = _make_fusion_config("gate")
        cfg["general"]["dropout_rate"] = 0.3
        layer = FusionLayer(layer_config=cfg, name="fusion_gate_train")
        out = layer((gnn_features, rnn_features, adjacency), training=True)
        assert out.shape == (BATCH, NUM_TRADES, 64)


class TestFusionLayerAdd:
    def test_output_shape(self, gnn_features, rnn_features, adjacency):
        layer = FusionLayer(layer_config=_make_fusion_config("add"), name="fusion_add")
        out = layer((gnn_features, rnn_features, adjacency), training=False)
        assert out.shape == (BATCH, NUM_TRADES, 64)


class TestFusionLayerInvalid:
    def test_invalid_mode_raises(self, gnn_features, rnn_features, adjacency):
        layer = FusionLayer(layer_config=_make_fusion_config("invalid_mode"), name="fusion_bad")
        with pytest.raises(ValueError, match="not recognised"):
            layer((gnn_features, rnn_features, adjacency))


class TestFusionLayerMultiHead:
    def test_multi_head_output_shape(self, gnn_features, rnn_features, adjacency):
        layer = FusionLayer(
            layer_config=_make_fusion_config("gate", units=64, num_heads=4),
            name="fusion_mh"
        )
        out = layer((gnn_features, rnn_features, adjacency), training=False)
        assert out.shape == (BATCH, NUM_TRADES, 64)


class TestFusionLayerSparse:
    def test_sparse_adjacency(self, gnn_features, rnn_features, adjacency):
        sparse_adj = tf.sparse.from_dense(adjacency)
        layer = FusionLayer(layer_config=_make_fusion_config("gate"), name="fusion_sparse")
        out = layer((gnn_features, rnn_features, sparse_adj), training=False)
        assert out.shape == (BATCH, NUM_TRADES, 64)


class TestFusionLayerConfig:
    def test_get_config(self):
        layer = FusionLayer(layer_config=_make_fusion_config(), name="fusion_cfg")
        config = layer.get_config()
        assert "layer_config" in config

    def test_compute_output_shape(self):
        layer = FusionLayer(layer_config=_make_fusion_config(), name="fusion_oshape")
        shape = layer.compute_output_shape((
            tf.TensorShape([NUM_TRADES, GNN_DIM]),
            tf.TensorShape([BATCH, RNN_DIM]),
            tf.TensorShape([NUM_TRADES, NUM_TRADES]),
        ))
        assert shape.as_list() == [BATCH, NUM_TRADES, 64]

    def test_units_not_divisible_by_heads_raises(self):
        with pytest.raises(AssertionError, match="divisible"):
            FusionLayer(
                layer_config=_make_fusion_config("gate", units=63, num_heads=4),
                name="fusion_bad_heads"
            )

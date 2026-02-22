"""Unit tests for rade_ml.models.hybrid_gnn_rnn.layers.gnn_layers."""
import numpy as np
import pytest
import tensorflow as tf

from src.rade_ml.models.hybrid_gnn_rnn.layers.gnn_layers import GnnBlock, GraphSage, MixedGraphSage


NUM_TRADES = 20
FEATURE_DIM = 10


@pytest.fixture
def features():
    np.random.seed(0)
    return tf.constant(np.random.randn(NUM_TRADES, FEATURE_DIM).astype(np.float32))


@pytest.fixture
def adjacency():
    np.random.seed(0)
    raw = np.random.rand(NUM_TRADES, NUM_TRADES).astype(np.float32)
    raw = (raw > 0.7).astype(np.float32)
    np.fill_diagonal(raw, 1.0)
    row_sums = raw.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return tf.constant(raw / row_sums)


@pytest.fixture
def sparse_adj(adjacency):
    return tf.sparse.from_dense(adjacency)


class _BaseGnnConfig:
    @staticmethod
    def make_config(layer_type="graph_sage", units=32):
        return {
            "general": {
                "architecture": "default",
                "layers": 2,
                "layer_type": layer_type,
                "dropout_rate": 0.0,
                "use_bias": True,
                "use_residual": True,
                "batch_norm": False,
                "aggregator_op": "mean",
            },
            "parameters": {
                "units": units,
                "activation": "relu",
                "kernel_initializer": "glorot_uniform",
                "bias_initializer": "zeros",
            }
        }


class TestGraphSage:
    def test_output_shape(self, features, adjacency):
        cfg = _BaseGnnConfig.make_config()
        layer = GraphSage(layer_config=cfg, name="test_gs")
        out = layer((features, adjacency), training=False)
        assert out.shape == (NUM_TRADES, 32)

    def test_sparse_adjacency(self, features, sparse_adj):
        cfg = _BaseGnnConfig.make_config()
        layer = GraphSage(layer_config=cfg, name="test_gs_sparse")
        out = layer((features, sparse_adj), training=False)
        assert out.shape == (NUM_TRADES, 32)

    def test_max_aggregator(self, features, adjacency):
        cfg = _BaseGnnConfig.make_config()
        cfg["general"]["aggregator_op"] = "max"
        layer = GraphSage(layer_config=cfg, name="test_gs_max")
        out = layer((features, adjacency), training=False)
        assert out.shape == (NUM_TRADES, 32)

    def test_invalid_aggregator_raises(self, features, adjacency):
        cfg = _BaseGnnConfig.make_config()
        cfg["general"]["aggregator_op"] = "invalid"
        layer = GraphSage(layer_config=cfg, name="test_gs_bad")
        with pytest.raises(ValueError, match="Unsupported aggregator"):
            layer((features, adjacency))

    def test_get_config(self):
        cfg = _BaseGnnConfig.make_config()
        layer = GraphSage(layer_config=cfg, name="test_gs_cfg")
        config = layer.get_config()
        assert "layer_config" in config

    def test_no_nan_in_output(self, features, adjacency):
        cfg = _BaseGnnConfig.make_config()
        layer = GraphSage(layer_config=cfg, name="test_gs_nan")
        out = layer((features, adjacency))
        assert not tf.reduce_any(tf.math.is_nan(out)).numpy()


class TestMixedGraphSage:
    def test_output_shape(self, features, adjacency):
        cfg = _BaseGnnConfig.make_config(layer_type="mixed_graph_sage")
        layer = MixedGraphSage(layer_config=cfg, name="test_mgs")
        out = layer((features, adjacency), training=False)
        assert out.shape == (NUM_TRADES, 32)

    def test_sparse_adjacency(self, features, sparse_adj):
        cfg = _BaseGnnConfig.make_config(layer_type="mixed_graph_sage")
        layer = MixedGraphSage(layer_config=cfg, name="test_mgs_sparse")
        out = layer((features, sparse_adj), training=False)
        assert out.shape == (NUM_TRADES, 32)

    def test_no_nan_in_output(self, features, adjacency):
        cfg = _BaseGnnConfig.make_config(layer_type="mixed_graph_sage")
        layer = MixedGraphSage(layer_config=cfg, name="test_mgs_nan")
        out = layer((features, adjacency))
        assert not tf.reduce_any(tf.math.is_nan(out)).numpy()

    def test_compute_output_shape(self, features, adjacency):
        cfg = _BaseGnnConfig.make_config(layer_type="mixed_graph_sage")
        layer = MixedGraphSage(layer_config=cfg, name="test_mgs_shape")
        shape = layer.compute_output_shape(
            (tf.TensorShape([NUM_TRADES, FEATURE_DIM]), tf.TensorShape([NUM_TRADES, NUM_TRADES]))
        )
        assert shape.as_list() == [NUM_TRADES, 32]


class TestGnnBlock:
    def test_output_shape_graph_sage(self, features, adjacency):
        cfg = _BaseGnnConfig.make_config(layer_type="graph_sage", units=64)
        block = GnnBlock(layer_config=cfg, name="test_block_gs")
        out = block((features, adjacency), training=False)
        assert out.shape == (NUM_TRADES, 64)

    def test_output_shape_mixed_graph_sage(self, features, adjacency):
        cfg = _BaseGnnConfig.make_config(layer_type="mixed_graph_sage", units=64)
        block = GnnBlock(layer_config=cfg, name="test_block_mgs")
        out = block((features, adjacency), training=False)
        assert out.shape == (NUM_TRADES, 64)

    def test_invalid_layer_type_raises(self, features, adjacency):
        cfg = _BaseGnnConfig.make_config(layer_type="invalid_type")
        block = GnnBlock(layer_config=cfg, name="test_block_bad")
        with pytest.raises(ValueError, match="Undefined layer type"):
            block((features, adjacency))

    def test_residual_connection(self, features, adjacency):
        cfg = _BaseGnnConfig.make_config(units=64)
        cfg["general"]["use_residual"] = True
        block = GnnBlock(layer_config=cfg, name="test_block_res")
        out = block((features, adjacency), training=False)
        assert out.shape == (NUM_TRADES, 64)

    def test_no_residual(self, features, adjacency):
        cfg = _BaseGnnConfig.make_config(units=64)
        cfg["general"]["use_residual"] = False
        block = GnnBlock(layer_config=cfg, name="test_block_nores")
        out = block((features, adjacency), training=False)
        assert out.shape == (NUM_TRADES, 64)

    def test_training_mode_dropout(self, features, adjacency):
        cfg = _BaseGnnConfig.make_config(units=32)
        cfg["general"]["dropout_rate"] = 0.5
        block = GnnBlock(layer_config=cfg, name="test_block_drop")
        out_train = block((features, adjacency), training=True)
        out_infer = block((features, adjacency), training=False)
        assert out_train.shape == out_infer.shape

    def test_get_config(self):
        cfg = _BaseGnnConfig.make_config()
        block = GnnBlock(layer_config=cfg, name="test_block_cfg")
        config = block.get_config()
        assert "layer_config" in config

    def test_no_nan_in_output(self, features, adjacency):
        cfg = _BaseGnnConfig.make_config(units=32)
        block = GnnBlock(layer_config=cfg, name="test_block_nan")
        out = block((features, adjacency), training=False)
        assert not tf.reduce_any(tf.math.is_nan(out)).numpy()

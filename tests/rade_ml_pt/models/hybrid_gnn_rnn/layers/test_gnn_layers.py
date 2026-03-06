"""Unit tests for rade_ml_pt.models.hybrid_gnn_rnn.layers.gnn_layers."""
import numpy as np
import pytest
import torch

from src.rade_ml_pt.models.hybrid_gnn_rnn.layers.gnn_layers import GnnBlock, GraphSage, MixedGraphSage


NUM_TRADES = 20
FEATURE_DIM = 10


@pytest.fixture
def features():
    """Random float32 node features [NUM_TRADES, FEATURE_DIM]."""
    np.random.seed(0)
    return torch.tensor(np.random.randn(NUM_TRADES, FEATURE_DIM).astype(np.float32))


@pytest.fixture
def adjacency():
    """Row-normalised dense adjacency matrix [NUM_TRADES, NUM_TRADES] with self-loops."""
    np.random.seed(0)
    raw = np.random.rand(NUM_TRADES, NUM_TRADES).astype(np.float32)
    raw = (raw > 0.7).astype(np.float32)
    np.fill_diagonal(raw, 1.0)
    row_sums = raw.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return torch.tensor(raw / row_sums)


@pytest.fixture
def sparse_adj(adjacency):
    """Sparse COO version of the dense adjacency fixture."""
    return adjacency.to_sparse_coo()


class _BaseGnnConfig:
    """Helper to build standard layer_config dicts for tests."""

    @staticmethod
    def make_config(layer_type="graph_sage", units=32):
        """Return a config dict matching the expected GNN layer schema."""
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
            },
        }


class TestGraphSage:
    """Tests for the GraphSage layer."""

    def test_output_shape(self, features, adjacency):
        """GraphSage with dense adjacency produces [T, units] output."""
        cfg = _BaseGnnConfig.make_config()
        layer = GraphSage(layer_config=cfg, name="test_gs")
        layer.eval()
        out = layer(features, adjacency)
        assert out.shape == (NUM_TRADES, 32)

    def test_sparse_adjacency(self, features, sparse_adj):
        """GraphSage with sparse COO adjacency produces [T, units] output."""
        cfg = _BaseGnnConfig.make_config()
        layer = GraphSage(layer_config=cfg, name="test_gs_sparse")
        layer.eval()
        out = layer(features, sparse_adj)
        assert out.shape == (NUM_TRADES, 32)

    def test_max_aggregator(self, features, adjacency):
        """GraphSage with max aggregation produces [T, units] output."""
        cfg = _BaseGnnConfig.make_config()
        cfg["general"]["aggregator_op"] = "max"
        layer = GraphSage(layer_config=cfg, name="test_gs_max")
        layer.eval()
        out = layer(features, adjacency)
        assert out.shape == (NUM_TRADES, 32)

    def test_max_aggregator_sparse(self, features, sparse_adj):
        """GraphSage with max aggregation on sparse adjacency produces [T, units] output."""
        cfg = _BaseGnnConfig.make_config()
        cfg["general"]["aggregator_op"] = "max"
        layer = GraphSage(layer_config=cfg, name="test_gs_max_sp")
        layer.eval()
        out = layer(features, sparse_adj)
        assert out.shape == (NUM_TRADES, 32)

    def test_invalid_aggregator_raises(self, features, adjacency):
        """GraphSage raises ValueError for an unknown aggregation operator."""
        cfg = _BaseGnnConfig.make_config()
        cfg["general"]["aggregator_op"] = "invalid"
        layer = GraphSage(layer_config=cfg, name="test_gs_bad")
        with pytest.raises(ValueError, match="Unsupported aggregator"):
            layer(features, adjacency)

    def test_get_config(self):
        """get_config returns a dict containing 'layer_config'."""
        cfg = _BaseGnnConfig.make_config()
        layer = GraphSage(layer_config=cfg, name="test_gs_cfg")
        config = layer.get_config()
        assert "layer_config" in config

    def test_no_nan_in_output(self, features, adjacency):
        """GraphSage output contains no NaN values."""
        cfg = _BaseGnnConfig.make_config()
        layer = GraphSage(layer_config=cfg, name="test_gs_nan")
        layer.eval()
        out = layer(features, adjacency)
        assert not torch.any(torch.isnan(out)).item()


class TestMixedGraphSage:
    """Tests for the MixedGraphSage layer."""

    def test_output_shape(self, features, adjacency):
        """MixedGraphSage with dense adjacency produces [T, units] output."""
        cfg = _BaseGnnConfig.make_config(layer_type="mixed_graph_sage")
        layer = MixedGraphSage(layer_config=cfg, name="test_mgs")
        layer.eval()
        out = layer(features, adjacency)
        assert out.shape == (NUM_TRADES, 32)

    def test_sparse_adjacency(self, features, sparse_adj):
        """MixedGraphSage with sparse COO adjacency produces [T, units] output."""
        cfg = _BaseGnnConfig.make_config(layer_type="mixed_graph_sage")
        layer = MixedGraphSage(layer_config=cfg, name="test_mgs_sparse")
        layer.eval()
        out = layer(features, sparse_adj)
        assert out.shape == (NUM_TRADES, 32)

    def test_no_nan_in_output(self, features, adjacency):
        """MixedGraphSage output contains no NaN values."""
        cfg = _BaseGnnConfig.make_config(layer_type="mixed_graph_sage")
        layer = MixedGraphSage(layer_config=cfg, name="test_mgs_nan")
        layer.eval()
        out = layer(features, adjacency)
        assert not torch.any(torch.isnan(out)).item()

    def test_from_config_roundtrip(self):
        """from_config reconstructs a MixedGraphSage from get_config output."""
        cfg = _BaseGnnConfig.make_config(layer_type="mixed_graph_sage")
        layer = MixedGraphSage(layer_config=cfg, name="test_mgs_rt")
        rebuilt = MixedGraphSage.from_config(layer.get_config())
        assert rebuilt.units == layer.units


class TestGnnBlock:
    """Tests for the GnnBlock wrapper."""

    def test_output_shape_graph_sage(self, features, adjacency):
        """GnnBlock with GraphSage sublayers produces [T, units] output."""
        cfg = _BaseGnnConfig.make_config(layer_type="graph_sage", units=64)
        block = GnnBlock(layer_config=cfg, name="test_block_gs")
        block.eval()
        out = block(features, adjacency)
        assert out.shape == (NUM_TRADES, 64)

    def test_output_shape_mixed_graph_sage(self, features, adjacency):
        """GnnBlock with MixedGraphSage sublayers produces [T, units] output."""
        cfg = _BaseGnnConfig.make_config(layer_type="mixed_graph_sage", units=64)
        block = GnnBlock(layer_config=cfg, name="test_block_mgs")
        block.eval()
        out = block(features, adjacency)
        assert out.shape == (NUM_TRADES, 64)

    def test_invalid_layer_type_raises(self, features, adjacency):
        """GnnBlock raises ValueError for an unknown layer_type during construction."""
        cfg = _BaseGnnConfig.make_config(layer_type="invalid_type")
        with pytest.raises(ValueError, match="Undefined layer type"):
            GnnBlock(layer_config=cfg, name="test_block_bad")

    def test_residual_connection(self, features, adjacency):
        """GnnBlock with residual=True produces [T, units] output."""
        cfg = _BaseGnnConfig.make_config(units=64)
        cfg["general"]["use_residual"] = True
        block = GnnBlock(layer_config=cfg, name="test_block_res")
        block.eval()
        out = block(features, adjacency)
        assert out.shape == (NUM_TRADES, 64)

    def test_no_residual(self, features, adjacency):
        """GnnBlock with residual=False produces [T, units] output (no projection)."""
        cfg = _BaseGnnConfig.make_config(units=64)
        cfg["general"]["use_residual"] = False
        block = GnnBlock(layer_config=cfg, name="test_block_nores")
        block.eval()
        out = block(features, adjacency)
        assert out.shape == (NUM_TRADES, 64)

    def test_training_mode_dropout(self, features, adjacency):
        """GnnBlock with dropout produces same-shaped output in train and eval mode."""
        cfg = _BaseGnnConfig.make_config(units=32)
        cfg["general"]["dropout_rate"] = 0.5
        block = GnnBlock(layer_config=cfg, name="test_block_drop")

        block.train()
        out_train = block(features, adjacency)

        block.eval()
        out_infer = block(features, adjacency)

        assert out_train.shape == out_infer.shape

    def test_batch_norm(self, features, adjacency):
        """GnnBlock with batch_norm=True applies LayerNorm between sublayers."""
        cfg = _BaseGnnConfig.make_config(units=32)
        cfg["general"]["batch_norm"] = True
        block = GnnBlock(layer_config=cfg, name="test_block_bn")
        block.eval()
        out = block(features, adjacency)
        assert out.shape == (NUM_TRADES, 32)

    def test_get_config(self):
        """get_config returns a dict containing 'layer_config'."""
        cfg = _BaseGnnConfig.make_config()
        block = GnnBlock(layer_config=cfg, name="test_block_cfg")
        config = block.get_config()
        assert "layer_config" in config

    def test_no_nan_in_output(self, features, adjacency):
        """GnnBlock output contains no NaN values."""
        cfg = _BaseGnnConfig.make_config(units=32)
        block = GnnBlock(layer_config=cfg, name="test_block_nan")
        block.eval()
        out = block(features, adjacency)
        assert not torch.any(torch.isnan(out)).item()

    def test_sparse_adjacency(self, features, sparse_adj):
        """GnnBlock with sparse COO adjacency produces [T, units] output."""
        cfg = _BaseGnnConfig.make_config(units=32)
        block = GnnBlock(layer_config=cfg, name="test_block_sp")
        block.eval()
        out = block(features, sparse_adj)
        assert out.shape == (NUM_TRADES, 32)

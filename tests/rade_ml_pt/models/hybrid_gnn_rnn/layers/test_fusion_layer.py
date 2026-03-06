"""Unit tests for rade_ml_pt.models.hybrid_gnn_rnn.layers.fusion_layer."""
import numpy as np
import pytest
import torch

from src.rade_ml_pt.models.hybrid_gnn_rnn.layers.fusion_layer import FusionLayer


NUM_TRADES = 20
BATCH = 4
GNN_DIM = 64
RNN_DIM = 64


def _make_fusion_config(fusion_mode="gate", units=64, num_heads=1):
    """Return a minimal FusionLayer configuration dictionary."""
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
        },
    }


@pytest.fixture
def gnn_features():
    """Random float32 GNN node features [NUM_TRADES, GNN_DIM]."""
    np.random.seed(42)
    return torch.tensor(np.random.randn(NUM_TRADES, GNN_DIM).astype(np.float32))


@pytest.fixture
def rnn_features():
    """Random float32 RNN temporal features [BATCH, RNN_DIM]."""
    np.random.seed(43)
    return torch.tensor(np.random.randn(BATCH, RNN_DIM).astype(np.float32))


@pytest.fixture
def adjacency():
    """Row-normalised dense adjacency [NUM_TRADES, NUM_TRADES] with self-loops."""
    np.random.seed(44)
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


class TestFusionLayerGate:
    """Tests for FusionLayer with gate fusion mode."""

    def test_output_shape(self, gnn_features, rnn_features, adjacency):
        """Gate fusion should produce [B, T, units] output."""
        cfg = _make_fusion_config(fusion_mode="gate", units=64)
        layer = FusionLayer(layer_config=cfg)
        layer.eval()
        out = layer((gnn_features, rnn_features, adjacency))
        assert out.shape == (BATCH, NUM_TRADES, 64)

    def test_no_nan(self, gnn_features, rnn_features, adjacency):
        """Gate fusion output should not contain NaN values."""
        cfg = _make_fusion_config(fusion_mode="gate", units=64)
        layer = FusionLayer(layer_config=cfg)
        layer.eval()
        out = layer((gnn_features, rnn_features, adjacency))
        assert not torch.isnan(out).any().item()

    def test_training_mode(self, gnn_features, rnn_features, adjacency):
        """Training and eval output shapes should match even with dropout."""
        cfg = _make_fusion_config(fusion_mode="gate", units=64)
        cfg["general"]["dropout_rate"] = 0.5
        layer = FusionLayer(layer_config=cfg)

        layer.train()
        out_train = layer((gnn_features, rnn_features, adjacency))
        layer.eval()
        out_eval = layer((gnn_features, rnn_features, adjacency))

        assert out_train.shape == out_eval.shape


class TestFusionLayerAdd:
    """Tests for FusionLayer with add fusion mode."""

    def test_output_shape(self, gnn_features, rnn_features, adjacency):
        """Add fusion should produce [B, T, units] output."""
        cfg = _make_fusion_config(fusion_mode="add", units=64)
        layer = FusionLayer(layer_config=cfg)
        layer.eval()
        out = layer((gnn_features, rnn_features, adjacency))
        assert out.shape == (BATCH, NUM_TRADES, 64)

    def test_no_nan(self, gnn_features, rnn_features, adjacency):
        """Add fusion output should not contain NaN values."""
        cfg = _make_fusion_config(fusion_mode="add", units=64)
        layer = FusionLayer(layer_config=cfg)
        layer.eval()
        out = layer((gnn_features, rnn_features, adjacency))
        assert not torch.isnan(out).any().item()


class TestFusionLayerInvalid:
    """Tests for invalid fusion mode."""

    def test_invalid_fusion_mode(self, gnn_features, rnn_features, adjacency):
        """An unrecognised fusion_mode should raise ValueError on forward."""
        cfg = _make_fusion_config(fusion_mode="invalid")
        layer = FusionLayer(layer_config=cfg)
        with pytest.raises(ValueError, match="Unsupported fusion_mode"):
            layer((gnn_features, rnn_features, adjacency))


class TestFusionLayerMultiHead:
    """Tests for multi-head attention."""

    def test_multi_head_output_shape(self, gnn_features, rnn_features, adjacency):
        """Multi-head (4 heads) should produce same [B, T, units] output shape."""
        cfg = _make_fusion_config(fusion_mode="gate", units=64, num_heads=4)
        layer = FusionLayer(layer_config=cfg)
        layer.eval()
        out = layer((gnn_features, rnn_features, adjacency))
        assert out.shape == (BATCH, NUM_TRADES, 64)

    def test_multi_head_no_nan(self, gnn_features, rnn_features, adjacency):
        """Multi-head output should not contain NaN values."""
        cfg = _make_fusion_config(fusion_mode="gate", units=64, num_heads=4)
        layer = FusionLayer(layer_config=cfg)
        layer.eval()
        out = layer((gnn_features, rnn_features, adjacency))
        assert not torch.isnan(out).any().item()

    def test_return_attention_weights(self, gnn_features, rnn_features, adjacency):
        """return_attention=True should return output and attention weights."""
        cfg = _make_fusion_config(fusion_mode="gate", units=64, num_heads=4)
        layer = FusionLayer(layer_config=cfg)
        layer.eval()
        out, attn = layer(
            (gnn_features, rnn_features, adjacency), return_attention=True
        )
        assert out.shape == (BATCH, NUM_TRADES, 64)
        # Dense attention weights shape: [B, H, T, T]
        assert attn.shape == (BATCH, 4, NUM_TRADES, NUM_TRADES)


class TestFusionLayerSparse:
    """Tests for sparse adjacency attention path."""

    def test_sparse_output_shape(self, gnn_features, rnn_features, sparse_adj):
        """Sparse adjacency should produce same [B, T, units] output shape."""
        cfg = _make_fusion_config(fusion_mode="gate", units=64)
        layer = FusionLayer(layer_config=cfg)
        layer.eval()
        out = layer((gnn_features, rnn_features, sparse_adj))
        assert out.shape == (BATCH, NUM_TRADES, 64)

    def test_sparse_no_nan(self, gnn_features, rnn_features, sparse_adj):
        """Sparse path output should not contain NaN values."""
        cfg = _make_fusion_config(fusion_mode="gate", units=64)
        layer = FusionLayer(layer_config=cfg)
        layer.eval()
        out = layer((gnn_features, rnn_features, sparse_adj))
        assert not torch.isnan(out).any().item()

    def test_sparse_multi_head(self, gnn_features, rnn_features, sparse_adj):
        """Sparse multi-head attention should produce correct output shape."""
        cfg = _make_fusion_config(fusion_mode="gate", units=64, num_heads=4)
        layer = FusionLayer(layer_config=cfg)
        layer.eval()
        out = layer((gnn_features, rnn_features, sparse_adj))
        assert out.shape == (BATCH, NUM_TRADES, 64)

    def test_sparse_return_attention(self, gnn_features, rnn_features, sparse_adj):
        """Sparse return_attention should return output and [B, H, T, k] weights."""
        cfg = _make_fusion_config(fusion_mode="gate", units=64, num_heads=2)
        layer = FusionLayer(layer_config=cfg)
        layer.eval()
        out, attn = layer(
            (gnn_features, rnn_features, sparse_adj), return_attention=True
        )
        assert out.shape == (BATCH, NUM_TRADES, 64)
        # Sparse weights have shape [B, H, T, k] where k <= k_nbrs
        assert attn.dim() == 4
        assert attn.shape[0] == BATCH
        assert attn.shape[1] == 2   # num_heads
        assert attn.shape[2] == NUM_TRADES


class TestFusionLayerConfig:
    """Tests for configuration serialization."""

    def test_get_config(self):
        """get_config should return a dict containing the original layer_config."""
        cfg = _make_fusion_config()
        layer = FusionLayer(layer_config=cfg)
        config = layer.get_config()
        assert "layer_config" in config
        assert config["layer_config"] == cfg

    def test_units_not_divisible(self):
        """units=63 with num_heads=4 should fail at construction."""
        with pytest.raises(AssertionError, match="divisible"):
            FusionLayer(layer_config=_make_fusion_config(units=63, num_heads=4))

    def test_from_config_roundtrip(self):
        """from_config(get_config()) should produce an equivalent FusionLayer."""
        cfg = _make_fusion_config(fusion_mode="add", units=32, num_heads=2)
        original = FusionLayer(layer_config=cfg)
        rebuilt = FusionLayer.from_config(original.get_config())
        assert rebuilt.layer_config == original.layer_config
        assert rebuilt.units == original.units
        assert rebuilt.num_heads == original.num_heads

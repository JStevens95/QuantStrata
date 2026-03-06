"""Unit tests for rade_ml_pt.models.hybrid_gnn_rnn.layers.attention_layer."""
import numpy as np
import pytest
import torch

from src.rade_ml_pt.models.hybrid_gnn_rnn.layers.attention_layer import TargetAttentionLayer


BATCH = 4
NUM_TRADES = 20
NUM_TARGETS = 8
FUSED_DIM = 64


def _make_attn_config(units=32, num_heads=1, dropout=0.0, k_nbrs=10):
    """Return a minimal TargetAttentionLayer configuration dictionary."""
    return {
        "general": {
            "layer_type": "standard",
            "use_residual": True,
            "use_layer_norm": True,
            "attention_mode": True,
            "num_heads": num_heads,
            "dropout_rate": dropout,
            "k_nbrs": k_nbrs,
        },
        "parameters": {
            "units": units,
            "activation": "tanh",
            "kernel_initializer": "he_uniform",
            "bias_initializer": "zeros",
        },
    }


@pytest.fixture
def target_idx():
    """Target node indices: first NUM_TARGETS nodes."""
    return torch.arange(NUM_TARGETS, dtype=torch.long)


@pytest.fixture
def fused_features():
    """Random fused features [BATCH, NUM_TRADES, FUSED_DIM]."""
    torch.manual_seed(0)
    return torch.randn(BATCH, NUM_TRADES, FUSED_DIM)


@pytest.fixture
def adjacency():
    """Row-normalised dense adjacency [NUM_TRADES, NUM_TRADES] with self-loops."""
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


# ======================================================================
# TestTargetAttentionLayerForward
# ======================================================================


class TestTargetAttentionLayerForward:
    """Tests for basic forward pass with dense adjacency."""

    def test_output_shape(self, fused_features, adjacency, target_idx):
        """Dense forward produces [B, n_tgt, units] output."""
        layer = TargetAttentionLayer(layer_config=_make_attn_config(units=32))
        layer.eval()
        out = layer(fused_features, adjacency, target_idx)
        assert out.shape == (BATCH, NUM_TARGETS, 32)

    def test_no_nan_output(self, fused_features, adjacency, target_idx):
        """Output contains no NaN values."""
        layer = TargetAttentionLayer(layer_config=_make_attn_config())
        layer.eval()
        out = layer(fused_features, adjacency, target_idx)
        assert not torch.any(torch.isnan(out)).item()

    def test_training_mode(self, fused_features, adjacency, target_idx):
        """Forward works in training mode with dropout enabled."""
        cfg = _make_attn_config(dropout=0.1)
        layer = TargetAttentionLayer(layer_config=cfg)
        layer.train()
        out = layer(fused_features, adjacency, target_idx)
        assert out.shape == (BATCH, NUM_TARGETS, 32)


# ======================================================================
# TestTargetAttentionLayerMultiHead
# ======================================================================


class TestTargetAttentionLayerMultiHead:
    """Tests for multi-head attention."""

    def test_multi_head_output(self, fused_features, adjacency, target_idx):
        """Multi-head (4 heads) attention produces correct output shape."""
        cfg = _make_attn_config(units=32, num_heads=4)
        layer = TargetAttentionLayer(layer_config=cfg)
        layer.eval()
        out = layer(fused_features, adjacency, target_idx)
        assert out.shape == (BATCH, NUM_TARGETS, 32)


# ======================================================================
# TestTargetAttentionLayerSparse
# ======================================================================


class TestTargetAttentionLayerSparse:
    """Tests for sparse adjacency path (padded-neighbor attention)."""

    def test_sparse_adjacency_output_shape(self, fused_features, sparse_adj, target_idx):
        """Sparse forward produces [B, n_tgt, units] output."""
        layer = TargetAttentionLayer(layer_config=_make_attn_config())
        layer.eval()
        out = layer(fused_features, sparse_adj, target_idx)
        assert out.shape == (BATCH, NUM_TARGETS, 32)

    def test_sparse_no_nan(self, fused_features, sparse_adj, target_idx):
        """Sparse path output contains no NaN values."""
        layer = TargetAttentionLayer(layer_config=_make_attn_config())
        layer.eval()
        out = layer(fused_features, sparse_adj, target_idx)
        assert not torch.any(torch.isnan(out)).item()

    def test_sparse_vs_dense_equivalence(self, fused_features, adjacency, sparse_adj, target_idx):
        """Sparse and dense paths produce numerically equivalent output."""
        cfg = _make_attn_config(units=32, num_heads=1)
        layer = TargetAttentionLayer(layer_config=cfg)
        layer.eval()

        # Run dense path first to initialise lazy modules, then sparse.
        out_dense = layer(fused_features, adjacency, target_idx)
        out_sparse = layer(fused_features, sparse_adj, target_idx)

        torch.testing.assert_close(out_sparse, out_dense, atol=1e-5, rtol=1e-5)

    def test_sparse_multi_head(self, fused_features, sparse_adj, target_idx):
        """Sparse multi-head (4 heads) produces correct output shape."""
        cfg = _make_attn_config(units=32, num_heads=4)
        layer = TargetAttentionLayer(layer_config=cfg)
        layer.eval()
        out = layer(fused_features, sparse_adj, target_idx)
        assert out.shape == (BATCH, NUM_TARGETS, 32)

    def test_sparse_multi_head_equivalence(self, fused_features, adjacency, sparse_adj, target_idx):
        """Sparse and dense multi-head paths produce equivalent output."""
        cfg = _make_attn_config(units=32, num_heads=4)
        layer = TargetAttentionLayer(layer_config=cfg)
        layer.eval()

        out_dense = layer(fused_features, adjacency, target_idx)
        out_sparse = layer(fused_features, sparse_adj, target_idx)

        torch.testing.assert_close(out_sparse, out_dense, atol=1e-5, rtol=1e-5)

    def test_sparse_training_mode(self, fused_features, sparse_adj, target_idx):
        """Sparse path works in training mode with dropout enabled."""
        cfg = _make_attn_config(dropout=0.1)
        layer = TargetAttentionLayer(layer_config=cfg)
        layer.train()
        out = layer(fused_features, sparse_adj, target_idx)
        assert out.shape == (BATCH, NUM_TARGETS, 32)


# ======================================================================
# TestExtractTargetSubmatrix
# ======================================================================


class TestExtractTargetSubmatrix:
    """Tests for _extract_target_submatrix (sparse and dense paths)."""

    def test_sparse_input_returns_sparse(self, sparse_adj, target_idx):
        """Sparse adjacency input produces a sparse submatrix output."""
        layer = TargetAttentionLayer(layer_config=_make_attn_config())
        sub = layer._extract_target_submatrix(sparse_adj, target_idx)
        assert sub.is_sparse

    def test_sparse_submatrix_shape(self, sparse_adj, target_idx):
        """Sparse submatrix has shape [n_tgt, n_tgt]."""
        layer = TargetAttentionLayer(layer_config=_make_attn_config())
        sub = layer._extract_target_submatrix(sparse_adj, target_idx)
        assert sub.shape == (NUM_TARGETS, NUM_TARGETS)

    def test_sparse_submatrix_matches_dense(self, adjacency, sparse_adj, target_idx):
        """Sparse submatrix matches dense submatrix when converted to dense."""
        layer = TargetAttentionLayer(layer_config=_make_attn_config())
        sub_sparse = layer._extract_target_submatrix(sparse_adj, target_idx).to_dense()
        sub_dense = layer._extract_target_submatrix(adjacency, target_idx)
        torch.testing.assert_close(sub_sparse, sub_dense, atol=1e-6, rtol=1e-6)

    def test_dense_input_returns_dense(self, adjacency, target_idx):
        """Dense adjacency input produces a dense submatrix output."""
        layer = TargetAttentionLayer(layer_config=_make_attn_config())
        sub = layer._extract_target_submatrix(adjacency, target_idx)
        assert not sub.is_sparse

    def test_sparse_submatrix_is_row_major(self, sparse_adj, target_idx):
        """Sparse submatrix is coalesced (sorted in row-major order)."""
        layer = TargetAttentionLayer(layer_config=_make_attn_config())
        sub = layer._extract_target_submatrix(sparse_adj, target_idx)
        assert sub.is_coalesced()


# ======================================================================
# TestTargetAttentionLayerConfig
# ======================================================================


class TestTargetAttentionLayerConfig:
    """Tests for configuration, serialization, and validation."""

    def test_get_config(self):
        """get_config returns a dict containing the original layer_config."""
        cfg = _make_attn_config()
        layer = TargetAttentionLayer(layer_config=cfg, name="test_attn")
        config = layer.get_config()
        assert "layer_config" in config
        assert config["layer_config"] == cfg

    def test_units_not_divisible_by_heads_raises(self):
        """ValueError raised when units is not divisible by num_heads."""
        with pytest.raises(ValueError, match="divisible"):
            TargetAttentionLayer(layer_config=_make_attn_config(units=33, num_heads=4))

    def test_k_nbrs_default(self):
        """Default k_nbrs is 10 when not overridden."""
        cfg = _make_attn_config()
        layer = TargetAttentionLayer(layer_config=cfg)
        assert layer.k_nbrs == 10

    def test_k_nbrs_from_config(self):
        """k_nbrs matches the value provided in config."""
        cfg = _make_attn_config(k_nbrs=20)
        layer = TargetAttentionLayer(layer_config=cfg)
        assert layer.k_nbrs == 20

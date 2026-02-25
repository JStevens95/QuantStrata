"""Unit tests for rade_ml.models.hybrid_gnn_rnn.layers.attention_layer."""
import numpy as np
import pytest
import tensorflow as tf

from src.rade_ml.models.hybrid_gnn_rnn.layers.attention_layer import TargetAttentionLayer


BATCH = 4
NUM_TRADES = 20
NUM_TARGETS = 8
FUSED_DIM = 64


def _make_attn_config(units=32, num_heads=1, dropout=0.0, k_nbrs=10):
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
def sparse_adjacency(adjacency):
    return tf.sparse.from_dense(adjacency)


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
    """Tests for the O(n_tgt * k) sparse target attention path."""

    def test_sparse_adjacency_output_shape(self, fused_features, sparse_adjacency, target_idx):
        layer = TargetAttentionLayer(layer_config=_make_attn_config(), name="attn_sparse")
        out = layer((fused_features, sparse_adjacency, target_idx), training=False)
        assert out.shape == (BATCH, NUM_TARGETS, 32)

    def test_sparse_no_nan(self, fused_features, sparse_adjacency, target_idx):
        layer = TargetAttentionLayer(layer_config=_make_attn_config(), name="attn_sparse_nan")
        out = layer((fused_features, sparse_adjacency, target_idx), training=False)
        assert not tf.reduce_any(tf.math.is_nan(out)).numpy()

    def test_sparse_vs_dense_equivalence(self, fused_features, adjacency, target_idx):
        """Sparse and dense paths should produce numerically equivalent results."""
        sparse_adj = tf.sparse.from_dense(adjacency)

        cfg = _make_attn_config(dropout=0.0)
        layer = TargetAttentionLayer(layer_config=cfg, name="attn_equiv")

        out_dense = layer((fused_features, adjacency, target_idx), training=False)
        out_sparse = layer((fused_features, sparse_adj, target_idx), training=False)

        np.testing.assert_allclose(
            out_sparse.numpy(), out_dense.numpy(), atol=1e-5, rtol=1e-5,
            err_msg="Sparse and dense attention paths diverged"
        )

    def test_sparse_multi_head(self, fused_features, sparse_adjacency, target_idx):
        layer = TargetAttentionLayer(
            layer_config=_make_attn_config(units=32, num_heads=4),
            name="attn_sparse_mh"
        )
        out = layer((fused_features, sparse_adjacency, target_idx), training=False)
        assert out.shape == (BATCH, NUM_TARGETS, 32)

    def test_sparse_multi_head_equivalence(self, fused_features, adjacency, target_idx):
        """Multi-head sparse and dense must match."""
        sparse_adj = tf.sparse.from_dense(adjacency)

        cfg = _make_attn_config(units=32, num_heads=4, dropout=0.0)
        layer = TargetAttentionLayer(layer_config=cfg, name="attn_mh_equiv")

        out_dense = layer((fused_features, adjacency, target_idx), training=False)
        out_sparse = layer((fused_features, sparse_adj, target_idx), training=False)

        np.testing.assert_allclose(
            out_sparse.numpy(), out_dense.numpy(), atol=1e-5, rtol=1e-5,
            err_msg="Multi-head sparse and dense attention paths diverged"
        )

    def test_sparse_training_mode(self, fused_features, sparse_adjacency, target_idx):
        cfg = _make_attn_config(dropout=0.3)
        layer = TargetAttentionLayer(layer_config=cfg, name="attn_sparse_train")
        out = layer((fused_features, sparse_adjacency, target_idx), training=True)
        assert out.shape == (BATCH, NUM_TARGETS, 32)


class TestExtractTargetSubmatrix:
    """Tests for _extract_target_submatrix sparse-preserving behavior."""

    def test_sparse_input_returns_sparse(self, adjacency, target_idx):
        sparse_adj = tf.sparse.from_dense(adjacency)
        result = TargetAttentionLayer._extract_target_submatrix(sparse_adj, target_idx)
        assert isinstance(result, tf.SparseTensor), \
            f"Expected SparseTensor, got {type(result).__name__}"

    def test_sparse_submatrix_shape(self, adjacency, target_idx):
        sparse_adj = tf.sparse.from_dense(adjacency)
        result = TargetAttentionLayer._extract_target_submatrix(sparse_adj, target_idx)
        assert tuple(result.dense_shape.numpy()) == (NUM_TARGETS, NUM_TARGETS)

    def test_sparse_submatrix_matches_dense(self, adjacency, target_idx):
        """Sparse submatrix, when densified, should match the dense path."""
        sparse_adj = tf.sparse.from_dense(adjacency)

        sub_sparse = TargetAttentionLayer._extract_target_submatrix(sparse_adj, target_idx)
        sub_dense = TargetAttentionLayer._extract_target_submatrix(adjacency, target_idx)

        sub_sparse_dense = tf.sparse.to_dense(sub_sparse).numpy()
        sub_dense_binary = (sub_dense.numpy() > 0).astype(np.float32)

        np.testing.assert_array_equal(
            sub_sparse_dense, sub_dense_binary,
            err_msg="Sparse and dense submatrix extraction differ"
        )

    def test_dense_input_returns_dense(self, adjacency, target_idx):
        result = TargetAttentionLayer._extract_target_submatrix(adjacency, target_idx)
        assert isinstance(result, tf.Tensor)
        assert not isinstance(result, tf.SparseTensor)

    def test_sparse_submatrix_is_row_major(self, adjacency, target_idx):
        """Row ids in sparse submatrix must be non-decreasing (row-major order)."""
        sparse_adj = tf.sparse.from_dense(adjacency)
        result = TargetAttentionLayer._extract_target_submatrix(sparse_adj, target_idx)
        rows = result.indices[:, 0].numpy()
        assert np.all(rows[1:] >= rows[:-1]), "Sparse submatrix is not in row-major order"


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

    def test_k_nbrs_default(self):
        cfg = _make_attn_config()
        del cfg["general"]["k_nbrs"]
        layer = TargetAttentionLayer(layer_config=cfg, name="attn_k_default")
        assert layer.k_nbrs == 50

    def test_k_nbrs_from_config(self):
        layer = TargetAttentionLayer(layer_config=_make_attn_config(k_nbrs=25), name="attn_k_cfg")
        assert layer.k_nbrs == 25

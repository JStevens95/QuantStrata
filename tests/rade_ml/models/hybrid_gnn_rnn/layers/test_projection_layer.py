"""Unit tests for rade_ml.models.hybrid_gnn_rnn.layers.projection_layer."""
import numpy as np
import pytest
import tensorflow as tf

from src.rade_ml.models.hybrid_gnn_rnn.layers.projection_layer import TargetPnlOutput


BATCH = 4
NUM_TRADES = 20
NUM_TARGETS = 8
ATTN_DIM = 32
FEATURE_DIM = 10


def _make_projection_config(
    baseline_new_mode="output_mix",
    use_baseline_weight_norm=True,
    use_attn_scale=False,
    use_attn_bias=False,
    knn_mode="cosine_softmax",
):
    return {
        "general": {
            "dropout_rate": 0.0,
            "baseline_new_mode": baseline_new_mode,
            "use_baseline_weight_norm": use_baseline_weight_norm,
            "use_attn_scale_new": use_attn_scale,
            "use_attn_bias_new": use_attn_bias,
            "knn_k": 5,
            "knn_mode": knn_mode,
            "knn_temperature": 5.0,
            "knn_power": 2.0,
            "residual_new_damp": 1.0,
        },
        "parameters": {
            "units": 32,
            "activation": "gelu",
            "kernel_initializer": "glorot_uniform",
            "bias_initializer": "zeros",
        }
    }


@pytest.fixture
def trade_features():
    np.random.seed(42)
    return tf.constant(np.random.randn(NUM_TRADES, FEATURE_DIM).astype(np.float32))


@pytest.fixture
def attn_features():
    np.random.seed(43)
    return tf.constant(np.random.randn(BATCH, NUM_TARGETS, ATTN_DIM).astype(np.float32))


@pytest.fixture
def target_idx():
    return tf.constant(list(range(NUM_TARGETS)), dtype=tf.int32)


class TestTargetPnlOutputForward:
    def test_output_shape(self, trade_features, attn_features, target_idx):
        layer = TargetPnlOutput(layer_config=_make_projection_config(), name="proj_test")
        out = layer((trade_features, attn_features, target_idx), training=False)
        assert out.shape == (BATCH, NUM_TARGETS)

    def test_no_nan_output(self, trade_features, attn_features, target_idx):
        layer = TargetPnlOutput(layer_config=_make_projection_config(), name="proj_nan")
        out = layer((trade_features, attn_features, target_idx), training=False)
        assert not tf.reduce_any(tf.math.is_nan(out)).numpy()

    def test_training_mode(self, trade_features, attn_features, target_idx):
        layer = TargetPnlOutput(layer_config=_make_projection_config(), name="proj_train")
        out_train = layer((trade_features, attn_features, target_idx), training=True)
        out_infer = layer((trade_features, attn_features, target_idx), training=False)
        assert out_train.shape == out_infer.shape


class TestTargetPnlOutputBaselineNorm:
    def test_with_baseline_norm(self, trade_features, attn_features, target_idx):
        layer = TargetPnlOutput(
            layer_config=_make_projection_config(use_baseline_weight_norm=True),
            name="proj_bnorm"
        )
        out = layer((trade_features, attn_features, target_idx), training=False)
        assert out.shape == (BATCH, NUM_TARGETS)

    def test_without_baseline_norm(self, trade_features, attn_features, target_idx):
        layer = TargetPnlOutput(
            layer_config=_make_projection_config(use_baseline_weight_norm=False),
            name="proj_no_bnorm"
        )
        out = layer((trade_features, attn_features, target_idx), training=False)
        assert out.shape == (BATCH, NUM_TARGETS)


class TestTargetPnlOutputKnnModes:
    def test_cosine_softmax(self, trade_features, attn_features, target_idx):
        layer = TargetPnlOutput(
            layer_config=_make_projection_config(knn_mode="cosine_softmax"),
            name="proj_cos"
        )
        out = layer((trade_features, attn_features, target_idx))
        assert out.shape == (BATCH, NUM_TARGETS)

    def test_idw(self, trade_features, attn_features, target_idx):
        layer = TargetPnlOutput(
            layer_config=_make_projection_config(knn_mode="idw"),
            name="proj_idw"
        )
        out = layer((trade_features, attn_features, target_idx))
        assert out.shape == (BATCH, NUM_TARGETS)

    def test_invalid_knn_mode_raises(self, trade_features, attn_features, target_idx):
        layer = TargetPnlOutput(
            layer_config=_make_projection_config(knn_mode="invalid"),
            name="proj_knn_bad"
        )
        with pytest.raises(ValueError, match="Unknown knn_mode"):
            layer((trade_features, attn_features, target_idx))


class TestTargetPnlOutputAttentionConditioning:
    def test_attn_scale_enabled(self, trade_features, attn_features, target_idx):
        layer = TargetPnlOutput(
            layer_config=_make_projection_config(use_attn_scale=True),
            name="proj_scale"
        )
        out = layer((trade_features, attn_features, target_idx))
        assert out.shape == (BATCH, NUM_TARGETS)

    def test_attn_bias_enabled(self, trade_features, attn_features, target_idx):
        layer = TargetPnlOutput(
            layer_config=_make_projection_config(use_attn_bias=True),
            name="proj_bias"
        )
        out = layer((trade_features, attn_features, target_idx))
        assert out.shape == (BATCH, NUM_TARGETS)


class TestTargetPnlOutputConfig:
    def test_get_config(self):
        layer = TargetPnlOutput(layer_config=_make_projection_config(), name="proj_cfg")
        config = layer.get_config()
        assert "layer_config" in config

    def test_compute_output_shape(self):
        layer = TargetPnlOutput(layer_config=_make_projection_config(), name="proj_oshape")
        shape = layer.compute_output_shape((
            tf.TensorShape([NUM_TRADES, FEATURE_DIM]),
            tf.TensorShape([BATCH, NUM_TARGETS, ATTN_DIM]),
            tf.TensorShape([NUM_TARGETS]),
        ))
        assert shape.as_list() == [BATCH, NUM_TARGETS]

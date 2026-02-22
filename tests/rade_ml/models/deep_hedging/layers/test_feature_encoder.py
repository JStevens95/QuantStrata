"""Unit tests for rade_ml.models.deep_hedging.layers.feature_encoder -- GatedResidualNetwork."""
import numpy as np
import pytest
import tensorflow as tf

from src.rade_ml.models.deep_hedging.layers.feature_encoder import GatedResidualNetwork


BATCH = 8
FEATURE_DIM = 5


@pytest.fixture
def default_config():
    return {
        "units": 32,
        "dropout_rate": 0.1,
        "activation": "elu",
        "kernel_initializer": "glorot_uniform",
        "bias_initializer": "zeros",
    }


@pytest.fixture
def encoder(default_config):
    return GatedResidualNetwork(layer_config=default_config, name="test_grn")


@pytest.fixture
def sample_input():
    np.random.seed(42)
    return tf.constant(np.random.randn(BATCH, FEATURE_DIM).astype(np.float32))


class TestGRNForward:
    def test_output_shape(self, encoder, sample_input):
        out = encoder(sample_input, training=False)
        assert out.shape == (BATCH, 32)

    def test_no_nan(self, encoder, sample_input):
        out = encoder(sample_input, training=False)
        assert not tf.reduce_any(tf.math.is_nan(out)).numpy()

    def test_no_inf(self, encoder, sample_input):
        out = encoder(sample_input, training=False)
        assert not tf.reduce_any(tf.math.is_inf(out)).numpy()

    def test_deterministic_eval(self, encoder, sample_input):
        out1 = encoder(sample_input, training=False)
        out2 = encoder(sample_input, training=False)
        np.testing.assert_allclose(out1.numpy(), out2.numpy(), rtol=1e-5)

    def test_training_mode(self, encoder, sample_input):
        out = encoder(sample_input, training=True)
        assert out.shape == (BATCH, 32)
        assert not tf.reduce_any(tf.math.is_nan(out)).numpy()


class TestGRNGating:
    def test_gate_values_bounded(self, encoder, sample_input):
        """Gate sigmoid output should be in [0, 1]."""
        skip = encoder.skip_proj(sample_input)
        h = encoder.dense_primary(sample_input)
        h = encoder.dense_hidden(h)
        gate_input = tf.concat([h, skip], axis=-1)
        gate = encoder.gate_dense(gate_input)
        assert tf.reduce_all(gate >= 0.0).numpy()
        assert tf.reduce_all(gate <= 1.0).numpy()

    def test_skip_projection_matches_units(self, encoder, sample_input):
        skip = encoder.skip_proj(sample_input)
        assert skip.shape[-1] == encoder.units


class TestGRNConfig:
    def test_get_config(self, encoder, default_config):
        cfg = encoder.get_config()
        assert "layer_config" in cfg
        assert cfg["layer_config"] == default_config

    def test_custom_units(self):
        enc = GatedResidualNetwork(layer_config={"units": 128}, name="large_grn")
        assert enc.units == 128

    def test_no_dropout(self):
        enc = GatedResidualNetwork(layer_config={"units": 32, "dropout_rate": 0.0}, name="no_drop")
        assert enc.dropout is None


class TestGRNGradients:
    def test_gradients_flow(self, encoder, sample_input):
        with tf.GradientTape() as tape:
            out = encoder(sample_input, training=True)
            loss = tf.reduce_mean(out)
        grads = tape.gradient(loss, encoder.trainable_variables)
        for v, g in zip(encoder.trainable_variables, grads):
            assert g is not None, f"No gradient for {v.name}"

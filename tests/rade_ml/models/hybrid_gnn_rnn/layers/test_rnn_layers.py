"""Unit tests for rade_ml.models.hybrid_gnn_rnn.layers.rnn_layers."""
import numpy as np
import pytest
import tensorflow as tf

from src.rade_ml.models.hybrid_gnn_rnn.layers.rnn_layers import RnnBlock


BATCH = 4
SEQUENCE_LEN = 15
NUM_ELEM = 12


@pytest.fixture
def pnl_input():
    np.random.seed(42)
    return tf.constant(np.random.randn(BATCH, SEQUENCE_LEN, NUM_ELEM).astype(np.float32))


def _make_rnn_config(layer_type="lstm", layers=2, units=64):
    return {
        "general": {
            "architecture": "default",
            "layers": layers,
            "layer_type": layer_type,
            "dropout_rate": 0.0,
            "use_bias": True,
        },
        "parameters": {
            "units": units,
            "activation": "relu",
            "recurrent_activation": "sigmoid",
            "kernel_initializer": "glorot_uniform",
            "recurrent_initializer": "orthogonal",
            "bias_initializer": "zeros",
        }
    }


class TestRnnBlockLSTM:
    def test_output_shape(self, pnl_input):
        block = RnnBlock(layer_config=_make_rnn_config("lstm", units=64), name="rnn_lstm")
        out = block(pnl_input, training=False)
        assert out.shape == (BATCH, 64)

    def test_no_nan_output(self, pnl_input):
        block = RnnBlock(layer_config=_make_rnn_config("lstm"), name="rnn_lstm_nan")
        out = block(pnl_input, training=False)
        assert not tf.reduce_any(tf.math.is_nan(out)).numpy()


class TestRnnBlockBiLSTM:
    def test_output_shape(self, pnl_input):
        block = RnnBlock(layer_config=_make_rnn_config("bilstm", units=64), name="rnn_bilstm")
        out = block(pnl_input, training=False)
        assert out.shape == (BATCH, 128)  # bidirectional doubles units


class TestRnnBlockGRU:
    def test_output_shape(self, pnl_input):
        block = RnnBlock(layer_config=_make_rnn_config("gru", units=64), name="rnn_gru")
        out = block(pnl_input, training=False)
        assert out.shape == (BATCH, 64)


class TestRnnBlockInvalidType:
    def test_invalid_layer_type_raises(self, pnl_input):
        block = RnnBlock(layer_config=_make_rnn_config("invalid"), name="rnn_bad")
        with pytest.raises(ValueError, match="Undefined layer type"):
            block(pnl_input)


class TestRnnBlockConfig:
    def test_get_config(self):
        cfg = _make_rnn_config()
        block = RnnBlock(layer_config=cfg, name="rnn_cfg")
        config = block.get_config()
        assert "layer_config" in config

    def test_compute_output_shape_lstm(self):
        block = RnnBlock(layer_config=_make_rnn_config("lstm", units=32), name="rnn_shape_lstm")
        shape = block.compute_output_shape(tf.TensorShape([BATCH, SEQUENCE_LEN, NUM_ELEM]))
        assert shape.as_list() == [BATCH, 32]

    def test_compute_output_shape_bilstm(self):
        block = RnnBlock(layer_config=_make_rnn_config("bilstm", units=32), name="rnn_shape_bilstm")
        shape = block.compute_output_shape(tf.TensorShape([BATCH, SEQUENCE_LEN, NUM_ELEM]))
        assert shape.as_list() == [BATCH, 64]


class TestRnnBlockTraining:
    def test_training_dropout(self, pnl_input):
        cfg = _make_rnn_config("lstm", units=32)
        cfg["general"]["dropout_rate"] = 0.5
        block = RnnBlock(layer_config=cfg, name="rnn_drop")
        out_train = block(pnl_input, training=True)
        out_infer = block(pnl_input, training=False)
        assert out_train.shape == out_infer.shape

    def test_single_layer(self, pnl_input):
        cfg = _make_rnn_config("lstm", layers=1, units=32)
        block = RnnBlock(layer_config=cfg, name="rnn_single")
        out = block(pnl_input, training=False)
        assert out.shape == (BATCH, 32)

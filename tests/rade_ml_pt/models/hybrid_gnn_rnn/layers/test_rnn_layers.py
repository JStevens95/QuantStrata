"""Unit tests for rade_ml_pt.models.hybrid_gnn_rnn.layers.rnn_layers."""
import numpy as np
import pytest
import torch

from src.rade_ml_pt.models.hybrid_gnn_rnn.layers.rnn_layers import RnnBlock


BATCH = 4
SEQUENCE_LEN = 15
NUM_ELEM = 12


@pytest.fixture
def pnl_input():
    """Create a deterministic random P&L input tensor [B, S, T_e]."""
    np.random.seed(42)
    return torch.tensor(np.random.randn(BATCH, SEQUENCE_LEN, NUM_ELEM).astype(np.float32))


def _make_rnn_config(layer_type="lstm", layers=2, units=64, activation="tanh",
                     recurrent_activation="sigmoid"):
    """Return a minimal RnnBlock configuration dictionary."""
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
            "activation": activation,
            "recurrent_activation": recurrent_activation,
            "kernel_initializer": "glorot_uniform",
            "recurrent_initializer": "orthogonal",
            "bias_initializer": "zeros",
        },
    }


class TestRnnBlockLSTM:
    """Tests for the standard LSTM variant of RnnBlock."""

    def test_output_shape(self, pnl_input):
        """LSTM should produce output shape [B, units]."""
        block = RnnBlock(layer_config=_make_rnn_config("lstm", units=64), name="test_lstm")
        block.eval()
        out = block(pnl_input)
        assert out.shape == (BATCH, 64)

    def test_no_nan_output(self, pnl_input):
        """LSTM output should not contain NaN values."""
        block = RnnBlock(layer_config=_make_rnn_config("lstm"), name="test_lstm_nan")
        block.eval()
        out = block(pnl_input)
        assert not torch.any(torch.isnan(out)).item()


class TestRnnBlockBiLSTM:
    """Tests for the bidirectional LSTM variant of RnnBlock."""

    def test_output_shape(self, pnl_input):
        """BiLSTM should double the output dimension to [B, 2*units]."""
        block = RnnBlock(layer_config=_make_rnn_config("bilstm", units=64), name="test_bilstm")
        block.eval()
        out = block(pnl_input)
        assert out.shape == (BATCH, 128)


class TestRnnBlockGRU:
    """Tests for the GRU variant of RnnBlock."""

    def test_output_shape(self, pnl_input):
        """GRU should produce output shape [B, units]."""
        block = RnnBlock(layer_config=_make_rnn_config("gru", units=64), name="test_gru")
        block.eval()
        out = block(pnl_input)
        assert out.shape == (BATCH, 64)


class TestRnnBlockDense:
    """Tests for the dense (linear) fallback variant of RnnBlock."""

    def test_output_shape(self, pnl_input):
        """Dense variant should produce output shape [B, units]."""
        block = RnnBlock(layer_config=_make_rnn_config("dense", layers=1, units=64),
                         name="test_dense")
        block.eval()
        out = block(pnl_input)
        assert out.shape == (BATCH, 64)

    def test_multi_layer_dense(self, pnl_input):
        """Multi-layer dense with activation should produce valid output."""
        block = RnnBlock(layer_config=_make_rnn_config("dense", layers=3, units=32,
                                                        activation="relu"),
                         name="test_dense_multi")
        block.eval()
        out = block(pnl_input)
        assert out.shape == (BATCH, 32)
        assert not torch.any(torch.isnan(out)).item()

    def test_dense_no_activation(self, pnl_input):
        """Dense with activation=None should produce a pure linear stack."""
        block = RnnBlock(layer_config=_make_rnn_config("dense", layers=2, units=32,
                                                        activation=None),
                         name="test_dense_linear")
        block.eval()
        out = block(pnl_input)
        assert out.shape == (BATCH, 32)


class TestRnnBlockInvalidType:
    """Tests for error handling on unsupported layer types."""

    def test_invalid_layer_type_raises(self, pnl_input):
        """An unrecognised layer_type should raise ValueError on forward()."""
        block = RnnBlock(layer_config=_make_rnn_config("invalid"), name="test_invalid")
        with pytest.raises(ValueError, match="Undefined layer type"):
            block(pnl_input)


class TestRnnBlockConfig:
    """Tests for configuration serialization and reconstruction."""

    def test_get_config(self):
        """get_config should return a dict containing the original layer_config and name."""
        cfg = _make_rnn_config()
        block = RnnBlock(layer_config=cfg, name="test_cfg")
        config = block.get_config()
        assert "layer_config" in config
        assert config["layer_config"] == cfg
        assert config["name"] == "test_cfg"

    def test_from_config_roundtrip(self):
        """from_config(get_config()) should produce an equivalent RnnBlock."""
        cfg = _make_rnn_config("gru", layers=3, units=32)
        original = RnnBlock(layer_config=cfg, name="test_roundtrip")
        rebuilt = RnnBlock.from_config(original.get_config())
        assert rebuilt.layer_config == original.layer_config
        assert rebuilt.layer_name == original.layer_name
        assert rebuilt.units == original.units
        assert rebuilt.num_layers == original.num_layers


class TestRnnBlockTraining:
    """Tests for training-mode behaviour (dropout, single-layer edge case)."""

    def test_training_dropout(self, pnl_input):
        """Training and eval output shapes should match even with non-zero dropout."""
        cfg = _make_rnn_config("lstm", units=32)
        cfg["general"]["dropout_rate"] = 0.5
        block = RnnBlock(layer_config=cfg, name="test_drop")

        block.train()
        out_train = block(pnl_input)

        block.eval()
        out_infer = block(pnl_input)

        assert out_train.shape == out_infer.shape

    def test_single_layer(self, pnl_input):
        """A single-layer LSTM should still produce the correct output shape."""
        cfg = _make_rnn_config("lstm", layers=1, units=32)
        block = RnnBlock(layer_config=cfg, name="test_single")
        block.eval()
        out = block(pnl_input)
        assert out.shape == (BATCH, 32)

    def test_gradient_flows(self, pnl_input):
        """Gradients should propagate back through the RNN block."""
        block = RnnBlock(layer_config=_make_rnn_config("lstm", units=16), name="test_grad")
        block.train()
        out = block(pnl_input)
        loss = out.sum()
        loss.backward()
        for name, param in block.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

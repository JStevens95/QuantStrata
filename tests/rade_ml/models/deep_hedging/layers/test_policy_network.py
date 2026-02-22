"""Unit tests for rade_ml.models.deep_hedging.layers.policy_network -- HedgingPolicy."""
import numpy as np
import pytest
import tensorflow as tf

from src.rade_ml.models.deep_hedging.layers.policy_network import HedgingPolicy


BATCH = 8
ENCODER_UNITS = 32
NUM_INSTRUMENTS = 1
TIMESTEPS = 10


@pytest.fixture
def gru_config():
    return {
        "rnn_type": "gru",
        "rnn_units": 64,
        "rnn_layers": 2,
        "dropout_rate": 0.0,
        "output_activation": None,
        "kernel_initializer": "glorot_uniform",
        "recurrent_initializer": "orthogonal",
        "bias_initializer": "zeros",
    }


@pytest.fixture
def lstm_config():
    return {
        "rnn_type": "lstm",
        "rnn_units": 64,
        "rnn_layers": 2,
        "dropout_rate": 0.0,
        "output_activation": None,
    }


@pytest.fixture
def policy(gru_config):
    return HedgingPolicy(layer_config=gru_config, num_instruments=NUM_INSTRUMENTS, name="test_policy")


@pytest.fixture
def step_input():
    np.random.seed(42)
    input_dim = ENCODER_UNITS + NUM_INSTRUMENTS
    return tf.constant(np.random.randn(BATCH, input_dim).astype(np.float32))


class TestPolicyStep:
    def test_step_output_shape(self, policy, step_input):
        states = policy.get_initial_state(BATCH)
        delta, new_states = policy.step(step_input, states, training=False)
        assert delta.shape == (BATCH, NUM_INSTRUMENTS)

    def test_step_returns_new_states(self, policy, step_input):
        states = policy.get_initial_state(BATCH)
        _, new_states = policy.step(step_input, states, training=False)
        assert len(new_states) == 2  # two GRU layers

    def test_step_no_nan(self, policy, step_input):
        states = policy.get_initial_state(BATCH)
        delta, _ = policy.step(step_input, states, training=False)
        assert not tf.reduce_any(tf.math.is_nan(delta)).numpy()

    def test_step_deterministic_eval(self, policy, step_input):
        states = policy.get_initial_state(BATCH)
        d1, _ = policy.step(step_input, states, training=False)
        states = policy.get_initial_state(BATCH)
        d2, _ = policy.step(step_input, states, training=False)
        np.testing.assert_allclose(d1.numpy(), d2.numpy(), rtol=1e-5)

    def test_multi_step_rollout(self, policy, step_input):
        """Verify that multiple steps can be chained without error."""
        states = policy.get_initial_state(BATCH)
        for _ in range(5):
            delta, states = policy.step(step_input, states, training=False)
        assert delta.shape == (BATCH, NUM_INSTRUMENTS)


class TestPolicyFullSequence:
    def test_call_output_shape(self, policy):
        np.random.seed(42)
        input_dim = ENCODER_UNITS + NUM_INSTRUMENTS
        seq = tf.constant(np.random.randn(BATCH, TIMESTEPS, input_dim).astype(np.float32))
        out = policy(seq, training=False)
        assert out.shape == (BATCH, TIMESTEPS, NUM_INSTRUMENTS)


class TestPolicyLSTM:
    def test_lstm_step(self, lstm_config):
        policy = HedgingPolicy(layer_config=lstm_config, num_instruments=1, name="lstm_policy")
        np.random.seed(42)
        input_dim = ENCODER_UNITS + 1
        x = tf.constant(np.random.randn(BATCH, input_dim).astype(np.float32))
        states = policy.get_initial_state(BATCH)
        delta, new_states = policy.step(x, states, training=False)
        assert delta.shape == (BATCH, 1)
        assert len(new_states) == 2


class TestPolicyInvalidRNN:
    def test_invalid_rnn_type_raises(self):
        config = {"rnn_type": "transformer", "rnn_units": 64, "rnn_layers": 1}
        with pytest.raises(ValueError, match="Unsupported rnn_type"):
            HedgingPolicy(layer_config=config, num_instruments=1, name="bad")


class TestPolicyMultiInstrument:
    def test_multi_instrument_output(self, gru_config):
        n_inst = 3
        policy = HedgingPolicy(layer_config=gru_config, num_instruments=n_inst, name="multi")
        np.random.seed(42)
        input_dim = ENCODER_UNITS + n_inst
        x = tf.constant(np.random.randn(BATCH, input_dim).astype(np.float32))
        states = policy.get_initial_state(BATCH)
        delta, _ = policy.step(x, states, training=False)
        assert delta.shape == (BATCH, n_inst)


class TestPolicyConfig:
    def test_get_config(self, policy, gru_config):
        cfg = policy.get_config()
        assert cfg["layer_config"] == gru_config
        assert cfg["num_instruments"] == NUM_INSTRUMENTS

    def test_initial_state_shapes(self, policy):
        states = policy.get_initial_state(BATCH)
        for state in states:
            # GRU returns [h], LSTM returns [h, c] - flatten for assertion
            tensors = state if isinstance(state, (list, tuple)) else [state]
            for t in tensors:
                assert t.shape[0] == BATCH

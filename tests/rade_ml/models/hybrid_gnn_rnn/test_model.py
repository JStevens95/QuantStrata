"""Unit tests for rade_ml.models.hybrid_gnn_rnn.model -- HybridGnnRnn."""
import numpy as np
import pytest
import tensorflow as tf

from src.rade_ml.models.hybrid_gnn_rnn.model import HybridGnnRnn
from src.rade_ml.models.hybrid_gnn_rnn.config import default_model_config
from src.rade_ml.core.base import BaseModel
from src.rade_ml.validation.exceptions import MissingKeyFields


BATCH = 4
NUM_TRADES = 20
NUM_TARGETS = 8
FEATURE_DIM = 10
SEQUENCE_LEN = 15
NUM_ELEM = 12


@pytest.fixture
def config():
    return default_model_config()


@pytest.fixture
def model(config):
    return HybridGnnRnn(config=config, name="test_hybrid")


@pytest.fixture
def inputs():
    np.random.seed(42)
    adj = np.random.rand(NUM_TRADES, NUM_TRADES).astype(np.float32)
    adj = (adj > 0.7).astype(np.float32)
    np.fill_diagonal(adj, 1.0)
    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    adj = adj / row_sums
    return {
        "trade_features": tf.constant(np.random.randn(NUM_TRADES, FEATURE_DIM).astype(np.float32)),
        "pnl_history": tf.constant(np.random.randn(BATCH, SEQUENCE_LEN, NUM_ELEM).astype(np.float32)),
        "adjacency_matrix": tf.constant(adj),
        "elementary_indices": tf.constant(list(range(NUM_ELEM)), dtype=tf.int32),
        "target_indices": tf.constant(list(range(NUM_TARGETS)), dtype=tf.int32),
    }


class TestHybridGnnRnnInheritance:
    def test_is_base_model(self, model):
        assert isinstance(model, BaseModel)

    def test_is_keras_model(self, model):
        assert isinstance(model, tf.keras.Model)


class TestHybridGnnRnnInit:
    def test_has_all_blocks(self, model):
        assert model.gnn_block is not None
        assert model.rnn_block is not None
        assert model.fusion_layer is not None
        assert model.attention_layer is not None
        assert model.projection_layer is not None

    def test_layer_norm_created(self, model):
        assert model.gnn_block_ln is not None
        assert model.rnn_block_ln is not None
        assert model.fusion_ln is not None

    def test_config_stored(self, model, config):
        assert model.model_config == config
        assert model.gnn_config == config["gnn_layer"]
        assert model.rnn_config == config["rnn_layer"]


class TestHybridGnnRnnForward:
    def test_output_shape(self, model, inputs):
        out = model(inputs, training=False)
        assert out.shape == (BATCH, NUM_TARGETS)

    def test_output_is_tensor(self, model, inputs):
        out = model(inputs, training=False)
        assert isinstance(out, tf.Tensor)

    def test_no_nan_in_output(self, model, inputs):
        out = model(inputs, training=False)
        assert not tf.reduce_any(tf.math.is_nan(out)).numpy()

    def test_no_inf_in_output(self, model, inputs):
        out = model(inputs, training=False)
        assert not tf.reduce_any(tf.math.is_inf(out)).numpy()

    def test_training_mode(self, model, inputs):
        out_train = model(inputs, training=True)
        out_infer = model(inputs, training=False)
        assert out_train.shape == out_infer.shape

    def test_deterministic_in_eval_mode(self, model, inputs):
        out1 = model(inputs, training=False)
        out2 = model(inputs, training=False)
        np.testing.assert_allclose(out1.numpy(), out2.numpy(), rtol=1e-5)


class TestHybridGnnRnnInputValidation:
    def test_missing_key_raises(self, model):
        incomplete_inputs = {"trade_features": tf.zeros((5, 10))}
        with pytest.raises(MissingKeyFields, match="Missing keys"):
            model(incomplete_inputs)

    def test_empty_dict_raises(self, model):
        with pytest.raises(MissingKeyFields):
            model({})


class TestHybridGnnRnnSparse:
    def test_sparse_adjacency(self, model, inputs):
        inputs["adjacency_matrix"] = tf.sparse.from_dense(inputs["adjacency_matrix"])
        out = model(inputs, training=False)
        assert out.shape == (BATCH, NUM_TARGETS)


class TestHybridGnnRnnMetadata:
    def test_metadata(self, model):
        meta = model.metadata
        assert meta["model_name"] == "test_hybrid"
        assert meta["model_class"] == "HybridGnnRnn"
        assert meta["framework"] == "tensorflow"


class TestHybridGnnRnnSerialisation:
    def test_get_config(self, model, config):
        cfg = model.get_config()
        assert "model_config" in cfg
        assert cfg["model_config"] == config

    def test_summary_dict_after_build(self, model, inputs):
        model(inputs, training=False)
        summary = model.summary_dict()
        assert summary["trainable_params"] > 0
        assert "layers" in summary


class TestHybridGnnRnnGradients:
    def test_gradients_flow(self, model, inputs):
        """Verify that gradients propagate through all layers."""
        targets = tf.random.normal((BATCH, NUM_TARGETS))
        with tf.GradientTape() as tape:
            preds = model(inputs, training=True)
            loss = tf.reduce_mean(tf.square(preds - targets))
        grads = tape.gradient(loss, model.trainable_variables)
        none_grads = [v.name for v, g in zip(model.trainable_variables, grads) if g is None]
        assert len(none_grads) == 0, f"No gradient for: {none_grads}"

    def test_gradients_are_finite(self, model, inputs):
        targets = tf.random.normal((BATCH, NUM_TARGETS))
        with tf.GradientTape() as tape:
            preds = model(inputs, training=True)
            loss = tf.reduce_mean(tf.square(preds - targets))
        grads = tape.gradient(loss, model.trainable_variables)
        for g in grads:
            if g is not None:
                assert not tf.reduce_any(tf.math.is_nan(g)).numpy(), "NaN gradient detected"
                assert not tf.reduce_any(tf.math.is_inf(g)).numpy(), "Inf gradient detected"


class TestHybridGnnRnnOutputShape:
    def test_compute_output_shape(self, model, inputs):
        shape_dict = {k: tf.TensorShape(v.shape) for k, v in inputs.items()}
        out_shape = model.compute_output_shape(shape_dict)
        assert out_shape.as_list() == [BATCH, NUM_TARGETS]


class TestHybridGnnRnnParamCount:
    def test_has_trainable_params(self, model, inputs):
        model(inputs, training=False)
        total_params = sum(tf.reduce_prod(w.shape).numpy() for w in model.trainable_weights)
        assert total_params > 0

    def test_param_count_reasonable(self, model, inputs):
        """Sanity check: a small config shouldn't produce millions of params."""
        model(inputs, training=False)
        total_params = sum(tf.reduce_prod(w.shape).numpy() for w in model.trainable_weights)
        assert total_params < 5_000_000

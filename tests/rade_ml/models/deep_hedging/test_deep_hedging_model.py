"""Unit tests for rade_ml.models.deep_hedging.model -- DeepHedgingModel."""
import numpy as np
import pytest
import tensorflow as tf

from src.rade_ml.models.deep_hedging.model import DeepHedgingModel
from src.rade_ml.models.deep_hedging.config import default_model_config
from src.rade_ml.core.base import BaseModel
from src.rade_ml.validation.exceptions import MissingKeyFields


BATCH = 8
NUM_STEPS = 20
NUM_FEATURES = 5


@pytest.fixture
def config():
    return default_model_config()


@pytest.fixture
def model(config):
    return DeepHedgingModel(config=config, name="test_deep_hedging")


@pytest.fixture
def inputs():
    np.random.seed(42)
    paths = np.abs(np.random.randn(BATCH, NUM_STEPS, NUM_FEATURES).astype(np.float32)) + 0.5
    payoffs = np.maximum(np.random.randn(BATCH).astype(np.float32) * 5, 0)
    return {
        "price_paths": tf.constant(paths),
        "payoffs": tf.constant(payoffs),
    }


class TestDeepHedgingInheritance:
    def test_is_base_model(self, model):
        assert isinstance(model, BaseModel)

    def test_is_keras_model(self, model):
        assert isinstance(model, tf.keras.Model)


class TestDeepHedgingInit:
    def test_has_strategy(self, model):
        assert model.strategy is not None

    def test_config_stored(self, model, config):
        assert model.model_config == config
        assert model.encoder_config == config["encoder"]
        assert model.policy_config == config["policy"]

    def test_general_config_parsed(self, model, config):
        assert model.general_config == config["general"]


class TestDeepHedgingForward:
    def test_output_shape(self, model, inputs):
        out = model(inputs, training=False)
        assert out.shape == (BATCH,)

    def test_output_is_tensor(self, model, inputs):
        out = model(inputs, training=False)
        assert isinstance(out, tf.Tensor)

    def test_no_nan_in_output(self, model, inputs):
        out = model(inputs, training=False)
        assert not tf.reduce_any(tf.math.is_nan(out)).numpy()

    def test_no_inf_in_output(self, model, inputs):
        out = model(inputs, training=False)
        assert not tf.reduce_any(tf.math.is_inf(out)).numpy()

    def test_training_mode_same_shape(self, model, inputs):
        out_train = model(inputs, training=True)
        out_infer = model(inputs, training=False)
        assert out_train.shape == out_infer.shape

    def test_deterministic_in_eval_mode(self, model, inputs):
        out1 = model(inputs, training=False)
        out2 = model(inputs, training=False)
        np.testing.assert_allclose(out1.numpy(), out2.numpy(), rtol=1e-5)


class TestDeepHedgingInputValidation:
    def test_missing_key_raises(self, model):
        with pytest.raises(MissingKeyFields, match="Missing keys"):
            model({"price_paths": tf.zeros((2, 10, 5))})

    def test_empty_dict_raises(self, model):
        with pytest.raises(MissingKeyFields):
            model({})


class TestDeepHedgingMetadata:
    def test_metadata_name(self, model):
        meta = model.metadata
        assert meta["model_name"] == "test_deep_hedging"

    def test_metadata_class(self, model):
        assert model.metadata["model_class"] == "DeepHedgingModel"

    def test_metadata_framework(self, model):
        assert model.metadata["framework"] == "tensorflow"


class TestDeepHedgingSerialisation:
    def test_get_config_has_model_config(self, model, config):
        cfg = model.get_config()
        assert "model_config" in cfg
        assert cfg["model_config"] == config

    def test_summary_dict_after_build(self, model, inputs):
        model(inputs, training=False)
        summary = model.summary_dict()
        assert summary["trainable_params"] > 0
        assert "layers" in summary


class TestDeepHedgingGradients:
    def test_gradients_flow(self, model, inputs):
        targets = tf.zeros((BATCH,))
        with tf.GradientTape() as tape:
            preds = model(inputs, training=True)
            loss = tf.reduce_mean(tf.square(preds - targets))
        grads = tape.gradient(loss, model.trainable_variables)
        none_grads = [v.name for v, g in zip(model.trainable_variables, grads) if g is None]
        assert len(none_grads) == 0, f"No gradient for: {none_grads}"

    def test_gradients_are_finite(self, model, inputs):
        targets = tf.zeros((BATCH,))
        with tf.GradientTape() as tape:
            preds = model(inputs, training=True)
            loss = tf.reduce_mean(tf.square(preds - targets))
        grads = tape.gradient(loss, model.trainable_variables)
        for g in grads:
            if g is not None:
                assert not tf.reduce_any(tf.math.is_nan(g)).numpy()
                assert not tf.reduce_any(tf.math.is_inf(g)).numpy()


class TestDeepHedgingPositionLimit:
    def test_with_position_limit(self, config, inputs):
        config["general"]["position_limit"] = 1.0
        model = DeepHedgingModel(config=config, name="limited_model")
        out = model(inputs, training=False)
        assert out.shape == (BATCH,)
        assert not tf.reduce_any(tf.math.is_nan(out)).numpy()

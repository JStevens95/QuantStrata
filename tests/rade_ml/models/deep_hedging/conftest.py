"""Shared fixtures for Deep Hedging model tests.

Generates synthetic market data that mirrors the shapes expected by the model
and its sub-layers: feature tensors, payoffs, and model inputs dictionary.
"""
import numpy as np
import pytest
import tensorflow as tf

from src.rade_ml.models.deep_hedging.config import default_model_config


BATCH_SIZE = 8
NUM_STEPS = 20
NUM_FEATURES = 5
NUM_INSTRUMENTS = 1


@pytest.fixture
def model_config():
    return default_model_config()


@pytest.fixture
def encoder_config(model_config):
    return model_config["encoder"]


@pytest.fixture
def policy_config(model_config):
    return model_config["policy"]


@pytest.fixture
def general_config(model_config):
    return model_config["general"]


@pytest.fixture
def price_paths():
    """Synthetic feature tensor: [batch, timesteps, num_features]."""
    np.random.seed(42)
    paths = np.random.randn(BATCH_SIZE, NUM_STEPS, NUM_FEATURES).astype(np.float32)
    paths[:, :, 0] = np.abs(paths[:, :, 0]) + 0.5  # spot must be positive
    return tf.constant(paths)


@pytest.fixture
def payoffs():
    """Synthetic option payoffs: [batch]."""
    np.random.seed(42)
    return tf.constant(np.maximum(np.random.randn(BATCH_SIZE) * 5, 0).astype(np.float32))


@pytest.fixture
def model_inputs(price_paths, payoffs):
    """Full dictionary input for DeepHedgingModel."""
    return {
        "price_paths": price_paths,
        "payoffs": payoffs,
    }


@pytest.fixture
def single_step_features():
    """Single-timestep feature vector: [batch, num_features]."""
    np.random.seed(42)
    return tf.constant(np.random.randn(BATCH_SIZE, NUM_FEATURES).astype(np.float32))

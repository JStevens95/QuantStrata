"""Shared fixtures for hybrid GNN-RNN model tests.

Generates synthetic graph-structured data that mirrors the shapes expected
by the model and its sub-layers.
"""
import numpy as np
import pytest
import tensorflow as tf

from src.rade_ml.models.hybrid_gnn_rnn.config import default_model_config


NUM_TRADES = 20
NUM_ELEM_TRADES = 12
NUM_TARGET_TRADES = 8
FEATURE_DIM = 10
BATCH_SIZE = 4
SEQUENCE_LEN = 15


@pytest.fixture
def model_config():
    return default_model_config()


@pytest.fixture
def gnn_config(model_config):
    return model_config["gnn_layer"]


@pytest.fixture
def rnn_config(model_config):
    return model_config["rnn_layer"]


@pytest.fixture
def fusion_config(model_config):
    return model_config["fusion_layer"]


@pytest.fixture
def attention_config(model_config):
    return model_config["attention_layer"]


@pytest.fixture
def projection_config(model_config):
    return model_config["projection_layer"]


@pytest.fixture
def trade_features():
    """Static trade features: [num_trades, feature_dim]."""
    np.random.seed(42)
    return tf.constant(np.random.randn(NUM_TRADES, FEATURE_DIM).astype(np.float32))


@pytest.fixture
def adjacency_matrix():
    """Row-normalised adjacency matrix: [num_trades, num_trades]."""
    np.random.seed(42)
    raw = np.random.rand(NUM_TRADES, NUM_TRADES).astype(np.float32)
    raw = (raw > 0.7).astype(np.float32)
    np.fill_diagonal(raw, 1.0)
    row_sums = raw.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    normalised = raw / row_sums
    return tf.constant(normalised)


@pytest.fixture
def sparse_adjacency(adjacency_matrix):
    """Sparse version of the adjacency matrix."""
    return tf.sparse.from_dense(adjacency_matrix)


@pytest.fixture
def pnl_history():
    """PnL history: [batch, sequence_len, num_elem_trades]."""
    np.random.seed(42)
    return tf.constant(np.random.randn(BATCH_SIZE, SEQUENCE_LEN, NUM_ELEM_TRADES).astype(np.float32))


@pytest.fixture
def elementary_indices():
    """Indices of elementary trades: [num_elem_trades]."""
    return tf.constant(list(range(NUM_ELEM_TRADES)), dtype=tf.int32)


@pytest.fixture
def target_indices():
    """Indices of target trades: [num_target_trades]."""
    return tf.constant(list(range(NUM_TARGET_TRADES)), dtype=tf.int32)


@pytest.fixture
def model_inputs(trade_features, pnl_history, adjacency_matrix, elementary_indices, target_indices):
    """Full dictionary input for the HybridGnnRnn model."""
    return {
        "trade_features": trade_features,
        "pnl_history": pnl_history,
        "adjacency_matrix": adjacency_matrix,
        "elementary_indices": elementary_indices,
        "target_indices": target_indices,
    }

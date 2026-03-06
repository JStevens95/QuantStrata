"""Shared fixtures for hybrid GNN-RNN model tests."""
import numpy as np
import pytest
import torch

from src.rade_ml_pt.models.hybrid_gnn_rnn.config import default_model_config

# Test dimensionality constants
NUM_TRADES = 20
NUM_ELEM_TRADES = 12
NUM_TARGET_TRADES = 8
FEATURE_DIM = 10
BATCH_SIZE = 4
SEQUENCE_LEN = 15


@pytest.fixture
def model_config():
    """Full default model configuration dict."""
    return default_model_config()


@pytest.fixture
def gnn_config(model_config):
    """GNN layer sub-config extracted from model config."""
    return model_config["gnn_layer"]


@pytest.fixture
def rnn_config(model_config):
    """RNN layer sub-config extracted from model config."""
    return model_config["rnn_layer"]


@pytest.fixture
def fusion_config(model_config):
    """Fusion layer sub-config extracted from model config."""
    return model_config["fusion_layer"]


@pytest.fixture
def attention_config(model_config):
    """Attention layer sub-config extracted from model config."""
    return model_config["attention_layer"]


@pytest.fixture
def projection_config(model_config):
    """Projection layer sub-config extracted from model config."""
    return model_config["projection_layer"]


@pytest.fixture
def trade_features():
    """Random float32 node features [NUM_TRADES, FEATURE_DIM]."""
    np.random.seed(42)
    return torch.from_numpy(np.random.randn(NUM_TRADES, FEATURE_DIM).astype(np.float32))


@pytest.fixture
def adjacency_matrix():
    """Row-normalised dense adjacency matrix [NUM_TRADES, NUM_TRADES] with self-loops."""
    np.random.seed(42)
    raw = np.random.rand(NUM_TRADES, NUM_TRADES).astype(np.float32)
    # Threshold to create a sparse binary connectivity pattern
    raw = (raw > 0.7).astype(np.float32)
    # Ensure self-loops are present
    np.fill_diagonal(raw, 1.0)
    # Row-normalise so each node's incoming weights sum to 1
    row_sums = raw.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    normalised = raw / row_sums
    return torch.from_numpy(normalised)


@pytest.fixture
def sparse_adjacency(adjacency_matrix):
    """Sparse COO version of the dense adjacency fixture."""
    return adjacency_matrix.to_sparse()


@pytest.fixture
def pnl_history():
    """Random PnL history tensor [BATCH_SIZE, SEQUENCE_LEN, NUM_ELEM_TRADES]."""
    np.random.seed(42)
    return torch.from_numpy(
        np.random.randn(BATCH_SIZE, SEQUENCE_LEN, NUM_ELEM_TRADES).astype(np.float32)
    )


@pytest.fixture
def elementary_indices():
    """Index tensor identifying elementary trades [NUM_ELEM_TRADES]."""
    return torch.arange(NUM_ELEM_TRADES, dtype=torch.int32)


@pytest.fixture
def target_indices():
    """Index tensor identifying target trades [NUM_TARGET_TRADES]."""
    return torch.arange(NUM_TARGET_TRADES, dtype=torch.int32)

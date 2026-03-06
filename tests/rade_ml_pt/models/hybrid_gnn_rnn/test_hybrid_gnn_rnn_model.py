"""Unit tests for rade_ml_pt.models.hybrid_gnn_rnn.model.HybridGnnRnn."""
import numpy as np
import pytest
import torch

from src.rade_ml_pt.models.hybrid_gnn_rnn.model import HybridGnnRnn
from src.rade_ml_pt.models.hybrid_gnn_rnn.config import default_model_config
from src.rade_ml_pt.core.base import BaseModel
from src.rade_ml_pt.validation.exceptions import MissingKeyFields

# Test dimensionality constants
BATCH = 4
NUM_TRADES = 20
NUM_TARGETS = 8
FEATURE_DIM = 10
SEQUENCE_LEN = 15
NUM_ELEM = 12


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config():
    """Model config with projection layer overrides needed for weight creation."""
    cfg = default_model_config()
    # Projection layer needs baseline_trade_count and attn_dim for weight creation
    cfg["projection_layer"]["general"]["baseline_trade_count"] = NUM_TARGETS
    cfg["projection_layer"]["general"]["attn_dim"] = cfg["attention_layer"]["parameters"]["units"]
    return cfg


@pytest.fixture
def inputs():
    """Complete input dict matching HybridGnnRnn.forward() expected schema."""
    np.random.seed(42)

    # Build a row-normalised adjacency matrix with self-loops
    adj = np.random.rand(NUM_TRADES, NUM_TRADES).astype(np.float32)
    adj = (adj > 0.7).astype(np.float32)
    np.fill_diagonal(adj, 1.0)
    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    adj = adj / row_sums

    # Extract sparse COO components for the adjacency representation
    nz = np.nonzero(adj)
    indices = np.stack(nz, axis=1).astype(np.int64)   # [nnz, 2] row-col pairs
    values = adj[nz].astype(np.float32)                # [nnz] edge weights
    shape = np.array(adj.shape, dtype=np.int64)        # [2] dense shape

    return {
        "trade_features": torch.from_numpy(
            np.random.randn(NUM_TRADES, FEATURE_DIM).astype(np.float32)
        ),
        "pnl_history": torch.from_numpy(
            np.random.randn(BATCH, SEQUENCE_LEN, NUM_ELEM).astype(np.float32)
        ),
        "adjacency_indices": torch.from_numpy(indices),
        "adjacency_values": torch.from_numpy(values),
        "adjacency_dense_shape": torch.from_numpy(shape),
        "elementary_indices": torch.arange(NUM_ELEM, dtype=torch.int32),
        "target_indices": torch.arange(NUM_TARGETS, dtype=torch.int32),
    }


@pytest.fixture
def model(config):
    """Instantiate a HybridGnnRnn model with test config."""
    return HybridGnnRnn(config=config)


# ---------------------------------------------------------------------------
# 1. Inheritance
# ---------------------------------------------------------------------------


class TestHybridGnnRnnInheritance:
    """Verify HybridGnnRnn satisfies expected type hierarchy."""

    def test_is_base_model(self, model):
        """Model must inherit from the library's BaseModel ABC."""
        assert isinstance(model, BaseModel)

    def test_is_nn_module(self, model):
        """Model must be a PyTorch nn.Module for autograd support."""
        assert isinstance(model, torch.nn.Module)


# ---------------------------------------------------------------------------
# 2. Initialisation
# ---------------------------------------------------------------------------


class TestHybridGnnRnnInit:
    """Verify internal sub-modules are created during __init__."""

    def test_has_all_blocks(self, model):
        """Model should expose GNN, RNN, fusion, attention, and projection blocks."""
        child_names = [name for name, _ in model.named_children()]
        for expected in ["gnn_block", "rnn_block", "fusion_layer", "attention_layer", "projection_layer"]:
            assert expected in child_names, f"Missing sub-module: {expected}"

    def test_layer_norm_created(self, model):
        """Model should contain at least one LayerNorm instance."""
        has_ln = any(isinstance(m, torch.nn.LayerNorm) for m in model.modules())
        assert has_ln, "Expected at least one LayerNorm module"

    def test_config_stored(self, model, config):
        """The config dict passed at construction should be retrievable."""
        stored = model.get_config()
        assert isinstance(stored, dict)


# ---------------------------------------------------------------------------
# 3. Forward pass
# ---------------------------------------------------------------------------


class TestHybridGnnRnnForward:
    """Validate forward() output properties."""

    def test_output_shape(self, model, inputs):
        """Output tensor shape must be [BATCH, NUM_TARGETS]."""
        model.eval()
        with torch.no_grad():
            out = model(inputs)
        assert out.shape == (BATCH, NUM_TARGETS)

    def test_output_is_tensor(self, model, inputs):
        """Forward pass must return a torch.Tensor."""
        model.eval()
        with torch.no_grad():
            out = model(inputs)
        assert isinstance(out, torch.Tensor)

    def test_no_nan(self, model, inputs):
        """Output must not contain NaN values."""
        model.eval()
        with torch.no_grad():
            out = model(inputs)
        assert not torch.any(torch.isnan(out)).item()

    def test_no_inf(self, model, inputs):
        """Output must not contain Inf values."""
        model.eval()
        with torch.no_grad():
            out = model(inputs)
        assert not torch.any(torch.isinf(out)).item()

    def test_training_mode(self, model, inputs):
        """Output shape should be identical in train and eval mode."""
        model.train()
        out_train = model(inputs)
        model.eval()
        with torch.no_grad():
            out_eval = model(inputs)
        assert out_train.shape == out_eval.shape

    def test_deterministic_in_eval_mode(self, model, inputs):
        """Two eval-mode forward passes with the same input must produce identical output."""
        # Run a forward pass first to initialize lazy modules
        model.eval()
        with torch.no_grad():
            _ = model(inputs)
        # Now run two forward passes and compare
        with torch.no_grad():
            out1 = model(inputs)
            out2 = model(inputs)
        np.testing.assert_allclose(out1.numpy(), out2.numpy(), rtol=1e-5)


# ---------------------------------------------------------------------------
# 4. Input validation
# ---------------------------------------------------------------------------


class TestHybridGnnRnnInputValidation:
    """Ensure the model rejects malformed inputs."""

    def test_missing_key_raises(self, model, inputs):
        """Removing a required key from the input dict must raise MissingKeyFields."""
        bad_inputs = {k: v for k, v in inputs.items() if k != "trade_features"}
        with pytest.raises(MissingKeyFields):
            model(bad_inputs)

    def test_empty_dict_raises(self, model):
        """An empty dict must raise MissingKeyFields."""
        with pytest.raises(MissingKeyFields):
            model({})


# ---------------------------------------------------------------------------
# 5. Metadata
# ---------------------------------------------------------------------------


class TestHybridGnnRnnMetadata:
    """Check model metadata populated by BaseModel."""

    def test_metadata(self, model):
        """Metadata must report 'pytorch' as the framework."""
        meta = model.metadata
        assert meta["framework"] == "pytorch"


# ---------------------------------------------------------------------------
# 6. Serialisation helpers
# ---------------------------------------------------------------------------


class TestHybridGnnRnnSerialisation:
    """Verify config round-tripping and summary utilities."""

    def test_get_config(self, model):
        """get_config must return a non-empty dict."""
        cfg = model.get_config()
        assert isinstance(cfg, dict)
        assert len(cfg) > 0

    def test_summary_dict_after_build(self, model, inputs):
        """summary_dict must report trainable param count after a forward pass."""
        # Run a forward pass so all lazy parameters (if any) are materialised
        model.eval()
        with torch.no_grad():
            model(inputs)
        summary = model.summary_dict()
        assert "trainable_params" in summary
        assert summary["trainable_params"] > 0


# ---------------------------------------------------------------------------
# 7. Gradient flow
# ---------------------------------------------------------------------------


class TestHybridGnnRnnGradients:
    """Verify autograd graph connects output to all trainable weights."""

    def test_gradients_flow(self, model, inputs):
        """Every trainable parameter must receive a non-None gradient after backward."""
        targets = torch.randn(BATCH, NUM_TARGETS)
        model.train()
        preds = model(inputs)
        loss = (preds - targets).pow(2).mean()
        loss.backward()
        # Collect names of parameters that require grad but received None
        none_grads = [
            n for n, p in model.named_parameters() if p.requires_grad and p.grad is None
        ]
        assert len(none_grads) == 0, f"Parameters with None grad: {none_grads}"

    def test_gradients_are_finite(self, model, inputs):
        """All gradients must be finite (no NaN or Inf)."""
        targets = torch.randn(BATCH, NUM_TARGETS)
        model.train()
        preds = model(inputs)
        loss = (preds - targets).pow(2).mean()
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.all(torch.isfinite(p.grad)).item(), (
                    f"Non-finite gradient in parameter: {name}"
                )


# ---------------------------------------------------------------------------
# 8. Parameter count
# ---------------------------------------------------------------------------


class TestHybridGnnRnnParamCount:
    """Sanity-check the number of trainable parameters."""

    def test_has_trainable_params(self, model, inputs):
        """Model must have at least one trainable parameter after initialization."""
        # Forward pass initializes lazy modules so .numel() works
        model.eval()
        with torch.no_grad():
            model(inputs)
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total > 0

    def test_param_count_reasonable(self, model, inputs):
        """Total trainable params should be within a plausible range (<10M for this config)."""
        model.eval()
        with torch.no_grad():
            model(inputs)
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total < 10_000_000, f"Unexpectedly large param count: {total}"

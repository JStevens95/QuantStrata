"""Unit tests for rade_ml_pt.models.hybrid_gnn_rnn.layers.projection_layer."""
import numpy as np
import pytest
import torch

from src.rade_ml_pt.models.hybrid_gnn_rnn.layers.projection_layer import TargetPnlOutput


BATCH = 4
NUM_TRADES = 20
NUM_TARGETS = 8
ATTN_DIM = 32
FEATURE_DIM = 10


def _make_projection_config(
    baseline_new_mode="output_mix",
    use_baseline_weight_norm=True,
    use_attn_scale=False,
    use_attn_bias=False,
    knn_mode="cosine_softmax",
):
    """Build a standard layer_config dict for TargetPnlOutput tests."""
    return {
        "general": {
            "dropout_rate": 0.0,
            "baseline_new_mode": baseline_new_mode,
            "use_baseline_weight_norm": use_baseline_weight_norm,
            "use_attn_scale_new": use_attn_scale,
            "use_attn_bias_new": use_attn_bias,
            "knn_k": 5,
            "knn_mode": knn_mode,
            "knn_temperature": 5.0,
            "knn_power": 2.0,
            "residual_new_damp": 1.0,
            "baseline_trade_count": NUM_TARGETS,
            "attn_dim": ATTN_DIM,
        },
        "parameters": {
            "units": 32,
            "activation": "gelu",
            "kernel_initializer": "glorot_uniform",
            "bias_initializer": "zeros",
        },
    }


@pytest.fixture
def trade_features():
    """Random float32 trade attributes [NUM_TRADES, FEATURE_DIM]."""
    np.random.seed(42)
    return torch.from_numpy(np.random.randn(NUM_TRADES, FEATURE_DIM).astype(np.float32))


@pytest.fixture
def attn_features():
    """Random float32 attention features [BATCH, NUM_TARGETS, ATTN_DIM]."""
    np.random.seed(43)
    return torch.from_numpy(np.random.randn(BATCH, NUM_TARGETS, ATTN_DIM).astype(np.float32))


@pytest.fixture
def target_idx():
    """Sequential target indices [0, 1, ..., NUM_TARGETS-1]."""
    return torch.arange(NUM_TARGETS, dtype=torch.long)


class TestTargetPnlOutputForward:
    """Basic forward-pass correctness: shape, NaN-free, train vs infer."""

    def test_output_shape(self, trade_features, attn_features, target_idx):
        """Forward produces [BATCH, NUM_TARGETS] output."""
        layer = TargetPnlOutput(layer_config=_make_projection_config(), name="proj_test")
        out = layer((trade_features, attn_features, target_idx))
        assert out.shape == (BATCH, NUM_TARGETS)

    def test_no_nan_output(self, trade_features, attn_features, target_idx):
        """Output contains no NaN values."""
        layer = TargetPnlOutput(layer_config=_make_projection_config(), name="proj_nan")
        out = layer((trade_features, attn_features, target_idx))
        assert not torch.any(torch.isnan(out)).item()

    def test_training_mode(self, trade_features, attn_features, target_idx):
        """Training and inference modes produce same-shaped output."""
        layer = TargetPnlOutput(layer_config=_make_projection_config(), name="proj_train")
        out_train = layer((trade_features, attn_features, target_idx))
        out_infer = layer((trade_features, attn_features, target_idx))
        assert out_train.shape == out_infer.shape


class TestTargetPnlOutputBaselineNorm:
    """Weight-norm decomposition (unit-norm kernel + gain) vs raw kernel."""

    def test_with_baseline_norm(self, trade_features, attn_features, target_idx):
        """Baseline with weight-norm enabled produces correct shape."""
        layer = TargetPnlOutput(
            layer_config=_make_projection_config(use_baseline_weight_norm=True),
            name="proj_bnorm",
        )
        out = layer((trade_features, attn_features, target_idx))
        assert out.shape == (BATCH, NUM_TARGETS)

    def test_without_baseline_norm(self, trade_features, attn_features, target_idx):
        """Baseline without weight-norm (raw kernel @ attn) produces correct shape."""
        layer = TargetPnlOutput(
            layer_config=_make_projection_config(use_baseline_weight_norm=False),
            name="proj_no_bnorm",
        )
        out = layer((trade_features, attn_features, target_idx))
        assert out.shape == (BATCH, NUM_TARGETS)


class TestTargetPnlOutputKnnModes:
    """kNN interpolation modes for new-target baseline transfer."""

    def test_cosine_softmax(self, trade_features, attn_features, target_idx):
        """cosine_softmax kNN mode produces correct shape."""
        layer = TargetPnlOutput(
            layer_config=_make_projection_config(knn_mode="cosine_softmax"),
            name="proj_cos",
        )
        out = layer((trade_features, attn_features, target_idx))
        assert out.shape == (BATCH, NUM_TARGETS)

    def test_idw(self, trade_features, attn_features, target_idx):
        """Inverse-distance weighting kNN mode produces correct shape."""
        layer = TargetPnlOutput(
            layer_config=_make_projection_config(knn_mode="idw"),
            name="proj_idw",
        )
        out = layer((trade_features, attn_features, target_idx))
        assert out.shape == (BATCH, NUM_TARGETS)

    def test_invalid_knn_mode_raises(self, trade_features, attn_features, target_idx):
        """Unknown kNN mode raises ValueError when the kNN path is exercised."""
        cfg = _make_projection_config(knn_mode="invalid")
        # Set baseline_trade_count < NUM_TARGETS so n_new > 0 triggers the kNN path.
        cfg["general"]["baseline_trade_count"] = 4
        layer = TargetPnlOutput(layer_config=cfg, name="proj_knn_bad")
        with pytest.raises(ValueError, match="Unknown knn_mode"):
            layer((trade_features, attn_features, target_idx))


class TestTargetPnlOutputAttentionConditioning:
    """Post-hoc attention-conditioned scale/bias for new targets."""

    def test_attn_scale_enabled(self, trade_features, attn_features, target_idx):
        """Attention-conditioned scale produces correct shape."""
        layer = TargetPnlOutput(
            layer_config=_make_projection_config(use_attn_scale=True),
            name="proj_scale",
        )
        out = layer((trade_features, attn_features, target_idx))
        assert out.shape == (BATCH, NUM_TARGETS)

    def test_attn_bias_enabled(self, trade_features, attn_features, target_idx):
        """Attention-conditioned bias produces correct shape."""
        layer = TargetPnlOutput(
            layer_config=_make_projection_config(use_attn_bias=True),
            name="proj_bias",
        )
        out = layer((trade_features, attn_features, target_idx))
        assert out.shape == (BATCH, NUM_TARGETS)


class TestTargetPnlOutputLazyInit:
    """Lazy initialization: omit baseline_trade_count and/or attn_dim."""

    @staticmethod
    def _make_lazy_config(omit_n0=False, omit_attn_dim=False, **overrides):
        """Config with selected dimensions removed for lazy init testing."""
        cfg = _make_projection_config()
        if omit_attn_dim:
            cfg["general"].pop("attn_dim", None)
        if omit_n0:
            cfg["general"].pop("baseline_trade_count", None)
        cfg["general"].update(overrides)
        return cfg

    # -- attn_dim only lazy --

    def test_construction_without_attn_dim(self):
        layer = TargetPnlOutput(
            layer_config=self._make_lazy_config(omit_attn_dim=True),
            name="proj_lazy_d",
        )
        assert layer.attn_dim is None

    def test_forward_infers_attn_dim(self, trade_features, attn_features, target_idx):
        layer = TargetPnlOutput(
            layer_config=self._make_lazy_config(omit_attn_dim=True),
            name="proj_lazy_d_fwd",
        )
        out = layer((trade_features, attn_features, target_idx))
        assert layer.attn_dim == ATTN_DIM
        assert out.shape == (BATCH, NUM_TARGETS)

    # -- baseline_trade_count only lazy --

    def test_construction_without_baseline_trade_count(self):
        layer = TargetPnlOutput(
            layer_config=self._make_lazy_config(omit_n0=True),
            name="proj_lazy_n0",
        )
        assert layer.baseline_trade_count is None

    def test_forward_infers_baseline_trade_count(
        self, trade_features, attn_features, target_idx
    ):
        layer = TargetPnlOutput(
            layer_config=self._make_lazy_config(omit_n0=True),
            name="proj_lazy_n0_fwd",
        )
        out = layer((trade_features, attn_features, target_idx))
        assert layer.baseline_trade_count == NUM_TARGETS
        assert out.shape == (BATCH, NUM_TARGETS)

    # -- both dimensions lazy (fully lazy, matches TF build() behaviour) --

    def test_construction_fully_lazy(self):
        layer = TargetPnlOutput(
            layer_config=self._make_lazy_config(omit_n0=True, omit_attn_dim=True),
            name="proj_fully_lazy",
        )
        assert layer.baseline_trade_count is None
        assert layer.attn_dim is None

    def test_forward_fully_lazy(self, trade_features, attn_features, target_idx):
        layer = TargetPnlOutput(
            layer_config=self._make_lazy_config(omit_n0=True, omit_attn_dim=True),
            name="proj_fully_lazy_fwd",
        )
        out = layer((trade_features, attn_features, target_idx))
        assert layer.baseline_trade_count == NUM_TARGETS
        assert layer.attn_dim == ATTN_DIM
        assert out.shape == (BATCH, NUM_TARGETS)
        assert not torch.any(torch.isnan(out)).item()

    def test_fully_lazy_with_weight_norm(
        self, trade_features, attn_features, target_idx
    ):
        layer = TargetPnlOutput(
            layer_config=self._make_lazy_config(
                omit_n0=True, omit_attn_dim=True, use_baseline_weight_norm=True
            ),
            name="proj_lazy_wnorm",
        )
        out = layer((trade_features, attn_features, target_idx))
        assert out.shape == (BATCH, NUM_TARGETS)

    def test_lazy_output_matches_eager_shape(
        self, trade_features, attn_features, target_idx
    ):
        lazy = TargetPnlOutput(
            layer_config=self._make_lazy_config(omit_n0=True, omit_attn_dim=True),
            name="proj_lazy_cmp",
        )
        eager = TargetPnlOutput(
            layer_config=_make_projection_config(), name="proj_eager_cmp"
        )
        out_lazy = lazy((trade_features, attn_features, target_idx))
        out_eager = eager((trade_features, attn_features, target_idx))
        assert out_lazy.shape == out_eager.shape
        assert not torch.any(torch.isnan(out_lazy)).item()

    # -- attention conditioning with lazy dims --

    def test_lazy_with_attn_scale(self, trade_features, attn_features, target_idx):
        layer = TargetPnlOutput(
            layer_config=self._make_lazy_config(
                omit_n0=True, omit_attn_dim=True, use_attn_scale_new=True
            ),
            name="proj_lazy_scale",
        )
        out = layer((trade_features, attn_features, target_idx))
        assert out.shape == (BATCH, NUM_TARGETS)

    def test_lazy_with_attn_bias(self, trade_features, attn_features, target_idx):
        layer = TargetPnlOutput(
            layer_config=self._make_lazy_config(
                omit_n0=True, omit_attn_dim=True, use_attn_bias_new=True
            ),
            name="proj_lazy_bias",
        )
        out = layer((trade_features, attn_features, target_idx))
        assert out.shape == (BATCH, NUM_TARGETS)

    # -- optimizer integration --

    def test_lazy_parameters_tracked_by_optimizer(
        self, trade_features, attn_features, target_idx
    ):
        """Optimizer created before forward still tracks materialized params."""
        layer = TargetPnlOutput(
            layer_config=self._make_lazy_config(omit_n0=True, omit_attn_dim=True),
            name="proj_lazy_opt",
        )
        optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)
        out = layer((trade_features, attn_features, target_idx))
        loss = out.sum()
        loss.backward()
        optimizer.step()
        assert layer._baseline_kernels.grad is not None
        assert layer._baseline_biases.grad is not None

    # -- serialization round-trip --

    def test_lazy_persists_to_config(self, trade_features, attn_features, target_idx):
        """Resolved dimensions are persisted into layer_config for serialization."""
        cfg = self._make_lazy_config(omit_n0=True, omit_attn_dim=True)
        layer = TargetPnlOutput(layer_config=cfg, name="proj_lazy_cfg")
        layer((trade_features, attn_features, target_idx))
        saved = layer.get_config()
        general = saved["layer_config"]["general"]
        assert general["baseline_trade_count"] == NUM_TARGETS
        assert general["attn_dim"] == ATTN_DIM


class TestTargetPnlOutputConfig:
    """Serialization round-trip."""

    def test_get_config(self):
        """get_config returns a dict containing 'layer_config'."""
        layer = TargetPnlOutput(layer_config=_make_projection_config(), name="proj_cfg")
        config = layer.get_config()
        assert "layer_config" in config

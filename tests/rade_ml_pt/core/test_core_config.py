"""Unit tests for rade_ml_pt.core.config -- Configuration dataclasses (PyTorch)."""
import json
import pytest
import torch

from src.rade_ml_pt.core.config import (
    DataPipelineConfig,
    TrainingConfig,
    OptimizerConfig,
    LrScheduleConfig,
    EarlyStoppingConfig,
    CheckpointConfig,
    ReduceLrConfig,
)


def _dummy_params():
    """Return a small parameter iterator for building optimizers in tests."""
    layer = torch.nn.Linear(4, 2)
    return layer.parameters()


class TestDataPipelineConfig:
    def test_defaults(self):
        """Verify sensible default values for DataPipelineConfig."""
        cfg = DataPipelineConfig()
        assert cfg.batch_size == 32
        assert cfg.shuffle is False
        assert cfg.seed == 42

    def test_to_dict_roundtrip(self):
        """Verify to_dict / from_dict round-trip preserves all fields."""
        cfg = DataPipelineConfig(batch_size=128, shuffle=True)
        d = cfg.to_dict()
        restored = DataPipelineConfig.from_dict(d)
        assert restored.batch_size == 128
        assert restored.shuffle is True

    def test_to_build_kwargs(self):
        """Verify to_build_kwargs includes expected keys."""
        cfg = DataPipelineConfig(batch_size=64)
        kw = cfg.to_build_kwargs()
        assert kw["batch_size"] == 64
        assert "seed" in kw

    def test_json_roundtrip(self, tmp_path):
        """Verify JSON save/load round-trip."""
        path = tmp_path / "dp_config.json"
        cfg = DataPipelineConfig(batch_size=256)
        cfg.to_json(path)
        loaded = DataPipelineConfig.from_json(path)
        assert loaded.batch_size == 256


class TestOptimizerConfig:
    def test_defaults_are_standard_adam(self):
        """Verify default betas match standard Adam defaults."""
        cfg = OptimizerConfig()
        assert cfg.beta_1 == 0.9
        assert cfg.beta_2 == 0.999

    def test_build_adam(self):
        """Verify build() produces a working Adam optimizer with correct LR."""
        cfg = OptimizerConfig(name="adam", learning_rate=1e-3)
        opt = cfg.build(_dummy_params())
        assert opt is not None
        # PyTorch stores lr in param_groups
        assert opt.param_groups[0]["lr"] == pytest.approx(1e-3, rel=1e-5)

    def test_build_adamw(self):
        """Verify build() produces an AdamW optimizer with weight decay."""
        cfg = OptimizerConfig(name="adamw", learning_rate=1e-4, weight_decay=0.01)
        opt = cfg.build(_dummy_params())
        assert isinstance(opt, torch.optim.AdamW)
        assert opt.param_groups[0]["weight_decay"] == pytest.approx(0.01)

    def test_build_sgd(self):
        """Verify build() produces an SGD optimizer with correct LR."""
        cfg = OptimizerConfig(name="sgd", learning_rate=0.01)
        opt = cfg.build(_dummy_params())
        assert opt.param_groups[0]["lr"] == pytest.approx(0.01, rel=1e-5)

    def test_build_rmsprop(self):
        """Verify build() produces an RMSprop optimizer."""
        cfg = OptimizerConfig(name="rmsprop", learning_rate=0.001)
        opt = cfg.build(_dummy_params())
        assert isinstance(opt, torch.optim.RMSprop)

    def test_build_unsupported_raises(self):
        """Verify build() raises ValueError for unknown optimizer names."""
        cfg = OptimizerConfig(name="nonexistent")
        with pytest.raises(ValueError, match="Unsupported optimizer"):
            cfg.build(_dummy_params())


class TestLrScheduleConfig:
    def test_constant_returns_none(self):
        """Verify constant schedule returns None (no scheduler needed)."""
        cfg = LrScheduleConfig(schedule="constant", initial_lr=5e-4)
        opt = torch.optim.Adam(_dummy_params(), lr=5e-4)
        result = cfg.build(opt)
        assert result is None

    def test_exponential_returns_scheduler(self):
        """Verify exponential schedule returns a StepLR scheduler."""
        cfg = LrScheduleConfig(schedule="exponential", decay_steps=100, decay_rate=0.9)
        opt = torch.optim.Adam(_dummy_params(), lr=1e-3)
        sched = cfg.build(opt)
        assert isinstance(sched, torch.optim.lr_scheduler.StepLR)

    def test_cosine_requires_total_steps(self):
        """Verify cosine schedule raises ValueError when total_steps is missing."""
        cfg = LrScheduleConfig(schedule="cosine")
        opt = torch.optim.Adam(_dummy_params(), lr=1e-3)
        with pytest.raises(ValueError, match="total_steps"):
            cfg.build(opt)

    def test_cosine_returns_scheduler(self):
        """Verify cosine schedule returns a CosineAnnealingLR scheduler."""
        cfg = LrScheduleConfig(schedule="cosine", min_lr=1e-6)
        opt = torch.optim.Adam(_dummy_params(), lr=1e-3)
        sched = cfg.build(opt, total_steps=1000)
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_unsupported_raises(self):
        """Verify unsupported schedule name raises ValueError."""
        cfg = LrScheduleConfig(schedule="nonexistent")
        opt = torch.optim.Adam(_dummy_params(), lr=1e-3)
        with pytest.raises(ValueError, match="Unsupported schedule"):
            cfg.build(opt)


class TestTrainingConfig:
    def test_defaults(self):
        """Verify sensible default values for TrainingConfig."""
        cfg = TrainingConfig()
        assert cfg.epochs == 100
        assert cfg.loss == "mae"

    def test_to_dict_roundtrip(self):
        """Verify to_dict / from_dict round-trip preserves all fields."""
        cfg = TrainingConfig(epochs=50, loss="mse")
        d = cfg.to_dict()
        restored = TrainingConfig.from_dict(d)
        assert restored.epochs == 50
        assert restored.loss == "mse"

    def test_json_roundtrip(self, tmp_path):
        """Verify JSON save/load round-trip."""
        path = tmp_path / "train_config.json"
        cfg = TrainingConfig(epochs=25)
        cfg.to_json(path)
        loaded = TrainingConfig.from_json(path)
        assert loaded.epochs == 25

    def test_nested_config_serialisation(self):
        """Verify nested sub-configs serialise correctly."""
        cfg = TrainingConfig(
            optimizer=OptimizerConfig(name="adamw", learning_rate=3e-4),
            early_stopping=EarlyStoppingConfig(patience=20),
        )
        d = cfg.to_dict()
        assert d["optimizer"]["name"] == "adamw"
        assert d["early_stopping"]["patience"] == 20

    def test_compile_model_field(self):
        """Verify compile_model field replaces TF's xla_compile."""
        cfg = TrainingConfig(compile_model=True)
        d = cfg.to_dict()
        assert d["compile_model"] is True

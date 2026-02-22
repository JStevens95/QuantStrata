"""Unit tests for rade_ml.core.config -- Configuration dataclasses."""
import json
import pytest

from src.rade_ml.core.config import (
    DataPipelineConfig,
    TrainingConfig,
    OptimizerConfig,
    LrScheduleConfig,
    EarlyStoppingConfig,
    CheckpointConfig,
    ReduceLrConfig,
)


class TestDataPipelineConfig:
    def test_defaults(self):
        cfg = DataPipelineConfig()
        assert cfg.batch_size == 32
        assert cfg.shuffle is False
        assert cfg.seed == 42

    def test_to_dict_roundtrip(self):
        cfg = DataPipelineConfig(batch_size=128, shuffle=True)
        d = cfg.to_dict()
        restored = DataPipelineConfig.from_dict(d)
        assert restored.batch_size == 128
        assert restored.shuffle is True

    def test_to_build_kwargs(self):
        cfg = DataPipelineConfig(batch_size=64)
        kw = cfg.to_build_kwargs()
        assert kw["batch_size"] == 64
        assert "seed" in kw

    def test_json_roundtrip(self, tmp_path):
        path = tmp_path / "dp_config.json"
        cfg = DataPipelineConfig(batch_size=256)
        cfg.to_json(path)
        loaded = DataPipelineConfig.from_json(path)
        assert loaded.batch_size == 256


class TestOptimizerConfig:
    def test_defaults_are_standard_adam(self):
        cfg = OptimizerConfig()
        assert cfg.beta_1 == 0.9
        assert cfg.beta_2 == 0.999

    def test_build_adam(self):
        cfg = OptimizerConfig(name="adam", learning_rate=1e-3)
        opt = cfg.build()
        assert opt is not None
        assert float(opt.learning_rate) == pytest.approx(1e-3, rel=1e-5)

    def test_build_sgd(self):
        cfg = OptimizerConfig(name="sgd", learning_rate=0.01)
        opt = cfg.build()
        assert float(opt.learning_rate) == pytest.approx(0.01, rel=1e-5)

    def test_build_unsupported_raises(self):
        cfg = OptimizerConfig(name="nonexistent")
        with pytest.raises(ValueError, match="Unsupported optimizer"):
            cfg.build()


class TestLrScheduleConfig:
    def test_constant_returns_float(self):
        cfg = LrScheduleConfig(schedule="constant", initial_lr=5e-4)
        result = cfg.build()
        assert result == 5e-4

    def test_exponential_returns_schedule(self):
        cfg = LrScheduleConfig(schedule="exponential")
        sched = cfg.build()
        assert callable(sched)

    def test_cosine_requires_total_steps(self):
        cfg = LrScheduleConfig(schedule="cosine")
        with pytest.raises(ValueError, match="total_steps"):
            cfg.build()

    def test_warmup_cosine(self):
        cfg = LrScheduleConfig(schedule="warmup_cosine", warmup_steps=100)
        sched = cfg.build(total_steps=1000)
        assert callable(sched)


class TestTrainingConfig:
    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.epochs == 100
        assert cfg.loss == "mae"

    def test_to_dict_roundtrip(self):
        cfg = TrainingConfig(epochs=50, loss="mse")
        d = cfg.to_dict()
        restored = TrainingConfig.from_dict(d)
        assert restored.epochs == 50
        assert restored.loss == "mse"

    def test_json_roundtrip(self, tmp_path):
        path = tmp_path / "train_config.json"
        cfg = TrainingConfig(epochs=25)
        cfg.to_json(path)
        loaded = TrainingConfig.from_json(path)
        assert loaded.epochs == 25

    def test_nested_config_serialisation(self):
        cfg = TrainingConfig(
            optimizer=OptimizerConfig(name="adamw", learning_rate=3e-4),
            early_stopping=EarlyStoppingConfig(patience=20),
        )
        d = cfg.to_dict()
        assert d["optimizer"]["name"] == "adamw"
        assert d["early_stopping"]["patience"] == 20

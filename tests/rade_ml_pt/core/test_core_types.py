"""Unit tests for rade_ml_pt.core.types -- Result dataclasses."""
import json
import pytest

from src.rade_ml_pt.core.types import (
    CheckpointInfo,
    TrainingResult,
    EvaluationResult,
    InferenceResult,
    DeepHedgingInferenceResult,
)


class TestCheckpointInfo:
    def test_to_dict_roundtrip(self):
        ci = CheckpointInfo(path="/tmp/ckpt", epoch=5, train_loss=0.1, val_loss=0.2, is_best=True)
        d = ci.to_dict()
        restored = CheckpointInfo.from_dict(d)
        assert restored.epoch == 5
        assert restored.is_best is True


class TestTrainingResult:
    def test_defaults(self):
        r = TrainingResult()
        assert r.best_train_loss == float("inf")
        assert r.final_epoch == 0

    def test_json_roundtrip(self, tmp_path):
        r = TrainingResult(
            history={"loss": [0.5, 0.3, 0.1]},
            best_epoch=3,
            best_val_loss=0.1,
        )
        path = tmp_path / "result.json"
        r.to_json(path)
        loaded = TrainingResult.from_json(path)
        assert loaded.best_epoch == 3
        assert loaded.history["loss"] == [0.5, 0.3, 0.1]

    def test_to_dict_converts_checkpoints(self):
        ci = CheckpointInfo(path="/tmp", epoch=1, train_loss=0.5, val_loss=0.4)
        r = TrainingResult(checkpoints=[ci])
        d = r.to_dict()
        assert isinstance(d["checkpoints"][0], dict)


class TestEvaluationResult:
    def test_repr(self):
        r = EvaluationResult(metrics={"mae": 0.01, "rmse": 0.02})
        s = repr(r)
        assert "mae" in s

    def test_summary(self):
        r = EvaluationResult(metrics={"mae": 0.01}, loss=0.01)
        summary = r.summary()
        assert "mae" in summary
        assert "0.01" in summary

    def test_to_dict_excludes_arrays(self):
        import numpy as np
        r = EvaluationResult(predictions=np.zeros(10), targets=np.ones(10))
        d = r.to_dict()
        assert "predictions" not in d
        assert "targets" not in d


class TestInferenceResult:
    def test_repr(self):
        r = InferenceResult(sample_ids=["A", "B"], n_samples=100, latency_seconds=0.5)
        s = repr(r)
        assert "samples=100" in s
        assert "ids=2" in s

    def test_json_roundtrip(self, tmp_path):
        r = InferenceResult(n_samples=50, model_version="v1")
        path = tmp_path / "inference.json"
        r.to_json(path)
        loaded = InferenceResult.from_json(path)
        assert loaded.n_samples == 50
        assert loaded.model_version == "v1"


class TestDeepHedgingInferenceResult:
    def test_inherits_base(self):
        r = DeepHedgingInferenceResult(n_samples=100, sample_ids=["opt_1", "opt_2"])
        assert r.scenario_count == 100
        assert r.trade_ids == ["opt_1", "opt_2"]

    def test_domain_aliases(self):
        r = DeepHedgingInferenceResult(n_samples=50)
        assert r.scenario_count == r.n_samples
        assert r.trade_ids is None

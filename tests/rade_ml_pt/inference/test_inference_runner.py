"""Unit tests for rade_ml_pt.inference.runner -- InferenceRunner."""
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.rade_ml_pt.core.types import InferenceResult, DeepHedgingInferenceResult
from src.rade_ml_pt.inference.runner import InferenceRunner


def _make_model_and_save(tmp_path):
    """Create a simple model, save it via torch.save, and return both."""
    model = nn.Linear(3, 1)
    model.eval()
    model_path = str(tmp_path / "model.pt")
    torch.save(model, model_path)
    return model, model_path


class TestInferenceRunnerFromPath:
    def test_from_path(self, tmp_path):
        _, model_path = _make_model_and_save(tmp_path)
        runner = InferenceRunner.from_path(model_path, model_version="test_v1")
        assert runner.model is not None
        assert runner.model_version == "test_v1"

    def test_predict_returns_inference_result(self, tmp_path):
        _, model_path = _make_model_and_save(tmp_path)
        runner = InferenceRunner.from_path(model_path)
        inputs = np.random.randn(10, 3).astype(np.float32)
        result = runner.predict(inputs)
        assert isinstance(result, InferenceResult)
        assert result.n_samples == 10
        assert result.latency_seconds > 0


class TestInferenceRunnerPredict:
    def test_sample_ids(self, tmp_path):
        model, model_path = _make_model_and_save(tmp_path)
        runner = InferenceRunner(model=model, model_path=model_path)
        inputs = np.random.randn(5, 3).astype(np.float32)
        result = runner.predict(inputs, sample_ids=["A", "B", "C", "D", "E"])
        assert result.sample_ids == ["A", "B", "C", "D", "E"]

    def test_input_hash_deterministic(self, tmp_path):
        model, model_path = _make_model_and_save(tmp_path)
        runner = InferenceRunner(model=model, model_path=model_path)
        inputs = np.ones((3, 3), dtype=np.float32)
        r1 = runner.predict(inputs, hash_inputs=True)
        r2 = runner.predict(inputs, hash_inputs=True)
        assert r1.input_hash == r2.input_hash

    def test_hash_disabled(self, tmp_path):
        model, model_path = _make_model_and_save(tmp_path)
        runner = InferenceRunner(model=model, model_path=model_path)
        result = runner.predict(np.ones((2, 3), dtype=np.float32), hash_inputs=False)
        assert result.input_hash is None

    def test_result_cls(self, tmp_path):
        _, model_path = _make_model_and_save(tmp_path)
        runner = InferenceRunner.from_path(model_path)
        inputs = np.random.randn(3, 3).astype(np.float32)
        result = runner.predict(inputs, result_cls=DeepHedgingInferenceResult)
        assert isinstance(result, DeepHedgingInferenceResult)
        assert result.n_samples == 3
        assert result.scenario_count == 3

    def test_metadata_merged(self, tmp_path):
        model, model_path = _make_model_and_save(tmp_path)
        runner = InferenceRunner(
            model=model, model_path=model_path, metadata={"source": "test"}
        )
        result = runner.predict(
            np.ones((2, 3), dtype=np.float32),
            metadata={"scenario": "base"},
        )
        assert result.metadata["source"] == "test"
        assert result.metadata["scenario"] == "base"


class TestInferenceRunnerHashInputs:
    def test_hash_numpy(self):
        h = InferenceRunner._hash_inputs(np.array([1.0, 2.0, 3.0]))
        assert isinstance(h, str)
        assert len(h) == 32  # MD5 hex digest

    def test_hash_dict(self):
        h = InferenceRunner._hash_inputs({
            "a": np.array([1.0]),
            "b": np.array([2.0]),
        })
        assert isinstance(h, str)

    def test_hash_tensor(self):
        h = InferenceRunner._hash_inputs(torch.tensor([1.0, 2.0]))
        assert isinstance(h, str)

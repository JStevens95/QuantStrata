"""Unit tests for rade_ml.tracking.run -- Run dataclass."""
import pytest

from src.rade_ml.tracking.run import Run


class TestRunCreation:
    def test_defaults(self):
        r = Run()
        assert len(r.run_id) == 12
        assert r.status == "running"
        assert r.metrics == {}
        assert r.params == {}

    def test_custom_name_and_tags(self):
        r = Run(name="experiment_1", tags=["lr-sweep"])
        assert r.name == "experiment_1"
        assert "lr-sweep" in r.tags


class TestRunLogging:
    def test_log_metric(self):
        r = Run()
        r.log_metric("loss", 0.5)
        assert r.metrics["loss"] == 0.5

    def test_log_metrics_batch(self):
        r = Run()
        r.log_metrics({"loss": 0.3, "mae": 0.1})
        assert r.metrics["loss"] == 0.3
        assert r.metrics["mae"] == 0.1

    def test_log_params(self):
        r = Run()
        r.log_params({"lr": 1e-3, "epochs": 100})
        assert r.params["lr"] == 1e-3

    def test_log_config_dict(self):
        r = Run()
        r.log_config({"optimizer": "adam", "lr": 1e-3})
        assert r.config == {"optimizer": "adam", "lr": 1e-3}

    def test_log_config_with_to_dict(self):
        class _FakeConfig:
            def to_dict(self):
                return {"k": "v"}
        r = Run()
        r.log_config(_FakeConfig())
        assert r.config == {"k": "v"}

    def test_set_model_version(self):
        r = Run()
        r.set_model_version("20260101_v1")
        assert r.model_version == "20260101_v1"


class TestRunLifecycle:
    def test_end(self):
        r = Run()
        r.end()
        assert r.status == "completed"
        assert r.end_time is not None

    def test_fail(self):
        r = Run()
        r.fail("OOM error")
        assert r.status == "failed"
        assert r.error == "OOM error"
        assert r.end_time is not None


class TestRunSerialisation:
    def test_to_dict_roundtrip(self):
        r = Run(name="test")
        r.log_metric("loss", 0.5)
        d = r.to_dict()
        restored = Run.from_dict(d)
        assert restored.name == "test"
        assert restored.metrics["loss"] == 0.5

    def test_json_roundtrip(self, tmp_path):
        r = Run(name="test_run")
        r.log_params({"lr": 0.001})
        path = tmp_path / "run.json"
        r.to_json(path)
        loaded = Run.from_json(path)
        assert loaded.name == "test_run"
        assert loaded.params["lr"] == 0.001

    def test_repr(self):
        r = Run(name="exp1", tags=["a", "b"])
        s = repr(r)
        assert "exp1" in s
        assert "a, b" in s

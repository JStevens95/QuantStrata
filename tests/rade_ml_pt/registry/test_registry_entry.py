"""Unit tests for rade_ml_pt.registry.entry -- RegistryEntry."""
import pytest

from src.rade_ml_pt.registry.entry import RegistryEntry


class TestRegistryEntry:
    def test_defaults(self):
        e = RegistryEntry()
        assert e.version == ""
        assert e.tags == []
        assert e.timestamp is not None

    def test_to_dict_roundtrip(self):
        e = RegistryEntry(
            version="20260101_120000_abc123",
            model_dir="/tmp/model",
            tags=["best", "prod"],
            metrics={"best_val_loss": 0.01},
        )
        d = e.to_dict()
        restored = RegistryEntry.from_dict(d)
        assert restored.version == e.version
        assert restored.tags == ["best", "prod"]
        assert restored.metrics["best_val_loss"] == 0.01

    def test_json_roundtrip(self, tmp_path):
        path = tmp_path / "entry.json"
        e = RegistryEntry(version="v1", description="test entry")
        e.to_json(path)
        loaded = RegistryEntry.from_json(path)
        assert loaded.version == "v1"
        assert loaded.description == "test entry"

    def test_repr(self):
        e = RegistryEntry(version="v1", tags=["best"], metrics={"best_val_loss": 0.05})
        s = repr(e)
        assert "v1" in s
        assert "best" in s

    def test_from_dict_ignores_unknown_keys(self):
        d = {"version": "v1", "unknown_key": 42, "model_dir": "/tmp"}
        e = RegistryEntry.from_dict(d)
        assert e.version == "v1"

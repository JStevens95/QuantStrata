"""Unit tests for rade_ml.registry.store -- ModelRegistry."""
import pytest
import tensorflow as tf

from src.rade_ml.core.types import TrainingResult
from src.rade_ml.registry.store import ModelRegistry
from src.rade_ml.registry.entry import RegistryEntry


def _make_simple_model():
    inp = tf.keras.Input(shape=(3,))
    out = tf.keras.layers.Dense(1)(inp)
    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="mse")
    return model


def _make_training_result(**kwargs):
    defaults = dict(
        best_val_loss=0.1,
        best_train_loss=0.05,
        final_epoch=10,
        best_epoch=8,
        training_time_seconds=60.0,
        history={"loss": [0.5, 0.3, 0.1]},
    )
    defaults.update(kwargs)
    return TrainingResult(**defaults)


class TestModelRegistryRegister:
    def test_register_creates_version_dir(self, tmp_path):
        registry = ModelRegistry(tmp_path / "registry")
        model = _make_simple_model()
        result = _make_training_result()
        entry = registry.register(model, result, tags=["test"])
        assert (tmp_path / "registry" / entry.version).is_dir()

    def test_register_returns_entry(self, tmp_path):
        registry = ModelRegistry(tmp_path / "registry")
        entry = registry.register(_make_simple_model(), _make_training_result())
        assert isinstance(entry, RegistryEntry)
        assert entry.version != ""
        assert entry.metrics["best_val_loss"] == 0.1

    def test_register_with_tags(self, tmp_path):
        registry = ModelRegistry(tmp_path / "registry")
        entry = registry.register(
            _make_simple_model(), _make_training_result(), tags=["best", "prod"]
        )
        assert "best" in entry.tags
        assert "prod" in entry.tags

    def test_latest_tag_set(self, tmp_path):
        registry = ModelRegistry(tmp_path / "registry")
        entry = registry.register(_make_simple_model(), _make_training_result())
        assert registry._index.get("latest") == entry.version


class TestModelRegistryLoad:
    def test_load_by_tag(self, tmp_path):
        registry = ModelRegistry(tmp_path / "registry")
        entry = registry.register(
            _make_simple_model(), _make_training_result(), tags=["best"]
        )
        model, loaded_entry = registry.load("best")
        assert isinstance(model, tf.keras.Model)
        assert loaded_entry.version == entry.version

    def test_load_latest(self, tmp_path):
        registry = ModelRegistry(tmp_path / "registry")
        registry.register(_make_simple_model(), _make_training_result())
        model, entry = registry.load("latest")
        assert model is not None

    def test_load_nonexistent_raises(self, tmp_path):
        registry = ModelRegistry(tmp_path / "registry")
        with pytest.raises(KeyError, match="not a known tag"):
            registry.load("nonexistent")


class TestModelRegistryListVersions:
    def test_list_all(self, tmp_path):
        registry = ModelRegistry(tmp_path / "registry")
        registry.register(_make_simple_model(), _make_training_result())
        registry.register(_make_simple_model(), _make_training_result())
        entries = registry.list_versions()
        assert len(entries) == 2

    def test_list_by_tag(self, tmp_path):
        registry = ModelRegistry(tmp_path / "registry")
        registry.register(_make_simple_model(), _make_training_result(), tags=["a"])
        registry.register(_make_simple_model(), _make_training_result(), tags=["b"])
        entries = registry.list_versions(tag_filter="a")
        assert len(entries) == 1


class TestModelRegistryTagging:
    def test_tag_and_untag(self, tmp_path):
        registry = ModelRegistry(tmp_path / "registry")
        entry = registry.register(_make_simple_model(), _make_training_result())
        registry.tag(entry.version, "new_tag")
        assert registry._index["new_tag"] == entry.version

        registry.untag(entry.version, "new_tag")
        assert "new_tag" not in registry._index

    def test_tag_nonexistent_raises(self, tmp_path):
        registry = ModelRegistry(tmp_path / "registry")
        with pytest.raises(FileNotFoundError):
            registry.tag("nonexistent_version", "tag")


class TestModelRegistryDelete:
    def test_delete_removes_version(self, tmp_path):
        registry = ModelRegistry(tmp_path / "registry")
        entry = registry.register(_make_simple_model(), _make_training_result())
        registry.delete(entry.version)
        assert not (tmp_path / "registry" / entry.version).exists()

    def test_delete_nonexistent_raises(self, tmp_path):
        registry = ModelRegistry(tmp_path / "registry")
        with pytest.raises(FileNotFoundError):
            registry.delete("nonexistent")


class TestModelRegistryGetBest:
    def test_get_best_min(self, tmp_path):
        registry = ModelRegistry(tmp_path / "registry")
        registry.register(_make_simple_model(), _make_training_result(best_val_loss=0.5))
        registry.register(_make_simple_model(), _make_training_result(best_val_loss=0.1))
        best = registry.get_best(metric="best_val_loss", mode="min")
        assert best.metrics["best_val_loss"] == 0.1

    def test_get_best_empty_raises(self, tmp_path):
        registry = ModelRegistry(tmp_path / "registry")
        with pytest.raises(ValueError, match="empty"):
            registry.get_best()

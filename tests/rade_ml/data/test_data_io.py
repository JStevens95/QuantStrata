"""Unit tests for rade_ml.data.io -- CacheLoader."""
import json
import pickle
import pytest
import pandas as pd

from src.rade_ml.data.io import CacheLoader
from src.rade_ml.validation.exceptions import FileLoadError, FileSaveError


@pytest.fixture(autouse=True)
def clear_cache():
    """Ensure a clean cache for every test."""
    CacheLoader.cache.clear()
    yield
    CacheLoader.cache.clear()


class TestCacheLoaderJson:
    def test_load_json(self, tmp_path):
        p = tmp_path / "data.json"
        p.write_text(json.dumps({"key": "value"}))
        data = CacheLoader.get("test_json", str(p))
        assert data == {"key": "value"}

    def test_cache_hit(self, tmp_path):
        p = tmp_path / "data.json"
        p.write_text(json.dumps({"a": 1}))
        CacheLoader.get("cached", str(p))
        result = CacheLoader.get("cached", str(p))
        assert result == {"a": 1}

    def test_save_json(self, tmp_path):
        p = tmp_path / "out.json"
        CacheLoader.save_data({"x": 42}, str(p))
        with open(p) as f:
            assert json.load(f) == {"x": 42}


class TestCacheLoaderPickle:
    def test_load_pickle(self, tmp_path):
        p = tmp_path / "data.pkl"
        with open(p, "wb") as f:
            pickle.dump([1, 2, 3], f)
        data = CacheLoader.get("pkl_test", str(p))
        assert data == [1, 2, 3]

    def test_save_pickle(self, tmp_path):
        p = tmp_path / "out.pkl"
        CacheLoader.save_data({"a": 1}, str(p))
        with open(p, "rb") as f:
            assert pickle.load(f) == {"a": 1}


class TestCacheLoaderCsv:
    def test_load_csv(self, tmp_path):
        p = tmp_path / "data.csv"
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df.to_csv(p)
        result = CacheLoader.get("csv_test", str(p))
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a", "b"]


class TestCacheLoaderErrors:
    def test_unsupported_extension_raises(self, tmp_path):
        p = tmp_path / "data.xyz"
        p.write_text("hello")
        with pytest.raises(FileLoadError, match="Unsupported"):
            CacheLoader.get("bad", str(p))

    def test_save_missing_cache_key_raises(self, tmp_path):
        with pytest.raises(FileSaveError, match="No cached data"):
            CacheLoader.save("nonexistent", str(tmp_path / "out.json"))


class TestCacheLoaderUtilities:
    def test_update_cache(self):
        CacheLoader.update_cache("mykey", [1, 2, 3])
        assert CacheLoader.get_cached("mykey") == [1, 2, 3]

    def test_reload(self, tmp_path):
        p = tmp_path / "data.json"
        p.write_text(json.dumps({"v": 1}))
        CacheLoader.get("reload_test", str(p))
        p.write_text(json.dumps({"v": 2}))
        data = CacheLoader.reload("reload_test", str(p))
        assert data == {"v": 2}

"""Unit tests for rade_ml.data.result -- DataBuildResult."""
import pytest
import tensorflow as tf

from src.rade_ml.data.result import DataBuildResult


class TestDataBuildResult:
    def test_defaults_are_none(self):
        r = DataBuildResult()
        assert r.train_ds is None
        assert r.val_ds is None
        assert r.test_ds is None
        assert r.metadata == {}

    def test_accepts_datasets(self):
        ds = tf.data.Dataset.from_tensor_slices(([1, 2], [3, 4]))
        r = DataBuildResult(train_ds=ds, metadata={"n_train": 2})
        assert r.train_ds is not None
        assert r.metadata["n_train"] == 2

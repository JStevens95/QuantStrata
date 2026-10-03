"""Unit tests for ``rade_ml_pt.monitoring.loaders``.

Round-trip ``save_feature_baseline`` → ``load_baseline`` to confirm the
JSON-encoded columns are decoded correctly into NumPy arrays — this is
the exact contract every drift consumer relies on, so we exercise the
edge cases the writer documents (all-NaN feature, constant feature).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.rade_ml_pt.monitoring.baselines import save_feature_baseline
from src.rade_ml_pt.monitoring.loaders import load_baseline


@pytest.fixture
def features() -> pd.DataFrame:
    """Four columns that exercise every code path the writer cares about."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "rf_a":        rng.normal(0, 1, 1000),
        "rf_b":        rng.normal(5, 2, 1000),
        "rf_constant": np.full(1000, 3.14),
        "rf_all_nan":  np.full(1000, np.nan),
    })


class TestLoadBaseline:
    def test_roundtrip_preserves_stats(self, features, tmp_path):
        path = tmp_path / "baseline.parquet"
        save_feature_baseline(path, features, cluster_id="c0")

        df    = load_baseline(path)
        rf_a  = df.set_index("feature_name").loc["rf_a"]
        assert rf_a["mean"] == pytest.approx(features["rf_a"].mean(),       rel=1e-4)
        assert rf_a["std"]  == pytest.approx(features["rf_a"].std(ddof=0), rel=1e-4)

    def test_histogram_columns_decoded_to_ndarrays(self, features, tmp_path):
        path = tmp_path / "baseline.parquet"
        save_feature_baseline(path, features, cluster_id="c0")

        df = load_baseline(path)
        for _, row in df.iterrows():
            assert isinstance(row["hist_edges"],  np.ndarray)
            assert isinstance(row["hist_counts"], np.ndarray)

        # rf_a has data → 51 edges + 50 counts (per writer's N_HIST_BINS=50)
        rf_a = df.set_index("feature_name").loc["rf_a"]
        assert rf_a["hist_edges"].shape  == (51,)
        assert rf_a["hist_counts"].shape == (50,)

    def test_all_nan_feature_yields_empty_arrays(self, features, tmp_path):
        path = tmp_path / "baseline.parquet"
        save_feature_baseline(path, features, cluster_id="c0")

        df     = load_baseline(path)
        rf_nan = df.set_index("feature_name").loc["rf_all_nan"]
        assert rf_nan["hist_edges"].shape  == (0,)
        assert rf_nan["hist_counts"].shape == (0,)

    def test_constant_feature_has_well_defined_edges(self, features, tmp_path):
        path = tmp_path / "baseline.parquet"
        save_feature_baseline(path, features, cluster_id="c0")

        df       = load_baseline(path)
        rf_const = df.set_index("feature_name").loc["rf_constant"]
        assert rf_const["hist_edges"].shape  == (51,)
        assert rf_const["hist_counts"].shape == (50,)
        # save_feature_baseline widens the range with a tiny epsilon so
        # np.histogram does not error — confirm the trick worked.
        assert rf_const["hist_edges"][-1] > rf_const["hist_edges"][0]

    def test_original_columns_retained(self, features, tmp_path):
        path = tmp_path / "baseline.parquet"
        save_feature_baseline(path, features, cluster_id="c0")
        df = load_baseline(path)
        # JSON columns stay on the frame for traceability
        assert "hist_edges_json"  in df.columns
        assert "hist_counts_json" in df.columns

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_baseline(tmp_path / "does_not_exist.parquet")

    def test_malformed_parquet_raises_value_error(self, tmp_path):
        path = tmp_path / "wrong.parquet"
        pd.DataFrame({"foo": [1, 2, 3]}).to_parquet(path)
        with pytest.raises(ValueError, match="missing required columns"):
            load_baseline(path)

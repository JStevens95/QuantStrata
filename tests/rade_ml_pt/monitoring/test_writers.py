"""Unit tests for ``rade_ml_pt.monitoring.writers``.

Round-trip writers ↔ readers + JSON sanitisation edge cases.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.rade_ml_pt.monitoring.drift import (
    SEVERITY_INFO,
    SEVERITY_NO_DATA,
)
from src.rade_ml_pt.monitoring.writers import (
    read_drift_summary_json,
    read_drift_table_parquet,
    read_monitoring_manifest_json,
    write_drift_summary_json,
    write_drift_table_parquet,
    write_monitoring_manifest_json,
)


# ═════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════

def _drift_table_fixture(cluster_id: str = "c0") -> pd.DataFrame:
    """Canonical drift table — matches build_drift_table output shape."""
    return pd.DataFrame([
        {
            "cluster_id":    cluster_id,
            "feature_name":  "rf_a",
            "psi":           0.05,
            "js_divergence": 0.01,
            "mean_shift":    0.10,
            "std_ratio":     1.02,
            "severity":      SEVERITY_INFO,
        },
        {
            "cluster_id":    cluster_id,
            "feature_name":  "rf_b",
            "psi":           float("nan"),
            "js_divergence": float("nan"),
            "mean_shift":    float("nan"),
            "std_ratio":     float("nan"),
            "severity":      SEVERITY_NO_DATA,
        },
    ]).astype({
        "cluster_id":    "string",
        "feature_name":  "string",
        "psi":           np.float32,
        "js_divergence": np.float32,
        "mean_shift":    np.float32,
        "std_ratio":     np.float32,
        "severity":      "string",
    })


# ═════════════════════════════════════════════════════════════════════
# write_drift_table_parquet ↔ read_drift_table_parquet
# ═════════════════════════════════════════════════════════════════════

class TestDriftTableParquet:
    def test_roundtrip_preserves_columns(self, tmp_path):
        df  = _drift_table_fixture("c0")
        out = tmp_path / "drift_table.parquet"
        write_drift_table_parquet(
            df, out,
            cluster_id="c0", ensemble_version="ens_v1", run_id="rid",
        )
        back = read_drift_table_parquet(out)
        assert list(back.columns) == list(df.columns)
        assert len(back) == len(df)
        assert (back["feature_name"] == df["feature_name"]).all()
        assert back.loc[0, "psi"] == pytest.approx(0.05, abs=1e-5)

    def test_creates_parent_dirs(self, tmp_path):
        df  = _drift_table_fixture()
        out = tmp_path / "deeply" / "nested" / "drift_table.parquet"
        write_drift_table_parquet(
            df, out,
            cluster_id="c0", ensemble_version="ens_v1", run_id="rid",
        )
        assert out.exists()

    def test_schema_metadata_embedded(self, tmp_path):
        df  = _drift_table_fixture("c0")
        out = tmp_path / "drift_table.parquet"
        write_drift_table_parquet(
            df, out,
            cluster_id="c0", ensemble_version="ens_v1", run_id="rid",
        )
        table = pq.read_table(out)
        meta  = table.schema.metadata or {}
        assert meta.get(b"cluster_id")       == b"c0"
        assert meta.get(b"ensemble_version") == b"ens_v1"
        assert meta.get(b"run_id")           == b"rid"
        assert b"_schema_version" in meta

    def test_missing_columns_raises(self, tmp_path):
        bad = pd.DataFrame([{"cluster_id": "c0", "feature_name": "x"}])  # missing psi etc.
        with pytest.raises(ValueError, match="missing columns"):
            write_drift_table_parquet(
                bad, tmp_path / "x.parquet",
                cluster_id="c0", ensemble_version="ens_v1", run_id="rid",
            )

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_drift_table_parquet(tmp_path / "nope.parquet")


# ═════════════════════════════════════════════════════════════════════
# write_drift_summary_json
# ═════════════════════════════════════════════════════════════════════

class TestDriftSummaryJson:
    def test_roundtrip(self, tmp_path):
        summary = {
            "n_clusters": 3,
            "mean_psi": 0.12,
            "severity": "warn",
            "worst_cluster": "c1",
        }
        out = tmp_path / "drift_summary.json"
        write_drift_summary_json(summary, out)
        back = read_drift_summary_json(out)
        assert back == summary

    def test_nan_coerced_to_null(self, tmp_path):
        summary = {
            "mean_psi": float("nan"),
            "max_psi":  float("inf"),
            "n_clusters": 0,
        }
        out = tmp_path / "drift_summary.json"
        write_drift_summary_json(summary, out)
        raw = out.read_text()
        assert "NaN" not in raw          # NOT in the JSON spec
        assert "Infinity" not in raw
        assert "null" in raw
        back = read_drift_summary_json(out)
        assert back["mean_psi"]   is None
        assert back["max_psi"]    is None
        assert back["n_clusters"] == 0

    def test_numpy_types_normalised(self, tmp_path):
        summary = {
            "n_clusters":   np.int64(3),
            "mean_psi":     np.float32(0.12),
            "feature_names": np.array(["a", "b"]),
        }
        out = tmp_path / "drift_summary.json"
        write_drift_summary_json(summary, out)
        back = read_drift_summary_json(out)
        assert back["n_clusters"] == 3
        assert back["mean_psi"] == pytest.approx(0.12, abs=1e-6)
        assert back["feature_names"] == ["a", "b"]

    def test_path_values_serialised(self, tmp_path):
        summary = {"some_path": tmp_path / "foo.parquet"}
        out = tmp_path / "drift_summary.json"
        write_drift_summary_json(summary, out)
        back = read_drift_summary_json(out)
        assert isinstance(back["some_path"], str)
        assert back["some_path"].endswith("foo.parquet")

    def test_creates_parent_dirs(self, tmp_path):
        summary = {"k": 1}
        out = tmp_path / "deeply" / "nested" / "ds.json"
        write_drift_summary_json(summary, out)
        assert out.exists()

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_drift_summary_json(tmp_path / "nope.json")


# ═════════════════════════════════════════════════════════════════════
# write_monitoring_manifest_json
# ═════════════════════════════════════════════════════════════════════

class TestMonitoringManifestJson:
    def test_roundtrip(self, tmp_path):
        manifest = {
            "schema_version":  1,
            "run_id":          "ens_v1__monitor__2026-05-26T14-30-00Z",
            "ensemble_version": "ens_v1",
            "n_clusters":      3,
            "drift_summary":   {"mean_psi": 0.05},
            "cluster_drift_tables": {
                "c0": "clusters/c0/drift_table.parquet",
            },
            "predictions": None,
        }
        out = tmp_path / "manifest.json"
        write_monitoring_manifest_json(manifest, out)
        back = read_monitoring_manifest_json(out)
        assert back == manifest

    def test_nested_nan_coerced(self, tmp_path):
        manifest = {
            "drift_summary": {
                "mean_psi": float("nan"),
                "nested":   {"deep_nan": float("nan")},
            },
        }
        out = tmp_path / "manifest.json"
        write_monitoring_manifest_json(manifest, out)
        back = read_monitoring_manifest_json(out)
        assert back["drift_summary"]["mean_psi"]            is None
        assert back["drift_summary"]["nested"]["deep_nan"]  is None

    def test_json_is_strictly_valid(self, tmp_path):
        """Confirm the file parses with stdlib json.loads (strict=True)."""
        manifest = {"k": float("nan"), "deep": [float("inf"), 1.0]}
        out = tmp_path / "manifest.json"
        write_monitoring_manifest_json(manifest, out)
        # Default json.loads is strict — would reject NaN/Infinity
        parsed = json.loads(out.read_text())
        assert parsed["k"]      is None
        assert parsed["deep"]   == [None, 1.0]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_monitoring_manifest_json(tmp_path / "nope.json")

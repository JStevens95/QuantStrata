"""Unit tests for ``rade_ml_pt.monitoring.run_paths``.

Pure path-resolution + run-id format tests — no filesystem touching
except for the explicit ``ensure_dirs`` assertion.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.rade_ml_pt.monitoring.run_paths import (
    DRIFT_SUMMARY_FILENAME,
    DRIFT_TABLE_FILENAME,
    MANIFEST_FILENAME,
    MONITORING_RUNS_DIRNAME,
    MONITORING_SUBDIRNAME,
    MonitoringRunPaths,
    monitoring_run_id,
    parse_monitoring_run_id,
    per_run_artifacts_dir,
)


# ═════════════════════════════════════════════════════════════════════
# monitoring_run_id
# ═════════════════════════════════════════════════════════════════════

class TestMonitoringRunId:
    def test_format_with_fixed_timestamp(self):
        ts = datetime(2026, 5, 26, 14, 30, 0, tzinfo=timezone.utc)
        rid = monitoring_run_id("ens_v1", timestamp=ts)
        assert rid == "ens_v1__monitor__2026-05-26T14-30-00Z"

    def test_naive_timestamp_assumed_utc(self):
        ts_naive = datetime(2026, 5, 26, 14, 30, 0)
        rid = monitoring_run_id("ens_v1", timestamp=ts_naive)
        assert rid.endswith("2026-05-26T14-30-00Z")

    def test_now_default_is_zulu_format(self):
        rid = monitoring_run_id("ens_v1")
        # Z suffix + colon-free → safe directory name on every OS
        assert rid.startswith("ens_v1__monitor__")
        assert rid.endswith("Z")
        assert ":" not in rid

    def test_ensemble_version_with_underscores_preserved(self):
        ts = datetime(2026, 5, 26, 14, 30, 0, tzinfo=timezone.utc)
        rid = monitoring_run_id("ens_v1_2_dev", timestamp=ts)
        assert rid.startswith("ens_v1_2_dev__monitor__")


# ═════════════════════════════════════════════════════════════════════
# parse_monitoring_run_id
# ═════════════════════════════════════════════════════════════════════

class TestParseMonitoringRunId:
    def test_roundtrip(self):
        ts  = datetime(2026, 5, 26, 14, 30, 0, tzinfo=timezone.utc)
        rid = monitoring_run_id("ens_v1", timestamp=ts)
        parsed = parse_monitoring_run_id(rid)
        assert parsed == {
            "ensemble_version": "ens_v1",
            "ts":               "2026-05-26T14-30-00Z",
        }

    def test_underscored_version_roundtrip(self):
        ts  = datetime(2026, 5, 26, 14, 30, 0, tzinfo=timezone.utc)
        rid = monitoring_run_id("ens_v1_2_dev", timestamp=ts)
        parsed = parse_monitoring_run_id(rid)
        assert parsed["ensemble_version"] == "ens_v1_2_dev"

    @pytest.mark.parametrize("bad", [
        "",
        "ens_v1__monitor__not-a-ts",
        "ens_v1__infer__2026-05-26T14-30-00Z",  # wrong segment
        "no_segments_at_all",
    ])
    def test_invalid_returns_empty_dict(self, bad):
        assert parse_monitoring_run_id(bad) == {}


# ═════════════════════════════════════════════════════════════════════
# per_run_artifacts_dir
# ═════════════════════════════════════════════════════════════════════

class TestPerRunArtifactsDir:
    def test_basic(self):
        out = per_run_artifacts_dir("/tmp/artifacts", "rid")
        assert out == str(Path("/tmp/artifacts") / MONITORING_RUNS_DIRNAME / "rid")

    def test_accepts_path_obj(self):
        out = per_run_artifacts_dir(Path("/tmp/artifacts"), "rid")
        assert MONITORING_RUNS_DIRNAME in out


# ═════════════════════════════════════════════════════════════════════
# MonitoringRunPaths
# ═════════════════════════════════════════════════════════════════════

class TestMonitoringRunPaths:
    def _paths(self, tmp_path: Path) -> MonitoringRunPaths:
        return MonitoringRunPaths(
            artifacts_dir    = tmp_path,
            run_id           = "ens_v1__monitor__2026-05-26T14-30-00Z",
            ensemble_version = "ens_v1",
        )

    def test_run_dir_layout(self, tmp_path):
        p = self._paths(tmp_path)
        assert p.run_dir == tmp_path / MONITORING_RUNS_DIRNAME / p.run_id

    def test_monitoring_dir_nested(self, tmp_path):
        p = self._paths(tmp_path)
        assert p.monitoring_dir == p.run_dir / MONITORING_SUBDIRNAME

    def test_manifest_path(self, tmp_path):
        p = self._paths(tmp_path)
        assert p.manifest_path == p.monitoring_dir / MANIFEST_FILENAME

    def test_drift_summary_path(self, tmp_path):
        p = self._paths(tmp_path)
        assert p.drift_summary_path == p.monitoring_dir / DRIFT_SUMMARY_FILENAME

    def test_cluster_drift_table_path(self, tmp_path):
        p = self._paths(tmp_path)
        cdp = p.cluster_drift_table_path("c0")
        assert cdp == p.clusters_dir / "c0" / DRIFT_TABLE_FILENAME

    def test_ensure_dirs_creates_layout(self, tmp_path):
        p = self._paths(tmp_path)
        assert not p.monitoring_dir.exists()
        p.ensure_dirs()
        assert p.monitoring_dir.is_dir()
        assert p.clusters_dir.is_dir()

    def test_ensure_dirs_idempotent(self, tmp_path):
        p = self._paths(tmp_path)
        p.ensure_dirs()
        p.ensure_dirs()  # second call must not raise
        assert p.monitoring_dir.is_dir()

"""Happy-path tests for the ``/prism/v1/monitoring`` API router.

Coverage matrix (one test per endpoint; system-failure / 409 transition
tests deferred — see M.4 scope decision):

    Control plane:
        POST /load                   → test_load_returns_n_clusters
        POST /scenarios              → test_load_scenarios_returns_report
        POST /validate               → test_validate_returns_routing_decisions
        POST /run                    → test_run_dispatches_and_completes
        POST /promote                → test_promote_dispatches_and_completes
        GET  /status                 → test_status_reflects_state
        GET  /events                 → test_events_cursor_pagination
        GET  /manifest               → test_active_manifest_reads_after_run

    Data plane:
        GET  /runs                   → test_list_runs_returns_summaries
        GET  /runs/{id}/manifest     → test_historical_manifest_read
        GET  /runs/{id}/drift_summary
                                     → test_drift_summary_read
        GET  /runs/{id}/clusters     → test_cluster_severity_index_read
        GET  /runs/{id}/clusters/{cid}/drift
                                     → test_cluster_drift_table_read

Mocking strategy
----------------
* Control-plane tests monkey-patch ``EnsembleMonitoringPipeline`` inside
  ``services.monitoring_state`` with the same fake pattern test_monitor.py
  uses — keeps the API tests independent of real ensemble artifacts.
* Data-plane tests write a real monitoring run directory tree (JSON +
  parquet) so the :class:`MonitoringResultReader` exercises its real
  on-disk layout code.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.rade_ml_pt.core.types import InferenceResult
from src.rade_ml_pt.ensemble.api.dependencies import (
    set_monitoring_result_reader,
    set_monitoring_state_manager,
)
from src.rade_ml_pt.ensemble.api.routers.monitoring import router as monitoring_router
from src.rade_ml_pt.ensemble.api.services.monitoring_reader import MonitoringResultReader
from src.rade_ml_pt.ensemble.api.services.monitoring_state import MonitoringStateManager
from src.rade_ml_pt.monitoring.run_paths import (
    DRIFT_SUMMARY_FILENAME,
    DRIFT_TABLE_FILENAME,
    MANIFEST_FILENAME,
    MONITORING_RUNS_DIRNAME,
    MONITORING_SUBDIRNAME,
)


# ═════════════════════════════════════════════════════════════════════
# Fakes — minimum surface area to drive the router
# ═════════════════════════════════════════════════════════════════════

class _FakeLoadedScenariosReport:
    """Minimal substitute for LoadedScenariosReport."""

    def __init__(self, new_scenario_dir: str, scenario_labels: List[str]):
        self.new_scenario_dir  = new_scenario_dir
        self.risk_factor_names = ["rf_x", "rf_y"]
        self.n_risk_factors    = 2
        self.n_scenarios       = len(scenario_labels)
        self.scenario_labels   = list(scenario_labels)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "new_scenario_dir":  self.new_scenario_dir,
            "risk_factor_names": list(self.risk_factor_names),
            "n_risk_factors":    self.n_risk_factors,
            "n_scenarios":       self.n_scenarios,
            "scenario_labels":   list(self.scenario_labels),
        }


class _FakeRoutingDecision:
    def __init__(self, cluster_id: str, is_affected: bool):
        self.cluster_id                = cluster_id
        self.is_affected               = is_affected
        self.intersecting_risk_factors = ["rf_x"] if is_affected else []
        self.n_elementary_trades       = 5
        self.n_target_trades           = 3
        self.missing_scenario_labels   = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id":                self.cluster_id,
            "is_affected":               self.is_affected,
            "intersecting_risk_factors": list(self.intersecting_risk_factors),
            "n_elementary_trades":       self.n_elementary_trades,
            "n_target_trades":           self.n_target_trades,
            "missing_scenario_labels":   list(self.missing_scenario_labels),
        }


class _FakeValidationReport:
    def __init__(
        self,
        ensemble_version:  str,
        scenario_labels:   List[str],
        cluster_decisions: List[_FakeRoutingDecision],
    ):
        self.ensemble_version  = ensemble_version
        self.n_scenarios       = len(scenario_labels)
        self.scenario_labels   = list(scenario_labels)
        self.cluster_decisions = list(cluster_decisions)
        self.errors:   List[str] = []
        self.warnings: List[str] = []

    @property
    def is_valid(self) -> bool:
        return not self.errors

    @property
    def affected_cluster_ids(self) -> List[str]:
        return [d.cluster_id for d in self.cluster_decisions if d.is_affected]

    @property
    def unaffected_cluster_ids(self) -> List[str]:
        return [d.cluster_id for d in self.cluster_decisions if not d.is_affected]

    @property
    def affected_count(self) -> int:
        return len(self.affected_cluster_ids)

    @property
    def unaffected_count(self) -> int:
        return len(self.unaffected_cluster_ids)


class _FakeMonitoringResult:
    """Substitute for monitor.MonitoringResult — only fields the router reads."""

    def __init__(
        self,
        run_id:          str,
        artifacts_dir:   Path,
        manifest_path:   Path,
        n_scenarios:     int = 5,
        n_clusters:      int = 2,
        n_affected:      int = 1,
        n_unaffected:    int = 1,
        wall_seconds:    float = 0.01,
    ):
        self.run_id           = run_id
        self.ensemble_version = "test_v1"
        self.artifacts_dir    = artifacts_dir
        self.manifest_path    = manifest_path
        self.portfolio_summary: Dict[str, Any] = {"severity": "info", "mean_psi": 0.05}
        self.drift_tables:      Dict[str, Any] = {}
        self.n_scenarios       = n_scenarios
        self.n_clusters        = n_clusters
        self.n_affected        = n_affected
        self.n_unaffected      = n_unaffected
        self.wall_seconds      = wall_seconds


class _FakePromoteResult:
    def __init__(
        self,
        run_id:          str,
        predictions_dir: Path,
        manifest_path:   Path,
    ):
        self.run_id               = run_id
        self.predictions_dir      = predictions_dir
        self.manifest_path        = manifest_path
        self.inference_result     = InferenceResult()
        self.promoted_at          = "2026-05-01T00:00:00+00:00"
        self.wall_seconds         = 0.01
        self.n_clusters_predicted = 1


class _FakeMonitoringPipeline:
    """In-memory replacement for EnsembleMonitoringPipeline.

    Public surface mirrors the real pipeline only where the router
    actually calls in: ``load``, ``load_scenarios``,
    ``validate_scenarios``, ``compute_drift``, ``promote_to_predictions``,
    plus the ``_inference_pipeline._ensemble.members`` access path
    the ``/load`` endpoint uses to count clusters.
    """

    def __init__(
        self,
        ensemble_config,
        ensemble_version: str,
        session=None,
        *,
        on_event=None,
    ):
        self.config           = ensemble_config
        self.ensemble_version = ensemble_version
        self._on_event        = on_event

        # Emulate the ``_inference_pipeline`` attribute the router peeks at.
        self._inference_pipeline = _FakeInnerInferencePipeline(ensemble_config)

        self._loaded:    bool                                    = False
        self._scenarios: Optional[_FakeLoadedScenariosReport]    = None
        self._report:    Optional[_FakeValidationReport]         = None
        self._drift_done: bool                                   = False
        self._promote_done: bool                                 = False

    # ── Stages the router drives ──────────────────────────────────

    def load(self) -> None:
        self._loaded = True
        if self._on_event is not None:
            self._on_event({
                "id":     "evt-load",
                "stage":  "monitoring",
                "phase":  "Ensemble loaded",
                "status": "ok",
                "ts":     "2026-05-01T00:00:00+00:00",
                "target": "test_v1",
                "detail": None,
            })

    def load_scenarios(self, new_scenario_dir=None) -> _FakeLoadedScenariosReport:
        self._scenarios = _FakeLoadedScenariosReport(
            new_scenario_dir = str(new_scenario_dir),
            scenario_labels  = ["s0", "s1", "s2", "s3", "s4"],
        )
        return self._scenarios

    def validate_scenarios(self) -> _FakeValidationReport:
        self._report = _FakeValidationReport(
            ensemble_version  = self.ensemble_version,
            scenario_labels   = ["s0", "s1", "s2", "s3", "s4"],
            cluster_decisions = [
                _FakeRoutingDecision("c0", is_affected=True),
                _FakeRoutingDecision("c1", is_affected=False),
            ],
        )
        return self._report

    def compute_drift(self) -> _FakeMonitoringResult:
        if not self._loaded or self._scenarios is None or self._report is None:
            raise RuntimeError("Stages not run in order")
        # Materialise a real monitoring run on disk so the active
        # ``/manifest`` endpoint can read it back.
        run_id = "test_v1__monitor__2026-05-01T00-00-00Z"
        artifacts_dir = Path(self.config.artifacts_dir)
        run_dir       = artifacts_dir / MONITORING_RUNS_DIRNAME / run_id
        mon_dir       = run_dir / MONITORING_SUBDIRNAME
        mon_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "run_id":           run_id,
            "ensemble_version": self.ensemble_version,
            "created_at":       "2026-05-01T00:00:00+00:00",
            "n_scenarios":      5,
            "n_clusters":       2,
            "n_clusters_affected":   1,
            "n_clusters_unaffected": 1,
            "drift_summary":    {"severity": "info", "mean_psi": 0.05, "max_psi": 0.1},
            "predictions":      None,
        }
        (mon_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest))
        self._drift_done = True
        return _FakeMonitoringResult(
            run_id        = run_id,
            artifacts_dir = run_dir,
            manifest_path = mon_dir / MANIFEST_FILENAME,
        )

    def promote_to_predictions(self) -> _FakePromoteResult:
        if not self._drift_done:
            raise RuntimeError("compute_drift not called")
        if self._promote_done:
            raise RuntimeError("Promote already done")
        artifacts_dir = Path(self.config.artifacts_dir)
        run_id = "test_v1__monitor__2026-05-01T00-00-00Z"
        run_dir = artifacts_dir / MONITORING_RUNS_DIRNAME / run_id
        mon_dir = run_dir / MONITORING_SUBDIRNAME
        predictions_dir = mon_dir / "inference"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        # Rewrite the manifest with the predictions block.
        manifest = json.loads((mon_dir / MANIFEST_FILENAME).read_text())
        manifest["predictions"] = {"predictions_dir": "monitoring/inference"}
        (mon_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest))
        self._promote_done = True
        return _FakePromoteResult(
            run_id          = run_id,
            predictions_dir = predictions_dir,
            manifest_path   = mon_dir / MANIFEST_FILENAME,
        )


class _FakeInnerEnsemble:
    """Substitute for the ensemble model — only ``members`` is read."""
    def __init__(self):
        self.members = {"c0": object(), "c1": object()}


class _FakeInnerInferencePipeline:
    """Stand-in for the composed EnsembleInferencePipeline."""
    def __init__(self, ensemble_config):
        self.config    = ensemble_config
        self._ensemble = _FakeInnerEnsemble()


# ═════════════════════════════════════════════════════════════════════
# Fixtures
# ═════════════════════════════════════════════════════════════════════

@pytest.fixture
def tmp_registry(tmp_path):
    """Build a minimal registry + artifacts tree so Settings can resolve."""
    registry = tmp_path / "registry"
    artifacts = tmp_path / "artifacts"
    (registry / "ensemble" / "test_v1").mkdir(parents=True)
    (registry / "ensemble" / "index.json").write_text(json.dumps({"latest": "test_v1"}))
    artifacts.mkdir(parents=True)
    return registry, artifacts


@pytest.fixture
def settings(tmp_registry):
    """Build PRISM Settings against the tmp registry / artifacts."""
    from src.rade_ml_pt.ensemble.api.config import Settings, set_settings
    registry, artifacts = tmp_registry
    s = Settings(
        artifacts_dir    = str(artifacts),
        registry_dir     = str(registry),
        ensemble_version = "latest",
    )
    set_settings(s)
    return s


@pytest.fixture
def patch_monitoring_pipeline(monkeypatch):
    """Swap the production EnsembleMonitoringPipeline for the fake."""
    from src.rade_ml_pt.ensemble.api.services import monitoring_state
    monkeypatch.setattr(
        monitoring_state, "EnsembleMonitoringPipeline", _FakeMonitoringPipeline,
    )
    return monitoring_state


@pytest.fixture
def app(settings, patch_monitoring_pipeline):
    """Build a minimal FastAPI app with the monitoring router mounted.

    Each test gets its own state manager + result reader (no cross-
    test leakage) — registered via the same dependency setters
    ``app.py``'s lifespan uses in production.
    """
    set_monitoring_state_manager(MonitoringStateManager())
    set_monitoring_result_reader(MonitoringResultReader(settings.artifacts_dir))

    a = FastAPI()
    a.include_router(monitoring_router)
    return a


@pytest.fixture
def client(app):
    return TestClient(app)


# ═════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════

def _wait_for_status(client: TestClient, target: str, timeout: float = 5.0) -> Dict[str, Any]:
    """Poll ``/status`` until ``status==target`` or timeout.

    Both ``/run`` and ``/promote`` dispatch on background threads, so
    the HTTP response returns before the work completes.  The tests
    follow the same polling pattern the real UI uses.
    """
    deadline = time.monotonic() + timeout
    last: Dict[str, Any] = {}
    while time.monotonic() < deadline:
        r = client.get("/prism/v1/monitoring/status")
        assert r.status_code == 200, r.text
        last = r.json()
        if last.get("status") == target:
            return last
        if last.get("status") == "failed":
            raise AssertionError(f"Run failed: {last.get('last_error')}")
        time.sleep(0.02)
    raise AssertionError(f"Timed out waiting for status={target}; last={last}")


def _walk_to_validated(client: TestClient, tmp_path: Path) -> None:
    """Drive the active run through load → scenarios → validate."""
    r1 = client.post("/prism/v1/monitoring/load")
    assert r1.status_code == 200, r1.text
    r2 = client.post(
        "/prism/v1/monitoring/scenarios",
        json={"new_scenario_dir": str(tmp_path / "scn")},
    )
    assert r2.status_code == 200, r2.text
    r3 = client.post("/prism/v1/monitoring/validate")
    assert r3.status_code == 200, r3.text


def _write_fixture_run(artifacts_dir: Path, run_id: str, with_predictions: bool = False) -> Path:
    """Write a real monitoring run directory tree on disk for data-plane tests."""
    run_dir = artifacts_dir / MONITORING_RUNS_DIRNAME / run_id
    mon_dir = run_dir / MONITORING_SUBDIRNAME
    mon_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id":           run_id,
        "ensemble_version": "test_v1",
        "created_at":       "2026-05-01T00:00:00+00:00",
        "n_scenarios":      5,
        "n_clusters":       2,
        "n_clusters_affected":   1,
        "n_clusters_unaffected": 1,
        "drift_summary":    {"severity": "info", "mean_psi": 0.05, "max_psi": 0.10},
        "predictions":      ({"predictions_dir": "monitoring/inference"} if with_predictions else None),
    }
    (mon_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest))

    drift_summary = {
        "severity":   "info",
        "mean_psi":   0.05,
        "max_psi":    0.10,
        "n_clusters": 2,
        "clusters":   [
            {"cluster_id": "c0", "severity": "info", "max_psi": 0.10, "mean_psi": 0.05, "n_features": 4},
            {"cluster_id": "c1", "severity": "no_data", "max_psi": None, "mean_psi": None, "n_features": 0},
        ],
    }
    (mon_dir / DRIFT_SUMMARY_FILENAME).write_text(json.dumps(drift_summary))

    cluster_dir = mon_dir / "clusters" / "c0"
    cluster_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "cluster_id":    ["c0"] * 4,
        "feature_name":  ["trade_0", "trade_1", "trade_2", "trade_3"],
        "psi":           [0.01, 0.03, 0.06, 0.12],
        "js_divergence": [0.005, 0.012, 0.022, 0.045],
        "mean_shift":    [0.0, 0.1, 0.2, 0.3],
        "std_ratio":     [1.0, 1.05, 1.1, 1.2],
        "severity":      ["info", "info", "warn", "warn"],
    })
    pq.write_table(pa.Table.from_pandas(df), cluster_dir / DRIFT_TABLE_FILENAME)
    return run_dir


# ═════════════════════════════════════════════════════════════════════
# Control-plane tests
# ═════════════════════════════════════════════════════════════════════

def test_load_returns_n_clusters(client):
    r = client.post("/prism/v1/monitoring/load")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"]           == "loaded"
    assert body["ensemble_version"] == "test_v1"
    assert body["n_clusters"]       == 2
    assert body["run_id"].startswith("test_v1__monitor__")


def test_load_scenarios_returns_report(client, tmp_path):
    client.post("/prism/v1/monitoring/load")
    r = client.post(
        "/prism/v1/monitoring/scenarios",
        json={"new_scenario_dir": str(tmp_path / "scn")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_scenarios"]       == 5
    assert body["scenario_labels"]   == ["s0", "s1", "s2", "s3", "s4"]
    assert body["n_risk_factors"]    == 2
    assert "rf_x" in body["risk_factor_names"]


def test_validate_returns_routing_decisions(client, tmp_path):
    client.post("/prism/v1/monitoring/load")
    client.post(
        "/prism/v1/monitoring/scenarios",
        json={"new_scenario_dir": str(tmp_path / "scn")},
    )
    r = client.post("/prism/v1/monitoring/validate")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["is_valid"]               is True
    assert body["n_scenarios"]            == 5
    assert body["affected_count"]         == 1
    assert body["unaffected_count"]       == 1
    assert body["affected_cluster_ids"]   == ["c0"]
    assert body["unaffected_cluster_ids"] == ["c1"]
    assert len(body["cluster_decisions"]) == 2


def test_run_dispatches_and_completes(client, tmp_path):
    _walk_to_validated(client, tmp_path)
    r = client.post("/prism/v1/monitoring/run", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "running"
    assert "monitoring_runs" in body["artifacts_dir"]

    final = _wait_for_status(client, "complete")
    assert final["manifest_path"] is not None
    assert Path(final["manifest_path"]).exists()


def test_promote_dispatches_and_completes(client, tmp_path):
    _walk_to_validated(client, tmp_path)
    client.post("/prism/v1/monitoring/run", json={})
    _wait_for_status(client, "complete")

    r = client.post("/prism/v1/monitoring/promote")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "promoting"

    final = _wait_for_status(client, "promoted")
    assert final["predictions_dir"] is not None
    assert Path(final["predictions_dir"]).exists()


def test_status_reflects_state(client):
    r0 = client.get("/prism/v1/monitoring/status")
    assert r0.status_code == 200
    assert r0.json()["has_active_run"] is False

    client.post("/prism/v1/monitoring/load")
    r1 = client.get("/prism/v1/monitoring/status")
    body = r1.json()
    assert body["has_active_run"]   is True
    assert body["status"]           == "loaded"
    assert body["ensemble_version"] == "test_v1"


def test_events_cursor_pagination(client):
    r0 = client.get("/prism/v1/monitoring/events", params={"cursor": 0})
    assert r0.status_code == 200
    # No active run yet — empty payload with cursor=0.
    assert r0.json() == {"events": [], "next_cursor": 0}

    client.post("/prism/v1/monitoring/load")
    r1 = client.get("/prism/v1/monitoring/events", params={"cursor": 0})
    body = r1.json()
    assert body["next_cursor"] >= 1
    assert len(body["events"])  == body["next_cursor"]
    # Polling again with the returned cursor should yield zero new events.
    r2 = client.get("/prism/v1/monitoring/events", params={"cursor": body["next_cursor"]})
    body2 = r2.json()
    assert body2["events"] == []
    assert body2["next_cursor"] == body["next_cursor"]


def test_active_manifest_reads_after_run(client, tmp_path):
    _walk_to_validated(client, tmp_path)
    client.post("/prism/v1/monitoring/run", json={})
    _wait_for_status(client, "complete")
    r = client.get("/prism/v1/monitoring/manifest")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["manifest"]["ensemble_version"] == "test_v1"
    assert body["manifest"]["n_clusters"]       == 2


# ═════════════════════════════════════════════════════════════════════
# Data-plane tests
# ═════════════════════════════════════════════════════════════════════

def test_list_runs_returns_summaries(client, settings):
    artifacts = Path(settings.artifacts_dir)
    _write_fixture_run(artifacts, "test_v1__monitor__2026-05-01T00-00-00Z")
    _write_fixture_run(
        artifacts, "test_v1__monitor__2026-05-02T00-00-00Z",
        with_predictions=True,
    )

    r = client.get("/prism/v1/monitoring/runs")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == 2
    # Most-recent first by run-id descending.
    assert body["runs"][0]["run_id"].endswith("2026-05-02T00-00-00Z")
    assert body["runs"][0]["has_predictions"] is True
    assert body["runs"][1]["has_predictions"] is False


def test_historical_manifest_read(client, settings):
    artifacts = Path(settings.artifacts_dir)
    run_id = "test_v1__monitor__2026-05-01T00-00-00Z"
    _write_fixture_run(artifacts, run_id)

    r = client.get(f"/prism/v1/monitoring/runs/{run_id}/manifest")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["run_id"] == run_id
    assert body["manifest"]["drift_summary"]["severity"] == "info"


def test_drift_summary_read(client, settings):
    artifacts = Path(settings.artifacts_dir)
    run_id = "test_v1__monitor__2026-05-01T00-00-00Z"
    _write_fixture_run(artifacts, run_id)

    r = client.get(f"/prism/v1/monitoring/runs/{run_id}/drift_summary")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["summary"]["severity"]   == "info"
    assert body["summary"]["n_clusters"] == 2


def test_cluster_severity_index_read(client, settings):
    artifacts = Path(settings.artifacts_dir)
    run_id = "test_v1__monitor__2026-05-01T00-00-00Z"
    _write_fixture_run(artifacts, run_id)

    r = client.get(f"/prism/v1/monitoring/runs/{run_id}/clusters")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == 2
    cids = {row["cluster_id"] for row in body["rows"]}
    assert cids == {"c0", "c1"}
    # Coercion: cluster c1 has null PSI in the fixture → must serialise to None.
    c1_row = next(row for row in body["rows"] if row["cluster_id"] == "c1")
    assert c1_row["max_psi"]  is None
    assert c1_row["severity"] == "no_data"


def test_cluster_drift_table_read(client, settings):
    artifacts = Path(settings.artifacts_dir)
    run_id = "test_v1__monitor__2026-05-01T00-00-00Z"
    _write_fixture_run(artifacts, run_id)

    r = client.get(f"/prism/v1/monitoring/runs/{run_id}/clusters/c0/drift")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["cluster_id"]   == "c0"
    assert body["n_features"]   == 4
    assert {row["feature_name"] for row in body["rows"]} == {
        "trade_0", "trade_1", "trade_2", "trade_3",
    }
    # Severity is preserved verbatim from the parquet.
    sev_for_trade_3 = next(
        row["severity"] for row in body["rows"] if row["feature_name"] == "trade_3"
    )
    assert sev_for_trade_3 == "warn"

"""Unit + integration tests for ``EnsembleMonitoringPipeline``.

Strategy:
* Test each helper (``_today_features``, ``_load_cluster_baseline``,
  ``_init_run_paths``, ``_build_manifest``, ``_write_run_artifacts``)
  in isolation with synthetic inputs.
* One end-to-end ``run()`` test stitches the helpers together via a
  monkey-patched ``EnsembleInferencePipeline`` so we never touch the
  real ensemble loader or any model.

Mocking convention: we monkey-patch the *names* the module imported
(``monitor.EnsembleInferencePipeline``, etc.) so the real classes are
free to evolve without breaking these tests.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from src.rade_ml_pt.core.types import InferenceResult
from src.rade_ml_pt.monitoring.baselines import save_feature_baseline
from src.rade_ml_pt.monitoring.drift import (
    SEVERITY_CRITICAL,
    SEVERITY_INFO,
    SEVERITY_NO_DATA,
)
from src.rade_ml_pt.monitoring.run_paths import MonitoringRunPaths


# ═════════════════════════════════════════════════════════════════════
# Lightweight stand-ins for the inference dataclasses
# ═════════════════════════════════════════════════════════════════════

@dataclass
class _FakeCtx:
    """Substitute for InferenceContext — only the fields monitor.py touches."""
    elementary_pnl:       Optional[pd.DataFrame] = None
    _cluster_assets_path: Optional[Path]         = None


@dataclass
class _FakeRoutingDecision:
    """Substitute for ClusterRoutingDecision."""
    cluster_id:               str
    is_affected:              bool
    missing_scenario_labels:  List[str] = field(default_factory=list)


@dataclass
class _FakeValidationReport:
    """Substitute for ValidationReport — exposes the fields run() reads."""
    ensemble_version:   str
    n_scenarios:        int
    scenario_labels:    List[str]
    cluster_decisions:  List[_FakeRoutingDecision]
    errors:             List[str] = field(default_factory=list)
    warnings:           List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    @property
    def affected_count(self) -> int:
        return sum(d.is_affected for d in self.cluster_decisions)

    @property
    def unaffected_count(self) -> int:
        return sum(not d.is_affected for d in self.cluster_decisions)


@dataclass
class _FakeLoadedScenarios:
    new_scenario_dir:  str
    n_scenarios:       int
    scenario_labels:   List[str]


class _FakeInferencePipeline:
    """In-memory replacement for ``EnsembleInferencePipeline``.

    Exposes the same private attrs ``monitor.py`` reads
    (``_inference_contexts``, ``_validation_report``, etc.) plus the
    three public methods (``load``, ``load_scenarios``,
    ``validate_scenarios``) the monitoring pipeline calls.
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
        self._session         = session
        self._inference_contexts: Dict[str, _FakeCtx]               = {}
        self._validation_report:  Optional[_FakeValidationReport]   = None
        self._loaded_scenarios:   Optional[_FakeLoadedScenarios]    = None
        self._new_scenario_shocks: Optional[Dict[str, Any]]         = None

        # Test hooks — populated by the test BEFORE run() is called.
        self._test_contexts:        Dict[str, _FakeCtx]              = {}
        self._test_decisions:       List[_FakeRoutingDecision]       = []
        self._test_scenario_labels: List[str]                        = []
        # Override for promote tests — set to a callable that returns
        # an InferenceResult and optionally drops files in
        # ``self.config.artifacts_dir / 'inference'`` to emulate
        # run_inference's real side effects.
        self._test_run_inference: Optional[Callable[[], InferenceResult]] = None

    def load(self) -> None:
        self._inference_contexts = dict(self._test_contexts)

    def load_scenarios(self, new_scenario_dir=None) -> None:
        self._loaded_scenarios = _FakeLoadedScenarios(
            new_scenario_dir = str(new_scenario_dir),
            n_scenarios      = len(self._test_scenario_labels),
            scenario_labels  = list(self._test_scenario_labels),
        )
        self._new_scenario_shocks = {"rf_x": {lab: 0.01 for lab in self._test_scenario_labels}}

    def validate_scenarios(self) -> _FakeValidationReport:
        self._validation_report = _FakeValidationReport(
            ensemble_version  = self.ensemble_version,
            n_scenarios       = len(self._test_scenario_labels),
            scenario_labels   = list(self._test_scenario_labels),
            cluster_decisions = list(self._test_decisions),
        )
        return self._validation_report

    def run_inference(self) -> InferenceResult:
        """Test-side stub for run_inference.

        Mirrors the real method's side-effect contract: reads
        ``self.config.artifacts_dir`` at start (i.e. AFTER the
        monitoring pipeline's stash-swap), so dropped artifacts
        land at the swapped location.  The actual artifact-drop is
        delegated to ``self._test_run_inference`` (set by the test);
        the default no-op returns an empty InferenceResult.
        """
        if self._test_run_inference is not None:
            return self._test_run_inference()
        return InferenceResult()


# ═════════════════════════════════════════════════════════════════════
# Fixtures
# ═════════════════════════════════════════════════════════════════════

@pytest.fixture
def patch_inference_pipeline(monkeypatch):
    """Swap the production EnsembleInferencePipeline for the in-memory fake."""
    from src.rade_ml_pt.pipelines.ensemble import monitor
    monkeypatch.setattr(monitor, "EnsembleInferencePipeline", _FakeInferencePipeline)
    return monitor


@pytest.fixture
def ensemble_config(tmp_path):
    """Minimal EnsembleConfig substitute — monitor.py only reads artifacts_dir."""
    return SimpleNamespace(
        registry_dir  = str(tmp_path / "registry"),
        artifacts_dir = str(tmp_path / "artifacts"),
        metadata      = {"inference": {}},
    )


@pytest.fixture
def baseline_parquet_path(tmp_path):
    """Write a real baseline parquet on disk for the unaffected-path test."""
    rng = np.random.default_rng(0)
    features = pd.DataFrame({
        "trade_0": rng.normal(0, 1, 5000),
        "trade_1": rng.normal(2, 1, 5000),
    })
    version_dir = tmp_path / "registry" / "cluster-c0-v1"
    out = version_dir / "monitoring" / "baseline_feature_stats.parquet"
    save_feature_baseline(out, features, cluster_id="c0")
    return out


# ═════════════════════════════════════════════════════════════════════
# Helpers — pipeline construction
# ═════════════════════════════════════════════════════════════════════

def _scenario_labels(n: int) -> List[str]:
    return [f"s{i}" for i in range(n)]


def _historical_pnl(scenario_labels: List[str], rng: np.random.Generator) -> pd.DataFrame:
    """Synthetic historical elementary PnL — index = scenario labels."""
    return pd.DataFrame(
        {
            "trade_0": rng.normal(0, 1, len(scenario_labels)),
            "trade_1": rng.normal(2, 1, len(scenario_labels)),
        },
        index=scenario_labels,
    )


def _make_pipeline(monitor_module, ensemble_config):
    """Construct EnsembleMonitoringPipeline with the patched fake injected."""
    return monitor_module.EnsembleMonitoringPipeline(
        ensemble_config  = ensemble_config,
        ensemble_version = "ens_v1",
    )


# ═════════════════════════════════════════════════════════════════════
# Helper-level tests
# ═════════════════════════════════════════════════════════════════════

class TestTodayFeatures:
    """``_today_features`` — unaffected path is pure; affected path delegates."""

    def test_unaffected_path_returns_historical_slice(self):
        from src.rade_ml_pt.pipelines.ensemble.monitor import EnsembleMonitoringPipeline

        labels = _scenario_labels(5)
        rng    = np.random.default_rng(0)
        ctx    = _FakeCtx(elementary_pnl=_historical_pnl(labels, rng))
        dec    = _FakeRoutingDecision(cluster_id="c0", is_affected=False)

        out = EnsembleMonitoringPipeline._today_features(
            ctx=ctx, decision=dec, shocks={}, scenario_labels=labels,
        )
        assert list(out.index) == labels
        assert set(out.columns) == {"trade_0", "trade_1"}

    def test_unaffected_path_raises_without_elementary_pnl(self):
        from src.rade_ml_pt.pipelines.ensemble.monitor import EnsembleMonitoringPipeline

        ctx = _FakeCtx(elementary_pnl=None)
        dec = _FakeRoutingDecision(cluster_id="c0", is_affected=False)
        with pytest.raises(RuntimeError, match="no elementary_pnl"):
            EnsembleMonitoringPipeline._today_features(
                ctx=ctx, decision=dec, shocks={}, scenario_labels=["s0"],
            )

    def test_affected_path_delegates_to_hybrid_builder(self, monkeypatch):
        """The affected path calls HybridGnnRnnInferencePipeline.build_new_scenario_inputs.

        We inject a fake module into ``sys.modules`` before the deferred
        import inside ``_today_features`` runs, so this test works
        even in environments without the hybrid_gnn_rnn dependency
        chain (``rade_sr.market_data_manager`` etc.).
        """
        import sys
        from src.rade_ml_pt.pipelines.ensemble.monitor import EnsembleMonitoringPipeline

        labels = _scenario_labels(3)
        synthetic_pnl = pd.DataFrame(
            {"trade_0": [0.1, 0.2, 0.3]},
            index=labels,
        )

        def _fake_builder(*, ctx, new_scenario_shocks, scenario_labels, is_affected):
            assert is_affected is True
            return {"inputs": SimpleNamespace(elementary_pnl=synthetic_pnl)}

        fake_module = SimpleNamespace(
            HybridGnnRnnInferencePipeline=SimpleNamespace(
                build_new_scenario_inputs=_fake_builder,
            ),
        )
        monkeypatch.setitem(
            sys.modules,
            "src.rade_ml_pt.pipelines.hybrid_gnn_rnn.infer",
            fake_module,
        )

        out = EnsembleMonitoringPipeline._today_features(
            ctx              = _FakeCtx(),
            decision         = _FakeRoutingDecision(cluster_id="c0", is_affected=True),
            shocks           = {"rf_x": {"s0": 0.01}},
            scenario_labels  = labels,
        )
        pd.testing.assert_frame_equal(out, synthetic_pnl)


class TestLoadClusterBaseline:
    """``_load_cluster_baseline`` — resolution via ``_cluster_assets_path.parent``."""

    def test_loads_real_baseline_via_ctx_path(self, baseline_parquet_path):
        from src.rade_ml_pt.pipelines.ensemble.monitor import EnsembleMonitoringPipeline

        version_dir = baseline_parquet_path.parent.parent
        # _cluster_assets_path.parent must equal version_dir
        ctx = _FakeCtx(_cluster_assets_path=version_dir / "cluster_assets.joblib")

        df = EnsembleMonitoringPipeline._load_cluster_baseline(ctx, cluster_id="c0")
        assert "hist_edges"  in df.columns
        assert "hist_counts" in df.columns
        assert set(df["feature_name"]) == {"trade_0", "trade_1"}

    def test_missing_assets_path_raises_file_not_found(self):
        from src.rade_ml_pt.pipelines.ensemble.monitor import EnsembleMonitoringPipeline

        ctx = _FakeCtx(_cluster_assets_path=None)
        with pytest.raises(FileNotFoundError, match="no _cluster_assets_path"):
            EnsembleMonitoringPipeline._load_cluster_baseline(ctx, cluster_id="c0")


class TestInitRunPaths:
    def test_creates_layout_and_returns_paths(self, patch_inference_pipeline, ensemble_config):
        pipe = _make_pipeline(patch_inference_pipeline, ensemble_config)
        paths = pipe._init_run_paths()
        assert isinstance(paths, MonitoringRunPaths)
        assert paths.monitoring_dir.is_dir()
        assert paths.clusters_dir.is_dir()
        assert paths.run_id.startswith("ens_v1__monitor__")


# ═════════════════════════════════════════════════════════════════════
# End-to-end run() — unaffected cluster against real baseline
# ═════════════════════════════════════════════════════════════════════

class TestRunEndToEnd:
    def test_drift_only_with_unaffected_cluster_writes_all_artifacts(
        self, patch_inference_pipeline, ensemble_config, tmp_path,
    ):
        # Build a real baseline parquet for cluster c0.
        rng = np.random.default_rng(0)
        training_features = pd.DataFrame({
            "trade_0": rng.normal(0, 1, 5000),
            "trade_1": rng.normal(0, 1, 5000),
        })
        version_dir = Path(ensemble_config.registry_dir) / "cluster-c0-v1"
        save_feature_baseline(
            version_dir / "monitoring" / "baseline_feature_stats.parquet",
            features   = training_features,
            cluster_id = "c0",
        )

        # 500 today obs against 5000 baseline obs keeps sample-size
        # noise in PSI low enough for stable assertions; under-sampling
        # alone can push PSI above the info threshold even when the
        # underlying distribution is identical.
        labels    = _scenario_labels(500)
        today_pnl = pd.DataFrame(
            {
                "trade_0": rng.normal(0, 1, len(labels)),
                "trade_1": rng.normal(0, 1, len(labels)),
            },
            index=labels,
        )

        pipe = _make_pipeline(patch_inference_pipeline, ensemble_config)
        # Inject the test fixtures BEFORE calling run() — the fake
        # inference pipeline copies them when load() / load_scenarios()
        # are called.
        fake = pipe._inference_pipeline
        fake._test_contexts = {
            "c0": _FakeCtx(
                elementary_pnl       = today_pnl,
                _cluster_assets_path = version_dir / "cluster_assets.joblib",
            ),
        }
        fake._test_decisions       = [_FakeRoutingDecision("c0", is_affected=False)]
        fake._test_scenario_labels = labels

        result = pipe.run(new_scenario_dir=tmp_path / "scenarios")

        # ── MonitoringResult shape ─────────────────────────────────
        assert result.run_id.startswith("ens_v1__monitor__")
        assert result.n_clusters     == 1
        assert result.n_affected     == 0
        assert result.n_unaffected   == 1
        assert result.n_scenarios    == 500
        assert "c0" in result.drift_tables

        # ── Portfolio summary ──────────────────────────────────────
        # The point of this test is pipeline wiring, not drift
        # numerics — only assert that summary keys are populated and
        # severity comes from the canonical set.
        summary = result.portfolio_summary
        assert summary["n_clusters"]    == 1
        assert summary["severity"]      in {"info", "warn", "critical", "no_data"}
        assert "mean_psi" in summary

        # ── On-disk artifacts exist ────────────────────────────────
        manifest_path = result.manifest_path
        assert manifest_path.is_file()
        drift_summary_path = manifest_path.parent / "drift_summary.json"
        assert drift_summary_path.is_file()
        cluster_parquet = manifest_path.parent / "clusters" / "c0" / "drift_table.parquet"
        assert cluster_parquet.is_file()

        # ── Manifest content ───────────────────────────────────────
        manifest = json.loads(manifest_path.read_text())
        assert manifest["schema_version"]       == 1
        assert manifest["ensemble_version"]     == "ens_v1"
        assert manifest["n_clusters"]            == 1
        assert manifest["n_clusters_unaffected"] == 1
        assert "c0" in manifest["cluster_drift_tables"]
        assert manifest["cluster_drift_tables"]["c0"].endswith("drift_table.parquet")
        # M.2 hasn't promoted yet
        assert manifest["predictions"] is None

    def test_invalid_validation_report_raises(
        self, patch_inference_pipeline, ensemble_config, tmp_path,
    ):
        pipe = _make_pipeline(patch_inference_pipeline, ensemble_config)
        fake = pipe._inference_pipeline
        labels = _scenario_labels(3)
        fake._test_contexts        = {"c0": _FakeCtx()}
        fake._test_decisions       = [_FakeRoutingDecision("c0", is_affected=False)]
        fake._test_scenario_labels = labels

        # Override validate_scenarios to inject an error
        def _bad_validate():
            report = _FakeValidationReport(
                ensemble_version  = "ens_v1",
                n_scenarios       = 3,
                scenario_labels   = labels,
                cluster_decisions = fake._test_decisions,
                errors            = ["synthetic validation error"],
            )
            fake._validation_report = report
            return report
        fake.validate_scenarios = _bad_validate  # type: ignore[assignment]

        with pytest.raises(ValueError, match="validation failed"):
            pipe.run(new_scenario_dir=tmp_path / "scenarios")

    def test_missing_baseline_emits_no_data_table(
        self, patch_inference_pipeline, ensemble_config, tmp_path,
    ):
        """Cluster with no baseline parquet → no_data drift row, run still succeeds."""
        labels    = _scenario_labels(5)
        rng       = np.random.default_rng(0)
        today_pnl = _historical_pnl(labels, rng)

        # Note: NO baseline parquet written, but ctx points to a fake
        # version_dir that doesn't have one.
        version_dir = Path(ensemble_config.registry_dir) / "cluster-c0-v1"
        version_dir.mkdir(parents=True, exist_ok=True)

        pipe = _make_pipeline(patch_inference_pipeline, ensemble_config)
        fake = pipe._inference_pipeline
        fake._test_contexts = {
            "c0": _FakeCtx(
                elementary_pnl       = today_pnl,
                _cluster_assets_path = version_dir / "cluster_assets.joblib",
            ),
        }
        fake._test_decisions       = [_FakeRoutingDecision("c0", is_affected=False)]
        fake._test_scenario_labels = labels

        result = pipe.run(new_scenario_dir=tmp_path / "scenarios")
        c0_drift = result.drift_tables["c0"]
        assert (c0_drift["severity"] == SEVERITY_NO_DATA).all()
        # Manifest still written
        assert result.manifest_path.is_file()


# ═════════════════════════════════════════════════════════════════════
# M.3 — promote_to_predictions()
# ═════════════════════════════════════════════════════════════════════

def _run_drift_only(
    monitor_module, ensemble_config, tmp_path,
    *, n_scenarios: int = 500,
):
    """Common setup: write a baseline + execute a drift run.

    Returns ``(pipeline, monitoring_result)`` ready for the promote
    tests to act on.  Kept as a helper rather than a fixture so each
    test can tune ``n_scenarios`` for its sensitivity needs.
    """
    rng = np.random.default_rng(0)
    training_features = pd.DataFrame({
        "trade_0": rng.normal(0, 1, 5000),
        "trade_1": rng.normal(0, 1, 5000),
    })
    version_dir = Path(ensemble_config.registry_dir) / "cluster-c0-v1"
    save_feature_baseline(
        version_dir / "monitoring" / "baseline_feature_stats.parquet",
        features   = training_features,
        cluster_id = "c0",
    )

    labels    = _scenario_labels(n_scenarios)
    today_pnl = pd.DataFrame(
        {
            "trade_0": rng.normal(0, 1, len(labels)),
            "trade_1": rng.normal(0, 1, len(labels)),
        },
        index=labels,
    )

    pipe = _make_pipeline(monitor_module, ensemble_config)
    fake = pipe._inference_pipeline
    fake._test_contexts = {
        "c0": _FakeCtx(
            elementary_pnl       = today_pnl,
            _cluster_assets_path = version_dir / "cluster_assets.joblib",
        ),
    }
    fake._test_decisions       = [_FakeRoutingDecision("c0", is_affected=False)]
    fake._test_scenario_labels = labels

    monitoring_result = pipe.run(new_scenario_dir=tmp_path / "scenarios")
    return pipe, monitoring_result


def _make_run_inference_stub(
    fake_pipeline,
    *,
    n_clusters_in_meta: Optional[int] = None,
    cluster_ids_to_drop: Optional[List[str]] = None,
):
    """Build a callable that emulates run_inference's side effects.

    Reads ``fake_pipeline.config.artifacts_dir`` at CALL time (not at
    bind time) so the monitoring pipeline's stash-swap is observed
    correctly.  Drops empty ``trade_predictions/<cid>_<space>.parquet``
    placeholder files at the swapped location so
    ``_derive_n_clusters_predicted`` has something to count.
    """
    cluster_ids_to_drop = cluster_ids_to_drop or []

    def _stub() -> InferenceResult:
        root = Path(fake_pipeline.config.artifacts_dir) / "inference"
        trade_dir = root / "trade_predictions"
        trade_dir.mkdir(parents=True, exist_ok=True)
        for cid in cluster_ids_to_drop:
            (trade_dir / f"{cid}_scaled.parquet").touch()
            (trade_dir / f"{cid}_original.parquet").touch()
        # Top-level manifest the real run_inference writes — its
        # presence lets the monitoring manifest's relative
        # ``manifest_path`` pointer resolve to a real file.
        (root / "manifest.json").write_text(
            json.dumps({"schema_version": 1, "ensemble_version": "ens_v1"})
        )
        meta = {"n_clusters": n_clusters_in_meta} if n_clusters_in_meta is not None else {}
        return InferenceResult(
            predictions     = np.zeros((1, 1), dtype=np.float32),
            n_samples       = 1,
            sample_ids      = ["s0"],
            model_version   = "ens_v1",
            latency_seconds = 0.01,
            metadata        = meta,
        )

    return _stub


class TestPromoteToPredictions:
    """In-session promote — exercises the stash-swap + manifest rewrite."""

    def test_promote_writes_predictions_inside_monitoring_run(
        self, patch_inference_pipeline, ensemble_config, tmp_path,
    ):
        pipe, mres = _run_drift_only(
            patch_inference_pipeline, ensemble_config, tmp_path,
        )
        fake = pipe._inference_pipeline
        fake._test_run_inference = _make_run_inference_stub(
            fake, n_clusters_in_meta=1, cluster_ids_to_drop=["c0"],
        )

        promote = pipe.promote_to_predictions()

        # Predictions dir is INSIDE the monitoring run, not the
        # global artifacts dir.
        assert promote.predictions_dir.parent == mres.manifest_path.parent
        assert promote.predictions_dir.name   == "inference"
        assert promote.predictions_dir.is_dir()
        assert (promote.predictions_dir / "manifest.json").is_file()
        assert promote.n_clusters_predicted == 1
        assert promote.wall_seconds >= 0.0
        assert promote.run_id == mres.run_id

    def test_promote_rewrites_monitoring_manifest_with_predictions_block(
        self, patch_inference_pipeline, ensemble_config, tmp_path,
    ):
        pipe, mres = _run_drift_only(
            patch_inference_pipeline, ensemble_config, tmp_path,
        )
        fake = pipe._inference_pipeline
        fake._test_run_inference = _make_run_inference_stub(
            fake, n_clusters_in_meta=1, cluster_ids_to_drop=["c0"],
        )

        # Drift-only manifest first — predictions block must be null.
        pre = json.loads(mres.manifest_path.read_text())
        assert pre["predictions"] is None

        pipe.promote_to_predictions()

        post = json.loads(mres.manifest_path.read_text())
        assert post["predictions"] is not None
        block = post["predictions"]
        # Relative paths only — make the run dir portable.
        assert block["subdir"]        == "monitoring/inference"
        assert block["manifest_path"] == "monitoring/inference/manifest.json"
        assert block["n_clusters"]    == 1
        assert "promoted_at"  in block
        assert "wall_seconds" in block
        # Other manifest fields must NOT have shifted across the rewrite.
        for key in (
            "schema_version", "run_id", "ensemble_version", "input_mode",
            "n_clusters", "n_clusters_affected", "n_clusters_unaffected",
            "cluster_drift_tables", "drift_summary",
        ):
            assert pre[key] == post[key], (
                f"manifest field '{key}' shifted across promote rewrite"
            )

    def test_promote_restores_artifacts_dir_on_success(
        self, patch_inference_pipeline, ensemble_config, tmp_path,
    ):
        pipe, _ = _run_drift_only(
            patch_inference_pipeline, ensemble_config, tmp_path,
        )
        fake = pipe._inference_pipeline
        original_artifacts_dir = fake.config.artifacts_dir
        fake._test_run_inference = _make_run_inference_stub(
            fake, n_clusters_in_meta=1, cluster_ids_to_drop=["c0"],
        )

        pipe.promote_to_predictions()
        assert fake.config.artifacts_dir == original_artifacts_dir

    def test_promote_restores_artifacts_dir_on_failure(
        self, patch_inference_pipeline, ensemble_config, tmp_path,
    ):
        pipe, _ = _run_drift_only(
            patch_inference_pipeline, ensemble_config, tmp_path,
        )
        fake = pipe._inference_pipeline
        original_artifacts_dir = fake.config.artifacts_dir

        def _explode():
            raise RuntimeError("synthetic forward-pass failure")
        fake._test_run_inference = _explode

        with pytest.raises(RuntimeError, match="synthetic forward-pass failure"):
            pipe.promote_to_predictions()
        # try/finally must restore even when run_inference raised
        assert fake.config.artifacts_dir == original_artifacts_dir

    def test_promote_requires_prior_run(
        self, patch_inference_pipeline, ensemble_config,
    ):
        pipe = _make_pipeline(patch_inference_pipeline, ensemble_config)
        with pytest.raises(RuntimeError, match="successful prior run"):
            pipe.promote_to_predictions()

    def test_promote_is_one_shot_per_instance(
        self, patch_inference_pipeline, ensemble_config, tmp_path,
    ):
        pipe, _ = _run_drift_only(
            patch_inference_pipeline, ensemble_config, tmp_path,
        )
        fake = pipe._inference_pipeline
        fake._test_run_inference = _make_run_inference_stub(
            fake, n_clusters_in_meta=1, cluster_ids_to_drop=["c0"],
        )

        pipe.promote_to_predictions()
        with pytest.raises(RuntimeError, match="already succeeded"):
            pipe.promote_to_predictions()

    def test_n_clusters_predicted_falls_back_to_disk_count(
        self, patch_inference_pipeline, ensemble_config, tmp_path,
    ):
        """When metadata['n_clusters'] is absent, count from disk."""
        pipe, _ = _run_drift_only(
            patch_inference_pipeline, ensemble_config, tmp_path,
        )
        fake = pipe._inference_pipeline
        # No metadata, three clusters on disk.
        fake._test_run_inference = _make_run_inference_stub(
            fake,
            n_clusters_in_meta  = None,
            cluster_ids_to_drop = ["c0", "c1", "c2"],
        )
        promote = pipe.promote_to_predictions()
        assert promote.n_clusters_predicted == 3

    def test_promote_emits_lifecycle_events(
        self, patch_inference_pipeline, ensemble_config, tmp_path,
    ):
        """Activity log gets 'Promote started' + 'Promote complete'."""
        events: List[Dict[str, Any]] = []

        # Build the pipeline with an event capture, then run.
        rng = np.random.default_rng(0)
        training_features = pd.DataFrame({
            "trade_0": rng.normal(0, 1, 5000),
            "trade_1": rng.normal(0, 1, 5000),
        })
        version_dir = Path(ensemble_config.registry_dir) / "cluster-c0-v1"
        save_feature_baseline(
            version_dir / "monitoring" / "baseline_feature_stats.parquet",
            features=training_features, cluster_id="c0",
        )

        labels    = _scenario_labels(500)
        today_pnl = pd.DataFrame(
            {
                "trade_0": rng.normal(0, 1, len(labels)),
                "trade_1": rng.normal(0, 1, len(labels)),
            },
            index=labels,
        )

        pipe = patch_inference_pipeline.EnsembleMonitoringPipeline(
            ensemble_config  = ensemble_config,
            ensemble_version = "ens_v1",
            on_event         = events.append,
        )
        fake = pipe._inference_pipeline
        fake._test_contexts = {
            "c0": _FakeCtx(
                elementary_pnl       = today_pnl,
                _cluster_assets_path = version_dir / "cluster_assets.joblib",
            ),
        }
        fake._test_decisions       = [_FakeRoutingDecision("c0", is_affected=False)]
        fake._test_scenario_labels = labels

        pipe.run(new_scenario_dir=tmp_path / "scenarios")
        fake._test_run_inference = _make_run_inference_stub(
            fake, n_clusters_in_meta=1, cluster_ids_to_drop=["c0"],
        )

        events.clear()  # we only care about promote events here
        pipe.promote_to_predictions()

        phases = [e.get("phase") for e in events]
        assert "Promote started"  in phases
        assert "Promote complete" in phases

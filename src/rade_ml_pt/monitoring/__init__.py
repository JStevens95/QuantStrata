"""Drift monitoring artifacts for the PRISM Model Monitoring surface.

This package houses:

* :mod:`monitoring.baselines` — training-time histogram + summary
  statistics writer (``save_feature_baseline``).
* :mod:`monitoring.loaders`   — reader that decodes baseline parquets
  back into NumPy-ready DataFrames (``load_baseline``).
* :mod:`monitoring.drift`     — pure-NumPy drift primitives + per-cluster
  / portfolio aggregators (PSI, JSD, severity classifier).
* :mod:`monitoring.writers`   — readers/writers for monitoring run
  artifacts (per-cluster drift table parquets + run-level JSON).
* :mod:`monitoring.run_paths` — on-disk layout conventions
  (``monitoring_runs/<run_id>/monitoring/...``) — symmetric with
  inference's ``inference_runs/<run_id>/inference/...``.

See ``docs/platform_designs/prism_retool_migration.md`` Phase 4 for the
full design and ``RADE_UI_DESIGN.md`` for the Monitoring tab consumer.
"""
from .drift import (  # noqa: F401  (re-exported public API)
    PSI_CRITICAL_THRESHOLD,
    PSI_WARN_THRESHOLD,
    SEVERITY_CRITICAL,
    SEVERITY_INFO,
    SEVERITY_NO_DATA,
    SEVERITY_WARN,
    build_drift_table,
    build_portfolio_drift_summary,
    classify_severity,
    js_divergence,
    population_stability_index,
)
from .loaders import load_baseline  # noqa: F401
from .run_paths import (  # noqa: F401
    MonitoringRunPaths,
    monitoring_run_id,
    parse_monitoring_run_id,
    per_run_artifacts_dir,
)
from .writers import (  # noqa: F401
    read_drift_summary_json,
    read_drift_table_parquet,
    read_monitoring_manifest_json,
    write_drift_summary_json,
    write_drift_table_parquet,
    write_monitoring_manifest_json,
)

__all__ = [
    # ─── drift.py ────────────────────────────────────────────────────
    "PSI_WARN_THRESHOLD", "PSI_CRITICAL_THRESHOLD",
    "SEVERITY_INFO", "SEVERITY_WARN", "SEVERITY_CRITICAL", "SEVERITY_NO_DATA",
    "population_stability_index",
    "js_divergence",
    "classify_severity",
    "build_drift_table",
    "build_portfolio_drift_summary",
    # ─── loaders.py ──────────────────────────────────────────────────
    "load_baseline",
    # ─── run_paths.py ────────────────────────────────────────────────
    "MonitoringRunPaths",
    "monitoring_run_id",
    "parse_monitoring_run_id",
    "per_run_artifacts_dir",
    # ─── writers.py ──────────────────────────────────────────────────
    "write_drift_table_parquet",
    "read_drift_table_parquet",
    "write_drift_summary_json",
    "read_drift_summary_json",
    "write_monitoring_manifest_json",
    "read_monitoring_manifest_json",
]

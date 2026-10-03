"""Plotly figure builders for the Monitoring page.

V1 ships **empty-state placeholders only** — every helper here returns
the standard themed empty figure (see :func:`figures._theme.empty_figure`)
with a chart-specific *awaiting-data* annotation.  Each placeholder
hints at what the production version of that chart will show, so the
Option-A preview reads as the proposed design rather than three
identical "no data" boxes.

When the Monitoring callbacks come online (Option B / V1 snapshot
mode), populated builders for each chart will land *next to* these
helpers in this same module, mirroring the layout used by
``cluster_deep_dive_charts.py`` (paired ``empty_*`` and ``populated_*``
factories).

Design anchor
-------------
``docs/platform_designs/rade_model_monitoring.png`` — Row 2 (residual
drift line + feature PSI bar) and Row 3 (latency histogram).
"""
from __future__ import annotations

import plotly.graph_objects as go

from ._theme import empty_figure


# ─────────────────────────────────────────────────────────────────────
# Empty-state placeholders
# ─────────────────────────────────────────────────────────────────────


def empty_residual_drift() -> go.Figure:
    """Placeholder for the *Residual Drift — last 30 days* line chart.

    Production version (Option B): per-scenario residual median + a
    P5–P95 envelope band derived from
    ``cluster_residuals_test.parquet``, with anomaly markers stamped
    on outlier scenarios.
    """
    return empty_figure("Awaiting eval-driven residual stats")


def empty_feature_drift_psi() -> go.Figure:
    """Placeholder for the *Feature Drift (PSI)* horizontal bar chart.

    Production version (Option B): top-10 features by PSI severity
    (info / warn / critical), computed from
    ``baseline_feature_stats.parquet`` (train baseline) vs recomputed
    test-side feature distributions.
    """
    return empty_figure("Awaiting feature drift PSI computation")


def empty_latency_histogram() -> go.Figure:
    """Placeholder for the *Latency Histogram* in Row 3.

    Production version (Stage 2): histogram of inference latencies
    pulled from the production logging pipeline, with annotated P50 /
    P95 / P99 lines.  Until that producer ships there is no real
    source for this chart — the V1 snapshot mode will keep this card
    on a synthesised gamma-shaped distribution.
    """
    return empty_figure("Awaiting inference latency telemetry")


__all__ = [
    "empty_residual_drift",
    "empty_feature_drift_psi",
    "empty_latency_histogram",
]

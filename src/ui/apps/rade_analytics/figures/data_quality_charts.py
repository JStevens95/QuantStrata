"""Plotly figure builders for the Data Quality page.

V1 ships **empty-state placeholders only** — every helper here returns
the standard themed empty figure (see :func:`figures._theme.empty_figure`)
with a chart-specific *awaiting-data* annotation.  Each placeholder
hints at what the production version of that chart will show, so the
Option-A preview reads as the proposed design rather than two identical
"no data" boxes.

When the Data Quality callbacks come online (Option B / V1 snapshot
mode), populated builders will land *next to* these helpers in this
same module, mirroring the layout used by ``cluster_deep_dive_charts.py``
(paired ``empty_*`` and ``populated_*`` factories).

Design anchor
-------------
``docs/platform_designs/rade_data_quality.png`` — Row 3 (completeness
heatmap, full-width) and Row 4 (feature summary table + distribution
explorer violin).
"""
from __future__ import annotations

import plotly.graph_objects as go

from ._theme import empty_figure


# ─────────────────────────────────────────────────────────────────────
# Empty-state placeholders
# ─────────────────────────────────────────────────────────────────────


def empty_completeness_heatmap() -> go.Figure:
    """Placeholder for the *Completeness Heatmap* in Row 3.

    Production version (Option B): X = ``cluster_id``, Y =
    ``feature_name``, Z = ``1 - null_rate`` (%).  Built from
    ``quality/completeness_{split}.parquet`` reshaped to a wide
    matrix; colour scale 0 % (purple) → 100 % (pink) matching the
    design palette.
    """
    return empty_figure("Awaiting completeness matrix from quality/completeness_{split}.parquet")


def empty_distribution_explorer() -> go.Figure:
    """Placeholder for the *Distribution Explorer* in Row 4.

    Production version (deferred): grouped violin per cluster for the
    feature selected in the right-hand dropdown.  Source TBD —
    ``quality/feature_summary_{split}.parquet`` only carries summary
    statistics (mean/std/percentiles), so V1 production either:

    1. Approximates a box from the existing percentile stats
       (p01 / p50 / p99 + min/max), or
    2. Adds a new ``quality/feature_samples_{split}.parquet`` writer
       on the eval side carrying sampled per-cluster values.

    Either path lights up this card without any layout change.
    """
    return empty_figure("Awaiting per-cluster feature distribution data")


__all__ = [
    "empty_completeness_heatmap",
    "empty_distribution_explorer",
]

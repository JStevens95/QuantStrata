"""Evaluation → Cross-Cluster sub-tab.

**Phase E.0 stub.**  The real implementation lands in E.2 and will
include:

* Full cluster × cluster correlation heatmap
  (:meth:`RadeBackend.cluster_correlations`).
* Top-k positively / negatively correlated pairs list.
* Drill-down on a cell → opens the pair in a side drawer with both
  residual timeseries overlaid.
"""
from __future__ import annotations

from typing import Optional

from dash import html

from ...components.state_wrappers import Empty
from ...data.session import Session


CROSS_CLUSTER_IDS = {
    "root": "evaluation-cross-cluster-root",
}


def build_cross_cluster(*, session: Optional[Session] = None) -> html.Div:
    """Stub layout for the Cross-Cluster sub-tab.

    The ``session`` kwarg is accepted for sub-tab builder uniformity
    (Page Contract §3 Rule L1) — currently unused while this is a
    placeholder; Phase E.2 will use it to seed the heatmap's initial
    selection / focus state from session.
    """
    del session  # reserved; Phase E.2 wires this in
    return html.Div(
        id=CROSS_CLUSTER_IDS["root"],
        className="rade-evaluation-subtab rade-evaluation-subtab--stub",
        children=Empty(
            title="Cross-Cluster",
            message="Correlation heatmap + top-k pairs land in Phase E.2.",
            icon="tabler:arrows-shuffle",
        ),
    )


__all__ = ["CROSS_CLUSTER_IDS", "build_cross_cluster"]

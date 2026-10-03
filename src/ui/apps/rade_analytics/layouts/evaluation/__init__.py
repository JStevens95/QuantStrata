"""Evaluation page — shell + four sub-tabs.

The shell (:mod:`.shell`) composes the global filter bar and the
``dmc.Tabs`` primary navigation; each sub-tab owns its own module:

* :mod:`.portfolio`          — ``/evaluation/portfolio`` (default)
* :mod:`.cross_cluster`      — ``/evaluation/cross-cluster``
* :mod:`.trade_graph`        — ``/evaluation/trade-graph``
* :mod:`.cluster_deep_dive`  — ``/evaluation/cluster``

Phase E.0 (this module) ships only stubs for the four sub-tabs so the
shell, filter bar, tabs router and URL handling can be exercised
end-to-end.  Real content lands in E.1–E.4.
"""
from __future__ import annotations

from .cluster_deep_dive import build_cluster_deep_dive
from .cross_cluster import build_cross_cluster
from .portfolio import build_portfolio
from .shell import (
    EVALUATION_IDS,
    EVALUATION_SUBTAB_SPECS,
    active_subtab_from_path,
    build_evaluation,
    build_subtab_content,
    path_for_subtab,
)
from .trade_graph import build_trade_graph


__all__ = [
    "EVALUATION_IDS",
    "EVALUATION_SUBTAB_SPECS",
    "active_subtab_from_path",
    "build_cluster_deep_dive",
    "build_cross_cluster",
    "build_evaluation",
    "build_portfolio",
    "build_subtab_content",
    "build_trade_graph",
    "path_for_subtab",
]

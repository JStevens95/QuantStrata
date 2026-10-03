"""Figure builders for the Rade Analytics Dash UI.

This is the single place every callback goes to for a Plotly
``go.Figure``.  Keeping chart construction out of the callback modules
means:

* Callbacks stay focused on fetch + state plumbing.
* Figures can be unit-tested headlessly (``fig.to_dict()`` snapshots).
* Visual tweaks land in one diff — no hunting through every callback
  module for a font-size bump.

Modules shipped
---------------
* :mod:`._theme`                   — shared layout defaults, palette, helpers.
* :mod:`.cluster_deep_dive_charts` — per-cluster PnL band + per-trade
  violin / scatter (Phase E.4).
* :mod:`.distributions`            — residual violin (aggregate + grouped).
* :mod:`.graph_charts`             — graph density histogram + edges vs nodes.
* :mod:`.scatter`                  — predicted-vs-actual scatter (+ focus).
* :mod:`.timeseries`               — portfolio PnL + rolling error band.
* :mod:`.trade_graph_stylesheet`   — Cytoscape stylesheet + legend body
  builders for the Trade-Graph color-by toggle (Phase E.3 rebuild).
* :mod:`.training_curves`          — per-cluster training loss + metric
  overlays (Phase E.4 Row 2).
"""
from __future__ import annotations

from ._theme import (
    CATEGORY_PALETTE,
    color_for_index,
    empty_figure,
    rade_layout,
    rgba,
)
from .cluster_deep_dive_charts import (
    elementary_pnl_multiline,
    per_trade_bias_scatter,
    per_trade_residual_histogram,
    per_trade_residual_violin,
    per_trade_scatter,
    predicted_vs_actual_band,
)
from .distributions import residual_violin
from .graph_charts import density_distribution, edges_vs_nodes_scatter
from .scatter import pred_actual_scatter
from .timeseries import error_over_time, portfolio_pnl
from .trade_graph_stylesheet import build_legend_body, build_stylesheet
from .training_curves import training_curves_chart

__all__ = [
    "CATEGORY_PALETTE",
    "build_legend_body",
    "build_stylesheet",
    "color_for_index",
    "density_distribution",
    "edges_vs_nodes_scatter",
    "elementary_pnl_multiline",
    "empty_figure",
    "error_over_time",
    "per_trade_bias_scatter",
    "per_trade_residual_histogram",
    "per_trade_residual_violin",
    "per_trade_scatter",
    "portfolio_pnl",
    "pred_actual_scatter",
    "predicted_vs_actual_band",
    "rade_layout",
    "residual_violin",
    "rgba",
    "training_curves_chart",
]

"""
Callbacks for Tab 1 — Overview.

All data comes from ``get_backend()`` — no ``EnsembleSession`` access.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html, no_update

from src.ui.apps.ensemble_analytics.config import METRIC_DISPLAY_NAMES
from src.ui.apps.ensemble_analytics.components.kpi_card import kpi_card
from src.ui.apps.ensemble_analytics.components.metric_table import metric_table
from src.ui.apps.ensemble_analytics.figures.scatter import pred_vs_target_scatter
from src.ui.apps.ensemble_analytics.figures.bar_charts import member_comparison_bar
from src.ui.apps.ensemble_analytics.figures.heatmaps import multi_metric_cluster_heatmap

logger = logging.getLogger(__name__)


def register(app):
    """Register Overview tab callbacks on *app*."""

    @app.callback(
        Output("overview-kpi-row", "children"),
        Output("overview-scatter-container", "children"),
        Output("overview-bar-container", "children"),
        Output("overview-heatmap-container", "children"),
        Output("overview-table-container", "children"),
        Input("overview-split-toggle", "value"),
    )
    def update_overview(split: str):
        """Rebuild all Overview visuals when the split toggle changes."""
        t0 = time.perf_counter()
        from src.ui.apps.ensemble_analytics_db.data.session_manager import get_backend
        from src.ui.apps.ensemble_analytics_db.data.prediction_store import get_portfolio_summary

        backend = get_backend()
        ens_metrics = backend.get_ensemble_metrics(split)
        pm_metrics = backend.get_member_metrics(split)
        cluster_ids = backend.get_cluster_ids()
        cluster_attrs = backend.get_cluster_attributes()
        cluster_mapping = backend.get_cluster_mapping()

        if not ens_metrics:
            return no_update, no_update, no_update, no_update, no_update

        # ── KPI cards ─────────────────────────────────────────────
        kpi_keys = ["mae", "rmse", "max_ae", "p95_ae", "p99_ae"]
        kpi_cards = []
        for key in kpi_keys:
            val = ens_metrics.get(key)
            display_val = f"{val:.4f}" if val is not None else "N/A"
            kpi_cards.append(
                dbc.Col(
                    kpi_card(
                        title=METRIC_DISPLAY_NAMES.get(key, key.upper()),
                        value=display_val,
                    ),
                    md=True,
                )
            )

        # ── Portfolio scatter (from pre-computed summary) ─────────
        scatter_fig = html.Div("No prediction data available.")
        portfolio = get_portfolio_summary(split)
        if portfolio is not None:
            scatter_fig = dcc.Graph(
                figure=pred_vs_target_scatter(
                    portfolio["predictions"], portfolio["targets"],
                    title=f"Portfolio PnL — {split.capitalize()}",
                ),
                config={"displayModeBar": False},
            )

        # ── Member comparison bar ─────────────────────────────────
        mae_by_cluster = {
            cid: pm_metrics.get(cid, {}).get("mae", 0.0)
            for cid in cluster_ids
        }
        hover_text = {}
        for cid in cluster_ids:
            ca = cluster_attrs.get(cid, {})
            if ca:
                hover_text[cid] = ", ".join(f"{k}={v}" for k, v in ca.items() if v is not None)
            else:
                hover_text[cid] = ""

        bar_fig = html.Div(
            dcc.Graph(
                figure=member_comparison_bar(
                    cluster_ids, mae_by_cluster,
                    metric_name="MAE",
                    title=f"MAE by Cluster — {split.capitalize()}",
                    hover_text=hover_text,
                ),
                config={"displayModeBar": False},
            ),
            style={"maxHeight": "450px", "overflowY": "auto"},
        )

        # ── Multi-metric cluster heatmap ──────────────────────────
        heatmap_fig = dcc.Graph(
            figure=multi_metric_cluster_heatmap(
                cluster_ids, pm_metrics,
                title=f"Cluster Metrics Heatmap — {split.capitalize()}",
            ),
            config={"displayModeBar": False},
        )

        # ── Member table ──────────────────────────────────────────
        column_defs = [
            {"field": "cluster_id", "headerName": "Cluster", "pinned": "left"},
        ]
        first_attrs = next(iter(cluster_attrs.values()), {}) if cluster_attrs else {}
        attr_cols = list(first_attrs.keys())
        for ac in attr_cols:
            column_defs.append({"field": ac, "headerName": ac.replace("_", " ").title()})

        metric_keys = list(next(iter(pm_metrics.values()), {}).keys()) if pm_metrics else []
        all_metric_vals = {mk: [] for mk in metric_keys}
        for cid in cluster_ids:
            for mk in metric_keys:
                v = pm_metrics.get(cid, {}).get(mk)
                if v is not None:
                    all_metric_vals[mk].append(v)

        p25 = {mk: float(np.percentile(vs, 25)) if vs else 0 for mk, vs in all_metric_vals.items()}
        p75 = {mk: float(np.percentile(vs, 75)) if vs else 0 for mk, vs in all_metric_vals.items()}

        for mk in metric_keys:
            column_defs.append({
                "field": mk,
                "headerName": METRIC_DISPLAY_NAMES.get(mk, mk.upper()),
                "valueFormatter": {"function": "d3.format('.4f')(params.value)"},
                "cellStyle": {
                    "styleConditions": [
                        {"condition": f"params.value < {p25[mk]}", "style": {"color": "#3fb950"}},
                        {"condition": f"params.value >= {p25[mk]} && params.value <= {p75[mk]}", "style": {"color": "#d29922"}},
                        {"condition": f"params.value > {p75[mk]}", "style": {"color": "#f85149"}},
                    ]
                },
            })

        column_defs.append({"field": "n_trades", "headerName": "# Trades"})

        row_data = []
        for cid in cluster_ids:
            row = {"cluster_id": cid}
            ca = cluster_attrs.get(cid, {})
            for ac in attr_cols:
                row[ac] = ca.get(ac, "")
            row.update(pm_metrics.get(cid, {}))
            row["n_trades"] = len(cluster_mapping.get(cid, []))
            row_data.append(row)

        table = metric_table(
            column_defs=column_defs,
            row_data=row_data,
            table_id="overview-member-table",
            sort_model=[{"colId": "mae", "sort": "asc"}],
        )

        logger.info("[CALLBACK] update_overview: %.3fs", time.perf_counter() - t0)
        return kpi_cards, scatter_fig, bar_fig, heatmap_fig, table

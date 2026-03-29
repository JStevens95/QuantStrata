"""
Callbacks for Tab 1 — Overview.

Single callback triggered by the split toggle.  Reads ensemble-level
display state and combined prediction arrays, builds all visual
elements, and returns them to the layout placeholders.
"""
from __future__ import annotations

import numpy as np
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html, no_update

from src.ui.apps.ensemble_analytics.config import METRIC_DISPLAY_NAMES
from src.ui.apps.ensemble_analytics.components.kpi_card import kpi_card
from src.ui.apps.ensemble_analytics.components.metric_table import metric_table
from src.ui.apps.ensemble_analytics.figures.scatter import pred_vs_target_scatter
from src.ui.apps.ensemble_analytics.figures.bar_charts import member_comparison_bar
from src.ui.apps.ensemble_analytics.figures.heatmaps import multi_metric_cluster_heatmap


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
        from src.ui.apps.ensemble_analytics.data.session_manager import get_session
        from src.ui.apps.ensemble_analytics.data.prediction_store import get_prediction_store

        session = get_session()
        ens_display = session.ensemble_display
        if ens_display is None:
            return no_update, no_update, no_update, no_update, no_update

        # ── KPI cards ─────────────────────────────────────────────
        ens_metrics = ens_display.ensemble_metrics.get(split, {})
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

        # ── Portfolio scatter ─────────────────────────────────────
        scatter_fig = html.Div("No prediction data available.")
        store = get_prediction_store(split)
        if store is not None:
            portfolio_preds = store.predictions.sum(axis=1)
            portfolio_targets = store.targets.sum(axis=1)
            scatter_fig = dcc.Graph(
                figure=pred_vs_target_scatter(
                    portfolio_preds, portfolio_targets,
                    title=f"Portfolio PnL — {split.capitalize()}",
                ),
                config={"displayModeBar": False},
            )

        # ── Member comparison bar ─────────────────────────────────
        pm_metrics = ens_display.per_member_metrics.get(split, {})
        cluster_ids = session.config.cluster_ids
        mae_by_cluster = {
            cid: pm_metrics.get(cid, {}).get("mae", 0.0)
            for cid in cluster_ids
        }
        bar_fig = dcc.Graph(
            figure=member_comparison_bar(
                cluster_ids, mae_by_cluster,
                metric_name="MAE",
                title=f"MAE by Cluster — {split.capitalize()}",
            ),
            config={"displayModeBar": False},
        )

        # ── Multi-metric cluster heatmap (G3) ─────────────────────
        heatmap_fig = dcc.Graph(
            figure=multi_metric_cluster_heatmap(
                cluster_ids, pm_metrics,
                title=f"Cluster Metrics Heatmap — {split.capitalize()}",
            ),
            config={"displayModeBar": False},
        )

        # ── Member table with cluster attributes (G1) + conditional formatting (G2) ──
        cluster_attrs = session.cluster_attributes
        column_defs = [
            {"field": "cluster_id", "headerName": "Cluster", "pinned": "left"},
        ]
        attr_cols = ["ccy", "desk", "product"]
        for ac in attr_cols:
            column_defs.append({"field": ac, "headerName": ac.upper()})

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
            row["n_trades"] = len(session.config.cluster_mapping.get(cid, []))
            row_data.append(row)

        table = metric_table(
            column_defs=column_defs,
            row_data=row_data,
            table_id="overview-member-table",
            sort_model=[{"colId": "mae", "sort": "asc"}],
        )

        return kpi_cards, scatter_fig, bar_fig, heatmap_fig, table

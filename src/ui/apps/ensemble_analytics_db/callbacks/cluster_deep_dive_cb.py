"""
Callbacks for Tab 3 — Cluster Deep Dive.

All data comes from ``get_backend()`` — no ``EnsembleSession`` access.
"""
from __future__ import annotations

import json
from typing import Any, Dict

import numpy as np
from dash import Input, Output, State, dcc, html, no_update

from src.ui.apps.ensemble_analytics.config import METRIC_DISPLAY_NAMES
from src.ui.apps.ensemble_analytics.components.metric_table import metric_table
from src.ui.apps.ensemble_analytics.figures.scatter import pred_vs_target_scatter
from src.ui.apps.ensemble_analytics.figures.timeseries import pnl_timeseries
from src.ui.apps.ensemble_analytics.figures.distributions import residual_histogram
from src.ui.apps.ensemble_analytics.theme.colors import TEXT_SECONDARY


def _ensure_dict(obj: Any) -> Dict[str, Any]:
    """Coerce API / cache values to a mapping (handles JSON-encoded blobs)."""
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, str):
        try:
            parsed = json.loads(obj)
        except (json.JSONDecodeError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def register(app):
    """Register Cluster Deep Dive callbacks on *app*."""

    @app.callback(
        Output("deep-dive-cluster-dropdown", "options"),
        Output("deep-dive-cluster-dropdown", "value"),
        Input("main-tabs", "value"),
    )
    def populate_dd_cluster(tab):
        if tab != "tab-cluster-deep-dive":
            return no_update, no_update
        from src.ui.apps.ensemble_analytics_db.data.session_manager import get_backend

        backend = get_backend()
        cluster_ids = backend.get_cluster_ids()
        attrs = backend.get_cluster_attributes()
        opts = []
        for cid in cluster_ids:
            if attrs and cid in attrs:
                row = _ensure_dict(attrs.get(cid, {}))
                parts = [f"{k}={v}" for k, v in row.items() if v is not None]
                label = f"{cid}  ({', '.join(parts)})" if parts else cid
            else:
                label = cid
            opts.append({"label": label, "value": cid})
        default = cluster_ids[0] if cluster_ids else None
        return opts, default

    @app.callback(
        Output("deep-dive-header", "children"),
        Output("deep-dive-split-table", "children"),
        Output("deep-dive-convergence", "children"),
        Output("deep-dive-scatter", "children"),
        Output("deep-dive-timeseries", "children"),
        Output("deep-dive-residual", "children"),
        Output("deep-dive-trade-dropdown", "options"),
        Output("deep-dive-trade-dropdown", "value"),
        Output("deep-dive-elementary", "children"),
        Output("deep-dive-model-config", "children"),
        Output("deep-dive-config", "children"),
        Input("deep-dive-cluster-dropdown", "value"),
        Input("deep-dive-split-toggle", "value"),
    )
    def update_deep_dive(cluster_id, split):
        import logging as _logging
        _logger = _logging.getLogger(__name__)

        n_out = 11
        if not cluster_id:
            return (no_update,) * n_out
        if not split:
            split = "test"

        try:
            return _update_deep_dive_inner(cluster_id, split)
        except Exception as exc:
            _logger.exception("update_deep_dive error for cluster=%s split=%s", cluster_id, split)
            err = html.Div(
                f"Error loading deep dive for {cluster_id}: {exc}",
                style={"color": "#f85149", "padding": "12px"},
            )
            empty_div = html.Div()
            return (
                err,            # deep-dive-header
                empty_div,      # deep-dive-split-table
                empty_div,      # deep-dive-convergence
                empty_div,      # deep-dive-scatter
                empty_div,      # deep-dive-timeseries
                empty_div,      # deep-dive-residual
                [],             # deep-dive-trade-dropdown.options
                [],             # deep-dive-trade-dropdown.value
                empty_div,      # deep-dive-elementary
                empty_div,      # deep-dive-model-config
                empty_div,      # deep-dive-config
            )

    def _update_deep_dive_inner(cluster_id, split):
        import json as _json
        import base64
        from src.ui.apps.ensemble_analytics_db.data.session_manager import get_backend
        from src.ui.apps.ensemble_analytics_db.data.prediction_store import get_cluster_predictions

        backend = get_backend()
        display = _ensure_dict(backend.get_cluster_display(cluster_id))
        attrs = _ensure_dict(backend.get_cluster_attributes().get(cluster_id, {}))
        cluster_mapping = backend.get_cluster_mapping()
        tu = _ensure_dict(display.get("trade_universe", {}))
        em = _ensure_dict(display.get("eval_metrics", {}))
        dc = _ensure_dict(em.get("data_config", {}))

        # ── Header ────────────────────────────────────────────────
        attr_str = ", ".join(f"{k}={v}" for k, v in attrs.items() if v)
        n_elem = len(tu.get("elementary_ids", []))
        n_target = len(tu.get("target_ids", cluster_mapping.get(cluster_id, [])))
        header = html.Div([
            html.Div(f"Cluster: {cluster_id}", style={"fontSize": "18px", "fontWeight": "700"}),
            html.Div(f"Version: {display.get('version', '?')}", style={"color": TEXT_SECONDARY, "fontSize": "13px"}),
            html.Div(attr_str, style={"color": TEXT_SECONDARY, "fontSize": "13px"}) if attr_str else None,
            html.Div(
                f"n_elementary: {n_elem} · n_target: {n_target} · "
                f"seq_length: {dc.get('seq_length', '?')} · "
                f"transform: {dc.get('transform_type', '?')}",
                style={"color": TEXT_SECONDARY, "fontSize": "13px"},
            ),
        ])

        # ── Split comparison table ────────────────────────────────
        split_rows = []
        for s in ["train", "val", "test"]:
            pm = backend.get_member_metrics(s).get(cluster_id, {})
            if pm:
                row = {"split": s.capitalize()}
                row.update(pm)
                split_rows.append(row)

        split_col_defs = [{"field": "split", "headerName": "Split", "pinned": "left"}]
        if split_rows:
            for key in split_rows[0]:
                if key != "split":
                    split_col_defs.append({
                        "field": key,
                        "headerName": METRIC_DISPLAY_NAMES.get(key, key.upper()),
                        "valueFormatter": {"function": "d3.format('.4f')(params.value)"},
                    })
        split_table = metric_table(split_col_defs, split_rows, "deep-dive-split-comp-table", height="180px")

        # ── Training convergence PNG ──────────────────────────────
        convergence_content = html.Div("No convergence plot available.", style={"color": TEXT_SECONDARY})
        png_bytes = backend.get_convergence_png(cluster_id)
        if png_bytes:
            encoded = base64.b64encode(png_bytes).decode()
            convergence_content = html.Img(
                src=f"data:image/png;base64,{encoded}",
                style={"width": "100%", "maxHeight": "400px", "objectFit": "contain"},
            )

        # ── Predictions: scatter, timeseries, residual ────────────
        scatter_content = html.Div("No predictions available.")
        ts_content = html.Div("No predictions available.")
        residual_content = html.Div("No predictions available.")
        trade_opts = []
        trade_default = []
        elementary_content = html.Div("Elementary PnL data not available.", style={"color": TEXT_SECONDARY})

        data = get_cluster_predictions(str(cluster_id), split)
        if data is not None:
            preds = data["predictions"]
            targets = data["targets"]
            if preds.ndim == 1:
                preds = preds.reshape(-1, 1)
                targets = targets.reshape(-1, 1)
            trade_ids = [str(t) for t in cluster_mapping.get(cluster_id, [])]
            trade_ids = trade_ids[:preds.shape[1]]

            cluster_pred = preds.sum(axis=1)
            cluster_target = targets.sum(axis=1)

            scatter_content = dcc.Graph(
                figure=pred_vs_target_scatter(cluster_pred, cluster_target,
                                              title=f"Pred vs Target — {split.capitalize()}"),
                config={"displayModeBar": False},
            )
            ts_content = dcc.Graph(
                figure=pnl_timeseries(cluster_pred, cluster_target,
                                      title=f"PnL Timeseries — {split.capitalize()}"),
                config={"displayModeBar": False},
            )
            residual_content = dcc.Graph(
                figure=residual_histogram(cluster_pred, cluster_target,
                                          title=f"Residuals — {split.capitalize()}"),
                config={"displayModeBar": False},
            )

            trade_opts = [{"label": tid, "value": tid} for tid in trade_ids]
            trade_default = trade_ids[:6]

        # ── Elementary PnL Explorer ───────────────────────────────
        elem_df = None
        elem_pnl_path = backend.get_elementary_pnl_path(cluster_id)
        if elem_pnl_path:
            from pathlib import Path
            import pandas as pd

            elem_path = Path(elem_pnl_path)
            if elem_path.exists():
                elem_df = pd.read_parquet(elem_path)
        if elem_df is None:
            df_getter = getattr(backend, "get_elementary_pnl_df", None)
            if callable(df_getter):
                elem_df = df_getter(cluster_id)

        if elem_df is not None and not elem_df.empty:
            import plotly.graph_objects as go
            from src.ui.apps.ensemble_analytics.theme.colors import CHART_COLORS

            n_scenarios, n_elem_cols = elem_df.shape

            stats_row = html.Div(
                f"{n_elem_cols} elementary trades × {n_scenarios} scenarios",
                style={"color": TEXT_SECONDARY, "fontSize": "13px", "marginBottom": "8px"},
            )

            MAX_LINES = 10
            show_df = elem_df.iloc[:, :MAX_LINES]
            x = np.arange(n_scenarios)
            fig = go.Figure()
            for i, col in enumerate(show_df.columns):
                fig.add_trace(go.Scattergl(
                    x=x, y=show_df[col].values,
                    mode="lines", name=str(col)[:20],
                    line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=1),
                ))
            fig.update_layout(
                title=f"Elementary PnL (first {len(show_df.columns)} trades)",
                xaxis_title="Scenario", yaxis_title="PnL",
                height=350,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=10)),
            )

            stats = elem_df.agg(["mean", "std", "min", "max"]).T.round(6)
            stats.insert(0, "trade_id", stats.index.astype(str))
            summary_data = stats.to_dict(orient="records")

            sum_col_defs = [
                {"field": "trade_id", "headerName": "Elementary Trade"},
                {"field": "mean", "headerName": "Mean"},
                {"field": "std", "headerName": "Std"},
                {"field": "min", "headerName": "Min"},
                {"field": "max", "headerName": "Max"},
            ]
            sum_table = metric_table(sum_col_defs, summary_data, "deep-dive-elem-table", height="300px")

            elementary_content = html.Div([
                stats_row,
                dcc.Graph(figure=fig, config={"displayModeBar": False}),
                sum_table,
            ])

        # ── Model configuration ───────────────────────────────────
        member_cfg = _ensure_dict(backend.get_member_configs().get(cluster_id, {}))
        if member_cfg:
            filtered = {
                k: v for k, v in member_cfg.items()
                if k not in ("metadata", "data_config")
            }
            cfg_json = _json.dumps(filtered, indent=2, default=str)
            model_config_content = html.Details([
                html.Summary(
                    "Expand Model / Training Config",
                    style={"cursor": "pointer", "color": TEXT_SECONDARY, "fontSize": "13px"},
                ),
                dcc.Markdown(
                    f"```json\n{cfg_json}\n```",
                    style={"fontSize": "12px", "maxHeight": "500px", "overflow": "auto"},
                ),
            ])
        else:
            model_config_content = html.Div("No member config available.", style={"color": TEXT_SECONDARY})

        # ── Data configuration ────────────────────────────────────
        if dc:
            dc_json = _json.dumps(dc, indent=2, default=str)
            data_config_content = html.Details([
                html.Summary(
                    "Expand Data Config",
                    style={"cursor": "pointer", "color": TEXT_SECONDARY, "fontSize": "13px"},
                ),
                dcc.Markdown(
                    f"```json\n{dc_json}\n```",
                    style={"fontSize": "12px", "maxHeight": "500px", "overflow": "auto"},
                ),
            ])
        else:
            data_config_content = html.Div("No data config available.", style={"color": TEXT_SECONDARY})

        return (header, split_table, convergence_content, scatter_content,
                ts_content, residual_content,
                trade_opts, trade_default,
                elementary_content, model_config_content, data_config_content)

    # ── Per-trade scatter ─────────────────────────────────────────
    @app.callback(
        Output("deep-dive-scatter-matrix", "children"),
        Input("deep-dive-trade-dropdown", "value"),
        State("deep-dive-cluster-dropdown", "value"),
        State("deep-dive-split-toggle", "value"),
    )
    def update_trade_scatter(selected_trades, cluster_id, split):
        if not selected_trades or not cluster_id:
            return html.Div("Select trades above.", style={"color": TEXT_SECONDARY})

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from src.ui.apps.ensemble_analytics_db.data.prediction_store import get_cluster_predictions
        from src.ui.apps.ensemble_analytics_db.data.session_manager import get_backend
        from src.ui.apps.ensemble_analytics.theme.colors import CHART_COLORS, TEXT_SECONDARY as TS

        backend = get_backend()
        data = get_cluster_predictions(str(cluster_id), split)
        if data is None:
            return html.Div("No prediction data.")

        preds = data["predictions"]
        targets = data["targets"]
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
            targets = targets.reshape(-1, 1)
        trade_ids = [str(t) for t in backend.get_cluster_mapping().get(cluster_id, [])]
        trade_ids = trade_ids[:preds.shape[1]]

        show = [t for t in selected_trades if t in trade_ids][:6]
        if not show:
            return html.Div("Selected trades not found in this cluster.")

        n_show = len(show)
        ncols = min(n_show, 3)
        nrows = (n_show + ncols - 1) // ncols
        titles = [t[:20] + "\u2026" if len(t) > 20 else t for t in show]
        fig = make_subplots(
            rows=nrows, cols=ncols,
            subplot_titles=titles,
            vertical_spacing=0.15,
            horizontal_spacing=0.10,
        )
        for j, tid in enumerate(show):
            idx = trade_ids.index(tid)
            r, c = j // ncols + 1, j % ncols + 1
            t = targets[:, idx]
            p = preds[:, idx]
            residuals = p - t
            fig.add_trace(go.Scattergl(
                x=t, y=p, mode="markers",
                marker=dict(size=3, color=CHART_COLORS[j % len(CHART_COLORS)], opacity=0.5),
                customdata=residuals,
                hovertemplate="Target: %{x:.4f}<br>Pred: %{y:.4f}<br>Residual: %{customdata:.4f}<extra></extra>",
                showlegend=False,
            ), row=r, col=c)
            vmin, vmax = min(t.min(), p.min()), max(t.max(), p.max())
            fig.add_trace(go.Scattergl(
                x=[vmin, vmax], y=[vmin, vmax], mode="lines",
                line=dict(color=TS, dash="dash", width=1),
                showlegend=False,
            ), row=r, col=c)

        fig.update_layout(height=300 * nrows, title="Per-Trade Pred vs Target")
        fig.update_annotations(font_size=11)
        return dcc.Graph(figure=fig, config={"displayModeBar": False})

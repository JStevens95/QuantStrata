"""
Tab 7 — Model Governance layout.

Static audit-trail view: ensemble manifest, member registry, config
inspector, and searchable trade-cluster map.
"""
from __future__ import annotations

import json

import dash_bootstrap_components as dbc
from dash import dcc, html

from src.ui.apps.ensemble_analytics.theme.styles import CARD_STYLE, SECTION_TITLE_STYLE, CARD_HEADER_STYLE
from src.ui.apps.ensemble_analytics.theme.colors import TEXT_SECONDARY


def layout() -> html.Div:
    """
    Build the Model Governance tab layout.

    Mostly static content populated at load time from session metadata.
    The trade-cluster search is handled by a client-side callback.

    Returns
    -------
    html.Div
    """
    from src.ui.apps.ensemble_analytics.data.session_manager import get_session
    from src.ui.apps.ensemble_analytics.components.metric_table import metric_table

    session = get_session()
    config = session.config
    ens_display = session.ensemble_display
    manifest = ens_display.manifest if ens_display else {}

    # ── Ensemble manifest card ────────────────────────────────────
    manifest_items = [
        ("Ensemble Version", session.ensemble_version),
        ("Clusters", str(config.n_members)),
        ("Total Trades", str(len(config.all_trade_ids))),
        ("Aggregation", config.aggregation),
        ("Execution Strategy", config.execution_strategy),
        ("Splits Available", ", ".join(manifest.get("splits_available", []))),
    ]
    manifest_card = html.Div(
        [html.Div("Ensemble Manifest", style=CARD_HEADER_STYLE)]
        + [
            html.Div(
                [
                    html.Span(f"{label}: ", style={"color": TEXT_SECONDARY, "fontWeight": "600"}),
                    html.Span(value),
                ],
                style={"marginBottom": "4px", "fontSize": "13px"},
            )
            for label, value in manifest_items
        ],
        style=CARD_STYLE,
    )

    # ── Member registry table ─────────────────────────────────────
    member_versions = session.member_versions or {}
    member_col_defs = [
        {"field": "cluster_id", "headerName": "Cluster"},
        {"field": "version", "headerName": "Version"},
        {"field": "n_trades", "headerName": "# Trades"},
    ]
    member_row_data = [
        {
            "cluster_id": cid,
            "version": member_versions.get(cid, "unknown"),
            "n_trades": len(config.cluster_mapping.get(cid, [])),
        }
        for cid in config.cluster_ids
    ]
    member_table = metric_table(
        column_defs=member_col_defs,
        row_data=member_row_data,
        table_id="governance-member-table",
        height="300px",
    )

    # ── Config inspector (collapsible JSON tree) ──────────────────
    summary = {
        k: v for k, v in config.to_dict().items()
        if k not in ("metadata", "member_configs")
    }
    config_json = json.dumps(summary, indent=2, default=str)
    config_card = html.Div(
        [
            html.Div("Configuration", style=CARD_HEADER_STYLE),
            html.Details(
                [
                    html.Summary(
                        "Expand EnsembleConfig JSON",
                        style={"cursor": "pointer", "color": TEXT_SECONDARY, "fontSize": "13px"},
                    ),
                    dcc.Markdown(
                        f"```json\n{config_json}\n```",
                        style={"fontSize": "12px", "maxHeight": "500px", "overflow": "auto"},
                    ),
                ],
            ),
        ],
        style=CARD_STYLE,
    )

    # ── Trade-cluster map (searchable) ────────────────────────────
    tcm = session.trade_cluster_map or {}
    tcm_col_defs = [
        {"field": "trade_id", "headerName": "Trade ID", "filter": True},
        {"field": "cluster_id", "headerName": "Cluster ID", "filter": True},
    ]
    tcm_row_data = [
        {"trade_id": tid, "cluster_id": cid}
        for tid, cid in tcm.items()
    ]
    tcm_table = metric_table(
        column_defs=tcm_col_defs,
        row_data=tcm_row_data,
        table_id="governance-tcm-table",
        height="400px",
    )

    # ── Version Comparison section (G16) ────────────────────────
    version_comparison = html.Div([
        html.Div("Version Comparison", style=CARD_HEADER_STYLE),
        html.Div([
            html.Label("Compare with:", style={"color": TEXT_SECONDARY, "fontSize": "12px", "marginRight": "6px"}),
            dcc.Dropdown(
                id="governance-compare-version",
                options=[], placeholder="Select version to compare...",
                style={"width": "400px", "fontSize": "13px"},
            ),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "12px"}),
        html.Div(id="governance-comparison-content"),
    ], style=CARD_STYLE)

    return html.Div([
        manifest_card,
        html.Div("Member Registry", style=SECTION_TITLE_STYLE),
        html.Div(member_table, style=CARD_STYLE),
        config_card,
        version_comparison,
        html.Div("Trade → Cluster Map", style=SECTION_TITLE_STYLE),
        html.Div(tcm_table, style=CARD_STYLE),
    ])

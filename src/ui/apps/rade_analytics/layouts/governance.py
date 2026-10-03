"""Governance page layout.

Mirrors ``docs/platform_designs/rade_governance.png`` region-for-region:

* **Row 0** — invisible mount tripwire (Page Contract §3 Rule L4).
* **Row 1** — Header band: 4 KPI chips (total / production / pending /
  last-activity) on the left, status filter ``SegmentedControl`` and a
  *Promote Version* primary button on the right.
* **Row 2** — *Model Registry* AG Grid: one row per registered
  ensemble version, populated from
  :meth:`backend.governance_registry_df`.  Status / version pair render
  as a single cell with a coloured pill.
* **Row 3** — Two-column row: *Lineage Timeline* (~2/3 width, derived
  from the same registry payload + a couple of static seed events) and
  *Approvals* card (~1/3 width, static V1) housing the pending-review
  CTA, Sign-offs checklist and Policy Checks list.
* **Row 4** — *Audit Log* AG Grid (V1: hard-coded rows; the producer
  side ships in Stage 2).

Mocked vs. real
---------------
The registry table, the four header KPIs and the lineage timeline read
from the live backend.  The right-hand approvals card, the sign-off
checklist, the policy-check list and the audit-log table are static
seed data baked into this module — labelled in code where they appear
so a future commit can replace them with a real source-of-truth
without hunting.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

import dash_mantine_components as dmc
from dash import dcc, html
from dash_iconify import DashIconify

from ..components.ag_grid_table import AgGridTable
from ..components.kpi_card import KpiCard

if TYPE_CHECKING:
    from ..data.session import Session


# ─────────────────────────────────────────────────────────────────────
# Stable id contract — every component a callback might target lives
# here so callbacks never hardcode strings (Page Contract §3 Rule L3).
# ─────────────────────────────────────────────────────────────────────


GOVERNANCE_IDS: Dict[str, str] = {
    "root":                      "governance-root",

    # Mount tripwire — Page Contract §3 Rule L4.
    "mount_signal":              "governance-mount-signal",

    # Row 1 — Header band.
    "status_filter":             "governance-status-filter",
    "promote_btn":                "governance-promote-btn",
    "kpi_total":                 "governance-kpi-total",
    "kpi_total_value":           "governance-kpi-total-value",
    "kpi_production":            "governance-kpi-production",
    "kpi_production_value":      "governance-kpi-production-value",
    "kpi_pending":               "governance-kpi-pending",
    "kpi_pending_value":         "governance-kpi-pending-value",
    "kpi_last_activity":         "governance-kpi-last-activity",
    "kpi_last_activity_value":   "governance-kpi-last-activity-value",

    # Row 2 — Model registry grid.
    "registry_grid":             "governance-registry-grid",

    # Row 3 — Lineage timeline.
    "lineage_timeline":          "governance-lineage-timeline",

    # Row 3 — Approvals card (static V1 — id present so the Stage-2
    # render callback can swap content without a layout refactor).
    "approvals_card":            "governance-approvals-card",
    "approvals_approve_btn":     "governance-approvals-approve-btn",
    "approvals_reject_btn":      "governance-approvals-reject-btn",

    # Row 4 — Audit log grid.
    "audit_log_grid":            "governance-audit-log-grid",
}


# ─────────────────────────────────────────────────────────────────────
# Static seed data — V1 placeholders, replaced in Stage 2 when the
# producer side (audit.sqlite + sign-off workflow) ships.
# ─────────────────────────────────────────────────────────────────────


# Pending review card — single open approval.  Hard-coded version is
# not* automatically tied to the active backend version because Stage 1
# has no real workflow producer; rendering a real version here without
# real workflow state would be misleading.
_PENDING_APPROVAL = {
    "version": "v2024.04.18",
    "approvers": ["pipeline-bot", "Joe Stevens"],
}

_SIGN_OFF_CHECKLIST: List[Dict[str, str]] = [
    {"label": "Risk",       "status": "approved"},
    {"label": "Quant",      "status": "approved"},
    {"label": "Production", "status": "approved"},
]

_POLICY_CHECKS: List[Dict[str, str]] = [
    {"label": "Backtest pass",     "status": "passed"},
    {"label": "Drift within band", "status": "passed"},
    {"label": "Coverage ≥ 95%",    "status": "passed"},
    {"label": "Peer review",       "status": "pending"},
]

_AUDIT_LOG_ROWS: List[Dict[str, Any]] = [
    {
        "timestamp": "2026-04-29 18:23:33",
        "actor":     "Joe Stevens",
        "action":    "Tag",
        "target":    "production",
        "result":    "approved",
    },
    {
        "timestamp": "2026-04-29 17:08:14",
        "actor":     "pipeline-bot",
        "action":    "Eval published",
        "target":    "ens_20260429_170201",
        "result":    "approved",
    },
    {
        "timestamp": "2026-04-29 16:42:01",
        "actor":     "Joe Stevens",
        "action":    "Drift check",
        "target":    "Coverage ≥ 95%",
        "result":    "rejected",
    },
    {
        "timestamp": "2026-04-29 14:11:07",
        "actor":     "pipeline-bot",
        "action":    "Register",
        "target":    "ens_20260429_141102",
        "result":    "approved",
    },
]


# Status filter values — keep in lock-step with the
# :class:`GovernanceRegistryRow.status` enum on the backend so the
# front-end SegmentedControl filters cleanly without an extra mapping.
_STATUS_FILTER_OPTIONS = [
    {"value": "all",        "label": "All"},
    {"value": "production", "label": "Production"},
    {"value": "staging",    "label": "Staging"},
    {"value": "candidate",  "label": "Candidate"},
    {"value": "archived",   "label": "Archived"},
]
_STATUS_FILTER_DEFAULT = "all"


# ─────────────────────────────────────────────────────────────────────
# Row builders
# ─────────────────────────────────────────────────────────────────────


def _kpi_strip() -> html.Div:
    """Four KPI cards across the top of the page (left half of Row 1).

    Values are em-dashes at build time and overwritten by the bootstrap
    render callback once the registry payload arrives.
    """
    return html.Div(
        className="grid grid-cols-4 gap-4",
        children=[
            KpiCard(
                label="Total Versions",
                value="—",
                card_id=GOVERNANCE_IDS["kpi_total"],
                value_id=GOVERNANCE_IDS["kpi_total_value"],
                icon="tabler:database",
            ),
            KpiCard(
                label="In Production",
                value="—",
                card_id=GOVERNANCE_IDS["kpi_production"],
                value_id=GOVERNANCE_IDS["kpi_production_value"],
                icon="tabler:circle-check",
            ),
            KpiCard(
                label="Pending Sign-offs",
                value="—",
                card_id=GOVERNANCE_IDS["kpi_pending"],
                value_id=GOVERNANCE_IDS["kpi_pending_value"],
                icon="tabler:hourglass",
            ),
            KpiCard(
                label="Last Activity",
                value="—",
                card_id=GOVERNANCE_IDS["kpi_last_activity"],
                value_id=GOVERNANCE_IDS["kpi_last_activity_value"],
                icon="tabler:clock",
            ),
        ],
    )


def _header_actions() -> html.Div:
    """Status filter SegmentedControl + *Promote Version* button (right half).

    The filter is bound to the registry grid via a render callback that
    re-emits ``rowData``.  The promote button is wired to a no-op
    notification today; Stage 2 lifts it into the workflow producer.
    """
    return html.Div(
        className="flex items-center justify-end gap-3",
        children=[
            dmc.SegmentedControl(
                id=GOVERNANCE_IDS["status_filter"],
                data=_STATUS_FILTER_OPTIONS,
                value=_STATUS_FILTER_DEFAULT,
                size="sm",
                color="violet",
                radius="md",
            ),
            dmc.Button(
                id=GOVERNANCE_IDS["promote_btn"],
                children="Promote Version",
                color="violet",
                size="sm",
                leftSection=DashIconify(icon="tabler:rocket", width=16),
            ),
        ],
    )


def _row_header() -> html.Div:
    """Row 1 — KPI strip + filter + promote CTA in a single visual band."""
    return html.Div(
        className="flex flex-col gap-3",
        children=[
            html.Div(
                className=(
                    "flex items-center justify-between gap-4 flex-wrap"
                ),
                children=[
                    html.Div("Governance", className="rade-page-title"),
                    _header_actions(),
                ],
            ),
            _kpi_strip(),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Row 2 — Model Registry grid
# ─────────────────────────────────────────────────────────────────────


# Status / Result colour rules — reused by the registry + audit
# grids.  Cells get exactly one of these classes via
# ``cellClassRules`` (AG Grid evaluates the predicates server-side
# during cell render); the matching CSS lives in ``rade.css`` under
# the ``.rade-pill--*`` selectors.
_STATUS_CLASS_RULES: Dict[str, str] = {
    "rade-pill rade-pill--production": "params.value === 'production'",
    "rade-pill rade-pill--staging":    "params.value === 'staging'",
    "rade-pill rade-pill--candidate":  "params.value === 'candidate'",
    "rade-pill rade-pill--archived":   "params.value === 'archived'",
}

_AUDIT_RESULT_CLASS_RULES: Dict[str, str] = {
    "rade-pill rade-pill--approved": "params.value === 'approved'",
    "rade-pill rade-pill--rejected": "params.value === 'rejected'",
    "rade-pill rade-pill--pending":  "params.value === 'pending'",
}


_REGISTRY_COLUMN_DEFS: List[Dict[str, Any]] = [
    {
        "field": "version",
        "headerName": "Version",
        "minWidth": 220,
        "pinned": "left",
        "cellClass": "rade-grid-mono",
    },
    {
        "field": "status",
        "headerName": "Status",
        "minWidth": 130,
        "cellClassRules": _STATUS_CLASS_RULES,
        # Render-time capitalisation — keeps the wire format lowercase
        # (so the SegmentedControl filter compares cleanly) while
        # showing a human-friendly label in the cell.
        "valueFormatter": {
            "function": (
                "params.value ? "
                "params.value.charAt(0).toUpperCase() + params.value.slice(1)"
                " : '—'"
            ),
        },
    },
    {
        "field": "created_by",
        "headerName": "Created By",
        "minWidth": 140,
    },
    {
        "field": "created_at",
        "headerName": "Created",
        "minWidth": 170,
        "valueFormatter": {
            "function": (
                "params.value ? "
                "new Date(params.value).toLocaleString('en-GB', "
                "{day:'2-digit', month:'short', year:'numeric', "
                " hour:'2-digit', minute:'2-digit'}) "
                ": '—'"
            ),
        },
    },
    {
        "field": "promoted_at",
        "headerName": "Promoted",
        "minWidth": 140,
        "valueFormatter": {
            "function": (
                "params.value ? "
                "new Date(params.value).toLocaleDateString('en-GB', "
                "{day:'2-digit', month:'short', year:'numeric'}) "
                ": '—'"
            ),
        },
    },
    {
        "field": "commit_sha",
        "headerName": "Commit SHA",
        "minWidth": 110,
        "cellClass": "rade-grid-mono",
    },
    {
        "field": "mae_test",
        "headerName": "MAE (test)",
        "type": "numericColumn",
        "minWidth": 110,
        "valueFormatter": {
            "function": (
                "params.value == null ? '—' : "
                "Number(params.value).toFixed(3)"
            ),
        },
    },
    {
        "field": "n_members",
        "headerName": "Clusters",
        "type": "numericColumn",
        "minWidth": 90,
    },
    {
        "field": "n_trades",
        "headerName": "Trades",
        "type": "numericColumn",
        "minWidth": 90,
        "valueFormatter": {
            "function": (
                "params.value == null ? '—' : "
                "Number(params.value).toLocaleString('en-GB')"
            ),
        },
    },
    {
        "field": "has_evaluation",
        "headerName": "Artifacts",
        "minWidth": 100,
        # Map the boolean to a glyph; AG Grid evaluates the function
        # at render time so callers can keep the wire format minimal.
        "valueFormatter": {
            "function": "params.value ? 'Available' : 'Pending'"
        },
        "cellClassRules": {
            "text-emerald-400": "params.value === true",
            "text-slate-500":   "params.value === false",
        },
    },
]


def _row_registry() -> html.Div:
    """Row 2 — full-width AG Grid for the Model Registry table."""
    return html.Div(
        className="rade-card flex flex-col gap-3",
        children=[
            html.Div(
                "Model Registry",
                className="text-sm font-semibold text-slate-200",
            ),
            AgGridTable(
                grid_id=GOVERNANCE_IDS["registry_grid"],
                row_data=[],
                column_defs=_REGISTRY_COLUMN_DEFS,
                grid_options={
                    "pagination": True,
                    "paginationPageSize": 25,
                    "paginationPageSizeSelector": [10, 25, 50, 100],
                    "rowHeight": 40,
                    "headerHeight": 38,
                    "animateRows": False,
                    "suppressCellFocus": True,
                    "domLayout": "normal",
                },
                height=340,
                className="rade-governance-registry-grid",
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Row 3 — Lineage timeline + Approvals card
# ─────────────────────────────────────────────────────────────────────


def _lineage_timeline_card() -> html.Div:
    """Left ~2/3 of Row 3 — recent lifecycle events, populated by callback.

    The body is an empty list at build time; the bootstrap render
    callback fills the timeline from the same registry payload that
    drives the table (most recent ``promoted_at`` / ``created_at``
    events get a one-line entry).  No-data fallback is rendered
    inline by the callback rather than baked here so the layout stays
    pure.
    """
    return html.Div(
        className="rade-card flex flex-col gap-3 col-span-2",
        children=[
            html.Div(
                "Lineage Timeline",
                className="text-sm font-semibold text-slate-200",
            ),
            html.Div(
                id=GOVERNANCE_IDS["lineage_timeline"],
                className="rade-feed",
                children=[],
            ),
        ],
    )


def _approvals_card() -> html.Div:
    """Right ~1/3 of Row 3 — pending approval + sign-offs + policy checks.

    Static V1 content.  The Stage-2 producer (workflow service) will
    drive this card via a callback that overwrites the
    ``approvals_card`` children — no layout change required.
    """
    avatars_row = html.Div(
        className="flex items-center gap-1",
        children=[
            DashIconify(
                icon="tabler:user-circle",
                width=22,
                className="text-slate-300",
            )
            for _ in _PENDING_APPROVAL["approvers"]
        ],
    )

    pending_block = html.Div(
        className="rade-card-compact flex flex-col gap-2",
        children=[
            html.Div(
                "Pending review:",
                className="text-xs text-slate-500 uppercase tracking-wider",
            ),
            html.Div(
                _PENDING_APPROVAL["version"],
                className="text-base font-semibold text-slate-100",
            ),
            html.Div(
                className="flex items-center justify-between gap-3 mt-1",
                children=[
                    avatars_row,
                    html.Div(
                        className="flex items-center gap-2",
                        children=[
                            dmc.Button(
                                id=GOVERNANCE_IDS["approvals_approve_btn"],
                                children="Approve",
                                color="teal",
                                size="xs",
                                variant="filled",
                            ),
                            dmc.Button(
                                id=GOVERNANCE_IDS["approvals_reject_btn"],
                                children="Reject",
                                color="red",
                                size="xs",
                                variant="light",
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )

    sign_offs_block = html.Div(
        className="flex flex-col gap-2",
        children=[
            html.Div(
                "Sign-offs",
                className="text-sm font-semibold text-slate-200",
            ),
            *[
                html.Div(
                    className="flex items-center gap-2 text-xs text-slate-300",
                    children=[
                        DashIconify(
                            icon=(
                                "tabler:circle-check-filled"
                                if row["status"] == "approved"
                                else "tabler:circle-dashed"
                            ),
                            width=14,
                            className=(
                                "text-emerald-400"
                                if row["status"] == "approved"
                                else "text-slate-500"
                            ),
                        ),
                        html.Span(row["label"]),
                    ],
                )
                for row in _SIGN_OFF_CHECKLIST
            ],
        ],
    )

    policy_block = html.Div(
        className="flex flex-col gap-2",
        children=[
            html.Div(
                "Policy Checks",
                className="text-sm font-semibold text-slate-200",
            ),
            *[
                html.Div(
                    className="flex items-center gap-2 text-xs",
                    children=[
                        DashIconify(
                            icon=(
                                "tabler:circle-check-filled"
                                if row["status"] == "passed"
                                else "tabler:alert-triangle-filled"
                            ),
                            width=14,
                            className=(
                                "text-emerald-400"
                                if row["status"] == "passed"
                                else "text-amber-400"
                            ),
                        ),
                        html.Span(
                            row["label"],
                            className=(
                                "text-slate-300"
                                if row["status"] == "passed"
                                else "text-amber-300"
                            ),
                        ),
                        html.Span(
                            "" if row["status"] == "passed" else " · pending",
                            className="text-amber-400",
                        ),
                    ],
                )
                for row in _POLICY_CHECKS
            ],
        ],
    )

    return html.Div(
        id=GOVERNANCE_IDS["approvals_card"],
        className="rade-card flex flex-col gap-3 col-span-1",
        children=[
            html.Div(
                "Approvals",
                className="text-sm font-semibold text-slate-200",
            ),
            pending_block,
            html.Div(
                className="grid grid-cols-2 gap-4 mt-2",
                children=[sign_offs_block, policy_block],
            ),
        ],
    )


def _row_lineage_and_approvals() -> html.Div:
    return html.Div(
        className="grid grid-cols-3 gap-4 items-stretch",
        children=[
            _lineage_timeline_card(),
            _approvals_card(),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Row 4 — Audit log grid
# ─────────────────────────────────────────────────────────────────────


_AUDIT_LOG_COLUMN_DEFS: List[Dict[str, Any]] = [
    {
        "field": "timestamp",
        "headerName": "Timestamp",
        "minWidth": 180,
        "cellClass": "rade-grid-mono",
    },
    {"field": "actor",  "headerName": "Actor",  "minWidth": 140},
    {"field": "action", "headerName": "Action", "minWidth": 160},
    {"field": "target", "headerName": "Target", "minWidth": 200},
    {
        "field": "result",
        "headerName": "Result",
        "minWidth": 120,
        "cellClassRules": _AUDIT_RESULT_CLASS_RULES,
        "valueFormatter": {
            "function": (
                "params.value ? "
                "params.value.charAt(0).toUpperCase() + params.value.slice(1)"
                " : '—'"
            ),
        },
    },
]


def _row_audit_log() -> html.Div:
    """Row 4 — full-width AG Grid for the audit log (V1: static rows)."""
    return html.Div(
        className="rade-card flex flex-col gap-3",
        children=[
            html.Div(
                className=(
                    "flex items-center justify-between gap-3 flex-wrap"
                ),
                children=[
                    html.Div(
                        "Audit Log (last 24h)",
                        className="text-sm font-semibold text-slate-200",
                    ),
                    html.Div(
                        "Governance data sourced from ensemble registry "
                        "+ audit.sqlite",
                        className="text-xs text-slate-500",
                    ),
                ],
            ),
            AgGridTable(
                grid_id=GOVERNANCE_IDS["audit_log_grid"],
                row_data=_AUDIT_LOG_ROWS,
                column_defs=_AUDIT_LOG_COLUMN_DEFS,
                grid_options={
                    "pagination": False,
                    "rowHeight": 36,
                    "headerHeight": 38,
                    "animateRows": False,
                    "suppressCellFocus": True,
                    "domLayout": "autoHeight",
                },
                height=200,
                className="rade-governance-audit-grid",
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────


def build_governance(*, session: Optional["Session"] = None) -> html.Div:
    """Build the full Governance page tree.

    The ``session`` kwarg is accepted for uniformity with every other
    page builder (Page Contract §2.1) but unused today — the page
    has no per-user persisted state.  Reserved so adding e.g. a
    ``governance_status_filter`` field to ``Session`` later is a
    one-line layout change.
    """
    del session  # unused today; reserved for forward-compat

    return html.Div(
        id=GOVERNANCE_IDS["root"],
        className="rade-page",
        children=[
            # Mount tripwire — Page Contract §3 Rule L4.
            dcc.Store(
                id=GOVERNANCE_IDS["mount_signal"],
                data=True,
                storage_type="memory",
            ),
            _row_header(),
            _row_registry(),
            _row_lineage_and_approvals(),
            _row_audit_log(),
        ],
    )


__all__ = [
    "GOVERNANCE_IDS",
    "build_governance",
]

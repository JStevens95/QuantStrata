"""Data Quality page layout.

Mirrors ``docs/platform_designs/rade_data_quality.png`` region-for-
region:

* **Row 0** — invisible mount tripwire (Page Contract §3 Rule L4) so
  the Stage-2 render callback can fire on layout mount without
  racing the DOM swap.
* **Row 1** — Header band: page title + subtitle on the left, and a
  filter bar (Cluster select + Search input) plus the *Export CSV*
  CTA on the right.  The Train/Val/Test split is owned by the
  chrome-level topbar toggle (``shell.SHELL_IDS["split_toggle"]``);
  this page reads ``session.split`` like every other page so the
  user's choice is honoured without duplicating the control.
* **Row 2** — KPI strip: five chips spanning the row (Total Features,
  Clusters, Complete Features, Features with Missing, Overall
  Completeness %).
* **Row 3** — *Completeness Heatmap* (full width).  X axis =
  ``cluster_id``; Y axis = ``feature_name``; cell colour = ``1 -
  null_rate`` (%).  Built from ``quality/completeness_{split}.parquet``.
* **Row 4** — Two-column row: *Feature Summary* AG Grid (~1/2 width)
  and *Distribution Explorer* (~1/2 width) with a "Selected feature"
  dropdown anchored top-right.
* **Row 5** — Footer caption: data-source disclosure naming the two
  parquets that drive the page.

V1 status — Option A (empty layout, no callbacks)
-------------------------------------------------
This module ships the full layout with **empty defaults everywhere**:

* KPI values default to em-dash placeholders.
* The Completeness Heatmap and Distribution Explorer mount with the
  matching ``empty_*`` figure from :mod:`figures.data_quality_charts`
  (each carries an "Awaiting …" annotation hinting at what the
  populated chart will show).
* The Feature Summary grid mounts with ``rowData=[]``.
* The cluster filter, search box and feature dropdown mount with
  neutral defaults so the chrome reads exactly as it will once
  callbacks land — but no callback fires today.

No callbacks are registered today (see ``callbacks/__init__.py``) so
no API request hits ``/prism/v1/quality/*`` when the page is
navigated to.  When the V1 snapshot-mode callbacks land they will
overwrite each region via ``Output`` — no layout change required.

Page Contract anchors
---------------------
* §3 Rule L1 — every primitive a callback might target has a stable
  id via :data:`DATA_QUALITY_IDS`.
* §3 Rule L3 — no hardcoded id strings outside this dict.
* §3 Rule L4 — :data:`DATA_QUALITY_IDS["mount_signal"]` is a Store
  with ``data=True`` so render callbacks can fire after the DOM swap.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

import dash_mantine_components as dmc
from dash import dcc, html
from dash_iconify import DashIconify

from ..components.ag_grid_table import AgGridTable
from ..components.chart_container import ChartContainer
from ..components.kpi_card import KpiCard
from ..figures.data_quality_charts import (
    empty_completeness_heatmap,
    empty_distribution_explorer,
)

if TYPE_CHECKING:
    from ..data.session import Session


# ─────────────────────────────────────────────────────────────────────
# Stable id contract — every component a callback might target lives
# here so callbacks never hardcode strings (Page Contract §3 Rule L3).
# ─────────────────────────────────────────────────────────────────────


DATA_QUALITY_IDS: Dict[str, str] = {
    "root":                       "data-quality-root",

    # Mount tripwire — Page Contract §3 Rule L4.
    "mount_signal":               "data-quality-mount-signal",

    # Row 1 — Header subtitle (carries the "updated N min ago" suffix
    # once the snapshot callback comes online).
    "subtitle":                   "data-quality-subtitle",

    # Row 1 — Filter bar.  The Train/Val/Test split toggle lives in
    # the chrome-level topbar (see ``shell.SHELL_IDS["split_toggle"]``);
    # this page reads ``session.split`` instead of duplicating the
    # control.
    "cluster_filter":             "data-quality-cluster-filter",
    "feature_search":             "data-quality-feature-search",
    "export_btn":                 "data-quality-export-btn",

    # Row 2 — KPI chips.  Each card carries a separate value id so
    # callbacks can target the value text without rebuilding chrome.
    "kpi_total_features":         "data-quality-kpi-total-features",
    "kpi_total_features_value":   "data-quality-kpi-total-features-value",
    "kpi_clusters":               "data-quality-kpi-clusters",
    "kpi_clusters_value":         "data-quality-kpi-clusters-value",
    "kpi_complete_features":      "data-quality-kpi-complete-features",
    "kpi_complete_features_value": "data-quality-kpi-complete-features-value",
    "kpi_with_missing":           "data-quality-kpi-with-missing",
    "kpi_with_missing_value":     "data-quality-kpi-with-missing-value",
    "kpi_overall_completeness":   "data-quality-kpi-overall-completeness",
    "kpi_overall_completeness_value": "data-quality-kpi-overall-completeness-value",

    # Row 3 — Completeness heatmap.
    "completeness_heatmap":       "data-quality-completeness-heatmap",

    # Row 4 — Feature summary grid + distribution explorer.
    "feature_summary_grid":       "data-quality-feature-summary-grid",
    "distribution_chart":         "data-quality-distribution-chart",
    "feature_picker":             "data-quality-feature-picker",
}


# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────


# Placeholder used for any KPI value that doesn't have a callback
# overwriting it yet.  Matches the convention used on every other
# Rade page (Overview / Governance / Monitoring / Cluster Deep-Dive).
_PLACEHOLDER = "—"


# Subtitle wording for V1 — honest about the empty state.  The
# snapshot-mode callback (Option B) replaces this with
# ``"Per-feature completeness & summary stats — split: <split> · N
# clusters · updated <N> min ago"``.
_V1_SUBTITLE = (
    "Per-feature completeness & summary statistics — preview "
    "(no live artifacts connected)"
)


# Sentinel that means "no cluster filter" in the Cluster select.
# Picked once here so the populate callback can compare against it
# without hardcoding the string in two places.
_ALL_CLUSTERS_VALUE = "__all__"


# Footer caption — surface the producer-side reality so reviewers
# don't infer anything about live data quality from a static layout.
# Wording mirrors the design's bottom-right grey strap line.
_FOOTER_CAPTION = (
    "Artifacts read from quality/completeness_{split}.parquet and "
    "quality/feature_summary_{split}.parquet"
)


# ─────────────────────────────────────────────────────────────────────
# Row 1 — Header band (title + subtitle + filter bar + export CTA)
# ─────────────────────────────────────────────────────────────────────


def _filter_bar() -> html.Div:
    """Cluster select + feature search + export CTA.

    The Train/Val/Test split is owned by the topbar toggle (see
    ``shell.py``); this filter bar covers only the page-local
    filters that don't make sense at the chrome level.  Mounted
    with neutral defaults so the chrome reads as the populated page
    would — no callback runs today.
    """
    return html.Div(
        className="flex items-center justify-between gap-3 flex-wrap",
        children=[
            html.Div(
                className="flex items-center gap-3 flex-wrap",
                children=[
                    html.Div(
                        className="flex items-center gap-2",
                        children=[
                            html.Div(
                                "Cluster:",
                                className="text-xs text-slate-400",
                            ),
                            dmc.Select(
                                id=DATA_QUALITY_IDS["cluster_filter"],
                                data=[
                                    {
                                        "label": "All clusters",
                                        "value": _ALL_CLUSTERS_VALUE,
                                    },
                                ],
                                value=_ALL_CLUSTERS_VALUE,
                                size="sm",
                                radius="md",
                                clearable=False,
                                searchable=True,
                                allowDeselect=False,
                                w=180,
                            ),
                        ],
                    ),
                    dmc.TextInput(
                        id=DATA_QUALITY_IDS["feature_search"],
                        placeholder="Search",
                        size="sm",
                        radius="md",
                        leftSection=DashIconify(
                            icon="tabler:search", width=14,
                        ),
                        w=220,
                    ),
                ],
            ),
            dmc.Button(
                id=DATA_QUALITY_IDS["export_btn"],
                children="Export CSV",
                variant="default",
                size="sm",
                radius="md",
                leftSection=DashIconify(icon="tabler:download", width=16),
            ),
        ],
    )


def _kpi_strip() -> html.Div:
    """Five KPI cards across Row 2.

    Values are em-dashes at build time and overwritten by the
    snapshot render callback once it lands.  Card icons stay static.

    Uses ``grid-cols-5`` so the five chips share the row width
    evenly, matching the density in the design.
    """
    return html.Div(
        className="grid grid-cols-5 gap-4",
        children=[
            KpiCard(
                label="Total Features",
                value=_PLACEHOLDER,
                card_id=DATA_QUALITY_IDS["kpi_total_features"],
                value_id=DATA_QUALITY_IDS["kpi_total_features_value"],
                icon="tabler:list-numbers",
            ),
            KpiCard(
                label="Clusters",
                value=_PLACEHOLDER,
                card_id=DATA_QUALITY_IDS["kpi_clusters"],
                value_id=DATA_QUALITY_IDS["kpi_clusters_value"],
                icon="tabler:apps",
            ),
            KpiCard(
                label="Complete Features",
                value=_PLACEHOLDER,
                card_id=DATA_QUALITY_IDS["kpi_complete_features"],
                value_id=DATA_QUALITY_IDS["kpi_complete_features_value"],
                icon="tabler:circle-check",
            ),
            KpiCard(
                label="Features with Missing",
                value=_PLACEHOLDER,
                card_id=DATA_QUALITY_IDS["kpi_with_missing"],
                value_id=DATA_QUALITY_IDS["kpi_with_missing_value"],
                icon="tabler:alert-triangle",
            ),
            KpiCard(
                label="Overall Completeness %",
                value=_PLACEHOLDER,
                card_id=DATA_QUALITY_IDS["kpi_overall_completeness"],
                value_id=DATA_QUALITY_IDS["kpi_overall_completeness_value"],
                icon="tabler:percentage",
            ),
        ],
    )


def _row_header() -> html.Div:
    """Row 1 — page title + subtitle on the left, filter bar on the right.

    The subtitle's id is part of the contract so the snapshot
    callback can drop a real "updated N min ago" suffix into it
    without rebuilding chrome.
    """
    return html.Div(
        className="flex flex-col gap-3",
        children=[
            html.Div(
                className=(
                    "flex items-end justify-between gap-4 flex-wrap"
                ),
                children=[
                    html.Div(
                        className="flex flex-col gap-1",
                        children=[
                            html.Div(
                                "Data Quality",
                                className="rade-page-title",
                            ),
                            html.Div(
                                _V1_SUBTITLE,
                                id=DATA_QUALITY_IDS["subtitle"],
                                className="text-xs text-slate-500",
                            ),
                        ],
                    ),
                ],
            ),
            _filter_bar(),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Row 3 — Completeness heatmap (full width)
# ─────────────────────────────────────────────────────────────────────


def _row_completeness_heatmap() -> html.Div:
    """Row 3 — full-width Completeness Heatmap card.

    Wider chart height than the standard 320 because the Y axis must
    accommodate one row per feature (real ensembles have ~50–200
    features); 360 gives the placeholder some breathing room and
    matches what the populated chart will need.
    """
    return ChartContainer(
        title="Completeness Heatmap",
        subtitle=(
            "Per-feature, per-cluster completeness — colour scale "
            "0% (purple) → 100% (pink)"
        ),
        graph_id=DATA_QUALITY_IDS["completeness_heatmap"],
        figure=empty_completeness_heatmap(),
        height=360,
    )


# ─────────────────────────────────────────────────────────────────────
# Row 4 — Feature Summary table + Distribution Explorer
# ─────────────────────────────────────────────────────────────────────


# Missing % colour rules — heat-mapped pill so high-missing rows pop
# out of the table at a glance.  Reuses the existing
# ``rade-pill--rejected/pending/archived`` palette so the V1 layout
# doesn't need a new CSS pass; the populate callback just emits
# numeric ``missing_pct`` values and AG Grid picks the matching class.
_MISSING_PCT_CLASS_RULES: Dict[str, str] = {
    "rade-pill rade-pill--rejected": (
        "params.value != null && params.value >= 5"
    ),
    "rade-pill rade-pill--pending": (
        "params.value != null && params.value >= 1 && params.value < 5"
    ),
    "rade-pill rade-pill--archived": (
        "params.value != null && params.value > 0 && params.value < 1"
    ),
}


_FEATURE_SUMMARY_COLUMN_DEFS: List[Dict[str, Any]] = [
    {
        "field":      "feature_name",
        "headerName": "Feature",
        "minWidth":   160,
        "pinned":     "left",
        "cellClass":  "rade-grid-mono",
    },
    {
        "field":      "cluster_id",
        "headerName": "Cluster",
        "minWidth":   90,
    },
    {
        "field":      "mean",
        "headerName": "Mean",
        "type":       "numericColumn",
        "minWidth":   100,
        "valueFormatter": {
            "function": (
                "params.value == null ? '—' : "
                "Number(params.value).toFixed(4)"
            ),
        },
    },
    {
        "field":      "std",
        "headerName": "Std",
        "type":       "numericColumn",
        "minWidth":   100,
        "valueFormatter": {
            "function": (
                "params.value == null ? '—' : "
                "Number(params.value).toFixed(4)"
            ),
        },
    },
    {
        "field":      "min",
        "headerName": "Min",
        "type":       "numericColumn",
        "minWidth":   90,
        "valueFormatter": {
            "function": (
                "params.value == null ? '—' : "
                "Number(params.value).toFixed(4)"
            ),
        },
    },
    {
        "field":      "max",
        "headerName": "Max",
        "type":       "numericColumn",
        "minWidth":   90,
        "valueFormatter": {
            "function": (
                "params.value == null ? '—' : "
                "Number(params.value).toFixed(4)"
            ),
        },
    },
    {
        "field":      "missing_pct",
        "headerName": "Missing %",
        "type":       "numericColumn",
        "minWidth":   110,
        "cellClassRules": _MISSING_PCT_CLASS_RULES,
        "valueFormatter": {
            "function": (
                "params.value == null ? '—' : "
                "Number(params.value).toFixed(1) + '%'"
            ),
        },
    },
    {
        "field":      "n_rows",
        "headerName": "N",
        "type":       "numericColumn",
        "minWidth":   80,
        "valueFormatter": {
            "function": (
                "params.value == null ? '—' : "
                "Number(params.value).toLocaleString('en-GB')"
            ),
        },
    },
]


def _feature_summary_card() -> html.Div:
    """Left half of Row 4 — feature summary AG Grid card.

    Mounts with ``rowData=[]`` so the populate callback (when it
    lands) can drop in whatever the active split's
    ``feature_summary_{split}.parquet`` carries.  Column defs are
    finalised today so adding rows later doesn't trigger a layout
    rebuild — the populate callback just emits ``rowData``.
    """
    return html.Div(
        className="rade-card flex flex-col gap-3",
        children=[
            html.Div(
                "Feature Summary",
                className="text-sm font-semibold text-slate-200",
            ),
            AgGridTable(
                grid_id=DATA_QUALITY_IDS["feature_summary_grid"],
                row_data=[],
                column_defs=_FEATURE_SUMMARY_COLUMN_DEFS,
                grid_options={
                    "pagination": True,
                    "paginationPageSize": 10,
                    "paginationPageSizeSelector": [10, 25, 50, 100],
                    "rowHeight": 36,
                    "headerHeight": 38,
                    "animateRows": False,
                    "suppressCellFocus": True,
                    "domLayout": "normal",
                },
                height=320,
                className="rade-data-quality-summary-grid",
            ),
        ],
    )


def _distribution_explorer_card() -> html.Div:
    """Right half of Row 4 — distribution explorer with a feature picker.

    The feature picker lives in the chart card's top-right action
    slot so the user can swap which feature is being inspected
    without leaving the row.  Mounts disabled / empty until the
    populate callback supplies the real feature list — keeping it
    visible (rather than hidden) makes the design intent obvious in
    the empty state.
    """
    feature_picker = dmc.Select(
        id=DATA_QUALITY_IDS["feature_picker"],
        data=[],
        value=None,
        size="xs",
        radius="md",
        clearable=False,
        searchable=True,
        allowDeselect=False,
        placeholder="Select feature",
        w=200,
        disabled=True,
    )

    return ChartContainer(
        title="Distribution Explorer",
        subtitle="Per-cluster distribution for the selected feature",
        graph_id=DATA_QUALITY_IDS["distribution_chart"],
        figure=empty_distribution_explorer(),
        height=320,
        actions=[feature_picker],
    )


def _row_summary_and_distribution() -> html.Div:
    """Row 4 — feature summary grid (~1/2) + distribution explorer (~1/2).

    Both children share equal width so the page reads as two paired
    cards, matching the design.  When the snapshot callback lands
    the grid populates from ``/prism/v1/quality/feature-summary``
    and the chart populates from whichever distribution source we
    pick (see ``empty_distribution_explorer`` docstring for the two
    candidate paths).
    """
    return html.Div(
        className="grid grid-cols-2 gap-4 items-stretch",
        children=[
            _feature_summary_card(),
            _distribution_explorer_card(),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Row 5 — Footer caption
# ─────────────────────────────────────────────────────────────────────


def _row_footer() -> html.Div:
    """Producer-disclosure caption.

    Rendered as plain text right-aligned underneath Row 4, matching
    the design's tiny grey footer line.  Stays static — it's not
    user-data; the wording becomes "data-quality-source-truthful"
    once the snapshot callback wires up the live parquet reads.
    """
    return html.Div(
        className="flex justify-end",
        children=[
            html.Div(
                _FOOTER_CAPTION,
                className="text-xs text-slate-500",
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────


def build_data_quality(*, session: Optional["Session"] = None) -> html.Div:
    """Build the full Data Quality page tree.

    The ``session`` kwarg is accepted for uniformity with every other
    page builder (Page Contract §2.1) but unused here today — the
    Train/Val/Test split is owned by the chrome-level topbar
    toggle, and the populate callback (when it lands) will read
    ``session.split`` directly via ``State`` rather than from the
    initial layout.  Every field in :data:`DATA_QUALITY_IDS` mounts
    with neutral / empty defaults that the populate callback will
    overwrite once it lands.
    """
    del session  # split toggle lives in the topbar; no per-page state today

    return html.Div(
        id=DATA_QUALITY_IDS["root"],
        className="rade-page",
        children=[
            # Mount tripwire — Page Contract §3 Rule L4.
            dcc.Store(
                id=DATA_QUALITY_IDS["mount_signal"],
                data=True,
                storage_type="memory",
            ),
            _row_header(),
            _kpi_strip(),
            _row_completeness_heatmap(),
            _row_summary_and_distribution(),
            _row_footer(),
        ],
    )


__all__ = [
    "DATA_QUALITY_IDS",
    "build_data_quality",
]

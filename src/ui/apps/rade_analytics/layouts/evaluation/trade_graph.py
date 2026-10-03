"""Evaluation → Trade-Graph sub-tab layout (Phase E.3 rebuild).

Page anatomy
------------
::

    Row 0 · mount tripwire           (invisible · drives _bootstrap)
    Row 1 · Header band              (sticky · cluster · layout · color-by ·
                                       threshold · fit · export)
    Row 2 · Main area                (grid lg:grid-cols-3 items-stretch)
            ├── Cytoscape pane (col-span-2)         · mini-map overlay
            └── Side panel (col-span-1)
                ├── Selected Trade  h-[200px]       · with neighbours popover
                ├── Node Legend     h-[140px]       · rebuilt by color-by
                └── Cluster Stats   h-[180px]       · 2×2 KPI grid
    Row 3 · Density distribution     (full width chart card)
    Row 4 · Edges vs nodes scatter   (full width chart card)

The Cytoscape pane stretches to match the **side panel's intrinsic
height** via ``items-stretch`` on the Row 2 grid + ``flex-1`` on the
inner cytoscape wrapper.  This keeps the graph as tall as the side
panel without hard-coding pixel heights — drop a card on the right
and the graph grows; tighten one and the graph shrinks.

Why a popover for "Nearest k"
-----------------------------
The Selected-Trade card is fixed-height by design (Page Contract §3
Rule L1, "no layout-time data fetching" — predictable mount frame).
Putting the neighbour list inline would either grow the card on each
node tap (jarring layout shift) or require an in-card scrollbar
(easy to miss).  A popover sits *above* the card, scrolls inside
itself, and disappears on outside-click — the card's height stays
fixed regardless of how many neighbours the user wants to see.

Mini-map
--------
A second :class:`dash_cytoscape.Cytoscape` instance positioned in
the pane's bottom-right corner.  Pan / zoom is disabled — the
mini-map is read-only.  It shares ``elements`` with the main graph
through a single render-callback that writes the same payload to
both ids (no JS sync needed).

All dynamic ids live in :data:`TRADE_GRAPH_IDS` so callbacks never
hardcode strings (Page Contract §3 Rule L3).  The build function
``build_trade_graph(*, session)`` is **pure** (Rule L1) — same
session in, same DOM out; backend lookups happen in the callback
module's render section.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import dash_cytoscape as cyto
import dash_mantine_components as dmc
from dash import dcc, html
from dash_iconify import DashIconify

from ...components.chart_container import ChartContainer
from ...components.kpi_card import KpiCard
from ...data.session import (
    DEFAULT_TRADE_GRAPH_COLOR_BY,
    DEFAULT_TRADE_GRAPH_LAYOUT,
    DEFAULT_TRADE_GRAPH_NEIGHBOUR_K,
    DEFAULT_TRADE_GRAPH_WEIGHT_THRESHOLD,
    EVALUATION_TRADE_GRAPH_COLOR_BY,
    EVALUATION_TRADE_GRAPH_LAYOUTS,
    Session,
)


# ─────────────────────────────────────────────────────────────────────
# ID contract — every component id used by the page lives here.
# ─────────────────────────────────────────────────────────────────────


TRADE_GRAPH_IDS: Dict[str, str] = {
    "root":                          "eval-trade-graph-root",

    # Mount tripwire — Page Contract §3 Rule L4.  Memory-store seeded
    # ``data=True`` at build time; the bootstrap callback fires off
    # this Store's ``data`` Input so it runs exactly once per fresh
    # mount of the page.
    "mount_signal":                  "eval-trade-graph-mount-signal",

    # Row 1 — Header band
    "cluster_select":                "eval-trade-graph-cluster-select",
    "layout_radio":                  "eval-trade-graph-layout-radio",
    "color_by_select":               "eval-trade-graph-color-by-select",
    "threshold_slider":              "eval-trade-graph-threshold-slider",
    "threshold_value_label":         "eval-trade-graph-threshold-value",
    "fit_btn":                       "eval-trade-graph-fit-btn",
    "export_btn":                    "eval-trade-graph-export-btn",

    # Row 2 — Cytoscape pane
    "cytoscape":                     "eval-trade-graph-cytoscape",
    "cytoscape_minimap":             "eval-trade-graph-cytoscape-minimap",
    "pane_status":                   "eval-trade-graph-pane-status",

    # Row 2 — Selected-Trade card
    "selected_card":                 "eval-trade-graph-selected-card",
    "selected_trade_id":             "eval-trade-graph-selected-trade-id",
    "selected_copy_btn":             "eval-trade-graph-selected-copy-btn",
    "selected_chip_strip":           "eval-trade-graph-selected-chip-strip",
    "selected_metrics":              "eval-trade-graph-selected-metrics",
    "selected_neighbours_btn":       "eval-trade-graph-selected-neighbours-btn",
    "selected_neighbours_btn_label": "eval-trade-graph-selected-neighbours-btn-label",
    "selected_neighbours_k_input":   "eval-trade-graph-selected-neighbours-k-input",
    "selected_neighbours_list":      "eval-trade-graph-selected-neighbours-list",
    "selected_deep_dive_btn":        "eval-trade-graph-selected-deep-dive-btn",

    # Row 2 — Node Legend card (body rebuilt by color-by render callback)
    "legend_card":                   "eval-trade-graph-legend-card",
    "legend_body":                   "eval-trade-graph-legend-body",

    # Row 2 — Cluster Stats card (2×2 KPI grid)
    "cluster_stats_card":            "eval-trade-graph-cluster-stats-card",
    "cluster_stats_grid":            "eval-trade-graph-cluster-stats-grid",

    # Row 3 / Row 4 — Secondary charts
    "density_chart":                 "eval-trade-graph-density-chart",
    "edges_vs_nodes_chart":          "eval-trade-graph-edges-vs-nodes-chart",

    # Ephemeral memory store — caches the last TradeGraphResponse so the
    # neighbours popover, threshold filter and stylesheet renderer don't
    # need a re-fetch.  Plain JSON; deserialised on every read.
    "store_graph":                   "eval-trade-graph-graph-store",
}


# ─────────────────────────────────────────────────────────────────────
# Cytoscape stylesheet base — render callback layers colour-by
# overrides on top of this.  ``trade_type`` colouring (target=amber,
# elementary=violet) is the safe default that ships in the layout so
# the first paint already shows something useful before any callback
# fires.
# ─────────────────────────────────────────────────────────────────────


CYTOSCAPE_STYLESHEET: List[Dict[str, Any]] = [
    {
        "selector": "node",
        "style": {
            "label":            "",
            "width":            12,
            "height":           12,
            "background-color": "#8b5cf6",     # violet — elementary
            "border-color":     "#0f172a",
            "border-width":     1,
            "transition-property": "background-color, width, height, border-color",
            "transition-duration": "150ms",
        },
    },
    {
        "selector": "node[trade_type = 'target']",
        "style": {
            "width":            18,
            "height":           18,
            "background-color": "#f59e0b",     # amber — targets
        },
    },
    {
        "selector": "node:selected",
        "style": {
            "border-color":  "#10b981",
            "border-width":  3,
            "width":         22,
            "height":        22,
        },
    },
    {
        "selector": "edge",
        "style": {
            "width":           "mapData(weight, 0, 1, 0.5, 3)",
            "line-color":      "rgba(148, 163, 184, 0.35)",
            "curve-style":     "haystack",
            "haystack-radius": 0.5,
        },
    },
]

# Mini-map uses a leaner stylesheet — we want a thumbnail, not a
# legible network.  No transitions, smaller / fixed-size dots, and
# faintest possible edges.
MINIMAP_STYLESHEET: List[Dict[str, Any]] = [
    {
        "selector": "node",
        "style": {
            "label":            "",
            "width":            6,
            "height":           6,
            "background-color": "#8b5cf6",
            "border-width":     0,
        },
    },
    {
        "selector": "node[trade_type = 'target']",
        "style": {"background-color": "#f59e0b", "width": 8, "height": 8},
    },
    {
        "selector": "edge",
        "style": {
            "width":      0.5,
            "line-color": "rgba(148, 163, 184, 0.20)",
            "curve-style": "haystack",
        },
    },
]


# ─────────────────────────────────────────────────────────────────────
# Header band labels & options
# ─────────────────────────────────────────────────────────────────────


# Human-readable label per color-by mode.  Used in the header select
# and (via the callback) to re-build the legend card body.
COLOR_BY_LABELS: Dict[str, str] = {
    "trade_type":  "Trade type",
    "residual":    "Residual",
    "asset_class": "Asset class",
    "currency":    "Currency",
    "product":     "Product",
}


def _layout_radio() -> dmc.SegmentedControl:
    return dmc.SegmentedControl(
        id=TRADE_GRAPH_IDS["layout_radio"],
        data=[
            {"value": name, "label": name.capitalize()}
            for name in EVALUATION_TRADE_GRAPH_LAYOUTS
        ],
        value=DEFAULT_TRADE_GRAPH_LAYOUT,
        size="xs",
        color="violet",
    )


def _color_by_select(*, value: str) -> dmc.Select:
    return dmc.Select(
        id=TRADE_GRAPH_IDS["color_by_select"],
        data=[
            {"value": k, "label": COLOR_BY_LABELS[k]}
            for k in EVALUATION_TRADE_GRAPH_COLOR_BY
        ],
        value=value,
        clearable=False,
        searchable=False,
        size="sm",
        # Fixed width so the header doesn't reflow when the user picks
        # a longer label like "Asset class".
        w=160,
    )


def _threshold_slider(*, value: float) -> html.Div:
    """Min-weight slider — drops edges whose weight < threshold."""
    return html.Div(
        className="flex flex-col gap-1 min-w-[180px]",
        children=[
            html.Div(
                className="flex items-center justify-between",
                children=[
                    html.Span(
                        "Min weight",
                        className="text-[11px] uppercase tracking-wider text-slate-400",
                    ),
                    html.Span(
                        f"{value:.2f}",
                        id=TRADE_GRAPH_IDS["threshold_value_label"],
                        className="text-[11px] font-mono text-slate-200",
                    ),
                ],
            ),
            dcc.Slider(
                id=TRADE_GRAPH_IDS["threshold_slider"],
                min=0.0,
                max=1.0,
                step=0.05,
                value=value,
                marks=None,
                tooltip={"always_visible": False, "placement": "bottom"},
                # ``mouseup`` so we don't trigger a re-filter every
                # frame of a drag — sub-60ms locally but the network
                # tab still gets noisy.
                updatemode="mouseup",
            ),
        ],
    )


def _header_band(
    *,
    initial_layout:    str,
    initial_color_by:  str,
    initial_threshold: float,
) -> html.Div:
    """Sticky header with picker + segmented controls + threshold slider."""
    cluster_picker = html.Div(
        className="flex flex-col gap-1 min-w-[220px]",
        children=[
            html.Span(
                "Cluster",
                className="text-[11px] uppercase tracking-wider text-slate-400",
            ),
            dmc.Select(
                id=TRADE_GRAPH_IDS["cluster_select"],
                data=[],   # bootstrap callback populates from /clusters
                placeholder="Select a cluster…",
                searchable=True,
                clearable=False,
                size="sm",
            ),
        ],
    )

    return html.Div(
        className=(
            "rade-card flex flex-wrap items-end gap-4 sticky top-0 z-10"
        ),
        children=[
            cluster_picker,
            html.Div(
                className="flex flex-col gap-1",
                children=[
                    html.Span(
                        "Layout",
                        className="text-[11px] uppercase tracking-wider text-slate-400",
                    ),
                    _layout_radio(),
                ],
            ),
            html.Div(
                className="flex flex-col gap-1",
                children=[
                    html.Span(
                        "Color by",
                        className="text-[11px] uppercase tracking-wider text-slate-400",
                    ),
                    _color_by_select(value=initial_color_by),
                ],
            ),
            _threshold_slider(value=initial_threshold),
            html.Div(className="flex-1"),    # spacer
            dmc.Button(
                "Fit view",
                id=TRADE_GRAPH_IDS["fit_btn"],
                variant="default",
                size="sm",
                leftSection=DashIconify(icon="tabler:focus-2", width=16),
            ),
            dmc.Button(
                "Export PNG",
                id=TRADE_GRAPH_IDS["export_btn"],
                variant="outline",
                color="violet",
                size="sm",
                leftSection=DashIconify(icon="tabler:download", width=16),
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Cytoscape pane + mini-map
# ─────────────────────────────────────────────────────────────────────


def _cytoscape_pane(*, initial_layout: str) -> html.Div:
    """Cytoscape graph + bottom-right mini-map overlay.

    Layout plumbing
    ---------------
    * The outer card is ``flex flex-col`` — title row sits on top
      (``shrink-0``), the cytoscape wrapper takes ``flex-1`` so it
      eats the remaining height, and the status row pins to the
      bottom (``shrink-0``).  Combined with the parent grid's
      ``items-stretch``, the cytoscape pane auto-stretches to match
      the side-panel's intrinsic height.
    * The inner cytoscape wrapper is ``flex items-center
      justify-center`` so when the rendered graph is smaller than
      the pane (e.g. 25-node cluster on a 600-px-tall pane), it's
      vertically + horizontally centred rather than glued to the
      top-left.  Cytoscape's own ``fit: True`` layout option zooms
      the graph to fill the viewport, so this only matters in the
      edge cases where ``fit`` would over-shrink.
    """
    title_row = html.Div(
        className="flex items-center justify-between shrink-0",
        children=[
            html.Span(
                "Trade network",
                className="text-sm font-semibold text-slate-200",
            ),
            html.Span(
                "—",
                id=TRADE_GRAPH_IDS["pane_status"],
                className="text-xs text-slate-500",
            ),
        ],
    )

    cytoscape = cyto.Cytoscape(
        id=TRADE_GRAPH_IDS["cytoscape"],
        elements=[],
        # ``fit: True`` zooms the graph to fill the viewport on every
        # layout run — keeps the network centred even on resize.
        # Cose tuning options match ``_build_layout_cfg()`` in the
        # callback module so the very first render after the bootstrap
        # callback ships elements has a sensible spread.
        layout={
            "name":           initial_layout,
            "fit":            True,
            "padding":        30,
            "animate":        True,
            "randomize":      True,
            "nodeRepulsion":  8000,
            "idealEdgeLength":80,
            "nodeOverlap":    20,
            "numIter":        1000,
        },
        stylesheet=CYTOSCAPE_STYLESHEET,
        style={"width": "100%", "height": "100%"},
        minZoom=0.2,
        maxZoom=3.0,
        boxSelectionEnabled=False,
    )

    minimap = html.Div(
        className=(
            "absolute bottom-2 right-2 z-20 rounded-md "
            "border border-slate-700 bg-slate-900/70 backdrop-blur "
            "p-1 shadow-md pointer-events-none"
        ),
        # Inline style for the explicit pixel size — Tailwind doesn't
        # have arbitrary 140×90 utilities by default.
        style={"width": "140px", "height": "90px"},
        children=[
            cyto.Cytoscape(
                id=TRADE_GRAPH_IDS["cytoscape_minimap"],
                elements=[],
                layout={"name": initial_layout, "fit": True, "padding": 4, "animate": False},
                stylesheet=MINIMAP_STYLESHEET,
                style={"width": "100%", "height": "100%"},
                # Read-only — pan / zoom / box-select all disabled.
                userZoomingEnabled=False,
                userPanningEnabled=False,
                boxSelectionEnabled=False,
                autoungrabify=True,
                autounselectify=True,
            ),
        ],
    )

    cyto_wrapper = html.Div(
        className="flex-1 flex items-center justify-center relative min-h-[400px]",
        children=[cytoscape, minimap],
    )

    return html.Div(
        className="rade-card flex flex-col gap-2 lg:col-span-2",
        children=[
            title_row,
            cyto_wrapper,
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Selected-Trade card (with neighbours popover)
# ─────────────────────────────────────────────────────────────────────


def _empty_chip_strip() -> html.Div:
    return html.Div(
        id=TRADE_GRAPH_IDS["selected_chip_strip"],
        className="flex flex-wrap gap-1 mt-1 min-h-[22px]",
        children=[],
    )


def _empty_metrics_row() -> html.Div:
    return html.Div(
        id=TRADE_GRAPH_IDS["selected_metrics"],
        className="text-xs text-slate-500 italic mt-1",
        children="Click a node in the graph to inspect the trade.",
    )


def _neighbours_popover(*, initial_k: int) -> dmc.Popover:
    """Popover overlaying the "Nearest k" button.

    Inside the popover:

    * **k input** at the top — :class:`dmc.NumberInput` bounded to
      ``[1, 20]``.  Capture callback writes the value to session.
    * **scrollable list** below — render callback rebuilds the list
      with one ``html.Div`` per neighbour.  Each row carries a
      pattern-matching id (``{"type": "tg-neighbour-row",
      "trade_id": <id>}``) so a single ``ALL`` callback handles
      every row's click.
    """
    return dmc.Popover(
        position="bottom-start",
        withArrow=True,
        shadow="md",
        offset=4,
        # ``width`` on the parent Popover is the canonical knob in
        # dmc — it propagates to the portal'd dropdown reliably,
        # whereas ``style={"width": ...}`` on PopoverDropdown is
        # honoured by some dmc minor versions and ignored by others.
        # Set both so trade ids stay readable across versions.
        width=340,
        children=[
            dmc.PopoverTarget(
                dmc.Button(
                    # The visible label is updated by a render callback
                    # to reflect the current k (e.g. "Nearest 8").
                    html.Span(
                        f"Nearest {initial_k}",
                        id=TRADE_GRAPH_IDS["selected_neighbours_btn_label"],
                    ),
                    id=TRADE_GRAPH_IDS["selected_neighbours_btn"],
                    rightSection=DashIconify(icon="tabler:chevron-down", width=14),
                    variant="light",
                    color="violet",
                    size="xs",
                    # NOTE: never `disabled=True` here — disabled HTML
                    # buttons set ``pointer-events: none`` which prevents
                    # the popover trigger from firing.  The dropdown
                    # body shows a placeholder when no node is selected,
                    # so a stray click without a selection is harmless.
                ),
            ),
            dmc.PopoverDropdown(
                # 340 px is the sweet spot — wide enough to show the
                # full ``trade_<n>`` ids most parquets emit, narrow
                # enough to stay inside the side-panel column on a
                # 1280-px viewport.  The list itself uses ``truncate``
                # only as a defensive cap for truly huge ids.
                style={"width": "340px", "padding": "10px"},
                children=[
                    html.Div(
                        className="flex items-center justify-between mb-2",
                        children=[
                            html.Span(
                                "Nearest",
                                className=(
                                    "text-[11px] uppercase tracking-wider "
                                    "text-slate-400"
                                ),
                            ),
                            dmc.NumberInput(
                                id=TRADE_GRAPH_IDS["selected_neighbours_k_input"],
                                value=initial_k,
                                min=1,
                                max=20,
                                step=1,
                                size="xs",
                                w=72,
                            ),
                        ],
                    ),
                    html.Div(
                        id=TRADE_GRAPH_IDS["selected_neighbours_list"],
                        # Scrollable list — caps the popover height
                        # at ~280 px so very large k values don't
                        # push the popover off-screen.
                        className="flex flex-col gap-1 overflow-y-auto",
                        style={"maxHeight": "240px"},
                        children=[
                            html.Span(
                                "Pick a node to load its neighbours.",
                                className="text-xs text-slate-500 italic",
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def _selected_trade_card(*, initial_k: int) -> html.Div:
    """Fixed-height card — content swaps in / out as the user taps nodes.

    The card height is fixed at 200 px so the right-column stack
    (Selected Trade + Legend + Cluster Stats) has a deterministic
    height, which the Cytoscape pane then matches.  The neighbours
    popover overlays on top of the card; the in-card content never
    grows.
    """
    header = html.Div(
        className="flex items-center justify-between shrink-0",
        children=[
            html.Span(
                "Selected trade",
                className="text-xs uppercase tracking-wider text-slate-400",
            ),
            dmc.ActionIcon(
                DashIconify(icon="tabler:copy", width=14),
                id=TRADE_GRAPH_IDS["selected_copy_btn"],
                size="sm",
                color="violet",
                variant="subtle",
                # Enabled by the render callback when a trade is selected.
                disabled=True,
            ),
        ],
    )

    trade_id_row = html.Div(
        className="shrink-0",
        children=[
            html.Code(
                "—",
                id=TRADE_GRAPH_IDS["selected_trade_id"],
                className=(
                    "font-mono text-sm text-slate-100 truncate "
                    "block max-w-full"
                ),
            ),
        ],
    )

    button_row = html.Div(
        className="flex gap-2 mt-auto shrink-0",
        children=[
            _neighbours_popover(initial_k=initial_k),
            dmc.Button(
                "Open Deep Dive",
                id=TRADE_GRAPH_IDS["selected_deep_dive_btn"],
                rightSection=DashIconify(icon="tabler:arrow-up-right", width=14),
                variant="filled",
                color="violet",
                size="xs",
                style={"flex": 1},
                disabled=True,
            ),
        ],
    )

    return html.Div(
        id=TRADE_GRAPH_IDS["selected_card"],
        # ``h-[260px]`` fixes the card height; ``flex flex-col`` +
        # ``mt-auto`` on the button row pins it to the bottom even
        # when the body content is short (placeholder state).
        # 260 + 140 (legend) + 240 (cluster stats) + 2*12 px gaps =
        # 664 px total side-panel height; the cytoscape pane stretches
        # to match via ``items-stretch`` on the row grid.
        className="rade-card-compact flex flex-col gap-1 h-[260px]",
        children=[
            header,
            trade_id_row,
            _empty_chip_strip(),
            _empty_metrics_row(),
            button_row,
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Node Legend card  (h-[140px], rebuilt per color-by)
# ─────────────────────────────────────────────────────────────────────


def _initial_legend_body() -> html.Div:
    """Default legend body — matches the ``trade_type`` color-by mode.

    Render callback (``_register_render_legend``) rebuilds this when
    the user changes the color-by select.
    """
    def _row(color: str, label: str, sublabel: str) -> html.Div:
        return html.Div(
            className="flex items-center gap-2",
            children=[
                html.Div(
                    className="w-3 h-3 rounded-full flex-shrink-0",
                    style={"backgroundColor": color},
                ),
                html.Div(
                    className="flex flex-col leading-tight",
                    children=[
                        html.Span(label, className="text-xs text-slate-200"),
                        html.Span(sublabel, className="text-[10px] text-slate-500"),
                    ],
                ),
            ],
        )

    return html.Div(
        id=TRADE_GRAPH_IDS["legend_body"],
        className="flex flex-col gap-2",
        children=[
            _row("#f59e0b", "Target",     "Priced directly"),
            _row("#8b5cf6", "Elementary", "Building block"),
        ],
    )


def _legend_card() -> html.Div:
    return html.Div(
        id=TRADE_GRAPH_IDS["legend_card"],
        className="rade-card-compact flex flex-col gap-2 h-[140px]",
        children=[
            html.Span(
                "Node legend",
                className="text-xs uppercase tracking-wider text-slate-400 shrink-0",
            ),
            _initial_legend_body(),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Cluster Stats card  (h-[180px], 2×2 KPI grid)
# ─────────────────────────────────────────────────────────────────────


def _cluster_stats_card() -> html.Div:
    return html.Div(
        id=TRADE_GRAPH_IDS["cluster_stats_card"],
        # ``h-[240px]`` gives the inner 2×2 KPI grid headroom — the
        # previous ``h-[180px]`` was tighter than the four KpiCards'
        # combined natural height, so the card visually shrunk under
        # its contents.  Keeps the side-panel total at exactly 664 px
        # together with the 260 px Selected card and the 140 px
        # Legend card.
        className="rade-card-compact flex flex-col gap-2 h-[240px]",
        children=[
            html.Span(
                "Cluster stats",
                className="text-xs uppercase tracking-wider text-slate-400 shrink-0",
            ),
            html.Div(
                id=TRADE_GRAPH_IDS["cluster_stats_grid"],
                className="grid grid-cols-2 gap-2",
                children=[
                    KpiCard(label="Nodes",       value="—"),
                    KpiCard(label="Edges",       value="—"),
                    KpiCard(label="Density",     value="—"),
                    KpiCard(label="Mean weight", value="—"),
                ],
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Row composition
# ─────────────────────────────────────────────────────────────────────


def _row_main_area(
    *,
    initial_layout:           str,
    initial_neighbour_k:      int,
) -> html.Div:
    side_panel = html.Div(
        className="flex flex-col gap-3",
        children=[
            _selected_trade_card(initial_k=initial_neighbour_k),
            _legend_card(),
            _cluster_stats_card(),
        ],
    )

    return html.Div(
        # ``items-stretch`` is the keystone — it makes the cytoscape
        # pane's outer card auto-stretch to match the side panel's
        # intrinsic height (sum of three fixed-height cards + gaps).
        className="grid grid-cols-1 lg:grid-cols-3 gap-3 items-stretch",
        children=[
            _cytoscape_pane(initial_layout=initial_layout),
            side_panel,
        ],
    )


def _row_density() -> html.Div:
    return ChartContainer(
        title="Density distribution",
        subtitle="Per-cluster · selected cluster highlighted",
        graph_id=TRADE_GRAPH_IDS["density_chart"],
        height=280,
    )


def _row_edges_vs_nodes() -> html.Div:
    return ChartContainer(
        title="Edges vs nodes",
        subtitle="Marker size · density   ·   colour · mean weight",
        graph_id=TRADE_GRAPH_IDS["edges_vs_nodes_chart"],
        height=280,
    )


# ─────────────────────────────────────────────────────────────────────
# Public entrypoint
# ─────────────────────────────────────────────────────────────────────


def build_trade_graph(*, session: Optional[Session] = None) -> html.Div:
    """Build the Trade-Graph sub-tab layout.

    Parameters
    ----------
    session
        Optional :class:`Session` — initial widget values are seeded
        from session per Page Contract §3 Rule L1 (no value-side
        hydration callback).  When ``None`` (e.g. early splash, tests),
        falls back to module defaults.

    Returns
    -------
    html.Div
        The full sub-tab layout, ready to be mounted under the
        Evaluation shell.  Contains the mount tripwire Store, the
        ephemeral graph-payload Store, four content rows and an
        otherwise-empty deep-dive `dcc.Location` companion.
    """
    if session is not None:
        ev = session.evaluation
        initial_layout      = ev.trade_graph_layout
        initial_color_by    = ev.trade_graph_color_by
        initial_threshold   = ev.trade_graph_weight_threshold
        initial_neighbour_k = ev.trade_graph_neighbour_k
    else:
        initial_layout      = DEFAULT_TRADE_GRAPH_LAYOUT
        initial_color_by    = DEFAULT_TRADE_GRAPH_COLOR_BY
        initial_threshold   = DEFAULT_TRADE_GRAPH_WEIGHT_THRESHOLD
        initial_neighbour_k = DEFAULT_TRADE_GRAPH_NEIGHBOUR_K

    return html.Div(
        id=TRADE_GRAPH_IDS["root"],
        className="rade-evaluation-subtab flex flex-col gap-4",
        children=[
            # Mount tripwire — Page Contract §3 Rule L4.
            dcc.Store(
                id=TRADE_GRAPH_IDS["mount_signal"],
                data=True,
                storage_type="memory",
            ),
            # Ephemeral graph payload — caches the last
            # ``TradeGraphResponse`` so node-tap rendering, neighbour
            # lookups and threshold filtering don't re-fetch.
            dcc.Store(
                id=TRADE_GRAPH_IDS["store_graph"],
                data={},
                storage_type="memory",
            ),
            _header_band(
                initial_layout=initial_layout,
                initial_color_by=initial_color_by,
                initial_threshold=initial_threshold,
            ),
            _row_main_area(
                initial_layout=initial_layout,
                initial_neighbour_k=initial_neighbour_k,
            ),
            _row_density(),
            _row_edges_vs_nodes(),
        ],
    )


__all__ = [
    "COLOR_BY_LABELS",
    "CYTOSCAPE_STYLESHEET",
    "MINIMAP_STYLESHEET",
    "TRADE_GRAPH_IDS",
    "build_trade_graph",
]

"""Evaluation → Cluster Deep-Dive sub-tab layout (Phase E.5 hybrid).

Four-row layout designed against ``rade_cluster_deep_dive.png``:

    Row 1 · Header band       (cluster picker · "open Trade-Graph" link;
                               split toggle stays on the topbar so the
                               header chrome stays light)

    Row 2 · Cluster context   (left rail: Cluster Attributes /
                               Cluster Metrics / Graph Statistics
                               stacked at exactly the same height as the
                               right pane; right pane: Cluster
                               Portfolio chart over Trade-Level Metrics
                               grid; click-row → Row 3 per-trade expands.)

    Row 2.5 · Convergence      Residual-over-time chart, then Training
                               curves (stacked full-width below with
                               chip group; long page, clear reading order.)

    Row 3 · Per-trade detail  (residual distribution + bias-vs-magnitude
                               scatter, both per-scenario for the
                               selected trade.  *Collapsed to zero
                               height when no trade is selected* —
                               the page stays compact on first paint.)

    Row 4 · Elementary PnL    (Elementary PnL Explorer table on the
                               left, Elementary PnL multi-line chart on
                               the right.  Chart shows an empty-state
                               placeholder until the user selects one
                               or more elementary trades.)

Why the hybrid (vs. a pure mock copy)?  The mock has the cluster
attributes / metrics / graph-stats / convergence as four stacked left-
rail cards.  We collapse that to three metric cards on the left rail;
the training-curves block sits in a full-width stack *under* residual
over time (not squeezed beside it), and we drop ``Avg Degree`` /
``Avg Path Length`` from the Graph Statistics card (those numbers would
need a new endpoint and the user explicitly said "show only what we have
today").

All dynamic ids live in :data:`CLUSTER_DEEP_DIVE_IDS` so callbacks
never hardcode strings.  Callbacks live in
:mod:`..callbacks.cluster_deep_dive_cb`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import dash_mantine_components as dmc
from dash import dcc, html
from dash_iconify import DashIconify

from ...components.ag_grid_table import AgGridTable
from ...components.chart_container import ChartContainer
from ...components.kpi_card import KpiCard
from ...data.session import Session


CLUSTER_DEEP_DIVE_IDS: Dict[str, str] = {
    # ── Page root ────────────────────────────────────────────────
    "root": "eval-cluster-root",

    # Mount tripwire — see ``mount_signal`` rationale below.
    "mount_signal": "eval-cluster-mount-signal",

    # ── Row 1 · Header band ──────────────────────────────────────
    "cluster_select":         "eval-cluster-cluster-select",
    "open_trade_graph_btn":   "eval-cluster-open-trade-graph-btn",

    # ── Row 2 · Left rail ────────────────────────────────────────
    # Cluster Attributes card (key/value rows fed by the render
    # callback from /clusters)
    "attributes_card":        "eval-cluster-attributes-card",
    "attributes_body":        "eval-cluster-attributes-body",

    # Cluster Metrics card — KpiCard ×4 with sparklines fed from
    # /trades (per-trade aggregate distribution across the cluster)
    "metrics_card":           "eval-cluster-metrics-card",
    "kpi_mae_card":           "eval-cluster-kpi-mae-card",
    "kpi_mae_value":          "eval-cluster-kpi-mae-value",
    "kpi_mae_spark":          "eval-cluster-kpi-mae-spark",
    "kpi_rmse_card":          "eval-cluster-kpi-rmse-card",
    "kpi_rmse_value":         "eval-cluster-kpi-rmse-value",
    "kpi_rmse_spark":         "eval-cluster-kpi-rmse-spark",
    "kpi_p95_card":           "eval-cluster-kpi-p95-card",
    "kpi_p95_value":          "eval-cluster-kpi-p95-value",
    "kpi_p95_spark":          "eval-cluster-kpi-p95-spark",
    "kpi_p99_card":           "eval-cluster-kpi-p99-card",
    "kpi_p99_value":          "eval-cluster-kpi-p99-value",
    "kpi_p99_spark":          "eval-cluster-kpi-p99-spark",

    # Graph Statistics card (Nodes / Edges / Density real values; Avg
    # Degree + Avg Path Length deliberately deferred — rendered as
    # ``—`` placeholders so the visual structure matches the mock.)
    "graph_stats_card":       "eval-cluster-graph-stats-card",
    "graph_stats_body":       "eval-cluster-graph-stats-body",

    # ── Row 2 · Right pane ───────────────────────────────────────
    # Cluster Portfolio (predicted vs target line, /cluster-timeseries)
    "portfolio_chart":        "eval-cluster-portfolio-chart",
    # Trade-Level Metrics (AgGrid, /trades)
    "trades_grid":            "eval-cluster-trades-grid",

    # ── Row 3 · Per-trade detail (collapse-on-no-selection) ──────
    "row3_wrapper":           "eval-cluster-row3-wrapper",
    "selected_trade_chip":    "eval-cluster-selected-trade-chip",
    "selected_trade_label":   "eval-cluster-selected-trade-label",
    "selected_trade_clear_btn": "eval-cluster-selected-trade-clear-btn",
    "per_trade_residual_hist": "eval-cluster-per-trade-residual-hist",
    "per_trade_bias_scatter":  "eval-cluster-per-trade-bias-scatter",

    # ── Row 4 · Elementary PnL Explorer ──────────────────────────
    # Multi-select, searchable dropdown of elementary trade ids — drives
    # the chart below.  Replaced the previous AgGrid + checkbox pattern
    # because elementary trades carry no predictions / metrics for the
    # /trades endpoint to surface, so the grid had nothing useful to
    # show alongside ``trade_id``; a typeahead picker is denser and
    # avoids paginating through hundreds of ids.
    "elementary_select":          "eval-cluster-elementary-select",
    "elementary_reset_btn":       "eval-cluster-elementary-reset-btn",
    "elementary_pnl_chart":       "eval-cluster-elementary-pnl-chart",
    "elementary_pnl_chart_card":  "eval-cluster-elementary-pnl-chart-card",
    "elementary_pnl_empty":       "eval-cluster-elementary-pnl-empty",

    # ── Convergence strip (residual → training curves, stacked) ──
    "training_curves_chart":      "eval-cluster-training-curves-chart",
    "training_curves_chip_group": "eval-cluster-training-curves-chip-group",
    "training_curves_chip_empty": "eval-cluster-training-curves-chip-empty",
    "residual_ts_chart":          "eval-cluster-residual-ts-chart",

    # ── Ephemeral stores ─────────────────────────────────────────
    # Trade-type map ``{trade_id: "target" | "elementary"}`` populated
    # from /trade-graph; consumed by Row 4's elementary explorer
    # (filter ``trade_type == "elementary"``) and by Row 3's selected-
    # trade chip subtitle.
    "store_trade_types":     "eval-cluster-trade-types-store",

    # Optional metric chip list (mirrors training_curves chip group's
    # ``data`` prop).  Avoids redundant /training-curves fetches when
    # only the chip *value* changes.
    "store_curve_metrics":   "eval-cluster-curve-metrics-store",
}


# ─────────────────────────────────────────────────────────────────────
# Row 1 — Header band
# ─────────────────────────────────────────────────────────────────────


def _header_band(*, initial_cluster_id: Optional[str] = None) -> html.Div:
    """Sticky header with cluster picker + Trade-Graph button.

    Split (train / val / test) lives on the topbar in this app, so the
    deep-dive header stays a thin context strip — cluster picker on
    the left, "Trade-Graph" deep link on the right.

    Parameters
    ----------
    initial_cluster_id
        Seed for the cluster :class:`dmc.Select`'s ``value`` prop, sourced
        from ``session.evaluation.deep_dive_cluster_id`` (or the top-level
        ``session.cluster_id`` fallback) at build time — Page Contract
        §3 Rule L1.  When ``None``, the picker shows its placeholder
        until the user / bootstrap picks one.  ``data`` is left empty
        here; the bootstrap callback populates it after fetching the
        version-keyed cluster list.
    """
    cluster_picker = html.Div(
        className="flex flex-col gap-1 min-w-[220px]",
        children=[
            html.Span(
                "Cluster",
                className="text-[11px] uppercase tracking-wider text-slate-400",
            ),
            dmc.Select(
                id=CLUSTER_DEEP_DIVE_IDS["cluster_select"],
                data=[],
                value=initial_cluster_id,
                placeholder="Select a cluster…",
                searchable=True,
                clearable=False,
                size="sm",
            ),
        ],
    )

    open_trade_graph_btn = dmc.Button(
        "Trade-Graph",
        id=CLUSTER_DEEP_DIVE_IDS["open_trade_graph_btn"],
        variant="light",
        color="violet",
        size="sm",
        leftSection=DashIconify(icon="tabler:share-2", width=16),
    )

    return html.Div(
        className="rade-card flex items-end justify-between gap-4 sticky top-0 z-10",
        children=[
            cluster_picker,
            open_trade_graph_btn,
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Row 2 — Left rail (3 stacked cards)
# ─────────────────────────────────────────────────────────────────────


_ATTRIBUTE_PLACEHOLDER_KEYS = (
    # Visual order in the card.  Matches the mock screenshot top-down.
    ("Asset Class",   "asset_class"),
    ("Currency",      "currency"),
    ("Desk",          "desk"),
    ("Product",       "product"),
    ("N Trades",      "n_trades"),
    ("N Scenarios",   "n_scenarios"),
)


def _attribute_row(label: str, value: str = "—") -> html.Div:
    """One key/value row inside Cluster Attributes / Graph Statistics."""
    return html.Div(
        className="flex items-center justify-between text-xs",
        children=[
            html.Span(label, className="text-slate-400"),
            html.Span(value, className="text-slate-100 font-medium"),
        ],
    )


def _cluster_attributes_card() -> html.Div:
    """Top card on the left rail — placeholder rows the callback fills.

    Pre-rendering placeholder rows means the card has the right
    *shape* on first paint (so the row 1 height calculation is
    stable), and the bootstrap callback only needs to swap text
    children — no children-replacement re-mount.

    No ``flex-1`` on the outer card — we sized the whole row to
    intrinsic content and rely on the grid's ``items-stretch`` to
    equalise the right pane's height.  ``flex-1`` on a card whose
    parent has no definite height collapses the card to its content
    (fine) or fights for non-existent leftover space with its
    siblings (also fine but confusing).  Dropping it keeps the
    layout predictable across viewport sizes.
    """
    body = html.Div(
        id=CLUSTER_DEEP_DIVE_IDS["attributes_body"],
        className="flex flex-col gap-1",
        children=[
            _attribute_row(label) for label, _ in _ATTRIBUTE_PLACEHOLDER_KEYS
        ],
    )
    return html.Div(
        id=CLUSTER_DEEP_DIVE_IDS["attributes_card"],
        className="rade-card-compact flex flex-col gap-2",
        children=[
            html.Div(
                "Cluster Attributes",
                className="text-sm font-semibold text-slate-200",
            ),
            body,
        ],
    )


def _cluster_metrics_card() -> html.Div:
    """Middle card on the left rail — KpiCard ×4 with sparklines.

    The four KPIs (MAE / RMSE / P95 / P99) come from the per-trade
    parquet (one row per trade).  Sparkline data is the per-trade
    distribution across the cluster's trades — gives the user a
    sense of distribution shape (single-mode vs heavy-tailed) without
    drilling into the table below.
    """
    return html.Div(
        id=CLUSTER_DEEP_DIVE_IDS["metrics_card"],
        className="rade-card-compact flex flex-col gap-2",
        children=[
            html.Div(
                "Cluster Metrics",
                className="text-sm font-semibold text-slate-200",
            ),
            html.Div(
                # 2×2 KpiCard grid; each KpiCard already has its own
                # padded background, so the outer wrapper just lays
                # them out.  No ``flex-1`` here — the KpiCards with
                # sparklines are ~110 px tall each, so this grid is
                # ~235 px and drives the Cluster Metrics card to its
                # natural height.
                className="grid grid-cols-2 gap-2",
                children=[
                    KpiCard(
                        label="MAE",
                        value="—",
                        card_id=CLUSTER_DEEP_DIVE_IDS["kpi_mae_card"],
                        value_id=CLUSTER_DEEP_DIVE_IDS["kpi_mae_value"],
                        sparkline_id=CLUSTER_DEEP_DIVE_IDS["kpi_mae_spark"],
                        icon="tabler:arrow-narrow-down",
                    ),
                    KpiCard(
                        label="RMSE",
                        value="—",
                        card_id=CLUSTER_DEEP_DIVE_IDS["kpi_rmse_card"],
                        value_id=CLUSTER_DEEP_DIVE_IDS["kpi_rmse_value"],
                        sparkline_id=CLUSTER_DEEP_DIVE_IDS["kpi_rmse_spark"],
                        icon="tabler:square-root",
                    ),
                    KpiCard(
                        label="P95",
                        value="—",
                        card_id=CLUSTER_DEEP_DIVE_IDS["kpi_p95_card"],
                        value_id=CLUSTER_DEEP_DIVE_IDS["kpi_p95_value"],
                        sparkline_id=CLUSTER_DEEP_DIVE_IDS["kpi_p95_spark"],
                        icon="tabler:percentage",
                    ),
                    KpiCard(
                        label="P99",
                        value="—",
                        card_id=CLUSTER_DEEP_DIVE_IDS["kpi_p99_card"],
                        value_id=CLUSTER_DEEP_DIVE_IDS["kpi_p99_value"],
                        sparkline_id=CLUSTER_DEEP_DIVE_IDS["kpi_p99_spark"],
                        icon="tabler:zoom-exclamation",
                    ),
                ],
            ),
        ],
    )


_GRAPH_STATS_PLACEHOLDER_ROWS = (
    # Real values populated by the render callback from /trade-graph.
    ("Nodes",            "graph_stats_n_nodes"),
    ("Edges",            "graph_stats_n_edges"),
    ("Density",          "graph_stats_density"),
    # Deferred — show "—" with a "(not yet computed)" tone.  Keeps the
    # card visually faithful to the mock without inventing numbers.
    ("Avg Degree",       None),
    ("Avg Path Length",  None),
)


def _graph_stats_card() -> html.Div:
    """Bottom card on the left rail — graph-level cluster summary.

    Real values: Nodes / Edges / Density (sourced from /trade-graph).
    Placeholders: Avg Degree / Avg Path Length (no endpoint yet — the
    user explicitly chose 'show only what we have today' over building
    a new endpoint just for these two numbers).
    """
    rows: List[Any] = []
    for label, _id in _GRAPH_STATS_PLACEHOLDER_ROWS:
        rows.append(_attribute_row(label, "—"))

    body = html.Div(
        id=CLUSTER_DEEP_DIVE_IDS["graph_stats_body"],
        className="flex flex-col gap-1",
        children=rows,
    )

    return html.Div(
        id=CLUSTER_DEEP_DIVE_IDS["graph_stats_card"],
        className="rade-card-compact flex flex-col gap-2",
        children=[
            html.Div(
                "Graph Statistics",
                className="text-sm font-semibold text-slate-200",
            ),
            body,
        ],
    )


def _row2_left_rail() -> html.Div:
    """Three stacked cards filling the left col of Row 2 evenly.

    ``min-h-0`` on the rail itself + ``flex-1`` on each card lets the
    flex children shrink below their intrinsic content height when
    the row is constrained by ``h-[560px]`` — without it a long
    Cluster-Attributes body would push the rail past the row.
    """
    return html.Div(
        className="lg:col-span-2 flex flex-col gap-3 min-h-0",
        children=[
            _cluster_attributes_card(),
            _cluster_metrics_card(),
            _graph_stats_card(),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Row 2 — Right pane (chart + grid)
# ─────────────────────────────────────────────────────────────────────


def _trades_grid_column_defs() -> List[Dict[str, Any]]:
    """Initial columnDefs — the callback overrides on data arrival."""
    return [
        {"field": "trade_id",      "headerName": "Trade",        "flex": 2, "minWidth": 160},
        {"field": "trade_type",    "headerName": "Type",         "flex": 1, "minWidth": 100},
        {"field": "mae",           "headerName": "MAE",          "flex": 1, "type": "numericColumn"},
        {"field": "rmse",          "headerName": "RMSE",         "flex": 1, "type": "numericColumn"},
        {"field": "p95_ae",        "headerName": "P95 |err|",    "flex": 1, "type": "numericColumn"},
        {"field": "mean_residual", "headerName": "Mean resid.",  "flex": 1, "type": "numericColumn"},
        {"field": "std_residual",  "headerName": "Std resid.",   "flex": 1, "type": "numericColumn"},
        {"field": "n_scenarios",   "headerName": "Scenarios",    "flex": 1, "type": "numericColumn"},
    ]


def _row2_right_pane() -> html.Div:
    """Cluster Portfolio chart over Trade-Level Metrics grid.

    Both children size to their natural content heights — we no
    longer impose a fixed row height on the grandparent, so each
    card renders the height its intrinsic chrome + graph / grid
    needs.  ``items-stretch`` on the row grid parent then equalises
    right pane ↔ left rail at whichever column is tallest.

    The grid uses ``rowSelection: 'single'`` so a row click fires
    the per-trade-detail row 3 expand.
    """
    portfolio = ChartContainer(
        title="Cluster Portfolio",
        subtitle="Predicted vs target PnL across scenarios for the active split",
        graph_id=CLUSTER_DEEP_DIVE_IDS["portfolio_chart"],
        height=240,            # graph height; card chrome adds ≈ 70 px
    )

    grid_header = html.Div(
        className="flex items-start justify-between",
        children=[
            html.Div(
                className="flex flex-col",
                children=[
                    html.Div(
                        "Trade-Level Metrics",
                        className="text-sm font-semibold text-slate-200",
                    ),
                    html.Div(
                        "Per-trade aggregate metrics for this cluster.  "
                        "Click a row to inspect that trade in detail.",
                        className="text-xs text-slate-500",
                    ),
                ],
            ),
        ],
    )

    grid = AgGridTable(
        grid_id=CLUSTER_DEEP_DIVE_IDS["trades_grid"],
        column_defs=_trades_grid_column_defs(),
        row_data=[],
        # Ag-grid needs an explicit height when its parent doesn't
        # impose one — without it the grid renders at 0 px because
        # the flex chain has no definite height to divide.  240 px
        # shows ~6 rows + header + pagination on a default viewport.
        height=240,
        className="rade-cluster-trades-grid",
        grid_options={"rowSelection": "single"},
        getRowId="params.data.trade_id",
    )

    return html.Div(
        # Gap between Cluster Portfolio chart and Trade-Level Metrics card —
        # slightly wider than intra-card spacing so reads as two deliberate
        # panels in the column.
        className="lg:col-span-3 flex flex-col gap-5 min-w-0",
        children=[
            portfolio,
            html.Div(
                # Inner wrapper card so the grid sits in the same
                # ``rade-card`` chrome as the chart above it.
                className="rade-card flex flex-col gap-2",
                children=[grid_header, grid],
            ),
        ],
    )


def _row2_main_area() -> html.Div:
    """Row 1 main-area wrapper — left-rail / right-pane grid.

    Intrinsic-height strategy — we let the left rail (Attributes +
    Metrics-with-sparklines + Graph Stats) dictate the row height
    (~600 px total) and rely on ``items-stretch`` to stretch the
    right pane to match.

    An earlier revision locked this row at ``h-[560px]`` with
    ``overflow-hidden``, which did exactly what it said — clipped
    the Cluster Metrics card's overflow — but the clip happened
    mid-KpiCard and the missing 40-80 px of content visually leaked
    into Row 2 through box-shadows and sparkline tails.  Letting
    content drive is self-healing: the cards fit, the columns
    line up, and sub-1024-px viewports (where ``lg:grid-cols-5``
    falls back to ``grid-cols-1``) stack the rail above the pane
    without any mystery overlap.
    """
    return html.Div(
        className="grid grid-cols-1 lg:grid-cols-5 gap-3 items-stretch",
        children=[
            _row2_left_rail(),
            _row2_right_pane(),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Row 2.5 — Residual-over-time + Training curves
# ─────────────────────────────────────────────────────────────────────


def _curves_chip_group(*, initial_metrics: Optional[List[str]] = None) -> html.Div:
    """ChipGroup for the training-curves overlay filter.

    ``train_loss`` is always shown on the chart and so is deliberately
    absent from the chip group — chips only pick *extra* series to
    overlay (``val_loss``, ``mae``, ``val_mae``, …).  The render
    callback populates the chip group's ``children`` based on the
    trainer's emitted metric list and shows the empty-state message
    when only ``train_loss`` is present.

    Default selection (``train_loss + val_loss``) is enforced server-
    side by the bootstrap callback — we just seed the existing session
    value here; the callback union/intersects it with the available
    metrics.
    """
    return html.Div(
        className="flex flex-col gap-2",
        children=[
            html.Div(
                className="flex items-center justify-between",
                children=[
                    html.Span(
                        "Overlay metrics",
                        className="text-[11px] uppercase tracking-wider text-slate-400",
                    ),
                    html.Span(
                        "train_loss always shown",
                        className="text-[11px] text-slate-500",
                    ),
                ],
            ),
            html.Div(
                className="flex items-center gap-1 flex-wrap min-h-[28px]",
                children=[
                    dmc.ChipGroup(
                        id=CLUSTER_DEEP_DIVE_IDS["training_curves_chip_group"],
                        multiple=True,
                        value=list(initial_metrics or []),
                        children=[],
                    ),
                    html.Span(
                        "No additional metrics emitted for this cluster.",
                        id=CLUSTER_DEEP_DIVE_IDS["training_curves_chip_empty"],
                        className="text-xs text-slate-500",
                        style={"display": "none"},
                    ),
                ],
            ),
        ],
    )


def _row_residual_and_curves(
    *,
    initial_curve_metrics: Optional[List[str]] = None,
) -> html.Div:
    """Residual-over-time on top; training curves full-width underneath.

    Stacked vertically with ``gap-8`` so both charts span the content width
    and the chip group under the convergence plot stays readable --- the user
    accepted a longer scrolling page instead of cramming curves beside the
    residual chart at ``lg`` breakpoints.
    """
    residual = ChartContainer(
        title="Residual over time",
        subtitle="Rolling absolute error with ±1σ band across the active split",
        graph_id=CLUSTER_DEEP_DIVE_IDS["residual_ts_chart"],
        height=300,
    )

    curves = html.Div(
        className="rade-card flex w-full min-w-0 flex-col gap-2 min-h-0",
        children=[
            html.Div(
                className="flex items-start justify-between",
                children=[
                    html.Div(
                        className="flex flex-col",
                        children=[
                            html.Div(
                                "Training curves",
                                className="text-sm font-semibold text-slate-200",
                            ),
                            html.Div(
                                "Per-epoch loss for this cluster's member.  "
                                "Default: train_loss + val_loss.",
                                className="text-xs text-slate-500",
                            ),
                        ],
                    ),
                ],
            ),
            dcc.Graph(
                id=CLUSTER_DEEP_DIVE_IDS["training_curves_chart"],
                figure={},
                style={"height": "280px"},
                config={
                    "displaylogo": False,
                    "modeBarButtonsToRemove": [
                        "select2d", "lasso2d", "autoScale2d", "toggleSpikelines",
                    ],
                },
            ),
            _curves_chip_group(initial_metrics=initial_curve_metrics),
        ],
    )

    return html.Div(
        className="flex w-full min-w-0 flex-col gap-8",
        children=[residual, curves],
    )


# ─────────────────────────────────────────────────────────────────────
# Row 3 — Per-trade detail (collapse-on-no-selection)
# ─────────────────────────────────────────────────────────────────────


def _selected_trade_chip() -> html.Div:
    """Header chip + clear button for the per-trade detail row.

    Shown only when a trade is selected (the row 3 wrapper handles
    ``display: none`` for the empty case, so this child can stay
    mounted and just have its label updated).
    """
    return html.Div(
        id=CLUSTER_DEEP_DIVE_IDS["selected_trade_chip"],
        className="rade-focus-chip flex items-center gap-2",
        children=[
            DashIconify(
                icon="tabler:target",
                width=14,
                className="text-emerald-400",
            ),
            html.Span(
                "Trade: —",
                id=CLUSTER_DEEP_DIVE_IDS["selected_trade_label"],
                className="text-xs text-slate-300",
            ),
            html.Button(
                "× Clear",
                id=CLUSTER_DEEP_DIVE_IDS["selected_trade_clear_btn"],
                className="rade-focus-chip-close",
                **{"aria-label": "Clear trade selection"},
            ),
        ],
    )


def _row_per_trade_detail() -> html.Div:
    """Row 3 — empty by default; render callback toggles ``display:none``.

    When no trade is selected the wrapper has ``display: none`` so the
    page is shorter and the elementary-trade row sits closer to the
    grid above it.  When a trade is clicked, the render callback flips
    the wrapper visible, fills both charts, and the focus chip in the
    header reads ``Trade: <id> · Type: <target|elementary>``.
    """
    histogram = ChartContainer(
        title="Per-trade residual distribution",
        subtitle="Histogram of (predicted − target) across all scenarios",
        graph_id=CLUSTER_DEEP_DIVE_IDS["per_trade_residual_hist"],
        height=320,
    )

    bias_scatter = ChartContainer(
        title="Per-trade bias vs magnitude",
        subtitle=(
            "x: predicted PnL  ·  y: residual (predicted − target)  ·  "
            "colour: |residual|"
        ),
        graph_id=CLUSTER_DEEP_DIVE_IDS["per_trade_bias_scatter"],
        height=320,
        actions=[_selected_trade_chip()],
        config={"doubleClick": "reset"},
    )

    return html.Div(
        id=CLUSTER_DEEP_DIVE_IDS["row3_wrapper"],
        # ``display: none`` until a trade is picked.  The render
        # callback flips this style + scrolls the row into view via a
        # clientside callback (see ``cluster_deep_dive_cb``).
        style={"display": "none"},
        className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-stretch",
        children=[histogram, bias_scatter],
    )


# ─────────────────────────────────────────────────────────────────────
# Row 4 — Elementary PnL Explorer + multi-line timeseries
# ─────────────────────────────────────────────────────────────────────


def _row_elementary_pnl(*, initial_selected: Optional[List[str]] = None) -> html.Div:
    """Row 4 — Elementary PnL Explorer (full-width).

    A single full-width card hosting a multi-select, searchable
    dropdown of elementary trade ids on top and the per-scenario PnL
    multi-line chart underneath.  The chart card lives behind an
    empty-state placeholder until the user picks ≥1 elementary trade;
    once selected, the chart fills with one line per trade
    (x = scenario_idx, y = raw PnL — no predictions / targets, since
    elementary trades are model *inputs*).

    Why a dropdown instead of a grid?

    * The /trades endpoint is target-only by data model so the previous
      AgGrid had no metric columns to display alongside ``trade_id``.
    * Clusters can carry hundreds of elementary trades; a typeahead
      picker is denser than a paginated grid and immediately
      searchable by partial id.
    * The Reset button stays on the same row so the user can clear
      the picker without scrolling.

    Parameters
    ----------
    initial_selected
        Trade ids to seed the picker with on first paint — usually
        sourced from the session's ``deep_dive_elementary_trade_ids``
        so that browser refresh / deep-link preserves the user's prior
        selection.  Defaults to the empty list (chart shows the
        empty-state placeholder).
    """
    header = html.Div(
        className="flex items-start justify-between",
        children=[
            html.Div(
                className="flex flex-col",
                children=[
                    html.Div(
                        "Elementary PnL Explorer",
                        className="text-sm font-semibold text-slate-200",
                    ),
                    html.Div(
                        "Atomic legs / hedge instruments in this cluster.  "
                        "Pick one or more from the dropdown to plot their "
                        "per-scenario PnL.",
                        className="text-xs text-slate-500",
                    ),
                ],
            ),
            dmc.Button(
                "Reset selection",
                id=CLUSTER_DEEP_DIVE_IDS["elementary_reset_btn"],
                variant="subtle",
                color="gray",
                size="xs",
                leftSection=DashIconify(icon="tabler:rotate", width=14),
            ),
        ],
    )

    # Searchable multi-select.  Options are populated on mount by the
    # render callback (it pulls ``elementary_set`` off
    # ``store_trade_types`` and rebuilds the option list whenever the
    # user switches cluster).  ``clearable`` so an empty selection
    # collapses back to the empty-state placeholder.
    picker = dmc.MultiSelect(
        id=CLUSTER_DEEP_DIVE_IDS["elementary_select"],
        data=[
            {"value": tid, "label": tid}
            for tid in (initial_selected or [])
        ],
        value=list(initial_selected or []),
        placeholder="Search and select elementary trades…",
        searchable=True,
        clearable=True,
        nothingFoundMessage="No matching trade ids in this cluster.",
        maxValues=12,
        # Comfortably wide on desktop, full-width on mobile.  ``min-w-0``
        # so the picker can shrink inside flex parents without overflow.
        className="min-w-0",
        style={"width": "100%"},
    )

    chart_empty_state = html.Div(
        id=CLUSTER_DEEP_DIVE_IDS["elementary_pnl_empty"],
        className=(
            "rade-list-empty flex flex-1 flex-col items-center justify-center "
            "gap-2 px-4 py-5 text-center min-h-[200px]"
        ),
        children=[
            DashIconify(
                icon="tabler:chart-line-off", width=24,
                className="text-slate-600",
            ),
            html.Div(
                "Pick one or more elementary trades from the dropdown "
                "above to plot their per-scenario PnL.",
                className="text-xs text-slate-500 max-w-sm",
            ),
        ],
    )

    chart = ChartContainer(
        title="Elementary PnL timeseries",
        subtitle="Raw PnL per scenario for the selected elementary trades",
        graph_id=CLUSTER_DEEP_DIVE_IDS["elementary_pnl_chart"],
        height=360,
        container_id=CLUSTER_DEEP_DIVE_IDS["elementary_pnl_chart_card"],
        # Hidden until the user picks ≥1 elementary trade — render cb
        # flips visibility vs ``elementary_pnl_empty``.
        style={"display": "none"},
    )

    return html.Div(
        className="rade-card flex flex-col gap-4",
        children=[header, picker, chart_empty_state, chart],
    )


# ─────────────────────────────────────────────────────────────────────
# Public builder
# ─────────────────────────────────────────────────────────────────────


def build_cluster_deep_dive(*, session: Optional[Session] = None) -> html.Div:
    """Assemble the Cluster Deep-Dive sub-tab (pure layout, no callbacks).

    Page Contract §3 Rule L1 — initial UI state is seeded from session
    at build time so the sub-tab paints the user's previously-chosen
    cluster + overlay metrics immediately on mount.  The bootstrap
    callback (``cluster_deep_dive_cb._register_bootstrap``) populates
    the ``Select.data`` option list and resolves the URL ``?cid=``
    deep-link / fresh-user fallback after the option list lands; it
    never overrides the layout-time seed for a value that's already
    valid.

    Parameters
    ----------
    session
        Live :class:`Session` whose :attr:`Session.evaluation` slice
        feeds the cluster-Select value and the curves-overlay chip
        group.  ``None`` falls back to a fresh :class:`Session` so
        unit tests / preview scripts that don't thread session
        through still render a sensible default (empty Select +
        empty chips).
    """
    sess = session or Session()
    eval_state = sess.evaluation
    initial_cluster_id = (
        eval_state.deep_dive_cluster_id or sess.cluster_id
    )

    return html.Div(
        id=CLUSTER_DEEP_DIVE_IDS["root"],
        # Vertical rhythm between sticky header strip, main rails, residual
        # curves row, optional per-trade detail, and elementary PnL.
        className="rade-evaluation-subtab flex flex-col gap-8",
        children=[
            _header_band(initial_cluster_id=initial_cluster_id),
            _row2_main_area(),
            _row_residual_and_curves(
                initial_curve_metrics=eval_state.deep_dive_curve_metrics,
            ),
            _row_per_trade_detail(),
            _row_elementary_pnl(
                initial_selected=eval_state.deep_dive_elementary_trade_ids,
            ),

            # ── Stores ────────────────────────────────────────────
            dcc.Store(
                id=CLUSTER_DEEP_DIVE_IDS["store_trade_types"],
                data={},
                storage_type="memory",
            ),
            dcc.Store(
                id=CLUSTER_DEEP_DIVE_IDS["store_curve_metrics"],
                data=[],
                storage_type="memory",
            ),

            # Mount tripwire — see ``CLUSTER_DEEP_DIVE_IDS["mount_signal"]``
            # for full rationale.  Briefly: ``data=True`` fires once
            # per fresh mount of this sub-tab, which is the trigger
            # the bootstrap callback uses to populate the cluster
            # ``Select.data`` option list and resolve URL ``?cid=``
            # deep-links.  ``pathname``-as-Input would race with the
            # subtab content swap; ``top_level_store.data`` only
            # changes on cross-top-level nav and so misses the
            # within-Evaluation Portfolio→Cluster transition.
            dcc.Store(
                id=CLUSTER_DEEP_DIVE_IDS["mount_signal"],
                data=True,
                storage_type="memory",
            ),
        ],
    )


__all__ = ["CLUSTER_DEEP_DIVE_IDS", "build_cluster_deep_dive"]

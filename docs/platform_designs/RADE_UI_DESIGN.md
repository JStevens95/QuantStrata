# Rade UI Design Spec

Short, living spec for the Rade Dash UI. The **mock images in this folder
(`rade_*.png`) are the design contract** — every new tab, card or callback
must visually and functionally align with one of them. When the mocks and
the code disagree, update one of them deliberately, never silently.

---

## 1. Brand

| Token | Value |
|---|---|
| Name | **Rade** |
| Tagline | Quantitative Model Intelligence |
| Logomark | Violet → cyan geometric "R" prism |
| Wordmark | Inter / Geist, weight 600, tracking -0.02em |
| Tone | Quietly confident, low-ornament, numerically dense. Linear / Vercel / Raycast school. |

---

## 2. Palette

Dark-only for v1. All tokens are Tailwind defaults — no bespoke colours.

| Role | Tailwind | Hex |
|---|---|---|
| Background | `slate-950` | `#020617` |
| Surface / card | `slate-900` | `#0f172a` |
| Sunken / track | `slate-950` inside card | `#020617` |
| Border | `slate-800` | `#1e293b` |
| Border subtle | `slate-800/60` | rgba |
| Text primary | `slate-100` | `#f1f5f9` |
| Text secondary | `slate-400` | `#94a3b8` |
| Text muted | `slate-500` | `#64748b` |
| Brand primary | `violet-500` | `#8b5cf6` |
| Brand accent | `cyan-400` | `#22d3ee` |
| Brand gradient | `from-violet-500 to-cyan-400` | — |
| Success | `emerald-500` | `#10b981` |
| Warning | `amber-500` | `#f59e0b` |
| Danger | `rose-500` | `#f43f5e` |
| Info | `sky-400` | `#38bdf8` |

**Rule:** Brand gradient is reserved for primary CTAs, selected nav
indicators, and the splash logomark. Do not apply it to card backgrounds
or chart fills — use 20–30% opacity of the violet for gradient chart fills.

---

## 3. Typography

- **Font**: Inter (fallback: system). Geist is acceptable for brand assets only.
- **Numeric font**: `JetBrains Mono` / `IBM Plex Mono` in tables and diff views
  (Tailwind `font-mono`).
- **Scale**:

| Usage | Size | Weight | Tracking |
|---|---|---|---|
| Page title | 28–36px | 600 | -0.01em |
| Section title | 18–20px | 600 | 0 |
| Card title | 14–15px | 600 | 0 |
| Body | 13–14px | 400 | 0 |
| Muted / caption | 11–12px | 400 | 0.01em |
| Code / numeric | 13px mono | 400 | 0 |

---

## 4. Layout grid

- **Viewport floor**: 1280px. Responsive down to 1024px (sidebar collapses).
- **Sidebar**: 220px wide, `bg-slate-900`, `border-r border-slate-800`.
- **Content padding**: `px-8 py-6` on the main area.
- **Card padding**: `p-5` (default), `p-4` (compact tiles).
- **Gap**: `gap-4` within a card, `gap-6` between cards.
- **Rounded**: `rounded-2xl` on cards, `rounded-xl` on inputs, `rounded-lg` on
  small chips.
- **Shadow**: none by default, `shadow-sm` max. No glow on real content.
  Glow is reserved for: splash logo, selected graph node, ai-assistant
  accent ring.

---

## 5. Page skeleton

Every tab follows the same top-down rhythm — see `rade_landing_dashboard.png`
and `rade_eval_portfolio.png`:

1. **Breadcrumb** (slate-400, 12–13px) — `Section › Sub-section › Tab`.
2. **Title row** — H1 + subtitle + right-aligned meta (version chip, bell,
   search).
3. **Filter bar** (optional) — on a `bg-slate-900 rounded-2xl` strip.
4. **KPI tile row** — 3–6 equal-width cards.
5. **Main content grid** — 1–2–3 column layouts; charts left-weighted, tables
   right-weighted.
6. **Footer strip** — source artifact caption, export actions.

---

## 6. Component library

| Concept | Dash component | Notes |
|---|---|---|
| App shell | `dash-mantine-components.AppShell` | `padding="md"`, custom sidebar. |
| Navigation | `html.A` + Tailwind | Selected = `bg-slate-800` + `border-l-2 border-violet-500`. |
| Breadcrumb | `html.Div` + chevron icons | `dash-iconify` for icons. |
| Segmented control | `dmc.SegmentedControl` | Default style, brand color on active. |
| Dropdown | `dcc.Dropdown` | Theme overridden via `className`. |
| Multi-select | `dmc.MultiSelect` | For cluster/trade pickers. |
| Range slider | `dmc.RangeSlider` | For scenario windows. |
| Button primary | `dmc.Button` with gradient | Variant `gradient`, from violet-500 to cyan-400. |
| Button outline | `dmc.Button variant="outline"` | |
| Pill / chip | `dmc.Badge` | With per-status color. |
| Metric tile | `html.Div` + Tailwind | `rounded-2xl bg-slate-900 border border-slate-800 p-5`. |
| Chart | `dcc.Graph` (Plotly) | Shared dark template — see §7. |
| Table | `dash-ag-grid` | Shared theme config — see §8. |
| Network graph | `dash-cytoscape` | Layout `cose-bilkent`. |
| File upload | `dcc.Upload` | Used in Inference Console. |
| Timeline | `dmc.Timeline` | Governance lineage. |
| Drag canvas | `dash-draggable` | Report Builder only. |
| Command palette | `dmc.Spotlight` | Cmd+K globally. |
| Modal | `dmc.Modal` | Approvals, confirmations. |
| Skeleton | `dmc.Skeleton` | Loading state for every card. |
| Alert | `dmc.Alert` | Error / warning banners. |
| Toast | `dmc.Notification` | Save/export confirmations. |

---

## 7. Chart defaults (Plotly)

One shared template `RADE_DARK` applied via `go.Figure(layout=...)` or a
helper `rade_figure()`:

- `template="plotly_dark"` as base, then override:
- `paper_bgcolor="rgba(0,0,0,0)"`, `plot_bgcolor="rgba(0,0,0,0)"` (transparent; card shows through).
- `font=dict(family="Inter, sans-serif", size=12, color="#94a3b8")`.
- Gridlines: `gridcolor="rgba(148,163,184,0.1)"`, `zerolinecolor` same.
- Axis ticks: `tickcolor="rgba(148,163,184,0.2)"`, `ticks="outside"`, `tickfont.color="#94a3b8"`.
- Margin: `dict(l=40, r=20, t=30, b=40)`.
- Legend: top-right, horizontal, chip-style (`bgcolor="rgba(15,23,42,0.6)"`, `bordercolor="#1e293b"`).
- Hovermode: `"x unified"` for time series, `"closest"` otherwise.
- **Primary series**: `violet-500`. **Secondary**: `cyan-400`. **Reference /
  actual**: dashed `slate-400`. **Residual band**: `rgba(139,92,246,0.18)` fill.
- Colormaps:
  - Residual heatmap: `[[0,"#7c3aed"],[0.5,"#334155"],[1,"#f43f5e"]]`.
  - Correlation heatmap: `"RdBu_r"` centred at 0.
  - Completeness heatmap: `[[0,"#f43f5e"],[0.5,"#334155"],[1,"#7c3aed"]]`.
  - Sequential: Plotly `Viridis` (keep for simple magnitudes).

All chart code lives under `ui/components/charts/` so the template is
enforced centrally.

---

## 8. Table defaults (`dash-ag-grid`)

Single shared `RADE_GRID_DEFAULTS` dict:

```python
RADE_GRID_DEFAULTS = dict(
    className="ag-theme-alpine-dark rade-grid",
    defaultColDef={
        "sortable": True,
        "filter": True,
        "resizable": True,
        "cellClass": "text-slate-200",
    },
    dashGridOptions={
        "rowHeight": 36,
        "headerHeight": 40,
        "suppressMovableColumns": False,
        "animateRows": True,
    },
)
```

- Numeric columns: `type="numericColumn"`, right-aligned, `font-mono`.
- Status pills: `cellRenderer` returning `dmc.Badge`.
- Inline bars: custom `cellRenderer` using a Tailwind-styled `<div>` with
  width proportional to value; bar colour follows status (`bg-rose-500/30`
  for breaches, `bg-emerald-500/30` for healthy).
- Row stripes: `ag-row-odd` = `bg-slate-950/40`.

---

## 9. State reference

Every card must handle the 6 states shown in `rade_states_reference.png`:

1. **Loading** — `dmc.Skeleton` matching the final layout's shape.
2. **Empty (no data)** — outlined icon + heading + body + primary action.
3. **Empty (filter too narrow)** — funnel icon + "Reset filters" link.
4. **Error (404 artifact)** — specific, names the missing file, lists
   available alternatives.
5. **Error (backend down)** — cites the `/health` status, offers Retry.
6. **Authz denied** — names the restricted resource, offers access request.

Do not ship a card without at least the Loading and one Empty state.

---

## 10. Accessibility

- All interactive elements reachable by keyboard; focus ring =
  `ring-2 ring-violet-500 ring-offset-2 ring-offset-slate-950`.
- Colour contrast ≥ 4.5:1 for body text against `slate-900` (checked: all
  palette entries pass).
- Charts ship an accompanying data table under a "View raw data" disclosure
  for screen-reader access.
- Status colour is never the only signal — pair with icon + label.
- Cmd+K command palette exposes every navigable destination.

---

## 11. Design contract — the 20 mocks

| # | File | Purpose | Data source(s) |
|---|---|---|---|
| 1 | `rade_splash.png` | Boot / loading | `/health`, `/versions` |
| 2 | `rade_login.png` | SSO sign-in | — |
| 3 | `rade_landing_dashboard.png` | Home / overview | `/overview`, `/metrics/ensemble`, `/portfolio`, `/clusters` |
| 4 | `rade_eval_portfolio.png` | Evaluation — portfolio split | `/portfolio`, `/metrics/ensemble` |
| 5 | `rade_eval_by_desk.png` | Evaluation grouped by attribute | `/metrics/per-member`, `/clusters` |
| 6 | `rade_cluster_deep_dive.png` | Per-cluster drill-down | `/clusters`, `/metrics/per-member`, `/cluster-timeseries`, `/trades`, `/graph-stats` |
| 7 | `rade_cross_cluster.png` | Correlation matrix + group stats | `/group-correlations`, `/clusters` |
| 8 | `rade_trade_graph.png` | Network graph | `/graph-stats`, `/trades` |
| 9 | `rade_data_quality.png` | Completeness + feature summary | `/quality/completeness`, `/quality/feature-summary` |
| 10 | `rade_model_monitoring.png` | Live drift + alerts | new `/monitoring/*` (future) |
| 11 | `rade_training_curves.png` | Per-cluster training traces | `training_curves.parquet` (served via `/training/curves`) |
| 12 | `rade_prediction_explorer.png` | Raw trade × scenario NPZ drill-in | `/predictions`, `/trades` |
| 13 | `rade_governance.png` | Registry + lineage + approvals | `/versions`, new `/governance/*` |
| 14 | `rade_version_comparison.png` | Side-by-side diff of two versions | compose any `/metrics/*` + `/portfolio` for two versions |
| 15 | `rade_inference.png` | Run ensemble on new trades | new `/inference/run` (future) |
| 16 | `rade_scenario_lab.png` | Synthetic scenario builder + score | new `/scenarios/*` (future) |
| 17 | `rade_report_builder.png` | Block-canvas report authoring | reads anything above, emits PDF / HTML |
| 18 | `rade_ai_assistant.png` | Anomaly investigation side-panel | LLM backend + tool calls into the rest of the API |
| 19 | `rade_command_palette.png` | Global cmd+K navigation | static + `/overview` |
| 20 | `rade_states_reference.png` | Loading / empty / error atlas | — |

---

## 12. Extension process

When a new tab or card is proposed:

1. Draft or regenerate a mock under `docs/platform_designs/rade_<name>.png`.
2. Append a row to §11 table naming its data source and required endpoints.
3. Check charts/tables against §7 / §8 defaults — add to shared helpers if
   a new pattern emerges.
4. Add state coverage (§9) to the implementation PR checklist.

No mock, no merge.

---

## Appendix A — `layouts/evaluation/cluster_deep_dive.py`

Full verbatim body of the Cluster Deep-Dive sub-tab layout. Paste into
`src/ui/apps/rade_analytics/layouts/evaluation/cluster_deep_dive.py`.

Five-row page scoped to a single cluster (no Evaluation filter bar,
split inherits from the topbar):

| Row | Purpose |
|----|---------|
| 1 | Header band · cluster picker · attribute chips · "Trade-Graph" link |
| 2 | 2×2 KPI grid (MAE / RMSE / R² / Coverage) + training-curves chart with overlay chip filter |
| 3 | Residual over time · predicted vs target PnL (shaded error band) |
| 4 | Per-trade residual violin (target / elementary) · per-trade scatter (click to highlight) |
| 5 | Trades AgGrid — per-trade MAE / RMSE / p95 / mean_residual; row click ↔ Row 4 scatter |

```python
"""Evaluation → Cluster Deep-Dive sub-tab layout (Phase E.4).

Five-row layout laser-focused on a single cluster:

    Row 1 · Header band       (cluster picker · attribute chips ·
                               open Trade-Graph link)
    Row 2 · Training          (KPI grid 2×2: MAE/RMSE/R²/Coverage  |
                               training curves with a metric chip
                               filter for optional overlays)
    Row 3 · Time-series       (residual over time  |  predicted vs
                               target PnL with error band)
    Row 4 · Per-trade charts  (residual violin target/elementary  |
                               per-trade scatter target/elementary,
                               with click-to-highlight into Row 5)
    Row 5 · Trades grid       (AgGrid — per-trade MAE / RMSE / p95 /
                               mean_residual; row click ↔ Row 4
                               scatter)

The page deliberately has no filter bar — Cluster Deep-Dive scopes
everything to the single selected cluster, so the top-level Evaluation
filter chrome is redundant here.  Split inherits from the topbar as
everywhere else.

All dynamic ids live in :data:`CLUSTER_DEEP_DIVE_IDS` so callbacks
never hardcode strings.  Callbacks live in
:mod:`..callbacks.cluster_deep_dive_cb`.
"""
from __future__ import annotations

from typing import Any, Dict, List

import dash_mantine_components as dmc
from dash import dcc, html
from dash_iconify import DashIconify

from ...components.ag_grid_table import AgGridTable
from ...components.chart_container import ChartContainer
from ...components.kpi_card import KpiCard


CLUSTER_DEEP_DIVE_IDS: Dict[str, str] = {
    "root":                "eval-cluster-root",

    # Header band
    "cluster_select":      "eval-cluster-cluster-select",
    "attribute_chips":     "eval-cluster-attribute-chips",
    "open_trade_graph_btn": "eval-cluster-open-trade-graph-btn",

    # Row 2 — KPI grid
    "kpi_mae_card":        "eval-cluster-kpi-mae-card",
    "kpi_mae_value":       "eval-cluster-kpi-mae-value",
    "kpi_rmse_card":       "eval-cluster-kpi-rmse-card",
    "kpi_rmse_value":      "eval-cluster-kpi-rmse-value",
    "kpi_r2_card":         "eval-cluster-kpi-r2-card",
    "kpi_r2_value":        "eval-cluster-kpi-r2-value",
    "kpi_coverage_card":   "eval-cluster-kpi-coverage-card",
    "kpi_coverage_value":  "eval-cluster-kpi-coverage-value",

    # Row 2 — training curves
    "curves_chart":        "eval-cluster-curves-chart",
    "curves_chip_group":   "eval-cluster-curves-chip-group",
    "curves_chip_empty":   "eval-cluster-curves-chip-empty",

    # Row 3 — timeseries
    "residual_ts_chart":   "eval-cluster-residual-ts-chart",
    "pnl_band_chart":      "eval-cluster-pnl-band-chart",

    # Row 4 — per-trade
    "per_trade_violin":    "eval-cluster-per-trade-violin",
    "per_trade_scatter":   "eval-cluster-per-trade-scatter",
    "selected_trade_chip": "eval-cluster-selected-trade-chip",
    "selected_trade_label": "eval-cluster-selected-trade-label",
    "selected_trade_clear_btn": "eval-cluster-selected-trade-clear-btn",

    # Row 5 — trades grid
    "trades_grid":         "eval-cluster-trades-grid",
    "trades_grid_card":    "eval-cluster-trades-grid-card",
    "trades_grid_empty":   "eval-cluster-trades-grid-empty",
    "trades_grid_wrap":    "eval-cluster-trades-grid-wrap",

    # Ephemeral stores
    "store_trade_types":   "eval-cluster-trade-types-store",
    "store_curve_metrics": "eval-cluster-curve-metrics-store",
}


# ─────────────────────────────────────────────────────────────────────
# Row 1 — Header band
# ─────────────────────────────────────────────────────────────────────


def _header_band() -> html.Div:
    """Sticky header with cluster picker + attribute chips.

    KPIs used to live here but Phase E.4 (Row 2 revision) moved them
    alongside the training-curves chart so the header stays a thin
    navigation / context band.
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
                placeholder="Select a cluster…",
                searchable=True,
                clearable=False,
                size="sm",
            ),
        ],
    )

    attribute_chips = html.Div(
        id=CLUSTER_DEEP_DIVE_IDS["attribute_chips"],
        className="flex items-center gap-1 flex-wrap min-h-[32px]",
    )

    open_trade_graph_btn = dmc.Button(
        "Trade-Graph",
        id=CLUSTER_DEEP_DIVE_IDS["open_trade_graph_btn"],
        variant="light",
        color="violet",
        size="sm",
        leftSection=DashIconify(icon="tabler:share-2", width=16),
    )

    top_row = html.Div(
        className="flex items-end gap-4 flex-wrap",
        children=[
            cluster_picker,
            html.Div(
                className="flex flex-col gap-1 flex-1 min-w-[240px]",
                children=[
                    html.Span(
                        "Attributes",
                        className="text-[11px] uppercase tracking-wider text-slate-400",
                    ),
                    attribute_chips,
                ],
            ),
            open_trade_graph_btn,
        ],
    )

    return html.Div(
        className="rade-card flex flex-col gap-3 sticky top-0 z-10",
        children=[top_row],
    )


# ─────────────────────────────────────────────────────────────────────
# Row 2 — KPI grid + training curves
# ─────────────────────────────────────────────────────────────────────


def _kpi_grid() -> html.Div:
    """2×2 KPI grid — MAE, RMSE, R², Coverage.

    R² and Coverage are derived client-side from the cluster time-series
    (``rade_analytics.callbacks.cluster_deep_dive_cb``); MAE and RMSE
    come straight from ``per_member_metrics``.
    """
    return html.Div(
        className="grid grid-cols-2 gap-3 self-start",
        children=[
            KpiCard(
                label="MAE",
                value="—",
                card_id=CLUSTER_DEEP_DIVE_IDS["kpi_mae_card"],
                value_id=CLUSTER_DEEP_DIVE_IDS["kpi_mae_value"],
                icon="tabler:arrow-narrow-down",
            ),
            KpiCard(
                label="RMSE",
                value="—",
                card_id=CLUSTER_DEEP_DIVE_IDS["kpi_rmse_card"],
                value_id=CLUSTER_DEEP_DIVE_IDS["kpi_rmse_value"],
                icon="tabler:square-root",
            ),
            KpiCard(
                label="R²",
                value="—",
                card_id=CLUSTER_DEEP_DIVE_IDS["kpi_r2_card"],
                value_id=CLUSTER_DEEP_DIVE_IDS["kpi_r2_value"],
                icon="tabler:chart-dots",
            ),
            KpiCard(
                label="Coverage",
                value="—",
                card_id=CLUSTER_DEEP_DIVE_IDS["kpi_coverage_card"],
                value_id=CLUSTER_DEEP_DIVE_IDS["kpi_coverage_value"],
                icon="tabler:target",
            ),
        ],
    )


def _curves_chip_group() -> html.Div:
    """Multi-select metric chips for the training-curves overlay filter.

    ``train_loss`` is always shown on the chart and so is deliberately
    absent from the chip group — chips only pick extra series to
    overlay (``val_loss``, ``mae``, ``val_mae``, …).  The callback
    populates the chip group based on ``df.attrs["metrics"]``; the
    empty state message renders while the list is still being fetched
    or when the trainer emitted only ``train_loss``.
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
                        id=CLUSTER_DEEP_DIVE_IDS["curves_chip_group"],
                        multiple=True,
                        value=[],
                        children=[],
                    ),
                    html.Span(
                        "No additional metrics emitted for this cluster.",
                        id=CLUSTER_DEEP_DIVE_IDS["curves_chip_empty"],
                        className="text-xs text-slate-500",
                        style={"display": "none"},
                    ),
                ],
            ),
        ],
    )


def _row_training() -> html.Div:
    """Row 2 — KPI grid on the left, training curves chart on the right."""
    return html.Div(
        # 2/5 : 3/5 split on wide screens — gives the KPI grid enough
        # room to breathe without crowding the chart.  Collapses to a
        # single stack on narrow viewports.
        className="grid grid-cols-1 lg:grid-cols-5 gap-3 items-stretch",
        children=[
            html.Div(
                className="lg:col-span-2 flex flex-col gap-3",
                children=[_kpi_grid()],
            ),
            html.Div(
                className="lg:col-span-3 flex flex-col gap-2",
                children=[
                    ChartContainer(
                        title="Training curves",
                        subtitle="Per-epoch train loss (+ selected overlays)",
                        graph_id=CLUSTER_DEEP_DIVE_IDS["curves_chart"],
                        height=300,
                    ),
                    _curves_chip_group(),
                ],
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Row 3 — timeseries
# ─────────────────────────────────────────────────────────────────────


def _row_timeseries() -> html.Div:
    return html.Div(
        className="grid grid-cols-1 lg:grid-cols-2 gap-3",
        children=[
            ChartContainer(
                title="Residual over time",
                subtitle="Rolling absolute error with ±1σ band",
                graph_id=CLUSTER_DEEP_DIVE_IDS["residual_ts_chart"],
                height=300,
            ),
            ChartContainer(
                title="Predicted vs Target PnL",
                subtitle="Shaded band = prediction error",
                graph_id=CLUSTER_DEEP_DIVE_IDS["pnl_band_chart"],
                height=300,
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Row 4 — per-trade charts
# ─────────────────────────────────────────────────────────────────────


def _selected_trade_chip() -> html.Div:
    """Focus-state chip shown in the per-trade scatter card header."""
    return html.Div(
        id=CLUSTER_DEEP_DIVE_IDS["selected_trade_chip"],
        className="rade-focus-chip flex items-center gap-1",
        style={"display": "none"},
        children=[
            DashIconify(
                icon="tabler:target",
                width=12,
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


def _row_per_trade_charts() -> html.Div:
    return html.Div(
        className="grid grid-cols-1 lg:grid-cols-2 gap-3",
        children=[
            ChartContainer(
                title="Per-trade residual distribution",
                subtitle="Split by target / elementary",
                graph_id=CLUSTER_DEEP_DIVE_IDS["per_trade_violin"],
                height=360,
            ),
            ChartContainer(
                title="Per-trade bias vs magnitude",
                subtitle=(
                    "x: mean_residual  ·  y: MAE  ·  click a point "
                    "to highlight in the grid"
                ),
                graph_id=CLUSTER_DEEP_DIVE_IDS["per_trade_scatter"],
                height=360,
                actions=[_selected_trade_chip()],
                config={"doubleClick": "reset"},
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Row 5 — trades grid
# ─────────────────────────────────────────────────────────────────────


def _row_trades_grid() -> html.Div:
    header = html.Div(
        className="flex items-center justify-between",
        children=[
            html.Div(
                className="flex flex-col",
                children=[
                    html.Div(
                        "Trades in this cluster",
                        className="text-sm font-semibold text-slate-200",
                    ),
                    html.Div(
                        "Per-trade metrics across the active split.  "
                        "Click a row to highlight the trade in the scatter above.",
                        className="text-xs text-slate-500",
                    ),
                ],
            ),
        ],
    )

    empty_state = html.Div(
        id=CLUSTER_DEEP_DIVE_IDS["trades_grid_empty"],
        className="rade-list-empty flex flex-col items-center justify-center gap-2 py-8",
        children=[
            DashIconify(icon="tabler:table-off", width=22, className="text-slate-600"),
            html.Div(
                "Pick a cluster to see its trades.",
                className="text-xs text-slate-500 text-center max-w-sm",
            ),
        ],
    )

    grid = AgGridTable(
        grid_id=CLUSTER_DEEP_DIVE_IDS["trades_grid"],
        column_defs=_initial_column_defs(),
        row_data=[],
        height=360,
        className="rade-cluster-trades-grid",
        grid_options={"rowSelection": "single"},
        getRowId="params.data.trade_id",
    )

    grid_wrapper = html.Div(
        id=CLUSTER_DEEP_DIVE_IDS["trades_grid_wrap"],
        className="rade-cluster-trades-grid-wrap",
        style={"display": "none"},
        children=grid,
    )

    return html.Div(
        id=CLUSTER_DEEP_DIVE_IDS["trades_grid_card"],
        className="rade-card flex flex-col gap-3",
        children=[header, empty_state, grid_wrapper],
    )


def _initial_column_defs() -> List[Dict[str, Any]]:
    """Bootstrap columnDefs — the callback rewrites these once data lands."""
    return [
        {"field": "trade_id",      "headerName": "Trade",        "flex": 2, "minWidth": 160},
        {"field": "trade_type",    "headerName": "Type",         "flex": 1, "minWidth": 100},
        {"field": "mae",           "headerName": "MAE",          "flex": 1, "type": "numericColumn"},
        {"field": "rmse",          "headerName": "RMSE",         "flex": 1, "type": "numericColumn"},
        {"field": "p95_ae",        "headerName": "P95 |err|",    "flex": 1, "type": "numericColumn"},
        {"field": "mean_residual", "headerName": "Mean resid.",  "flex": 1, "type": "numericColumn"},
        {"field": "n_scenarios",   "headerName": "Scenarios",    "flex": 1, "type": "numericColumn"},
    ]


# ─────────────────────────────────────────────────────────────────────
# Public builder
# ─────────────────────────────────────────────────────────────────────


def build_cluster_deep_dive() -> html.Div:
    """Assemble the Cluster Deep-Dive sub-tab (pure layout, no callbacks)."""
    return html.Div(
        id=CLUSTER_DEEP_DIVE_IDS["root"],
        className="rade-evaluation-subtab flex flex-col gap-4",
        children=[
            _header_band(),
            _row_training(),
            _row_timeseries(),
            _row_per_trade_charts(),
            _row_trades_grid(),
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
        ],
    )


__all__ = ["CLUSTER_DEEP_DIVE_IDS", "build_cluster_deep_dive"]
```

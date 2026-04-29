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

Full layout module for the **Cluster Deep-Dive sub-tab** (Phase E.5
hybrid).  Lives at `src/ui/apps/rade_analytics/layouts/evaluation/cluster_deep_dive.py`;
mounted by the Evaluation tab's sub-tab router via
`build_cluster_deep_dive(session=...)`.

Key contract points:

* All dynamic ids are exported from `CLUSTER_DEEP_DIVE_IDS` so the
  callback module never hard-codes string literals.
* `mount_signal` is a `dcc.Store(data=True)` mounted at the bottom
  of the page; it is the bootstrap callback's *only* trigger
  (Page Contract §5 — `pathname` races the sub-tab content swap).
* `store_trade_types` carries the `{trade_id: "target" | "elementary"}`
  classification map populated from `/trade-graph` so Row 4's
  Elementary PnL Explorer can filter to `trade_type == "elementary"`
  client-side without a second `/trades` round-trip.
* Row 3 (per-trade detail) ships with `style={"display": "none"}`
  by default — the render callback toggles it visible only when the
  user picks a row in the Trade-Level Metrics grid.
* Row 4's elementary PnL chart card sits behind an empty-state
  placeholder; both are mounted simultaneously and the render
  callback flips their `display` styles in opposite directions.

```python
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
                               grid, click-row → Row 3 expands)

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
rail cards.  We collapse that to three cards by promoting the
training-curves chart to its own row (Row 2 right) where it's actually
readable, and we drop ``Avg Degree`` / ``Avg Path Length`` from the
Graph Statistics card (those numbers would need a new endpoint and
the user explicitly said "show only what we have today").

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
    "elementary_explorer_grid":   "eval-cluster-elementary-explorer-grid",
    "elementary_reset_btn":       "eval-cluster-elementary-reset-btn",
    "elementary_pnl_chart":       "eval-cluster-elementary-pnl-chart",
    "elementary_pnl_chart_card":  "eval-cluster-elementary-pnl-chart-card",
    "elementary_pnl_empty":       "eval-cluster-elementary-pnl-empty",

    # ── Row 2 · Training curves (chart + chip group) ─────────────
    # (Visually under the per-trade row in the mock; kept here so the
    # layout sub-rows stay numbered top-to-bottom.)
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
    """
    body = html.Div(
        id=CLUSTER_DEEP_DIVE_IDS["attributes_body"],
        className="flex flex-col gap-1 flex-1",
        children=[
            _attribute_row(label) for label, _ in _ATTRIBUTE_PLACEHOLDER_KEYS
        ],
    )
    return html.Div(
        id=CLUSTER_DEEP_DIVE_IDS["attributes_card"],
        # ``flex-1`` so this card splits the left-rail height equally
        # with the metrics + graph-stats cards (Row 2 wrapper sits at
        # ``h-[560px]`` and the three cards' flex-1 each grab 1/3).
        className="rade-card-compact flex flex-col gap-2 flex-1",
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
        className="rade-card-compact flex flex-col gap-2 flex-1",
        children=[
            html.Div(
                "Cluster Metrics",
                className="text-sm font-semibold text-slate-200",
            ),
            html.Div(
                # 2×2 KpiCard grid; each KpiCard already has its own
                # padded background, so the outer wrapper just lays
                # them out.
                className="grid grid-cols-2 gap-2 flex-1",
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
        className="flex flex-col gap-1 flex-1",
        children=rows,
    )

    return html.Div(
        id=CLUSTER_DEEP_DIVE_IDS["graph_stats_card"],
        className="rade-card-compact flex flex-col gap-2 flex-1",
        children=[
            html.Div(
                "Graph Statistics",
                className="text-sm font-semibold text-slate-200",
            ),
            body,
        ],
    )


def _row2_left_rail() -> html.Div:
    """Three stacked cards filling the left col of Row 2 evenly."""
    return html.Div(
        # ``lg:col-span-2`` of the 5-col grid; ``flex-col gap-3`` stacks
        # the cards; each card has ``flex-1`` so they share the height
        # equally (Row 2 wrapper sits at fixed ``h-[560px]``).
        className="lg:col-span-2 flex flex-col gap-3",
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

    Both children carry ``flex-1`` so they split the row's 560-px
    height evenly (~270 px each).  The grid uses ``rowSelection: 'single'``
    so a row click fires the per-trade-detail row 3 expand.
    """
    portfolio = ChartContainer(
        title="Cluster Portfolio",
        subtitle="Predicted vs target PnL across scenarios for the active split",
        graph_id=CLUSTER_DEEP_DIVE_IDS["portfolio_chart"],
        height=240,            # leaves padding for the title strip
        className="flex-1",
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
        # ``height=None`` lets ag-grid stretch inside the card body
        # whose own height is dictated by the parent ``flex-1``.
        height=240,
        className="rade-cluster-trades-grid flex-1",
        grid_options={"rowSelection": "single"},
        getRowId="params.data.trade_id",
    )

    return html.Div(
        className="lg:col-span-3 flex flex-col gap-3 min-w-0",
        children=[
            portfolio,
            html.Div(
                # Inner wrapper card so the grid sits in the same
                # ``rade-card`` chrome as the chart above it.
                className="rade-card flex flex-col gap-2 flex-1 min-h-0",
                children=[grid_header, grid],
            ),
        ],
    )


def _row2_main_area() -> html.Div:
    """Row 2 wrapper — left-rail / right-pane grid at fixed 560 px."""
    return html.Div(
        # ``items-stretch`` makes both columns grow to the row height
        # (which we lock to ``h-[560px]``).  Without it the left rail
        # would naturally collapse to its content height.
        className="grid grid-cols-1 lg:grid-cols-5 gap-3 items-stretch h-[560px]",
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
    """Two-up row: residual-over-time on the left, training curves on the right."""
    residual = ChartContainer(
        title="Residual over time",
        subtitle="Rolling absolute error with ±1σ band across the active split",
        graph_id=CLUSTER_DEEP_DIVE_IDS["residual_ts_chart"],
        height=300,
    )

    curves = html.Div(
        className="rade-card flex flex-col gap-2 min-h-0",
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
                style={"height": "260px"},
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
        className="grid grid-cols-1 lg:grid-cols-2 gap-3 items-stretch h-[340px]",
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
        # Grid + height locked the same way as the other rows so the
        # row's footprint is predictable when it does appear.
        className="grid grid-cols-1 lg:grid-cols-2 gap-3 items-stretch h-[360px]",
        children=[histogram, bias_scatter],
    )


# ─────────────────────────────────────────────────────────────────────
# Row 4 — Elementary PnL Explorer + multi-line timeseries
# ─────────────────────────────────────────────────────────────────────


def _elementary_explorer_column_defs() -> List[Dict[str, Any]]:
    """Column defs for the elementary-trade explorer grid.

    Subset of the Trade-Level Metrics columns — same data source
    (/trades) but filtered client-side to ``trade_type=='elementary'``
    in the render callback (we already have the trade-type map in
    ``store_trade_types``).
    """
    return [
        # Checkbox column — multi-select drives the timeseries chart.
        {
            "field": "trade_id",
            "headerName": "Trade",
            "flex": 2,
            "minWidth": 160,
            "checkboxSelection": True,
            "headerCheckboxSelection": False,
        },
        {"field": "mae",           "headerName": "MAE",         "flex": 1, "type": "numericColumn"},
        {"field": "rmse",          "headerName": "RMSE",        "flex": 1, "type": "numericColumn"},
        {"field": "mean_residual", "headerName": "Mean resid.", "flex": 1, "type": "numericColumn"},
        {"field": "n_scenarios",   "headerName": "Scenarios",   "flex": 1, "type": "numericColumn"},
    ]


def _row_elementary_pnl() -> html.Div:
    """Row 4 — Elementary PnL Explorer (left) + multi-line chart (right).

    The chart card lives behind an empty-state placeholder until the
    user selects ≥1 elementary trade in the explorer grid; then the
    chart fills with one line per selected trade, x = scenario_idx,
    y = raw PnL value (no predictions / targets — elementary trades
    are model *inputs*).
    """
    explorer_header = html.Div(
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
                        "Tick rows to plot their per-scenario PnL on the right.",
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

    explorer_grid = AgGridTable(
        grid_id=CLUSTER_DEEP_DIVE_IDS["elementary_explorer_grid"],
        column_defs=_elementary_explorer_column_defs(),
        row_data=[],
        height=300,
        className="rade-cluster-elementary-grid flex-1",
        grid_options={
            "rowSelection": "multiple",
            # Without ``suppressRowClickSelection``, clicking a non-
            # checkbox cell also toggles selection, which conflicts
            # with the user's expectation that the checkbox column is
            # the explicit selector.  Off keeps cell clicks neutral.
            "suppressRowClickSelection": True,
        },
        getRowId="params.data.trade_id",
    )

    explorer_card = html.Div(
        className="rade-card flex flex-col gap-2 min-h-0",
        children=[explorer_header, explorer_grid],
    )

    chart_empty_state = html.Div(
        id=CLUSTER_DEEP_DIVE_IDS["elementary_pnl_empty"],
        className=(
            "rade-list-empty flex flex-col items-center justify-center "
            "gap-2 py-8 flex-1"
        ),
        children=[
            DashIconify(
                icon="tabler:chart-line-off", width=24,
                className="text-slate-600",
            ),
            html.Div(
                "Pick one or more elementary trades from the explorer "
                "table to plot their per-scenario PnL.",
                className="text-xs text-slate-500 text-center max-w-sm",
            ),
        ],
    )

    # Chart card is mounted but its outer wrapper carries
    # ``display: none`` until the user selects ≥1 elementary trade.
    # The render callback flips the empty state and the chart card
    # in opposite directions (one shows, one hides) so we never have
    # both visible simultaneously.  ``container_id`` lets the
    # callback toggle the *inner* card's display style; we don't
    # nest a redundant wrapper around it.
    chart = ChartContainer(
        title="Elementary PnL timeseries",
        subtitle="Raw PnL per scenario for the selected elementary trades",
        graph_id=CLUSTER_DEEP_DIVE_IDS["elementary_pnl_chart"],
        height=300,
        container_id=CLUSTER_DEEP_DIVE_IDS["elementary_pnl_chart_card"],
        className="flex-1",
    )

    chart_panel = html.Div(
        className="flex flex-col flex-1 min-h-0",
        children=[chart_empty_state, chart],
    )

    return html.Div(
        className="grid grid-cols-1 lg:grid-cols-2 gap-3 items-stretch h-[400px]",
        children=[explorer_card, chart_panel],
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
        className="rade-evaluation-subtab flex flex-col gap-4",
        children=[
            _header_band(initial_cluster_id=initial_cluster_id),
            _row2_main_area(),
            _row_residual_and_curves(
                initial_curve_metrics=eval_state.deep_dive_curve_metrics,
            ),
            _row_per_trade_detail(),
            _row_elementary_pnl(),

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
```

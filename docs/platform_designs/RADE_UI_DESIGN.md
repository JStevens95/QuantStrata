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

## Appendix A — Monitoring page (Option A: empty layout, no callbacks)

Single appendix (replaces all prior appendix blocks).  Wires the
`/monitoring` route to a fully-structured layout that mirrors
`rade_model_monitoring.png` region-for-region, but with **no
callbacks and no API calls** — KPI values are em-dashes, charts
mount with themed empty-state placeholders, the alerts grid is
empty.  The page is honest about its state via subtitle copy
("Drift & residual surveillance — preview (no live telemetry
connected)") and a footer producer-disclosure caption.

This shape gives reviewers a navigable preview of the proposed
design today, with zero risk of `ReadTimeout` / API failures, and
collapses to a one-line restore once the V1 snapshot-mode callbacks
ship — the layout's `MONITORING_IDS` contract is already in place
so the eventual `monitoring_cb` module can target each region via
`Output` without a layout refactor.

### How to apply

Three targets in this appendix.  Two are **new files** — copy the
contents into the path called out in the section header.  One is a
**patch to an existing file** — apply the targeted find/replace
shown in §A.3.

| § | Path | Action |
|---|---|---|
| A.1 | `src/ui/apps/rade_analytics/figures/monitoring_charts.py` | NEW |
| A.2 | `src/ui/apps/rade_analytics/layouts/monitoring.py`         | NEW |
| A.3 | `src/ui/apps/rade_analytics/router.py`                     | PATCH |

Files **deliberately untouched**:

* `callbacks/__init__.py` — no `monitoring_cb` exists, nothing to
  register.  When V1 snapshot-mode lands, add a single
  `monitoring_cb.register(app, backend)` call here.
* `assets/rade.css` — the existing `.rade-page-title`,
  `.rade-grid-mono`, `.rade-pill--*` classes (added for the
  Governance page) cover everything the Monitoring V1 layout uses.
  Severity-specific pills (`--critical` / `--warn` / `--info`) and
  KPI-badge variants are deferred to the callback-on phase.
* Backend stack (`services/`, `routers/`, `client.py`,
  `backend.py`, Pydantic models) — Option A intentionally ships
  zero backend.  When Option B (V1 snapshot mode) lands, a single
  `services/monitoring.py` derives every populated value from
  existing eval artefacts.

See §A.4 for the verification checklist.

---

### A.1 — `src/ui/apps/rade_analytics/figures/monitoring_charts.py` (NEW)

```python
"""Plotly figure builders for the Monitoring page.

V1 ships **empty-state placeholders only** — every helper here returns
the standard themed empty figure (see :func:`figures._theme.empty_figure`)
with a chart-specific *awaiting-data* annotation.  Each placeholder
hints at what the production version of that chart will show, so the
Option-A preview reads as the proposed design rather than three
identical "no data" boxes.

When the Monitoring callbacks come online (Option B / V1 snapshot
mode), populated builders for each chart will land *next to* these
helpers in this same module, mirroring the layout used by
``cluster_deep_dive_charts.py`` (paired ``empty_*`` and ``populated_*``
factories).

Design anchor
-------------
``docs/platform_designs/rade_model_monitoring.png`` — Row 2 (residual
drift line + feature PSI bar) and Row 3 (latency histogram).
"""
from __future__ import annotations

import plotly.graph_objects as go

from ._theme import empty_figure


# ─────────────────────────────────────────────────────────────────────
# Empty-state placeholders
# ─────────────────────────────────────────────────────────────────────


def empty_residual_drift() -> go.Figure:
    """Placeholder for the *Residual Drift — last 30 days* line chart.

    Production version (Option B): per-scenario residual median + a
    P5–P95 envelope band derived from
    ``cluster_residuals_test.parquet``, with anomaly markers stamped
    on outlier scenarios.
    """
    return empty_figure("Awaiting eval-driven residual stats")


def empty_feature_drift_psi() -> go.Figure:
    """Placeholder for the *Feature Drift (PSI)* horizontal bar chart.

    Production version (Option B): top-10 features by PSI severity
    (info / warn / critical), computed from
    ``baseline_feature_stats.parquet`` (train baseline) vs recomputed
    test-side feature distributions.
    """
    return empty_figure("Awaiting feature drift PSI computation")


def empty_latency_histogram() -> go.Figure:
    """Placeholder for the *Latency Histogram* in Row 3.

    Production version (Stage 2): histogram of inference latencies
    pulled from the production logging pipeline, with annotated P50 /
    P95 / P99 lines.  Until that producer ships there is no real
    source for this chart — the V1 snapshot mode will keep this card
    on a synthesised gamma-shaped distribution.
    """
    return empty_figure("Awaiting inference latency telemetry")


__all__ = [
    "empty_residual_drift",
    "empty_feature_drift_psi",
    "empty_latency_histogram",
]
```

**Why a separate figures module rather than inlining `empty_figure(...)`
calls in the layout?**  When Option B / V1 snapshot mode lands the
populated chart builders will need ~50–100 lines each (envelope
shading, anomaly markers, severity-coloured PSI bars, percentile
lines).  Putting the empty-state shells here today means the populated
versions land next to their `empty_*` counterparts — same import path
the layout already references, no second-order layout edit needed.

---

### A.2 — `src/ui/apps/rade_analytics/layouts/monitoring.py` (NEW)

```python
"""Monitoring page layout.

Mirrors ``docs/platform_designs/rade_model_monitoring.png`` region-for-
region:

* **Row 0** — invisible mount tripwire (Page Contract §3 Rule L4) so
  the Stage-2 render callback can fire on layout mount without
  racing the DOM swap.
* **Row 1** — Header band: page title + subtitle on the left, four KPI
  chips spanning the row (Baseline-vs-Live KS, PSI, Rolling MAE 7d,
  Alerts Open).
* **Row 2** — Two-column row: *Residual Drift — last 30 days* line
  chart on the left (~2/3 width) and *Feature Drift (PSI)* horizontal
  bar chart on the right (~1/3 width).
* **Row 3** — Two-column row: *Active Alerts* AG Grid (~1/2 width) and
  *Latency Histogram* (~1/2 width).
* **Row 4** — Footer caption: data-source disclosure.

V1 status — Option A (empty layout, no callbacks)
-------------------------------------------------
This module ships the full layout with **empty defaults everywhere**:

* KPI values default to em-dash placeholders.
* Charts mount with the matching ``empty_*`` figure from
  :mod:`figures.monitoring_charts` (each carries an "Awaiting …"
  annotation hinting at what the populated chart will show).
* The Active Alerts grid mounts with ``rowData=[]``.

No callbacks are registered today (see ``callbacks/__init__.py``) so
no API request fires when the page is navigated to.  When the V1
snapshot-mode callbacks land they will overwrite each region via
``Output`` — no layout change required.

Page Contract anchors
---------------------
* §3 Rule L1 — every primitive a callback might target has a stable
  id via :data:`MONITORING_IDS`.
* §3 Rule L3 — no hardcoded id strings outside this dict.
* §3 Rule L4 — :data:`MONITORING_IDS["mount_signal"]` is a Store with
  ``data=True`` so render callbacks can fire after the DOM swap.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from dash import dcc, html

from ..components.ag_grid_table import AgGridTable
from ..components.chart_container import ChartContainer
from ..components.kpi_card import KpiCard
from ..figures.monitoring_charts import (
    empty_feature_drift_psi,
    empty_latency_histogram,
    empty_residual_drift,
)

if TYPE_CHECKING:
    from ..data.session import Session


# ─────────────────────────────────────────────────────────────────────
# Stable id contract — every component a callback might target lives
# here so callbacks never hardcode strings (Page Contract §3 Rule L3).
# ─────────────────────────────────────────────────────────────────────


MONITORING_IDS: Dict[str, str] = {
    "root":                     "monitoring-root",

    # Mount tripwire — Page Contract §3 Rule L4.
    "mount_signal":             "monitoring-mount-signal",

    # Row 1 — Header subtitle (carries the "updated N min ago" suffix
    # once the snapshot callback comes online).
    "subtitle":                 "monitoring-subtitle",

    # Row 1 — KPI chips.  Each card carries a separate value id so
    # callbacks can target the value text without rebuilding chrome.
    "kpi_ks":                   "monitoring-kpi-ks",
    "kpi_ks_value":             "monitoring-kpi-ks-value",
    "kpi_psi":                  "monitoring-kpi-psi",
    "kpi_psi_value":            "monitoring-kpi-psi-value",
    "kpi_rolling_mae":          "monitoring-kpi-rolling-mae",
    "kpi_rolling_mae_value":    "monitoring-kpi-rolling-mae-value",
    "kpi_alerts_open":          "monitoring-kpi-alerts-open",
    "kpi_alerts_open_value":    "monitoring-kpi-alerts-open-value",

    # Row 2 — Drift charts.
    "residual_drift_chart":     "monitoring-residual-drift-chart",
    "feature_drift_chart":      "monitoring-feature-drift-chart",

    # Row 3 — Alerts grid + Latency histogram.
    "alerts_grid":              "monitoring-alerts-grid",
    "latency_histogram_chart":  "monitoring-latency-histogram-chart",
}


# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────


# Placeholder used for any KPI value that doesn't have a callback
# overwriting it yet.  Matches the convention used on every other
# Rade page (Overview / Governance / Cluster Deep-Dive).
_PLACEHOLDER = "—"


# Subtitle wording for V1 — honest about the empty state.  The
# snapshot-mode callback (Option B) replaces this with
# ``"Train vs Test drift snapshot — updated <N> min ago"``.
_V1_SUBTITLE = (
    "Drift & residual surveillance — preview "
    "(no live telemetry connected)"
)


# Footer caption — surface the producer-side reality so reviewers
# don't infer anything about live monitoring from a static layout.
_FOOTER_CAPTION = (
    "Monitoring data sourced from production logging pipeline; "
    "live feed wires up in Stage 2."
)


# ─────────────────────────────────────────────────────────────────────
# Row 1 — Header band (title + subtitle + KPI strip)
# ─────────────────────────────────────────────────────────────────────


def _kpi_strip() -> html.Div:
    """Four KPI cards across Row 1.

    Values are em-dashes at build time and overwritten by the
    snapshot render callback once it lands.  Card icons stay static.
    """
    return html.Div(
        className="grid grid-cols-4 gap-4",
        children=[
            KpiCard(
                label="Baseline vs Live KS",
                value=_PLACEHOLDER,
                card_id=MONITORING_IDS["kpi_ks"],
                value_id=MONITORING_IDS["kpi_ks_value"],
                icon="tabler:wave-square",
            ),
            KpiCard(
                label="PSI (Population Stability)",
                value=_PLACEHOLDER,
                card_id=MONITORING_IDS["kpi_psi"],
                value_id=MONITORING_IDS["kpi_psi_value"],
                icon="tabler:chart-histogram",
            ),
            KpiCard(
                label="Rolling MAE (7d)",
                value=_PLACEHOLDER,
                card_id=MONITORING_IDS["kpi_rolling_mae"],
                value_id=MONITORING_IDS["kpi_rolling_mae_value"],
                icon="tabler:trending-down",
            ),
            KpiCard(
                label="Alerts Open",
                value=_PLACEHOLDER,
                card_id=MONITORING_IDS["kpi_alerts_open"],
                value_id=MONITORING_IDS["kpi_alerts_open_value"],
                icon="tabler:bell-ringing",
            ),
        ],
    )


def _row_header() -> html.Div:
    """Row 1 — page title + subtitle on the left, KPI strip below.

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
                                "Production Monitoring",
                                className="rade-page-title",
                            ),
                            html.Div(
                                _V1_SUBTITLE,
                                id=MONITORING_IDS["subtitle"],
                                className="text-xs text-slate-500",
                            ),
                        ],
                    ),
                ],
            ),
            _kpi_strip(),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Row 2 — Drift charts (residual line + feature PSI bar)
# ─────────────────────────────────────────────────────────────────────


def _row_drift_charts() -> html.Div:
    """Row 2 — residual drift line (~2/3) + feature drift PSI bar (~1/3).

    Layout uses a 3-column CSS grid: residual drift spans 2 cols,
    feature drift takes the remaining 1.  Mirrors the
    Overview / Cluster-Deep-Dive split conventions so visual rhythm
    stays consistent across pages.
    """
    return html.Div(
        className="grid grid-cols-3 gap-4 items-stretch",
        children=[
            html.Div(
                className="col-span-2",
                children=[
                    ChartContainer(
                        title="Residual Drift — last 30 days",
                        subtitle=(
                            "Median residual + P5–P95 envelope · "
                            "anomaly markers"
                        ),
                        graph_id=MONITORING_IDS["residual_drift_chart"],
                        figure=empty_residual_drift(),
                        height=320,
                    ),
                ],
            ),
            html.Div(
                className="col-span-1",
                children=[
                    ChartContainer(
                        title="Feature Drift (PSI)",
                        subtitle="Top 10 features · severity-coloured",
                        graph_id=MONITORING_IDS["feature_drift_chart"],
                        figure=empty_feature_drift_psi(),
                        height=320,
                    ),
                ],
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Row 3 — Active alerts + Latency histogram
# ─────────────────────────────────────────────────────────────────────


# Column defs for the Active Alerts grid.  Severity column is wired
# to ``cellClassRules`` against the existing ``rade-pill`` palette
# (rose / amber / slate via reused ``rejected`` / ``pending`` /
# ``archived`` variants) so when the alerts callback comes online
# styling lights up automatically — no second CSS pass.
_ALERTS_COLUMN_DEFS: List[Dict[str, Any]] = [
    {
        "field":      "alert_id",
        "headerName": "Alert ID",
        "minWidth":   140,
        "pinned":     "left",
        "cellClass":  "rade-grid-mono",
    },
    {
        "field":      "cluster",
        "headerName": "Cluster",
        "minWidth":   110,
    },
    {
        "field":      "type",
        "headerName": "Type",
        "minWidth":   120,
    },
    {
        "field":      "severity",
        "headerName": "Severity",
        "minWidth":   120,
        "cellClassRules": {
            # Until severity-specific pills land, reuse the existing
            # governance palette: critical → rose, warn → amber,
            # info → slate.  Colour intent stays correct; the class
            # names are reused not duplicated.
            "rade-pill rade-pill--rejected": (
                "params.value === 'critical' || params.value === 'rose'"
            ),
            "rade-pill rade-pill--pending": (
                "params.value === 'warn' || params.value === 'amber'"
            ),
            "rade-pill rade-pill--archived": (
                "params.value === 'info' || params.value === 'slate'"
            ),
        },
        "valueFormatter": {
            "function": (
                "params.value ? "
                "params.value.charAt(0).toUpperCase() + params.value.slice(1)"
                " : '—'"
            ),
        },
    },
    {
        "field":      "opened",
        "headerName": "Opened",
        "minWidth":   140,
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
        "field":      "owner",
        "headerName": "Owner",
        "minWidth":   140,
    },
]


def _row_alerts_and_latency() -> html.Div:
    """Row 3 — active alerts grid (~1/2) + latency histogram (~1/2).

    Both children share equal width so the page reads as two paired
    cards, matching the design.  When the snapshot callback lands
    the latency histogram becomes a synthesised gamma distribution
    until inference telemetry exists; alerts stay empty until the
    Stage-2 alerts producer ships.
    """
    return html.Div(
        className="grid grid-cols-2 gap-4 items-stretch",
        children=[
            html.Div(
                className="rade-card flex flex-col gap-3",
                children=[
                    html.Div(
                        "Active Alerts",
                        className="text-sm font-semibold text-slate-200",
                    ),
                    AgGridTable(
                        grid_id=MONITORING_IDS["alerts_grid"],
                        row_data=[],
                        column_defs=_ALERTS_COLUMN_DEFS,
                        grid_options={
                            "pagination": False,
                            "rowHeight": 36,
                            "headerHeight": 38,
                            "animateRows": False,
                            "suppressCellFocus": True,
                            "domLayout": "autoHeight",
                        },
                        height=240,
                        className="rade-monitoring-alerts-grid",
                    ),
                ],
            ),
            ChartContainer(
                title="Latency Histogram",
                subtitle="Inference latency · P50 / P95 / P99",
                graph_id=MONITORING_IDS["latency_histogram_chart"],
                figure=empty_latency_histogram(),
                height=260,
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Row 4 — Footer caption
# ─────────────────────────────────────────────────────────────────────


def _row_footer() -> html.Div:
    """Producer-disclosure caption.

    Rendered as plain text right-aligned underneath Row 3, matching
    the design's tiny grey footer line.  Stays static — it's not
    user-data; the wording becomes "monitoring-data-source-truthful"
    once Stage 2 wires up production logging.
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


def build_monitoring(*, session: Optional["Session"] = None) -> html.Div:
    """Build the full Monitoring page tree.

    The ``session`` kwarg is accepted for uniformity with every other
    page builder (Page Contract §2.1) but unused today — the page
    has no per-user persisted state.  Reserved so adding e.g. a
    ``monitoring_split_filter`` field to ``Session`` later is a
    one-line layout change.
    """
    del session  # unused today; reserved for forward-compat

    return html.Div(
        id=MONITORING_IDS["root"],
        className="rade-page",
        children=[
            # Mount tripwire — Page Contract §3 Rule L4.
            dcc.Store(
                id=MONITORING_IDS["mount_signal"],
                data=True,
                storage_type="memory",
            ),
            _row_header(),
            _row_drift_charts(),
            _row_alerts_and_latency(),
            _row_footer(),
        ],
    )


__all__ = [
    "MONITORING_IDS",
    "build_monitoring",
]
```

**Implementation notes for reviewers**

* `MONITORING_IDS` is the only public id surface.  Adding a new
  component to a callback later → add the id here first, then in
  the layout helper, then in the callback's `Input/Output`.  Same
  workflow as Governance.
* The Active Alerts grid uses **`domLayout: "autoHeight"`** so an
  empty grid still renders the header row + the AG Grid "No Rows
  to Show" overlay at a sensible height.  The container's
  `height=240` is the *card* height — AG Grid sizes itself to fit.
* `_ALERTS_COLUMN_DEFS` reuses the Governance pill palette
  (`rade-pill--rejected` / `--pending` / `--archived`) for severity
  colours.  When monitoring-specific severity pills land in
  `rade.css` (`--critical` / `--warn` / `--info`), update the four
  class names in `cellClassRules` and nothing else.
* The subtitle has its **own id** (`subtitle`) — when V1 snapshot
  mode lands, the render callback emits a new value into
  `Output(MONITORING_IDS["subtitle"], "children")` carrying
  "Train vs Test drift snapshot — updated 2 min ago"-style copy.

---

### A.3 — `src/ui/apps/rade_analytics/router.py` (PATCH)

Two targeted patches.  The router currently imports every page
builder at the top of the module and references `_placeholder("Monitoring", "Phase F")`
in the `/monitoring` `PageSpec`; we add the import and swap the
builder.

**Patch 1 — add `build_monitoring` to the imports**

```diff
 from .layouts.evaluation import build_evaluation
 from .layouts.governance import build_governance
+from .layouts.monitoring import build_monitoring
 from .layouts.overview import build_overview
 from .layouts.shell import SHELL_IDS, build_chrome
 from .layouts.splash import build_splash
```

**Patch 2 — wire the route**

```diff
     # ── Remaining top-level pages (stubs until Phase F) ─────────
     "/monitoring": PageSpec(
         path="/monitoring",
         title="Monitoring",
-        build=_placeholder("Monitoring", "Phase F"),
+        build=build_monitoring,
     ),
```

No other lines in `router.py` change.  The breadcrumb title stays
`"Monitoring"`, the sidebar nav item already exists, and there is
no sub-route registration to add (the page is single-route, unlike
Evaluation which fans out to `/evaluation/portfolio`,
`/evaluation/cluster`, etc.).

---

### A.4 — Verification (after pasting all three targets)

Run from the repo root.  No live API needed — Option A makes zero
backend calls.

1. **Static checks** — imports + lints:

   ```bash
   python -c "
   from src.ui.apps.rade_analytics.layouts.monitoring import (
       build_monitoring, MONITORING_IDS,
   )
   from src.ui.apps.rade_analytics.figures.monitoring_charts import (
       empty_residual_drift,
       empty_feature_drift_psi,
       empty_latency_histogram,
   )
   from src.ui.apps.rade_analytics.router import ROUTES

   tree = build_monitoring()
   assert tree.id == 'monitoring-root'
   assert len(tree.children) == 5, 'Store + 4 rows expected'
   assert ROUTES['/monitoring'].build is build_monitoring
   assert empty_residual_drift().layout.template is not None
   print('monitoring imports + layout build + route wiring: OK')
   "
   ```

2. **Live UI** — boot the existing Dash runner and click
   *Monitoring* in the sidebar.  Expected:

   * Page title `"Production Monitoring"` and the V1 honest
     subtitle render.
   * Four KPI cards each show `—` for value, with the icons
     visible.
   * Residual Drift card renders the empty figure with the
     "Awaiting eval-driven residual stats" annotation centred.
   * Feature Drift card renders the empty figure with
     "Awaiting feature drift PSI computation".
   * Active Alerts card renders the column header row and an
     "No Rows To Show" overlay (AG Grid default for `rowData=[]`).
   * Latency Histogram card renders the empty figure with
     "Awaiting inference latency telemetry".
   * Footer caption right-aligned under Row 3.
   * **No HTTP request to `/prism/v1/monitoring/*` ever fires** —
     verify in the browser DevTools network tab.
   * Browser console has no "missing callback" warnings.

3. **Re-enable callbacks later (one-line restore)**.  When V1
   snapshot mode (Option B) is ready, two adjustments bring the
   page live:

   ```python
   # callbacks/__init__.py
   from . import monitoring_cb           # ← add
   monitoring_cb.register(app, backend)  # ← add inside register_all
   ```

   At that point the layout doesn't change at all — the same
   `MONITORING_IDS` are already there, and every empty-state
   placeholder gets overwritten via `Output`.

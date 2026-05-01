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

## Appendix A — Inference Console (scenario ingest · Option A: empty layout, no callbacks)

Single appendix entry (this block replaces whatever lived in Appendix A before).
Implementation pairs with [`docs/rade_analytics/page_contract.md`](../rade_analytics/page_contract.md)
and the mechanical workflow in [`docs/rade_analytics/page_template/README.md`](../rade_analytics/page_template/README.md).

**Design anchor:** scenario-first variant of [`rade_inference.png`](rade_inference.png) — global **version +
split live in chrome** (`TOPBAR_IDS["version_select"]`, `"split_toggle"`).  This option ships **navigation +
layout shells only**: no callbacks in `callbacks/__init__.py`, therefore no REST calls yet.

---

### Appendix A · 1 — Contract + template checklist (pre-flight before callbacks)

Pass criteria for reviewers — should match Cluster Deep‑Dive / Data Quality parity.

| Item | Requirement (page contract / template) | This page |
|------|---------------------------------------|-----------|
| §2·1 Pure layout | `build_inference(*, session=…)` is side-effect-free; zero backend imports under `layouts/`. | **Pass** · only `components/`, `figures/`, `dcc`/`dmc`/`html`. |
| §2·1 `Session` signature | Builders accept optional `session` typed as `Optional[Session]` + `del session`/`use` honestly. | **Pass** · `del session` with note until capture persists state. |
| §3 Rule L2 | **`<PAGE>_IDS`** dict owns every callable id (`INFERENCE_IDS`). | **Pass** (`inference-*` prefixes). |
| §3 Rule L3 | Prefer shared styles in `rade.css`; primitives use Tailwind like sibling pages until a page-specific block is warranted. | **Pass** reuse `rade-card`, `rade-page-title`, pills. |
| §3 Rule L4 | **`dcc.Store(id=mount_signal, data=True, storage_type="memory")`** sits under page root early in children. | **Pass** (`inference-mount-signal`). |
| Chrome scope | Ensemble **version** + **split** come from shell top bar (`TOPBAR_IDS`); do not duplicate on the page. | **Pass** · caption references top bar. |
| Template layout | Mirrors template row order philosophy: tripwire → header → strips → grids; future `callbacks/inference_cb.py` pairs with `layouts/inference.py` per README step 4. | **Pass** Phase A layout only (`inference_cb` not merged yet). |
| §11 mock coverage | Honour palette / card cadence §4–§6 (gradient CTA sparingly — Run button only). | **Pass** violet→cyan mirrors splash CTA conventions. |

**Known platform limitation (`dcc.Upload` 3.4):** Dash does **not** expose `directory=True`. **Multi-file** browse +
**paste server path** is the authoritative story until a packaged `clientside_callback` attaches `webkitdirectory`.

---

### Appendix A · 2 — File touch summary

| Action | Path |
|--------|------|
| **NEW** | `src/ui/apps/rade_analytics/figures/inference_charts.py` |
| **NEW** | `src/ui/apps/rade_analytics/layouts/inference.py` |
| **PATCH** | `src/ui/apps/rade_analytics/router.py` — alphabetised imports + `/inference` route + breadcrumb title |
| Deferred | `src/ui/apps/rade_analytics/callbacks/inference_cb.py` (+ `callbacks/__init__.py` registration) |

No `data/backend.py` changes for Option A. CSS: optional scoped hooks under `.rade-inference` if overrides become necessary — reuse existing `rade-card`, `rade-page-title`, pill classes today.

---

### Appendix A · 3 — Stage 2 callback playbook (outline only)

Implementers bootstrap off `docs/rade_analytics/page_template/template_cb.py`:

1. **Capture** paths: paste field + last `dcc.Upload` `contents`; write structured ingest payload to `dcc.Store` or session extension.
2. **Upload scenarios** button: `Output(upload_scenarios_btn, "loading")` + `Output(ingest_status, "children")` success `DashIconify(tabler:circle-check, className="text-emerald-500")`.
3. **Render** listens on `Input(mount_signal, "data")` + `State(SHELL_IDS["url"], "pathname")` gate `== "/inference"` — never `Input(pathname)` on page render (Anti-pattern A8).
4. **Chart mode** `chart_view_mode` → swap `empty_pnl_*` vs live figures on `chart_main`.
5. **AgGrid** `rowSelection="single"` → future `selectedRows` callback filters chart series.

---

### Appendix A · 4 — `figures/inference_charts.py` (NEW · full file)

```python
"""Plotly figure builders for the Inference Console page.

V1 ships **empty-state placeholders only**.  When inference callbacks
wire up, populate builders pair with these helpers in this module —
same convention as ``data_quality_charts.py`` /
``monitoring_charts.py``.

Modes
-----
* **Distribution** — histogram of aggregated P&L (or PV change)
  across scenarios for the priced book.
* **Timeseries** — scenario index or chronological ordering on X,
  aggregated curve with optional percentile band.
* **Overlay** — reserved third view (e.g. train vs live scenario
  kernel density) until the product contract names it.
"""
from __future__ import annotations

import plotly.graph_objects as go

from ._theme import empty_figure


def empty_pnl_distribution() -> go.Figure:
    """Placeholder for *Predicted PnL distribution across scenarios*."""
    return empty_figure("Awaiting scenario run — distribution view")


def empty_pnl_timeseries() -> go.Figure:
    """Placeholder for *PnL / exposure vs scenario order or time*."""
    return empty_figure("Awaiting scenario run — timeseries view")


def empty_pnl_overlay() -> go.Figure:
    """Reserved third chart mode (e.g. density overlay, fan chart)."""
    return empty_figure("Third chart mode — reserved")


__all__ = [
    "empty_pnl_distribution",
    "empty_pnl_overlay",
    "empty_pnl_timeseries",
]

```

---

### Appendix A · 5 — `layouts/inference.py` (NEW · full file)

```python
"""Inference Console page layout — **scenario ingestion** path (V1).

This iteration targets **uploading scenario definitions** from a filesystem
folder (paste path vs browser folder pick), not ad-hoc new trades.

Row map
-------

* **Row 0** — ``dcc.Store`` mount tripwire (Page Contract §3 Rule L4).

* **Row 1** — Title + subtitle.  Ensemble **version** and **train /
  val / test** split remain in the chrome topbar (:data:`TOPBAR_IDS`)
  — not duplicated here — so this page stays aligned with the rest of
  Rade.

* **Row 2** — Scenario folder sourcing bar:

    * ``dmc.TextInput`` — paste a server-accessible folder path.

    * ``dcc.Upload`` — ``multiple=True``, wraps a *Browse files* button so
      analysts can pick many scenario files at once.  True **folder**
      selection (``webkitdirectory``) is not exposed by ``dcc.Upload`` in
      Dash 3.4 — use the pasted path for recursive directory ingest on the
      server, or extend with a small clientside bundle later.

    * *Upload scenarios* ``dmc.Button`` — exposes ``loading=`` /
      ``loaderProps`` at ``False``; the hydrate callback toggles
      ``loading``, then swaps the ingest status slot for an emerald tick
      when ingest completes.

    * Ingest feedback slot ``#inference-ingest-status`` — empty shell
      the callback swaps to ``tabler:circle-check``, error icon, etc.

* **Row 3** — Two-column split:

    * **Left · INPUT** — scrollable scenario manifest preview
      placeholder; **Run** (gradient violet→cyan — matches splash CTAs)
      and **Validate only** footer actions.

    * **Right · RESULTS** — four KPI tiles (sparkline slots via
      ``sparkline_id``); ``dmc.Tabs`` (**Charts · Sensitivity ·
      Diagnostics**).  The **Charts** tab holds the segmented control
      (**Distribution** / **Timeseries** / **More · TBD**) and the main
      ``dcc.Graph`` (:class:`~components.chart_container.ChartContainer`).
      Then *Save run* / Publish / Export, then AG Grid (**single-row
      selection**) for per-scenario aggregates — clicking a row filters
      the charts upstream in Phase 2.

Beyond these regions, roadmap analytics (add as tabs or accordion later):

    * Shock attribution tornado — dominant risk legs per scenario slice.

    * Cluster routing matrix — ensemble coverage vs fallback buckets.

    * Train-baseline Δ overlay — run distribution vs frozen eval split.

    * Dispersion funnel — predictive band calibration when labels exist.

    * Per-member latency waterfall — exposes which cluster dominates wall
      clock during batch runs.

Page Contract anchors
---------------------

* §3 Rule L3 — ids only through :data:`INFERENCE_IDS`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import dash_mantine_components as dmc
from dash import dcc, html
from dash_iconify import DashIconify

from ..components.ag_grid_table import AgGridTable
from ..components.chart_container import ChartContainer
from ..components.kpi_card import KpiCard
from ..figures.inference_charts import (
    empty_pnl_distribution,
)

if TYPE_CHECKING:
    from ..data.session import Session


# ─────────────────────────────────────────────────────────────────────


INFERENCE_IDS: Dict[str, str] = {
    "root":                      "inference-root",
    "mount_signal":              "inference-mount-signal",
    "subtitle":                  "inference-subtitle",

    # Row 2 — scenario folder ingestion.
    "scenario_folder_path":      "inference-scenario-folder-path",
    "scenario_folder_upload":    "inference-scenario-folder-upload",
    "upload_scenarios_btn":      "inference-upload-scenarios-btn",
    "ingest_status":             "inference-ingest-status",

    # Row 3 · left — INPUT.
    "scenario_manifest":         "inference-scenario-manifest",
    "validate_only_btn":         "inference-validate-only-btn",
    "run_btn":                   "inference-run-btn",

    # Row 3 · right — KPI strip.
    "kpi_scenarios":             "inference-kpi-scenarios",
    "kpi_scenarios_value":       "inference-kpi-scenarios-value",
    "kpi_scenarios_spark":       "inference-kpi-scenarios-spark",
    "kpi_clusters":              "inference-kpi-clusters",
    "kpi_clusters_value":        "inference-kpi-clusters-value",
    "kpi_clusters_spark":        "inference-kpi-clusters-spark",
    "kpi_latency":               "inference-kpi-latency",
    "kpi_latency_value":         "inference-kpi-latency-value",
    "kpi_latency_spark":         "inference-kpi-latency-spark",
    "kpi_portfolio":             "inference-kpi-portfolio",
    "kpi_portfolio_value":       "inference-kpi-portfolio-value",
    "kpi_portfolio_spark":       "inference-kpi-portfolio-spark",

    # Row 3 · right — analytics tabs + chart mode + graph.
    "analytics_tabs":            "inference-analytics-tabs",
    "chart_view_mode":           "inference-chart-view-mode",
    "chart_main":                "inference-chart-main",

    # Row 3 · right — run footer actions.
    "save_run_as":               "inference-save-run-as",
    "publish_btn":               "inference-publish-btn",
    "export_json_btn":           "inference-export-json-btn",
    "export_csv_btn":            "inference-export-csv-btn",

    # Row 3 · right — results grid (row click → filters charts Phase 2).
    "scenario_results_grid":     "inference-scenario-results-grid",
}


_PLACEHOLDER = "—"


_V1_SUBTITLE = (
    "Price the book under freshly ingested scenarios — preview "
    "(no inference executor connected)"
)


_FOOTER_CAPTION = (
    "Runs hashed + persisted under inference_runs/ · audit hooks from "
    "Governance (Stage 2)"
)


def _scenario_ingestion_bar() -> html.Div:
    path_field = dmc.TextInput(
        id=INFERENCE_IDS["scenario_folder_path"],
        label="Scenario folder path",
        description="Filesystem path reachable by the app server",
        placeholder="/data/market/scenarios/sim_2028q3",
        size="sm",
        radius="md",
        classNames={"input": "font-mono text-xs"},
        style={"flex": "1 1 280px", "minWidth": "260px"},
    )

    browse = dcc.Upload(
        id=INFERENCE_IDS["scenario_folder_upload"],
        multiple=True,
        style={
            "display":               "inline-block",
            "border":                "none",
            "padding":               0,
            "margin":                0,
            "background":            "transparent",
            "cursor":                "pointer",
            "verticalAlign":         "middle",
        },
            children=dmc.Button(
            children="Browse files",
            variant="default",
            size="sm",
            radius="md",
            leftSection=DashIconify(icon="tabler:folder-open", width=14),
        ),
    )

    upload_trigger = dmc.Button(
        id=INFERENCE_IDS["upload_scenarios_btn"],
        children="Upload scenarios",
        variant="filled",
        color="violet",
        size="sm",
        radius="md",
        leftSection=DashIconify(icon="tabler:cloud-upload", width=16),
        loading=False,
        loaderProps={"type": "oval", "size": "xs"},
        n_clicks=0,
    )

    ingest_status = html.Div(
        id=INFERENCE_IDS["ingest_status"],
        title="Shows a green tick once scenarios are ingested",
        className=(
            "flex items-center justify-center w-10 h-10 "
            "rounded-md border border-slate-700/70 bg-slate-900/40"
        ),
        children=[],
    )

    return html.Div(
        className="rade-card flex flex-col gap-3",
        children=[
            html.Div(
                "Scenario source",
                className="text-sm font-semibold text-slate-200",
            ),
            html.Div(
                className="flex flex-row flex-wrap items-end gap-3 gap-y-5",
                children=[
                    path_field,
                    browse,
                    upload_trigger,
                    ingest_status,
                ],
            ),
            html.Div(
                "Browser multi-file pick gathers individual files; paste a "
                "folder path below for full-tree ingest on the server.",
                className="text-xs text-slate-500 leading-snug max-w-[720px]",
            ),
        ],
    )


def _input_panel() -> html.Div:
    manifest_box = html.Div(
        id=INFERENCE_IDS["scenario_manifest"],
        className=(
            "min-h-[200px] max-h-[340px] overflow-y-auto rounded-md "
            "border border-slate-800 bg-slate-950/50 p-3 text-xs "
            "text-slate-400 leading-relaxed"
        ),
        children=[
            html.Div(
                "No scenarios ingested yet.",
                className="text-slate-500 italic",
            ),
            html.Ul(
                className="list-disc mt-3 ps-5 space-y-1 text-slate-500",
                children=[
                    html.Li("Upload populates filenames, horizons, shocks."),
                    html.Li("Validate only checks manifest + shock schema."),
                    html.Li("Run executes ensemble inference on the bundle."),
                ],
            ),
        ],
    )

    actions = dmc.Group(
        gap="sm",
        grow=True,
        className="w-full mt-4",
        children=[
            dmc.Button(
                id=INFERENCE_IDS["validate_only_btn"],
                children="Validate only",
                variant="default",
                size="sm",
                radius="md",
                flex=1,
                leftSection=DashIconify(icon="tabler:checks", width=16),
                n_clicks=0,
            ),
            dmc.Button(
                id=INFERENCE_IDS["run_btn"],
                children="Run",
                variant="gradient",
                gradient={"from": "violet", "to": "cyan", "deg": 135},
                size="sm",
                radius="md",
                flex=1,
                leftSection=DashIconify(icon="tabler:player-play", width=16),
                n_clicks=0,
            ),
        ],
    )

    return html.Div(
        className="rade-card flex flex-col gap-3 flex-1 min-w-0",
        children=[
            html.Div(
                "Scenario bundle",
                className="text-sm font-semibold text-slate-200",
            ),
            html.Div(
                "Ingest summary",
                className="text-[11px] uppercase tracking-wide text-slate-500",
            ),
            manifest_box,
            actions,
        ],
    )


def _kpi_row() -> html.Div:
    return html.Div(
        className="grid grid-cols-4 gap-3",
        children=[
            KpiCard(
                label="Scenarios priced",
                value=_PLACEHOLDER,
                icon="tabler:stack-pop",
                card_id=INFERENCE_IDS["kpi_scenarios"],
                value_id=INFERENCE_IDS["kpi_scenarios_value"],
                sparkline_data=None,
                sparkline_id=INFERENCE_IDS["kpi_scenarios_spark"],
            ),
            KpiCard(
                label="Clusters touched",
                value=_PLACEHOLDER,
                icon="tabler:hierarchy",
                card_id=INFERENCE_IDS["kpi_clusters"],
                value_id=INFERENCE_IDS["kpi_clusters_value"],
                sparkline_data=None,
                sparkline_id=INFERENCE_IDS["kpi_clusters_spark"],
            ),
            KpiCard(
                label="Avg inference",
                value=_PLACEHOLDER,
                icon="tabler:gauge",
                card_id=INFERENCE_IDS["kpi_latency"],
                value_id=INFERENCE_IDS["kpi_latency_value"],
                sparkline_data=None,
                sparkline_id=INFERENCE_IDS["kpi_latency_spark"],
            ),
            KpiCard(
                label="Portfolio P&L (est)",
                value=_PLACEHOLDER,
                icon="tabler:trending-up",
                card_id=INFERENCE_IDS["kpi_portfolio"],
                value_id=INFERENCE_IDS["kpi_portfolio_value"],
                sparkline_data=None,
                sparkline_id=INFERENCE_IDS["kpi_portfolio_spark"],
            ),
        ],
    )


def _placeholder_panel(title: str, body: str) -> html.Div:
    return html.Div(
        className=(
            "flex flex-col gap-2 min-h-[200px] justify-center px-6 py-10 "
            "rounded-md border border-dashed border-slate-700/80 bg-slate-900/35"
        ),
        children=[
            html.Div(title, className="text-xs font-semibold text-slate-300"),
            html.Div(body, className="text-xs text-slate-500 max-w-xl"),
        ],
    )


def _charts_tab_body() -> html.Div:
    mode_toggle = dmc.SegmentedControl(
        id=INFERENCE_IDS["chart_view_mode"],
        value="distribution",
        size="xs",
        color="violet",
        radius="md",
        mb="xs",
        data=[
            {"label": "Distribution", "value": "distribution"},
            {"label": "Timeseries",   "value": "timeseries"},
            {"label": "More · TBD",   "value": "overlay"},
        ],
    )

    chart_card = ChartContainer(
        title="Aggregate book response",
        subtitle=(
            "Distribution / timeseries / overlay — callback swaps figure on "
            "``chart_main`` while honouring segmented state"
        ),
        graph_id=INFERENCE_IDS["chart_main"],
        figure=empty_pnl_distribution(),
        height=300,
        actions=[
            html.Div(
                className="flex flex-wrap justify-end",
                children=[mode_toggle],
            ),
        ],
    )

    return html.Div(className="flex flex-col gap-2", children=[chart_card])


def _analytics_tabs_block() -> dmc.Tabs:
    charts_tab = _charts_tab_body()

    sensitivity = _placeholder_panel(
        title="Sensitivity sweep",
        body=(
            "Tornado / Sobol indices / shock elasticity vs baseline — "
            "second-class analytics kept out of the hero chart lane."
        ),
    )

    diagnostics = _placeholder_panel(
        title="Diagnostics",
        body=(
            "Residual vs baseline, routing confidence histograms, dispersion "
            "checks — absorbs 'third view' exploratory analytics until we "
            "split them into first-class charts."
        ),
    )

    return dmc.Tabs(
        id=INFERENCE_IDS["analytics_tabs"],
        value="charts",
        color="violet",
        variant="outline",
        radius="md",
        className="w-full inference-analytics-tabs",
        children=[
            dmc.TabsList(
                grow=True,
                children=[
                    dmc.TabsTab(
                        value="charts",
                        flex=1,
                        leftSection=DashIconify(icon="tabler:chart-histogram", width=14),
                        children="Charts",
                    ),
                    dmc.TabsTab(
                        value="sensitivity",
                        flex=1,
                        leftSection=DashIconify(icon="tabler:chart-arcs", width=14),
                        children="Sensitivity",
                    ),
                    dmc.TabsTab(
                        value="diagnostics",
                        flex=1,
                        leftSection=DashIconify(icon="tabler:wave-sine", width=14),
                        children="Diagnostics",
                    ),
                ],
            ),
            dmc.TabsPanel(value="charts", pt="sm", pb=0, children=charts_tab),
            dmc.TabsPanel(value="sensitivity", pt="sm", children=sensitivity),
            dmc.TabsPanel(value="diagnostics", pt="sm", children=diagnostics),
        ],
    )


def _scenario_results_footer() -> html.Div:
    note = html.Div(
        "Selecting a scenario / trade row filters charts above — Stage 2",
        className="text-[11px] text-slate-500 ps-1",
    )

    actions = html.Div(
        className="flex flex-row flex-wrap items-end gap-2",
        children=[
            dmc.TextInput(
                id=INFERENCE_IDS["save_run_as"],
                placeholder="Save run as…",
                size="xs",
                radius="md",
                style={"flex": "2 1 180px"},
            ),
            dmc.Button(
                id=INFERENCE_IDS["publish_btn"],
                children="Publish to Governance",
                color="gray",
                variant="filled",
                size="xs",
                radius="md",
                leftSection=DashIconify(icon="tabler:clipboard-check", width=14),
            ),
            dmc.Button(
                id=INFERENCE_IDS["export_json_btn"],
                children="Export JSON",
                variant="default",
                size="xs",
                radius="md",
                leftSection=DashIconify(icon="tabler:brand-json", width=14),
            ),
            dmc.Button(
                id=INFERENCE_IDS["export_csv_btn"],
                children="Export CSV",
                variant="default",
                size="xs",
                radius="md",
                leftSection=DashIconify(icon="tabler:download", width=14),
            ),
        ],
    )

    return html.Div(
        className="flex flex-col gap-3",
        children=[
            html.Div(
                "Scenario / trade results",
                className="text-sm font-semibold text-slate-200",
            ),
            actions,
            note,
        ],
    )


def _results_ag_grid_defs() -> List[Dict[str, Any]]:
    conf_rules = {
        "rade-pill rade-pill--approved": (
            "params.value === 'High' || params.value === 'high'"
        ),
        "rade-pill rade-pill--pending": (
            "params.value === 'Medium' || params.value === 'medium'"
        ),
        "rade-pill rade-pill--archived": (
            "params.value === 'Low' || params.value === 'low'"
        ),
    }

    fmt_cap = (
        "params.value ? "
        "params.value.charAt(0).toUpperCase() + params.value.slice(1) "
        ": '—'"
    )

    return [
        {
            "field":       "scenario_id",
            "headerName": "Scenario",
            "pinned":     "left",
            "minWidth":   120,
            "cellClass":  "rade-grid-mono",
        },
        {
            "field":      "cluster",
            "headerName": "Cluster",
            "minWidth":   100,
            "cellClass":  "rade-grid-mono",
        },
        {
            "field":       "predicted",
            "headerName": "Predicted",
            "type":        "numericColumn",
            "minWidth":    100,
            "valueFormatter": {
                "function": (
                    "params.value == null ? '—' : "
                    "Number(params.value).toPrecision(5)"
                ),
            },
        },
        {"field": "p95_band", "headerName": "P95 band",
         "type": "numericColumn", "minWidth": 100},
        {
            "field":           "confidence",
            "headerName":      "Confidence",
            "minWidth":        110,
            "cellClassRules": conf_rules,
            "valueFormatter": {"function": fmt_cap},
        },
    ]


def _results_panel() -> html.Div:
    defs = _results_ag_grid_defs()
    return html.Div(
        className="rade-card flex flex-col gap-4 flex-1 min-w-0 p-5",
        children=[
            html.Div(
                "Results",
                className="text-sm font-semibold text-slate-200 mb-4",
            ),
            _kpi_row(),
            html.Div(
                children=[_analytics_tabs_block()],
                className="w-full shrink-0",
            ),
            _scenario_results_footer(),
            AgGridTable(
                grid_id=INFERENCE_IDS["scenario_results_grid"],
                row_data=[],
                column_defs=defs,
                grid_options={
                    "pagination": True,
                    "paginationPageSize": 12,
                    "paginationPageSizeSelector": [10, 25, 50, 100],
                    "rowHeight": 38,
                    "headerHeight": 38,
                    "animateRows": False,
                    "suppressCellFocus": True,
                    "rowSelection": "single",
                    "domLayout": "normal",
                },
                height=280,
                className="rade-inference-results-grid",
            ),
        ],
    )


def _row_main_workspace() -> html.Div:
    return html.Div(
        className="grid grid-cols-12 gap-4 items-start",
        children=[
            html.Div(className="col-span-12 lg:col-span-5", children=[_input_panel()]),
            html.Div(className="col-span-12 lg:col-span-7", children=[_results_panel()]),
        ],
    )


def _page_footer_line() -> html.Div:
    return html.Div(
        className="flex justify-end mt-8",
        children=[html.Span(_FOOTER_CAPTION, className="text-xs text-slate-500")],
    )


def build_inference(*, session: Optional["Session"] = None) -> html.Div:
    """Compose the Inference Console tree (Option A — no callbacks).

    ``session`` follows Page Contract §2.1 (uniform ``build_*`` signature).
    Chrome seeds version + split via ``TOPBAR_IDS``; no per-page hydration
    reads today — reserved for ingest-to-session persistence.
    """
    del session  # unused until capture callbacks hydrate inference state

    return html.Div(
        id=INFERENCE_IDS["root"],
        className="rade-page rade-inference flex flex-col gap-5",
        children=[
            dcc.Store(
                id=INFERENCE_IDS["mount_signal"],
                data=True,
                storage_type="memory",
            ),
            html.Div(
                className="flex flex-col gap-6",
                children=[
                    html.Div(
                        className="flex flex-col gap-1",
                        children=[
                            html.Div(
                                className="rade-page-title",
                                children="Inference Console",
                            ),
                            html.Span(
                                _V1_SUBTITLE,
                                id=INFERENCE_IDS["subtitle"],
                                className="text-xs text-slate-500",
                            ),
                            html.Span(
                                "Active ensemble version + split controlled from "
                                "the top bar.",
                                className="text-[11px] text-slate-600",
                            ),
                        ],
                    ),
                    _scenario_ingestion_bar(),
                    _row_main_workspace(),
                    _page_footer_line(),
                ],
            ),
        ],
    )


__all__ = ["INFERENCE_IDS", "build_inference"]

```

---

### Appendix A · 6 — `router.py` (PATCHES)

**1. Imports** — keep alphabetical among layout builders:

```python
from .layouts.data_quality import build_data_quality
from .layouts.evaluation import build_evaluation
from .layouts.governance import build_governance
from .layouts.inference import build_inference
from .layouts.monitoring import build_monitoring
from .layouts.overview import build_overview
```

**2. Route registration** — replace placeholder with real builder:

```python
    "/inference": PageSpec(
        path="/inference",
        title="Inference Console",
        build=build_inference,
    ),
```

---

### Appendix A · 7 — Smoke import

```bash
cd /path/to/QuantStrata && python -c '
from src.ui.apps.rade_analytics.layouts.inference import build_inference, INFERENCE_IDS
from src.ui.apps.rade_analytics.router import ROUTES
tree = build_inference()
assert tree.id == "inference-root"
assert len(tree.children) == 2
assert getattr(tree.children[0], "id", None) == "inference-mount-signal"
assert len(INFERENCE_IDS) == 30
assert ROUTES["/inference"].build.__name__ == "build_inference"
assert ROUTES["/inference"].title == "Inference Console"
print("OK")
'
```


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

## Appendix A — Data Quality page (Option A: empty layout, no callbacks)

Single appendix (replaces all prior appendix blocks).  Wires the
`/data-quality` route to a fully-structured layout that mirrors
`rade_data_quality.png` region-for-region, but with **no callbacks
and no API calls** — KPI values are em-dashes, the Completeness
Heatmap and Distribution Explorer mount with themed empty-state
placeholders, the Feature Summary grid mounts empty, and the filter
bar / feature picker mount with neutral defaults.  The page is honest
about its state via subtitle copy ("Per-feature completeness &
summary statistics — preview (no live artifacts connected)") and a
footer producer-disclosure caption naming the two parquets that will
drive it.

This shape gives reviewers a navigable preview of the proposed design
today, with **zero risk of `ReadTimeout` / API failures** even though
the entire backend pipe (`/prism/v1/quality/completeness`,
`/prism/v1/quality/feature-summary`, the typed client and the reader)
is already live.  When the populate callbacks ship the layout doesn't
change at all — the same `DATA_QUALITY_IDS` are already there, and
every empty-state placeholder gets overwritten via `Output`.

### File touch list

| Action | File | Purpose |
|---|---|---|
| **NEW** | `src/ui/apps/rade_analytics/figures/data_quality_charts.py` | `empty_completeness_heatmap()`, `empty_distribution_explorer()` — themed empty placeholders with "Awaiting …" annotations. |
| **NEW** | `src/ui/apps/rade_analytics/layouts/data_quality.py` | `DATA_QUALITY_IDS` contract + `build_data_quality(session=…)` — header + filter bar + KPI strip + heatmap + summary grid + distribution explorer + footer. |
| **PATCH** | `src/ui/apps/rade_analytics/router.py` | Import `build_data_quality`, swap the `/data-quality` placeholder for the real builder. |

### Files NOT touched

* `src/ui/apps/rade_analytics/data/backend.py` — no `data_quality_*`
  methods needed yet (every API call is deferred to Option B).
* `src/ui/apps/rade_analytics/callbacks/__init__.py` — no
  `data_quality_cb` registration (no callbacks today).
* `src/ui/apps/rade_analytics/data/session.py` — no Data Quality
  state persists yet; the split toggle reads from the existing
  `session.split` field, no schema bump required.
* `src/ui/apps/rade_analytics/assets/rade.css` — existing
  `rade-page-title` / `rade-card` / `rade-pill--*` / `rade-grid-mono`
  classes cover the V1 page.  The Missing % column already references
  the `rade-pill--rejected/pending/archived` palette via
  `cellClassRules` so when the populate callback emits numeric
  `missing_pct` values, severity colouring lights up automatically.
* The whole `/prism/v1/quality/*` backend pipe (router, models,
  reader, typed client) — already in place, untouched by this PR.

### Backend artifacts the page will read (when callbacks ship)

| Artifact | Schema | Driven components |
|---|---|---|
| `quality/completeness_{split}.parquet` | `cluster_id, feature_name, dtype, n_rows, n_null, null_rate, n_distinct, n_zero, n_inf, n_nan` | KPI strip (Total Features / Clusters / Complete / With Missing / Overall %) · Completeness Heatmap · Missing % column on Feature Summary table |
| `quality/feature_summary_{split}.parquet` | `cluster_id, feature_name, count, mean, std, p01, p50, p99, min, max` | Feature Summary table (Mean / Std / Min / Max / N columns) |
| Per-cluster raw feature samples (not yet exposed) | n/a | Distribution Explorer violin — **deferred**.  Two candidate paths: (a) approximate a box from the existing percentile stats, or (b) add a new `quality/feature_samples_{split}.parquet` writer.  V1 layout stays empty until that decision lands. |

Both parquets are written by the eval pipeline in
`src/rade_ml_pt/pipelines/ensemble/eval.py` Phase 5e (see
`completeness_frames` / `feature_summary_frames` accumulator block).

---

### A.1 — `figures/data_quality_charts.py` (NEW)

Two empty-state placeholders, both delegating to the shared
`empty_figure()` helper from `figures/_theme.py`.  Each carries a
chart-specific annotation hinting at the production source so the
preview reads as the proposed design rather than two identical "no
data" boxes.

```python
"""Plotly figure builders for the Data Quality page.

V1 ships **empty-state placeholders only** — every helper here returns
the standard themed empty figure (see :func:`figures._theme.empty_figure`)
with a chart-specific *awaiting-data* annotation.  Each placeholder
hints at what the production version of that chart will show, so the
Option-A preview reads as the proposed design rather than two identical
"no data" boxes.

When the Data Quality callbacks come online (Option B / V1 snapshot
mode), populated builders will land *next to* these helpers in this
same module, mirroring the layout used by ``cluster_deep_dive_charts.py``
(paired ``empty_*`` and ``populated_*`` factories).

Design anchor
-------------
``docs/platform_designs/rade_data_quality.png`` — Row 3 (completeness
heatmap, full-width) and Row 4 (feature summary table + distribution
explorer violin).
"""
from __future__ import annotations

import plotly.graph_objects as go

from ._theme import empty_figure


# ─────────────────────────────────────────────────────────────────────
# Empty-state placeholders
# ─────────────────────────────────────────────────────────────────────


def empty_completeness_heatmap() -> go.Figure:
    """Placeholder for the *Completeness Heatmap* in Row 3.

    Production version (Option B): X = ``cluster_id``, Y =
    ``feature_name``, Z = ``1 - null_rate`` (%).  Built from
    ``quality/completeness_{split}.parquet`` reshaped to a wide
    matrix; colour scale 0 % (purple) → 100 % (pink) matching the
    design palette.
    """
    return empty_figure("Awaiting completeness matrix from quality/completeness_{split}.parquet")


def empty_distribution_explorer() -> go.Figure:
    """Placeholder for the *Distribution Explorer* in Row 4.

    Production version (deferred): grouped violin per cluster for the
    feature selected in the right-hand dropdown.  Source TBD —
    ``quality/feature_summary_{split}.parquet`` only carries summary
    statistics (mean/std/percentiles), so V1 production either:

    1. Approximates a box from the existing percentile stats
       (p01 / p50 / p99 + min/max), or
    2. Adds a new ``quality/feature_samples_{split}.parquet`` writer
       on the eval side carrying sampled per-cluster values.

    Either path lights up this card without any layout change.
    """
    return empty_figure("Awaiting per-cluster feature distribution data")


__all__ = [
    "empty_completeness_heatmap",
    "empty_distribution_explorer",
]
```

---

### A.2 — `layouts/data_quality.py` (NEW)

The `DATA_QUALITY_IDS` dict locks every callable target up front so
the populate callbacks (when they land) never hardcode strings —
Page Contract §3 Rule L1 / L3.  The `mount_signal` Store (Rule L4)
gives the future render callback a reliable trigger that fires after
the DOM swap.

`build_data_quality()` reads `session.split` for the initial Train /
Val / Test toggle value (Rule L1: initial UI state from session at
build time, no hydration callback) and falls through to `"test"`
when no session is supplied — matching the design's default active
pill.

```python
"""Data Quality page layout.

Mirrors ``docs/platform_designs/rade_data_quality.png`` region-for-
region:

* **Row 0** — invisible mount tripwire (Page Contract §3 Rule L4) so
  the Stage-2 render callback can fire on layout mount without
  racing the DOM swap.
* **Row 1** — Header band: page title + subtitle on the left, and a
  filter bar (Train/Val/Test segmented control + Cluster select +
  Search input) plus the *Export CSV* CTA on the right.
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
* The split toggle, cluster filter, search box and feature dropdown
  mount with neutral defaults so the chrome reads exactly as it will
  once callbacks land — but no callback fires today.

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

    # Row 1 — Filter bar.
    "split_toggle":               "data-quality-split-toggle",
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


# Split toggle options — matches the design pill order.  Values stay
# lowercase to align with the ``Session.split`` literal type and the
# ``/prism/v1/quality/*`` ``split`` query parameter.
_SPLIT_OPTIONS: List[Dict[str, str]] = [
    {"label": "Train", "value": "train"},
    {"label": "Val",   "value": "val"},
    {"label": "Test",  "value": "test"},
]
_SPLIT_DEFAULT = "test"


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


def _filter_bar(*, initial_split: str) -> html.Div:
    """Train / Val / Test toggle + cluster select + search + export CTA.

    Mounted with neutral defaults so the chrome reads as the populated
    page would — no callback runs today.  The split SegmentedControl
    seeds from the live session so navigating in from another page
    keeps the user's split intact.
    """
    return html.Div(
        className="flex items-center justify-between gap-3 flex-wrap",
        children=[
            html.Div(
                className="flex items-center gap-3 flex-wrap",
                children=[
                    dmc.SegmentedControl(
                        id=DATA_QUALITY_IDS["split_toggle"],
                        data=_SPLIT_OPTIONS,
                        value=initial_split,
                        size="sm",
                        color="violet",
                        radius="md",
                    ),
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


def _row_header(*, initial_split: str) -> html.Div:
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
            _filter_bar(initial_split=initial_split),
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

    The ``session`` kwarg is read for the initial split toggle value
    (Page Contract §3 Rule L1 — initial UI state from session at
    build time, no hydration callback needed).  Every other field
    in :data:`DATA_QUALITY_IDS` mounts with neutral / empty
    defaults that the populate callback will overwrite once it
    lands.
    """
    initial_split = session.split if session is not None else _SPLIT_DEFAULT

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
            _row_header(initial_split=initial_split),
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
```

---

### A.3 — `router.py` (PATCHED)

Two targeted edits, both in `src/ui/apps/rade_analytics/router.py`.
Imports stay alphabetical; the route table swaps the placeholder for
the real builder.

**Patch 1 — add the import**

Replace:

```python
from .layouts.evaluation import build_evaluation
from .layouts.governance import build_governance
from .layouts.monitoring import build_monitoring
from .layouts.overview import build_overview
```

with:

```python
from .layouts.data_quality import build_data_quality
from .layouts.evaluation import build_evaluation
from .layouts.governance import build_governance
from .layouts.monitoring import build_monitoring
from .layouts.overview import build_overview
```

**Patch 2 — wire the route**

Replace:

```python
    "/data-quality": PageSpec(
        path="/data-quality",
        title="Data Quality",
        build=_placeholder("Data Quality", "Phase E"),
    ),
```

with:

```python
    "/data-quality": PageSpec(
        path="/data-quality",
        title="Data Quality",
        build=build_data_quality,
    ),
```

The `_placeholder` helper stays in the file — it still backs
`/inference`, `/scenario-lab`, `/report-builder`, `/assistant`.

---

### A.4 — Smoke test

After applying the three changes, the following sequence should run
clean from the repo root:

```bash
python -c "
from src.ui.apps.rade_analytics.layouts.data_quality import build_data_quality, DATA_QUALITY_IDS
from src.ui.apps.rade_analytics.figures.data_quality_charts import (
    empty_completeness_heatmap, empty_distribution_explorer,
)
from src.ui.apps.rade_analytics.router import ROUTES

tree = build_data_quality()
assert tree.id == 'data-quality-root'
assert len(tree.children) == 6  # mount Store + 5 visible rows
assert ROUTES['/data-quality'].build.__name__ == 'build_data_quality'
assert len(DATA_QUALITY_IDS) == 21
empty_completeness_heatmap()
empty_distribution_explorer()
print('OK')
"
```

Expected output: `OK` (along with the same id / count diagnostics
already validated in the implementation PR).

---

### A.5 — Unblocking the populate callback (one-line restore)

When the V1 snapshot mode (Option B) is ready, two adjustments turn
the page live without touching the layout:

```python
# callbacks/__init__.py
from . import data_quality_cb           # ← add
data_quality_cb.register(app, backend)  # ← add inside register_all
```

At that point the layout doesn't change at all — the same
`DATA_QUALITY_IDS` are already there, every empty-state placeholder
gets overwritten via `Output`, and the typed PRISM client
(`backend.completeness(split, …)` / `backend.feature_summary(split,
…)`) — backed by `/prism/v1/quality/*` — returns ready-shaped
DataFrames the moment the eval pipeline writes the underlying
parquets.


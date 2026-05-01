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
## Appendix A — Inference Console (V2 · scenario ingest + activity log + 5-tab analytics)

This appendix supersedes whatever lived in Appendix A before.
Implementation pairs with [`docs/rade_analytics/page_contract.md`](../rade_analytics/page_contract.md)
and the mechanical workflow in [`docs/rade_analytics/page_template/README.md`](../rade_analytics/page_template/README.md).

**Design anchor:** scenario-first variant of [`rade_inference.png`](rade_inference.png) — global **version + split**
live in chrome (`TOPBAR_IDS["version_select"]`, `"split_toggle"`).
This is still **Option A** (layout-only, no callbacks); Stage 2 lights up
the contract in §4 below.

---

### Appendix A · 1 — Contract + template checklist (pre-flight before callbacks)

Pass criteria for reviewers — should match Cluster Deep-Dive / Data Quality parity.

| Item | Requirement | This page |
| --- | --- | --- |
| §2·1 Pure layout | `build_inference(*, session=…)` is side-effect-free; zero backend imports under `layouts/`. | **Pass** · only `components/`, `figures/`, `dcc`/`dmc`/`html`. |
| §2·1 `Session` signature | Builders accept `Optional[Session]` + `del session` honestly. | **Pass** · `del session` with note until capture persists state. |
| §3 Rule L2 | `<PAGE>_IDS` dict owns every callable id (`INFERENCE_IDS`). | **Pass** · 42 stable `inference-*` ids. |
| §3 Rule L3 | Prefer shared styles in `rade.css`; primitives use Tailwind like sibling pages until a page-specific block is warranted. | **Pass** reuse `rade-card`, `rade-page-title`, pills + new `rade-activity-*` / `rade-stress-mini-*` blocks (§5.3). |
| §3 Rule L4 | `dcc.Store(id=mount_signal, data=True, storage_type="memory")` sits under page root early in children. | **Pass** · `inference-mount-signal` plus four data Stores (§4.1). |
| Chrome scope | Ensemble **version** + **split** come from shell top bar (`TOPBAR_IDS`); do not duplicate on the page. | **Pass** · caption references top bar. |
| Template layout | Mirrors template row order: tripwire + Stores → header → ingest bar → 2-col workspace; future `callbacks/inference_cb.py` pairs with `layouts/inference.py`. | **Pass** Phase A layout only (`inference_cb` not merged yet). |
| §11 mock coverage | Honour palette / card cadence §4–§6 (gradient CTA sparingly — *Run* button only). | **Pass** violet→cyan mirrors splash CTA conventions. |

**Known platform limitation (`dcc.Upload` in Dash 3.4):** does **not** expose `directory=True`. Multi-file browse + paste-server-path is the authoritative story until a packaged `clientside_callback` attaches `webkitdirectory`.

---

### Appendix A · 2 — File touch summary

| Action | Path |
| --- | --- |
| **NEW** | `src/ui/apps/rade_analytics/figures/inference_charts.py` (now exports 5 empties — adds `empty_risk_attribution`, `empty_stress_tails`) |
| **NEW** | `src/ui/apps/rade_analytics/layouts/inference.py` |
| **PATCH** | `src/ui/apps/rade_analytics/router.py` — alphabetised imports + `/inference` route + breadcrumb title |
| **PATCH** | `src/ui/apps/rade_analytics/assets/rade.css` — appends `rade-activity-*` and `rade-stress-mini-*` blocks |
| Deferred | `src/ui/apps/rade_analytics/callbacks/inference_cb.py` (+ `callbacks/__init__.py` registration) — see §4.3 wiring table |

No `data/backend.py` changes for Option A.  Stage 2 will add a `RadeBackend.run_inference(...)` returning a typed `RunMeta`-shaped payload (see §4.1).

---

### Appendix A · 3 — V2 delta vs V1 (what's actually new)

| Region | V1 | V2 |
| --- | --- | --- |
| INPUT (Row 3 left) | One scrollable manifest box + Run / Validate buttons. | **Two stacked cards**: Activity log (top, scrollable, status-coded) → Manifest preview (bottom, scrollable, holds Run / Validate buttons). |
| Page-level Stores | `mount_signal` only. | `mount_signal` + `activity_log_store` + `ingest_meta_store` + `run_meta_store` + `selected_scenario_store`. |
| Analytics tabs | 3 — Charts · Sensitivity · Diagnostics. | **5** — Charts · Sensitivity · **Risk attribution** (new) · **Stress & tails** (new) · Diagnostics. |
| Empty figures | 3 — distribution / timeseries / overlay. | **5** — adds `empty_risk_attribution`, `empty_stress_tails`. |
| Workspace grid | (V1.1 fix) `lg:grid-cols-5` + `lg:col-span-2/3` (Cluster Deep-Dive idiom). | Unchanged — same parent. |
| Public render helpers | none | `render_activity_entries(entries)` — used by both the empty-state layout and Stage-2 callbacks. |

---

### Appendix A · 4 — Callback functional spec (Stage 2 contract)

This page is intentionally **stateful via Stores** so Stage-2 callbacks
remain pure functions of `(stores, controls)` — no globals, no module
state.  The store contracts below are the source of truth; everything
else (chart / KPI / grid populates) derives from them.

#### 4.1 Store contracts

| Store ID | Initial | Shape | Owner / writer |
| --- | --- | --- | --- |
| `inference-mount-signal` | `True` | bool | layout — Page Contract Rule L4 tripwire |
| `inference-activity-log-store` | `[]` | `List[ActivityEntry]` (append-only) | ingest, validate, run callbacks |
| `inference-ingest-meta-store` | `None` | `IngestMeta \| None` | upload-scenarios callback |
| `inference-run-meta-store` | `None` | `RunMeta \| None` | run callback |
| `inference-selected-scenario-store` | `None` | `str \| None` (scenario id) | scenario-grid row-click callback |

```python
# Append-only feed entry (see render_activity_entries):
ActivityEntry = TypedDict("ActivityEntry", {
    "id":     str,                        # uuid for stable React keys
    "stage":  Literal["ingest", "validate", "inference"],
    "phase":  str,                        # human readable phase
    "target": NotRequired[str],           # filename / cluster id / risk factor
    "status": Literal["ok", "fail", "running", "pending"],
    "ts":     str,                        # ISO-8601 or HH:MM:SS
    "detail": NotRequired[str],           # error message / extra detail
})

IngestMeta = TypedDict("IngestMeta", {
    "source":       str,                  # folder path or "uploaded"
    "files":        List[FileMeta],
    "started_ts":   str,
    "completed_ts": str,
    "n_files":      int,
    "n_scenarios":  int,
})

FileMeta = TypedDict("FileMeta", {
    "name":         str,
    "scenarios":    int,
    "risk_factors": List[str],
    "valid":        bool,
    "errors":       List[str],
})

RunMeta = TypedDict("RunMeta", {
    "run_id":       str,
    "started_ts":   str,
    "completed_ts": str,
    "elapsed_ms":   int,
    "n_clusters":   int,
    "n_trades":     int,
    "kpi": {
        "scenarios":         int,
        "clusters":          int,
        "avg_inference_ms":  float,
        "portfolio_pnl":     float,
    },
})
```

#### 4.2 Activity-log lifecycle (golden-path narrative)

The activity log is the user-facing *story* of what's happening.  Every
state transition pushes one or more entries.  Stage-2 callbacks should
emit at least the events below; intermediate progress is welcome but
not required.

**Phase 1 — ingest (per file in the bundle):**

```text
[ingest][running] File loaded                  · scenario_001.json
[ingest][ok]      Manifest parsed              · scenario_001.json
[ingest][ok]      Scenarios ingested into RF1  · scenario_001.json (100 scenarios)
[ingest][fail]    Schema mismatch              · scenario_002.json (detail: missing 'horizon')
```

**Phase 2 — validate (when *Validate only* clicked, or pre-flight before run):**

```text
[validate][running] Cross-checking shock schema vs ensemble version v0.4.2
[validate][ok]      All risk factors covered
[validate][ok]      No NaN / Inf shocks
```

**Phase 3 — inference (when *Run* clicked):**

```text
[inference][running] Cluster 1 (RF1) model loading
[inference][ok]      Cluster 1 (RF1) model loaded
[inference][ok]      Cluster 1 (RF1) input batch ready (100 scenarios x 8 trades)
[inference][ok]      Cluster 1 (RF1) forward pass complete (3.4 ms)
[inference][ok]      Cluster 1 (RF1) predictions ready
... per cluster ...
[inference][ok]      Aggregation complete · KPIs published
```

#### 4.3 Callback wiring (one row per Stage-2 callback)

| # | Trigger | Outputs | Reads | Notes |
| --- | --- | --- | --- | --- |
| 1 | `Input(upload_scenarios_btn, "n_clicks")` | `Output(upload_scenarios_btn, "loading")`, `Output(ingest_status, "children")`, `Output(ingest_meta_store, "data")`, `Output(activity_log_store, "data")`, `Output(manifest_preview_container, "children")` | `State(scenario_folder_path, "value")`, `State(scenario_folder_upload, "contents")`, `State(scenario_folder_upload, "filename")`, `State(activity_log_store, "data")` | Streams ingest events into the activity log; final write fills `ingest_meta_store` and renders manifest preview. Use a long-running callback if ingest > ~200 ms. |
| 2 | `Input(activity_log_store, "data")` | `Output(activity_log_container, "children")` | — | Pure render: `render_activity_entries(data)`. |
| 3 | `Input(validate_only_btn, "n_clicks")` | `Output(activity_log_store, "data", allow_duplicate=True)` | `State(ingest_meta_store, "data")`, `State(activity_log_store, "data")` | Validates the ingested manifest only; emits validate events. |
| 4 | `Input(run_btn, "n_clicks")` | `Output(activity_log_store, "data", allow_duplicate=True)`, `Output(run_meta_store, "data")`, `Output(scenario_results_grid, "rowData")` | `State(ingest_meta_store, "data")`, `State(activity_log_store, "data")`, `State(TOPBAR_IDS["version_select"], "value")` | Executes inference; streams per-cluster events; writes per-scenario rows. |
| 5 | `Input(run_meta_store, "data")` | `Output(kpi_*_value, "children")` × 4, `Output(kpi_*_spark, "data")` × 4 | — | KPI hydrate from `RunMeta.kpi`. |
| 6 | `Input(chart_view_mode, "value")`, `Input(run_meta_store, "data")`, `Input(selected_scenario_store, "data")` | `Output(chart_main, "figure")` | — | Swaps distribution / timeseries / overlay; re-slices when row selected. |
| 7 | `Input(risk_attribution_breakdown, "value")`, `Input(run_meta_store, "data")`, `Input(selected_scenario_store, "data")` | `Output(risk_attribution_chart, "figure")` | — | Attribution view; bar when 1-deep, treemap when nested. |
| 8 | `Input(stress_tails_mode, "value")`, `Input(run_meta_store, "data")`, `Input(selected_scenario_store, "data")` | `Output(stress_tails_chart, "figure")`, `Output(stress_kpi_var, "children")`, `Output(stress_kpi_cvar, "children")`, `Output(stress_kpi_worst, "children")` | — | Recomputes tail stats over current slice. |
| 9 | `Input(scenario_results_grid, "selectedRows")` | `Output(selected_scenario_store, "data")` | — | Single-select; null when deselected. |
| 10 | `Input(publish_btn, "n_clicks")` | `Output(activity_log_store, "data", allow_duplicate=True)` (governance event) | `State(run_meta_store, "data")`, `State(save_run_as, "value")`, `State(TOPBAR_IDS["version_select"], "value")` | Posts to Governance writer (Stage 3); emits *Published to governance* event. |

#### 4.4 Anti-patterns to avoid

* **Do not** use `Input(scenario_folder_upload, "contents")` as a callback
  trigger — analysts want to confirm the bundle before ingest fires.
  Use the *Upload scenarios* button as the explicit trigger.
* **Do not** filter charts by mutating `run_meta_store` — slice from a
  read-only copy and key chart callbacks off `selected_scenario_store`.
* **Do not** mutate `activity_log_store` in-place; always return a new
  list.  React reconciles on identity, and append-in-place will drop
  entries silently.


---

### Appendix A · 5 — Source files (full code)

#### 5.1 `figures/inference_charts.py` (NEW · full file)

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
* **Risk attribution** — bar / treemap of P&L contribution split by
  cluster, risk-factor, and trade type.  Tells the user *why* the
  book moved under the new scenarios.
* **Stress & tails** — VaR / CVaR cross-sections, tail histogram and
  fan-chart percentile band.  Regulatory / risk-management view of
  the same scenarios.

Builders are intentionally trivial wrappers around ``empty_figure``
so the page renders cleanly before any inference run; replace each
helper body once the inference executor lands a typed payload.
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


def empty_risk_attribution() -> go.Figure:
    """Placeholder for *P&L attribution by cluster · risk-factor · trade-type*.

    Stage 2 will replace this with either a horizontal bar chart
    (cluster contribution, signed) or a treemap (cluster → risk
    factor → trade type) depending on the user's last selected
    breakdown axis.
    """
    return empty_figure("Awaiting scenario run — attribution view")


def empty_stress_tails() -> go.Figure:
    """Placeholder for *VaR / CVaR / tail-quantile* view.

    Stage 2 candidates: percentile band fan chart over scenario
    index, tail-bar (VaR vs CVaR), or worst-N scenarios sparkline.
    """
    return empty_figure("Awaiting scenario run — tail-risk view")


__all__ = [
    "empty_pnl_distribution",
    "empty_pnl_overlay",
    "empty_pnl_timeseries",
    "empty_risk_attribution",
    "empty_stress_tails",
]
```

#### 5.2 `layouts/inference.py` (NEW · full file)

```python
"""Inference Console page layout — **scenario ingestion** path (V2).

This iteration upgrades the original "scenario folder ingest" mock with
two product-driven additions:

1.  **Activity log** in the INPUT column — a streaming, status-coded
    feed that narrates *every* lifecycle event (file loaded, scenarios
    ingested into a risk factor, model loaded, forward pass, predictions
    ready, …) with green-tick / red-cross / pulsing-circle icons.

2.  **Two new analytics tabs** in the RESULTS column — *Risk
    attribution* (P&L breakdown by cluster · risk factor · trade type)
    and *Stress & tails* (VaR, CVaR, worst-N).  These sit alongside the
    existing **Charts**, **Sensitivity** and **Diagnostics** tabs.

Row map
-------

* **Row 0** — ``dcc.Store`` mount tripwire (Page Contract §3 Rule L4)
  *plus* four data Stores driving the page state machine: activity
  log, ingest meta, run meta, selected scenario.

* **Row 1** — Title + subtitle.  Ensemble **version** and **train /
  val / test** split remain in the chrome topbar (:data:`TOPBAR_IDS`)
  — not duplicated here — so this page stays aligned with the rest of
  Rade.

* **Row 2** — Scenario folder sourcing bar:

    * ``dmc.TextInput`` — paste a server-accessible folder path.

    * ``dcc.Upload`` — ``multiple=True``, wraps a *Browse files* button
      so analysts can pick many scenario files at once.  True
      **folder** selection (``webkitdirectory``) is not exposed by
      ``dcc.Upload`` in Dash 3.4 — use the pasted path for recursive
      directory ingest on the server, or extend with a small
      clientside bundle later.

    * *Upload scenarios* ``dmc.Button`` — exposes ``loading=`` /
      ``loaderProps`` at ``False``; the hydrate callback toggles
      ``loading``, then swaps the ingest status slot for an emerald
      tick when ingest completes.

    * Ingest feedback slot ``#inference-ingest-status`` — empty shell
      the callback swaps to ``tabler:circle-check``, error icon, etc.

* **Row 3** — Two-column workspace:

    * **Left · INPUT** — *two stacked cards*.

        - **Activity log card** — scrollable feed bound to
          :data:`INFERENCE_IDS["activity_log_store"]`.  Each entry is
          a dict (see *Callback contract* below) and the renderer
          emits one ``rade-activity-row`` per entry.

        - **Manifest card** — scrollable manifest preview of the
          ingested bundle (filenames, scenario counts, risk-factor
          coverage, validity).  Footer holds **Validate only** and
          **Run** (gradient violet→cyan) actions; both gated until
          ingest completes.

    * **Right · RESULTS** — four KPI tiles (sparkline slots via
      ``sparkline_id``); ``dmc.Tabs`` (**Charts · Sensitivity · Risk
      attribution · Stress & tails · Diagnostics**).

      * **Charts** holds the segmented control (Distribution /
        Timeseries / *More · TBD*) and the main ``dcc.Graph``
        (:class:`~components.chart_container.ChartContainer`).

      * **Risk attribution** holds a *breakdown* segmented control
        (Cluster / Risk factor / Trade type) and an attribution chart
        (bar / treemap depending on the breakdown).

      * **Stress & tails** holds a 3-mini-KPI strip (VaR · CVaR ·
        Worst loss) and a tail-view chart (percentile fan / worst-N
        sparkline / tail histogram).

      Below the tabs sit *Save run* / *Publish* / *Export*, then the
      ``rade-inference-results-grid`` AG Grid (single-row select) for
      per-scenario aggregates.  Selecting a row fills
      :data:`INFERENCE_IDS["selected_scenario_store"]`, which the
      Stage-2 chart callbacks key off to filter every chart above.

Callback contract (Stage 2)
---------------------------

The page is intentionally **stateful via Stores** so callbacks can
remain pure functions.  Use these as the contract:

* :data:`INFERENCE_IDS["activity_log_store"]` — ``List[ActivityEntry]``.
  Append-only.  Each entry::

      {
          "id":     "<uuid>",          # stable React key
          "stage":  "ingest|validate|inference",
          "phase":  "File loaded",     # human readable
          "target": "scenario_001.json",  # optional
          "status": "ok|fail|running|pending",
          "ts":     "2028-04-01T12:30:01Z",
          "detail": "Optional error / extra detail string",
      }

  See :func:`render_activity_entries` — exposed publicly so
  callbacks can rebuild the rendered list cheaply each time the
  store mutates.

* :data:`INFERENCE_IDS["ingest_meta_store"]` — ``IngestMeta | None``::

      {
          "source":     "/data/.../sim_2028q3" | "uploaded",
          "files":      [
              {"name": "...", "scenarios": 100, "risk_factors": [...],
               "valid": true, "errors": []},
              ...
          ],
          "started_ts": "...",
          "completed_ts": "...",
          "n_files":    3,
          "n_scenarios": 300,
      }

* :data:`INFERENCE_IDS["run_meta_store"]` — ``RunMeta | None`` —
  populated when the ensemble inference run completes::

      {
          "run_id":      "<uuid>",
          "started_ts":  "...",
          "completed_ts": "...",
          "elapsed_ms":  1234,
          "n_clusters":  4,
          "n_trades":    120,
          "kpi": {"scenarios": 300, "clusters": 4,
                  "avg_inference_ms": 4.2, "portfolio_pnl": 12345.6},
      }

* :data:`INFERENCE_IDS["selected_scenario_store"]` — ``str | None``
  scenario id; row click on the results grid sets this; chart
  callbacks observe it and slice the underlying frame.

Beyond V2, roadmap analytics (add as additional tabs / accordions):

    * Cluster routing matrix — ensemble coverage vs fallback buckets.

    * Train-baseline Δ overlay — run distribution vs frozen eval split.

    * Per-member latency waterfall — exposes which cluster dominates
      wall-clock during batch runs.

    * Comparison tab — scenario-A vs scenario-B side-by-side.

Page Contract anchors
---------------------

* §3 Rule L3 — ids only through :data:`INFERENCE_IDS`.
* §3 Rule L4 — ``mount_signal`` Store + ``rade-page`` root class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import dash_mantine_components as dmc
from dash import dcc, html
from dash_iconify import DashIconify

from ..components.ag_grid_table import AgGridTable
from ..components.chart_container import ChartContainer
from ..components.kpi_card import KpiCard
from ..figures.inference_charts import (
    empty_pnl_distribution,
    empty_risk_attribution,
    empty_stress_tails,
)

if TYPE_CHECKING:
    from ..data.session import Session


# ─────────────────────────────────────────────────────────────────────


INFERENCE_IDS: Dict[str, str] = {
    "root":                       "inference-root",
    "mount_signal":               "inference-mount-signal",
    "subtitle":                   "inference-subtitle",

    # Row 0 — page-level data Stores driving the state machine.
    "activity_log_store":         "inference-activity-log-store",
    "ingest_meta_store":          "inference-ingest-meta-store",
    "run_meta_store":             "inference-run-meta-store",
    "selected_scenario_store":    "inference-selected-scenario-store",

    # Row 2 — scenario folder ingestion.
    "scenario_folder_path":       "inference-scenario-folder-path",
    "scenario_folder_upload":     "inference-scenario-folder-upload",
    "upload_scenarios_btn":       "inference-upload-scenarios-btn",
    "ingest_status":              "inference-ingest-status",

    # Row 3 · left — INPUT column.
    "activity_log_container":     "inference-activity-log-container",
    "manifest_preview_container": "inference-manifest-preview-container",
    "validate_only_btn":          "inference-validate-only-btn",
    "run_btn":                    "inference-run-btn",

    # Row 3 · right — KPI strip.
    "kpi_scenarios":              "inference-kpi-scenarios",
    "kpi_scenarios_value":        "inference-kpi-scenarios-value",
    "kpi_scenarios_spark":        "inference-kpi-scenarios-spark",
    "kpi_clusters":               "inference-kpi-clusters",
    "kpi_clusters_value":         "inference-kpi-clusters-value",
    "kpi_clusters_spark":         "inference-kpi-clusters-spark",
    "kpi_latency":                "inference-kpi-latency",
    "kpi_latency_value":          "inference-kpi-latency-value",
    "kpi_latency_spark":          "inference-kpi-latency-spark",
    "kpi_portfolio":              "inference-kpi-portfolio",
    "kpi_portfolio_value":        "inference-kpi-portfolio-value",
    "kpi_portfolio_spark":        "inference-kpi-portfolio-spark",

    # Row 3 · right — analytics tabs (5 tabs).
    "analytics_tabs":             "inference-analytics-tabs",

    # Charts tab — segmented mode + main graph.
    "chart_view_mode":            "inference-chart-view-mode",
    "chart_main":                 "inference-chart-main",

    # Risk-attribution tab — breakdown axis + chart.
    "risk_attribution_breakdown": "inference-risk-attribution-breakdown",
    "risk_attribution_chart":     "inference-risk-attribution-chart",

    # Stress & tails tab — mini KPI strip + chart-mode + chart.
    "stress_tails_mode":          "inference-stress-tails-mode",
    "stress_tails_chart":         "inference-stress-tails-chart",
    "stress_kpi_var":             "inference-stress-kpi-var",
    "stress_kpi_cvar":            "inference-stress-kpi-cvar",
    "stress_kpi_worst":           "inference-stress-kpi-worst",

    # Row 3 · right — run footer actions.
    "save_run_as":                "inference-save-run-as",
    "publish_btn":                "inference-publish-btn",
    "export_json_btn":            "inference-export-json-btn",
    "export_csv_btn":             "inference-export-csv-btn",

    # Row 3 · right — results grid (row click → filters charts Stage 2).
    "scenario_results_grid":      "inference-scenario-results-grid",
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


# ─────────────────────────────────────────────────────────────────────
# Activity log — public render helper used by both layout + callbacks.
# ─────────────────────────────────────────────────────────────────────


_STATUS_ICON = {
    "ok":      ("tabler:circle-check",   "rade-activity-icon--ok"),
    "fail":    ("tabler:circle-x",       "rade-activity-icon--fail"),
    "running": ("tabler:loader-2",       "rade-activity-icon--running"),
    "pending": ("tabler:circle-dashed",  "rade-activity-icon--pending"),
}


_STAGE_LABEL = {
    "ingest":    "Ingest",
    "validate":  "Validate",
    "inference": "Inference",
}


def _activity_row(entry: Dict[str, Any]) -> html.Div:
    """Render one activity feed entry into a ``rade-activity-row``."""
    status = str(entry.get("status", "pending"))
    icon_name, icon_class = _STATUS_ICON.get(status, _STATUS_ICON["pending"])
    stage = str(entry.get("stage", "ingest"))
    phase = str(entry.get("phase", "—"))
    target = entry.get("target")
    detail = entry.get("detail")
    ts = str(entry.get("ts", ""))

    body_children: List[Any] = [
        html.Span(_STAGE_LABEL.get(stage, stage.title()),
                  className="rade-activity-stage"),
        html.Span(phase, className="rade-activity-phase"),
    ]
    if target:
        body_children.append(
            html.Span(target, className="rade-activity-target font-mono")
        )
    if detail:
        body_children.append(
            html.Div(detail, className="rade-activity-detail")
        )

    return html.Div(
        className="rade-activity-row",
        children=[
            html.Div(
                className=f"rade-activity-icon {icon_class}",
                children=DashIconify(icon=icon_name, width=16),
            ),
            html.Div(className="rade-activity-body", children=body_children),
            html.Span(ts, className="rade-activity-ts font-mono"),
        ],
    )


def render_activity_entries(
    entries: Optional[Sequence[Dict[str, Any]]],
) -> List[Any]:
    """Render an activity log store payload into row children.

    Intentionally exposed for Stage-2 callbacks: pass the current
    contents of :data:`INFERENCE_IDS["activity_log_store"]` and return
    the children of :data:`INFERENCE_IDS["activity_log_container"]`.
    Callbacks should append entries to the store, then call this with
    the *full* list — the diff is cheap enough at typical bundle sizes
    (<200 entries per run).
    """
    if not entries:
        return [
            html.Div(
                className="rade-activity-empty",
                children=[
                    DashIconify(
                        icon="tabler:wave-square",
                        width=18,
                        className="text-slate-600",
                    ),
                    html.Div(
                        "Activity feed will populate as scenarios are "
                        "uploaded, ingested, validated and priced.",
                        className="text-xs text-slate-500 leading-snug",
                    ),
                ],
            ),
        ]
    return [_activity_row(e) for e in entries]


# ─────────────────────────────────────────────────────────────────────
# Row 2 — Scenario ingestion bar
# ─────────────────────────────────────────────────────────────────────


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
            "display":        "inline-block",
            "border":         "none",
            "padding":        0,
            "margin":         0,
            "background":     "transparent",
            "cursor":         "pointer",
            "verticalAlign":  "middle",
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
            "rounded-md border border-slate-700 bg-slate-950/40"
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
                # ``flex-wrap`` + ``gap-y-4`` so the path field stretches
                # horizontally on wide viewports but the trio of buttons
                # drops below it cleanly on narrow ones.
                className="flex flex-row flex-wrap items-end gap-3 gap-y-4",
                children=[
                    path_field,
                    browse,
                    upload_trigger,
                    ingest_status,
                ],
            ),
            html.Div(
                "Browser multi-file pick gathers individual files; paste a "
                "folder path above for full-tree ingest on the server.",
                className="text-xs text-slate-500 leading-snug max-w-2xl",
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Row 3 · left — INPUT column (Activity log + Manifest)
# ─────────────────────────────────────────────────────────────────────


def _activity_log_card() -> html.Div:
    return html.Div(
        className="rade-card flex flex-col gap-2 min-w-0",
        children=[
            html.Div(
                className="flex items-center justify-between",
                children=[
                    html.Div(
                        "Activity log",
                        className="text-sm font-semibold text-slate-200",
                    ),
                    html.Div(
                        "Live — append-only",
                        className=(
                            "text-[11px] uppercase tracking-wide "
                            "text-slate-500"
                        ),
                    ),
                ],
            ),
            html.Div(
                id=INFERENCE_IDS["activity_log_container"],
                className=(
                    "rade-activity-log overflow-y-auto rounded-md "
                    "border border-slate-800 bg-slate-950/40 p-3"
                ),
                style={"minHeight": "180px", "maxHeight": "320px"},
                children=render_activity_entries(None),
            ),
        ],
    )


def _manifest_card() -> html.Div:
    manifest_box = html.Div(
        id=INFERENCE_IDS["manifest_preview_container"],
        className=(
            "overflow-y-auto rounded-md border border-slate-800 "
            "bg-slate-950/40 p-3 text-xs text-slate-400 leading-relaxed"
        ),
        style={"minHeight": "160px", "maxHeight": "260px"},
        children=[
            html.Div(
                "No scenarios ingested yet.",
                className="text-slate-500",
            ),
            html.Div(
                className="flex flex-col gap-1 mt-3 text-slate-500",
                children=[
                    html.Div("• Upload populates filenames, horizons, shocks."),
                    html.Div("• Validate only checks manifest + shock schema."),
                    html.Div("• Run executes ensemble inference on the bundle."),
                ],
            ),
        ],
    )

    actions = dmc.Group(
        gap="sm",
        grow=True,
        className="w-full mt-3",
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
        className="rade-card flex flex-col gap-2 min-w-0",
        children=[
            html.Div(
                className="flex items-center justify-between",
                children=[
                    html.Div(
                        "Scenario bundle",
                        className="text-sm font-semibold text-slate-200",
                    ),
                    html.Div(
                        "Manifest preview",
                        className=(
                            "text-[11px] uppercase tracking-wide "
                            "text-slate-500"
                        ),
                    ),
                ],
            ),
            manifest_box,
            actions,
        ],
    )


def _input_panel() -> html.Div:
    return html.Div(
        className="flex flex-col gap-4 min-w-0",
        children=[
            _activity_log_card(),
            _manifest_card(),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Row 3 · right — RESULTS column
# ─────────────────────────────────────────────────────────────────────


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
    # Dashed-border + arbitrary min-height live in inline ``style``
    # because ``border-dashed`` and ``min-h-[200px]`` aren't in the
    # compiled ``rade.css`` utility bundle.
    return html.Div(
        className=(
            "flex flex-col gap-2 justify-center px-6 py-6 rounded-md "
            "border border-slate-800"
        ),
        style={
            "minHeight":   "200px",
            "borderStyle": "dashed",
            "background":  "rgba(15, 23, 42, 0.45)",
        },
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
            "Distribution / timeseries / overlay — callback swaps figure "
            "on ``chart_main`` while honouring segmented state and the "
            "active row in the scenario grid."
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


def _risk_attribution_tab_body() -> html.Div:
    """Risk-attribution view — bar/treemap by selected breakdown axis."""
    breakdown = dmc.SegmentedControl(
        id=INFERENCE_IDS["risk_attribution_breakdown"],
        value="cluster",
        size="xs",
        color="violet",
        radius="md",
        mb="xs",
        data=[
            {"label": "By cluster",      "value": "cluster"},
            {"label": "By risk factor",  "value": "risk_factor"},
            {"label": "By trade type",   "value": "trade_type"},
        ],
    )

    chart_card = ChartContainer(
        title="P&L attribution",
        subtitle=(
            "Signed contribution of each bucket to portfolio P&L for the "
            "current scenario set — bar when grouped one-deep, treemap "
            "when nested (Stage 2 swaps ``figure`` shape via callback)."
        ),
        graph_id=INFERENCE_IDS["risk_attribution_chart"],
        figure=empty_risk_attribution(),
        height=300,
        actions=[
            html.Div(
                className="flex flex-wrap justify-end",
                children=[breakdown],
            ),
        ],
    )

    return html.Div(className="flex flex-col gap-2", children=[chart_card])


def _stress_tails_tab_body() -> html.Div:
    """Stress / tail-risk view — mini KPI strip + chart-mode toggle."""
    mini_kpis = html.Div(
        className="grid grid-cols-3 gap-3",
        children=[
            html.Div(
                id=INFERENCE_IDS["stress_kpi_var"],
                className="rade-stress-mini-kpi",
                children=[
                    html.Div("VaR (95%)", className="rade-stress-mini-label"),
                    html.Div(_PLACEHOLDER, className="rade-stress-mini-value"),
                ],
            ),
            html.Div(
                id=INFERENCE_IDS["stress_kpi_cvar"],
                className="rade-stress-mini-kpi",
                children=[
                    html.Div("CVaR (95%)", className="rade-stress-mini-label"),
                    html.Div(_PLACEHOLDER, className="rade-stress-mini-value"),
                ],
            ),
            html.Div(
                id=INFERENCE_IDS["stress_kpi_worst"],
                className="rade-stress-mini-kpi",
                children=[
                    html.Div("Worst loss", className="rade-stress-mini-label"),
                    html.Div(_PLACEHOLDER, className="rade-stress-mini-value"),
                ],
            ),
        ],
    )

    mode_toggle = dmc.SegmentedControl(
        id=INFERENCE_IDS["stress_tails_mode"],
        value="fan",
        size="xs",
        color="violet",
        radius="md",
        mb="xs",
        data=[
            {"label": "Percentile fan", "value": "fan"},
            {"label": "Tail histogram", "value": "tail"},
            {"label": "Worst N",        "value": "worst"},
        ],
    )

    chart_card = ChartContainer(
        title="Tail-risk view",
        subtitle=(
            "Percentile band over scenario index, tail histogram of "
            "scenario P&L, or top-N worst-loss scenarios — callback "
            "swaps figure shape on ``stress_tails_chart``."
        ),
        graph_id=INFERENCE_IDS["stress_tails_chart"],
        figure=empty_stress_tails(),
        height=260,
        actions=[
            html.Div(
                className="flex flex-wrap justify-end",
                children=[mode_toggle],
            ),
        ],
    )

    return html.Div(
        className="flex flex-col gap-3",
        children=[mini_kpis, chart_card],
    )


def _analytics_tabs_block() -> dmc.Tabs:
    charts_tab      = _charts_tab_body()
    risk_attr_tab   = _risk_attribution_tab_body()
    stress_tab      = _stress_tails_tab_body()

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
            "Residual vs baseline, routing confidence histograms, "
            "dispersion checks — absorbs exploratory analytics until "
            "they earn first-class chart status."
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
                        leftSection=DashIconify(
                            icon="tabler:chart-histogram", width=14,
                        ),
                        children="Charts",
                    ),
                    dmc.TabsTab(
                        value="sensitivity",
                        flex=1,
                        leftSection=DashIconify(
                            icon="tabler:chart-arcs", width=14,
                        ),
                        children="Sensitivity",
                    ),
                    dmc.TabsTab(
                        value="risk_attribution",
                        flex=1,
                        leftSection=DashIconify(
                            icon="tabler:chart-treemap", width=14,
                        ),
                        children="Risk attribution",
                    ),
                    dmc.TabsTab(
                        value="stress_tails",
                        flex=1,
                        leftSection=DashIconify(
                            icon="tabler:chart-area-line", width=14,
                        ),
                        children="Stress & tails",
                    ),
                    dmc.TabsTab(
                        value="diagnostics",
                        flex=1,
                        leftSection=DashIconify(
                            icon="tabler:wave-sine", width=14,
                        ),
                        children="Diagnostics",
                    ),
                ],
            ),
            dmc.TabsPanel(value="charts",           pt="sm", pb=0,
                          children=charts_tab),
            dmc.TabsPanel(value="sensitivity",      pt="sm",
                          children=sensitivity),
            dmc.TabsPanel(value="risk_attribution", pt="sm",
                          children=risk_attr_tab),
            dmc.TabsPanel(value="stress_tails",     pt="sm",
                          children=stress_tab),
            dmc.TabsPanel(value="diagnostics",      pt="sm",
                          children=diagnostics),
        ],
    )


def _scenario_results_footer() -> html.Div:
    note = html.Div(
        "Selecting a scenario / trade row filters charts above — Stage 2",
        className="text-[11px] text-slate-500",
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
            "field":      "scenario_id",
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
            "field":      "predicted",
            "headerName": "Predicted",
            "type":       "numericColumn",
            "minWidth":   100,
            "valueFormatter": {
                "function": (
                    "params.value == null ? '—' : "
                    "Number(params.value).toPrecision(5)"
                ),
            },
        },
        {"field": "p95_band", "headerName": "P95 band",
         "type":  "numericColumn", "minWidth": 100},
        {
            "field":          "confidence",
            "headerName":     "Confidence",
            "minWidth":       110,
            "cellClassRules": conf_rules,
            "valueFormatter": {"function": fmt_cap},
        },
    ]


def _results_panel() -> html.Div:
    defs = _results_ag_grid_defs()
    return html.Div(
        # ``rade-card`` already supplies padding, border + radius;
        # don't double-pad with an extra ``p-5`` here.
        className="rade-card flex flex-col gap-4 min-w-0",
        children=[
            html.Div(
                "Results",
                className="text-sm font-semibold text-slate-200",
            ),
            _kpi_row(),
            html.Div(
                children=[_analytics_tabs_block()],
                className="w-full",
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
    """Two-column workspace.

    The compiled ``rade.css`` only ships responsive grid utilities for
    ``lg:grid-cols-5`` + ``lg:col-span-2`` / ``lg:col-span-3`` (Cluster
    Deep-Dive idiom).  Keep this idiom — ``grid-cols-12`` / wider
    spans are not in the bundle and would silently no-op.
    """
    return html.Div(
        className="grid grid-cols-1 lg:grid-cols-5 gap-4 items-stretch",
        children=[
            html.Div(
                className="lg:col-span-2 flex flex-col min-w-0",
                children=[_input_panel()],
            ),
            html.Div(
                className="lg:col-span-3 flex flex-col min-w-0",
                children=[_results_panel()],
            ),
        ],
    )


def _page_footer_line() -> html.Div:
    return html.Div(
        className="flex justify-end mt-8",
        children=[html.Span(_FOOTER_CAPTION, className="text-xs text-slate-500")],
    )


def _page_stores() -> List[Any]:
    """All ``dcc.Store`` mounts driving the page state machine.

    Kept as a list so ``build_inference`` can splat them at the top of
    the page tree without tracking individual ids.
    """
    return [
        dcc.Store(
            id=INFERENCE_IDS["mount_signal"],
            data=True,
            storage_type="memory",
        ),
        dcc.Store(
            id=INFERENCE_IDS["activity_log_store"],
            data=[],
            storage_type="memory",
        ),
        dcc.Store(
            id=INFERENCE_IDS["ingest_meta_store"],
            data=None,
            storage_type="memory",
        ),
        dcc.Store(
            id=INFERENCE_IDS["run_meta_store"],
            data=None,
            storage_type="memory",
        ),
        dcc.Store(
            id=INFERENCE_IDS["selected_scenario_store"],
            data=None,
            storage_type="memory",
        ),
    ]


def build_inference(*, session: Optional["Session"] = None) -> html.Div:
    """Compose the Inference Console tree (V2 — empty layout, no callbacks).

    ``session`` follows Page Contract §2.1 (uniform ``build_*`` signature).
    Chrome seeds version + split via ``TOPBAR_IDS``; no per-page hydration
    reads today — reserved for ingest-to-session persistence.
    """
    del session  # unused until capture callbacks hydrate inference state

    return html.Div(
        id=INFERENCE_IDS["root"],
        className="rade-page rade-inference flex flex-col gap-5",
        children=[
            *_page_stores(),
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
                                "Active ensemble version + split controlled "
                                "from the top bar.",
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


__all__ = [
    "INFERENCE_IDS",
    "build_inference",
    "render_activity_entries",
]
```

#### 5.3 `assets/rade.css` (PATCH · appended block)

Append to the bottom of `rade.css`. The `rade-activity-*` block styles
the streaming feed in the INPUT column; the `rade-stress-mini-*` block
styles the 3-tile mini KPI strip inside the *Stress & tails* tab.

```css
/* ── Inference Console — activity log feed ─────────────────────── */

/* The activity log lives inside the INPUT column (Row 3 left).  It
   renders an append-only event stream where each row pairs a
   status-coloured icon with a stage tag, a phase string, an
   optional target (file / cluster / risk-factor) and an optional
   detail string.  Markup-wise the layout uses
   ``html.Div(className="rade-activity-row")`` for each event — see
   ``layouts/inference.py::_activity_row``. */

.rade-activity-log {
  display: flex;
  flex-direction: column;
  gap: 0;
}

.rade-activity-row {
  display: grid;
  grid-template-columns: 28px 1fr auto;
  gap: 0.625rem;
  align-items: flex-start;
  padding: 0.5rem 0;
  border-bottom: 1px solid rgba(30, 41, 59, 0.6);
}

.rade-activity-row:first-child { padding-top: 0; }
.rade-activity-row:last-child  { border-bottom: 0; padding-bottom: 0; }

.rade-activity-icon {
  width: 24px;
  height: 24px;
  border-radius: 9999px;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: rgba(148, 163, 184, 0.12);   /* slate-400/12 */
  color: #94a3b8;                                /* slate-400 */
}

.rade-activity-icon--ok {
  background-color: rgba(16, 185, 129, 0.16);
  color: #34d399;                                /* emerald-400 */
}

.rade-activity-icon--fail {
  background-color: rgba(244, 63, 94, 0.16);
  color: #fb7185;                                /* rose-400 */
}

.rade-activity-icon--running {
  background-color: rgba(139, 92, 246, 0.18);
  color: #a78bfa;                                /* violet-400 */
  animation: rade-activity-spin 1.2s linear infinite;
}

.rade-activity-icon--pending {
  background-color: rgba(245, 158, 11, 0.16);
  color: #fbbf24;                                /* amber-400 */
}

@keyframes rade-activity-spin {
  to { transform: rotate(360deg); }
}

.rade-activity-body {
  display: flex;
  flex-direction: column;
  gap: 0.125rem;
  min-width: 0;
}

.rade-activity-stage {
  display: inline-block;
  font-size: 0.65rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: #94a3b8;                                /* slate-400 */
  margin-right: 0.5rem;
}

.rade-activity-phase {
  font-size: 0.8125rem;
  color: #e2e8f0;                                /* slate-200 */
}

.rade-activity-target {
  display: inline-block;
  font-size: 0.7rem;
  color: #cbd5e1;                                /* slate-300 */
  background-color: rgba(15, 23, 42, 0.7);       /* slate-900/70 */
  border: 1px solid rgba(30, 41, 59, 0.6);
  padding: 0.0625rem 0.4rem;
  border-radius: 0.375rem;
  margin-left: 0.5rem;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 100%;
}

.rade-activity-detail {
  font-size: 0.7rem;
  color: #94a3b8;                                /* slate-400 */
  margin-top: 0.125rem;
}

.rade-activity-ts {
  font-size: 0.65rem;
  color: #64748b;                                /* slate-500 */
  white-space: nowrap;
  align-self: center;
}

.rade-activity-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  padding: 1rem;
  color: #475569;                                /* slate-600 */
  text-align: center;
}

/* ── Inference Console — stress / tails mini KPI strip ─────────── */

/* Three mini KPI tiles inside the *Stress & tails* tab — purposely
   smaller than ``KpiCard`` so they fit above the chart without
   stealing visual weight.  Markup uses
   ``html.Div(className="rade-stress-mini-kpi")``. */

.rade-stress-mini-kpi {
  display: flex;
  flex-direction: column;
  gap: 0.125rem;
  padding: 0.5rem 0.75rem;
  border-radius: 0.5rem;
  border: 1px solid rgba(30, 41, 59, 0.6);
  background-color: rgba(15, 23, 42, 0.7);       /* slate-900/70 */
}

.rade-stress-mini-label {
  font-size: 0.65rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: #94a3b8;                                /* slate-400 */
}

.rade-stress-mini-value {
  font-family: "JetBrains Mono", "IBM Plex Mono", ui-monospace,
               SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 1rem;
  font-weight: 600;
  color: #e2e8f0;                                /* slate-200 */
}
```

#### 5.4 `router.py` (PATCH)

```python
# In src/ui/apps/rade_analytics/router.py — alphabetised import list:
from .layouts.inference         import build_inference

# In ROUTES table (alphabetised):
"/inference":         RouteSpec(
    build=build_inference,
    title="Inference Console",
),
```

---

### Appendix A · 6 — Smoke import

```bash
cd /path/to/QuantStrata && python -c '
from src.ui.apps.rade_analytics.layouts.inference import (
    build_inference, INFERENCE_IDS, _row_main_workspace, _input_panel,
    _analytics_tabs_block, _page_stores, render_activity_entries,
)
from src.ui.apps.rade_analytics.router import ROUTES

tree = build_inference()
assert tree.id == "inference-root"
assert len(INFERENCE_IDS) == 42

stores = _page_stores()
assert {s.id for s in stores} == {
    "inference-mount-signal",
    "inference-activity-log-store",
    "inference-ingest-meta-store",
    "inference-run-meta-store",
    "inference-selected-scenario-store",
}

ws = _row_main_workspace()
assert ws.className == "grid grid-cols-1 lg:grid-cols-5 gap-4 items-stretch"
assert "lg:col-span-2" in ws.children[0].className
assert "lg:col-span-3" in ws.children[1].className

ip = _input_panel()
assert len(ip.children) == 2  # activity card + manifest card

tabs = _analytics_tabs_block()
tab_values = [t.value for t in tabs.children[0].children]
assert tab_values == [
    "charts", "sensitivity", "risk_attribution",
    "stress_tails", "diagnostics",
]

# render helper round-trips
rows = render_activity_entries([
    {"stage": "ingest", "phase": "File loaded",
     "target": "sim_001.json", "status": "ok", "ts": "12:30:01"},
])
assert len(rows) == 1 and rows[0].className == "rade-activity-row"

assert ROUTES["/inference"].build.__name__ == "build_inference"
assert ROUTES["/inference"].title == "Inference Console"
print("OK")
'
```

---

### Appendix A · 7 — Layout gotcha: `rade.css` is hand-compiled, not JIT

`src/ui/apps/rade_analytics/assets/rade.css` is a **hand-compiled
Tailwind bundle**, not a runtime JIT pass.  Only utilities that
already shipped (see `tailwind.input.css` + the previous build
scan) are present at runtime.  Classes that look fine in the source
but **silently no-op in the browser** include:

- `grid-cols-12`, `col-span-5`, `col-span-7`, `col-span-12`
- `lg:col-span-5`, `lg:col-span-7`
- `gap-y-5`
- arbitrary values: `min-h-[…]`, `max-h-[…]`, `max-w-[…]`, `min-w-[…]`
- opacity variants: `bg-slate-900/35`, `bg-slate-900/40`,
  `bg-slate-950/50`, `border-slate-700/70`, `border-slate-700/80`
- `border-dashed`, `italic`, `list-disc`, `space-y-1`,
  logical-property `ps-*` / `pe-*`

**Approved substitutions** (already in `rade.css`):

| Need | Use instead |
| --- | --- |
| Two-column workspace | `grid grid-cols-1 lg:grid-cols-5 gap-4 items-stretch` parent + `lg:col-span-2` / `lg:col-span-3` (Cluster Deep-Dive idiom) |
| Soft transparent surface | `bg-slate-900/60` (or `bg-slate-950/40`) |
| Soft border | `border-slate-800/60` or `border-slate-700` |
| Arbitrary heights | inline `style={"minHeight": "…", "maxHeight": "…"}` |
| Dashed border | inline `style={"borderStyle": "dashed"}` |
| Bullet lists | plain stacked `html.Div`s with leading `"• "` |
| Italic text | drop or replace with `text-slate-500` weight cue |

Whenever you add a new utility class, sanity-check it appears in
`rade.css` (or rebuild the bundle per `assets/README.md`); the page
will render but content will collapse to natural width if the
parent grid loses its column-span utilities.

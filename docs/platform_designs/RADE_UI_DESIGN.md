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


## Appendix A — Phase E.1 Portfolio polish (`assets/rade.css` append)

> **Usage.**  Four tiny component classes the Portfolio sub-tab
> references that aren't in `rade.css` yet:
>
> * `rade-section-divider` — the "Error analysis" divider row.
> * `rade-focus-chip` + `rade-focus-chip-close` — the "Focused: …
>   [× Show all]" chip in the scatter card header.
> * `rade-portfolio-groupby-select` — `Break down by` dropdown tint.
> * `rade-portfolio-leaderboard-grid-wrap` / `rade-portfolio-leaderboard`
>   — leaderboard layout hooks + AgGrid theme overrides.
>
> **Action.**  Append the block below to the end of
> `src/ui/apps/rade_analytics/assets/rade.css`.  Safe to paste as-is —
> every rule is scoped under a `rade-*` class and won't collide with
> anything existing.  Reuses the design tokens already in the file:
> slate-950 `#0f172a`, slate-800 `#1e293b`, violet `#8b5cf6`,
> slate-400 `#94a3b8`, slate-500 `#64748b`.
>
> Strip this appendix once the classes are in `rade.css` (same
> convention as B / D / E).

```css
/* ─────────────────────────────────────────────────────────────────── */
/* PHASE E.1 — Portfolio sub-tab (evaluation)                          */
/* ─────────────────────────────────────────────────────────────────── */

/* "Error analysis" divider row — section caption + horizontal rule + */
/* break-down control on one line.  Tailwind utility classes provide  */
/* the flex layout; this file adds the subtle top / bottom spacing    */
/* and the uppercase caption polish.                                   */
.rade-section-divider {
  padding-top:    0.5rem;
  padding-bottom: 0.25rem;
  margin-top:     0.5rem;
  margin-bottom:  0.25rem;
}
.rade-section-divider .rade-section-divider-caption {
  font-size: 11px;
  line-height: 1.4;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #94a3b8;
}

/* Focus chip — surfaced inside ChartContainer's `actions` slot when   */
/* a point is clicked on the grouped scatter.                          */
.rade-focus-chip {
  display: inline-flex;
  align-items: center;
  gap: 0.375rem;
  padding: 0.1875rem 0.5rem;
  border-radius: 9999px;
  background-color: rgba(139, 92, 246, 0.12);   /* violet @ 12% */
  border: 1px solid rgba(139, 92, 246, 0.35);
  color: #cbd5e1;
  font-size: 11px;
  line-height: 1.4;
  font-weight: 500;
  transition: background-color 150ms, border-color 150ms;
}
.rade-focus-chip:hover {
  background-color: rgba(139, 92, 246, 0.18);
  border-color:     rgba(139, 92, 246, 0.5);
}

/* The "× Show all" button inside the chip.  Styled as a link so it    */
/* reads as an action without competing visually with the chip label.  */
.rade-focus-chip-close {
  appearance: none;
  background: transparent;
  border: 0;
  padding: 0 0 0 0.25rem;
  margin-left: 0.125rem;
  color: #94a3b8;
  font-size: 11px;
  line-height: 1;
  cursor: pointer;
  transition: color 120ms;
}
.rade-focus-chip-close:hover {
  color: #f1f5f9;
}
.rade-focus-chip-close:focus-visible {
  outline: 2px solid #8b5cf6;
  outline-offset: 2px;
  border-radius: 0.25rem;
}

/* Break-down Select — nudges DMC's base tint into the slate palette   */
/* the rest of the page already uses.                                  */
.rade-portfolio-groupby-select input {
  background-color: #0f172a;
  border-color: #1e293b;
  color: #e2e8f0;
}
.rade-portfolio-groupby-select input::placeholder {
  color: #64748b;
}
.rade-portfolio-groupby-select input:focus-visible {
  border-color: #8b5cf6;
  box-shadow: 0 0 0 1px rgba(139, 92, 246, 0.35);
}

/* Leaderboard AgGrid — the grid itself is styled by AgGrid's          */
/* alpine-dark theme; these rules just give the wrapper a consistent   */
/* border + radius so it reads as part of the leaderboard card.        */
.rade-portfolio-leaderboard-grid-wrap {
  border-radius: 0.5rem;
  overflow: hidden;
  border: 1px solid #1e293b;
}
.rade-portfolio-leaderboard {
  --ag-background-color:          #0f172a;
  --ag-odd-row-background-color:  #0b1220;
  --ag-header-background-color:   #111c2e;
  --ag-border-color:              #1e293b;
  --ag-row-border-color:          #1e293b;
  --ag-foreground-color:          #e2e8f0;
  --ag-header-foreground-color:   #94a3b8;
  --ag-secondary-foreground-color: #64748b;
  --ag-selected-row-background-color: rgba(139, 92, 246, 0.12);
  --ag-range-selection-background-color: rgba(139, 92, 246, 0.18);
  --ag-font-family: "Inter", system-ui, sans-serif;
  --ag-font-size: 12px;
}
```

After pasting, hard-reload the browser (`Cmd+Shift+R`) so the updated
`rade.css` is fetched — Flask's dev server hands out a new `ETag` every
time the file changes, but the browser can cling to the cached copy
between sub-tab switches.

---

## Appendix B — Phase E.3 Trade-Graph paste blocks

Phase E.3 ships the Evaluation → Trade-Graph sub-tab:

* New `/prism/v1/trade-graph` endpoint (cluster-scoped Cytoscape payload).
* Eval pipeline stages `graph_results.joblib` + `trade_universe.json`
  into `evaluation/members/{cluster_id}/` so the bundle is shippable.
* New Cytoscape-based layout with 4 side-panel cards (Selected trade,
  Legend, Cluster stats, Ensemble summary) and two secondary charts
  (density histogram, edges vs nodes scatter).

### B.1 File actions

| File | Action |
|---|---|
| `src/rade_ml_pt/pipelines/ensemble/eval.py` | MODIFY — add `_copy_member_graph_artifacts` helper + call site |
| `src/rade_ml_pt/ensemble/api/services/paths.py` | MODIFY — add `member_graph_results` / `member_trade_universe` |
| `src/rade_ml_pt/ensemble/api/models/trade_graph.py` | CREATE |
| `src/rade_ml_pt/ensemble/api/services/reader.py` | MODIFY — `_load_joblib` + `trade_graph` method |
| `src/rade_ml_pt/ensemble/api/routers/trade_graph.py` | CREATE |
| `src/rade_ml_pt/ensemble/api/app.py` | MODIFY — register `trade_graph_router` |
| `src/rade_ml_pt/ensemble/api/client.py` | MODIFY — `trade_graph` method |
| `src/ui/apps/rade_analytics/data/backend.py` | MODIFY — cached `trade_graph` method |
| `src/ui/apps/rade_analytics/data/session.py` | MODIFY — bump `SESSION_SCHEMA_VERSION` → 4, extend `EvaluationState` |
| `src/ui/apps/rade_analytics/figures/graph_charts.py` | CREATE |
| `src/ui/apps/rade_analytics/figures/__init__.py` | MODIFY — re-export `density_distribution` / `edges_vs_nodes_scatter` |
| `src/ui/apps/rade_analytics/layouts/evaluation/trade_graph.py` | REPLACE stub |
| `src/ui/apps/rade_analytics/callbacks/trade_graph_cb.py` | CREATE |
| `src/ui/apps/rade_analytics/callbacks/__init__.py` | MODIFY — register `trade_graph_cb` |
| `examples/rade_analytics/_mock_backend.py` | MODIFY — synthetic `trade_graph` + `graph_stats_df` |
| `examples/rade_analytics/08_trade_graph_preview_live.py` | CREATE |
| `requirements-rade-ui.txt` | already lists `dash-cytoscape>=1.0.0` — run `pip install -r requirements-rade-ui.txt` |

### B.2 `src/rade_ml_pt/pipelines/ensemble/eval.py` — modify

Add `import shutil` near the other stdlib imports, then in
`_save_all_artifacts` append a new call after `_save_graph_stats_parquet`:

```python
_copy_member_graph_artifacts(
    eval_dir=eval_dir,
    config=config,
    member_versions=member_versions,
)
```

Then add this helper module-local (after `_save_graph_stats_parquet`):

```python
def _copy_member_graph_artifacts(
    eval_dir: Path,
    config: EnsembleConfig,
    member_versions: Optional[Dict[str, str]] = None,
) -> None:
    """Stage per-member graph artefacts into the eval directory.

    For each cluster, copies ``graph_results.joblib`` and
    ``trade_universe.json`` from ``{registry_dir}/{member_version}/``
    into ``{eval_dir}/members/{cluster_id}/``.  Makes the evaluation
    bundle self-contained so the API can read everything it needs
    without touching the training registry.
    """
    if not member_versions:
        member_versions = config.metadata.get("job", {}).get("member_versions", {})

    if not isinstance(member_versions, dict) or not member_versions:
        logger.info("  skipping graph-artefact staging (no member versions)")
        return

    files_to_copy = ("graph_results.joblib", "trade_universe.json")
    n_clusters_staged = 0
    n_files_staged = 0

    for cid in config.cluster_ids:
        member_version = member_versions.get(cid)
        if not member_version:
            continue

        src_dir = Path(config.registry_dir) / member_version
        dst_dir = eval_dir / "members" / cid
        dst_dir.mkdir(parents=True, exist_ok=True)

        cluster_files_staged = 0
        for filename in files_to_copy:
            src = src_dir / filename
            if not src.exists():
                logger.debug(
                    "  member graph artefact missing: %s (cluster=%s)",
                    src, cid,
                )
                continue
            try:
                shutil.copy2(src, dst_dir / filename)
                cluster_files_staged += 1
            except Exception as exc:
                logger.warning(
                    "  could not stage %s for cluster '%s': %s",
                    filename, cid, exc,
                )

        if cluster_files_staged > 0:
            n_clusters_staged += 1
            n_files_staged += cluster_files_staged

    logger.info(
        "  staged member graph artefacts: %d files across %d clusters",
        n_files_staged, n_clusters_staged,
    )
```

### B.3 `src/rade_ml_pt/ensemble/api/services/paths.py` — modify

Inside the `ArtifactPaths` class, add two properties next to the other
`member_*` path builders:

```python
def member_graph_results(self, cluster_id: str) -> Path:
    """Sparse adjacency joblib staged from the training registry."""
    return self.eval_dir / "members" / cluster_id / "graph_results.joblib"

def member_trade_universe(self, cluster_id: str) -> Path:
    """Trade-universe JSON staged from the training registry."""
    return self.eval_dir / "members" / cluster_id / "trade_universe.json"
```

### B.4 `src/rade_ml_pt/ensemble/api/models/trade_graph.py` — CREATE

```python
"""Pydantic response schemas for the ``/prism/v1/trade-graph`` endpoint.

Serves one cluster's full trade-level graph — the sparse adjacency
recorded in ``graph_results.joblib`` plus the ``target`` / ``elementary``
trade split from ``trade_universe.json`` (both staged into the eval
artefact bundle by :mod:`rade_ml_pt.pipelines.ensemble.eval`).
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class TradeGraphNode(BaseModel):
    trade_id: str = Field(..., description="Trade identifier (string).")
    cluster_id: str = Field(..., description="Parent cluster identifier.")
    trade_type: str = Field(
        ...,
        description=(
            "``target`` or ``elementary`` — classifies the trade for the "
            "UI's node colouring and the Selected-Trade card."
        ),
    )


class TradeGraphEdge(BaseModel):
    source: str = Field(..., description="Source trade_id.")
    target: str = Field(..., description="Target trade_id.")
    weight: float = Field(
        ...,
        description="Adjacency entry from ``sparse_values`` — graph-builder units.",
    )


class TradeGraphStats(BaseModel):
    n_nodes: int
    n_edges: int
    density: float
    mean_weight: float


class TradeGraphResponse(BaseModel):
    cluster_id: str
    n_target_trades: int
    n_elementary_trades: int
    stats: TradeGraphStats
    nodes: List[TradeGraphNode]
    edges: List[TradeGraphEdge]
    warnings: Optional[List[str]] = None


__all__ = [
    "TradeGraphEdge",
    "TradeGraphNode",
    "TradeGraphResponse",
    "TradeGraphStats",
]
```

### B.5 `src/rade_ml_pt/ensemble/api/services/reader.py` — modify

Add these imports at the top with the others:

```python
import numpy as np
from src.rade_ml_pt.ensemble.api.models.trade_graph import (
    TradeGraphEdge,
    TradeGraphNode,
    TradeGraphResponse,
    TradeGraphStats,
)
```

Add this module-level helper (near `_load_json`):

```python
def _load_joblib(path: Path) -> Any:
    """Load a joblib file — deferred import to keep ``joblib`` optional."""
    import joblib
    return joblib.load(path)
```

Then add this method inside `ArtifactReader` (after `graph_stats_df`):

```python
def trade_graph(self, cluster_id: str) -> TradeGraphResponse:
    """Build the full trade-graph payload for one cluster.

    Reads ``members/{cluster_id}/graph_results.joblib`` and the
    adjacent ``trade_universe.json``; both are staged by the eval
    pipeline (``_copy_member_graph_artifacts``).

    Raises
    ------
    FileNotFoundError
        If ``graph_results.joblib`` is missing for the cluster.
    """
    graph_path = self.paths.member_graph_results(cluster_id)
    tu_path = self.paths.member_trade_universe(cluster_id)

    if not graph_path.exists():
        raise FileNotFoundError(
            f"graph_results.joblib missing for cluster '{cluster_id}' "
            f"(looked for {graph_path}). Re-run the eval pipeline to "
            f"stage member graph artefacts."
        )

    gr = read_with_mtime_cache(graph_path, _load_joblib)

    indices = np.asarray(gr.get("sparse_indices", []), dtype=int)
    values = np.asarray(gr.get("sparse_values", []), dtype=float)
    shape = gr.get("sparse_shape", [0, 0])
    n = int(shape[0]) if shape and shape[0] > 0 else 0

    warnings: List[str] = []

    target_ids: List[str] = []
    elementary_ids: List[str] = []
    if tu_path.exists():
        try:
            tu = read_with_mtime_cache(tu_path, _load_json)
            target_ids = [str(x) for x in (tu.get("target_ids") or [])]
            elementary_ids = [str(x) for x in (tu.get("elementary_ids") or [])]
        except Exception as exc:
            warnings.append(f"Could not read trade_universe.json: {exc!r}")
    else:
        warnings.append(
            "trade_universe.json missing — all nodes classified as "
            "'elementary'. Re-run the eval pipeline to stage this file."
        )

    node_ordering = target_ids + elementary_ids
    target_set = set(target_ids)

    nodes: List[TradeGraphNode] = []
    for i in range(n):
        if i < len(node_ordering):
            tid = node_ordering[i]
            ttype = "target" if tid in target_set else "elementary"
        else:
            tid = f"{cluster_id}_trade_{i}"
            ttype = "elementary"
            warnings.append(
                f"Node index {i} out of trade-universe range "
                f"({len(node_ordering)}); synthetic id assigned."
            )
        nodes.append(
            TradeGraphNode(
                trade_id=tid, cluster_id=cluster_id, trade_type=ttype,
            )
        )

    edges: List[TradeGraphEdge] = []
    if indices.size > 0 and indices.ndim == 2 and indices.shape[0] == 2:
        src_arr = indices[0]
        dst_arr = indices[1]
        for k in range(src_arr.shape[0]):
            i, j = int(src_arr[k]), int(dst_arr[k])
            if i == j:
                continue
            if i >= n or j >= n:
                continue
            edges.append(
                TradeGraphEdge(
                    source=nodes[i].trade_id,
                    target=nodes[j].trade_id,
                    weight=float(values[k]),
                )
            )

    n_edges_total = int(values.size)
    density = float(n_edges_total / (n * n)) if n > 0 else 0.0
    mean_w = float(np.mean(values)) if values.size > 0 else 0.0

    n_target = sum(1 for node in nodes if node.trade_type == "target")
    n_elem = len(nodes) - n_target

    dedup_warnings = sorted(set(warnings)) if warnings else None

    return TradeGraphResponse(
        cluster_id=cluster_id,
        n_target_trades=n_target,
        n_elementary_trades=n_elem,
        stats=TradeGraphStats(
            n_nodes=n,
            n_edges=n_edges_total,
            density=density,
            mean_weight=mean_w,
        ),
        nodes=nodes,
        edges=edges,
        warnings=dedup_warnings,
    )
```

### B.6 `src/rade_ml_pt/ensemble/api/routers/trade_graph.py` — CREATE

```python
"""``/prism/v1/trade-graph`` — per-cluster trade graph (nodes + edges)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from src.rade_ml_pt.ensemble.api.dependencies import get_reader
from src.rade_ml_pt.ensemble.api.models.trade_graph import TradeGraphResponse
from src.rade_ml_pt.ensemble.api.services.reader import ArtifactReader

router = APIRouter(prefix="/prism/v1", tags=["trade-graph"])


@router.get("/trade-graph", response_model=TradeGraphResponse)
def get_trade_graph(
    cluster_id: str = Query(
        ...,
        description="Cluster to render. Required — the endpoint is cluster-scoped.",
    ),
    reader: ArtifactReader = Depends(get_reader),
) -> TradeGraphResponse:
    try:
        return reader.trade_graph(cluster_id=cluster_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
```

### B.7 `src/rade_ml_pt/ensemble/api/app.py` — modify

Add the import next to the other router imports:

```python
from src.rade_ml_pt.ensemble.api.routers.trade_graph import (
    router as trade_graph_router,
)
```

Register it alongside the others:

```python
app.include_router(graph_stats_router)
app.include_router(trade_graph_router)     # ← new
app.include_router(quality_router)
app.include_router(predictions_router)
```

### B.8 `src/rade_ml_pt/ensemble/api/client.py` — modify

Add the import:

```python
from src.rade_ml_pt.ensemble.api.models.trade_graph import TradeGraphResponse
```

Add this method to `RadeApiClient` (next to `graph_stats`):

```python
def trade_graph(self, *, cluster_id: str) -> TradeGraphResponse:
    """Nodes + edges for one cluster's trade graph."""
    return TradeGraphResponse(
        **self._get_json(
            "/prism/v1/trade-graph",
            {"cluster_id": cluster_id},
        )
    )
```

### B.9 `src/ui/apps/rade_analytics/data/backend.py` — modify

Add the import:

```python
from src.rade_ml_pt.ensemble.api.models.trade_graph import TradeGraphResponse
```

In `_bind_cached_methods` add the cache binding next to the others:

```python
self._trade_graph_cached = cache.memoize(timeout=ttl)(
    self._fetch_trade_graph
)
```

Add the private fetcher and public wrapper (near the other methods):

```python
def _fetch_trade_graph(self, cluster_id: str) -> TradeGraphResponse:
    return self._client.trade_graph(cluster_id=cluster_id)

def trade_graph(
    self, *, cluster_id: str,
) -> BackendResult[TradeGraphResponse]:
    """Full per-cluster trade graph (nodes + edges + summary stats)."""
    return self._wrap(self._trade_graph_cached, cluster_id)
```

### B.10 `src/ui/apps/rade_analytics/data/session.py` — modify

Bump the schema version:

```python
SESSION_SCHEMA_VERSION = 4
```

Add module-level constants next to the other evaluation defaults:

```python
EVALUATION_TRADE_GRAPH_LAYOUTS = ("cose", "concentric", "circle", "grid", "breadthfirst")
DEFAULT_TRADE_GRAPH_LAYOUT = "cose"
DEFAULT_TRADE_GRAPH_WEIGHT_THRESHOLD = 0.0
```

Extend the `EvaluationState` dataclass with these fields:

```python
trade_graph_cluster_id:        Optional[str] = None
trade_graph_layout:            str = DEFAULT_TRADE_GRAPH_LAYOUT
trade_graph_weight_threshold:  float = DEFAULT_TRADE_GRAPH_WEIGHT_THRESHOLD
trade_graph_selected_trade_id: Optional[str] = None
```

Update `EvaluationState.to_dict` / `from_dict` to round-trip the new
fields (guard the threshold with a `float()` cast so stale stores with
missing / non-numeric values still deserialise), and add the new
constants to the module `__all__`.

### B.11 `src/ui/apps/rade_analytics/figures/graph_charts.py` — CREATE

```python
"""Figures supporting the Evaluation → Trade-Graph sub-tab."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ._theme import empty_figure, rade_layout, rgba


_PRIMARY = "#8b5cf6"
_MUTED = "#475569"
_POSITIVE = "#10b981"


def density_distribution(
    graph_stats: pd.DataFrame,
    *,
    selected_cluster_id: Optional[str] = None,
    nbins: int = 18,
) -> go.Figure:
    """Histogram of per-cluster graph density."""
    if graph_stats is None or graph_stats.empty or "density" not in graph_stats:
        return empty_figure("No graph stats available.")

    densities = graph_stats["density"].astype(float).to_numpy()
    if densities.size == 0:
        return empty_figure("No graph stats available.")

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=densities,
            nbinsx=nbins,
            marker={
                "color":  rgba(_PRIMARY, 0.45),
                "line":   {"color": _PRIMARY, "width": 1},
            },
            hovertemplate="Density ≈ %{x:.3g}<br>Clusters: %{y}<extra></extra>",
            name="Clusters",
        )
    )

    if selected_cluster_id and "cluster_id" in graph_stats.columns:
        row = graph_stats[graph_stats["cluster_id"] == selected_cluster_id]
        if not row.empty:
            sel_density = float(row.iloc[0]["density"])
            fig.add_vline(
                x=sel_density,
                line_color=_POSITIVE,
                line_width=2,
                annotation_text=f"{selected_cluster_id}<br>{sel_density:.3g}",
                annotation_position="top right",
                annotation_font={"color": _POSITIVE, "size": 10},
            )

    fig.update_layout(
        **rade_layout(
            show_legend=False,
            margin={"l": 44, "r": 16, "t": 16, "b": 40},
            xaxis={"title": {"text": "Density", "standoff": 10}},
            yaxis={"title": {"text": "Clusters", "standoff": 10}},
        ),
        bargap=0.08,
    )
    return fig


def edges_vs_nodes_scatter(
    graph_stats: pd.DataFrame,
    *,
    selected_cluster_id: Optional[str] = None,
) -> go.Figure:
    """Per-cluster scatter of node-count vs edge-count."""
    required = {"cluster_id", "n_nodes", "n_edges", "density", "mean_weight"}
    if (
        graph_stats is None
        or graph_stats.empty
        or not required.issubset(graph_stats.columns)
    ):
        return empty_figure("No graph stats available.")

    df = graph_stats.copy()
    df["n_nodes"] = df["n_nodes"].astype(float)
    df["n_edges"] = df["n_edges"].astype(float)
    df["density"] = df["density"].astype(float)
    df["mean_weight"] = df["mean_weight"].astype(float)

    d_min, d_max = df["density"].min(), df["density"].max()
    if np.isclose(d_max, d_min):
        sizes = np.full(len(df), 18.0)
    else:
        sizes = 10.0 + (df["density"] - d_min) / (d_max - d_min) * 22.0

    hover_text = [
        (
            f"<b>{row.cluster_id}</b><br>"
            f"Nodes: {int(row.n_nodes)}<br>"
            f"Edges: {int(row.n_edges)}<br>"
            f"Density: {row.density:.3g}<br>"
            f"Mean weight: {row.mean_weight:.3g}"
        )
        for row in df.itertuples()
    ]

    line_widths = [
        3 if cid == selected_cluster_id else 1
        for cid in df["cluster_id"]
    ]
    line_colors = [
        _POSITIVE if cid == selected_cluster_id else "rgba(15, 23, 42, 0.8)"
        for cid in df["cluster_id"]
    ]

    fig = go.Figure(
        data=go.Scatter(
            x=df["n_nodes"],
            y=df["n_edges"],
            mode="markers",
            marker={
                "size":      sizes,
                "color":     df["mean_weight"],
                "colorscale": "Viridis",
                "showscale": True,
                "colorbar": {
                    "title":      {"text": "Mean weight", "side": "right"},
                    "thickness":  10,
                    "len":        0.9,
                    "tickfont":   {"color": "#94a3b8", "size": 10},
                    "title_font": {"color": "#94a3b8", "size": 11},
                },
                "line": {
                    "color": line_colors,
                    "width": line_widths,
                },
            },
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            customdata=df["cluster_id"],
        )
    )

    fig.update_layout(
        **rade_layout(
            show_legend=False,
            margin={"l": 48, "r": 72, "t": 16, "b": 40},
            xaxis={"title": {"text": "Nodes (n)", "standoff": 10}},
            yaxis={"title": {"text": "Edges (n)", "standoff": 10}},
        ),
    )
    return fig


__all__ = ["density_distribution", "edges_vs_nodes_scatter"]
```

### B.12 `src/ui/apps/rade_analytics/figures/__init__.py` — modify

Add the import and extend `__all__`:

```python
from .graph_charts import density_distribution, edges_vs_nodes_scatter
```

```python
__all__ = [
    # … existing entries …
    "density_distribution",
    "edges_vs_nodes_scatter",
]
```

### B.13 `src/ui/apps/rade_analytics/layouts/evaluation/trade_graph.py` — REPLACE

Replace the stub's entire contents with the file at
`src/ui/apps/rade_analytics/layouts/evaluation/trade_graph.py` in this
repo (see the working copy for the full module).  Key attachment points
the callbacks module depends on:

* `TRADE_GRAPH_IDS` dict — every id the callbacks reference.
* `CYTOSCAPE_STYLESHEET` — violet elementary / amber target node styling.
* `build_trade_graph()` — public builder returning the full sub-tab div
  with the sticky header band, Cytoscape pane + side-panel cards, and
  the secondary-chart row.

### B.14 `src/ui/apps/rade_analytics/callbacks/trade_graph_cb.py` — CREATE

Copy the full contents of `src/ui/apps/rade_analytics/callbacks/trade_graph_cb.py`
from the working tree.  The module exposes a single `register(app, backend)`
entry point and is wired by Appendix B.15.

Six callbacks are registered:

1. `_hydrate` — on `/evaluation/trade-graph` entry, populate the cluster
   `Select`, initialise layout radio + threshold slider from the session.
2. `_sync` — cluster picker / layout radio / threshold slider →
   `session.evaluation.trade_graph_*`.  Changing the cluster clears the
   selected-trade id.
3. `_render_graph` — fetches the trade-graph payload once per cluster,
   stashes nodes / edges in a memory store, writes cluster-stats card +
   pane status.
4. `_apply` — threshold slider re-filters stored edges on the client side
   (sub-60 ms even on thousands of edges) and updates the threshold label.
5. `_on_tap` — `tapNodeData` → Selected-Trade card + session write;
   attribute rows come from `clusters_df(cluster_id=…)`.
6. `_render` (ensemble) — builds the Ensemble-summary KPIs and both
   secondary charts from `graph_stats_df()`.

### B.15 `src/ui/apps/rade_analytics/callbacks/__init__.py` — modify

```python
from . import (
    evaluation_cb,
    overview_cb,
    portfolio_cb,
    splash_cb,
    trade_graph_cb,
)
```

And in `register_all`, append:

```python
trade_graph_cb.register(app, backend)
```

### B.16 `examples/rade_analytics/_mock_backend.py` — modify

Add imports:

```python
from src.rade_ml_pt.ensemble.api.models.trade_graph import (
    TradeGraphEdge,
    TradeGraphNode,
    TradeGraphResponse,
    TradeGraphStats,
)
```

Add these two methods inside `MockRadeBackend` (after
`cluster_timeseries_df`):

```python
def graph_stats_df(
    self, *, cluster_id: Optional[str] = None,
) -> BackendResult[pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    for cid in self._cluster_ids:
        payload = self._trade_graph_payload(cid)
        rows.append(
            {
                "cluster_id":  cid,
                "n_nodes":     int(payload.stats.n_nodes),
                "n_edges":     int(payload.stats.n_edges),
                "density":     float(payload.stats.density),
                "mean_weight": float(payload.stats.mean_weight),
            }
        )
    df = pd.DataFrame(rows)
    if cluster_id:
        df = df[df["cluster_id"] == cluster_id].reset_index(drop=True)
    return BackendResult.success(df)

def trade_graph(
    self, *, cluster_id: str,
) -> BackendResult[TradeGraphResponse]:
    if cluster_id not in self._cluster_ids:
        return BackendResult.failure(
            error=f"unknown cluster '{cluster_id}'",
            status_code=404,
        )
    return BackendResult.success(self._trade_graph_payload(cluster_id))

def _trade_graph_payload(self, cluster_id: str) -> TradeGraphResponse:
    return _trade_graph_cached(
        mock_id=id(self),
        cluster_id=cluster_id,
        seed=self._seed,
    )
```

And add the module-level seeded generator at the bottom of the file,
**above** the `__all__` assignment:

```python
_TRADE_NODE_MIN = 40
_TRADE_NODE_MAX = 90
_TARGET_FRACTION = 0.2


@lru_cache(maxsize=256)
def _trade_graph_cached(
    *,
    mock_id:    int,
    cluster_id: str,
    seed:       int,
) -> TradeGraphResponse:
    del mock_id
    rng = np.random.default_rng(seed + (abs(hash(cluster_id)) % 10_000))

    n = int(rng.integers(_TRADE_NODE_MIN, _TRADE_NODE_MAX + 1))
    n_target = max(1, int(round(n * _TARGET_FRACTION)))
    n_elem = n - n_target

    target_ids = [f"{cluster_id}_T{i:03d}" for i in range(n_target)]
    elem_ids = [f"{cluster_id}_E{i:03d}" for i in range(n_elem)]

    nodes: List[TradeGraphNode] = []
    for tid in target_ids:
        nodes.append(TradeGraphNode(trade_id=tid, cluster_id=cluster_id, trade_type="target"))
    for tid in elem_ids:
        nodes.append(TradeGraphNode(trade_id=tid, cluster_id=cluster_id, trade_type="elementary"))

    n_edges_target = max(n, 2 * n)
    edges: List[TradeGraphEdge] = []
    seen: set[tuple[int, int]] = set()
    tries = 0
    while len(edges) < n_edges_target and tries < n_edges_target * 4:
        tries += 1
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i == j:
            continue
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        weight = float(rng.uniform(0.05, 1.0))
        edges.append(
            TradeGraphEdge(
                source=nodes[i].trade_id,
                target=nodes[j].trade_id,
                weight=weight,
            )
        )

    density = float(len(edges) / (n * n)) if n > 0 else 0.0
    mean_w = float(np.mean([e.weight for e in edges])) if edges else 0.0

    return TradeGraphResponse(
        cluster_id=cluster_id,
        n_target_trades=n_target,
        n_elementary_trades=n_elem,
        stats=TradeGraphStats(
            n_nodes=n,
            n_edges=len(edges),
            density=density,
            mean_weight=mean_w,
        ),
        nodes=nodes,
        edges=edges,
        warnings=None,
    )
```

### B.17 `examples/rade_analytics/08_trade_graph_preview_live.py` — CREATE

Mirrors `06_portfolio_preview_live.py` but on port **8054** with
`title="Rade — Trade-Graph live preview (mock)"`.  Open
[http://localhost:8054/evaluation/trade-graph](http://localhost:8054/evaluation/trade-graph)
after launch.

### B.18 Smoke test

From the project root, with the venv active:

```bash
pip install -r requirements-rade-ui.txt
python examples/rade_analytics/08_trade_graph_preview_live.py
```

Visit `http://localhost:8054/evaluation/trade-graph`.  Expected:

* Cluster `Select` populated with 12 synthetic clusters.
* Graph renders ~50-90 nodes, a mix of amber (target) and violet
  (elementary) dots.
* Dragging "Min weight" above 0 thins out edges without a server hit.
* Switching layout (`cose` → `concentric` → …) re-runs Cytoscape's
  client-side layout.
* Tapping any node fills the Selected-Trade card with the trade id,
  cluster badge, trade type + asset / currency / desk / product rows;
  the "Open in Cluster Deep Dive" button becomes enabled.
* Right-hand Cluster stats and Ensemble summary reflect the fetched
  payload and the full `graph_stats` frame respectively.
* Density histogram + Edges-vs-nodes scatter both highlight the
  selected cluster (emerald vline / thicker ring).

Once live data is available, rerun the eval pipeline (picks up
`_copy_member_graph_artifacts`) and swap `MockRadeBackend` for the real
`RadeBackend` — no other code changes required.

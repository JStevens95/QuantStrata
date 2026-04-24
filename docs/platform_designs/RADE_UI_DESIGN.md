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

## Appendix A — `src/ui/apps/rade_analytics/callbacks/trade_graph_cb.py`

CREATE this file (full contents below) and register it in
`src/ui/apps/rade_analytics/callbacks/__init__.py`:

```python
from . import (
    evaluation_cb,
    overview_cb,
    portfolio_cb,
    splash_cb,
    trade_graph_cb,
)
```

```python
# …
register_router(app, backend)
splash_cb.register(app, backend)
overview_cb.register(app, backend)
evaluation_cb.register(app, backend)
portfolio_cb.register(app, backend)
trade_graph_cb.register(app, backend)
```

Then paste the full module:

```python
"""Evaluation → Trade-Graph sub-tab callbacks (Phase E.3).

Six callbacks drive the tab:

1. ``_hydrate_cluster_options``   — on entry to ``/evaluation/trade-graph``
   populate the cluster ``Select`` from ``RadeBackend.clusters_df``.
2. ``_sync_header_to_session``    — cluster picker / layout radio /
   threshold slider → ``session.evaluation.trade_graph_*``.  Clearing
   the cluster also wipes the selected-trade id.
3. ``_render_graph``              — session change → Cytoscape elements,
   pane status, cluster-stats card.  Fetches the trade-graph payload
   once per cluster and stashes nodes / edges in a memory store so
   threshold changes can re-filter without another API hit.
4. ``_apply_threshold``           — threshold slider drags the stored
   nodes / edges through a ``weight >= threshold`` filter.  Pure
   client-side, so the interaction stays sub-60ms even on thousands
   of edges.
5. ``_render_node_tap``           — Cytoscape ``tapNodeData`` →
   Selected-Trade card + session's ``trade_graph_selected_trade_id``.
6. ``_render_ensemble_context``   — graph_stats DataFrame feeds the
   Ensemble-summary KPIs and the two secondary charts.

Every fetch goes through :class:`RadeBackend`.  When the cluster has no
graph staged (older runs without ``graph_results.joblib``) we degrade
gracefully to empty states — the rest of the UI keeps working.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import pandas as pd
from dash import Input, Output, State, ctx, html, no_update
from dash.exceptions import PreventUpdate

from ..data.session import (
    DEFAULT_TRADE_GRAPH_LAYOUT,
    EVALUATION_TRADE_GRAPH_LAYOUTS,
    Session,
)
from ..figures import density_distribution, edges_vs_nodes_scatter
from ..layouts.evaluation.trade_graph import TRADE_GRAPH_IDS
from ..layouts.shell import SHELL_IDS

if TYPE_CHECKING:
    from dash import Dash

    from ..data.backend import RadeBackend


logger = logging.getLogger(__name__)


_TRADE_GRAPH_PATH = "/evaluation/trade-graph"
_PLACEHOLDER = "—"


# ─────────────────────────────────────────────────────────────────────
# Formatters — kept small so the callback bodies stay readable.
# ─────────────────────────────────────────────────────────────────────


def _fmt_int(x: Optional[float]) -> str:
    if x is None:
        return _PLACEHOLDER
    try:
        return f"{int(x):,}"
    except (TypeError, ValueError):
        return _PLACEHOLDER


def _fmt_float(x: Optional[float], *, precision: int = 3) -> str:
    if x is None:
        return _PLACEHOLDER
    try:
        return f"{float(x):.{precision}g}"
    except (TypeError, ValueError):
        return _PLACEHOLDER


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────


def register(app: "Dash", backend: "RadeBackend") -> None:
    """Attach every Trade-Graph-sub-tab callback to ``app``."""
    _register_hydrate_cluster_options(app, backend)
    _register_sync_header_to_session(app)
    _register_render_graph(app, backend)
    _register_apply_threshold(app)
    _register_render_node_tap(app, backend)
    _register_render_ensemble_context(app, backend)


# ═════════════════════════════════════════════════════════════════════
# 1. Hydrate cluster Select options on URL entry
# ═════════════════════════════════════════════════════════════════════


def _register_hydrate_cluster_options(
    app: "Dash", backend: "RadeBackend",
) -> None:
    @app.callback(
        Output(TRADE_GRAPH_IDS["cluster_select"], "data"),
        Output(TRADE_GRAPH_IDS["cluster_select"], "value"),
        Output(TRADE_GRAPH_IDS["layout_radio"],   "value"),
        Output(TRADE_GRAPH_IDS["threshold_slider"], "value"),
        Input(SHELL_IDS["url"],                   "pathname"),
        State(SHELL_IDS["session_store"],         "data"),
    )
    def _hydrate(
        pathname:     Optional[str],
        session_data: Optional[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, str]], Optional[str], str, float]:
        if pathname != _TRADE_GRAPH_PATH:
            raise PreventUpdate

        session = Session.from_store(session_data)
        res = backend.clusters_df()
        if not res.ok or res.data is None or res.data.empty:
            return [], None, session.evaluation.trade_graph_layout, session.evaluation.trade_graph_weight_threshold

        df = res.data
        options = [
            {"value": cid, "label": cid} for cid in sorted(df["cluster_id"].unique())
        ]

        # Prefer the session override, fall back to the top-bar cluster,
        # then the first cluster in the list.  Never return ``None`` when
        # we have clusters — the graph pane is useless without one.
        session_cluster = (
            session.evaluation.trade_graph_cluster_id
            or session.cluster_id
            or (options[0]["value"] if options else None)
        )
        if session_cluster not in {o["value"] for o in options}:
            session_cluster = options[0]["value"] if options else None

        return (
            options,
            session_cluster,
            session.evaluation.trade_graph_layout,
            session.evaluation.trade_graph_weight_threshold,
        )


# ═════════════════════════════════════════════════════════════════════
# 2. Header band → session
# ═════════════════════════════════════════════════════════════════════


def _register_sync_header_to_session(app: "Dash") -> None:
    @app.callback(
        Output(SHELL_IDS["session_store"],             "data", allow_duplicate=True),
        Input(TRADE_GRAPH_IDS["cluster_select"],       "value"),
        Input(TRADE_GRAPH_IDS["layout_radio"],         "value"),
        Input(TRADE_GRAPH_IDS["threshold_slider"],     "value"),
        State(SHELL_IDS["session_store"],              "data"),
        prevent_initial_call=True,
    )
    def _sync(
        cluster:     Optional[str],
        layout_name: Optional[str],
        threshold:   Optional[float],
        session_data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        trigger = ctx.triggered_id
        if trigger is None:
            raise PreventUpdate

        session = Session.from_store(session_data)
        ev = session.evaluation
        changed = False

        if trigger == TRADE_GRAPH_IDS["cluster_select"]:
            new_cluster = cluster if cluster else None
            if ev.trade_graph_cluster_id != new_cluster:
                ev.trade_graph_cluster_id = new_cluster
                # Changing the cluster invalidates the selected trade.
                ev.trade_graph_selected_trade_id = None
                changed = True

        elif trigger == TRADE_GRAPH_IDS["layout_radio"]:
            if layout_name in EVALUATION_TRADE_GRAPH_LAYOUTS and ev.trade_graph_layout != layout_name:
                ev.trade_graph_layout = layout_name
                changed = True

        elif trigger == TRADE_GRAPH_IDS["threshold_slider"]:
            try:
                new_threshold = max(0.0, float(threshold))
            except (TypeError, ValueError):
                raise PreventUpdate
            if abs(ev.trade_graph_weight_threshold - new_threshold) > 1e-9:
                ev.trade_graph_weight_threshold = new_threshold
                changed = True

        if not changed:
            raise PreventUpdate
        return session.to_store()


# ═════════════════════════════════════════════════════════════════════
# 3. Render graph — elements + cluster-stats card
# ═════════════════════════════════════════════════════════════════════


def _register_render_graph(
    app: "Dash", backend: "RadeBackend",
) -> None:
    @app.callback(
        Output(TRADE_GRAPH_IDS["store_nodes_edges"],   "data"),
        Output(TRADE_GRAPH_IDS["cytoscape"],           "layout"),
        Output(TRADE_GRAPH_IDS["pane_status"],         "children"),
        Output(TRADE_GRAPH_IDS["cluster_stats"],       "children"),
        Input(SHELL_IDS["url"],                        "pathname"),
        Input(SHELL_IDS["session_store"],              "data"),
    )
    def _render(
        pathname:     Optional[str],
        session_data: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Dict[str, Any], str, List[Any]]:
        if pathname != _TRADE_GRAPH_PATH:
            raise PreventUpdate

        session = Session.from_store(session_data)
        ev = session.evaluation
        layout_cfg = {
            "name":    ev.trade_graph_layout or DEFAULT_TRADE_GRAPH_LAYOUT,
            "animate": False,
        }
        cluster_id = ev.trade_graph_cluster_id or session.cluster_id

        empty_stats = _cluster_stats_children(None)
        if not cluster_id:
            return (
                {},
                layout_cfg,
                "No cluster selected.",
                empty_stats,
            )

        res = backend.trade_graph(cluster_id=cluster_id)
        if not res.ok or res.data is None:
            logger.info(
                "trade-graph fetch failed for cluster '%s': %s",
                cluster_id, res.error,
            )
            return (
                {},
                layout_cfg,
                f"Graph unavailable for {cluster_id}.",
                empty_stats,
            )

        payload = res.data
        # Serialise to plain dicts for the dcc.Store and the later
        # threshold filter.  Keeping the shape Cytoscape-ready avoids a
        # second transform at render time.
        nodes_payload = [
            {
                "data": {
                    "id":         n.trade_id,
                    "trade_type": n.trade_type,
                    "cluster_id": n.cluster_id,
                },
            }
            for n in payload.nodes
        ]
        edges_payload = [
            {
                "data": {
                    "source": e.source,
                    "target": e.target,
                    "weight": float(e.weight),
                },
            }
            for e in payload.edges
        ]

        status = (
            f"{payload.cluster_id} · "
            f"{payload.stats.n_nodes:,} nodes · "
            f"{payload.stats.n_edges:,} edges "
            f"({payload.n_target_trades} target / "
            f"{payload.n_elementary_trades} elementary)"
        )
        stats_children = _cluster_stats_children(
            {
                "n_nodes":     payload.stats.n_nodes,
                "n_edges":     payload.stats.n_edges,
                "density":     payload.stats.density,
                "mean_weight": payload.stats.mean_weight,
            }
        )

        return (
            {"nodes": nodes_payload, "edges": edges_payload},
            layout_cfg,
            status,
            stats_children,
        )


def _cluster_stats_children(stats: Optional[Dict[str, Any]]) -> List[Any]:
    """Rebuild the cluster-stats card children with fresh numbers.

    Imported lazily to avoid a cyclic import between callbacks and the
    layout module (layouts import components; callbacks import layouts).
    """
    from ..components.kpi_card import KpiCard   # local to avoid cycle

    caption = html.Span(
        "Cluster stats",
        className="text-xs uppercase tracking-wider text-slate-400",
    )
    if stats is None:
        grid_children = [
            KpiCard(label="Nodes",       value=_PLACEHOLDER),
            KpiCard(label="Edges",       value=_PLACEHOLDER),
            KpiCard(label="Density",     value=_PLACEHOLDER),
            KpiCard(label="Mean weight", value=_PLACEHOLDER),
        ]
    else:
        grid_children = [
            KpiCard(label="Nodes",       value=_fmt_int(stats.get("n_nodes"))),
            KpiCard(label="Edges",       value=_fmt_int(stats.get("n_edges"))),
            KpiCard(
                label="Density",
                value=_fmt_float(stats.get("density")),
            ),
            KpiCard(
                label="Mean weight",
                value=_fmt_float(stats.get("mean_weight")),
            ),
        ]
    return [
        caption,
        html.Div(
            className="grid grid-cols-2 gap-2",
            children=grid_children,
        ),
    ]


# ═════════════════════════════════════════════════════════════════════
# 4. Threshold filter — stored nodes / edges → Cytoscape elements
# ═════════════════════════════════════════════════════════════════════


def _register_apply_threshold(app: "Dash") -> None:
    @app.callback(
        Output(TRADE_GRAPH_IDS["cytoscape"],        "elements"),
        Output(TRADE_GRAPH_IDS["threshold_value_label"], "children"),
        Input(TRADE_GRAPH_IDS["store_nodes_edges"], "data"),
        Input(TRADE_GRAPH_IDS["threshold_slider"],  "value"),
    )
    def _apply(
        store_data: Optional[Dict[str, Any]],
        threshold:  Optional[float],
    ) -> Tuple[List[Dict[str, Any]], str]:
        try:
            thr = max(0.0, float(threshold)) if threshold is not None else 0.0
        except (TypeError, ValueError):
            thr = 0.0

        label = f"{thr:.2f}"

        if not store_data:
            return [], label

        nodes = store_data.get("nodes") or []
        edges = store_data.get("edges") or []

        if thr > 0.0:
            edges = [
                e for e in edges
                if float(e.get("data", {}).get("weight", 0.0)) >= thr
            ]

        return [*nodes, *edges], label


# ═════════════════════════════════════════════════════════════════════
# 5. Node-tap → Selected-Trade card + session
# ═════════════════════════════════════════════════════════════════════


def _register_render_node_tap(
    app: "Dash", backend: "RadeBackend",
) -> None:
    @app.callback(
        Output(TRADE_GRAPH_IDS["selected_trade_id"],   "children"),
        Output(TRADE_GRAPH_IDS["selected_cluster_chip"], "children"),
        Output(TRADE_GRAPH_IDS["selected_attrs"],       "children"),
        Output(TRADE_GRAPH_IDS["selected_deep_dive_btn"], "disabled"),
        Output(SHELL_IDS["session_store"],             "data", allow_duplicate=True),
        Input(TRADE_GRAPH_IDS["cytoscape"],            "tapNodeData"),
        State(SHELL_IDS["session_store"],              "data"),
        prevent_initial_call=True,
    )
    def _on_tap(
        node_data:    Optional[Dict[str, Any]],
        session_data: Optional[Dict[str, Any]],
    ) -> Tuple[Any, Any, Any, bool, Any]:
        if not node_data:
            raise PreventUpdate

        trade_id = node_data.get("id")
        cluster_id = node_data.get("cluster_id")
        trade_type = node_data.get("trade_type", "elementary")

        if not trade_id or not cluster_id:
            raise PreventUpdate

        session = Session.from_store(session_data)
        if session.evaluation.trade_graph_selected_trade_id == trade_id:
            # Re-tap is a no-op — no need to write the store.
            session_out: Any = no_update
        else:
            session.evaluation.trade_graph_selected_trade_id = trade_id
            session_out = session.to_store()

        # Attribute lookup — clusters_df carries the attributes we need
        # (asset_class, currency, desk, product).  One cached call, so
        # repeated node taps pay nothing extra.
        attrs_children = _build_attrs_children(backend, cluster_id, trade_type)

        return (
            trade_id,
            cluster_id,
            attrs_children,
            False,
            session_out,
        )


def _build_attrs_children(
    backend: "RadeBackend", cluster_id: str, trade_type: str,
) -> Any:
    """Render the attribute list inside the Selected-Trade card."""
    res = backend.clusters_df(cluster_id=cluster_id)
    rows: List[Tuple[str, str]] = [("Trade type", trade_type.capitalize())]

    if res.ok and res.data is not None and not res.data.empty:
        row = res.data.iloc[0]
        for label, column in (
            ("Asset class", "asset_class"),
            ("Currency",    "currency_code"),
            ("Desk",        "desk"),
            ("Product",     "product_code"),
        ):
            if column in res.data.columns:
                value = row.get(column)
                if value is not None and not pd.isna(value):
                    rows.append((label, str(value)))

    return html.Div(
        className="flex flex-col gap-1 mt-2",
        children=[
            html.Div(
                className="flex items-center justify-between text-xs",
                children=[
                    html.Span(label, className="text-slate-500"),
                    html.Span(value, className="text-slate-200"),
                ],
            )
            for label, value in rows
        ],
    )


# ═════════════════════════════════════════════════════════════════════
# 6. Ensemble context — summary KPIs + two secondary charts
# ═════════════════════════════════════════════════════════════════════


def _register_render_ensemble_context(
    app: "Dash", backend: "RadeBackend",
) -> None:
    @app.callback(
        Output(TRADE_GRAPH_IDS["ensemble_stats"],       "children"),
        Output(TRADE_GRAPH_IDS["density_chart"],        "figure"),
        Output(TRADE_GRAPH_IDS["edges_vs_nodes_chart"], "figure"),
        Input(SHELL_IDS["url"],                         "pathname"),
        Input(SHELL_IDS["session_store"],               "data"),
    )
    def _render(
        pathname:     Optional[str],
        session_data: Optional[Dict[str, Any]],
    ) -> Tuple[List[Any], Any, Any]:
        if pathname != _TRADE_GRAPH_PATH:
            raise PreventUpdate

        session = Session.from_store(session_data)
        selected_cluster = (
            session.evaluation.trade_graph_cluster_id or session.cluster_id
        )

        res = backend.graph_stats_df()
        if not res.ok or res.data is None or res.data.empty:
            return (
                _ensemble_stats_children(None),
                density_distribution(pd.DataFrame(),
                                     selected_cluster_id=selected_cluster),
                edges_vs_nodes_scatter(pd.DataFrame(),
                                       selected_cluster_id=selected_cluster),
            )

        df = res.data.copy()
        # Guard against any row where the cluster has no graph (n_nodes
        # == 0).  We include those in the histogram (user should see the
        # zero bar) but exclude them from the avg-density / mean-weight
        # averages so they don't drag the summary to zero.
        has_graph = df["n_nodes"] > 0
        n_clusters = int(len(df))
        total_edges = int(df["n_edges"].sum())
        avg_density = (
            float(df.loc[has_graph, "density"].mean()) if has_graph.any() else 0.0
        )
        avg_mean_weight = (
            float(df.loc[has_graph, "mean_weight"].mean()) if has_graph.any() else 0.0
        )

        stats_payload = {
            "n_clusters":      n_clusters,
            "total_edges":     total_edges,
            "avg_density":     avg_density,
            "avg_mean_weight": avg_mean_weight,
        }

        return (
            _ensemble_stats_children(stats_payload),
            density_distribution(df, selected_cluster_id=selected_cluster),
            edges_vs_nodes_scatter(df, selected_cluster_id=selected_cluster),
        )


def _ensemble_stats_children(stats: Optional[Dict[str, Any]]) -> List[Any]:
    """Rebuild the Ensemble-summary card children with fresh numbers."""
    from ..components.kpi_card import KpiCard   # local to avoid cycle

    caption = html.Span(
        "Ensemble summary",
        className="text-xs uppercase tracking-wider text-slate-400",
    )
    if stats is None:
        grid_children = [
            KpiCard(label="Total clusters",  value=_PLACEHOLDER),
            KpiCard(label="Total edges",     value=_PLACEHOLDER),
            KpiCard(label="Avg density",     value=_PLACEHOLDER),
            KpiCard(label="Avg mean weight", value=_PLACEHOLDER),
        ]
    else:
        grid_children = [
            KpiCard(label="Total clusters",
                    value=_fmt_int(stats.get("n_clusters"))),
            KpiCard(label="Total edges",
                    value=_fmt_int(stats.get("total_edges"))),
            KpiCard(label="Avg density",
                    value=_fmt_float(stats.get("avg_density"))),
            KpiCard(label="Avg mean weight",
                    value=_fmt_float(stats.get("avg_mean_weight"))),
        ]
    return [
        caption,
        html.Div(
            className="grid grid-cols-2 gap-2",
            children=grid_children,
        ),
    ]


__all__ = ["register"]
```

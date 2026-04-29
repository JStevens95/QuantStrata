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

## Appendix A — `callbacks/trade_graph_cb.py`

Full callback module for the **Trade-Graph sub-tab** (Phase E.3
rebuild).  Lives at `src/ui/apps/rade_analytics/callbacks/trade_graph_cb.py`;
auto-registered via `callbacks/__init__.py::register_all`.

### What this module owns

| Bucket | Callbacks |
|---|---|
| **Capture** (input → session) | `_register_sync_header_to_session` · `_register_sync_node_tap_to_session` · `_register_sync_neighbours_k_to_session` · `_register_sync_neighbour_click_to_session` |
| **Render** (state → DOM) | `_register_bootstrap` · `_register_render_graph` · `_register_render_selected_card` · `_register_render_neighbours_list` · `_register_render_legend` · `_register_render_threshold_label` · `_register_render_density_chart` · `_register_render_edges_vs_nodes_chart` |
| **Clientside** | `_register_clientside` (wires `fit_view` + `export_png` from `assets/js/trade_graph.js`) |

The single render callback that writes the `store_graph` payload is
`_register_render_graph` — every other render callback reads from
that store rather than re-fetching, so toggling color-by, dragging
the threshold slider or chaining through neighbours never re-hits
the backend.

The bootstrap callback owns the page's only render-side capture-edges
(option-list population + fresh-user-default override + stale-cluster
fallback), per Page Contract §3 Rule L4.  Every other render callback
gates on `pathname == "/evaluation/trade-graph"` and returns `no_update`
otherwise.

### Prerequisites already in place

For this module to import cleanly, the following must be present
(all delivered earlier in the rebuild — listed here as a checklist
before paste):

| File | Symbol(s) consumed |
|---|---|
| `data/session.py` | `Session`, `EVALUATION_TRADE_GRAPH_COLOR_BY`, `EVALUATION_TRADE_GRAPH_LAYOUTS`, `DEFAULT_TRADE_GRAPH_LAYOUT` |
| `layouts/evaluation/trade_graph.py` | `TRADE_GRAPH_IDS`, `COLOR_BY_LABELS` |
| `layouts/shell.py` | `SHELL_IDS` (existing) |
| `figures/__init__.py` | `build_legend_body`, `build_stylesheet`, `density_distribution`, `edges_vs_nodes_scatter`, `empty_figure` |
| `components/kpi_card.py` | `KpiCard` (existing) |
| `assets/js/trade_graph.js` | `window.dash_clientside.trade_graph.fit_view` + `export_png` |
| `data/backend.py` | `RadeBackend` with `clusters_df()`, `trade_graph()`, `trades_df()`, `graph_stats_df()` |

Drop the file in place — `callbacks/__init__.py` already calls
`trade_graph_cb.register(app, backend)` so no further wiring is
required.

```python
"""Evaluation → Trade-Graph sub-tab callbacks (Phase E.3 rebuild).

Page-Contract structure (§2 capture / render split)
---------------------------------------------------
:func:`register` delegates to two section helpers.

Capture callbacks (input gestures → session writes)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* :func:`_register_sync_header_to_session`     — cluster / layout /
  color-by / threshold widgets → session.
* :func:`_register_sync_node_tap_to_session`   — Cytoscape ``tapNodeData``
  → ``session.trade_graph_selected_trade_id``.
* :func:`_register_sync_neighbours_k_to_session` — popover NumberInput
  → ``session.trade_graph_neighbour_k``.
* :func:`_register_sync_neighbour_click_to_session` — pattern-matching
  click on a neighbour row → ``session.trade_graph_selected_trade_id``.
* :func:`_register_sync_deep_dive_button`      — "Open Deep Dive" button
  → writes deep_dive session fields + URL navigation.

Render callbacks (state → DOM, no session writes)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* :func:`_register_bootstrap`                  — mount tripwire fires
  once per fresh mount; populates the cluster Select's ``data`` and
  handles the two narrow capture-edges (URL deep-link, fresh-user
  default).
* :func:`_register_render_graph`               — cluster / threshold /
  color-by / layout → cytoscape ``elements``, mini-map ``elements``,
  ``stylesheet``, pane status, store_graph, cluster-stats grid.
  Only render callback that writes the ``store_graph`` payload.
* :func:`_register_render_selected_card`       — ``store_graph`` +
  selected_trade_id → trade_id text, chip strip, metrics row,
  neighbours-button enabled state, deep-dive button enabled state.
* :func:`_register_render_neighbours_list`     — store_graph + selected
  + k → popover scroll-list children.
* :func:`_register_render_legend`              — color-by + store_graph
  → legend body children.
* :func:`_register_render_threshold_label`     — slider value → label
  (kept here so the slider doesn't leak threshold-state to the rest
  of the app via the session-store on every drag).
* :func:`_register_render_density_chart`       — ensemble graph_stats →
  density distribution figure.
* :func:`_register_render_edges_vs_nodes_chart` — same → edges-vs-nodes
  scatter figure.

Clientside callbacks
~~~~~~~~~~~~~~~~~~~~
* "Fit view" button — clientside ``cy.fit()`` via
  :data:`_FIT_VIEW_CLIENTSIDE`.
* "Export PNG" button — clientside download via
  :data:`_EXPORT_PNG_CLIENTSIDE`.

Why pathname-gating
-------------------
Every render callback returns ``no_update`` (or :class:`PreventUpdate`)
when ``pathname`` isn't ``/evaluation/trade-graph``.  Page Contract §4
Rule C2 — cheap, idempotent, prevents render storms when the user
navigates between sub-tabs.

Why the ``store_graph`` ephemeral store
---------------------------------------
``RadeBackend.trade_graph(cluster_id)`` is the heaviest single fetch
on the page (full node + edge list).  We pay it once per cluster
selection and cache the deserialised payload in a memory store.
Threshold filter, color-by toggle, neighbours popover, selected-card
re-renders all read from the store.
"""
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from dash import (
    ALL,
    ClientsideFunction,
    Input,
    Output,
    State,
    ctx,
    html,
    no_update,
)
from dash.exceptions import PreventUpdate

from ..components.kpi_card import KpiCard
from ..data.session import (
    DEFAULT_TRADE_GRAPH_LAYOUT,
    EVALUATION_TRADE_GRAPH_COLOR_BY,
    EVALUATION_TRADE_GRAPH_LAYOUTS,
    Session,
)
from ..figures import (
    build_legend_body,
    build_stylesheet,
    density_distribution,
    edges_vs_nodes_scatter,
    empty_figure,
)
from ..layouts.evaluation.trade_graph import (
    COLOR_BY_LABELS,
    TRADE_GRAPH_IDS,
)
from ..layouts.shell import SHELL_IDS

if TYPE_CHECKING:
    from dash import Dash

    from ..data.backend import RadeBackend


logger = logging.getLogger(__name__)


_TRADE_GRAPH_PATH = "/evaluation/trade-graph"
_PLACEHOLDER      = "—"


# ─────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────


def register(app: "Dash", backend: "RadeBackend") -> None:
    """Attach every Trade-Graph sub-tab callback to ``app``.

    Mirrors the template_cb.py structure — one capture section, one
    render section, plus the clientside callbacks at the end.
    """
    _register_capture(app)
    _register_render(app, backend)
    _register_clientside(app)


# ─────────────────────────────────────────────────────────────────────
# Section dispatchers
# ─────────────────────────────────────────────────────────────────────


def _register_capture(app: "Dash") -> None:
    _register_sync_header_to_session(app)
    _register_sync_node_tap_to_session(app)
    _register_sync_neighbours_k_to_session(app)
    _register_sync_neighbour_click_to_session(app)
    # Deep-dive button → URL navigation lives in
    # ``cluster_deep_dive_cb._register_navigate_from_trade_graph`` so
    # the navigation logic sits next to the page that consumes the
    # deep-link.  We only own the button's enabled/disabled state
    # (rendered out of ``_register_render_selected_card`` below).


def _register_render(app: "Dash", backend: "RadeBackend") -> None:
    _register_bootstrap(app, backend)
    _register_render_graph(app, backend)
    _register_render_selected_card(app)
    _register_render_neighbours_list(app)
    _register_render_legend(app)
    _register_render_threshold_label(app)
    _register_render_density_chart(app, backend)
    _register_render_edges_vs_nodes_chart(app, backend)


# ═════════════════════════════════════════════════════════════════════
# CAPTURE — input gestures → session writes
# ═════════════════════════════════════════════════════════════════════


def _register_sync_header_to_session(app: "Dash") -> None:
    """Header widgets → session.

    The four header inputs (cluster, layout, color-by, threshold) all
    converge on a single capture callback.  ``ctx.triggered_id`` tells
    us which one fired so we mutate the matching session field and
    leave the others alone.

    Threshold is included here (not in a separate callback) so dragging
    the slider doesn't write to session on every frame — we already use
    ``updatemode="mouseup"`` on the slider to throttle to release.
    """

    @app.callback(
        Output(SHELL_IDS["session_store"],            "data", allow_duplicate=True),
        Input(TRADE_GRAPH_IDS["cluster_select"],      "value"),
        Input(TRADE_GRAPH_IDS["layout_radio"],        "value"),
        Input(TRADE_GRAPH_IDS["color_by_select"],     "value"),
        Input(TRADE_GRAPH_IDS["threshold_slider"],    "value"),
        State(SHELL_IDS["session_store"],             "data"),
        prevent_initial_call=True,
    )
    def _sync(
        cluster:     Optional[str],
        layout_name: Optional[str],
        color_by:    Optional[str],
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
                # Changing cluster invalidates the selected trade — the
                # node id would no longer match anything in the new
                # graph.
                ev.trade_graph_selected_trade_id = None
                changed = True

        elif trigger == TRADE_GRAPH_IDS["layout_radio"]:
            if (
                layout_name in EVALUATION_TRADE_GRAPH_LAYOUTS
                and ev.trade_graph_layout != layout_name
            ):
                ev.trade_graph_layout = layout_name
                changed = True

        elif trigger == TRADE_GRAPH_IDS["color_by_select"]:
            if (
                color_by in EVALUATION_TRADE_GRAPH_COLOR_BY
                and ev.trade_graph_color_by != color_by
            ):
                ev.trade_graph_color_by = color_by
                changed = True

        elif trigger == TRADE_GRAPH_IDS["threshold_slider"]:
            try:
                new_threshold = max(0.0, min(1.0, float(threshold)))
            except (TypeError, ValueError):
                raise PreventUpdate
            if abs(ev.trade_graph_weight_threshold - new_threshold) > 1e-9:
                ev.trade_graph_weight_threshold = new_threshold
                changed = True

        if not changed:
            raise PreventUpdate
        return session.to_store()


def _register_sync_node_tap_to_session(app: "Dash") -> None:
    """Cytoscape node tap → ``session.trade_graph_selected_trade_id``."""

    @app.callback(
        Output(SHELL_IDS["session_store"],     "data", allow_duplicate=True),
        Input(TRADE_GRAPH_IDS["cytoscape"],    "tapNodeData"),
        State(SHELL_IDS["session_store"],      "data"),
        prevent_initial_call=True,
    )
    def _on_tap(
        node_data:    Optional[Dict[str, Any]],
        session_data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not node_data:
            raise PreventUpdate
        trade_id = node_data.get("id")
        if not trade_id:
            raise PreventUpdate

        session = Session.from_store(session_data)
        if session.evaluation.trade_graph_selected_trade_id == trade_id:
            # Re-tap on the same node is a no-op — saves a session round
            # trip + the cascade of render callbacks that would follow.
            raise PreventUpdate

        session.evaluation.trade_graph_selected_trade_id = trade_id
        return session.to_store()


def _register_sync_neighbours_k_to_session(app: "Dash") -> None:
    """Popover NumberInput → ``session.trade_graph_neighbour_k``."""

    @app.callback(
        Output(SHELL_IDS["session_store"], "data", allow_duplicate=True),
        Input(TRADE_GRAPH_IDS["selected_neighbours_k_input"], "value"),
        State(SHELL_IDS["session_store"], "data"),
        prevent_initial_call=True,
    )
    def _sync_k(
        new_k:        Any,
        session_data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        try:
            new_k_int = int(new_k)
        except (TypeError, ValueError):
            raise PreventUpdate
        new_k_int = max(1, min(20, new_k_int))

        session = Session.from_store(session_data)
        if session.evaluation.trade_graph_neighbour_k == new_k_int:
            raise PreventUpdate
        session.evaluation.trade_graph_neighbour_k = new_k_int
        return session.to_store()


def _register_sync_neighbour_click_to_session(app: "Dash") -> None:
    """Click on a neighbour row inside the popover → drill to that trade.

    Each row's id is a pattern-matching dict
    (``{"type": "tg-neighbour-row", "trade_id": <id>}``); a single
    callback handles every row's click via the ``ALL`` wildcard.
    """

    @app.callback(
        Output(SHELL_IDS["session_store"], "data", allow_duplicate=True),
        Input(
            {"type": "tg-neighbour-row", "trade_id": ALL},
            "n_clicks",
        ),
        State(SHELL_IDS["session_store"], "data"),
        prevent_initial_call=True,
    )
    def _on_click(
        n_clicks_list: Sequence[Optional[int]],
        session_data:  Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        # The ``ALL`` pattern fires this callback on initial render
        # too (every n_clicks=None).  Filter to genuine clicks.
        if not n_clicks_list or not any(n_clicks_list):
            raise PreventUpdate

        triggered = ctx.triggered_id
        if not isinstance(triggered, dict) or "trade_id" not in triggered:
            raise PreventUpdate

        target_trade = triggered["trade_id"]
        session = Session.from_store(session_data)
        if session.evaluation.trade_graph_selected_trade_id == target_trade:
            raise PreventUpdate

        session.evaluation.trade_graph_selected_trade_id = target_trade
        return session.to_store()


# ═════════════════════════════════════════════════════════════════════
# RENDER — bootstrap (mount tripwire)
# ═════════════════════════════════════════════════════════════════════


def _register_bootstrap(app: "Dash", backend: "RadeBackend") -> None:
    """Populate the cluster Select on fresh mount of the page.

    The ``mount_signal`` Store fires this callback once per fresh
    mount (Page Contract §3 Rule L4).  We fetch the cluster list from
    the backend and write the option set to the Select.

    Override edges
    ~~~~~~~~~~~~~~
    * **Fresh-user default** — if ``session.trade_graph_cluster_id``
      is unset and we have clusters, we pick the first one and write
      both ``Select.value`` and ``session`` in the same return tuple.
    * **Stale session id** — if the session-stored cluster id no
      longer exists in the backend (e.g. version flipped), we fall
      back to the first option and overwrite session.

    URL deep-link (``?cluster=<id>``) is intentionally *not* wired
    yet — falls under the broader "deep-link the eval sub-tabs"
    Stage 4.x effort.  The callback structure leaves a clean seam for
    that follow-up.
    """

    @app.callback(
        Output(TRADE_GRAPH_IDS["cluster_select"], "data"),
        Output(TRADE_GRAPH_IDS["cluster_select"], "value", allow_duplicate=True),
        Output(SHELL_IDS["session_store"],        "data", allow_duplicate=True),
        Input(TRADE_GRAPH_IDS["mount_signal"],    "data"),
        State(SHELL_IDS["session_store"],         "data"),
        prevent_initial_call="initial_duplicate",
    )
    def _bootstrap(
        _trigger:     Any,
        session_data: Optional[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, str]], Any, Any]:
        res = backend.clusters_df()
        if not res.ok or res.data is None or res.data.empty:
            return [], no_update, no_update

        df = res.data
        options = [
            {"value": cid, "label": cid}
            for cid in sorted(df["cluster_id"].unique())
        ]

        session = Session.from_store(session_data)
        ev = session.evaluation
        valid_ids = {o["value"] for o in options}

        # Fresh-user default — neither the trade-graph nor the global
        # cluster_id is set.
        if not ev.trade_graph_cluster_id and not session.cluster_id:
            default_value = options[0]["value"]
            ev.trade_graph_cluster_id = default_value
            return options, default_value, session.to_store()

        # Stale session id — fall back to first option.
        active = ev.trade_graph_cluster_id or session.cluster_id
        if active not in valid_ids:
            default_value = options[0]["value"]
            ev.trade_graph_cluster_id = default_value
            return options, default_value, session.to_store()

        # Steady state — option list only; the layout-time seeded
        # value already matches session, no value-side write needed.
        return options, no_update, no_update


# ═════════════════════════════════════════════════════════════════════
# RENDER — graph elements + stylesheet + cluster-stats card
# ═════════════════════════════════════════════════════════════════════


def _register_render_graph(app: "Dash", backend: "RadeBackend") -> None:
    """Cluster / threshold / color-by → graph elements + stylesheet.

    Single fetch per cluster — node + edge lists are stashed in
    ``store_graph`` for downstream callbacks (selected-card,
    neighbours popover, legend).  Color-by changes don't re-fetch;
    they just rebuild the stylesheet from the cached payload.

    Trade attributes (residual gradient, asset_class / currency /
    product categorical modes) are **enriched** here from
    :meth:`RadeBackend.trades_df` so the stylesheet helper can
    paint nodes by the chosen mode.
    """

    @app.callback(
        Output(TRADE_GRAPH_IDS["cytoscape"],          "elements"),
        Output(TRADE_GRAPH_IDS["cytoscape"],          "stylesheet"),
        Output(TRADE_GRAPH_IDS["cytoscape"],          "layout"),
        Output(TRADE_GRAPH_IDS["cytoscape_minimap"],  "elements"),
        Output(TRADE_GRAPH_IDS["cytoscape_minimap"],  "layout"),
        Output(TRADE_GRAPH_IDS["pane_status"],        "children"),
        Output(TRADE_GRAPH_IDS["cluster_stats_grid"], "children"),
        Output(TRADE_GRAPH_IDS["store_graph"],        "data"),
        Input(SHELL_IDS["url"],                       "pathname"),
        Input(SHELL_IDS["session_store"],             "data"),
    )
    def _render(
        pathname:     Optional[str],
        session_data: Optional[Dict[str, Any]],
    ) -> Tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
        if pathname != _TRADE_GRAPH_PATH:
            raise PreventUpdate

        session = Session.from_store(session_data)
        ev = session.evaluation
        cluster_id = ev.trade_graph_cluster_id or session.cluster_id

        layout_name = ev.trade_graph_layout or DEFAULT_TRADE_GRAPH_LAYOUT
        layout_cfg = {
            "name":    layout_name,
            "fit":     True,
            "padding": 30,
            "animate": False,
        }
        minimap_layout_cfg = {**layout_cfg, "padding": 4}

        empty_grid = _cluster_stats_grid(None)

        if not cluster_id:
            return (
                [],
                build_stylesheet(ev.trade_graph_color_by, nodes_payload=[])[0],
                layout_cfg,
                [],
                minimap_layout_cfg,
                "No cluster selected.",
                empty_grid,
                {},
            )

        res = backend.trade_graph(cluster_id=cluster_id)
        if not res.ok or res.data is None:
            logger.info(
                "trade-graph fetch failed for cluster %s: %s",
                cluster_id, res.error,
            )
            return (
                [],
                build_stylesheet(ev.trade_graph_color_by, nodes_payload=[])[0],
                layout_cfg,
                [],
                minimap_layout_cfg,
                f"Graph unavailable for {cluster_id}.",
                empty_grid,
                {},
            )

        payload = res.data

        # Trade-level attribute lookup for color-by enrichment.  Per-
        # node residual lives on ``mean_residual``; categorical
        # attributes (asset_class / currency / product) live on the
        # cluster row, so every node in this cluster shares the same
        # value (the colour-by helper falls back gracefully when
        # values are missing).
        residuals_by_trade = _per_trade_residuals(backend, session.split, cluster_id)
        cluster_attrs = _cluster_attrs(backend, cluster_id)

        nodes_payload: List[Dict[str, Any]] = []
        for n in payload.nodes:
            data = {
                "id":         n.trade_id,
                "trade_type": n.trade_type,
                "cluster_id": n.cluster_id,
            }
            residual = residuals_by_trade.get(n.trade_id)
            if residual is not None and not _is_nan(residual):
                data["residual"] = float(residual)
            for k, v in cluster_attrs.items():
                data[k] = v
            nodes_payload.append({"data": data})

        # Threshold filter — drops weak edges before painting.  Cheap
        # enough to redo in this callback (we already have the
        # payload in hand) so the threshold slider's own render
        # callback only owns the label.
        thr = ev.trade_graph_weight_threshold
        edges_payload: List[Dict[str, Any]] = []
        for e in payload.edges:
            w = float(e.weight)
            if w < thr:
                continue
            edges_payload.append({"data": {"source": e.source, "target": e.target, "weight": w}})

        n_hidden = len(payload.edges) - len(edges_payload)
        status = (
            f"{payload.cluster_id} · "
            f"{payload.stats.n_nodes:,} nodes · "
            f"{len(edges_payload):,} edges"
        )
        if n_hidden > 0:
            status += f" · {n_hidden:,} hidden"

        stylesheet, _legend_pairs = build_stylesheet(
            ev.trade_graph_color_by, nodes_payload=nodes_payload,
        )

        elements = [*nodes_payload, *edges_payload]

        cluster_stats_children = _cluster_stats_grid(
            {
                "n_nodes":     payload.stats.n_nodes,
                "n_edges":     len(edges_payload),
                "density":     payload.stats.density,
                "mean_weight": payload.stats.mean_weight,
            }
        )

        # Store payload for downstream callbacks — keep the raw node
        # list, the filtered edge list, and a mapping of edges by
        # source so the neighbours popover can find top-k weights
        # without re-walking the whole edge list.
        store_payload: Dict[str, Any] = {
            "cluster_id":   payload.cluster_id,
            "nodes":        nodes_payload,
            "edges":        edges_payload,
            "n_target":     payload.n_target_trades,
            "n_elementary": payload.n_elementary_trades,
        }

        return (
            elements,
            stylesheet,
            layout_cfg,
            elements,           # mini-map shares the same elements
            minimap_layout_cfg,
            status,
            cluster_stats_children,
            store_payload,
        )


def _cluster_stats_grid(stats: Optional[Dict[str, Any]]) -> List[Any]:
    """2×2 KPI grid children for the Cluster Stats card."""
    if stats is None:
        return [
            KpiCard(label="Nodes",       value=_PLACEHOLDER),
            KpiCard(label="Edges",       value=_PLACEHOLDER),
            KpiCard(label="Density",     value=_PLACEHOLDER),
            KpiCard(label="Mean weight", value=_PLACEHOLDER),
        ]
    return [
        KpiCard(label="Nodes",       value=_fmt_int(stats.get("n_nodes"))),
        KpiCard(label="Edges",       value=_fmt_int(stats.get("n_edges"))),
        KpiCard(label="Density",     value=_fmt_float(stats.get("density"))),
        KpiCard(label="Mean weight", value=_fmt_float(stats.get("mean_weight"))),
    ]


def _per_trade_residuals(
    backend: "RadeBackend", split: str, cluster_id: str,
) -> Dict[str, float]:
    """Return ``{trade_id: mean_residual}`` for every trade in the cluster.

    Returns an empty dict on any backend failure — the caller treats
    "no residuals available" as "skip residual colouring", which
    falls back to ``trade_type`` via the stylesheet helper.
    """
    res = backend.trades_df(split, cluster_id=cluster_id)
    if not res.ok or res.data is None or res.data.empty:
        return {}
    df = res.data
    if "trade_id" not in df.columns or "mean_residual" not in df.columns:
        return {}
    return dict(zip(df["trade_id"], df["mean_residual"]))


def _cluster_attrs(
    backend: "RadeBackend", cluster_id: str,
) -> Dict[str, Any]:
    """Return the per-cluster categorical attributes.

    Every node in a cluster shares the same asset_class / currency /
    product (clusters are *defined* by these attributes), so a single
    lookup feeds every node.  When a column is missing we omit the
    key — the stylesheet helper's fallback handles the missing case.
    """
    res = backend.clusters_df(cluster_id=cluster_id)
    if not res.ok or res.data is None or res.data.empty:
        return {}
    row = res.data.iloc[0]
    out: Dict[str, Any] = {}
    for column in ("asset_class", "currency_code", "product_code"):
        if column in res.data.columns:
            value = row.get(column)
            if value is not None and not _is_nan(value):
                out[column] = str(value)
    return out


# ═════════════════════════════════════════════════════════════════════
# RENDER — Selected-Trade card
# ═════════════════════════════════════════════════════════════════════


def _register_render_selected_card(app: "Dash") -> None:
    """Reflect the active selection inside the Selected-Trade card.

    All inputs come from the local store + session — no backend hit
    so card paints synchronously after every node tap.
    """

    @app.callback(
        Output(TRADE_GRAPH_IDS["selected_trade_id"],       "children"),
        Output(TRADE_GRAPH_IDS["selected_chip_strip"],     "children"),
        Output(TRADE_GRAPH_IDS["selected_metrics"],        "children"),
        Output(TRADE_GRAPH_IDS["selected_neighbours_btn"], "disabled"),
        Output(TRADE_GRAPH_IDS["selected_copy_btn"],       "disabled"),
        Output(TRADE_GRAPH_IDS["selected_deep_dive_btn"],  "disabled"),
        Input(SHELL_IDS["url"],                            "pathname"),
        Input(SHELL_IDS["session_store"],                  "data"),
        Input(TRADE_GRAPH_IDS["store_graph"],              "data"),
    )
    def _render(
        pathname:     Optional[str],
        session_data: Optional[Dict[str, Any]],
        store_data:   Optional[Dict[str, Any]],
    ) -> Tuple[Any, Any, Any, bool, bool, bool]:
        if pathname != _TRADE_GRAPH_PATH:
            raise PreventUpdate

        session = Session.from_store(session_data)
        selected_id = session.evaluation.trade_graph_selected_trade_id

        if not selected_id or not store_data or not store_data.get("nodes"):
            return (
                _PLACEHOLDER,
                [],
                "Click a node in the graph to inspect the trade.",
                True,    # neighbours disabled
                True,    # copy disabled
                True,    # deep-dive disabled
            )

        node = _find_node(store_data["nodes"], selected_id)
        if node is None:
            return (
                _PLACEHOLDER,
                [],
                f"Trade '{selected_id}' is not in the active cluster.",
                True, True, True,
            )

        chips = _build_chips(node)
        metrics = _build_metrics_row(node)

        return (
            selected_id,
            chips,
            metrics,
            False,
            False,
            False,
        )


def _find_node(
    nodes_payload: Sequence[Dict[str, Any]], trade_id: str,
) -> Optional[Dict[str, Any]]:
    for n in nodes_payload:
        if n.get("data", {}).get("id") == trade_id:
            return n.get("data", {})
    return None


def _build_chips(node_data: Dict[str, Any]) -> List[Any]:
    """Return the chip strip for the Selected-Trade card.

    Renders one chip per categorical attribute we know about.  The
    chips share the ``rade-filter-chip`` style from rade.css so they
    visually match the global filter bar's chips.
    """
    rows: List[Tuple[str, str]] = []
    if (tt := node_data.get("trade_type")):
        rows.append(("Type", tt.capitalize()))
    if (ac := node_data.get("asset_class")):
        rows.append(("Asset", ac))
    if (cc := node_data.get("currency_code")):
        rows.append(("CCY", cc))
    if (pc := node_data.get("product_code")):
        rows.append(("Product", pc))

    return [
        html.Span(
            f"{label}: {value}",
            className=(
                "px-2 py-0.5 rounded-md text-[11px] "
                "bg-slate-800 text-slate-200 border border-slate-700"
            ),
        )
        for label, value in rows
    ]


def _build_metrics_row(node_data: Dict[str, Any]) -> Any:
    residual = node_data.get("residual")
    if residual is None or _is_nan(residual):
        return html.Span(
            "No metrics available for this trade.",
            className="text-[11px] text-slate-500 italic",
        )
    return html.Div(
        className="text-xs text-slate-300 flex items-center gap-2",
        children=[
            html.Span("Mean residual", className="text-slate-500"),
            html.Span(_fmt_float(residual), className="font-mono text-slate-100"),
        ],
    )


# ═════════════════════════════════════════════════════════════════════
# RENDER — Neighbours popover list
# ═════════════════════════════════════════════════════════════════════


def _register_render_neighbours_list(app: "Dash") -> None:
    """Selected trade + k + cached graph → top-k neighbour rows."""

    @app.callback(
        Output(TRADE_GRAPH_IDS["selected_neighbours_list"],     "children"),
        Output(TRADE_GRAPH_IDS["selected_neighbours_btn_label"], "children"),
        Input(SHELL_IDS["url"],                                  "pathname"),
        Input(SHELL_IDS["session_store"],                        "data"),
        Input(TRADE_GRAPH_IDS["store_graph"],                    "data"),
    )
    def _render(
        pathname:     Optional[str],
        session_data: Optional[Dict[str, Any]],
        store_data:   Optional[Dict[str, Any]],
    ) -> Tuple[Any, Any]:
        if pathname != _TRADE_GRAPH_PATH:
            raise PreventUpdate

        session = Session.from_store(session_data)
        ev = session.evaluation
        selected_id = ev.trade_graph_selected_trade_id
        k = max(1, min(20, ev.trade_graph_neighbour_k))
        button_label = f"Nearest {k}"

        if not selected_id or not store_data or not store_data.get("edges"):
            return (
                [
                    html.Span(
                        "Pick a node to load its neighbours.",
                        className="text-xs text-slate-500 italic",
                    ),
                ],
                button_label,
            )

        neighbours = _top_k_neighbours(store_data["edges"], selected_id, k=k)
        if not neighbours:
            return (
                [
                    html.Span(
                        "This trade has no neighbours under the active threshold.",
                        className="text-xs text-slate-500 italic",
                    ),
                ],
                button_label,
            )

        rows = [
            html.Div(
                id={"type": "tg-neighbour-row", "trade_id": tid},
                className=(
                    "flex items-center justify-between gap-2 "
                    "px-2 py-1 rounded-md cursor-pointer "
                    "hover:bg-slate-800 transition-colors"
                ),
                # ``n_clicks`` initialised so the pattern-matching
                # callback's ``any(n_clicks_list)`` filter works.
                n_clicks=0,
                children=[
                    html.Code(
                        tid,
                        className="font-mono text-xs text-slate-200 truncate",
                    ),
                    html.Span(
                        f"ρ {weight:.2f}",
                        className="text-[11px] font-mono text-violet-300",
                    ),
                ],
            )
            for tid, weight in neighbours
        ]

        return rows, button_label


def _top_k_neighbours(
    edges_payload: Sequence[Dict[str, Any]],
    trade_id: str,
    *,
    k: int,
) -> List[Tuple[str, float]]:
    """Walk the edge list, return the k strongest connections to ``trade_id``.

    Edges are undirected — we look at both ``source`` and ``target``
    sides.  Self-loops are dropped server-side already, but we
    defensively skip them here too.
    """
    pairs: List[Tuple[str, float]] = []
    for e in edges_payload:
        data = e.get("data", {})
        src, tgt = data.get("source"), data.get("target")
        if src == trade_id and tgt and tgt != trade_id:
            pairs.append((str(tgt), float(data.get("weight", 0.0))))
        elif tgt == trade_id and src and src != trade_id:
            pairs.append((str(src), float(data.get("weight", 0.0))))

    pairs.sort(key=lambda p: p[1], reverse=True)
    return pairs[:k]


# ═════════════════════════════════════════════════════════════════════
# RENDER — Node Legend
# ═════════════════════════════════════════════════════════════════════


def _register_render_legend(app: "Dash") -> None:
    """Color-by + cached payload → legend body children."""

    @app.callback(
        Output(TRADE_GRAPH_IDS["legend_body"], "children"),
        Input(SHELL_IDS["url"],                "pathname"),
        Input(SHELL_IDS["session_store"],      "data"),
        Input(TRADE_GRAPH_IDS["store_graph"],  "data"),
    )
    def _render(
        pathname:     Optional[str],
        session_data: Optional[Dict[str, Any]],
        store_data:   Optional[Dict[str, Any]],
    ) -> Any:
        if pathname != _TRADE_GRAPH_PATH:
            raise PreventUpdate

        session = Session.from_store(session_data)
        color_by = session.evaluation.trade_graph_color_by
        nodes_payload = (store_data or {}).get("nodes") or []

        # Run the stylesheet helper just for its (mode, pairs) tuple
        # — we discard the rules and use the legend pairs to render
        # the legend card body.  Both helpers stay in lockstep this
        # way (no chance of legend / stylesheet drift).
        _rules, pairs = build_stylesheet(color_by, nodes_payload=nodes_payload)
        return build_legend_body(color_by, pairs=pairs)


# ═════════════════════════════════════════════════════════════════════
# RENDER — Threshold label
# ═════════════════════════════════════════════════════════════════════


def _register_render_threshold_label(app: "Dash") -> None:
    """Slider value → "Min weight" pill label.

    Lives separately from the session-sync because the label is a
    pure local mirror of the slider — keeping it in its own callback
    means dragging the slider doesn't churn unrelated session
    consumers.
    """

    @app.callback(
        Output(TRADE_GRAPH_IDS["threshold_value_label"], "children"),
        Input(TRADE_GRAPH_IDS["threshold_slider"],       "value"),
    )
    def _render(threshold: Optional[float]) -> str:
        try:
            return f"{max(0.0, min(1.0, float(threshold))):.2f}"
        except (TypeError, ValueError):
            return "0.00"


# ═════════════════════════════════════════════════════════════════════
# RENDER — Density distribution + Edges-vs-nodes scatter
# ═════════════════════════════════════════════════════════════════════


def _register_render_density_chart(
    app: "Dash", backend: "RadeBackend",
) -> None:
    @app.callback(
        Output(TRADE_GRAPH_IDS["density_chart"], "figure"),
        Input(SHELL_IDS["url"],                  "pathname"),
        Input(SHELL_IDS["session_store"],        "data"),
    )
    def _render(
        pathname:     Optional[str],
        session_data: Optional[Dict[str, Any]],
    ) -> Any:
        if pathname != _TRADE_GRAPH_PATH:
            raise PreventUpdate

        session = Session.from_store(session_data)
        selected_cluster = (
            session.evaluation.trade_graph_cluster_id or session.cluster_id
        )

        res = backend.graph_stats_df()
        if not res.ok or res.data is None or res.data.empty:
            return empty_figure("No graph stats available.")

        return density_distribution(
            res.data, selected_cluster_id=selected_cluster,
        )


def _register_render_edges_vs_nodes_chart(
    app: "Dash", backend: "RadeBackend",
) -> None:
    @app.callback(
        Output(TRADE_GRAPH_IDS["edges_vs_nodes_chart"], "figure"),
        Input(SHELL_IDS["url"],                         "pathname"),
        Input(SHELL_IDS["session_store"],               "data"),
    )
    def _render(
        pathname:     Optional[str],
        session_data: Optional[Dict[str, Any]],
    ) -> Any:
        if pathname != _TRADE_GRAPH_PATH:
            raise PreventUpdate

        session = Session.from_store(session_data)
        selected_cluster = (
            session.evaluation.trade_graph_cluster_id or session.cluster_id
        )

        res = backend.graph_stats_df()
        if not res.ok or res.data is None or res.data.empty:
            return empty_figure("No graph stats available.")

        return edges_vs_nodes_scatter(
            res.data, selected_cluster_id=selected_cluster,
        )


# ═════════════════════════════════════════════════════════════════════
# CLIENTSIDE — Fit view + Export PNG
# ═════════════════════════════════════════════════════════════════════


def _register_clientside(app: "Dash") -> None:
    """Wire the two clientside actions to their JS implementations.

    Both use the existing ``window.dash_clientside.trade_graph``
    namespace from ``assets/js/trade_graph.js``.  Output goes to a
    placeholder Div property to satisfy Dash's "every callback must
    have an Output" rule — neither action needs to round-trip server
    state.
    """
    app.clientside_callback(
        ClientsideFunction(namespace="trade_graph", function_name="fit_view"),
        # The button itself is the receiver of the no-op write; we
        # never read this property so it's a safe sink.
        Output(TRADE_GRAPH_IDS["fit_btn"], "n_clicks"),
        Input(TRADE_GRAPH_IDS["fit_btn"],  "n_clicks"),
        State(TRADE_GRAPH_IDS["cytoscape"], "id"),
        prevent_initial_call=True,
    )

    app.clientside_callback(
        ClientsideFunction(namespace="trade_graph", function_name="export_png"),
        Output(TRADE_GRAPH_IDS["export_btn"], "n_clicks"),
        Input(TRADE_GRAPH_IDS["export_btn"],  "n_clicks"),
        State(TRADE_GRAPH_IDS["cytoscape"],   "id"),
        prevent_initial_call=True,
    )


# ─────────────────────────────────────────────────────────────────────
# Formatters / helpers
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


def _is_nan(value: Any) -> bool:
    """``True`` for ``NaN`` / ``None``; survives non-numeric inputs."""
    if value is None:
        return True
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


__all__ = ["register"]
```

---

## Appendix B — `assets/rade.css` patch (Trade-Graph utilities)

**One-time** addition to the shipped `rade.css`.  The pre-E.3 build of
`rade.css` was JIT-compiled before `layouts/evaluation/trade_graph.py`
existed, so several Tailwind utilities the new layout references were
never emitted.  Without them the page renders as default block flow
(side panel below the graph, cards growing to content) — visually
indistinguishable from the old Phase E.0 stub.

### How to apply

1. Open `src/ui/apps/rade_analytics/assets/rade.css`.
2. **Append** the block below to the very end of the file
   (immediately after the last existing rule — typically the
   `.rade-evaluation-subtab--stub { … }` block).
3. Hard-refresh the browser (`Ctrl+Shift+R`) to bust the cached old
   stylesheet.

No existing rules are removed or modified — this is pure additive.

### Why each rule is needed

| Class(es) | Used by | Symptom if missing |
|---|---|---|
| `.h-[200px]`, `.h-[180px]`, `.h-[140px]` | Selected-Trade / Cluster-Stats / Legend cards | Side panel grows tall to content; "items-stretch" can't lock the cytoscape pane to a sensible height |
| `.min-h-[400px]` | Cytoscape wrapper | Pane collapses to zero on short viewports → graph hidden |
| `.min-h-[22px]` | Empty chip strip | Strip jumps when chips populate after a node tap |
| `.min-w-[180px]`, `.min-w-[220px]` | Header layout-radio + cluster-select | Pickers ellipsis-clip cluster ids |
| `.text-[10px]`, `.text-[11px]` | Legend captions, KPI labels, threshold pill | Labels render at parent font-size (way too large) |
| `.flex-shrink-0` | Legend swatches (figures/trade_graph_stylesheet.py) | Swatches squeeze and disappear |
| `.bg-slate-900/70` | Mini-map glass background | Mini-map shows transparent (only `/60` and `/80` alpha existed) |
| `.backdrop-blur` | Mini-map glass | No blur effect (only the `-sm` 4 px variant existed) |
| `.lg:grid-cols-3`, `.lg:col-span-2` | Row 2 main grid + cytoscape pane | **The keystone fix** — without these the row collapses to a single column on lg+ viewports, putting the side panel below the graph |

The long-term fix is to re-run the Tailwind CLI so `rade.css`
regenerates from scratch and tracks every class the source files
reference automatically:

```bash
npx tailwindcss \
  -c src/ui/apps/rade_analytics/assets/tailwind.config.js \
  -i src/ui/apps/rade_analytics/assets/tailwind.input.css \
  -o src/ui/apps/rade_analytics/assets/rade.css \
  --minify
```

Until then, this manual block stays in `rade.css`.

### The patch

```css
/* ─────────────────────────────────────────────────────────────────── */
/* PHASE E.3 — TRADE-GRAPH UTILITIES                                    */
/* Tailwind classes consumed by                                         */
/* ``layouts/evaluation/trade_graph.py`` and                            */
/* ``figures/trade_graph_stylesheet.py`` that were not present in the   */
/* pre-E.3 compile of rade.css.  Rebuilding via the Tailwind CLI would  */
/* regenerate these automatically; until then we ship them by hand.    */
/* ─────────────────────────────────────────────────────────────────── */

/* ── Arbitrary fixed heights ───────────────────────────────────── */
/* Side-panel cards anchor to these heights so the cytoscape pane
   can match their combined intrinsic height via ``items-stretch``
   on the row grid.  See trade_graph.py:_selected_trade_card,
   _legend_card, _cluster_stats_card. */
.h-\[200px\] { height: 200px; }
.h-\[180px\] { height: 180px; }
.h-\[140px\] { height: 140px; }

/* ── Arbitrary minimum dimensions ──────────────────────────────── */
/* Cytoscape pane never collapses below 400 px even on short
   viewports; selector header pickers reserve a sensible minimum
   width so the cluster id labels don't ellipsis-clip. */
.min-h-\[400px\] { min-height: 400px; }
.min-h-\[22px\]  { min-height: 22px; }
.min-w-\[180px\] { min-width: 180px; }
.min-w-\[220px\] { min-width: 220px; }

/* ── Arbitrary text sizes ──────────────────────────────────────── */
/* 10–11 px micro-labels for the legend, chip strip, threshold
   pill and 2×2 KPI grid. */
.text-\[10px\] { font-size: 10px; line-height: 1; }
.text-\[11px\] { font-size: 11px; line-height: 1.4; }

/* ── Aliases / supplementary single-purpose utilities ──────────── */
/* ``flex-shrink-0`` is the Tailwind v2 alias for ``shrink-0``;
   the legend swatch builder still emits the long form. */
.flex-shrink-0 { flex-shrink: 0; }

/* Mini-map background uses 70 % alpha; existing rade.css covers
   60 % and 80 % only. */
.bg-slate-900\/70 { background-color: rgb(15 23 42 / 0.7); }

/* Default ``backdrop-blur`` (8 px) used by the mini-map glass card;
   existing CSS only ships ``backdrop-blur-sm`` (4 px). */
.backdrop-blur { backdrop-filter: blur(8px); }

/* ── Responsive grid (lg: ≥ 1024 px) ───────────────────────────── */
/* Row 2 of the Trade-Graph layout collapses to one column on
   narrow screens and switches to a 2-fr-cytoscape + 1-fr-side-
   panel split at the lg breakpoint.  Without these, the cytoscape
   pane silently drops back to col-span-1 and the page reads like
   a vertical stack. */
@media (min-width: 1024px) {
  .lg\:grid-cols-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  .lg\:col-span-2  { grid-column: span 2 / span 2; }
}
```

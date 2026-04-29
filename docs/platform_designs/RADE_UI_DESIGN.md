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

## Appendix A — `figures/trade_graph_stylesheet.py`

Dynamic Cytoscape stylesheet + legend body generators for the
**Trade-Graph sub-tab** (Phase E.3 rebuild).  Drives the header-band's
"Color by" select — one helper produces the rule list Cytoscape
consumes, the other produces the legend card body that explains the
colours.  Lives at `src/ui/apps/rade_analytics/figures/trade_graph_stylesheet.py`;
re-exported from `figures/__init__.py` as `build_stylesheet` and
`build_legend_body`.

Five colour-by modes are supported:

* `trade_type` — target=amber, elementary=violet (default).
* `residual` — diverging gradient (negative=teal → ~0=slate → positive=magenta), keyed on each node's `residual` data field.
* `asset_class` / `currency` / `product` — categorical 8-tone palette.

Both helpers consume the same `nodes_payload` shape that the callback
module stashes in `store_graph` so the legend and the stylesheet stay
in lockstep — there's no second source of truth for palettes / labels.

Drop the file in place.  Dash auto-discovers it via the
`figures/__init__.py` re-export; no other plumbing needed.

```python
"""Dynamic Cytoscape stylesheet + legend body generators for the
Trade-Graph sub-tab.

The header-band's "Color by" select drives node colouring.  Five modes
are supported:

* ``trade_type``  — target=amber, elementary=violet (the safe default
  that ships in the layout's static stylesheet).
* ``residual``    — gradient from sub-zero teal → over-zero magenta,
  keyed on each node's residual metric.
* ``asset_class`` — categorical palette mapped from the distinct values
  observed in the cluster's nodes.
* ``currency``    — categorical palette.
* ``product``     — categorical palette.

The two helpers in this module produce, for a given (mode, node-list)
pair:

1. :func:`build_stylesheet` — a Cytoscape stylesheet ready to be
   handed to ``dash_cytoscape.Cytoscape.stylesheet``.  Always
   includes the four base selectors (``node``, ``node:selected``,
   ``node[trade_type='target']`` for size, ``edge``) — colour-by
   modes layer additional selectors on top.
2. :func:`build_legend_body` — a Dash component tree that the
   ``_register_render_legend`` callback drops into the legend card's
   ``legend_body`` slot.  Mirrors the active stylesheet so users
   never have to guess what a colour means.

Why a dedicated module
----------------------
The legend body and the stylesheet must agree on every palette,
threshold and label — keeping them next to each other (one builds
the visual rules, the other builds the legend that explains them)
makes drift impossible.  Both consume the same ``nodes_payload``
shape that the callback module stashes in ``store_graph``.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from dash import html


# ─────────────────────────────────────────────────────────────────────
# Palettes
# ─────────────────────────────────────────────────────────────────────

# Eight-tone categorical palette — colour-blind-friendly tones picked
# from the Tailwind 400 band so they render evenly on the dark theme.
# Cycled if the cluster carries more distinct values than this length.
_CATEGORICAL_PALETTE: Tuple[str, ...] = (
    "#8b5cf6",   # violet-400
    "#f59e0b",   # amber-400
    "#10b981",   # emerald-400
    "#60a5fa",   # blue-400
    "#f472b6",   # pink-400
    "#facc15",   # yellow-400
    "#22d3ee",   # cyan-400
    "#a3e635",   # lime-400
)

# Residual gradient endpoints — teal for negative, slate for ~zero,
# magenta for positive.  Cytoscape supports ``mapData(...)`` linear
# interpolation between two colours; we lay down two selectors so
# the gradient pivots through ~zero.
_RESIDUAL_NEGATIVE_COLOR = "#0ea5e9"   # sky-500
_RESIDUAL_ZERO_COLOR     = "#475569"   # slate-600
_RESIDUAL_POSITIVE_COLOR = "#ec4899"   # pink-500


# Map color-by mode → node-payload column name.  ``None`` for
# ``trade_type`` because that mode reads the ``trade_type`` selector
# directly without needing a numeric / categorical column.
_NODE_COLUMN_FOR_MODE: Dict[str, Optional[str]] = {
    "trade_type":  None,
    "residual":    "residual",
    "asset_class": "asset_class",
    "currency":    "currency_code",
    "product":     "product_code",
}


# ─────────────────────────────────────────────────────────────────────
# Stylesheet builder
# ─────────────────────────────────────────────────────────────────────


def _base_stylesheet() -> List[Dict[str, Any]]:
    """Stylesheet rules that apply regardless of color-by mode.

    These cover sizing, edge rendering and the selection ring.  Mode-
    specific colour rules are layered on top by :func:`build_stylesheet`.
    """
    return [
        {
            "selector": "node",
            "style": {
                "label":            "",
                "width":            12,
                "height":           12,
                "background-color": "#8b5cf6",
                "border-color":     "#0f172a",
                "border-width":     1,
                "transition-property": "background-color, width, height, border-color",
                "transition-duration": "150ms",
            },
        },
        {
            "selector": "node[trade_type = 'target']",
            "style": {"width": 18, "height": 18},
        },
        {
            "selector": "node:selected",
            "style": {
                "border-color":  "#10b981",
                "border-width":  3,
                "width":         22,
                "height":        22,
            },
        },
        {
            "selector": "edge",
            "style": {
                "width":           "mapData(weight, 0, 1, 0.5, 3)",
                "line-color":      "rgba(148, 163, 184, 0.35)",
                "curve-style":     "haystack",
                "haystack-radius": 0.5,
            },
        },
    ]


def _trade_type_rules() -> List[Dict[str, Any]]:
    """Restore the legacy amber / violet trade-type colouring."""
    return [
        {
            "selector": "node",
            "style": {"background-color": "#8b5cf6"},   # elementary
        },
        {
            "selector": "node[trade_type = 'target']",
            "style": {"background-color": "#f59e0b"},   # target
        },
    ]


def _residual_rules(values: Sequence[float]) -> List[Dict[str, Any]]:
    """Two-stop gradient keyed on the per-node ``residual`` data field.

    Cytoscape's ``mapData(field, min, max, color_lo, color_hi)`` does a
    linear interpolation; to get a teal → grey → magenta diverging
    palette we lay down two rules — one for the negative half, one for
    the positive half — split at zero.
    """
    if not values:
        return []

    v_min = float(min(values))
    v_max = float(max(values))

    # If every residual is zero (or NaN-only after filtering), fall
    # back to the trade-type colouring so we never paint the whole
    # graph the same flat tone.
    if v_min == v_max:
        return _trade_type_rules()

    # Anchor the diverging palette at zero so positive and negative
    # residuals are visually distinct.  When the data is one-sided
    # (e.g. all positive), the negative selector simply matches no
    # nodes — Cytoscape silently no-ops.
    return [
        {
            "selector": f"node[residual <= 0]",
            "style": {
                "background-color": (
                    f"mapData(residual, {min(v_min, 0.0)}, 0, "
                    f"{_RESIDUAL_NEGATIVE_COLOR}, {_RESIDUAL_ZERO_COLOR})"
                ),
            },
        },
        {
            "selector": f"node[residual > 0]",
            "style": {
                "background-color": (
                    f"mapData(residual, 0, {max(v_max, 1e-9)}, "
                    f"{_RESIDUAL_ZERO_COLOR}, {_RESIDUAL_POSITIVE_COLOR})"
                ),
            },
        },
    ]


def _categorical_rules(
    values: Sequence[str],
    *,
    column_name: str,
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
    """Per-distinct-value selector → palette colour.

    Returns the rules **and** the (value, colour) pairs the legend
    builder needs — so we walk the unique values once.
    """
    distinct = sorted({v for v in values if v is not None and v != ""})
    pairs: List[Tuple[str, str]] = []
    rules: List[Dict[str, Any]] = []

    for idx, value in enumerate(distinct):
        colour = _CATEGORICAL_PALETTE[idx % len(_CATEGORICAL_PALETTE)]
        pairs.append((value, colour))
        # Cytoscape selectors need single-quoted string values; we
        # escape any embedded apostrophes defensively.
        safe = str(value).replace("'", "\\'")
        rules.append(
            {
                "selector": f"node[{column_name} = '{safe}']",
                "style": {"background-color": colour},
            }
        )

    return rules, pairs


def build_stylesheet(
    color_by: str,
    *,
    nodes_payload: Iterable[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Optional[List[Tuple[str, str]]]]:
    """Build the Cytoscape stylesheet for the given color-by mode.

    Parameters
    ----------
    color_by
        One of :data:`session.EVALUATION_TRADE_GRAPH_COLOR_BY`.  When
        the mode is unknown (e.g. user hand-edited the store), falls
        back to ``trade_type`` and the legend reflects that.
    nodes_payload
        The same dict payload the render callback stashed in
        ``store_graph["nodes"]`` — each item is
        ``{"data": {"id": ..., "trade_type": ..., "residual": ...,
        "asset_class": ..., "currency_code": ..., "product_code": ...}}``.

    Returns
    -------
    (stylesheet, legend_pairs)
        * ``stylesheet`` — the rule list, ready for
          ``Cytoscape.stylesheet``.
        * ``legend_pairs`` — for the categorical / trade_type modes,
          a list of ``(label, colour)`` tuples the legend builder
          uses to reconstruct the swatches.  ``None`` for
          ``residual`` (the legend renders a gradient bar instead).

    Notes
    -----
    The base sizing / edge rules come first so mode-specific colour
    rules can override the default ``background-color`` simply by
    appearing later in the list (Cytoscape applies rules in order).
    """
    rules = _base_stylesheet()

    if color_by == "residual":
        residuals = [
            float(n.get("data", {}).get("residual"))
            for n in nodes_payload
            if n.get("data", {}).get("residual") is not None
        ]
        rules.extend(_residual_rules(residuals))
        return rules, None

    column = _NODE_COLUMN_FOR_MODE.get(color_by)
    if column is None or color_by == "trade_type":
        # ``trade_type`` mode (default + fallback for unknown mode).
        rules.extend(_trade_type_rules())
        return rules, [("Target", "#f59e0b"), ("Elementary", "#8b5cf6")]

    values = [
        str(n.get("data", {}).get(column))
        for n in nodes_payload
        if n.get("data", {}).get(column) is not None
    ]
    cat_rules, pairs = _categorical_rules(values, column_name=column)
    rules.extend(cat_rules)

    # Edge case — cluster nodes carry no entry under this column
    # (e.g. residual-only metadata).  Fall back to trade-type so we
    # never paint a transparent / broken graph.
    if not pairs:
        rules.extend(_trade_type_rules())
        return rules, [("Target", "#f59e0b"), ("Elementary", "#8b5cf6")]

    return rules, pairs


# ─────────────────────────────────────────────────────────────────────
# Legend body builder
# ─────────────────────────────────────────────────────────────────────


def _swatch(color: str) -> html.Div:
    return html.Div(
        className="w-3 h-3 rounded-full flex-shrink-0",
        style={"backgroundColor": color},
    )


def _legend_row(color: str, label: str, sublabel: Optional[str] = None) -> html.Div:
    children: List[Any] = [_swatch(color)]
    text_block_children: List[Any] = [
        html.Span(label, className="text-xs text-slate-200"),
    ]
    if sublabel:
        text_block_children.append(
            html.Span(sublabel, className="text-[10px] text-slate-500"),
        )
    children.append(
        html.Div(
            className="flex flex-col leading-tight",
            children=text_block_children,
        ),
    )
    return html.Div(
        className="flex items-center gap-2",
        children=children,
    )


def _residual_gradient_bar() -> html.Div:
    """Horizontal gradient bar with min / mid / max tick labels.

    The renderer doesn't know the exact residual extrema (we pass the
    raw bar; the user reads "negative ← → positive" as semantic, not
    quantitative).  A future enhancement can drop the per-cluster
    ``[min, max]`` numbers in here once we surface them through the
    payload's metadata.
    """
    gradient = (
        f"linear-gradient(to right, "
        f"{_RESIDUAL_NEGATIVE_COLOR}, "
        f"{_RESIDUAL_ZERO_COLOR}, "
        f"{_RESIDUAL_POSITIVE_COLOR})"
    )
    return html.Div(
        className="flex flex-col gap-1",
        children=[
            html.Div(
                className="h-2 rounded-full w-full",
                style={"background": gradient},
            ),
            html.Div(
                className="flex justify-between text-[10px] text-slate-500",
                children=[
                    html.Span("Negative"),
                    html.Span("0"),
                    html.Span("Positive"),
                ],
            ),
        ],
    )


def build_legend_body(
    color_by: str,
    *,
    pairs: Optional[List[Tuple[str, str]]],
) -> List[Any]:
    """Build the ``legend_body``'s children for the given mode.

    The legend card is a fixed-height container (140 px); for modes
    with many distinct categorical values, the legend wraps onto a
    second / third row.  We don't paginate or scroll the legend —
    if a cluster has 30 distinct currencies, the user has bigger
    problems than legend layout.

    Parameters
    ----------
    color_by
        One of :data:`session.EVALUATION_TRADE_GRAPH_COLOR_BY`.
    pairs
        From :func:`build_stylesheet`'s return tuple.  ``None`` for
        ``residual`` (gradient bar); a list of ``(label, colour)``
        for ``trade_type`` and the categoricals.
    """
    if color_by == "residual":
        return [_residual_gradient_bar()]

    if not pairs:
        return [
            html.Span(
                "No data for legend.",
                className="text-xs text-slate-500 italic",
            ),
        ]

    # Two-column flex-wrap so up to ~8 categoricals stay legible
    # within the 140-px card without overflowing.  Beyond 8, the
    # palette cycles (callers see this in `_categorical_rules`) and
    # the swatch repeats — visually the user understands "more
    # categories than colours".
    return [
        html.Div(
            className="grid grid-cols-2 gap-x-3 gap-y-1",
            children=[_legend_row(color, label) for label, color in pairs],
        ),
    ]


__all__ = [
    "build_legend_body",
    "build_stylesheet",
]
```

Don't forget to add the re-export to `figures/__init__.py`:

```python
from .trade_graph_stylesheet import build_legend_body, build_stylesheet

__all__ = [
    # ... existing entries ...
    "build_legend_body",
    "build_stylesheet",
    # ... existing entries ...
]
```

---

## Appendix B — `assets/js/trade_graph.js`

Two clientside callbacks for the **Trade-Graph sub-tab**'s Cytoscape
toolbar — `fit_view` (re-runs `cy.fit() + cy.center()`) and
`export_png` (downloads a 2× scale base64 PNG of the current graph
state).  Lives at `src/ui/apps/rade_analytics/assets/js/trade_graph.js`;
Dash auto-loads every file under `assets/` at startup, so dropping
this file in place is all that's required — no bundler, no import
statement.

The functions are referenced from Python via:

```python
ClientsideFunction(namespace="trade_graph", function_name="fit_view")
ClientsideFunction(namespace="trade_graph", function_name="export_png")
```

Both pull the live Cytoscape instance via `dash_cytoscape`'s internal
registry (`el._cyreg.cy`, with a fallback for older builds).  When the
graph isn't ready yet (user clicks Fit before mount completes) both
silently no-op rather than throwing.

```javascript
/* ──────────────────────────────────────────────────────────────────
 * Trade-Graph sub-tab — clientside callbacks.
 *
 * Two actions live here, both targeting the Cytoscape instance the
 * Dash app mounts at id="eval-trade-graph-cytoscape":
 *
 *   1. fit_view   — re-runs cy.fit() so the graph centres + zooms
 *                   to fill the viewport.  Useful after the user
 *                   has panned / zoomed away.
 *   2. export_png — calls cy.png({...}) and triggers a download
 *                   via a synthetic <a> element.
 *
 * Both functions are referenced from Python via:
 *
 *     ClientsideFunction(namespace="trade_graph",
 *                        function_name="fit_view")
 *
 * Page Contract reference: §6 Lever P2 (clientside_callback for
 * trivial UI / non-stateful actions).
 *
 * Why we hunt for the cy instance via dash_cytoscape's internal
 * registry: dash_cytoscape doesn't expose a clean "give me the
 * cy instance for id=X" hook.  The component stashes the cy
 * reference on the underlying React component, which we reach via
 * ``window.cy`` (set by some integrations) or by reading the
 * canvas DOM and walking the React fiber tree as a last resort.
 *
 * The lookup intentionally guards against "Cytoscape not ready
 * yet" — when the user clicks Fit before the graph has finished
 * its first render, we silently no-op rather than throwing.
 * ────────────────────────────────────────────────────────────────── */

window.dash_clientside = window.dash_clientside || {};
window.dash_clientside.trade_graph = (function () {
    "use strict";

    /* Locate the live Cytoscape instance for the given DOM id.
     *
     * dash_cytoscape attaches the cy instance as a property on the
     * outer container element; we read it back via the cy global
     * pattern. */
    function getCy(cytoscapeId) {
        const el = document.getElementById(cytoscapeId);
        if (!el) {
            return null;
        }
        /* dash_cytoscape (v0.3+) exposes the cy instance as a
         * property on the wrapping div.  Older versions stashed it
         * on the canvas element — we check both. */
        if (el._cyreg && el._cyreg.cy) {
            return el._cyreg.cy;
        }
        if (el.__cy) {
            return el.__cy;
        }
        /* Last resort — walk the children for a canvas with a cy
         * back-reference (older dash_cytoscape builds). */
        const canvas = el.querySelector("canvas");
        if (canvas && canvas._cy) {
            return canvas._cy;
        }
        return null;
    }

    /* Trigger a download of a base64 PNG payload as a file. */
    function triggerDownload(dataUrl, fileName) {
        const a = document.createElement("a");
        a.href = dataUrl;
        a.download = fileName;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }

    /* Fit-view — re-runs the layout's fit/center logic.
     *
     * Returns dash.no_update so we don't churn the button's n_clicks
     * (which is the Dash callback's Output sink). */
    function fit_view(_n_clicks, cytoscapeId) {
        if (!_n_clicks) {
            return window.dash_clientside.no_update;
        }
        const cy = getCy(cytoscapeId);
        if (!cy) {
            return window.dash_clientside.no_update;
        }
        cy.fit(undefined, 30);    // 30 px padding — matches layout default
        cy.center();
        return window.dash_clientside.no_update;
    }

    /* Export-PNG — generates a high-res PNG of the current graph
     * state and triggers a browser download. */
    function export_png(_n_clicks, cytoscapeId) {
        if (!_n_clicks) {
            return window.dash_clientside.no_update;
        }
        const cy = getCy(cytoscapeId);
        if (!cy) {
            return window.dash_clientside.no_update;
        }

        /* 2× scale gives us a crisp image for retina displays
         * without bloating the file size. */
        const dataUrl = cy.png({
            output: "base64uri",
            full:   true,
            scale:  2,
            bg:     "#0f172a",     // slate-900 — matches dark theme
        });

        const ts = new Date().toISOString().replace(/[:.]/g, "-");
        triggerDownload(dataUrl, `trade-graph-${ts}.png`);

        return window.dash_clientside.no_update;
    }

    return {
        fit_view:   fit_view,
        export_png: export_png,
    };
})();
```

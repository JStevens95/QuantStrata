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

## Appendix A — `figures/cluster_deep_dive_charts.py`

Full verbatim body of the Cluster Deep-Dive sub-tab figure builders.
Paste into `src/ui/apps/rade_analytics/figures/cluster_deep_dive_charts.py`.

Three builders, each tuned to a single-cluster diagnostic view:

| Function | Used by | Purpose |
|----|----|----|
| `predicted_vs_actual_band` | Row 3 right | Predicted + actual PnL lines with the residual zone shaded rose between them. |
| `per_trade_residual_violin` | Row 4 left | Per-trade residual distribution, split by `trade_type` (target / elementary). Falls back to a single aggregate violin when the trade-graph payload is missing. |
| `per_trade_scatter` | Row 4 right | Per-trade scatter `mean_residual × mae`, coloured by `trade_type`. The optional `selected_trade_id` renders an emerald-ring highlight marker so cross-highlighting from the Row 5 AgGrid is visible at a glance. |

All three share the same data contract: callers pass a pandas frame
with the columns listed in each docstring and a `trade_type_map`
(`{trade_id: "target" | "elementary"}`) derived from the trade-graph
payload. Missing columns / empty frames gracefully return an
`empty_figure` — the UI never shows a broken axis.

```python
"""Cluster Deep-Dive specific figure builders (Phase E.4).

Three builders land here, all tuned to single-cluster diagnostic views:

* :func:`predicted_vs_actual_band` — Row 3 right.  Predicted + actual
  PnL lines for one cluster with the residual zone between them shaded
  (rose) so the user can *see* where the model is over- or
  under-predicting.  Distinct from :func:`figures.portfolio_pnl`, which
  has no error shading and is used on the Portfolio tab.
* :func:`per_trade_residual_violin` — Row 4 left.  Distribution of
  per-trade metrics (mean_residual or mae), split by ``trade_type``
  (target / elementary).  Uses :class:`plotly.graph_objects.Violin` so
  the visual language matches the Portfolio residual violin.
* :func:`per_trade_scatter` — Row 4 right.  Per-trade scatter:
  ``mean_residual`` (x) vs ``mae`` (y), coloured by ``trade_type``.
  A ``selected_trade_id`` gets a larger, emerald-bordered marker so
  cross-highlighting from the trades AgGrid is visible at a glance.

All three share the same data contract: callers pass a pandas frame
with the columns listed in each function's docstring, and a
``trade_type_map`` (``{trade_id: "target" | "elementary"}``) derived
from the trade-graph payload.  Missing columns / empty frames gracefully
return an :func:`empty_figure` — the UI never shows a broken axis.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ._theme import color_for_index, empty_figure, rade_layout, rgba

# Fixed colours for the two trade types so violin + scatter + Cytoscape
# legend always match.  Must stay in sync with the Cytoscape stylesheet
# in ``layouts/evaluation/trade_graph.py``.
_TRADE_TYPE_COLOR = {
    "target":     "#f59e0b",   # amber
    "elementary": "#8b5cf6",   # violet
}
_TRADE_TYPE_ORDER: tuple[str, ...] = ("target", "elementary")


# ─────────────────────────────────────────────────────────────────────
# Row 3 right — predicted vs actual with shaded error band
# ─────────────────────────────────────────────────────────────────────


def predicted_vs_actual_band(df: pd.DataFrame) -> go.Figure:
    """Two-line PnL chart with the residual zone shaded rose.

    Parameters
    ----------
    df
        Single-cluster timeseries frame.  Required columns:
        ``predicted``, ``actual``.  Optional but preferred:
        ``scenario_idx`` (drives sort order) and ``scenario_label``
        (drives the x-axis tick labels).  Extra columns are ignored.

    Notes
    -----
    The shaded band is built as ``predicted`` fill-to-zero *minus*
    ``actual`` fill-to-zero — i.e. two scatter traces with
    ``fill='tonexty'`` form the envelope between the two lines.  Using
    ``rgba(0.18)`` for the fill keeps the band visible against the
    dark theme without drowning the line strokes.
    """
    if (
        df is None
        or df.empty
        or "predicted" not in df.columns
        or "actual" not in df.columns
    ):
        return empty_figure("No cluster timeseries for this selection.")

    df_sorted = df.sort_values("scenario_idx") if "scenario_idx" in df.columns else df
    x_vals: Sequence
    if "scenario_label" in df_sorted.columns:
        x_vals = df_sorted["scenario_label"].tolist()
    else:
        x_vals = list(range(len(df_sorted)))

    pred_color = color_for_index(0)       # violet
    actual_color = "#cbd5e1"              # slate-300 — neutral reference
    band_color = color_for_index(3)       # rose — "error zone"

    fig = go.Figure()

    # Baseline trace — predicted.  Rendered first so the "actual" trace
    # below can fill-to-next and produce the between-lines band.
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=df_sorted["predicted"],
            mode="lines",
            name="Predicted PnL",
            line={"color": pred_color, "width": 2.5},
            hovertemplate="%{y:.4f}<extra>Predicted</extra>",
        )
    )
    # Error band — actual with fill='tonexty' fills the area between
    # the two traces.
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=df_sorted["actual"],
            mode="lines",
            name="Actual PnL",
            line={"color": actual_color, "width": 1.5, "dash": "dash"},
            fill="tonexty",
            fillcolor=rgba(band_color, 0.18),
            hovertemplate="%{y:.4f}<extra>Actual</extra>",
        )
    )

    fig.update_layout(
        **rade_layout(
            show_legend=True,
            hovermode="x unified",
            xaxis={"showticklabels": True},
            yaxis={"title": {"text": "PnL", "font": {"color": "#94a3b8"}}},
        ),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────
# Row 4 left — per-trade residual violin (target vs elementary)
# ─────────────────────────────────────────────────────────────────────


def per_trade_residual_violin(
    trades_df:      pd.DataFrame,
    *,
    trade_type_map: Optional[Mapping[str, str]] = None,
    value_column:   str = "mean_residual",
    y_axis_title:   str = "Mean residual per trade",
) -> go.Figure:
    """Violin of a per-trade metric, split by ``trade_type``.

    Parameters
    ----------
    trades_df
        ``trades_df``-shaped frame (one row per trade).  Must carry
        ``trade_id`` and :paramref:`value_column`.
    trade_type_map
        ``{trade_id: "target" | "elementary"}``, typically derived from
        ``trade_graph.nodes``.  Trades with no entry are dropped — the
        graph payload is the authoritative source for trade type.  When
        the map is empty / ``None`` the chart falls back to an
        aggregate single-violin view.
    value_column
        Column whose distribution is plotted.  Default
        ``"mean_residual"`` — callers can swap to ``"mae"`` /
        ``"rmse"`` without changing anything else.
    y_axis_title
        Y-axis caption.
    """
    if (
        trades_df is None
        or trades_df.empty
        or value_column not in trades_df.columns
        or "trade_id" not in trades_df.columns
    ):
        return empty_figure("No per-trade metrics for this cluster.")

    values = trades_df[value_column].astype(float).to_numpy()
    if values.size == 0:
        return empty_figure("No per-trade metrics for this cluster.")

    if not trade_type_map:
        return _aggregate_trade_violin(values, y_axis_title=y_axis_title)

    trade_types = (
        trades_df["trade_id"].map(trade_type_map).fillna("").to_numpy(dtype=object)
    )
    # Drop any rows we couldn't classify — keeps the violin honest.
    keep = trade_types != ""
    if not keep.any():
        return _aggregate_trade_violin(values, y_axis_title=y_axis_title)
    values = values[keep]
    trade_types = trade_types[keep]

    fig = go.Figure()
    for group in _TRADE_TYPE_ORDER:
        mask = trade_types == group
        if not mask.any():
            continue
        color = _TRADE_TYPE_COLOR[group]
        fig.add_trace(
            go.Violin(
                y=values[mask],
                name=group.capitalize(),
                x=[group.capitalize()] * int(mask.sum()),
                line_color=color,
                fillcolor=rgba(color, 0.2),
                box_visible=True,
                meanline_visible=True,
                points="outliers",
                hoveron="violins",
                hovertemplate=(
                    f"<b>{group.capitalize()}</b><br>"
                    "median: %{median:.4f}<br>"
                    "Q1: %{q1:.4f} / Q3: %{q3:.4f}<br>"
                    "min: %{lowerfence:.4f} / max: %{upperfence:.4f}"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig.update_layout(
        **rade_layout(
            show_legend=False,
            xaxis={
                "title": {"text": "Trade type", "font": {"color": "#94a3b8", "size": 11}},
                "showgrid": False,
            },
            yaxis={"title": {"text": y_axis_title, "font": {"color": "#94a3b8"}}},
        ),
    )
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="rgba(148, 163, 184, 0.4)",
        line_width=1,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────
# Row 4 right — per-trade scatter (mean_residual vs mae, by trade_type)
# ─────────────────────────────────────────────────────────────────────


def per_trade_scatter(
    trades_df:          pd.DataFrame,
    *,
    trade_type_map:     Optional[Mapping[str, str]] = None,
    selected_trade_id:  Optional[str] = None,
    x_column:           str = "mean_residual",
    y_column:           str = "mae",
) -> go.Figure:
    """Per-trade scatter, coloured by ``trade_type``.

    Parameters
    ----------
    trades_df
        ``trades_df``-shaped frame.  Must carry ``trade_id`` plus
        :paramref:`x_column` and :paramref:`y_column`.
    trade_type_map
        ``{trade_id: "target" | "elementary"}``.  Trades not in the
        map render as "unknown" in a muted slate colour — we keep them
        rather than drop them so the user notices when the trade-graph
        payload is incomplete.
    selected_trade_id
        Optional trade id to highlight with an emerald ring + larger
        marker.  Pass ``None`` for no highlight.
    x_column, y_column
        Defaults put ``mean_residual`` on x (bias) and ``mae`` on y
        (magnitude), which diagnoses both systematic drift and error
        scale at a glance.

    Notes
    -----
    Each point's ``customdata`` is ``[trade_id, trade_type]``.  Callbacks
    read ``clickData["points"][0]["customdata"][0]`` to sync the grid /
    session selection.  Hovertemplate uses ``%{customdata[0]}`` for the
    trade id so every point surfaces an identifier on hover.
    """
    required = {"trade_id", x_column, y_column}
    if trades_df is None or trades_df.empty or not required.issubset(trades_df.columns):
        return empty_figure("No per-trade scatter data.")

    df = trades_df.copy()
    if trade_type_map:
        df["trade_type"] = df["trade_id"].map(trade_type_map).fillna("unknown")
    else:
        df["trade_type"] = "unknown"

    fig = go.Figure()
    order = [*_TRADE_TYPE_ORDER, "unknown"]
    seen_any = False
    for group in order:
        sub = df[df["trade_type"] == group]
        if sub.empty:
            continue
        seen_any = True
        color = _TRADE_TYPE_COLOR.get(group, "#64748b")
        fig.add_trace(
            go.Scattergl(
                x=sub[x_column],
                y=sub[y_column],
                mode="markers",
                marker={
                    "size":    7,
                    "color":   color,
                    "opacity": 0.78,
                    "line":    {"width": 0},
                },
                name=group.capitalize(),
                customdata=np.stack(
                    [sub["trade_id"].astype(str).to_numpy(),
                     sub["trade_type"].astype(str).to_numpy()],
                    axis=-1,
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    f"{x_column}: %{{x:.4f}}<br>"
                    f"{y_column}: %{{y:.4f}}<br>"
                    "type: %{customdata[1]}"
                    "<extra></extra>"
                ),
                showlegend=True,
            )
        )

    if not seen_any:
        return empty_figure("No per-trade scatter data.")

    # Highlight the selected trade on top of everything else so it pops.
    if selected_trade_id:
        sel = df[df["trade_id"] == selected_trade_id]
        if not sel.empty:
            fig.add_trace(
                go.Scattergl(
                    x=sel[x_column],
                    y=sel[y_column],
                    mode="markers",
                    marker={
                        "size":    14,
                        "color":   "rgba(16, 185, 129, 0)",
                        "line":    {"color": "#10b981", "width": 2.5},
                    },
                    name=f"Selected · {selected_trade_id}",
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    fig.update_layout(
        **rade_layout(
            show_legend=True,
            hovermode="closest",
            xaxis={"title": {"text": x_column, "font": {"color": "#94a3b8"}}},
            yaxis={"title": {"text": y_column, "font": {"color": "#94a3b8"}}},
        ),
    )
    # x=0 reference line helps read bias at a glance.
    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="rgba(148, 163, 184, 0.4)",
        line_width=1,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _aggregate_trade_violin(
    values:       np.ndarray,
    *,
    y_axis_title: str,
) -> go.Figure:
    """Single-violin fallback when trade_type classification is missing.

    Kept visually consistent with the grouped variant so the chart
    shape doesn't flicker when the trade-graph payload arrives and the
    callback re-renders with a populated map.
    """
    color = color_for_index(0)
    fig = go.Figure()
    fig.add_trace(
        go.Violin(
            y=values,
            x=[""] * values.size,
            name="All",
            line_color=color,
            fillcolor=rgba(color, 0.2),
            box_visible=True,
            meanline_visible=True,
            points="outliers",
            hoveron="violins",
            showlegend=False,
        )
    )
    fig.update_layout(
        **rade_layout(
            show_legend=False,
            xaxis={"visible": False},
            yaxis={"title": {"text": y_axis_title, "font": {"color": "#94a3b8"}}},
        ),
    )
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="rgba(148, 163, 184, 0.4)",
        line_width=1,
    )
    return fig


__all__ = [
    "per_trade_residual_violin",
    "per_trade_scatter",
    "predicted_vs_actual_band",
]
```

## Appendix B — `assets/js/evaluation.js`

Clientside callbacks for the Evaluation page.  Lives in
`src/ui/apps/rade_analytics/assets/js/evaluation.js` so Dash auto-loads
it at startup (every file under `assets/` is served and injected by
the framework).

Wired from Python via:

```python
from dash import ClientsideFunction

app.clientside_callback(
    ClientsideFunction(namespace="evaluation",
                       function_name="update_filter_ui"),
    Output(EVAL_FILTER_IDS["chips"],         "children"),
    Output(EVAL_FILTER_IDS["toggle_label"],  "children"),
    Output(EVAL_FILTER_IDS["clear_all"],     "style"),
    Input(EVAL_FILTER_IDS["asset_class"],    "value"),
    Input(EVAL_FILTER_IDS["currency"],       "value"),
    Input(EVAL_FILTER_IDS["desk"],           "value"),
    Input(EVAL_FILTER_IDS["product"],        "value"),
    Input(EVAL_FILTER_IDS["date_range"],     "value"),
)
```

The DOM trees produced here MUST match the structure produced by
`components.evaluation_filter_bar.render_filter_chips` on the Python
side, otherwise the initial server-side render and the post-mount
clientside updates will diverge visually.  Both paths render each chip
as:

```html
<div class="rade-filter-chip">
  <span class="rade-filter-chip-label">{label}: {value}</span>
  <button id='{"type":"eval-filter-chip-close","dimension":"{dim}"}'
          class="rade-filter-chip-close"
          aria-label="Remove {dim} filter">×</button>
</div>
```

Visual styling lives in `rade.css` under `.rade-filter-chip*`.

```javascript
/* ──────────────────────────────────────────────────────────────────
 * Evaluation page — clientside callbacks.
 *
 * Runs entirely in the browser; no server round-trips.  Functions are
 * referenced from Python via:
 *
 *     ClientsideFunction(namespace="evaluation",
 *                        function_name="update_filter_ui")
 *
 * Page Contract reference: §6 Lever P2 (clientside_callback for trivial
 * UI), §3 Rule L1 (initial render still happens server-side via
 * render_filter_chips so the page is usable before any callback fires).
 *
 * The DOM trees produced here MUST match the structure produced by
 * components.evaluation_filter_bar.render_filter_chips on the Python
 * side, otherwise the initial render and the post-mount updates will
 * diverge visually.  Both paths render the chip as
 *
 *     <div class="rade-filter-chip">
 *       <span class="rade-filter-chip-label">{label}: {value}</span>
 *       <button id={type:"eval-filter-chip-close",dimension:dim}
 *               class="rade-filter-chip-close"
 *               aria-label="Remove {dim} filter">×</button>
 *     </div>
 *
 * All visual styling lives in rade.css under `.rade-filter-chip*`.
 * ────────────────────────────────────────────────────────────────── */

window.dash_clientside = window.dash_clientside || {};
window.dash_clientside.evaluation = (function () {
    "use strict";

    /* Human-readable label per dimension; mirrors _CHIP_LABELS in
     * components/evaluation_filter_bar.py.  Keep the two in sync. */
    const CHIP_LABELS = {
        asset_class: "Asset",
        currency:    "CCY",
        desk:        "Desk",
        product:     "Product",
        date:        "Date",
    };

    /* Build a Dash component dict for one chip.  Using the html.* layer
     * (dash_html_components) instead of dmc.Badge keeps us free of
     * Mantine-internal serialisation; CSS does the visual work. */
    function makeChip(dimension, text) {
        return {
            namespace: "dash_html_components",
            type: "Div",
            props: {
                className: "rade-filter-chip",
                children: [
                    {
                        namespace: "dash_html_components",
                        type: "Span",
                        props: {
                            className: "rade-filter-chip-label",
                            children: text,
                        },
                    },
                    {
                        namespace: "dash_html_components",
                        type: "Button",
                        props: {
                            id: {
                                type: "eval-filter-chip-close",
                                dimension: dimension,
                            },
                            className: "rade-filter-chip-close",
                            children: "\u00D7",  // multiplication-sign ×
                            "aria-label": "Remove " + dimension + " filter",
                        },
                    },
                ],
            },
        };
    }

    /* Compose the value-text for a multi-value dimension chip — first
     * three values verbatim, "+N" overflow indicator after that. */
    function formatValueList(values) {
        if (!Array.isArray(values) || values.length === 0) return "";
        const head = values.slice(0, 3).join(", ");
        const overflow = values.length > 3 ? " +" + (values.length - 3) : "";
        return head + overflow;
    }

    /* Compose the "from – to" text for the date range chip.  Mirrors
     * the Python helper in render_filter_chips. */
    function formatDateRange(dateRange) {
        if (!Array.isArray(dateRange)) return null;
        const from = dateRange[0] || null;
        const to   = dateRange[1] || null;
        if (!from && !to) return null;
        if (from && to)   return from + "\u2013" + to;  // en-dash –
        return from || to;
    }

    /* Public entry point.  Inputs are the raw values from the five
     * filter controls (four MultiSelects + DatePickerInput).  Returns
     * a 3-tuple matching the Output ordering in evaluation_cb.py:
     *
     *     [chips, toggleLabel, clearAllStyle]
     *
     * - chips: list of chip component dicts (or [] for none).
     * - toggleLabel: "{n} active" or "" when no filters are set.
     * - clearAllStyle: {} (visible) when ≥1 filter, {display: "none"}
     *                  when none.  Inline style is the simplest
     *                  visibility lever; CSS could replace this later. */
    function update_filter_ui(asset_class, currency, desk, product, dateRange) {
        const dimensions = [
            ["asset_class", asset_class],
            ["currency",    currency],
            ["desk",        desk],
            ["product",     product],
        ];

        const chips = [];
        let activeCount = 0;

        for (const [dim, values] of dimensions) {
            if (Array.isArray(values) && values.length > 0) {
                activeCount += 1;
                const text = CHIP_LABELS[dim] + ": " + formatValueList(values);
                chips.push(makeChip(dim, text));
            }
        }

        const dateText = formatDateRange(dateRange);
        if (dateText) {
            activeCount += 1;
            chips.push(makeChip("date", "Date: " + dateText));
        }

        const toggleLabel  = activeCount > 0 ? activeCount + " active" : "";
        const clearAllStyle = activeCount > 0 ? {} : { display: "none" };

        return [chips, toggleLabel, clearAllStyle];
    }

    return {
        update_filter_ui: update_filter_ui,
    };
})();
```

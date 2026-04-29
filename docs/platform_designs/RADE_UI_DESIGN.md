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

## Appendix A — Cluster Deep-Dive Phase E.5 callback wiring

Single appendix (replaces all prior appendix blocks). Three files;
each replaces an existing file in your work tree wholesale — paste the
contents into the path called out in the section header and you're
done.

### How to apply

1. **A.1 — `figures/cluster_deep_dive_charts.py`** — replace the file in
   place. Adds three Phase E.5 builders (`per_trade_residual_histogram`,
   `per_trade_bias_scatter`, `elementary_pnl_multiline`) alongside the
   Phase E.4 ones, which stay for backwards compatibility.
2. **A.2 — `figures/__init__.py`** — replace the file in place. Re-exports
   the three new builders so callbacks can `from ..figures import …` them.
3. **A.3 — `callbacks/cluster_deep_dive_cb.py`** — replace the file in
   place. Full rewrite for the Phase E.5 layout: 5 capture callbacks +
   8 render callbacks, no duplicate outputs.

No layout / CSS / `KpiCard` / session changes are required for this
patch — those landed in the previous appendix and the layout file
already calls `KpiCard(..., sparkline_id=…)`. After pasting all three
files restart the dashboard. Smoke check: registering the cluster-
deep-dive callbacks should yield 13 entries in `app.callback_map`,
44 across the whole app.

### A.1 — `src/ui/apps/rade_analytics/figures/cluster_deep_dive_charts.py` (full file)

Adds the three new Phase E.5 figure builders **at the bottom** of the
existing Phase E.4 file, plus an updated module docstring + `__all__`.
Easiest is to paste the whole file:

```python
"""Cluster Deep-Dive specific figure builders (Phase E.4 + E.5).

Six builders land here, all tuned to single-cluster diagnostic views:

* :func:`predicted_vs_actual_band` — *Phase E.4 legacy.*  Predicted +
  actual PnL lines for one cluster with the residual zone shaded (rose)
  so the user can *see* where the model is over- or under-predicting.
  Distinct from :func:`figures.portfolio_pnl`, which has no error
  shading and is used on the Portfolio tab.
* :func:`per_trade_residual_violin` — *Phase E.4 legacy.*  Distribution
  of per-trade metrics (mean_residual or mae), split by ``trade_type``
  (target / elementary).  Uses :class:`plotly.graph_objects.Violin` so
  the visual language matches the Portfolio residual violin.
* :func:`per_trade_scatter` — *Phase E.4 legacy.*  Per-trade aggregate
  scatter: ``mean_residual`` (x) vs ``mae`` (y), coloured by
  ``trade_type``.  A ``selected_trade_id`` gets a larger, emerald-
  bordered marker so cross-highlighting from the trades AgGrid is
  visible at a glance.
* :func:`per_trade_residual_histogram` — *Phase E.5 Row 3 left.*
  Histogram of (predicted − target) for **one** target trade across
  every scenario in the active split.  Replaces the per-cluster violin
  with a per-trade detail view that diagnoses systematic drift +
  heavy-tailed residuals at a glance.
* :func:`per_trade_bias_scatter` — *Phase E.5 Row 3 right.*  Per-scenario
  scatter for **one** target trade: x = predicted PnL, y = residual
  (predicted − target), colour-coded by ``|residual|`` so the user can
  spot under-/over-predicted clusters of scenarios immediately.
* :func:`elementary_pnl_multiline` — *Phase E.5 Row 4.*  One line per
  selected elementary trade, x = scenario index, y = raw PnL value.
  Drives the Elementary PnL Explorer's right pane.

All builders share the same data contract: callers pass a pandas frame
with the columns listed in each function's docstring, and a
``trade_type_map`` (``{trade_id: "target" | "elementary"}``) derived
from the trade-graph payload where relevant.  Missing columns / empty
frames gracefully return an :func:`empty_figure` — the UI never shows
a broken axis.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ._theme import color_for_index, empty_figure, rade_layout, rgba, sort_chronologically

# Fixed colours for the two trade types so violin + scatter + Cytoscape
# legend always match.  Must stay in sync with the Cytoscape stylesheet
# in ``layouts/evaluation/trade_graph.py``.
_TRADE_TYPE_COLOR = {
    "target":     "#f59e0b",   # amber
    "elementary": "#8b5cf6",   # violet
}
_TRADE_TYPE_ORDER: tuple[str, ...] = ("target", "elementary")


# ─────────────────────────────────────────────────────────────────────
# Row 2 right — predicted vs actual with shaded error band
# ─────────────────────────────────────────────────────────────────────


def predicted_vs_actual_band(
    df: pd.DataFrame,
    *,
    uirevision_key: Optional[str] = None,
) -> go.Figure:
    """Two-line PnL chart with the residual zone shaded rose.

    Parameters
    ----------
    df
        Single-cluster timeseries frame.  Required columns:
        ``predictions``, ``targets``.  Optional but preferred:
        ``scenario_idx`` (drives sort order) and ``scenario_label``
        (drives the x-axis tick labels).  Extra columns are ignored.
    uirevision_key
        Optional Plotly ``uirevision`` value.  Page Contract §6 — when
        the same key recurs across re-renders the user's zoom / pan
        / legend toggles are preserved; a different key resets UI
        state to the new data domain.  Callers typically pass
        ``f"{split}::{cluster_id}"`` so navigating across clusters /
        splits resets, but in-place re-renders (e.g. session-store
        churn from another widget) preserve the user's view.

    Notes
    -----
    The shaded band is built as ``predictions`` fill-to-zero *minus*
    ``targets`` fill-to-zero — i.e. two scatter traces with
    ``fill='tonexty'`` form the envelope between the two lines.  Using
    ``rgba(0.18)`` for the fill keeps the band visible against the
    dark theme without drowning the line strokes.
    """
    if (
        df is None
        or df.empty
        or "predictions" not in df.columns
        or "targets" not in df.columns
    ):
        return empty_figure("No cluster timeseries for this selection.")

    # Train split is shuffled at fit time; sort by parsed scenario_label so
    # the predicted/actual bands read chronologically left-to-right.
    df_sorted = sort_chronologically(df)
    x_vals: Sequence
    if "scenario_label" in df_sorted.columns:
        x_vals = df_sorted["scenario_label"].tolist()
    else:
        x_vals = list(range(len(df_sorted)))

    pred_color = color_for_index(0)       # violet
    actual_color = "#cbd5e1"              # slate-300 — neutral reference
    band_color = color_for_index(3)       # rose — "error zone"

    fig = go.Figure()

    # Baseline trace — predictions.  Rendered first so the "targets"
    # trace below can fill-to-next and produce the between-lines band.
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=df_sorted["predictions"],
            mode="lines",
            name="Predicted PnL",
            line={"color": pred_color, "width": 2.5},
            hovertemplate="%{y:.4f}<extra>Predicted</extra>",
        )
    )
    # Error band — targets with fill='tonexty' fills the area between
    # the two traces.
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=df_sorted["targets"],
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
    if uirevision_key is not None:
        fig.update_layout(uirevision=uirevision_key)
    return fig


# ─────────────────────────────────────────────────────────────────────
# Row 3 left — per-trade residual violin (target vs elementary)
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
# Row 3 right — per-trade scatter (mean_residual vs mae, by trade_type)
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


# ─────────────────────────────────────────────────────────────────────
# Phase E.5 Row 3 left — per-trade residual histogram (per scenario)
# ─────────────────────────────────────────────────────────────────────


def per_trade_residual_histogram(
    predictions:    np.ndarray,
    targets:        np.ndarray,
    *,
    trade_id:       Optional[str] = None,
    n_bins:         int = 40,
    uirevision_key: Optional[str] = None,
) -> go.Figure:
    """Histogram of ``predictions − targets`` for one trade.

    Parameters
    ----------
    predictions, targets
        1-D numpy arrays of equal length, one entry per scenario for
        the *single* trade currently in focus.  The column-slicing
        from the cluster-level NPZ (``predictions[:, target_idx]``)
        happens in the caller — by the time we get here the input is
        already trade-shaped.
    trade_id
        Optional id to thread through to the trace name; surfaces
        on hover when the user has multiple traces overlaid (we don't
        today, but the contract is symmetric with the bias scatter).
    n_bins
        Histogram bin count.  40 is the design-spec default and reads
        well at the 320-px row-3 chart height; callers can override
        for very long tails.
    uirevision_key
        See :func:`predicted_vs_actual_band`'s docstring for the
        contract — typically ``f"{split}::{cluster_id}::{trade_id}"``
        so toggling between trades resets zoom but session-store churn
        from another widget preserves it.
    """
    if (
        predictions is None or targets is None
        or predictions.size == 0 or targets.size == 0
        or predictions.shape != targets.shape
    ):
        return empty_figure("No per-scenario data for this trade.")

    residuals = (predictions - targets).astype(float)
    finite_mask = np.isfinite(residuals)
    if not finite_mask.any():
        return empty_figure("Residuals are all non-finite for this trade.")
    residuals = residuals[finite_mask]

    primary = color_for_index(0)
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=residuals,
            nbinsx=int(n_bins),
            marker={
                "color":    rgba(primary, 0.55),
                "line":     {"color": primary, "width": 1.0},
            },
            name=trade_id or "Residual",
            hovertemplate=(
                "residual: %{x:.4f}<br>"
                "scenarios: %{y}"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )
    fig.add_vline(
        x=float(np.mean(residuals)),
        line_dash="dot",
        line_color="rgba(16, 185, 129, 0.8)",   # emerald — mean marker
        line_width=1.5,
        annotation_text="mean",
        annotation_position="top left",
        annotation_font_color="#10b981",
    )
    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="rgba(148, 163, 184, 0.5)",
        line_width=1,
    )
    fig.update_layout(
        **rade_layout(
            show_legend=False,
            hovermode="x",
            xaxis={
                "title": {
                    "text": "Residual (predicted − target)",
                    "font": {"color": "#94a3b8"},
                },
            },
            yaxis={
                "title": {"text": "Scenarios", "font": {"color": "#94a3b8"}},
            },
        ),
        bargap=0.04,
    )
    if uirevision_key is not None:
        fig.update_layout(uirevision=uirevision_key)
    return fig


# ─────────────────────────────────────────────────────────────────────
# Phase E.5 Row 3 right — per-scenario bias-vs-magnitude scatter
# ─────────────────────────────────────────────────────────────────────


def per_trade_bias_scatter(
    predictions:    np.ndarray,
    targets:        np.ndarray,
    *,
    trade_id:       Optional[str] = None,
    uirevision_key: Optional[str] = None,
) -> go.Figure:
    """Per-scenario scatter for one trade — predictions vs residuals.

    Parameters
    ----------
    predictions, targets
        Same trade-shaped 1-D arrays as
        :func:`per_trade_residual_histogram`.  The caller handles
        column-slicing the NPZ.
    trade_id
        Optional trade id for the hover footer.  No visual effect
        when omitted.
    uirevision_key
        See :func:`per_trade_residual_histogram` — typically the same
        key so both row-3 figures share zoom-reset semantics.

    Notes
    -----
    The colour scale runs from cool (low ``|residual|``) to warm
    (high ``|residual|``) so the user can find the pathological
    scenarios at a glance.  Plotly's ``RdYlBu_r`` is the closest
    perceptually-uniform diverging scale to the rest of the
    dashboard's diagnostic charts.
    """
    if (
        predictions is None or targets is None
        or predictions.size == 0 or targets.size == 0
        or predictions.shape != targets.shape
    ):
        return empty_figure("No per-scenario data for this trade.")

    pred_arr = predictions.astype(float)
    tgt_arr  = targets.astype(float)
    residuals = pred_arr - tgt_arr
    abs_res = np.abs(residuals)

    finite_mask = (
        np.isfinite(pred_arr) & np.isfinite(tgt_arr) & np.isfinite(residuals)
    )
    if not finite_mask.any():
        return empty_figure("Residuals are all non-finite for this trade.")

    pred_arr = pred_arr[finite_mask]
    residuals = residuals[finite_mask]
    abs_res = abs_res[finite_mask]
    scenario_idx = np.arange(pred_arr.size, dtype=int)

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=pred_arr,
            y=residuals,
            mode="markers",
            marker={
                "size":       6,
                "color":      abs_res,
                "colorscale": "RdYlBu_r",
                "showscale":  True,
                "colorbar":   {
                    "title":       {"text": "|residual|", "side": "right"},
                    "thickness":   8,
                    "outlinewidth": 0,
                    "tickfont":    {"color": "#94a3b8", "size": 10},
                },
                "opacity":    0.78,
                "line":       {"width": 0},
            },
            customdata=np.stack([scenario_idx, abs_res], axis=-1),
            name=trade_id or "Per-scenario",
            hovertemplate=(
                "scenario %{customdata[0]}<br>"
                "predicted: %{x:.4f}<br>"
                "residual:  %{y:.4f}<br>"
                "|residual|: %{customdata[1]:.4f}"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="rgba(148, 163, 184, 0.5)",
        line_width=1,
    )
    fig.update_layout(
        **rade_layout(
            show_legend=False,
            hovermode="closest",
            xaxis={
                "title": {
                    "text": "Predicted PnL",
                    "font": {"color": "#94a3b8"},
                },
            },
            yaxis={
                "title": {
                    "text": "Residual (predicted − target)",
                    "font": {"color": "#94a3b8"},
                },
            },
        ),
    )
    if uirevision_key is not None:
        fig.update_layout(uirevision=uirevision_key)
    return fig


# ─────────────────────────────────────────────────────────────────────
# Phase E.5 Row 4 — elementary PnL multi-line timeseries
# ─────────────────────────────────────────────────────────────────────


def elementary_pnl_multiline(
    df:             pd.DataFrame,
    *,
    uirevision_key: Optional[str] = None,
) -> go.Figure:
    """Multi-line PnL chart, one trace per selected elementary trade.

    Parameters
    ----------
    df
        Wide DataFrame from :meth:`RadeBackend.elementary_pnl_df` —
        index = scenario index (int), one column per elementary
        trade id.  An empty / all-NaN frame falls through to an
        :func:`empty_figure` placeholder so the empty-state branch
        is identical for "no selection" and "all selections returned
        empty".
    uirevision_key
        See :func:`predicted_vs_actual_band` — typically
        ``f"{cluster_id}::{','.join(sorted(trade_ids))}"`` so adding
        / removing an elementary trade resets zoom while session-
        store churn from elsewhere preserves it.
    """
    if df is None or df.empty:
        return empty_figure(
            "Pick one or more elementary trades to plot their PnL."
        )

    x_vals = df.index.tolist()
    fig = go.Figure()
    plotted = 0
    for i, col in enumerate(df.columns):
        series = df[col]
        if series.dropna().empty:
            continue
        color = color_for_index(i)
        fig.add_trace(
            go.Scattergl(
                x=x_vals,
                y=series.tolist(),
                mode="lines",
                name=str(col),
                line={"color": color, "width": 1.6},
                hovertemplate=(
                    f"<b>{col}</b><br>"
                    "scenario %{x}<br>"
                    "PnL: %{y:.4f}"
                    "<extra></extra>"
                ),
                showlegend=True,
            )
        )
        plotted += 1

    if plotted == 0:
        return empty_figure(
            "Selected elementary trades have no PnL data on this split."
        )

    fig.update_layout(
        **rade_layout(
            show_legend=True,
            hovermode="x unified",
            xaxis={
                "title": {
                    "text": "Scenario index",
                    "font": {"color": "#94a3b8"},
                },
            },
            yaxis={
                "title": {"text": "Elementary PnL", "font": {"color": "#94a3b8"}},
            },
        ),
    )
    if uirevision_key is not None:
        fig.update_layout(uirevision=uirevision_key)
    return fig


__all__ = [
    "elementary_pnl_multiline",
    "per_trade_bias_scatter",
    "per_trade_residual_histogram",
    "per_trade_residual_violin",
    "per_trade_scatter",
    "predicted_vs_actual_band",
]
```

### A.2 — `src/ui/apps/rade_analytics/figures/__init__.py` (full file)

Re-exports the three new builders plus updates `__all__`. Replace the
whole file:

```python
"""Figure builders for the Rade Analytics Dash UI.

This is the single place every callback goes to for a Plotly
``go.Figure``.  Keeping chart construction out of the callback modules
means:

* Callbacks stay focused on fetch + state plumbing.
* Figures can be unit-tested headlessly (``fig.to_dict()`` snapshots).
* Visual tweaks land in one diff — no hunting through every callback
  module for a font-size bump.

Modules shipped
---------------
* :mod:`._theme`                   — shared layout defaults, palette, helpers.
* :mod:`.cluster_deep_dive_charts` — per-cluster PnL band + per-trade
  violin / scatter (Phase E.4).
* :mod:`.distributions`            — residual violin (aggregate + grouped).
* :mod:`.graph_charts`             — graph density histogram + edges vs nodes.
* :mod:`.scatter`                  — predicted-vs-actual scatter (+ focus).
* :mod:`.timeseries`               — portfolio PnL + rolling error band.
* :mod:`.trade_graph_stylesheet`   — Cytoscape stylesheet + legend body
  builders for the Trade-Graph color-by toggle (Phase E.3 rebuild).
* :mod:`.training_curves`          — per-cluster training loss + metric
  overlays (Phase E.4 Row 2).
"""
from __future__ import annotations

from ._theme import (
    CATEGORY_PALETTE,
    color_for_index,
    empty_figure,
    rade_layout,
    rgba,
)
from .cluster_deep_dive_charts import (
    elementary_pnl_multiline,
    per_trade_bias_scatter,
    per_trade_residual_histogram,
    per_trade_residual_violin,
    per_trade_scatter,
    predicted_vs_actual_band,
)
from .distributions import residual_violin
from .graph_charts import density_distribution, edges_vs_nodes_scatter
from .scatter import pred_actual_scatter
from .timeseries import error_over_time, portfolio_pnl
from .trade_graph_stylesheet import build_legend_body, build_stylesheet
from .training_curves import training_curves_chart

__all__ = [
    "CATEGORY_PALETTE",
    "build_legend_body",
    "build_stylesheet",
    "color_for_index",
    "density_distribution",
    "edges_vs_nodes_scatter",
    "elementary_pnl_multiline",
    "empty_figure",
    "error_over_time",
    "per_trade_bias_scatter",
    "per_trade_residual_histogram",
    "per_trade_residual_violin",
    "per_trade_scatter",
    "portfolio_pnl",
    "pred_actual_scatter",
    "predicted_vs_actual_band",
    "rade_layout",
    "residual_violin",
    "rgba",
    "training_curves_chart",
]
```

### A.3 — `src/ui/apps/rade_analytics/callbacks/cluster_deep_dive_cb.py` (full file)

Full rewrite of the callback module against the Phase E.5 layout
ids. **Replace the whole file** in your work tree:

```python
"""Evaluation → Cluster Deep-Dive sub-tab callbacks (Phase E.5 hybrid).

Page Contract structure
-----------------------
The public surface is a single :func:`register` that delegates to two
section helpers, matching Page Contract §2 (capture / render split):

* :func:`_register_capture` — user-input gestures and cross-page
  navigation → :class:`Session` writes (no UI side-effects).
* :func:`_register_render`  — derived state → DOM updates (no
  :class:`Session` writes, except for the bootstrap capture-edge
  described below).

Capture (5 callbacks)
~~~~~~~~~~~~~~~~~~~~~
* ``_sync_selection``               — cluster picker, trades-grid row
  click, clear-chip button → ``deep_dive_cluster_id`` /
  ``deep_dive_selected_trade_id``.
* ``_sync_curve_metrics``           — overlay-metric chip group
  ``value`` → ``deep_dive_curve_metrics``.
* ``_sync_elementary_selection``    — Elementary PnL Explorer
  ``selectedRows`` and reset button → ``deep_dive_elementary_trade_ids``.
* ``_navigate_to_trade_graph``      — "Trade-Graph" button on this
  page → ``/evaluation/trade-graph`` with the active cluster pinned.
* ``_navigate_from_trade_graph``    — "Open in Cluster Deep Dive" on
  the Trade-Graph tab → ``/evaluation/cluster`` with the trade-
  graph's cluster + selected trade copied into the deep-dive slots.

Render (8 callbacks)
~~~~~~~~~~~~~~~~~~~~
* ``_bootstrap``                    — mount-signal-triggered fetch of
  the cluster ``Select.data``; coalesces the URL ``?cid=`` deep-link
  + fresh-user defaults into the same round-trip.
* ``_render_attributes``            — session → cluster-attributes
  card body, graph-statistics card body, "Trade-Graph" button enabled
  state, ``store_trade_types`` (the canonical trade-type map +
  ordered ``target_ids`` / ``elementary_ids`` lists for downstream
  callbacks).
* ``_render_kpis``                  — session → 4 KPI values
  (MAE / RMSE / P95 / P99) + 4 sparkline figures showing the per-trade
  distribution shape across the cluster.
* ``_render_timeseries``            — session → cluster portfolio
  chart (predicted vs target line) + residual-over-time chart.
* ``_render_training_curves``       — session → training-curves
  figure + chip group children + chip empty-state + curve-metrics
  store.
* ``_render_grids``                 — session + store_trade_types →
  Trade-Level Metrics grid (all trades) and Elementary PnL Explorer
  grid (filtered to elementary).  Also re-asserts the elementary
  ``selectedRows`` from session so deep-links / browser-back paint
  correctly.
* ``_render_per_trade_detail``      — session + store_trade_types →
  Row 3 wrapper visibility + selected-trade chip label + per-trade
  residual histogram + bias-vs-magnitude scatter.  Lazy-loads the
  per-cluster predictions NPZ on demand via ``backend.predictions``.
* ``_render_elementary_pnl``        — session → empty-state vs chart
  visibility on Row 4 right + elementary-pnl multi-line chart figure.

Data contract
-------------
Trade-type classification comes from the trade-graph endpoint and is
shared across the page via ``store_trade_types``::

    {
        "types":          {trade_id: "target" | "elementary"},
        "target_ids":     [tid, …],     # NPZ column ordering
        "elementary_ids": [tid, …],
    }

Why the ordered ``target_ids`` list?  The predictions NPZ is shaped
``(n_scenarios, n_target_trades_in_cluster)``; column ``i`` corresponds
to ``target_ids[i]``.  Threading the list through the store means the
per-trade detail callback never re-fetches the trade graph just to
recover ordering — the cached fetch in :func:`_render_attributes` is
the single source of truth.

Every fetch goes through :class:`RadeBackend`; the cache layer there
coalesces duplicate requests within a render tick.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs

import dash_mantine_components as dmc
import numpy as np
import pandas as pd
from dash import Input, Output, State, ctx, html, no_update
from dash.exceptions import PreventUpdate

from ..data.result_helpers import figure_with_fallback
from ..data.session import Session
from ..figures import (
    elementary_pnl_multiline,
    empty_figure,
    error_over_time,
    per_trade_bias_scatter,
    per_trade_residual_histogram,
    portfolio_pnl,
    training_curves_chart,
)
from ..layouts.evaluation.cluster_deep_dive import CLUSTER_DEEP_DIVE_IDS
from ..layouts.evaluation.trade_graph import TRADE_GRAPH_IDS
from ..layouts.shell import SHELL_IDS

if TYPE_CHECKING:
    from dash import Dash

    from ..data.backend import RadeBackend


logger = logging.getLogger(__name__)


_DEEP_DIVE_PATH   = "/evaluation/cluster"
_TRADE_GRAPH_PATH = "/evaluation/trade-graph"
_PLACEHOLDER      = "—"

# Visual order in the Cluster Attributes card.  Matches the mock
# screenshot top-down.  Each entry is ``(label, candidate_columns)``
# — the first candidate column present in the clusters_df row wins,
# so we silently degrade when the ensemble's attribute schema drops
# an entry without leaving a phantom row in the UI.
_ATTRIBUTE_ROWS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("Asset Class",  ("asset_class", "AssetClassCode")),
    ("Currency",     ("currency_code", "CurrencyCode", "currency")),
    ("Desk",         ("desk", "DeskCode", "desk_code")),
    ("Product",      ("product_code", "ProductCode", "product")),
    ("N Trades",     ("n_trades",)),
    ("N Scenarios", ("n_scenarios",)),
)

# Graph-stats rows.  The first three carry real values from
# ``trade_graph.stats``; the last two are deferred and always render
# as ``—`` so the card visual structure matches the mock.
_GRAPH_STATS_ROWS: Tuple[Tuple[str, Optional[str]], ...] = (
    ("Nodes",           "n_nodes"),
    ("Edges",           "n_edges"),
    ("Density",         "density"),
    ("Avg Degree",      None),
    ("Avg Path Length", None),
)


# ─────────────────────────────────────────────────────────────────────
# Formatters
# ─────────────────────────────────────────────────────────────────────


def _fmt_float(x: Optional[float], *, precision: int = 4) -> str:
    if x is None:
        return _PLACEHOLDER
    try:
        val = float(x)
    except (TypeError, ValueError):
        return _PLACEHOLDER
    if pd.isna(val):
        return _PLACEHOLDER
    return f"{val:.{precision}f}"


def _fmt_int(x: Any) -> str:
    if x is None:
        return _PLACEHOLDER
    try:
        val = int(x)
    except (TypeError, ValueError):
        try:
            val = int(float(x))
        except (TypeError, ValueError):
            return _PLACEHOLDER
    return f"{val:,}"


def _fmt_density(x: Optional[float]) -> str:
    if x is None:
        return _PLACEHOLDER
    try:
        val = float(x)
    except (TypeError, ValueError):
        return _PLACEHOLDER
    if pd.isna(val):
        return _PLACEHOLDER
    return f"{val:.3f}"


def _nullable_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return None
    if pd.isna(fv):
        return None
    return fv


def _parse_cid_from_search(search: Optional[str]) -> Optional[str]:
    """Extract the ``?cid=`` query param from the URL search string."""
    if not search:
        return None
    try:
        params = parse_qs(search.lstrip("?"))
    except (ValueError, TypeError):
        return None
    values = params.get("cid")
    if not values:
        return None
    candidate = values[0]
    return candidate if isinstance(candidate, str) and candidate else None


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────


def register(app: "Dash", backend: "RadeBackend") -> None:
    """Attach every Cluster-Deep-Dive sub-tab callback to ``app``."""
    _register_capture(app)
    _register_render(app, backend)


def _register_capture(app: "Dash") -> None:
    """Capture-side callbacks (input gestures → session writes only)."""
    _register_sync_selection(app)
    _register_sync_curve_metrics(app)
    _register_sync_elementary_selection(app)
    _register_navigate_to_trade_graph(app)
    _register_navigate_from_trade_graph(app)


def _register_render(app: "Dash", backend: "RadeBackend") -> None:
    """Render-side callbacks (state → DOM, no session writes except bootstrap)."""
    _register_bootstrap(app, backend)
    _register_render_attributes(app, backend)
    _register_render_kpis(app, backend)
    _register_render_timeseries(app, backend)
    _register_render_training_curves(app, backend)
    _register_render_grids(app, backend)
    _register_render_per_trade_detail(app, backend)
    _register_render_elementary_pnl(app, backend)


# ═════════════════════════════════════════════════════════════════════
# 1. Bootstrap — populate Select option list + resolve initial cluster
# ═════════════════════════════════════════════════════════════════════


def _register_bootstrap(app: "Dash", backend: "RadeBackend") -> None:
    @app.callback(
        Output(CLUSTER_DEEP_DIVE_IDS["cluster_select"], "data"),
        Output(CLUSTER_DEEP_DIVE_IDS["cluster_select"], "value",
               allow_duplicate=True),
        Output(SHELL_IDS["session_store"],              "data",
               allow_duplicate=True),
        Input(CLUSTER_DEEP_DIVE_IDS["mount_signal"],    "data"),
        State(SHELL_IDS["url"],                         "search"),
        State(SHELL_IDS["url"],                         "pathname"),
        State(SHELL_IDS["session_store"],               "data"),
        prevent_initial_call="initial_duplicate",
    )
    def _bootstrap(
        _mount_signal: Any,
        search:        Optional[str],
        pathname:      Optional[str],
        session_data:  Optional[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, str]], Any, Any]:
        if pathname != _DEEP_DIVE_PATH:
            raise PreventUpdate

        res = backend.clusters_df()
        if not res.ok or res.data is None or res.data.empty:
            return [], no_update, no_update

        session = Session.from_store(session_data)
        df = res.data
        options = [
            {"value": cid, "label": cid}
            for cid in sorted(df["cluster_id"].unique())
        ]
        valid_ids = {o["value"] for o in options}

        layout_seed = (
            session.evaluation.deep_dive_cluster_id or session.cluster_id
        )
        url_override = _parse_cid_from_search(search)
        canonical = (
            url_override
            or session.evaluation.deep_dive_cluster_id
            or session.cluster_id
            or (options[0]["value"] if options else None)
        )
        if canonical not in valid_ids:
            canonical = options[0]["value"] if options else None

        if canonical == layout_seed:
            return options, no_update, no_update

        session.evaluation.deep_dive_cluster_id = canonical
        return options, canonical, session.to_store()


# ═════════════════════════════════════════════════════════════════════
# 2. Sync picker / grid-click / clear-btn → session
# ═════════════════════════════════════════════════════════════════════


def _register_sync_selection(app: "Dash") -> None:
    @app.callback(
        Output(SHELL_IDS["session_store"],                       "data",
               allow_duplicate=True),
        Input(CLUSTER_DEEP_DIVE_IDS["cluster_select"],           "value"),
        Input(CLUSTER_DEEP_DIVE_IDS["trades_grid"],              "cellClicked"),
        Input(CLUSTER_DEEP_DIVE_IDS["selected_trade_clear_btn"], "n_clicks"),
        State(SHELL_IDS["session_store"],                        "data"),
        prevent_initial_call=True,
    )
    def _sync(
        cluster:      Optional[str],
        cell_clicked: Optional[Dict[str, Any]],
        clear_clicks: Optional[int],
        session_data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        trigger = ctx.triggered_id
        if trigger is None:
            raise PreventUpdate

        session = Session.from_store(session_data)
        ev = session.evaluation
        changed = False

        if trigger == CLUSTER_DEEP_DIVE_IDS["cluster_select"]:
            new_cluster = cluster if cluster else None
            if ev.deep_dive_cluster_id != new_cluster:
                ev.deep_dive_cluster_id = new_cluster
                # Cluster change invalidates per-trade + per-scenario
                # selections — those ids almost certainly don't live
                # in the new cluster.  Overlay-metric chips also reset
                # because metric availability is per-cluster.
                ev.deep_dive_selected_trade_id = None
                ev.deep_dive_elementary_trade_ids = []
                ev.deep_dive_curve_metrics = []
                changed = True

        elif trigger == CLUSTER_DEEP_DIVE_IDS["trades_grid"]:
            trade_id = _trade_id_from_cell(cell_clicked)
            if trade_id and ev.deep_dive_selected_trade_id != trade_id:
                ev.deep_dive_selected_trade_id = trade_id
                changed = True

        elif trigger == CLUSTER_DEEP_DIVE_IDS["selected_trade_clear_btn"]:
            if clear_clicks and ev.deep_dive_selected_trade_id is not None:
                ev.deep_dive_selected_trade_id = None
                changed = True

        if not changed:
            raise PreventUpdate
        return session.to_store()


def _trade_id_from_cell(cell_data: Optional[Dict[str, Any]]) -> Optional[str]:
    """Pull ``trade_id`` out of AgGrid's ``cellClicked`` payload."""
    if not cell_data:
        return None
    row = cell_data.get("data") or {}
    tid = row.get("trade_id")
    return tid if isinstance(tid, str) and tid else None


# ═════════════════════════════════════════════════════════════════════
# 3. Overlay-metric chip group → session.deep_dive_curve_metrics
# ═════════════════════════════════════════════════════════════════════


def _register_sync_curve_metrics(app: "Dash") -> None:
    @app.callback(
        Output(SHELL_IDS["session_store"],                          "data",
               allow_duplicate=True),
        Input(CLUSTER_DEEP_DIVE_IDS["training_curves_chip_group"],  "value"),
        State(SHELL_IDS["session_store"],                           "data"),
        prevent_initial_call=True,
    )
    def _sync(
        selected:     Optional[List[str]],
        session_data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        session = Session.from_store(session_data)
        normalised = _normalise_str_list(selected)
        if session.evaluation.deep_dive_curve_metrics == normalised:
            raise PreventUpdate
        session.evaluation.deep_dive_curve_metrics = normalised
        return session.to_store()


# ═════════════════════════════════════════════════════════════════════
# 4. Elementary explorer multi-select / reset → session
# ═════════════════════════════════════════════════════════════════════


def _register_sync_elementary_selection(app: "Dash") -> None:
    @app.callback(
        Output(SHELL_IDS["session_store"],                            "data",
               allow_duplicate=True),
        Input(CLUSTER_DEEP_DIVE_IDS["elementary_explorer_grid"],      "selectedRows"),
        Input(CLUSTER_DEEP_DIVE_IDS["elementary_reset_btn"],          "n_clicks"),
        State(SHELL_IDS["session_store"],                             "data"),
        prevent_initial_call=True,
    )
    def _sync(
        selected_rows: Optional[List[Dict[str, Any]]],
        n_clicks:      Optional[int],
        session_data:  Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        trigger = ctx.triggered_id
        if trigger is None:
            raise PreventUpdate

        session = Session.from_store(session_data)
        ev = session.evaluation

        if trigger == CLUSTER_DEEP_DIVE_IDS["elementary_reset_btn"]:
            if not n_clicks:
                raise PreventUpdate
            if not ev.deep_dive_elementary_trade_ids:
                raise PreventUpdate
            ev.deep_dive_elementary_trade_ids = []
            return session.to_store()

        # selectedRows trigger: extract trade ids and dedupe.
        new_ids = _normalise_str_list(
            [
                row.get("trade_id") for row in (selected_rows or [])
                if isinstance(row, dict)
            ]
        )
        if sorted(ev.deep_dive_elementary_trade_ids) == sorted(new_ids):
            raise PreventUpdate
        # Preserve user-visible click order rather than re-sorting —
        # the multiline chart respects the order columns arrive in.
        ev.deep_dive_elementary_trade_ids = new_ids
        return session.to_store()


def _normalise_str_list(raw: Optional[Sequence[Any]]) -> List[str]:
    """Stable, deduped list of non-empty strings."""
    if not isinstance(raw, (list, tuple)):
        return []
    seen: set[str] = set()
    out: List[str] = []
    for v in raw:
        if not isinstance(v, str) or not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


# ═════════════════════════════════════════════════════════════════════
# 5. Render — Cluster Attributes + Graph Stats + nav button + store
# ═════════════════════════════════════════════════════════════════════


def _register_render_attributes(
    app: "Dash", backend: "RadeBackend",
) -> None:
    @app.callback(
        Output(CLUSTER_DEEP_DIVE_IDS["attributes_body"],      "children"),
        Output(CLUSTER_DEEP_DIVE_IDS["graph_stats_body"],     "children"),
        Output(CLUSTER_DEEP_DIVE_IDS["open_trade_graph_btn"], "disabled"),
        Output(CLUSTER_DEEP_DIVE_IDS["store_trade_types"],    "data"),
        Input(SHELL_IDS["url"],                               "pathname"),
        Input(SHELL_IDS["session_store"],                     "data"),
        prevent_initial_call=False,
    )
    def _render(
        pathname:     Optional[str],
        session_data: Optional[Dict[str, Any]],
    ) -> Tuple[List[Any], List[Any], bool, Dict[str, Any]]:
        if pathname != _DEEP_DIVE_PATH:
            raise PreventUpdate

        session = Session.from_store(session_data)
        cluster_id = session.evaluation.deep_dive_cluster_id or session.cluster_id

        if not cluster_id:
            return (
                _attribute_rows(None),
                _graph_stats_rows(None),
                True,
                _empty_trade_types_store(),
            )

        clusters_res = backend.clusters_df(cluster_id=cluster_id)
        attrs_row: Optional[Dict[str, Any]] = None
        if (
            clusters_res.ok
            and clusters_res.data is not None
            and not clusters_res.data.empty
        ):
            attrs_row = clusters_res.data.iloc[0].to_dict()

        graph_res = backend.trade_graph(cluster_id=cluster_id)
        stats: Optional[Dict[str, Any]] = None
        store_payload = _empty_trade_types_store()
        if graph_res.ok and graph_res.data is not None:
            tg = graph_res.data
            stats = tg.stats.model_dump() if tg.stats else None
            types_map: Dict[str, str] = {}
            target_ids: List[str] = []
            elementary_ids: List[str] = []
            for node in tg.nodes:
                tid = str(node.trade_id)
                ttype = str(node.trade_type)
                types_map[tid] = ttype
                if ttype == "target":
                    target_ids.append(tid)
                elif ttype == "elementary":
                    elementary_ids.append(tid)
            store_payload = {
                "types":          types_map,
                "target_ids":     target_ids,
                "elementary_ids": elementary_ids,
                "cluster_id":     cluster_id,
            }

        # n_trades / n_scenarios fall back to the trades_df shape if
        # they aren't carried on clusters_df (older parquets).  The
        # cheap fetch is cached so this doesn't add a round-trip on
        # the steady-state path.
        if attrs_row is None:
            attrs_row = {}
        if "n_trades" not in attrs_row or attrs_row.get("n_trades") is None:
            attrs_row["n_trades"] = (
                len(store_payload["types"])
                if store_payload["types"] else None
            )

        return (
            _attribute_rows(attrs_row),
            _graph_stats_rows(stats),
            False,
            store_payload,
        )


def _empty_trade_types_store() -> Dict[str, Any]:
    return {
        "types":          {},
        "target_ids":     [],
        "elementary_ids": [],
        "cluster_id":     None,
    }


def _attribute_rows(row: Optional[Dict[str, Any]]) -> List[Any]:
    """Render each (label, value) row — placeholder ``—`` on missing data."""
    children: List[Any] = []
    for label, candidates in _ATTRIBUTE_ROWS:
        value: Optional[Any] = None
        if row:
            for col in candidates:
                if col in row and row[col] is not None:
                    candidate = row[col]
                    if not (isinstance(candidate, float) and pd.isna(candidate)):
                        value = candidate
                        break
        if isinstance(value, float):
            display = _fmt_float(value, precision=2)
        elif isinstance(value, (int,)) and value is not None:
            display = _fmt_int(value)
        elif label in ("N Trades", "N Scenarios"):
            display = _fmt_int(value) if value is not None else _PLACEHOLDER
        elif value is None:
            display = _PLACEHOLDER
        else:
            display = str(value)
        children.append(_kv_row(label, display))
    return children


def _graph_stats_rows(stats: Optional[Dict[str, Any]]) -> List[Any]:
    """Render Graph Statistics body — Nodes / Edges / Density real,
    Avg Degree / Avg Path Length deferred (always ``—``).
    """
    children: List[Any] = []
    for label, key in _GRAPH_STATS_ROWS:
        if key is None or stats is None:
            children.append(_kv_row(label, _PLACEHOLDER))
            continue
        value = stats.get(key)
        if key == "density":
            display = _fmt_density(_nullable_float(value))
        else:
            display = _fmt_int(value)
        children.append(_kv_row(label, display))
    return children


def _kv_row(label: str, value: str) -> html.Div:
    return html.Div(
        className="flex items-center justify-between text-xs",
        children=[
            html.Span(label, className="text-slate-400"),
            html.Span(value, className="text-slate-100 font-medium"),
        ],
    )


# ═════════════════════════════════════════════════════════════════════
# 6. Render — Cluster Metrics KPI grid (4 KPIs + 4 sparklines)
# ═════════════════════════════════════════════════════════════════════


_KPI_CONFIG: Tuple[Tuple[str, str, str, str], ...] = (
    # (kpi_key, candidate column,         value_id key,             spark_id key)
    ("MAE",      "mae",            "kpi_mae_value",   "kpi_mae_spark"),
    ("RMSE",     "rmse",           "kpi_rmse_value",  "kpi_rmse_spark"),
    ("P95",      "p95_ae",         "kpi_p95_value",   "kpi_p95_spark"),
    ("P99",      "p99_ae",         "kpi_p99_value",   "kpi_p99_spark"),
)


def _register_render_kpis(
    app: "Dash", backend: "RadeBackend",
) -> None:
    @app.callback(
        Output(CLUSTER_DEEP_DIVE_IDS["kpi_mae_value"],   "children"),
        Output(CLUSTER_DEEP_DIVE_IDS["kpi_rmse_value"],  "children"),
        Output(CLUSTER_DEEP_DIVE_IDS["kpi_p95_value"],   "children"),
        Output(CLUSTER_DEEP_DIVE_IDS["kpi_p99_value"],   "children"),
        Output(CLUSTER_DEEP_DIVE_IDS["kpi_mae_spark"],   "figure"),
        Output(CLUSTER_DEEP_DIVE_IDS["kpi_rmse_spark"],  "figure"),
        Output(CLUSTER_DEEP_DIVE_IDS["kpi_p95_spark"],   "figure"),
        Output(CLUSTER_DEEP_DIVE_IDS["kpi_p99_spark"],   "figure"),
        Input(SHELL_IDS["url"],                          "pathname"),
        Input(SHELL_IDS["session_store"],                "data"),
        prevent_initial_call=False,
    )
    def _render(
        pathname:     Optional[str],
        session_data: Optional[Dict[str, Any]],
    ) -> Tuple[Any, ...]:
        if pathname != _DEEP_DIVE_PATH:
            raise PreventUpdate

        session = Session.from_store(session_data)
        cluster_id = session.evaluation.deep_dive_cluster_id or session.cluster_id
        if not cluster_id:
            blank_spark = _sparkline_payload([])
            return (
                _PLACEHOLDER, _PLACEHOLDER, _PLACEHOLDER, _PLACEHOLDER,
                blank_spark, blank_spark, blank_spark, blank_spark,
            )

        # KPI headline values come from per_member_metrics (cluster
        # aggregate); sparklines come from trades_df (per-trade
        # distribution across the cluster's trades).
        member_res = backend.per_member_metrics_df(
            split=session.split, cluster_id=cluster_id,
        )
        member_row: Dict[str, Any] = {}
        if (
            member_res.ok
            and member_res.data is not None
            and not member_res.data.empty
        ):
            member_row = member_res.data.iloc[0].to_dict()

        trades_res = backend.trades_df(session.split, cluster_id=cluster_id)
        trades_df: Optional[pd.DataFrame] = (
            trades_res.data
            if trades_res.ok and trades_res.data is not None
            else None
        )

        values: List[str] = []
        sparks: List[Dict[str, Any]] = []
        for _, col, _vid, _sid in _KPI_CONFIG:
            headline = _nullable_float(member_row.get(col))
            # If the per-member parquet doesn't carry the column,
            # fall back to the median of the trades_df column so
            # the user still sees a number rather than ``—``.
            if headline is None and trades_df is not None and col in trades_df.columns:
                series = trades_df[col].dropna()
                if not series.empty:
                    headline = float(series.median())
            values.append(_fmt_float(headline))

            spark_data: List[float] = []
            if trades_df is not None and col in trades_df.columns:
                series = trades_df[col].dropna().astype(float)
                if not series.empty:
                    spark_data = series.tolist()
            sparks.append(_sparkline_payload(spark_data))

        return (
            values[0], values[1], values[2], values[3],
            sparks[0], sparks[1], sparks[2], sparks[3],
        )


def _sparkline_payload(data: Sequence[float]) -> Dict[str, Any]:
    """Tiny line trace with no chrome — matches ``KpiCard``'s built-in
    ``_sparkline_figure`` helper so the visual baseline is consistent."""
    if not data:
        return {
            "data": [],
            "layout": {
                "xaxis":         {"visible": False},
                "yaxis":         {"visible": False},
                "margin":        {"l": 0, "r": 0, "t": 0, "b": 0},
                "paper_bgcolor": "rgba(0,0,0,0)",
                "plot_bgcolor":  "rgba(0,0,0,0)",
                "showlegend":    False,
                "height":        36,
            },
        }
    return {
        "data": [
            {
                "type": "scatter",
                "mode": "lines",
                "x":    list(range(len(data))),
                "y":    list(data),
                "line": {"color": "#94a3b8", "width": 1.4},
                "hoverinfo": "skip",
            }
        ],
        "layout": {
            "xaxis":         {"visible": False},
            "yaxis":         {"visible": False},
            "margin":        {"l": 0, "r": 0, "t": 0, "b": 0},
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor":  "rgba(0,0,0,0)",
            "showlegend":    False,
            "height":        36,
        },
    }


# ═════════════════════════════════════════════════════════════════════
# 7. Render — Cluster Portfolio + Residual-over-time
# ═════════════════════════════════════════════════════════════════════


def _register_render_timeseries(
    app: "Dash", backend: "RadeBackend",
) -> None:
    @app.callback(
        Output(CLUSTER_DEEP_DIVE_IDS["portfolio_chart"],   "figure"),
        Output(CLUSTER_DEEP_DIVE_IDS["residual_ts_chart"], "figure"),
        Input(SHELL_IDS["url"],                            "pathname"),
        Input(SHELL_IDS["session_store"],                  "data"),
        prevent_initial_call=False,
    )
    def _render(
        pathname:     Optional[str],
        session_data: Optional[Dict[str, Any]],
    ) -> Tuple[Any, Any]:
        if pathname != _DEEP_DIVE_PATH:
            raise PreventUpdate

        session = Session.from_store(session_data)
        cluster_id = session.evaluation.deep_dive_cluster_id or session.cluster_id
        if not cluster_id:
            return (
                empty_figure("Pick a cluster to see its portfolio."),
                empty_figure("Pick a cluster to see its residual-over-time."),
            )

        res = backend.cluster_timeseries_df(
            session.split, cluster_id=cluster_id,
        )
        ui_key = f"{session.split}::{cluster_id}"
        return (
            figure_with_fallback(
                res,
                on_ok=lambda df: portfolio_pnl(df, uirevision_key=ui_key),
                empty_msg="No timeseries data for this cluster.",
            ),
            figure_with_fallback(
                res,
                on_ok=lambda df: error_over_time(df, uirevision_key=ui_key),
                empty_msg="No residual data for this cluster.",
            ),
        )


# ═════════════════════════════════════════════════════════════════════
# 8. Render — Training curves (figure + chip group + empty msg + store)
# ═════════════════════════════════════════════════════════════════════


def _register_render_training_curves(
    app: "Dash", backend: "RadeBackend",
) -> None:
    @app.callback(
        Output(CLUSTER_DEEP_DIVE_IDS["training_curves_chart"],       "figure"),
        Output(CLUSTER_DEEP_DIVE_IDS["training_curves_chip_group"],  "children"),
        Output(CLUSTER_DEEP_DIVE_IDS["training_curves_chip_group"],  "value"),
        Output(CLUSTER_DEEP_DIVE_IDS["training_curves_chip_empty"],  "style"),
        Output(CLUSTER_DEEP_DIVE_IDS["store_curve_metrics"],         "data"),
        Input(SHELL_IDS["url"],                                      "pathname"),
        Input(SHELL_IDS["session_store"],                            "data"),
        prevent_initial_call=False,
    )
    def _render(
        pathname:     Optional[str],
        session_data: Optional[Dict[str, Any]],
    ) -> Tuple[Any, List[Any], List[str], Dict[str, str], List[str]]:
        if pathname != _DEEP_DIVE_PATH:
            raise PreventUpdate

        session = Session.from_store(session_data)
        cluster_id = session.evaluation.deep_dive_cluster_id or session.cluster_id

        empty_chip_hidden  = {"display": "none"}
        empty_chip_visible = {"display": "inline"}

        if not cluster_id:
            return (
                empty_figure("Pick a cluster to see its training curves."),
                [],
                [],
                empty_chip_visible,
                [],
            )

        res = backend.training_curves_df(cluster_id=cluster_id)
        if not res.ok or res.data is None or res.data.empty:
            err = res.error if not res.ok else "no data"
            logger.info(
                "training_curves fetch failed / empty for cluster %s: %s",
                cluster_id, err,
            )
            return (
                empty_figure("No training curves staged for this cluster."),
                [],
                [],
                empty_chip_visible,
                [],
            )

        df = res.data
        available = list(df.attrs.get("metrics") or [])
        persisted = list(session.evaluation.deep_dive_curve_metrics)
        validated = [m for m in persisted if m in available]

        chip_children = [
            dmc.Chip(m, value=m, size="xs", variant="outline")
            for m in available
        ]
        fig = training_curves_chart(
            df,
            selected_metrics=validated,
            available_metrics=available,
            uirevision_key=cluster_id,
        )
        empty_style = empty_chip_hidden if available else empty_chip_visible
        return fig, chip_children, validated, empty_style, available


# ═════════════════════════════════════════════════════════════════════
# 9. Render — Trade-Level Metrics + Elementary Explorer grids
# ═════════════════════════════════════════════════════════════════════


_TRADES_GRID_COLUMN_HEADERS: Dict[str, str] = {
    "trade_id":       "Trade",
    "trade_type":     "Type",
    "mae":            "MAE",
    "mse":            "MSE",
    "rmse":           "RMSE",
    "max_ae":         "Max |err|",
    "p95_ae":         "P95 |err|",
    "p99_ae":         "P99 |err|",
    "mean_residual":  "Mean resid.",
    "std_residual":   "Std resid.",
    "n_scenarios":    "Scenarios",
}
_TRADES_GRID_PRIORITY: Tuple[str, ...] = (
    "trade_id", "trade_type",
    "mae", "rmse", "p95_ae", "p99_ae",
    "mean_residual", "std_residual", "n_scenarios",
)


def _register_render_grids(
    app: "Dash", backend: "RadeBackend",
) -> None:
    @app.callback(
        Output(CLUSTER_DEEP_DIVE_IDS["trades_grid"],              "rowData"),
        Output(CLUSTER_DEEP_DIVE_IDS["trades_grid"],              "columnDefs"),
        Output(CLUSTER_DEEP_DIVE_IDS["elementary_explorer_grid"], "rowData"),
        Output(CLUSTER_DEEP_DIVE_IDS["elementary_explorer_grid"], "columnDefs"),
        Output(CLUSTER_DEEP_DIVE_IDS["elementary_explorer_grid"], "selectedRows"),
        Input(SHELL_IDS["url"],                                   "pathname"),
        Input(SHELL_IDS["session_store"],                         "data"),
        Input(CLUSTER_DEEP_DIVE_IDS["store_trade_types"],         "data"),
        prevent_initial_call=False,
    )
    def _render(
        pathname:        Optional[str],
        session_data:    Optional[Dict[str, Any]],
        types_payload:   Optional[Dict[str, Any]],
    ) -> Tuple[
        List[Dict[str, Any]], List[Dict[str, Any]],
        List[Dict[str, Any]], List[Dict[str, Any]],
        List[Dict[str, Any]],
    ]:
        if pathname != _DEEP_DIVE_PATH:
            raise PreventUpdate

        session = Session.from_store(session_data)
        cluster_id = session.evaluation.deep_dive_cluster_id or session.cluster_id
        elem_columns = _elementary_grid_column_defs()
        trade_columns = _trades_grid_column_defs_initial()

        if not cluster_id:
            return [], trade_columns, [], elem_columns, []

        res = backend.trades_df(session.split, cluster_id=cluster_id)
        if not res.ok or res.data is None or res.data.empty:
            return [], trade_columns, [], elem_columns, []

        df = res.data.copy()
        types_map: Dict[str, str] = {}
        elementary_set: set = set()
        if isinstance(types_payload, dict):
            types_map = dict(types_payload.get("types") or {})
            elementary_set = set(types_payload.get("elementary_ids") or [])

        df["trade_type"] = (
            df["trade_id"].map(types_map).fillna("unknown")
            if "trade_id" in df.columns
            else "unknown"
        )

        trades_columns = _trades_grid_column_defs(df)
        trades_rows = df[
            [c["field"] for c in trades_columns if c["field"] in df.columns]
        ].to_dict(orient="records")

        # Elementary explorer — filter to elementary trades.  Use the
        # graph-derived set when present; degrade to ``trade_type``
        # column when the graph payload is missing.
        if elementary_set:
            elem_df = df[df["trade_id"].isin(elementary_set)]
        else:
            elem_df = df[df["trade_type"] == "elementary"]

        elem_rows = (
            elem_df[
                [c["field"] for c in elem_columns if c["field"] in elem_df.columns]
            ].to_dict(orient="records")
            if not elem_df.empty
            else []
        )

        # Re-assert selectedRows from session so deep-link / browser-
        # back paints the user's prior selection.  Match by trade_id;
        # any session id that's no longer in the cluster is silently
        # dropped (the capture callback will reconcile session on the
        # next user gesture).
        selected_ids = set(
            session.evaluation.deep_dive_elementary_trade_ids or []
        )
        selected_rows = [r for r in elem_rows if r.get("trade_id") in selected_ids]

        return trades_rows, trades_columns, elem_rows, elem_columns, selected_rows


def _trades_grid_column_defs_initial() -> List[Dict[str, Any]]:
    """Stable column defs for the empty-state trades grid."""
    return [
        {"field": "trade_id",      "headerName": "Trade",        "flex": 2, "minWidth": 160},
        {"field": "trade_type",    "headerName": "Type",         "flex": 1, "minWidth": 100},
        {"field": "mae",           "headerName": "MAE",          "flex": 1, "type": "numericColumn"},
        {"field": "rmse",          "headerName": "RMSE",         "flex": 1, "type": "numericColumn"},
        {"field": "p95_ae",        "headerName": "P95 |err|",    "flex": 1, "type": "numericColumn"},
        {"field": "p99_ae",        "headerName": "P99 |err|",    "flex": 1, "type": "numericColumn"},
        {"field": "mean_residual", "headerName": "Mean resid.",  "flex": 1, "type": "numericColumn"},
        {"field": "std_residual",  "headerName": "Std resid.",   "flex": 1, "type": "numericColumn"},
        {"field": "n_scenarios",   "headerName": "Scenarios",    "flex": 1, "type": "numericColumn"},
    ]


def _trades_grid_column_defs(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Dynamic columnDefs from the available ``trades_df`` columns."""
    numeric_cols = {
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col].dtype)
    }
    ordered_cols: List[str] = []
    seen: set = set()
    for col in _TRADES_GRID_PRIORITY:
        if col in df.columns and col not in seen:
            ordered_cols.append(col)
            seen.add(col)
    for col in df.columns:
        if col in seen or col in ("cluster_id", "split"):
            continue
        ordered_cols.append(col)
        seen.add(col)

    defs: List[Dict[str, Any]] = []
    for col in ordered_cols:
        col_def: Dict[str, Any] = {
            "field":      str(col),
            "headerName": _TRADES_GRID_COLUMN_HEADERS.get(col, str(col)),
        }
        if col == "trade_id":
            col_def.update({"flex": 2, "minWidth": 160})
        elif col == "trade_type":
            col_def.update({"flex": 1, "minWidth": 100})
        else:
            col_def["flex"] = 1
            if col in numeric_cols:
                col_def["type"] = "numericColumn"
                col_def["valueFormatter"] = {
                    "function": "d3.format('.4~g')(params.value)",
                }
        defs.append(col_def)
    return defs


def _elementary_grid_column_defs() -> List[Dict[str, Any]]:
    """Static column defs for the Elementary PnL Explorer grid.

    Mirrors ``layouts.evaluation.cluster_deep_dive
    ._elementary_explorer_column_defs`` so the rowSelection /
    checkbox setup the layout configures stays the source of truth on
    initial paint, and the callback re-asserts the same shape after
    data lands so AgGrid doesn't drop the checkboxSelection flag
    when it sees a non-empty rowData for the first time.
    """
    return [
        {
            "field":                  "trade_id",
            "headerName":             "Trade",
            "flex":                   2,
            "minWidth":               160,
            "checkboxSelection":      True,
            "headerCheckboxSelection": False,
        },
        {"field": "mae",           "headerName": "MAE",         "flex": 1, "type": "numericColumn"},
        {"field": "rmse",          "headerName": "RMSE",        "flex": 1, "type": "numericColumn"},
        {"field": "mean_residual", "headerName": "Mean resid.", "flex": 1, "type": "numericColumn"},
        {"field": "n_scenarios",   "headerName": "Scenarios",   "flex": 1, "type": "numericColumn"},
    ]


# ═════════════════════════════════════════════════════════════════════
# 10. Render — Per-trade detail (Row 3, collapse-on-no-selection)
# ═════════════════════════════════════════════════════════════════════


def _register_render_per_trade_detail(
    app: "Dash", backend: "RadeBackend",
) -> None:
    @app.callback(
        Output(CLUSTER_DEEP_DIVE_IDS["row3_wrapper"],            "style"),
        Output(CLUSTER_DEEP_DIVE_IDS["selected_trade_label"],    "children"),
        Output(CLUSTER_DEEP_DIVE_IDS["per_trade_residual_hist"], "figure"),
        Output(CLUSTER_DEEP_DIVE_IDS["per_trade_bias_scatter"],  "figure"),
        Input(SHELL_IDS["url"],                                  "pathname"),
        Input(SHELL_IDS["session_store"],                        "data"),
        Input(CLUSTER_DEEP_DIVE_IDS["store_trade_types"],        "data"),
        prevent_initial_call=False,
    )
    def _render(
        pathname:      Optional[str],
        session_data:  Optional[Dict[str, Any]],
        types_payload: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], str, Any, Any]:
        if pathname != _DEEP_DIVE_PATH:
            raise PreventUpdate

        session = Session.from_store(session_data)
        cluster_id = session.evaluation.deep_dive_cluster_id or session.cluster_id
        selected_trade_id = session.evaluation.deep_dive_selected_trade_id

        hidden_style  = {"display": "none"}
        visible_style = {"display": "grid"}

        if not selected_trade_id or not cluster_id:
            return hidden_style, "Trade: —", no_update, no_update

        types_map: Dict[str, str] = {}
        target_ids: List[str] = []
        if isinstance(types_payload, dict):
            types_map = dict(types_payload.get("types") or {})
            target_ids = list(types_payload.get("target_ids") or [])

        trade_type = types_map.get(selected_trade_id, "unknown")
        chip_label = f"Trade: {selected_trade_id}  ·  Type: {trade_type}"

        # Residuals only make sense for *target* trades — the model
        # doesn't predict elementary trades (those are inputs).  Show
        # the row, but render an explanatory empty figure so the user
        # learns *why* the chart is blank rather than seeing a stale
        # plot or the row silently snapping shut.
        if trade_type != "target":
            placeholder = empty_figure(
                "Residual diagnostics are only available for target trades."
            )
            return visible_style, chip_label, placeholder, placeholder

        if not target_ids:
            placeholder = empty_figure(
                "Trade-graph payload missing target ordering — "
                "re-run the eval pipeline to stage trade_universe.json."
            )
            return visible_style, chip_label, placeholder, placeholder

        try:
            target_idx = target_ids.index(selected_trade_id)
        except ValueError:
            placeholder = empty_figure(
                f"Trade {selected_trade_id!r} not found in this cluster's "
                "target ordering."
            )
            return visible_style, chip_label, placeholder, placeholder

        pred_res = backend.predictions(cluster_id=cluster_id, split=session.split)
        if not pred_res.ok or pred_res.data is None:
            err = pred_res.error or "predictions fetch failed"
            placeholder = empty_figure(
                f"Could not load per-scenario predictions ({err})."
            )
            return visible_style, chip_label, placeholder, placeholder

        npz: Dict[str, np.ndarray] = pred_res.data
        predictions_arr = np.asarray(npz.get("predictions"))
        targets_arr     = np.asarray(npz.get("targets"))
        if (
            predictions_arr.size == 0
            or targets_arr.size == 0
            or predictions_arr.ndim != 2
            or targets_arr.shape != predictions_arr.shape
        ):
            placeholder = empty_figure(
                "Predictions NPZ is empty or malformed for this cluster."
            )
            return visible_style, chip_label, placeholder, placeholder

        if target_idx >= predictions_arr.shape[1]:
            placeholder = empty_figure(
                f"Target index {target_idx} out of range for predictions "
                f"shape {predictions_arr.shape}."
            )
            return visible_style, chip_label, placeholder, placeholder

        pred_slice   = predictions_arr[:, target_idx]
        target_slice = targets_arr[:, target_idx]
        ui_key = f"{cluster_id}::{session.split}::{selected_trade_id}"

        histogram = per_trade_residual_histogram(
            pred_slice, target_slice,
            trade_id=selected_trade_id,
            uirevision_key=ui_key,
        )
        scatter = per_trade_bias_scatter(
            pred_slice, target_slice,
            trade_id=selected_trade_id,
            uirevision_key=ui_key,
        )
        return visible_style, chip_label, histogram, scatter


# ═════════════════════════════════════════════════════════════════════
# 11. Render — Elementary PnL multi-line (Row 4 right)
# ═════════════════════════════════════════════════════════════════════


def _register_render_elementary_pnl(
    app: "Dash", backend: "RadeBackend",
) -> None:
    @app.callback(
        Output(CLUSTER_DEEP_DIVE_IDS["elementary_pnl_empty"],      "style"),
        Output(CLUSTER_DEEP_DIVE_IDS["elementary_pnl_chart_card"], "style"),
        Output(CLUSTER_DEEP_DIVE_IDS["elementary_pnl_chart"],      "figure"),
        Input(SHELL_IDS["url"],                                    "pathname"),
        Input(SHELL_IDS["session_store"],                          "data"),
        prevent_initial_call=False,
    )
    def _render(
        pathname:     Optional[str],
        session_data: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Any]:
        if pathname != _DEEP_DIVE_PATH:
            raise PreventUpdate

        session = Session.from_store(session_data)
        cluster_id = session.evaluation.deep_dive_cluster_id or session.cluster_id
        selected_ids = list(session.evaluation.deep_dive_elementary_trade_ids or [])

        empty_style_visible = {
            "display": "flex",
            "flex":    "1 1 auto",
            "min-height": "160px",
        }
        empty_style_hidden  = {"display": "none"}

        if not cluster_id or not selected_ids:
            return (
                empty_style_visible,
                {"display": "none"},
                empty_figure(
                    "Pick one or more elementary trades to plot their PnL."
                ),
            )

        res = backend.elementary_pnl_df(
            cluster_id=cluster_id, trade_ids=selected_ids,
        )
        if not res.ok:
            err = res.error or "elementary-pnl fetch failed"
            return (
                empty_style_visible,
                {"display": "none"},
                empty_figure(f"Could not load elementary PnL ({err})."),
            )
        df = res.data
        if df is None or df.empty:
            return (
                empty_style_visible,
                {"display": "none"},
                empty_figure(
                    "Selected elementary trades have no PnL data on this split."
                ),
            )

        ui_key = f"{cluster_id}::{','.join(sorted(selected_ids))}"
        fig = elementary_pnl_multiline(df, uirevision_key=ui_key)
        return (
            empty_style_hidden,
            {"display": "block"},
            fig,
        )


# ═════════════════════════════════════════════════════════════════════
# 12. Cross-page navigation — to / from Trade-Graph sub-tab
# ═════════════════════════════════════════════════════════════════════


def _register_navigate_to_trade_graph(app: "Dash") -> None:
    @app.callback(
        Output(SHELL_IDS["url"],                             "pathname",
               allow_duplicate=True),
        Output(SHELL_IDS["url"],                             "search",
               allow_duplicate=True),
        Output(SHELL_IDS["session_store"],                   "data",
               allow_duplicate=True),
        Input(CLUSTER_DEEP_DIVE_IDS["open_trade_graph_btn"], "n_clicks"),
        State(SHELL_IDS["session_store"],                    "data"),
        prevent_initial_call=True,
    )
    def _navigate(
        n_clicks:     Optional[int],
        session_data: Optional[Dict[str, Any]],
    ) -> Tuple[str, str, Dict[str, Any]]:
        if not n_clicks:
            raise PreventUpdate

        session = Session.from_store(session_data)
        cid = session.evaluation.deep_dive_cluster_id or session.cluster_id
        if not cid:
            raise PreventUpdate

        session.evaluation.trade_graph_cluster_id = cid
        session.evaluation.trade_graph_selected_trade_id = None

        return _TRADE_GRAPH_PATH, "", session.to_store()


def _register_navigate_from_trade_graph(app: "Dash") -> None:
    @app.callback(
        Output(SHELL_IDS["url"],                          "pathname",
               allow_duplicate=True),
        Output(SHELL_IDS["url"],                          "search",
               allow_duplicate=True),
        Output(SHELL_IDS["session_store"],                "data",
               allow_duplicate=True),
        Input(TRADE_GRAPH_IDS["selected_deep_dive_btn"],  "n_clicks"),
        State(SHELL_IDS["session_store"],                 "data"),
        prevent_initial_call=True,
    )
    def _navigate(
        n_clicks:     Optional[int],
        session_data: Optional[Dict[str, Any]],
    ) -> Tuple[str, str, Dict[str, Any]]:
        if not n_clicks:
            raise PreventUpdate

        session = Session.from_store(session_data)
        cid = (
            session.evaluation.trade_graph_cluster_id
            or session.cluster_id
        )
        if not cid:
            raise PreventUpdate

        session.evaluation.deep_dive_cluster_id = cid
        session.evaluation.deep_dive_selected_trade_id = (
            session.evaluation.trade_graph_selected_trade_id
        )

        return _DEEP_DIVE_PATH, f"?cid={cid}", session.to_store()


__all__ = ["register"]
```

### Verification (after pasting all three files)

Restart the dashboard, then check:

| Sanity probe | Expected |
|---|---|
| Browser console | No `Duplicate callback outputs` errors. |
| `/evaluation/cluster` page | All cards render: Cluster Attributes, KPI grid (4 sparklines), Graph Statistics, Cluster Portfolio chart, Trade-Level Metrics grid, Convergence strip (residual + curves stacked), Elementary PnL Explorer + empty-state placeholder. |
| Click a row in the Trade-Level Metrics grid | Row 3 (per-trade detail) expands.  For target trades you see histogram + bias scatter populated from the predictions NPZ.  For elementary trades you see an explanatory empty figure. |
| Tick rows in the Elementary PnL Explorer | Row 4 right swaps the empty placeholder for the multi-line PnL chart. |
| Click "Reset selection" | Multi-line chart collapses back to the empty-state placeholder. |
| Switch cluster in the picker | Selected trade + elementary multi-select + curve overlay metrics all reset (intentional — those ids belong to the prior cluster). |
| Browser-back on a `?cid=` deep-link | Cluster + selected trade + elementary checkboxes all repaint from session. |

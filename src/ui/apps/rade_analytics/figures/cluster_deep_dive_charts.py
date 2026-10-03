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

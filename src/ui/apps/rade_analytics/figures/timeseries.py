"""Portfolio-level time-series figures.

Two public builders:

* :func:`portfolio_pnl` — predicted vs actual PnL line chart with a
  semi-transparent fill under the prediction (the overview page uses
  the same visual so the two pages compose cleanly).
* :func:`error_over_time` — rolling absolute-error line with a ±1σ
  band, so users can spot *when* the model breaks, not just how often.

Both operate on the ``portfolio_df`` DataFrame shape the backend
already exposes — columns: ``scenario_idx``, ``scenario_label``,
``predictions``, ``targets``, ``error``, ``abs_error``, ``squared_error``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ._theme import (
    color_for_index,
    empty_figure,
    is_all_nan,
    pnl_axis_title,
    pnl_hover_format,
    rade_layout,
    rgba,
    sort_chronologically,
)


def portfolio_pnl(
    df: pd.DataFrame,
    *,
    uirevision_key: Optional[str] = None,
    space:          str = "scaled",
) -> go.Figure:
    """Predicted-vs-actual line chart.

    Parameters
    ----------
    df
        ``portfolio_df``-shaped frame.  Must have ``predictions`` and
        ``targets`` columns; the x-axis uses ``scenario_label`` when
        present, otherwise ``scenario_idx`` (or a plain range).
    uirevision_key
        Optional Plotly ``uirevision`` value.  When the same key is
        passed across re-renders, the user's zoom / pan / legend
        toggles are preserved; when the key changes (e.g. user flips
        split test→val) the UI state resets to the new data domain.
        Page Contract §6 mandates uirevision on every time-series
        figure so filter changes don't snap the user's zoom back to
        full extent.  Callers typically pass ``f"{split}:{space}"``.
    space
        PnL space the ``df`` columns are in (Phase 3.4).  Drives the
        y-axis title (``"PnL (scaled)"`` vs ``"PnL (notional)"``) and
        the hover numeric format.  Default ``"scaled"`` preserves the
        legacy behaviour for any caller that doesn't pass it yet.
    """
    if (
        df is None or df.empty
        or "predictions" not in df.columns
        or "targets" not in df.columns
    ):
        return empty_figure("No portfolio data for the active filter set.")

    # Phase 3.4 — when the user toggled to Original on a run without
    # scaler coverage the API surfaces all-null measures; render a
    # dedicated empty-state rather than a flat-zero chart.
    if (
        space == "original"
        and is_all_nan(df["predictions"])
        and is_all_nan(df["targets"])
    ):
        return empty_figure(
            "Original-space PnL is unavailable for this run "
            "(no scaler coverage). Toggle to Scaled to view this chart."
        )

    # Sort by real chronology, not training order (train split is shuffled).
    df_sorted = sort_chronologically(df)
    x_vals = (
        df_sorted["scenario_label"]
        if "scenario_label" in df_sorted.columns
        else list(range(len(df_sorted)))
    )

    pred_color = color_for_index(0)
    hover_fmt  = pnl_hover_format(space)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=df_sorted["predictions"],
            mode="lines",
            name="Predicted PnL",
            line={"color": pred_color, "width": 2.5},
            fill="tozeroy",
            fillcolor=rgba(pred_color, 0.18),
            hovertemplate=f"%{{y:{hover_fmt.lstrip(':')}}}<extra>Predicted</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=df_sorted["targets"],
            mode="lines",
            name="Actual PnL",
            line={"color": "#cbd5e1", "width": 1.5, "dash": "dash"},
            hovertemplate=f"%{{y:{hover_fmt.lstrip(':')}}}<extra>Actual</extra>",
        )
    )
    fig.update_layout(
        **rade_layout(
            show_legend=True,
            hovermode="x unified",
            xaxis={"showticklabels": True},
            yaxis={"title": {"text": pnl_axis_title(space), "font": {"color": "#94a3b8"}}},
        ),
    )
    if uirevision_key is not None:
        fig.update_layout(uirevision=uirevision_key)
    return fig


def error_over_time(
    df:             pd.DataFrame,
    *,
    window:         Optional[int] = None,
    band_std:       float = 1.0,
    uirevision_key: Optional[str] = None,
    space:          str = "scaled",
) -> go.Figure:
    """Rolling absolute error time series with a ±``band_std``·σ band.

    Parameters
    ----------
    df
        ``portfolio_df``-shaped frame.  Must carry either ``abs_error``
        or both ``predictions`` + ``targets`` (in which case abs-error
        is derived on the fly).
    window
        Rolling window size in scenarios.  When ``None`` we pick
        ``max(3, len(df) // 10)`` — a gentle smoother that scales with
        dataset size without drowning short series.
    band_std
        Width of the shaded band in rolling σ units.  ``1.0`` renders
        the ±1σ zone; set ``2.0`` for a wider envelope.
    uirevision_key
        Optional Plotly ``uirevision`` value.  See :func:`portfolio_pnl`
        for the contract; callers on the Cluster Deep-Dive page
        typically pass ``f"{split}::{cluster_id}:{space}"`` so the
        user's zoom survives re-renders triggered by an unrelated
        session change but resets when the data domain genuinely
        shifts.
    space
        PnL space the ``df`` columns are in (Phase 3.4).  Drives the
        y-axis title (``"Absolute error (scaled)"`` vs
        ``"Absolute error (notional)"``) and the hover format on the
        rolling-mean / spike traces.  Default ``"scaled"`` preserves
        the legacy behaviour for any caller that doesn't pass it yet.
    """
    if df is None or df.empty:
        return empty_figure("No error-over-time data for the active filter set.")

    if "abs_error" in df.columns:
        abs_err = df["abs_error"].astype(float)
    elif {"predictions", "targets"}.issubset(df.columns):
        abs_err = (df["predictions"] - df["targets"]).abs().astype(float)
    else:
        return empty_figure(
            "Error-over-time requires predictions + targets columns."
        )

    df_sorted = sort_chronologically(df.assign(_abs_error=abs_err))
    x_vals = (
        df_sorted["scenario_label"]
        if "scenario_label" in df_sorted.columns
        else list(range(len(df_sorted)))
    )

    n = len(df_sorted)
    win = int(window) if window else max(3, n // 10)
    win = min(max(win, 3), n)

    rolling_mean = df_sorted["_abs_error"].rolling(win, min_periods=1).mean()
    rolling_std  = df_sorted["_abs_error"].rolling(win, min_periods=1).std().fillna(0.0)
    upper = rolling_mean + band_std * rolling_std
    lower = (rolling_mean - band_std * rolling_std).clip(lower=0)

    err_color = color_for_index(3)   # rose — signals "error"

    fig = go.Figure()
    # Upper / lower band as a filled envelope (upper first, then lower
    # with fill='tonexty' to close the shape).
    fig.add_trace(
        go.Scatter(
            x=x_vals, y=upper,
            mode="lines",
            line={"color": "rgba(244, 63, 94, 0)", "width": 0},
            hoverinfo="skip",
            showlegend=False,
            name="upper",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_vals, y=lower,
            mode="lines",
            line={"color": "rgba(244, 63, 94, 0)", "width": 0},
            fill="tonexty",
            fillcolor="rgba(244, 63, 94, 0.18)",
            hoverinfo="skip",
            showlegend=False,
            name="lower",
        )
    )

    hover_fmt = pnl_hover_format(space)
    # Raw abs-error as a thin line — kept to show scenario-level spikes.
    fig.add_trace(
        go.Scatter(
            x=x_vals, y=df_sorted["_abs_error"],
            mode="lines",
            line={"color": "rgba(244, 63, 94, 0.35)", "width": 1},
            name="abs error",
            hovertemplate=f"%{{y:{hover_fmt.lstrip(':')}}}<extra>abs error</extra>",
        )
    )
    # Rolling mean — the headline signal.
    fig.add_trace(
        go.Scatter(
            x=x_vals, y=rolling_mean,
            mode="lines",
            line={"color": err_color, "width": 2.5},
            name=f"rolling mean (w={win})",
            hovertemplate=f"%{{y:{hover_fmt.lstrip(':')}}}<extra>rolling mean</extra>",
        )
    )

    fig.update_layout(
        **rade_layout(
            show_legend=True,
            hovermode="x unified",
            xaxis={"showticklabels": True},
            yaxis={"title": {
                "text": pnl_axis_title(space, prefix="Absolute error"),
                "font": {"color": "#94a3b8"},
            }},
        ),
    )
    if uirevision_key is not None:
        fig.update_layout(uirevision=uirevision_key)
    return fig


def _safe_std(values: np.ndarray) -> float:
    """Numpy std that returns 0.0 on empty / constant input."""
    if values.size == 0:
        return 0.0
    return float(np.std(values))


__all__ = ["error_over_time", "portfolio_pnl"]

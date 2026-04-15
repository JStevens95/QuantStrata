"""
Time-series / scenario-indexed line-chart figure builders.

Scenarios are treated as ordered indices (0, 1, 2, ...) since the
underlying data is scenario-based, not calendar-dated.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import plotly.graph_objects as go

from src.ui.apps.ensemble_analytics.theme.colors import ACCENT_BLUE, ACCENT_GREEN, CHART_COLORS


def pnl_timeseries(
    predictions: np.ndarray,
    targets: np.ndarray,
    title: str = "Portfolio PnL — Predictions vs Targets",
) -> go.Figure:
    """
    Overlay predicted and target PnL as line charts over scenario index.

    Parameters
    ----------
    predictions : np.ndarray
        1-D scenario-ordered predicted PnL.
    targets : np.ndarray
        1-D scenario-ordered target PnL.
    title : str
        Figure title.

    Returns
    -------
    go.Figure
    """
    x = np.arange(len(predictions))

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=x, y=targets, mode="lines",
        name="Target", line=dict(color=ACCENT_GREEN, width=1.5),
        hovertemplate="Scenario %{x}<br>Target: %{y:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scattergl(
        x=x, y=predictions, mode="lines",
        name="Prediction", line=dict(color=ACCENT_BLUE, width=1.5),
        hovertemplate="Scenario %{x}<br>Prediction: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Scenario",
        yaxis_title="PnL",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def overlaid_group_timeseries(
    group_data: Dict[str, np.ndarray],
    title: str = "PnL by Group",
    y_label: str = "PnL",
) -> go.Figure:
    """
    Overlay multiple groups as separate lines (e.g. desks or ccys).

    Parameters
    ----------
    group_data : dict
        ``{group_label: 1-D np.ndarray}``.
    title : str
        Figure title.
    y_label : str
        Y-axis label.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()
    for i, (label, values) in enumerate(group_data.items()):
        fig.add_trace(go.Scattergl(
            x=np.arange(len(values)),
            y=values,
            mode="lines",
            name=label,
            line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=1.5),
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Scenario",
        yaxis_title=y_label,
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig

"""
Scatter plot figure builders.

All functions return ``go.Figure`` with the global template applied.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go

from src.ui.apps.ensemble_analytics.theme.colors import ACCENT_BLUE, ACCENT_RED, TEXT_SECONDARY


def pred_vs_target_scatter(
    predictions: np.ndarray,
    targets: np.ndarray,
    title: str = "Predictions vs Targets",
    max_points: int = 5_000,
) -> go.Figure:
    """
    Scatter plot of predicted vs actual values with a 45-degree reference line.

    Parameters
    ----------
    predictions : np.ndarray
        1-D array of predicted values.
    targets : np.ndarray
        1-D array of actual values.
    title : str
        Figure title.
    max_points : int
        Downsample to this many points if the array is larger.

    Returns
    -------
    go.Figure
    """
    if len(predictions) > max_points:
        idx = np.random.default_rng(42).choice(len(predictions), max_points, replace=False)
        predictions, targets = predictions[idx], targets[idx]

    vmin = min(predictions.min(), targets.min())
    vmax = max(predictions.max(), targets.max())

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=targets,
        y=predictions,
        mode="markers",
        marker=dict(size=3, color=ACCENT_BLUE, opacity=0.5),
        name="Predicted",
    ))
    fig.add_trace(go.Scattergl(
        x=[vmin, vmax],
        y=[vmin, vmax],
        mode="lines",
        line=dict(color=TEXT_SECONDARY, dash="dash", width=1),
        name="Perfect",
        showlegend=False,
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Target",
        yaxis_title="Prediction",
        height=450,
    )
    return fig


def residual_scatter(
    predictions: np.ndarray,
    targets: np.ndarray,
    title: str = "Residuals",
) -> go.Figure:
    """
    Scatter plot of residuals (prediction - target) vs target.

    Parameters
    ----------
    predictions : np.ndarray
        1-D predicted values.
    targets : np.ndarray
        1-D actual values.
    title : str
        Figure title.

    Returns
    -------
    go.Figure
    """
    residuals = predictions - targets

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=targets,
        y=residuals,
        mode="markers",
        marker=dict(
            size=3,
            color=np.where(residuals >= 0, ACCENT_BLUE, ACCENT_RED),
            opacity=0.5,
        ),
        name="Residual",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color=TEXT_SECONDARY, line_width=1)
    fig.update_layout(
        title=title,
        xaxis_title="Target",
        yaxis_title="Residual (Pred − Target)",
        height=400,
    )
    return fig

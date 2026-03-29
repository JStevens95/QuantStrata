"""
Distribution plot figure builders — histograms, violins, QQ plots.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import plotly.graph_objects as go

from src.ui.apps.ensemble_analytics.theme.colors import (
    ACCENT_BLUE,
    ACCENT_RED,
    ACCENT_GREEN,
    CHART_COLORS,
    TEXT_SECONDARY,
)


def residual_histogram(
    predictions: np.ndarray,
    targets: np.ndarray,
    title: str = "Residual Distribution",
    nbins: int = 80,
) -> go.Figure:
    """
    Histogram of residuals (prediction - target) with annotation box
    showing mean, std, skew, kurtosis, and % within ±1σ/±2σ.

    Parameters
    ----------
    predictions : np.ndarray
        1-D predicted values.
    targets : np.ndarray
        1-D target values.
    title : str
        Figure title.
    nbins : int
        Number of histogram bins.

    Returns
    -------
    go.Figure
    """
    from scipy import stats as _stats

    residuals = predictions - targets
    mu = float(residuals.mean())
    sigma = float(residuals.std())
    skew = float(_stats.skew(residuals))
    kurt = float(_stats.kurtosis(residuals))
    pct_1s = float(np.mean(np.abs(residuals - mu) <= sigma) * 100)
    pct_2s = float(np.mean(np.abs(residuals - mu) <= 2 * sigma) * 100)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=residuals, nbinsx=nbins,
        marker_color=ACCENT_BLUE, opacity=0.8,
        name="Residuals",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color=TEXT_SECONDARY, line_width=1)

    annotation_text = (
        f"μ={mu:.4f}  σ={sigma:.4f}<br>"
        f"skew={skew:.2f}  kurt={kurt:.2f}<br>"
        f"±1σ: {pct_1s:.1f}%  ±2σ: {pct_2s:.1f}%"
    )
    fig.add_annotation(
        text=annotation_text, xref="paper", yref="paper",
        x=0.98, y=0.95, showarrow=False,
        font=dict(size=11, color=TEXT_SECONDARY),
        align="right", bgcolor="rgba(22,27,34,0.8)", bordercolor=ACCENT_BLUE,
    )
    fig.update_layout(
        title=title,
        xaxis_title="Residual (Pred − Target)",
        yaxis_title="Count",
        height=350,
    )
    return fig


def violin_overlay(
    group_data: Dict[str, np.ndarray],
    title: str = "Distribution Comparison",
) -> go.Figure:
    """
    Overlay violin plots for multiple groups.

    Parameters
    ----------
    group_data : dict
        ``{group_label: 1-D values}``.
    title : str
        Figure title.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()
    for i, (label, values) in enumerate(group_data.items()):
        fig.add_trace(go.Violin(
            y=values, name=label,
            line_color=CHART_COLORS[i % len(CHART_COLORS)],
            meanline_visible=True,
            box_visible=True,
        ))
    fig.update_layout(title=title, height=400, showlegend=False)
    return fig


def qq_plot(
    sample: np.ndarray,
    title: str = "Q-Q Plot (Normal)",
) -> go.Figure:
    """
    Quantile-quantile plot against a standard normal distribution.

    Parameters
    ----------
    sample : np.ndarray
        1-D sample values.
    title : str
        Figure title.

    Returns
    -------
    go.Figure
    """
    from scipy import stats

    sorted_sample = np.sort(sample)
    n = len(sorted_sample)
    theoretical = stats.norm.ppf(np.linspace(0.001, 0.999, n))

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=theoretical, y=sorted_sample,
        mode="markers",
        marker=dict(size=3, color=ACCENT_BLUE, opacity=0.6),
        name="Sample",
    ))
    qmin, qmax = theoretical.min(), theoretical.max()
    fig.add_trace(go.Scattergl(
        x=[qmin, qmax], y=[qmin, qmax],
        mode="lines",
        line=dict(color=ACCENT_RED, dash="dash", width=1),
        showlegend=False,
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Sample Quantiles",
        height=400,
    )
    return fig

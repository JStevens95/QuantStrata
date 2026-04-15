"""
Heatmap figure builders — cluster performance, RF scenarios, adjacency.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go

from src.ui.apps.ensemble_analytics.theme.colors import BG_CARD, TEXT_SECONDARY


def cluster_heatmap(
    cluster_ids: List[str],
    metric_values: Dict[str, float],
    metric_name: str = "MAE",
    title: str = "Cluster Performance Heatmap",
) -> go.Figure:
    """
    Single-row heatmap showing one metric per cluster.

    Parameters
    ----------
    cluster_ids : list of str
        Cluster identifiers (x-axis).
    metric_values : dict
        ``{cluster_id: metric_value}``.
    metric_name : str
        Metric label for the colour bar.
    title : str
        Figure title.

    Returns
    -------
    go.Figure
    """
    values = [[metric_values.get(cid, 0.0) for cid in cluster_ids]]

    fig = go.Figure(go.Heatmap(
        z=values,
        x=cluster_ids,
        y=[metric_name],
        colorscale="RdYlGn_r",
        text=[[f"{v:.4f}" for v in values[0]]],
        texttemplate="%{text}",
        textfont=dict(size=11),
        colorbar=dict(title=metric_name, len=0.5),
    ))
    fig.update_layout(
        title=title,
        height=160 + 30 * max(1, len(cluster_ids) // 10),
        xaxis=dict(tickangle=-45),
        yaxis=dict(showticklabels=False),
    )
    return fig


def multi_metric_cluster_heatmap(
    cluster_ids: List[str],
    per_member_metrics: Dict[str, Dict[str, float]],
    metric_keys: Optional[List[str]] = None,
    title: str = "Cluster Performance Heatmap",
) -> go.Figure:
    """
    Multi-row heatmap: rows = metric names, cols = clusters.

    Designed for the Overview tab — shows MAE, RMSE, MaxAE, P95, P99
    for every cluster in one view with colour intensity encoding.

    Parameters
    ----------
    cluster_ids : list of str
        Cluster identifiers (x-axis).
    per_member_metrics : dict
        ``{cluster_id: {metric: value, ...}}``.
    metric_keys : list of str, optional
        Metrics to show.  Defaults to ``["mae", "rmse", "max_ae", "p95_ae", "p99_ae"]``.
    title : str
        Figure title.

    Returns
    -------
    go.Figure
    """
    if metric_keys is None:
        metric_keys = ["mae", "rmse", "max_ae", "p95_ae", "p99_ae"]

    z = []
    text = []
    for mk in metric_keys:
        row = [per_member_metrics.get(cid, {}).get(mk, 0.0) for cid in cluster_ids]
        z.append(row)
        text.append([f"{v:.4f}" for v in row])

    display_names = {
        "mae": "MAE", "rmse": "RMSE", "max_ae": "Max AE",
        "p95_ae": "P95 AE", "p99_ae": "P99 AE", "mape": "MAPE", "r2": "R²",
    }
    y_labels = [display_names.get(mk, mk.upper()) for mk in metric_keys]

    fig = go.Figure(go.Heatmap(
        z=z, x=cluster_ids, y=y_labels,
        colorscale="RdYlGn_r",
        text=text, texttemplate="%{text}", textfont=dict(size=10),
        colorbar=dict(title="Value", len=0.6),
    ))
    fig.update_layout(
        title=title,
        height=max(200, 40 * len(metric_keys) + 100),
        xaxis=dict(tickangle=-45),
    )
    return fig


def rf_scenario_heatmap(
    rf_names: List[str],
    shock_matrix: np.ndarray,
    title: str = "RF × Scenario Heatmap",
    max_scenarios: int = 200,
) -> go.Figure:
    """
    Heatmap of risk-factor shocks across scenarios.

    Parameters
    ----------
    rf_names : list of str
        Risk factor names (y-axis).
    shock_matrix : np.ndarray
        Shape ``[n_scenarios, n_rfs]``.
    title : str
        Figure title.
    max_scenarios : int
        Downsample scenarios if larger.

    Returns
    -------
    go.Figure
    """
    if shock_matrix.shape[0] > max_scenarios:
        idx = np.linspace(0, shock_matrix.shape[0] - 1, max_scenarios, dtype=int)
        shock_matrix = shock_matrix[idx]

    fig = go.Figure(go.Heatmap(
        z=shock_matrix.T,
        x=list(range(shock_matrix.shape[0])),
        y=rf_names,
        colorscale="RdBu_r",
        zmid=0,
        colorbar=dict(title="Shock"),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Scenario",
        height=max(300, 20 * len(rf_names) + 100),
    )
    return fig


def adjacency_spy(
    indices: np.ndarray,
    values: np.ndarray,
    shape: List[int],
    title: str = "Adjacency Matrix (Spy Plot)",
) -> go.Figure:
    """
    Sparse matrix spy plot from COO-format adjacency data.

    Parameters
    ----------
    indices : np.ndarray
        Shape ``[2, nnz]`` — row and column indices.
    values : np.ndarray
        Edge weights, shape ``[nnz]``.
    shape : list of int
        ``[n_nodes, n_nodes]``.
    title : str
        Figure title.

    Returns
    -------
    go.Figure
    """
    if indices.ndim == 2 and indices.shape[0] == 2:
        rows, cols = indices[0], indices[1]
    elif indices.ndim == 2 and indices.shape[1] == 2:
        rows, cols = indices[:, 0], indices[:, 1]
    else:
        rows, cols = indices[0], indices[1]

    n = shape[0]
    fig = go.Figure(go.Scattergl(
        x=cols,
        y=rows,
        mode="markers",
        marker=dict(size=2, color=values, colorscale="Viridis", showscale=True),
    ))
    fig.update_layout(
        title=title,
        xaxis=dict(title="Column", range=[0, n], autorange=False),
        yaxis=dict(title="Row", range=[n, 0], autorange=False),
        height=500,
    )
    return fig

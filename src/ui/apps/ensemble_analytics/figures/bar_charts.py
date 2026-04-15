"""
Bar chart figure builders — member comparisons, grouped split bars.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import plotly.graph_objects as go

from src.ui.apps.ensemble_analytics.theme.colors import ACCENT_BLUE, ACCENT_GREEN, ACCENT_AMBER, CHART_COLORS


def member_comparison_bar(
    cluster_ids: List[str],
    metric_values: Dict[str, float],
    metric_name: str = "MAE",
    title: str = "Member Comparison",
    hover_text: Optional[Dict[str, str]] = None,
) -> go.Figure:
    """
    Horizontal bar chart comparing one metric across cluster members.

    Parameters
    ----------
    cluster_ids : list of str
        Cluster identifiers.
    metric_values : dict
        ``{cluster_id: float}``.
    metric_name : str
        Metric display name.
    title : str
        Figure title.
    hover_text : dict, optional
        ``{cluster_id: description}`` shown on hover.

    Returns
    -------
    go.Figure
    """
    values = [metric_values.get(cid, 0.0) for cid in cluster_ids]
    custom = [hover_text.get(cid, "") for cid in cluster_ids] if hover_text else None

    bar_kwargs: dict = dict(
        x=values,
        y=cluster_ids,
        orientation="h",
        marker_color=ACCENT_BLUE,
        text=[f"{v:.4f}" for v in values],
        textposition="auto",
    )
    if custom:
        bar_kwargs["customdata"] = custom
        bar_kwargs["hovertemplate"] = (
            "<b>%{y}</b><br>"
            + f"{metric_name}: " + "%{x:.4f}<br>"
            + "%{customdata}"
            + "<extra></extra>"
        )

    fig = go.Figure(go.Bar(**bar_kwargs))
    fig.update_layout(
        title=title,
        xaxis_title=metric_name,
        height=max(300, 28 * len(cluster_ids) + 80),
        yaxis=dict(categoryorder="total ascending"),
    )
    return fig


def grouped_split_bar(
    labels: List[str],
    split_values: Dict[str, List[float]],
    title: str = "Metric by Split",
    y_label: str = "Value",
) -> go.Figure:
    """
    Grouped bar chart with one group per split.

    Parameters
    ----------
    labels : list of str
        X-axis category labels.
    split_values : dict
        ``{split_name: [float values per label]}``.
    title : str
        Figure title.
    y_label : str
        Y-axis label.

    Returns
    -------
    go.Figure
    """
    split_colors = {"test": ACCENT_BLUE, "val": ACCENT_AMBER, "train": ACCENT_GREEN}

    fig = go.Figure()
    for split_name, values in split_values.items():
        fig.add_trace(go.Bar(
            x=labels,
            y=values,
            name=split_name.capitalize(),
            marker_color=split_colors.get(split_name, ACCENT_BLUE),
        ))
    fig.update_layout(
        title=title,
        yaxis_title=y_label,
        barmode="group",
        height=400,
    )
    return fig

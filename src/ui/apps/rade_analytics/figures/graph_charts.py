"""Figures supporting the Evaluation → Trade-Graph sub-tab.

Two companion charts sit in the second row underneath the Cytoscape
pane:

* :func:`density_distribution` — histogram of ``density`` across all
  clusters, with the currently-selected cluster highlighted.  Lets the
  user eyeball whether the cluster they're looking at is an outlier.
* :func:`edges_vs_nodes_scatter` — per-cluster scatter of ``n_nodes``
  (x) against ``n_edges`` (y), with marker size encoding density and
  colour encoding ``mean_weight``.  Selected cluster is emphasised with
  a ring.

Both builders take a pandas DataFrame with the ``graph_stats``
schema (``cluster_id``, ``n_nodes``, ``n_edges``, ``density``,
``mean_weight``) — the exact shape returned by
``RadeBackend.graph_stats_df``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ._theme import empty_figure, rade_layout, rgba


_PRIMARY = "#8b5cf6"          # violet — default / focus highlight
_MUTED = "#475569"             # slate  — unfocused bars / markers
_POSITIVE = "#10b981"          # emerald — used for ring on selected


def density_distribution(
    graph_stats: pd.DataFrame,
    *,
    selected_cluster_id: Optional[str] = None,
    nbins: int = 18,
) -> go.Figure:
    """Histogram of per-cluster graph density.

    Parameters
    ----------
    graph_stats
        DataFrame matching ``graph_stats.parquet``.  Must contain
        ``cluster_id`` and ``density`` columns.  Zero-node clusters
        (density == 0 and ``n_nodes`` == 0 when available) are kept —
        the user deserves to see when a cluster has no graph at all.
    selected_cluster_id
        If set, overlays a vertical guide at that cluster's density so
        the user can place "their" cluster in the distribution.
    nbins
        Number of histogram bins.  Small enough to keep every bar
        readable even with 6-10 clusters; Plotly auto-bins inside this
        budget.
    """
    if graph_stats is None or graph_stats.empty or "density" not in graph_stats:
        return empty_figure("No graph stats available.")

    densities = graph_stats["density"].astype(float).to_numpy()
    if densities.size == 0:
        return empty_figure("No graph stats available.")

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=densities,
            nbinsx=nbins,
            marker={
                "color":  rgba(_PRIMARY, 0.45),
                "line":   {"color": _PRIMARY, "width": 1},
            },
            hovertemplate="Density ≈ %{x:.3g}<br>Clusters: %{y}<extra></extra>",
            name="Clusters",
        )
    )

    # Overlay selected-cluster guide line.
    if selected_cluster_id and "cluster_id" in graph_stats.columns:
        row = graph_stats[graph_stats["cluster_id"] == selected_cluster_id]
        if not row.empty:
            sel_density = float(row.iloc[0]["density"])
            fig.add_vline(
                x=sel_density,
                line_color=_POSITIVE,
                line_width=2,
                annotation_text=f"{selected_cluster_id}<br>{sel_density:.3g}",
                annotation_position="top right",
                annotation_font={"color": _POSITIVE, "size": 10},
            )

    fig.update_layout(
        **rade_layout(
            show_legend=False,
            margin={"l": 44, "r": 16, "t": 16, "b": 40},
            xaxis={"title": {"text": "Density", "standoff": 10}},
            yaxis={"title": {"text": "Clusters", "standoff": 10}},
        ),
        bargap=0.08,
    )
    return fig


def edges_vs_nodes_scatter(
    graph_stats: pd.DataFrame,
    *,
    selected_cluster_id: Optional[str] = None,
) -> go.Figure:
    """Per-cluster scatter of node-count vs edge-count.

    Marker size encodes ``density`` (rescaled to the 10-32 px band so
    sparse clusters stay visible) and marker colour encodes
    ``mean_weight`` via Plotly's built-in Viridis.  The currently-
    selected cluster gets an emerald ring so it's findable even inside
    a dense blob of markers.
    """
    required = {"cluster_id", "n_nodes", "n_edges", "density", "mean_weight"}
    if (
        graph_stats is None
        or graph_stats.empty
        or not required.issubset(graph_stats.columns)
    ):
        return empty_figure("No graph stats available.")

    df = graph_stats.copy()
    df["n_nodes"] = df["n_nodes"].astype(float)
    df["n_edges"] = df["n_edges"].astype(float)
    df["density"] = df["density"].astype(float)
    df["mean_weight"] = df["mean_weight"].astype(float)

    # Marker size: rescale density to [10, 32] so the dots read at a
    # glance across the range.  Guard against the degenerate single-
    # density case by clamping to a mid-range size.
    d_min, d_max = df["density"].min(), df["density"].max()
    if np.isclose(d_max, d_min):
        sizes = np.full(len(df), 18.0)
    else:
        sizes = 10.0 + (df["density"] - d_min) / (d_max - d_min) * 22.0

    hover_text = [
        (
            f"<b>{row.cluster_id}</b><br>"
            f"Nodes: {int(row.n_nodes)}<br>"
            f"Edges: {int(row.n_edges)}<br>"
            f"Density: {row.density:.3g}<br>"
            f"Mean weight: {row.mean_weight:.3g}"
        )
        for row in df.itertuples()
    ]

    # Emphasise the selected cluster with a thicker ring; others get
    # a thin subtle border.
    line_widths = [
        3 if cid == selected_cluster_id else 1
        for cid in df["cluster_id"]
    ]
    line_colors = [
        _POSITIVE if cid == selected_cluster_id else "rgba(15, 23, 42, 0.8)"
        for cid in df["cluster_id"]
    ]

    fig = go.Figure(
        data=go.Scatter(
            x=df["n_nodes"],
            y=df["n_edges"],
            mode="markers",
            marker={
                "size":      sizes,
                "color":     df["mean_weight"],
                "colorscale": "Viridis",
                "showscale": True,
                "colorbar": {
                    "title":     {"text": "Mean weight", "side": "right"},
                    "thickness": 10,
                    "len":       0.9,
                    "tickfont":  {"color": "#94a3b8", "size": 10},
                    "title_font": {"color": "#94a3b8", "size": 11},
                },
                "line": {
                    "color": line_colors,
                    "width": line_widths,
                },
            },
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            customdata=df["cluster_id"],
        )
    )

    fig.update_layout(
        **rade_layout(
            show_legend=False,
            margin={"l": 48, "r": 72, "t": 16, "b": 40},
            xaxis={"title": {"text": "Nodes (n)", "standoff": 10}},
            yaxis={"title": {"text": "Edges (n)", "standoff": 10}},
        ),
    )
    return fig


__all__ = ["density_distribution", "edges_vs_nodes_scatter"]

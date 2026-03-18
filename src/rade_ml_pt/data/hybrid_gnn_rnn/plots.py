"""
Visualisations and plots for the Hybrid GNN-RNN model.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from pathlib import Path
from scipy import stats
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from typing import Optional, Union, List

# define module level logging.
logger = logging.getLogger(__name__)


_ELEM_COLOUR = "#3B82F6"
_TARGET_COLOUR = "#EF4444"
_EDGE_COLOUR = "#94A3B8"
_TGT_EDGE = "#EF4444"
_CROSS_EDGE = "#8B5CF6"
_BG_COLOUR = "#FAFAFA"
_GRID_COLOUR = "#E2E8F0"


def plot_trade_graph(
        adjacency_indices: np.ndarray, adjacency_values: np.ndarray, adjacency_dense_shape: np.ndarray,
        is_target: np.ndarray, trade_ids: Optional[List[str]] = None, features: Optional[np.ndarray] = None,
        title: str = "Trade Relationship Graph", save_path: Optional[Union[str, Path]] = None,
        figsize: tuple = (14, 10), max_edges_drawn: int = 2000, max_nodes_heatmap: int = 5000,
) -> plt.Figure:
    """
    Professional four-panel visualisation of the trade relationship graph.

    Uses sparse adjacency components (indices, values, dense_shape). Dense
    conversion is done only for the heatmap panel; other panels work directly
    from the edge list to avoid memory blow-up on large graphs.

    Panels:
      (a) Adjacency heatmap — row-normalised edge weights with elementary /
          target partition lines (skipped if n > max_nodes_heatmap).
      (b) Network layout — spring-directed graph with nodes coloured by trade
          type and edges scaled by weight.
      (c) Degree & weight distribution — connectivity and total edge weight
          histograms (computed from sparse edges).
      (d) Feature space projection — 2-D PCA with k-NN edges overlaid.

    :param adjacency_indices: [nnz, 2] int array of (row, col) for each edge.
    :param adjacency_values: [nnz] float array of edge weights.
    :param adjacency_dense_shape: [2] shape (n, n) of the adjacency matrix.
    :param is_target: boolean array [n], True = target trade.
    :param trade_ids: optional list of trade identifiers.
    :param features: optional feature matrix [n, d] for PCA projection.
    :param title: suptitle for the figure.
    :param save_path: if provided, saves the figure to this path.
    :param figsize: overall figure size.
    :param max_edges_drawn: cap on edges rendered in network/feature panels.
    :param max_nodes_heatmap: skip heatmap if n exceeds this (avoids O(n^2) memory).
    :return: matplotlib Figure.
    """
    # define adjacency elements.
    indices = np.asarray(adjacency_indices, dtype=np.intp)
    values = np.asarray(adjacency_values, dtype=np.float64)
    dense_shape = np.asarray(adjacency_dense_shape, dtype=np.intp)

    # assign basic shapes and dimensions.
    n = int(dense_shape[0])
    is_target = np.asarray(is_target, dtype=bool)
    n_elem = int(np.sum(~is_target))
    n_tgt = int(np.sum(is_target))

    # assign node colours for elementary and target trades.
    node_colours = np.where(is_target, _TARGET_COLOUR, _ELEM_COLOUR)

    # dense adjacency only for heatmap (skip if too large).
    adj_dense = None
    if n <= max_nodes_heatmap:
        adj_dense = np.zeros((n, n), dtype=np.float32)
        rows, cols = indices[:, 0], indices[:, 1]
        adj_dense[rows, cols] = values.astype(np.float32)

    # build plt figure and panels.
    fig = plt.figure(figsize=figsize, facecolor="white", constrained_layout=True)
    fig.suptitle(title, fontsize=18, fontweight="bold", y=1.01)

    # add subplots / panels in figure.
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.25)
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_net = fig.add_subplot(gs[0, 1])
    ax_deg = fig.add_subplot(gs[1, 0])
    ax_feat = fig.add_subplot(gs[1, 1])

    # --- (a) Adjacency heatmap ---
    _plot_adjacency_heatmap(ax_heat, adj_dense, is_target, n_elem, n_tgt, n)

    # --- (b) Network layout ---
    g = _build_networkx_graph(indices, values, n, is_target, trade_ids)
    _plot_network(ax_net, g, node_colours, n, max_edges_drawn)

    # --- (c) Degree & weight distributions ---
    _plot_degree_distribution(ax_deg, indices, values, n, is_target)

    # --- (d) Feature space PCA projection ---
    _plot_feature_projection(ax_feat, indices, values, is_target, features, node_colours, max_edges_drawn)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(save_path / "trade_graph_analytics.png"), dpi=200, bbox_inches="tight", facecolor="white")
        logger.info(f"Trade graph plot saved to {save_path}")
    return fig


def _plot_adjacency_heatmap(
        ax: plt.Axes, adj: Optional[np.ndarray], is_target: np.ndarray, n_elem: int, n_tgt: int, n: int,
) -> None:
    """
    Panel (a): adjacency heatmap with elementary/target labelling.

    Reorders the matrix so elementary trades come first, then target. If adj
    is None (skipped for large graphs), shows a placeholder message.
    """

    # Reorder so elementary (0..n_elem-1) then target (n_elem..n-1)
    order = np.concatenate([np.where(~is_target)[0], np.where(is_target)[0]])
    adj_plot = adj[np.ix_(order, order)]

    # Main heatmap (edge weights)
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    cmap.set_bad(color=_BG_COLOUR)

    masked = np.ma.masked_where(adj_plot == 0, adj_plot)
    im = ax.imshow(masked, cmap=cmap, aspect="auto", interpolation="nearest")
    plt.colorbar(im, ax=ax, shrink=0.8, pad=0.01, fraction=0.046, label="Edge weight")

    # Colour strip above heatmap (axes coords): blue=elementary, red=target
    strip = np.array([[0 if i < n_elem else 1 for i in range(n)]], dtype=float)
    strip_cmap = mcolors.ListedColormap([_ELEM_COLOUR, _TARGET_COLOUR])
    strip_ax = ax.inset_axes([0, 0.97, 1, 0.03])
    strip_ax.imshow(strip, aspect="auto", cmap=strip_cmap, interpolation="nearest")
    strip_ax.set_xticks([])
    strip_ax.set_yticks([])

    if n_elem > 0 and n_tgt > 0:
        ax.axhline(y=n_elem - 0.5, color="white", linewidth=1.5, linestyle="--")
        ax.axvline(x=n_elem - 0.5, color="white", linewidth=1.5, linestyle="--")
    ax.set_title("(a)  Adjacency Matrix", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Trade index (elementary | target)")
    ax.set_ylabel("Trade index (elementary | target)")

    legend = [
        Patch(facecolor=_ELEM_COLOUR, label=f"Elementary ({n_elem})"),
        Patch(facecolor=_TARGET_COLOUR, label=f"Target ({n_tgt})"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=9, framealpha=0.9)


def _build_networkx_graph(
        indices: np.ndarray, values: np.ndarray, n: int, is_target: np.ndarray, trade_ids: Optional[List[str]],
) -> nx.DiGraph:
    """Build a networkx DiGraph from sparse edge list (indices, values)."""
    g = nx.DiGraph()
    labels = trade_ids if trade_ids and len(trade_ids) == n else [str(i) for i in range(n)]

    for i in range(n):
        g.add_node(i, label=labels[i], is_target=bool(is_target[i]))
    rows, cols = indices[:, 0], indices[:, 1]
    for r, c, w in zip(rows, cols, values):
        if r != c:
            g.add_edge(int(r), int(c), weight=float(w))
    return g


def _plot_network(
        ax: plt.Axes, g: nx.DiGraph, node_colours: np.ndarray, n: int, max_edges: int,
) -> None:
    """Panel (b): force-directed network layout."""
    ax.set_facecolor(_BG_COLOUR)

    pos = nx.spring_layout(g, seed=42, k=2.5 / max(1, np.sqrt(n)), iterations=100)
    edge_weights = np.array([d["weight"] for _, _, d in g.edges(data=True)])

    if len(edge_weights) > max_edges:
        threshold = np.percentile(edge_weights, 100 * (1 - max_edges / len(edge_weights)))
        draw_edges = [(u, v) for u, v, d in g.edges(data=True) if d["weight"] >= threshold]
        draw_weights = [d["weight"] for u, v, d in g.edges(data=True) if d["weight"] >= threshold]
    else:
        draw_edges = list(g.edges())
        draw_weights = list(edge_weights)

    if draw_weights:
        w_arr = np.array(draw_weights)
        edge_alphas = 0.15 + 0.55 * (w_arr / max(w_arr.max(), 1e-9))
        edge_widths = 0.4 + 1.4 * (w_arr / max(w_arr.max(), 1e-9))
    else:
        edge_alphas, edge_widths = [], []

    for (u, v), alpha, width in zip(draw_edges, edge_alphas, edge_widths):
        u_tgt = g.nodes[u]["is_target"]
        v_tgt = g.nodes[v]["is_target"]
        if u_tgt and v_tgt:
            color = _TGT_EDGE
        elif u_tgt or v_tgt:
            color = _CROSS_EDGE
        else:
            color = _EDGE_COLOUR
        ax.annotate(
            "", xy=pos[v], xytext=pos[u], arrowprops=dict(
                arrowstyle="->", color=color, alpha=float(alpha), lw=float(width), connectionstyle="arc3,rad=0.05"
            ),
        )

    # Draw nodes first (low zorder) so edges render on top
    node_sizes = np.where(
        np.array([g.nodes[i]["is_target"] for i in g.nodes()]), 60, 30,
    )
    nx.draw_networkx_nodes(
        g, pos, ax=ax, node_color=list(node_colours), node_size=node_sizes, edgecolors="white", linewidths=0.3,
        alpha=0.7,
    )

    ax.set_title("(b)  Network Layout", fontsize=13, fontweight="bold", pad=10)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_ELEM_COLOUR, markersize=8, label="Elementary"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_TARGET_COLOUR, markersize=10, label="Target"),
        Line2D([0], [0], color=_CROSS_EDGE, alpha=0.6, linewidth=1.2, label="Target ↔ Elementary"),
        Line2D([0], [0], color=_TGT_EDGE, alpha=0.6, linewidth=1.2, label="Target ↔ Target"),
        Line2D([0], [0], color=_EDGE_COLOUR, alpha=0.6, linewidth=1.2, label="Elementary ↔ Elementary"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=9, framealpha=0.9)


def _plot_degree_distribution(
        ax: plt.Axes, indices: np.ndarray, values: np.ndarray, n: int, is_target: np.ndarray,
) -> None:
    """Panel (c): degree and total edge-weight histograms, computed from sparse edges."""
    rows, cols = indices[:, 0], indices[:, 1]
    # Exclude self-loops for degree
    non_self = rows != cols
    rows_ns, cols_ns = rows[non_self], cols[non_self]
    vals_ns = values[non_self]

    degrees = np.bincount(rows_ns, minlength=n)
    weight_sums = np.zeros(n, dtype=np.float64)
    np.add.at(weight_sums, rows_ns, vals_ns)

    elem_mask = ~is_target
    tgt_mask = is_target

    max_deg = int(degrees.max()) if len(degrees) > 0 else 0
    bins_deg = np.arange(0, max_deg + 2) - 0.5
    ax.hist(degrees[elem_mask], bins=bins_deg, alpha=0.7, color=_ELEM_COLOUR, label="Elementary", edgecolor="white")
    ax.hist(degrees[tgt_mask], bins=bins_deg, alpha=0.7, color=_TARGET_COLOUR, label="Target", edgecolor="white")

    ax.set_xlabel("Node degree (excl. self-loop)")
    ax.set_ylabel("Count")
    ax.set_title("(c)  Degree Distribution", fontsize=13, fontweight="bold", pad=10)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.set_facecolor(_BG_COLOUR)
    ax.grid(axis="y", color=_GRID_COLOUR, linewidth=0.5)

    ax2 = ax.twinx()
    ax2.hist(
        weight_sums[elem_mask], bins=25, alpha=0.25, color=_ELEM_COLOUR,
        histtype="step", linewidth=1.5, linestyle="--",
    )
    ax2.hist(
        weight_sums[tgt_mask], bins=25, alpha=0.25, color=_TARGET_COLOUR,
        histtype="step", linewidth=1.5, linestyle="--",
    )
    ax2.set_ylabel("Weight sum (dashed)", fontsize=9, color="#64748B")
    ax2.tick_params(axis="y", labelcolor="#64748B", labelsize=8)


def _plot_feature_projection(
        ax: plt.Axes,
        indices: np.ndarray,
        values: np.ndarray,
        is_target: np.ndarray,
        features: Optional[np.ndarray],
        node_colours: np.ndarray,
        max_edges: int,
) -> None:
    """Panel (d): 2-D PCA of trade features with k-NN edges overlaid (from sparse edges)."""
    ax.set_facecolor(_BG_COLOUR)

    if features is None or features.shape[0] == 0:
        ax.text(0.5, 0.5, "Features not provided", ha="center", va="center", fontsize=12, color="#94A3B8")
        ax.set_title("(d)  Feature Space (PCA)", fontsize=13, fontweight="bold", pad=10)
        return

    from sklearn.decomposition import PCA
    proj = PCA(n_components=2, random_state=42).fit_transform(features)

    rows, cols = indices[:, 0], indices[:, 1]
    self_mask = rows != cols
    rows, cols = rows[self_mask], cols[self_mask]
    weights = values[self_mask]

    if len(rows) > max_edges:
        threshold = np.percentile(weights, 100 * (1 - max_edges / len(weights)))
        keep = weights >= threshold
        rows, cols, weights = rows[keep], cols[keep], weights[keep]

    # Draw nodes first (low zorder) so edges render on top
    sizes = np.where(is_target, 20, 8)
    ax.scatter(
        proj[:, 0], proj[:, 1], c=list(node_colours), s=sizes,
        edgecolors="white", linewidths=0.2, alpha=0.7, zorder=1,
    )

    # Draw edges on top of nodes
    w_norm = weights / max(weights.max(), 1e-9)
    for r, c, w in zip(rows, cols, w_norm):
        ax.plot(
            [proj[r, 0], proj[c, 0]], [proj[r, 1], proj[c, 1]],
            color=_EDGE_COLOUR, alpha=float(0.25 + 0.55 * w), linewidth=0.7, zorder=3,
        )

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("(d)  Feature Space (PCA)", fontsize=13, fontweight="bold", pad=10)
    ax.grid(color=_GRID_COLOUR, linewidth=0.5)

    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_ELEM_COLOUR, markersize=7, label="Elementary"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_TARGET_COLOUR, markersize=9, label="Target"),
        Line2D([0], [0], color=_EDGE_COLOUR, alpha=0.4, linewidth=1, label="k-NN edge"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=9, framealpha=0.9)


def plot_portfolio_pnl(
        pnl_df: pd.DataFrame, save_path: Optional[Union[str, Path]] = None, period: Optional[str] = None,
        scale_type: Optional[str] = "scaled"
) -> None:
    """
    Plot difference in aggregated pnl per scenarios.

    :param pnl_df:
    :param save_path:
    :param period:
    :param scale_type:
    :return:
    """
    # check pnl dataframe contains required columns.
    scale_type = scale_type.lower()
    assert {"target_pnl", "predicted_pnl"}.issubset(pnl_df.columns), \
        "Inputs missing required columns ['target_pnl', 'predicted_pnl']"

    # if save path, ensure folder exists.
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # ---- plot scatter graph ----
    plt.figure()
    # create scatter plot using linear regression.
    b2, b1, r2, _, _ = stats.linregress(pnl_df.target_pnl, pnl_df.predicted_pnl)
    r_squared = r2 ** 2

    # calculate regression line.
    reg_line = "y={0:.2f}+{1:.2f}x ({2:.2f}%)".format(b1, b2, r_squared * 100)

    plt.figure()
    ax = sns.regplot(
        x="target_pnl", y="predict_pnl", data=pnl_df[["target_pnl", "predicted_pnl"]],
        scatter_kws={"color": "black", "alpha": 0.25}, ci=99, line_kws={"label": reg_line},
    )
    ax.legend()
    plt.xlabel("Benchmark Pnl")
    plt.xticks(rotation=45)
    plt.ylabel("Model Pnl")
    plt.title(f"{period.upper()}: Hybrid Model vs Actual PnL")
    plt.tight_layout()
    if save_path:
        if period and scale_type:
            plt.savefig(Path(save_path, f"{period}_{scale_type}_model_vs_actual_scatter.png"))
        elif period:
            plt.savefig(Path(save_path, f"{scale_type}_model_vs_actual_scatter.png"))
        else:
            plt.savefig(Path(save_path, "model_vs_actual_scatter.png"))

    # ---- plot line graph ----
    plt.figure()
    sns.lineplot(pnl_df[["target_pnl", "predicted_pnl"]])
    plt.ylabel("PnL")
    plt.xticks(rotation=45)
    plt.title(f"{period.upper()}: Hybrid Model vs Actual PnL")
    plt.tight_layout()
    if save_path:
        if period and scale_type:
            plt.savefig(Path(save_path, f"{period}_{scale_type}_model_vs_actual_line.png"))
        elif period:
            plt.savefig(Path(save_path, f"{scale_type}_model_vs_actual_line.png"))
        else:
            plt.savefig(Path(save_path, "model_vs_actual_line.png"))
    plt.close()


def plot_kde_distributions(df: pd.DataFrame, save_path: Union[str, Path]) -> None:
    """
    Plot kernel density estimates to compare distributions of calibration and validation periods and elementary vs
    target trades

    :param df: Dataframe containing elementary pnl target pnl and period columns.
    :param save_path: file path to save plots.
    :return:
    """
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # KDE for elementary pnl
    plt.figure(figsize=(12, 6))
    sns.kdeplot(
        data=df, x='elementary_pnl', hue='period', fill=True, common_norm=False, palette='crest', alpha=0.5,
        linewidth=0
    )
    plt.title('Elementary PnL Distribution: Training vs Validation')
    plt.xlabel('PnL')
    plt.ylabel('Density')
    plt.savefig(Path(save_path, 'elem_distribution.png'))

    # CKDE for elementary pnl.
    plt.figure(figsize=(12, 6))
    sns.kdeplot(
        data=df, x='elementary_pnl', hue='period', fill=True, common_norm=False, cumulative=True, palette='crest',
        alpha=0.5,
    )
    plt.title('Elementary PnL Cumulative Distribution: Training vs Validation')
    plt.xlabel('PnL')
    plt.ylabel('Density')
    plt.savefig(Path(save_path, 'elem_c_distribution.png'))

    # KDE for target pnl.
    plt.figure(figsize=(12, 6))
    sns.kdeplot(
        data=df, x='target_pnl', hue='period', fill=True, common_norm=False, palette='crest', alpha=0.5, linewidth=0
    )
    plt.title('Target PnL Distribution: Training vs Validation')
    plt.xlabel('PnL')
    plt.ylabel('Density')
    plt.savefig(Path(save_path, 'targ_distribution.png'))

    # CKDE for target pnl.
    plt.figure(figsize=(12, 6))
    sns.kdeplot(
        data=df, x='target_pnl', hue='period', common_norm=False, alpha=0.5, cumulative=True, common_grid=True,
        palette='crest',
    )
    plt.title('Target PnL Cumulative Distribution: Training vs Validation')
    plt.xlabel('PnL')
    plt.ylabel('Density')
    plt.savefig(Path(save_path, 'targ_c_distribution.png'))

    # KDE for elementary and target pnl.
    plt.figure(figsize=(12, 6))
    sns.kdeplot(
        data=df, x='elementary_pnl', y='target_pnl', hue='period', fill=True, common_norm=False, palette='crest',
        alpha=0.5, linewidth=0
    )
    plt.title('Elementary vs Target PnL Distribution: Training vs Validation')
    plt.xlabel('Elementary PnL')
    plt.ylabel('Target PnL')
    plt.savefig(Path(save_path, 'elem_targ_distribution.png'))


def plot_pnl_distribution(
        elementary_pnl: pd.DataFrame, target_pnl: pd.DataFrame, train_indices: np.ndarray, save_path: Union[str, Path]
) -> None:
    """
    Plot showing pnl distribution across different sample periods.

    :param elementary_pnl: dataframe of elementary trade pnl history.
    :param target_pnl: dataframe of target trade pnl history.
    :param metadata: metadata from transformation concerning different sample periods.
    :param save_path: file path to save plots
    :return:
    """
    # assert pnl scenario dimensions align.
    assert elementary_pnl.shape[0] == target_pnl.shape[0], "Scenario mismatch between elementary & target pnl."

    # aggregate pnl across trades for each scenario.
    elem_pnl_agg = elementary_pnl.sum(axis=1)
    target_pnl_agg = target_pnl.sum(axis=1)

    # create dataframe for plotting.
    sample_df = pd.DataFrame(
        {
        "scenario_id": elem_pnl_agg.index,
        "elementary_pnl": elem_pnl_agg,
        "target_pnl": target_pnl_agg,
        }
    ).reset_index(drop=True)

    # create column to detail whether scenario is in training or validation
    sample_df["period"] = sample_df.index.map(lambda idx: 'training' if idx in train_indices else 'validation')

    # plot kernel density estimate distribution.
    plot_kde_distributions(df=sample_df, save_path=save_path)

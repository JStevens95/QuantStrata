"""
Visualisations and plots for the Hybrid GNN-RNN model.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

logger = logging.getLogger(__name__)


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
        elementary_pnl: pd.DataFrame, target_pnl: pd.DataFrame, metadata: Dict[str, Any], save_path: Union[str, Path]
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

    # create column to detail whether scenario is in training or validation or test.
    train_set = set(metadata["train_indices"])
    sample_df["period"] = sample_df.index.map(
        lambda idx: "training" if idx in train_set else "validation"
    )

    # plot kernel density estimate distribution.
    plot_kde_distributions(df=sample_df, save_path=save_path)


# ======================================================================
# Trade graph visualisation
# ======================================================================

_ELEM_COLOUR = "#3B82F6"
_TARGET_COLOUR = "#EF4444"
_EDGE_COLOUR = "#94A3B8"
_BG_COLOUR = "#FAFAFA"
_GRID_COLOUR = "#E2E8F0"


def plot_trade_graph(
        adjacency: np.ndarray,
        is_target: np.ndarray,
        trade_ids: Optional[List[str]] = None,
        features: Optional[np.ndarray] = None,
        title: str = "Trade Relationship Graph",
        save_path: Optional[Union[str, Path]] = None,
        figsize: tuple = (22, 16),
        max_edges_drawn: int = 2000,
) -> plt.Figure:
    """
    Professional four-panel visualisation of the trade relationship graph.

    Panels:
      (a) Adjacency heatmap — row-normalised edge weights with elementary /
          target partition lines.
      (b) Network layout — spring-directed graph with nodes coloured by trade
          type and edges scaled by weight.
      (c) Degree & weight distribution — connectivity and total edge weight
          histograms.
      (d) Feature space projection — 2-D PCA of the weighted features used to
          build the graph, with k-NN edges overlaid.

    :param adjacency: dense adjacency matrix [n, n] (row-normalised weights).
    :param is_target: boolean array [n], True = target trade.
    :param trade_ids: optional list of trade identifiers for hover / labels.
    :param features: optional weighted feature matrix [n, d] for PCA projection.
    :param title: suptitle for the figure.
    :param save_path: if provided, saves the figure to this path.
    :param figsize: overall figure size.
    :param max_edges_drawn: cap on edges rendered in the network panel to keep
        the plot readable for large graphs.
    :return: matplotlib Figure.
    """
    n = adjacency.shape[0]
    is_target = np.asarray(is_target, dtype=bool)
    n_elem = int(np.sum(~is_target))
    n_tgt = int(np.sum(is_target))

    node_colours = np.where(is_target, _TARGET_COLOUR, _ELEM_COLOUR)

    fig = plt.figure(figsize=figsize, facecolor="white", constrained_layout=True)
    fig.suptitle(title, fontsize=18, fontweight="bold", y=1.01)

    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.25)
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_net = fig.add_subplot(gs[0, 1])
    ax_deg = fig.add_subplot(gs[1, 0])
    ax_feat = fig.add_subplot(gs[1, 1])

    # --- (a) Adjacency heatmap ---
    _plot_adjacency_heatmap(ax_heat, adjacency, is_target, n_elem, n_tgt)

    # --- (b) Network layout ---
    G = _build_networkx_graph(adjacency, is_target, trade_ids)
    _plot_network(ax_net, G, node_colours, n, max_edges_drawn)

    # --- (c) Degree & weight distributions ---
    _plot_degree_distribution(ax_deg, adjacency, is_target)

    # --- (d) Feature space PCA projection ---
    _plot_feature_projection(ax_feat, adjacency, is_target, features, node_colours, max_edges_drawn)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
        logger.info(f"Trade graph plot saved to {save_path}")

    return fig


# ------------------------------------------------------------------
#  Panel helpers
# ------------------------------------------------------------------

def _plot_adjacency_heatmap(
        ax: plt.Axes, adj: np.ndarray, is_target: np.ndarray, n_elem: int, n_tgt: int,
) -> None:
    """Panel (a): adjacency heatmap with partition markers."""
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    cmap.set_bad(color=_BG_COLOUR)

    masked = np.ma.masked_where(adj == 0, adj)
    im = ax.imshow(masked, cmap=cmap, aspect="auto", interpolation="nearest")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Edge weight")

    if n_elem > 0 and n_tgt > 0:
        ax.axhline(y=n_elem - 0.5, color="white", linewidth=1.5, linestyle="--")
        ax.axvline(x=n_elem - 0.5, color="white", linewidth=1.5, linestyle="--")

    ax.set_title("(a)  Adjacency Matrix", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Trade index")
    ax.set_ylabel("Trade index")

    legend = [
        Patch(facecolor=_ELEM_COLOUR, label=f"Elementary ({n_elem})"),
        Patch(facecolor=_TARGET_COLOUR, label=f"Target ({n_tgt})"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=9, framealpha=0.9)


def _build_networkx_graph(
        adj: np.ndarray, is_target: np.ndarray, trade_ids: Optional[List[str]],
) -> nx.DiGraph:
    """Build a networkx DiGraph from the dense adjacency."""
    G = nx.DiGraph()
    n = adj.shape[0]
    labels = trade_ids if trade_ids and len(trade_ids) == n else [str(i) for i in range(n)]

    for i in range(n):
        G.add_node(i, label=labels[i], is_target=bool(is_target[i]))

    rows, cols = np.nonzero(adj)
    for r, c in zip(rows, cols):
        if r != c:
            G.add_edge(int(r), int(c), weight=float(adj[r, c]))
    return G


def _plot_network(
        ax: plt.Axes, G: nx.DiGraph, node_colours: np.ndarray, n: int, max_edges: int,
) -> None:
    """Panel (b): force-directed network layout."""
    ax.set_facecolor(_BG_COLOUR)

    pos = nx.spring_layout(G, seed=42, k=2.5 / max(1, np.sqrt(n)), iterations=80)

    edge_weights = np.array([d["weight"] for _, _, d in G.edges(data=True)])
    if len(edge_weights) > max_edges:
        threshold = np.percentile(edge_weights, 100 * (1 - max_edges / len(edge_weights)))
        draw_edges = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] >= threshold]
        draw_weights = [d["weight"] for u, v, d in G.edges(data=True) if d["weight"] >= threshold]
    else:
        draw_edges = list(G.edges())
        draw_weights = list(edge_weights)

    if draw_weights:
        w_arr = np.array(draw_weights)
        edge_alphas = 0.08 + 0.5 * (w_arr / max(w_arr.max(), 1e-9))
        edge_widths = 0.3 + 1.2 * (w_arr / max(w_arr.max(), 1e-9))
    else:
        edge_alphas, edge_widths = [], []

    for (u, v), alpha, width in zip(draw_edges, edge_alphas, edge_widths):
        ax.annotate(
            "", xy=pos[v], xytext=pos[u],
            arrowprops=dict(
                arrowstyle="-", color=_EDGE_COLOUR, alpha=float(alpha),
                lw=float(width), connectionstyle="arc3,rad=0.05",
            ),
        )

    node_sizes = np.where(
        np.array([G.nodes[i]["is_target"] for i in G.nodes()]), 60, 30,
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=list(node_colours), node_size=node_sizes,
        edgecolors="white", linewidths=0.4,
    )

    ax.set_title("(b)  Network Layout", fontsize=13, fontweight="bold", pad=10)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_ELEM_COLOUR, markersize=8, label="Elementary"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_TARGET_COLOUR, markersize=10, label="Target"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=9, framealpha=0.9)


def _plot_degree_distribution(
        ax: plt.Axes, adj: np.ndarray, is_target: np.ndarray,
) -> None:
    """Panel (c): degree and total edge-weight histograms."""
    binary = (adj > 0).astype(float)
    np.fill_diagonal(binary, 0)
    degrees = binary.sum(axis=1).astype(int)
    weight_sums = adj.sum(axis=1) - np.diag(adj)

    elem_mask = ~is_target
    tgt_mask = is_target

    bins_deg = np.arange(0, degrees.max() + 2) - 0.5
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
        ax: plt.Axes, adj: np.ndarray, is_target: np.ndarray,
        features: Optional[np.ndarray], node_colours: np.ndarray, max_edges: int,
) -> None:
    """Panel (d): 2-D PCA of trade features with k-NN edges overlaid."""
    ax.set_facecolor(_BG_COLOUR)

    if features is None or features.shape[0] == 0:
        ax.text(0.5, 0.5, "Features not provided", ha="center", va="center", fontsize=12, color="#94A3B8")
        ax.set_title("(d)  Feature Space (PCA)", fontsize=13, fontweight="bold", pad=10)
        return

    from sklearn.decomposition import PCA
    proj = PCA(n_components=2, random_state=42).fit_transform(features)

    rows, cols = np.nonzero(adj)
    self_mask = rows != cols
    rows, cols = rows[self_mask], cols[self_mask]
    weights = adj[rows, cols]

    if len(rows) > max_edges:
        threshold = np.percentile(weights, 100 * (1 - max_edges / len(weights)))
        keep = weights >= threshold
        rows, cols, weights = rows[keep], cols[keep], weights[keep]

    w_norm = weights / max(weights.max(), 1e-9)
    for r, c, w in zip(rows, cols, w_norm):
        ax.plot(
            [proj[r, 0], proj[c, 0]], [proj[r, 1], proj[c, 1]],
            color=_EDGE_COLOUR, alpha=float(0.05 + 0.35 * w), linewidth=0.4, zorder=1,
        )

    sizes = np.where(is_target, 40, 18)
    ax.scatter(
        proj[:, 0], proj[:, 1], c=list(node_colours), s=sizes,
        edgecolors="white", linewidths=0.3, zorder=2,
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


"""
Ensemble-specific visualisations.

All functions accept pre-computed data (metrics dicts, numpy arrays) and
produce matplotlib figures.  They share a consistent dark theme and can
save to disk or display interactively.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_PALETTE = [
    "#3B82F6", "#EF4444", "#10B981", "#F59E0B", "#8B5CF6",
    "#EC4899", "#06B6D4", "#F97316", "#6366F1", "#14B8A6",
]


def plot_member_comparison(
    member_metrics: Dict[str, Dict[str, float]],
    metric: str = "mae",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Horizontal bar chart comparing one metric across all members.

    Parameters
    ----------
    member_metrics : dict
        ``{cluster_id: {mae: float, ...}}``.
    metric : str
        Which metric to plot.
    """
    import matplotlib.pyplot as plt

    clusters = sorted(member_metrics.keys())
    values = [member_metrics[c].get(metric, 0.0) for c in clusters]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(clusters))))
    colours = [_PALETTE[i % len(_PALETTE)] for i in range(len(clusters))]

    bars = ax.barh(clusters, values, color=colours, edgecolor="white", linewidth=0.5)
    ax.set_xlabel(metric.upper())
    ax.set_title(title or f"Member Comparison — {metric.upper()}")
    ax.invert_yaxis()

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=8,
        )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_cluster_performance_heatmap(
    member_metrics: Dict[str, Dict[str, float]],
    metrics: Optional[List[str]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Heatmap with clusters on the y-axis and metrics on the x-axis.

    Parameters
    ----------
    member_metrics : dict
        ``{cluster_id: {mae: float, mse: float, ...}}``.
    metrics : list of str or None
        Which metrics to include.  Defaults to all numeric keys.
    """
    import matplotlib.pyplot as plt

    clusters = sorted(member_metrics.keys())

    if metrics is None:
        all_keys = set()
        for m in member_metrics.values():
            all_keys.update(k for k, v in m.items() if isinstance(v, (int, float)))
        metrics = sorted(all_keys)

    data = np.array([
        [member_metrics[c].get(m, 0.0) for m in metrics]
        for c in clusters
    ])

    fig, ax = plt.subplots(figsize=(max(6, len(metrics) * 1.2), max(3, len(clusters) * 0.5)))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(clusters)))
    ax.set_yticklabels(clusters, fontsize=8)

    for i in range(len(clusters)):
        for j in range(len(metrics)):
            ax.text(j, i, f"{data[i, j]:.4f}", ha="center", va="center", fontsize=7)

    plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    ax.set_title(title or "Cluster Performance Heatmap")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_version_comparison(
    comparison: Dict[str, Any],
    version_a: str = "A",
    version_b: str = "B",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Grouped bar chart comparing two ensemble versions metric-by-metric.

    Parameters
    ----------
    comparison : dict
        Output of ``build_version_comparison()``.
    """
    import matplotlib.pyplot as plt

    metric_names = sorted(comparison.keys())
    vals_a = [comparison[m].get(version_a, 0.0) or 0.0 for m in metric_names]
    vals_b = [comparison[m].get(version_b, 0.0) or 0.0 for m in metric_names]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(metric_names) * 1.0), 5))
    ax.bar(x - width / 2, vals_a, width, label=version_a, color=_PALETTE[0])
    ax.bar(x + width / 2, vals_b, width, label=version_b, color=_PALETTE[1])

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Value")
    ax.set_title(title or "Version Comparison")
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_ensemble_vs_members(
    ensemble_preds: np.ndarray,
    member_preds: Dict[str, np.ndarray],
    targets: np.ndarray,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Scatter comparing ensemble MAE vs per-member MAE for each target.

    One point per target; x-axis is ensemble MAE, y-axis is the member
    MAE (coloured by cluster).
    """
    import matplotlib.pyplot as plt

    ensemble_mae = np.mean(np.abs(ensemble_preds - targets), axis=0)

    fig, ax = plt.subplots(figsize=(7, 7))
    colour_idx = 0
    for cid, preds in sorted(member_preds.items()):
        member_targets_slice = targets  # placeholder — caller should pass aligned targets
        member_mae = np.mean(np.abs(preds - member_targets_slice[:, :preds.shape[-1]]), axis=0)

        ax.scatter(
            ensemble_mae[:len(member_mae)], member_mae,
            s=15, alpha=0.6, color=_PALETTE[colour_idx % len(_PALETTE)],
            label=cid,
        )
        colour_idx += 1

    lim = max(ensemble_mae.max(), 0.01) * 1.1
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, linewidth=0.8)
    ax.set_xlabel("Ensemble MAE per target")
    ax.set_ylabel("Member MAE per target")
    ax.set_title(title or "Ensemble vs Member MAE")
    ax.legend(fontsize=8)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def save_ensemble_plots(
    member_metrics: Dict[str, Dict[str, float]],
    save_dir: Path,
    ensemble_metrics: Optional[Dict[str, float]] = None,
) -> None:
    """
    Convenience function to save standard ensemble plots to *save_dir*.

    Used by the ensemble eval pipeline for automated artifact generation.
    """
    import matplotlib
    matplotlib.use("Agg")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        plot_member_comparison(
            member_metrics, metric="mae",
            save_path=str(save_dir / "member_comparison_mae.png"), show=False,
        )
    except Exception as exc:
        logger.warning("Could not generate member comparison plot: %s", exc)

    try:
        plot_cluster_performance_heatmap(
            member_metrics,
            save_path=str(save_dir / "cluster_performance_heatmap.png"), show=False,
        )
    except Exception as exc:
        logger.warning("Could not generate cluster heatmap: %s", exc)

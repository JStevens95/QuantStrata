"""
Residual diagnostic plots for model evaluation.

All functions accept an EvaluationResult (which carries .predictions, .targets,
.residuals as numpy arrays) and produce matplotlib figures. An optional
*save_path* persists the figure to disk.

Usage:
    from rade_ml.evaluation.plots.residuals import plot_residual_distribution

    result = evaluator.run(test_ds)
    plot_residual_distribution(result, save_path="reports/")
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from pathlib import Path
from typing import Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from src.rade_ml_pt.core.types import EvaluationResult


def plot_residual_distribution(
    result: "EvaluationResult",
    bins: int = 80,
    figsize: tuple = (14, 5),
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Histogram + KDE of residuals with summary statistics overlay.

    :param result: EvaluationResult with .residuals populated.
    :param bins: number of histogram bins.
    :param figsize: figure size (width, height).
    :param save_path: directory to save figure. None to skip saving.
    """
    residuals = np.asarray(result.residuals).flatten()

    fig, (ax_hist, ax_box) = plt.subplots(1, 2, figsize=figsize)

    # --- histogram + KDE ---
    ax_hist.hist(residuals, bins=bins, density=True, alpha=0.6, edgecolor="black", linewidth=0.3)
    xs = np.linspace(residuals.min(), residuals.max(), 300)
    kde = stats.gaussian_kde(residuals)
    ax_hist.plot(xs, kde(xs), linewidth=1.5, label="KDE")

    mu, sigma = float(np.mean(residuals)), float(np.std(residuals))
    ax_hist.axvline(mu, color="red", linestyle="--", alpha=0.8, label=f"Mean={mu:.4f}")
    ax_hist.axvline(mu + 2 * sigma, color="orange", linestyle=":", alpha=0.7, label=f"+2\u03c3={mu + 2 * sigma:.4f}")
    ax_hist.axvline(mu - 2 * sigma, color="orange", linestyle=":", alpha=0.7, label=f"-2\u03c3={mu - 2 * sigma:.4f}")

    ax_hist.set_xlabel("Residual")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Residual Distribution")
    ax_hist.legend(fontsize=8)
    ax_hist.grid(True, alpha=0.3)

    # --- box plot ---
    bp = ax_box.boxplot(residuals, vert=True, patch_artist=True, showmeans=True)
    bp["boxes"][0].set_facecolor("#89CFF0")
    ax_box.set_ylabel("Residual")
    ax_box.set_title("Residual Box Plot")
    ax_box.grid(True, alpha=0.3)

    plt.tight_layout()
    _maybe_save(fig, save_path, "residual_distribution.png")
    plt.show()


def plot_qq(
    result: "EvaluationResult",
    figsize: tuple = (6, 6),
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    QQ plot of residuals against a standard normal distribution.

    Heavy tails will deviate from the diagonal, signalling non-Gaussian
    residual structure (common in PnL prediction).

    :param result: EvaluationResult with .residuals populated.
    :param figsize: figure size (width, height).
    :param save_path: directory to save figure. None to skip saving.
    """
    residuals = np.asarray(result.residuals).flatten()
    standardised = (residuals - np.mean(residuals)) / (np.std(residuals) + 1e-12)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    stats.probplot(standardised, dist="norm", plot=ax)
    ax.set_title("QQ Plot (Residuals vs Normal)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _maybe_save(fig, save_path, "residual_qq.png")
    plt.show()


def plot_residual_scatter(
    result: "EvaluationResult",
    figsize: tuple = (10, 5),
    alpha: float = 0.3,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Scatter plot of residuals against predicted values.

    Heteroscedasticity appears as a fan/funnel shape; systematic bias
    appears as a non-zero trend line.

    :param result: EvaluationResult with .predictions and .residuals populated.
    :param figsize: figure size (width, height).
    :param alpha: scatter point transparency.
    :param save_path: directory to save figure. None to skip saving.
    """
    preds = np.asarray(result.predictions).flatten()
    residuals = np.asarray(result.residuals).flatten()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.scatter(preds, residuals, alpha=alpha, s=8, edgecolors="none")
    ax.axhline(0, color="red", linewidth=1, linestyle="--")

    z = np.polyfit(preds, residuals, deg=1)
    trend = np.poly1d(z)
    xs = np.linspace(preds.min(), preds.max(), 100)
    ax.plot(xs, trend(xs), color="orange", linewidth=1.5, label=f"Trend (slope={z[0]:.4f})")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals vs Predicted")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _maybe_save(fig, save_path, "residual_scatter.png")
    plt.show()


def plot_residual_by_target(
    result: "EvaluationResult",
    target_labels: Optional[list] = None,
    top_n: int = 20,
    figsize: tuple = (12, 6),
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Per-target MAE bar chart showing which targets the model struggles with.

    Useful for the GNN-RNN model where each column is a different trade.

    :param result: EvaluationResult with .residuals populated (2D: [samples, targets]).
    :param target_labels: optional list of target names for the x-axis.
    :param top_n: show only the top N worst targets.
    :param figsize: figure size (width, height).
    :param save_path: directory to save figure. None to skip saving.
    """
    residuals = np.asarray(result.residuals)
    if residuals.ndim == 1:
        return

    per_target_mae = np.mean(np.abs(residuals), axis=0)
    n_targets = len(per_target_mae)

    if target_labels is None:
        target_labels = [f"T{i}" for i in range(n_targets)]

    order = np.argsort(per_target_mae)[::-1][:top_n]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    labels = [target_labels[i] for i in order]
    values = per_target_mae[order]

    ax.barh(range(len(order)), values, color="#4C72B0", edgecolor="black", linewidth=0.3)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("MAE")
    ax.set_title(f"Top {top_n} Targets by MAE")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    _maybe_save(fig, save_path, "residual_by_target.png")
    plt.show()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _maybe_save(fig, save_path: Optional[Union[str, Path]], filename: str) -> None:
    """Save figure to *save_path/filename* if a path is provided."""
    if save_path is not None:
        out_dir = Path(save_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")

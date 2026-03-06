"""
Prediction diagnostic plots for model evaluation.

All functions accept an EvaluationResult and produce matplotlib figures
comparing predictions against ground truth. An optional *save_path*
persists the figure to disk.

Usage:
    from rade_ml.evaluation.plots.predictions import plot_predicted_vs_actual

    result = evaluator.run(test_ds)
    plot_predicted_vs_actual(result, save_path="reports/")
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from src.rade_ml_pt.core.types import EvaluationResult


def plot_predicted_vs_actual(
    result: "EvaluationResult",
    figsize: tuple = (7, 7),
    alpha: float = 0.3,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Scatter plot of predicted vs actual values with a 45-degree reference line.

    Points clustering on the diagonal indicate good fit; systematic
    deviation reveals bias.

    :param result: EvaluationResult with .predictions and .targets populated.
    :param figsize: figure size (width, height).
    :param alpha: scatter point transparency.
    :param save_path: directory to save figure. None to skip saving.
    """
    preds = np.asarray(result.predictions).flatten()
    targets = np.asarray(result.targets).flatten()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.scatter(targets, preds, alpha=alpha, s=8, edgecolors="none")

    lo = min(targets.min(), preds.min())
    hi = max(targets.max(), preds.max())
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            color="red", linewidth=1, linestyle="--", label="Perfect fit")

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs Actual")
    ax.legend(fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _maybe_save(fig, save_path, "predicted_vs_actual.png")
    plt.show()


def plot_error_distribution(
    result: "EvaluationResult",
    bins: int = 60,
    figsize: tuple = (10, 5),
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Histogram of absolute errors with P95 and P99 vertical lines.

    :param result: EvaluationResult with .residuals populated.
    :param bins: number of histogram bins.
    :param figsize: figure size (width, height).
    :param save_path: directory to save figure. None to skip saving.
    """
    abs_errors = np.abs(np.asarray(result.residuals).flatten())

    p95 = float(np.percentile(abs_errors, 95))
    p99 = float(np.percentile(abs_errors, 99))

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.hist(abs_errors, bins=bins, density=True, alpha=0.6, edgecolor="black", linewidth=0.3)

    ax.axvline(p95, color="orange", linestyle="--", linewidth=1.5, label=f"P95 = {p95:.4f}")
    ax.axvline(p99, color="red", linestyle="--", linewidth=1.5, label=f"P99 = {p99:.4f}")

    ax.set_xlabel("Absolute Error")
    ax.set_ylabel("Density")
    ax.set_title("Absolute Error Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _maybe_save(fig, save_path, "error_distribution.png")
    plt.show()


def plot_cumulative_error(
    result: "EvaluationResult",
    figsize: tuple = (10, 5),
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Empirical CDF of absolute errors.

    Shows what fraction of predictions fall within a given error threshold.

    :param result: EvaluationResult with .residuals populated.
    :param figsize: figure size (width, height).
    :param save_path: directory to save figure. None to skip saving.
    """
    abs_errors = np.sort(np.abs(np.asarray(result.residuals).flatten()))
    cdf = np.arange(1, len(abs_errors) + 1) / len(abs_errors)

    p95_val = float(np.percentile(abs_errors, 95))

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(abs_errors, cdf, linewidth=1.5)
    ax.axhline(0.95, color="orange", linestyle=":", alpha=0.7, label="95%")
    ax.axvline(p95_val, color="orange", linestyle=":", alpha=0.7)

    ax.set_xlabel("Absolute Error")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Cumulative Error Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _maybe_save(fig, save_path, "cumulative_error.png")
    plt.show()


def plot_prediction_timeseries(
    result: "EvaluationResult",
    target_idx: int = 0,
    scenario_labels: Optional[list] = None,
    figsize: tuple = (14, 5),
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Time-series overlay of predicted vs actual for a single target.

    For multi-target models (e.g. GNN-RNN with N target trades), select
    which target column to plot via *target_idx*.

    :param result: EvaluationResult with .predictions and .targets populated.
    :param target_idx: column index of the target to plot (for 2D outputs).
    :param scenario_labels: optional x-axis labels (e.g. scenario dates).
    :param figsize: figure size (width, height).
    :param save_path: directory to save figure. None to skip saving.
    """
    preds = np.asarray(result.predictions)
    targets = np.asarray(result.targets)

    if preds.ndim == 2:
        preds = preds[:, target_idx]
        targets = targets[:, target_idx]
    else:
        preds = preds.flatten()
        targets = targets.flatten()

    n = len(preds)
    x = np.arange(n) if scenario_labels is None else scenario_labels

    fig, (ax_ts, ax_err) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                         gridspec_kw={"height_ratios": [3, 1]})

    ax_ts.plot(x, targets, linewidth=1, label="Actual", alpha=0.9)
    ax_ts.plot(x, preds, linewidth=1, label="Predicted", alpha=0.9)
    ax_ts.set_ylabel("PnL")
    ax_ts.set_title(f"Prediction vs Actual (target {target_idx})")
    ax_ts.legend(fontsize=8)
    ax_ts.grid(True, alpha=0.3)

    errors = preds - targets
    ax_err.bar(x, errors, color=np.where(errors >= 0, "#4C72B0", "#C44E52"),
               alpha=0.7, width=1.0)
    ax_err.axhline(0, color="black", linewidth=0.5)
    ax_err.set_ylabel("Error")
    ax_err.set_xlabel("Scenario")
    ax_err.grid(True, alpha=0.3)

    plt.tight_layout()
    _maybe_save(fig, save_path, f"prediction_timeseries_t{target_idx}.png")
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

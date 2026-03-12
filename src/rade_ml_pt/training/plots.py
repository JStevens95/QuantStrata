"""
Training plot utilities for rade ML framework.

Provides multi-panel training dynamics figures (loss curves, train-val gap,
val/train ratio, other metrics) for inclusion in training reports.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.rade_ml_pt.core.types import TrainingResult


def _create_training_figure(
        result: TrainingResult,
        figsize: Tuple[float, float] = (14, 10),
) -> Optional[plt.Figure]:
    """
    Build the multi-panel training dynamics figure.

    Panels:
      (a) Loss curves — train & val vs epoch with best-epoch marker.
      (b) Train-val gap — (val_loss - train_loss) vs epoch (overfitting indicator).
      (c) Val/train ratio — val_loss / train_loss vs epoch.
      (d) Other metrics — any additional history keys (e.g. mae, val_mae).

    :param result: TrainingResult with history dict.
    :param figsize: figure dimensions.
    :return: matplotlib Figure or None if history is empty.
    """
    history = result.history or {}
    loss_vals = history.get("loss", [])
    if not history or not loss_vals:
        return None

    n_epochs = len(loss_vals)
    epochs = range(1, n_epochs + 1)
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    ax_a, ax_b, ax_c, ax_d = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]

    train_loss = np.array(loss_vals)
    val_vals = history.get("val_loss")
    has_val = val_vals is not None and len(val_vals) == n_epochs
    val_loss_arr = np.array(val_vals) if has_val else np.full(n_epochs, np.nan)

    # --- (a) Loss curves ---
    ax_a.plot(epochs, train_loss, label="train", linewidth=1.5, color="#2563eb")
    if has_val:
        ax_a.plot(epochs, val_loss_arr, label="val", linewidth=1.5, color="#dc2626")
    if result.best_epoch > 0:
        ax_a.axvline(
            result.best_epoch,
            color="green",
            linestyle="--",
            alpha=0.7,
            label=f"Best ({result.best_epoch})",
        )
    ax_a.set_xlabel("Epoch")
    ax_a.set_ylabel("Loss")
    ax_a.set_title("(a)  Loss Curves")
    ax_a.legend()
    ax_a.grid(True, alpha=0.3)

    # --- (b) Train-val gap ---
    if has_val:
        gap = val_loss_arr - train_loss
        ax_b.plot(epochs, gap, color="#7c3aed", linewidth=1.5)
        ax_b.axhline(0, color="gray", linestyle="-", alpha=0.4)
        if result.best_epoch > 0:
            ax_b.axvline(result.best_epoch, color="green", linestyle="--", alpha=0.7)
    else:
        ax_b.text(0.5, 0.5, "No val loss", ha="center", va="center", transform=ax_b.transAxes)
    ax_b.set_xlabel("Epoch")
    ax_b.set_ylabel("Val − Train")
    ax_b.set_title("(b)  Train-Val Gap (Overfitting)")
    ax_b.grid(True, alpha=0.3)

    # --- (c) Val/train ratio ---
    if has_val:
        eps = 1e-8
        ratio = val_loss_arr / (train_loss + eps)
        ax_c.plot(epochs, ratio, color="#059669", linewidth=1.5)
        ax_c.axhline(1, color="gray", linestyle="-", alpha=0.4)
        if result.best_epoch > 0:
            ax_c.axvline(result.best_epoch, color="green", linestyle="--", alpha=0.7)
    else:
        ax_c.text(0.5, 0.5, "No val loss", ha="center", va="center", transform=ax_c.transAxes)
    ax_c.set_xlabel("Epoch")
    ax_c.set_ylabel("Val / Train")
    ax_c.set_title("(c)  Val/Train Ratio")
    ax_c.grid(True, alpha=0.3)

    # --- (d) Other metrics (val_mae, val_mse, etc. from TrainingConfig.metrics) ---
    skip = {"loss", "val_loss"}
    other = [(k, v) for k, v in history.items() if k not in skip and isinstance(v, (list, tuple)) and len(v) == n_epochs]
    if other:
        for k, v in other:
            ax_d.plot(epochs, np.asarray(v), label=k.replace("_", " "), linewidth=1.2)
        if result.best_epoch > 0:
            ax_d.axvline(result.best_epoch, color="green", linestyle="--", alpha=0.7)
        ax_d.legend(fontsize=8)
    else:
        ax_d.text(
            0.5, 0.5,
            "No additional metrics\n\nSet training_config.metrics to e.g.\n[\"mae\", \"mse\"] to plot val_mae, val_mse.",
            ha="center", va="center", transform=ax_d.transAxes, fontsize=10,
        )
    ax_d.set_xlabel("Epoch")
    ax_d.set_ylabel("Metric")
    ax_d.set_title("(d)  Other Metrics")
    ax_d.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def save_training_plots(result: TrainingResult, save_dir: Path) -> Path:
    """
    Save multi-panel training dynamics figure to disk.

    :param result: TrainingResult with history dict.
    :param save_dir: directory to save the PNG.
    :return: path to the saved figure.
    """
    plot_path = save_dir / "training_plots.png"

    prev_backend = matplotlib.get_backend()
    matplotlib.use("Agg")
    try:
        fig = _create_training_figure(result)
        if fig is not None:
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
    finally:
        matplotlib.use(prev_backend)

    return plot_path


def show_training_plots(result: TrainingResult) -> None:
    """
    Display multi-panel training dynamics figure on screen.

    :param result: TrainingResult with history dict.
    """
    fig = _create_training_figure(result)
    if fig is not None:
        plt.show()

"""
Training plot utilities for rade ML framework.

Provides multi-panel training dynamics figures (loss curves, train-val gap,
val/train ratio, other metrics) for inclusion in training reports.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.rade_ml.core.types import TrainingResult


def save_training_plots(result: TrainingResult, save_dir: Path) -> Path:
    """
    Save multi-panel training dynamics figure.

    Panels:
      (a) Loss curves — train & val vs epoch with best-epoch marker.
      (b) Train-val gap — (val_loss - train_loss) vs epoch (overfitting indicator).
      (c) Val/train ratio — val_loss / train_loss vs epoch.
      (d) Other metrics — any additional history keys (e.g. mae, val_mae).

    :param result: TrainingResult with history dict.
    :param save_dir: directory to save the PNG.
    :return: path to the saved figure.
    """
    history = result.history or {}
    loss_vals = history.get("loss", [])
    if not history or not loss_vals:
        return save_dir / "training_plots.png"

    n_epochs = len(loss_vals)
    epochs = range(1, n_epochs + 1)
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
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

    # --- (d) Other metrics ---
    skip = {"loss", "val_loss"}
    other = [(k, v) for k, v in history.items() if k not in skip and len(v) == n_epochs]
    if other:
        for k, v in other:
            ax_d.plot(epochs, v, label=k.replace("_", " "), linewidth=1.2)
        if result.best_epoch > 0:
            ax_d.axvline(result.best_epoch, color="green", linestyle="--", alpha=0.7)
        ax_d.legend(fontsize=8)
    else:
        ax_d.text(
            0.5, 0.5,
            "No additional metrics\n(loss & val_loss only)",
            ha="center", va="center", transform=ax_d.transAxes, fontsize=11,
        )
    ax_d.set_xlabel("Epoch")
    ax_d.set_ylabel("Metric")
    ax_d.set_title("(d)  Other Metrics")
    ax_d.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = save_dir / "training_plots.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return plot_path

"""
Training plot utilities for rade ML framework.

Provides multi-panel training dynamics figures (loss curves, train-val gap,
val/train ratio, other metrics) for inclusion in training reports.
"""
from __future__ import annotations

import matplotlib

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from src.rade_ml_pt.core.types import TrainingResult

matplotlib.use("Agg")


def plot_training_analytics(result: TrainingResult, save_dir: Path) -> Path:
    """
    Save multi-panel training dynamic figures.

    Panels:
        1. Loss curves - train and val epoch with best-epoch marker.
        2. Train-val gap - (val loss vs train loss) vs epoch (overfitting indicator).
        3. Val/train ratio - val loss / train loss vs epoch.
        4. Other metrics - any additional history keys (e.g. mae, val_mae).
    :param result:
    :param save_dir:
    :return:
    """
    history = result.history or {}
    loss_vals = history.get("loss", [])
    if not history or not loss_vals:
        return save_dir / "training_analytics.png"

    n_epochs = len(loss_vals)
    epochs = range(1, n_epochs + 1)
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
    ax_a, ax_b, ax_c, ax_d = axs[0 ,0], axs[0 ,1], axs[1, 0], axs[1, 1]

    train_loss = np.array(loss_vals)
    val_vals = history.get("val_loss", [])
    has_val = val_vals is not None and len(val_vals) == n_epochs
    val_loss_arr = np.array(val_vals) if has_val else np.full(n_epochs, np.nan)

    # ---- loss curve ----
    ax_a.plot(epochs, train_loss, label="train", linewidth=1.5, color="£2563eb")
    if has_val:
        ax_a.plot(epochs, val_loss_arr, label="val", linewidth=1.5, color="#dc2626")
    if result.best_epoch > 0:
        ax_a.axvline(result.best_epoch, color="green", linestyle="--", alpha=0.7, label=f"Best ({result.best_epoch})")
    ax_a.set_xlabel("Epoch")
    ax_a.set_ylabel("Loss")
    ax_a.set_title("(1)     Loss Curves")
    ax_a.legend()
    ax_a.grid(True, alpha=0.3)

    # ---- train-val gap ----
    if has_val:
        gap = val_loss_arr - train_loss
        ax_b.plot(epochs, gap, color="#7c3aed", linewidth=1.5)
        ax_b.axhline(0, color="grey", linestyle="--", alpha=0.4)
        if result.best_epoch > 0:
            ax_b.axvline(result.best_epoch, color="green", linestyle="--", alpha=0.7)
    else:
        ax_b.text(0.5, 0.5, "No Val Loss", ha="center", va="center", transform=ax_b.transAxes)
    ax_b.set_xlabel("Epoch")
    ax_b.set_ylabel("Val - Train")
    ax_b.set_title("(2)     Train-Val Gap (Overfitting).")
    ax_b.grid(True, alpha=0.3)

    # ---- Val/Train ratio ----.
    if has_val:
        eps = 1e-8
        ratio = val_loss_arr / (train_loss + eps)
        ax_c.plot(epochs, ratio, color="#059669", linewidth=1.5)
        ax_c.axhline(1, color="grey", linestyle="--", alpha=0.4)
        if result.best_epoch > 0:
            ax_c.axvline(result.best_epoch, color="green", linestyle="--", alpha=0.7)
    else:
        ax_c.text(0.5, 0.5, "No Val Loss", ha="center", va="center", transform=ax_c.transAxes)
    ax_c.set_xlabel("Epoch")
    ax_c.set_ylabel("Val / Train")
    ax_c.set_title("(3)     Val/Train Ratio.")
    ax_c.grid(True, alpha=0.3)

    # ---- Other metrics ----
    skip = {"loss", "val_loss"}
    other = [(k, v) for k, v in history.items() if k not in skip and len(v) == n_epochs]
    if other:
        for k, v in other:
            ax_d.plot(epochs, v, label=k.replace("_", " "), linewidth=1.2)
        if result.best_epoch > 0:
            ax_d.axvline(result.best_epoch, color="green", linestyle="--", alpha=0.7)
        ax_d.legend(fontsize=8)
    else:
        ax_d.text(0.5, 0.5, "No additional metrics\n(loss & val loss only)", ha="center", va="center",
                  transform=ax_d.transAxes, fontsize=11)
    ax_d.set_xlabel("Epoch")
    ax_d.set_ylabel("Metric")
    ax_d.set_title("(4)     Other Metrics")
    ax_d.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle(" --- Model Training Analytics ---", fontsize=18, fontweight="bold", y=1.01)
    plot_path = save_dir / "training_analytics.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return plot_path

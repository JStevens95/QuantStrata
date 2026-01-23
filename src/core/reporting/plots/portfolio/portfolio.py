"""
Pricing plots: spot sweeps for portfolios (PV / PnL / Greeks).

Design goals
------------
- Matplotlib-only (no seaborn dependency; deterministic; CI-friendly).
- Professional-ish visuals via a tiny local style helper (no global rcParams changes).
- Simple APIs that accept numpy arrays / dataclasses (easy to test).
- Save .png (raster) + .pdf (vector) for report-quality output.

This module is intentionally generic:
- It does not know about orchestrator Context or pipelines.
- It does not know how you built the Market or Portfolio.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# =============================================================================
# Plot configuration (small, explicit, stable)
# =============================================================================

@dataclass(frozen=True, slots=True)
class PlotOptions:
    """
    Small plot configuration for examples.

    Notes
    -----
    - We avoid storing matplotlib objects in config to keep this serializable-ish.
    - "block" only matters when show=True.
    """
    show: bool = True
    save: bool = False
    out_dir: Path = Path("outputs/plots")
    dpi: int = 160
    block: bool = True
    close: bool = False


# =============================================================================
# Style helpers (local, minimal, no global side effects)
# =============================================================================

def _comma_formatter(x: float, _pos: int) -> str:
    """Format large tick labels with commas (e.g. 1000000 -> 1,000,000)."""
    return f"{x:,.0f}"


def _apply_plot_style(ax: plt.Axes) -> None:
    """
    Apply a clean, presentation-friendly style to the axes.

    Why local styling?
    ------------------
    - No seaborn dependency.
    - No global matplotlib style changes that leak into other plots/tests.
    """
    # Light grid for readability (especially on PV/PnL curves).
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.5)

    # Slightly smaller tick labels (helps on dense sweeps).
    ax.tick_params(axis="both", labelsize=10)

    # Remove top/right spines to reduce visual clutter.
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _ensure_out_dir(cfg: PlotOptions) -> None:
    """Create output directory if saving is enabled."""
    if cfg.save:
        cfg.out_dir.mkdir(parents=True, exist_ok=True)


def _save_or_show(fig: plt.Figure, *, cfg: PlotOptions, filename: str) -> None:
    """
    Save/show a matplotlib figure following PlotOptions.

    Save policy
    -----------
    - Always saves .png (raster) when cfg.save=True.
    - Also saves a matching .pdf (vector) for report-quality outputs.
    """
    if cfg.save:
        _ensure_out_dir(cfg)

        # Save PNG for quick viewing.
        png_path = cfg.out_dir / filename
        fig.savefig(png_path, dpi=int(cfg.dpi), bbox_inches="tight")

        # Save PDF for high-quality (vector) reports.
        pdf_path = png_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")

    if cfg.show:
        plt.show(block=bool(cfg.block))

    if cfg.close:
        plt.close(fig)


# =============================================================================
# Data container (sweep result)
# =============================================================================

@dataclass(frozen=True, slots=True)
class SpotSweepResult:
    """
    Container for a spot sweep.

    Arrays are 1D and aligned by index:
      spot_grid[i] corresponds to pv_total[i], greeks_total[...][i], etc.
    """
    spot0: float
    spot_grid: np.ndarray                # shape [N]
    pv_total: np.ndarray                 # shape [N]
    greeks_total: Dict[str, np.ndarray]  # each shape [N]
    pv_by_position: Dict[str, np.ndarray]  # position_id -> shape [N]


# =============================================================================
# Plotters
# =============================================================================

def plot_total_pv_vs_spot(*, sweep: SpotSweepResult, cfg: PlotOptions) -> None:
    """Plot total portfolio PV vs spot."""
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(1, 1, 1)

    # PV curve.
    ax.plot(sweep.spot_grid, sweep.pv_total, marker="o", linewidth=1.6)

    # Vertical marker for base spot.
    ax.axvline(float(sweep.spot0), linestyle="--", linewidth=1.2)

    ax.set_title("Total PV vs Spot")
    ax.set_xlabel("Spot")
    ax.set_ylabel("PV (domestic)")

    # Format PV axis nicely.
    ax.yaxis.set_major_formatter(FuncFormatter(_comma_formatter))
    _apply_plot_style(ax)

    fig.tight_layout()
    _save_or_show(fig, cfg=cfg, filename="01_total_pv_vs_spot.png")


def plot_total_pnl_vs_spot(*, sweep: SpotSweepResult, cfg: PlotOptions) -> None:
    """
    Plot total portfolio PnL vs spot (relative to PV at base spot).

    PnL definition
    --------------
    pnl(spot) = PV(spot) - PV(spot0)
    """
    # Interpolate PV at spot0 to be robust to grids where spot0 isn't exactly a grid point.
    base_pv = float(np.interp(float(sweep.spot0), sweep.spot_grid, sweep.pv_total))
    pnl = sweep.pv_total - base_pv

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(sweep.spot_grid, pnl, marker="o", linewidth=1.6)
    ax.axvline(float(sweep.spot0), linestyle="--", linewidth=1.2)

    ax.set_title("Total PnL vs Spot (relative to spot0)")
    ax.set_xlabel("Spot")
    ax.set_ylabel("PnL (domestic)")

    ax.yaxis.set_major_formatter(FuncFormatter(_comma_formatter))
    _apply_plot_style(ax)

    fig.tight_layout()
    _save_or_show(fig, cfg=cfg, filename="02_total_pnl_vs_spot.png")


def plot_total_greek_vs_spot(*, sweep: SpotSweepResult, greek_key: str, cfg: PlotOptions) -> None:
    """
    Plot a single total greek vs spot.

    Notes
    -----
    - If greek_key is missing from sweep.greeks_total, this is a no-op.
      This keeps examples robust if some pricers do not provide some greeks.
    """
    if greek_key not in sweep.greeks_total:
        return

    y = sweep.greeks_total[greek_key]

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(sweep.spot_grid, y, marker="o", linewidth=1.6)
    ax.axvline(float(sweep.spot0), linestyle="--", linewidth=1.2)

    ax.set_title(f"Total {greek_key} vs Spot")
    ax.set_xlabel("Spot")
    ax.set_ylabel(greek_key)

    # For large greeks, comma formatting is helpful.
    ax.yaxis.set_major_formatter(FuncFormatter(_comma_formatter))
    _apply_plot_style(ax)

    fig.tight_layout()
    _save_or_show(fig, cfg=cfg, filename=f"03_total_{greek_key}_vs_spot.png")


def plot_position_pv_vs_spot(
    *,
    sweep: SpotSweepResult,
    position_id: str,
    cfg: PlotOptions,
    title: Optional[str] = None,
) -> None:
    """
    Plot a single position PV vs spot.

    Notes
    -----
    - If the position_id was not captured, this is a no-op.
    """
    if position_id not in sweep.pv_by_position:
        return

    y = sweep.pv_by_position[position_id]

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(sweep.spot_grid, y, marker="o", linewidth=1.6)
    ax.axvline(float(sweep.spot0), linestyle="--", linewidth=1.2)

    ax.set_title(title or f"Position PV vs Spot: {position_id}")
    ax.set_xlabel("Spot")
    ax.set_ylabel("PV (domestic)")

    ax.yaxis.set_major_formatter(FuncFormatter(_comma_formatter))
    _apply_plot_style(ax)

    fig.tight_layout()
    _save_or_show(fig, cfg=cfg, filename=f"10_position_pv_{position_id}.png")


def plot_positions_pv_vs_spot(
    *,
    sweep: SpotSweepResult,
    position_ids: Iterable[str],
    cfg: PlotOptions,
) -> None:
    """
    Plot multiple position PV curves on one chart (use sparingly).

    Recommendation
    --------------
    Keep this to 2–4 positions to avoid spaghetti charts.
    """
    position_ids = [str(x) for x in position_ids if str(x) in sweep.pv_by_position]
    if not position_ids:
        return

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(1, 1, 1)

    for pid in position_ids:
        ax.plot(sweep.spot_grid, sweep.pv_by_position[pid], marker="o", linewidth=1.4, label=pid)

    ax.axvline(float(sweep.spot0), linestyle="--", linewidth=1.2)

    ax.set_title("Selected Positions PV vs Spot")
    ax.set_xlabel("Spot")
    ax.set_ylabel("PV (domestic)")
    ax.legend(loc="best", fontsize=9)

    ax.yaxis.set_major_formatter(FuncFormatter(_comma_formatter))
    _apply_plot_style(ax)

    fig.tight_layout()
    _save_or_show(fig, cfg=cfg, filename="11_positions_pv_vs_spot.png")
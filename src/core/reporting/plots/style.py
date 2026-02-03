"""
Publication-quality report styling for risk and analytics plots.

Provides consistent grid, spines, tick size, and optional figure/rc defaults
used by vol surfaces, Greeks heatmaps, scenario plots, and portfolio plots.
No global rcParams changes; apply explicitly per axes/figure.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib import rcParams


def apply_report_style(ax: plt.Axes) -> None:
    """
    Apply a clean, presentation-friendly style to the axes.

    - Light grid for readability.
    - Slightly smaller tick labels.
    - Remove top/right spines to reduce clutter.

    Use this for risk reports, vol surfaces, Greeks heatmaps, and scenario plots.
    """
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.tick_params(axis="both", labelsize=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def get_report_figsize() -> Tuple[float, float]:
    """Default figure size for report-quality plots (width, height in inches)."""
    return (9.0, 5.0)


@contextmanager
def report_rc():
    """
    Context manager for report-style matplotlib defaults.

    Use when creating multiple figures that should share the same base style.
    Restores previous rcParams on exit.
    """
    backup = dict(rcParams)
    try:
        rcParams["font.size"] = 10
        rcParams["axes.titlesize"] = 11
        rcParams["axes.labelsize"] = 10
        rcParams["xtick.labelsize"] = 9
        rcParams["ytick.labelsize"] = 9
        yield
    finally:
        rcParams.update(backup)


__all__ = ["apply_report_style", "get_report_figsize", "report_rc"]

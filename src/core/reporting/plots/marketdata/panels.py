from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Optional

import numpy as np
import matplotlib.pyplot as plt

from src.marketdata.core.panel import Panel


def plot_panel_scalar_timeseries(
    panel: Panel,
    dates: Sequence[str],
    *,
    scenario_idx: int = 0,
    title: str = "Panel Scalar Time Series",
    ylabel: str = "value",
) -> plt.Figure:
    """
    Plot a scalar Panel (quote-like) through time.

    Expects panel to support scalar_at(time_idx, scenario_idx).
    """
    n_t = len(dates)
    xs = np.arange(n_t, dtype=float)
    ys = np.array([float(panel.scalar_at(time_idx=i, scenario_idx=int(scenario_idx))) for i in range(n_t)], dtype=float)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xs, ys)
    ax.set_title(title)
    ax.set_xlabel("time index")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    return fig
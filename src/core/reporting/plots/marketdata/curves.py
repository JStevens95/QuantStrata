from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from src.marketdata.core.interfaces import Curve


def plot_curve_df(curve: Curve, title: str = "Discount factor DF(t)", t_max: float = 10.0, n: int = 200) -> plt.Figure:
    ts = np.linspace(0.0, float(t_max), int(n), dtype=float)
    dfs = np.array([curve.df(float(t)) for t in ts], dtype=float)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ts, dfs)
    ax.set_title(title)
    ax.set_xlabel("t (years)")
    ax.set_ylabel("DF(t)")
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_curve_zero_rate(curve: Curve, title: str = "Zero rate r(t)", t_max: float = 10.0, n: int = 200) -> plt.Figure:
    ts = np.linspace(1e-6, float(t_max), int(n), dtype=float)  # avoid t=0 edge
    rs = np.array([curve.zero_rate(float(t)) for t in ts], dtype=float)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ts, rs)
    ax.set_title(title)
    ax.set_xlabel("t (years)")
    ax.set_ylabel("r(t) (cont. comp)")
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_curve_forward_rate(
    curve: Curve,
    *,
    t1: float = 0.5,
    t2_max: float = 10.0,
    n: int = 200,
    title: Optional[str] = None,
) -> Figure:
    t2 = np.linspace(float(t1) + 1e-6, float(t2_max), int(n))
    f = np.array([curve.forward_rate(float(t1), float(tt)) for tt in t2], dtype=float)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t2, f)
    ax.set_xlabel("t2 (years)")
    ax.set_ylabel(f"f({t1}, t2) (cc)")
    ax.set_title(title or f"Forward rate f({t1}, t)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
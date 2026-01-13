from __future__ import annotations

from typing import Mapping, Tuple

import numpy as np
import matplotlib.pyplot as plt

from src.marketdata.scenarios.interfaces import MarketView
from src.marketdata.core.ids import MarketId


def plot_spot_comparison(
    base: MarketView,
    shocked: Mapping[str, MarketView],
    spot_id: MarketId,
) -> Tuple[plt.Figure, plt.Axes]:
    names = ["BASE"] + list(shocked.keys())
    vals = [float(base.quote(spot_id))] + [float(m.quote(spot_id)) for m in shocked.values()]

    fig, ax = plt.subplots()
    ax.bar(np.arange(len(names)), vals)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_title("Spot comparison")
    ax.set_ylabel("Spot")
    ax.grid(True, axis="y")
    return fig, ax


def plot_curve_df_comparison(
    base: MarketView,
    shocked: Mapping[str, MarketView],
    curve_id: MarketId,
    times: np.ndarray,
) -> Tuple[plt.Figure, plt.Axes]:
    t = np.asarray(times, dtype=float)

    fig, ax = plt.subplots()

    c0 = base.curve(curve_id)
    ax.plot(t, [float(c0.df(float(x))) for x in t], label="BASE")

    for name, mv in shocked.items():
        c = mv.curve(curve_id)
        ax.plot(t, [float(c.df(float(x))) for x in t], label=name)

    ax.set_title("DF(t) comparison")
    ax.set_xlabel("t (years)")
    ax.set_ylabel("DF(t)")
    ax.grid(True)
    ax.legend()
    return fig, ax


def plot_vol_comparison(
    base: MarketView,
    shocked: Mapping[str, MarketView],
    vol_id: MarketId,
    expiry: float,
    strikes: np.ndarray,
) -> Tuple[plt.Figure, plt.Axes]:
    k = np.asarray(strikes, dtype=float)
    t = float(expiry)

    fig, ax = plt.subplots()

    s0 = base.vol_surface(vol_id)
    ax.plot(k, [float(s0.vol(t, float(kk))) for kk in k], label="BASE")

    for name, mv in shocked.items():
        s = mv.vol_surface(vol_id)
        ax.plot(k, [float(s.vol(t, float(kk))) for kk in k], label=name)

    ax.set_title(f"Vol comparison @ T={t:g}")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied Vol")
    ax.grid(True)
    ax.legend()
    return fig, ax
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True, slots=True)
class FanSpec:
    """
    Configuration for a fan chart (quantile envelope).
    """
    q_low: float = 0.05
    q_mid: float = 0.50
    q_high: float = 0.95
    max_scenario_lines: int = 10  # draw a few sample scenario paths for intuition


def plot_spot_fan_chart(
    *,
    dates: Sequence[str],
    spot_paths: np.ndarray,
    title: str,
    fan: FanSpec = FanSpec(),
) -> plt.Figure:
    """
    Plot a spot fan chart from a panel shaped [T,S].

    What this shows (front-office useful)
    -------------------------------------
    - Median path (q_mid)
    - Uncertainty envelope (q_low..q_high)
    - A few scenario lines for texture
    """
    # Coerce to float arrays and validate shape explicitly (examples should be defensive).
    x = np.asarray(list(dates))
    y = np.asarray(spot_paths, dtype=float)

    if y.ndim != 2:
        raise ValueError(f"spot_paths must be 2D [T,S], got shape={y.shape}.")
    if y.shape[0] != len(x):
        raise ValueError(f"spot_paths T dimension must match len(dates). Got T={y.shape[0]} vs {len(x)}.")

    # Compute quantiles across scenarios at each time.
    ql = np.quantile(y, float(fan.q_low), axis=1)
    qm = np.quantile(y, float(fan.q_mid), axis=1)
    qh = np.quantile(y, float(fan.q_high), axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Fan band.
    ax.fill_between(np.arange(len(x)), ql, qh, alpha=0.25, label=f"q{fan.q_low:.0%}..q{fan.q_high:.0%}")
    # Median line.
    ax.plot(np.arange(len(x)), qm, linewidth=2.0, label=f"median (q{fan.q_mid:.0%})")

    # Optional: draw a handful of scenario lines (deterministically choose the first ones).
    s_count = int(y.shape[1])
    n_lines = int(min(max(0, fan.max_scenario_lines), s_count))
    for s in range(n_lines):
        ax.plot(np.arange(len(x)), y[:, s], linewidth=0.8, alpha=0.35)

    # X ticks: avoid clutter; show ~10 labels.
    n = len(x)
    step = max(1, n // 10)
    tick_idx = np.arange(0, n, step, dtype=int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([x[i] for i in tick_idx], rotation=30, ha="right")

    ax.set_title(title)
    ax.set_ylabel("Spot")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_log_return_timeseries(
    *,
    dates: Sequence[str],
    spot_paths: np.ndarray,
    title: str,
) -> plt.Figure:
    """
    Plot median log-returns over time from spot paths [T,S].

    Why median?
    ----------
    In scenario datasets, the median gives a robust “typical” path of returns.
    """
    x = np.asarray(list(dates))
    y = np.asarray(spot_paths, dtype=float)

    if y.ndim != 2:
        raise ValueError(f"spot_paths must be 2D [T,S], got shape={y.shape}.")
    if y.shape[0] != len(x):
        raise ValueError(f"spot_paths T dimension must match len(dates). Got T={y.shape[0]} vs {len(x)}.")

    # Compute log returns per scenario, then take median across scenarios.
    # r_t = log(S_t / S_{t-1})
    logy = np.log(np.maximum(y, 1e-12))
    r = logy[1:, :] - logy[:-1, :]
    r_med = np.median(r, axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(np.arange(len(r_med)), r_med, linewidth=2.0)

    # X ticks aligned to dates[1:].
    n = len(x) - 1
    step = max(1, n // 10)
    tick_idx = np.arange(0, n, step, dtype=int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([x[i + 1] for i in tick_idx], rotation=30, ha="right")

    ax.set_title(title)
    ax.set_ylabel("log return")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_return_correlation_heatmap(
    *,
    returns_by_label: Mapping[str, np.ndarray],
    title: str = "Return correlation",
) -> plt.Figure:
    """
    Plot a correlation heatmap from a mapping label -> returns vector.

    Input contract
    --------------
    returns_by_label[label] must be 1D array-like (any length >= 2).
    """
    labels = list(returns_by_label.keys())
    if not labels:
        raise ValueError("returns_by_label must be non-empty.")

    # Stack into a 2D array [N_assets, N_obs].
    series = []
    for k in labels:
        v = np.asarray(returns_by_label[k], dtype=float).reshape(-1)
        if v.size < 2:
            raise ValueError(f"Need at least 2 observations for correlation: label={k!r}, size={v.size}")
        series.append(v)

    # Align by trimming all series to the same length (examples should be explicit).
    min_len = min(int(v.size) for v in series)
    X = np.vstack([v[-min_len:] for v in series])

    # Corr across assets.
    C = np.corrcoef(X)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    im = ax.imshow(C, aspect="auto", origin="lower")
    ax.set_title(title)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)

    fig.colorbar(im, ax=ax, shrink=0.85, label="corr")
    fig.tight_layout()
    return fig


def flatten_log_returns_all_scenarios(spot_paths: np.ndarray) -> np.ndarray:
    """
    Convert spot paths [T,S] into a single long 1D vector of log-returns.

    This is useful for:
    - correlation across instruments in scenario datasets
    - simple distribution diagnostics

    Returns
    -------
    np.ndarray
        1D vector of length (T-1)*S
    """
    y = np.asarray(spot_paths, dtype=float)
    if y.ndim != 2:
        raise ValueError(f"spot_paths must be 2D [T,S], got shape={y.shape}.")

    logy = np.log(np.maximum(y, 1e-12))
    r = logy[1:, :] - logy[:-1, :]
    return r.reshape(-1)
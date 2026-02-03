"""
Greeks surface plotter: 2D heatmap of a greek (e.g. delta, vega) vs expiry and strike.

Data source is left to the caller: e.g. FD pricer surface, or a grid built from
SensitivitiesReport when SensitivityKey has expiry/strike.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from src.core.reporting.plots.style import apply_report_style, get_report_figsize
from src.risk.sensitivities.result import SensitivitiesReport


def plot_greeks_surface(
    expiries: np.ndarray,
    strikes: np.ndarray,
    greek_matrix_2d: np.ndarray,
    *,
    greek_name: str = "delta",
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a single greek as a 2D heatmap over (expiry, strike).

    Parameters
    ----------
    expiries : np.ndarray
        1D array of expiries (T).
    strikes : np.ndarray
        1D array of strikes (K).
    greek_matrix_2d : np.ndarray
        Shape (len(expiries), len(strikes)); greek_matrix_2d[i, j] = greek at (expiries[i], strikes[j]).
    greek_name : str
        Label for the greek (e.g. "delta", "vega").
    title : str, optional
        Figure title.

    Returns
    -------
    plt.Figure
    """
    expiries = np.asarray(expiries, dtype=float).reshape(-1)
    strikes = np.asarray(strikes, dtype=float).reshape(-1)
    z = np.asarray(greek_matrix_2d, dtype=float)
    if z.shape != (expiries.size, strikes.size):
        raise ValueError(
            f"greek_matrix_2d shape {z.shape} must be (len(expiries)={expiries.size}, len(strikes)={strikes.size})"
        )

    fig = plt.figure(figsize=get_report_figsize())
    ax = fig.add_subplot(111)
    im = ax.imshow(
        z,
        aspect="auto",
        origin="lower",
        extent=[float(strikes[0]), float(strikes[-1]), float(expiries[0]), float(expiries[-1])],
    )
    ax.set_title(title or f"{greek_name} surface")
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Expiry T")
    fig.colorbar(im, ax=ax, label=greek_name)
    apply_report_style(ax)
    fig.tight_layout()
    return fig


def greek_grid_from_sensitivities(
    report: SensitivitiesReport,
    greek_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (expiries, strikes, greek_matrix_2d) from a SensitivitiesReport when
    SensitivityKey has expiry and strike set.

    Rows with matching greek_name and non-None expiry/strike are binned into a grid.
    Missing cells are filled with 0.0. If no such rows exist, returns empty 1D arrays
    and a 0x0 matrix.

    Returns
    -------
    expiries : np.ndarray
        Sorted unique expiries.
    strikes : np.ndarray
        Sorted unique strikes.
    greek_matrix_2d : np.ndarray
        Shape (len(expiries), len(strikes)).
    """
    exp_set: set = set()
    strike_set: set = set()
    value_by_et_k: dict = {}
    for row in report.rows:
        if str(row.key.greek) != greek_name:
            continue
        e, k = row.key.expiry, row.key.strike
        if e is None or k is None:
            continue
        exp_set.add(float(e))
        strike_set.add(float(k))
        value_by_et_k[(float(e), float(k))] = float(row.value)
    expiries = np.array(sorted(exp_set), dtype=float)
    strikes = np.array(sorted(strike_set), dtype=float)
    if expiries.size == 0 or strikes.size == 0:
        return expiries, strikes, np.empty((0, 0), dtype=float)
    z = np.zeros((expiries.size, strikes.size), dtype=float)
    for i, e in enumerate(expiries):
        for j, k in enumerate(strikes):
            z[i, j] = value_by_et_k.get((e, k), 0.0)
    return expiries, strikes, z


__all__ = ["greek_grid_from_sensitivities", "plot_greeks_surface"]

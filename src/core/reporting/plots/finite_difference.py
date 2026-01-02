from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union


@dataclass(frozen=True, slots=True)
class PlotPaths:
    """Convenience container for plot outputs written by the helpers below."""
    price_curve: Path
    error_curve: Path
    delta_profile: Path
    gamma_profile: Path
    convergence: Path
    surface_heatmap: Optional[Path] = None


PlotReturn = Union[Path, plt.Figure]


def _ensure_dir(dir_path: Path) -> Path:
    """Create directory if missing and return it."""
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def _finalize_plot(fig: plt.Figure, out_path: Optional[Path]) -> PlotReturn:
    """
    If out_path is provided: save figure and close it, returning Path.
    If out_path is None: return the Figure (caller can display or save).
    """
    if out_path is None:
        return fig

    out_path = Path(out_path)
    _ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_price_curve_fd_vs_reference(
    *,
    out_path: Optional[Path] = None,
    spot_grid: np.ndarray,
    fd_values_per_unit: np.ndarray,
    ref_values_per_unit: np.ndarray,
    spot0: float,
    title: str = "V(S,0): FD vs reference",
) -> PlotReturn:
    """
    Plot the FD price curve V(S,0) against a reference curve (e.g., BSM),
    both expressed *per unit notional*.

    If out_path is None, returns a matplotlib Figure. Otherwise saves and returns Path.
    """
    fig = plt.figure()
    ax = fig.gca()

    ax.plot(spot_grid, fd_values_per_unit, label="FD")
    ax.plot(spot_grid, ref_values_per_unit, label="Reference")
    ax.axvline(float(spot0), label="S0")

    ax.set_title(title)
    ax.set_xlabel("Spot S")
    ax.set_ylabel("PV per unit notional")
    ax.legend()

    return _finalize_plot(fig, out_path)


def plot_error_curve(
    *,
    out_path: Optional[Path] = None,
    spot_grid: np.ndarray,
    fd_values_per_unit: np.ndarray,
    ref_values_per_unit: np.ndarray,
    spot0: float,
    title: str = "Error curve: FD - reference",
) -> PlotReturn:
    """
    Plot the pricing error curve: V_FD(S,0) - V_ref(S,0) per unit notional.

    If out_path is None, returns a matplotlib Figure. Otherwise saves and returns Path.
    """
    err = np.asarray(fd_values_per_unit, dtype=np.float64) - np.asarray(ref_values_per_unit, dtype=np.float64)

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(spot_grid, err, label="FD - Reference")
    ax.axvline(float(spot0), label="S0")

    ax.set_title(title)
    ax.set_xlabel("Spot S")
    ax.set_ylabel("PV error per unit")
    ax.legend()

    return _finalize_plot(fig, out_path)


def plot_delta_profile(
    *,
    out_path: Optional[Path] = None,
    spot_grid: np.ndarray,
    fd_values_per_unit: np.ndarray,
    ref_delta_per_unit: np.ndarray,
    spot0: float,
    title: str = "Delta profile: FD (surface) vs reference",
) -> PlotReturn:
    """
    Plot delta(S) from the FD surface vs a reference delta curve.

    Implementation detail
    ---------------------
    We compute FD delta by differentiating the FD price curve w.r.t. S using numpy.gradient.
    This is a diagnostic plot (not the production greek).

    If out_path is None, returns a matplotlib Figure. Otherwise saves and returns Path.
    """
    fd_delta = np.gradient(
        np.asarray(fd_values_per_unit, dtype=np.float64),
        np.asarray(spot_grid, dtype=np.float64),
    )

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(spot_grid, fd_delta, label="FD delta (from surface)")
    ax.plot(spot_grid, ref_delta_per_unit, label="Reference delta")
    ax.axvline(float(spot0), label="S0")

    ax.set_title(title)
    ax.set_xlabel("Spot S")
    ax.set_ylabel("Delta per unit")
    ax.legend()

    return _finalize_plot(fig, out_path)


def plot_gamma_profile(
    *,
    out_path: Optional[Path] = None,
    spot_grid: np.ndarray,
    fd_values_per_unit: np.ndarray,
    ref_gamma_per_unit: np.ndarray,
    spot0: float,
    title: str = "Gamma profile: FD (surface) vs reference",
) -> PlotReturn:
    """
    Plot gamma(S) from the FD surface vs a reference gamma curve.

    Implementation detail
    ---------------------
    We compute FD gamma as the second derivative of the FD price curve w.r.t. S.

    If out_path is None, returns a matplotlib Figure. Otherwise saves and returns Path.
    """
    spot_grid = np.asarray(spot_grid, dtype=np.float64)
    v = np.asarray(fd_values_per_unit, dtype=np.float64)

    fd_delta = np.gradient(v, spot_grid)
    fd_gamma = np.gradient(fd_delta, spot_grid)

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(spot_grid, fd_gamma, label="FD gamma (from surface)")
    ax.plot(spot_grid, ref_gamma_per_unit, label="Reference gamma")
    ax.axvline(float(spot0), label="S0")

    ax.set_title(title)
    ax.set_xlabel("Spot S")
    ax.set_ylabel("Gamma per unit")
    ax.legend()

    return _finalize_plot(fig, out_path)


def plot_convergence_pv_error(
    *,
    out_path: Optional[Path] = None,
    n_space_list: Sequence[int],
    pv_error_list: Sequence[float],
    title: str = "Convergence: PV_FD(S0) - PV_ref vs n_space",
) -> PlotReturn:
    """
    Plot convergence of PV error at S0 versus the spatial resolution n_space.

    If out_path is None, returns a matplotlib Figure. Otherwise saves and returns Path.
    """
    n_space_arr = np.asarray(n_space_list, dtype=np.float64)
    err_arr = np.asarray(pv_error_list, dtype=np.float64)

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(n_space_arr, err_arr, marker="o", label="PV error")
    ax.set_title(title)
    ax.set_xlabel("n_space")
    ax.set_ylabel("PV error (domestic)")
    ax.legend()

    return _finalize_plot(fig, out_path)


def plot_surface_heatmap(
    *,
    out_path: Optional[Path] = None,
    spot_grid: np.ndarray,
    time_grid: np.ndarray,
    surface: np.ndarray,
    title: str = "FD surface V(t,S) (per unit)",
) -> PlotReturn:
    """
    Plot a heatmap of the full FD solution surface.

    Parameters
    ----------
    surface:
        Array shaped (n_t, n_x), aligned with (time_grid, spot_grid).

    If out_path is None, returns a matplotlib Figure. Otherwise saves and returns Path.
    """
    spot_grid = np.asarray(spot_grid, dtype=np.float64)
    time_grid = np.asarray(time_grid, dtype=np.float64)
    surface = np.asarray(surface, dtype=np.float64)

    fig = plt.figure()
    ax = fig.gca()

    im = ax.imshow(
        surface,
        aspect="auto",
        origin="lower",
        extent=[float(spot_grid[0]), float(spot_grid[-1]), float(time_grid[0]), float(time_grid[-1])],
    )
    fig.colorbar(im, ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Spot S")
    ax.set_ylabel("Time t")

    return _finalize_plot(fig, out_path)
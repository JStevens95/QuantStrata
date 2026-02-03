from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Optional, Tuple

from src.core.reporting.plots.style import apply_report_style
from src.marketdata.core.interfaces import VolSurface


def _resolve_grid_axes(surface: VolSurface) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resolve (expiries, strikes) for plotting.

    Priority:
    1) surface.expiries / surface.strikes (GridVolSurface)
    2) surface.base_surface.expiries / strikes (scenario wrapper over grid)
    3) raise with a clear message
    """
    if hasattr(surface, "expiries") and hasattr(surface, "strikes"):
        expiries = np.asarray(getattr(surface, "expiries"), dtype=float).reshape(-1)
        strikes = np.asarray(getattr(surface, "strikes"), dtype=float).reshape(-1)
        return expiries, strikes

    # Common for wrapper shocks: surface.base_surface is the real grid surface
    base = getattr(surface, "base_surface", None)
    if base is not None and hasattr(base, "expiries") and hasattr(base, "strikes"):
        expiries = np.asarray(getattr(base, "expiries"), dtype=float).reshape(-1)
        strikes = np.asarray(getattr(base, "strikes"), dtype=float).reshape(-1)
        return expiries, strikes

    raise AttributeError(
        "Cannot infer plotting grid. Provide a GridVolSurface-like object "
        "with `.expiries` and `.strikes`, or ensure your wrapper exposes "
        "`base_surface.expiries/strikes`."
    )


def _sample_implied_vol(surface: VolSurface, expiry: float, strike: float) -> float:
    """
    Robust sampler for implied vol. Supports either:
      - surface.implied_vol(T,K)
      - surface.vol(T,K)  (alias)
    """
    if hasattr(surface, "implied_vol"):
        return float(surface.implied_vol(float(expiry), float(strike)))
    if hasattr(surface, "vol"):
        return float(surface.vol(float(expiry), float(strike)))
    raise AttributeError("VolSurface must implement implied_vol(T,K) or vol(T,K).")


def _sample_vol_grid(surface: VolSurface, expiries: np.ndarray, strikes: np.ndarray) -> np.ndarray:
    """
    Build Z[T_i, K_j] by sampling the surface callable API.

    This is critical for scenario wrappers: they change implied_vol(T,K) but not stored grids.
    """
    expiries = np.asarray(expiries, dtype=float).reshape(-1)
    strikes = np.asarray(strikes, dtype=float).reshape(-1)

    z = np.empty((expiries.size, strikes.size), dtype=float)
    for i, t in enumerate(expiries):
        for j, k in enumerate(strikes):
            z[i, j] = _sample_implied_vol(surface, float(t), float(k))
    return z


def plot_vol_surface_heatmap(surface: VolSurface, title: str = "Vol surface") -> plt.Figure:
    """
    Heatmap of σ(T,K) built by sampling implied_vol(T,K) on the surface's own grid.
    """
    expiries, strikes = _resolve_grid_axes(surface)
    vols = _sample_vol_grid(surface, expiries, strikes)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(
        vols,
        aspect="auto",
        origin="lower",
        extent=[float(strikes[0]), float(strikes[-1]), float(expiries[0]), float(expiries[-1])],
    )
    ax.set_title(title)
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Expiry T")
    fig.colorbar(im, ax=ax, label="Implied vol")
    apply_report_style(ax)
    fig.tight_layout()
    return fig


def plot_vol_smile_slices(surface: VolSurface, title: str = "Smile slices") -> plt.Figure:
    """
    Plot σ(T,K) vs K for each expiry T on the surface's grid.

    Important:
    - This samples implied_vol(T,K) so scenario wrappers plot correctly.
    """
    expiries, strikes = _resolve_grid_axes(surface)
    vols = _sample_vol_grid(surface, expiries, strikes)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, t in enumerate(expiries.tolist()):
        ax.plot(strikes, vols[i, :], label=f"T={float(t):g}")

    ax.set_title(title)
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Implied vol")
    ax.grid(True)
    ax.legend()
    apply_report_style(ax)
    fig.tight_layout()
    return fig


def plot_vol_surface(
    surface: VolSurface,
    *,
    expiries: Optional[np.ndarray] = None,
    strikes: Optional[np.ndarray] = None,
    title: str = "Implied Vol Surface (3D)",
    n_expiries: int = 25,
    n_strikes: int = 41,
) -> plt.Figure:
    """
    Plot implied vol as a 3D surface σ(T,K).

    - If the surface has a grid (expiries/strikes), we use it by default.
    - Otherwise we fall back to a demo grid unless the caller provides grids.
    """
    if expiries is None and strikes is None:
        try:
            expiries, strikes = _resolve_grid_axes(surface)
        except Exception:
            expiries = None
            strikes = None

    if expiries is None:
        expiries = np.linspace(0.1, 2.0, int(n_expiries), dtype=float)
    if strikes is None:
        strikes = np.linspace(0.8, 1.2, int(n_strikes), dtype=float)

    expiries = np.asarray(expiries, dtype=float).reshape(-1)
    strikes = np.asarray(strikes, dtype=float).reshape(-1)

    Z = _sample_vol_grid(surface, expiries, strikes)
    T, K = np.meshgrid(expiries, strikes, indexing="ij")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    norm = mpl.colors.Normalize(vmin=float(np.min(Z)), vmax=float(np.max(Z)))

    surf = ax.plot_surface(
        T, K, Z,
        cmap="viridis",
        norm=norm,
        linewidth=0.0,
        antialiased=True,
    )

    cbar = fig.colorbar(surf, ax=ax, shrink=0.7, pad=0.12)
    cbar.set_label("Implied vol σ")

    ax.set_title(title)
    ax.set_xlabel("Expiry T")
    ax.set_ylabel("Strike K")
    ax.set_zlabel("Implied vol σ")
    fig.tight_layout()
    return fig
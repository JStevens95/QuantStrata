from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Optional

from src.marketdata.core.interfaces import VolSurface


def plot_vol_surface_heatmap(surface: VolSurface, title: str = "Vol surface") -> plt.Figure:
    # assumes GridVolSurface-like attributes exist (expiries/strikes/implied_vols)
    expiries = np.asarray(getattr(surface, "expiries"), dtype=float).reshape(-1)
    strikes = np.asarray(getattr(surface, "strikes"), dtype=float).reshape(-1)
    vols = np.asarray(getattr(surface, "implied_vols"), dtype=float)

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
    fig.tight_layout()
    return fig


def plot_vol_smile_slices(surface: VolSurface, title: str = "Smile slices") -> plt.Figure:
    expiries = np.asarray(getattr(surface, "expiries"), dtype=float).reshape(-1)
    strikes = np.asarray(getattr(surface, "strikes"), dtype=float).reshape(-1)
    vols = np.asarray(getattr(surface, "implied_vols"), dtype=float)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, t in enumerate(expiries.tolist()):
        ax.plot(strikes, vols[i, :], label=f"T={float(t):g}")

    ax.set_title(title)
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Implied vol")
    ax.grid(True)
    ax.legend()
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

    Notes
    -----
    - If `surface` is a GridVolSurface-like object with `.expiries` and `.strikes`,
      those are used by default.
    - Otherwise, you must pass `expiries` and `strikes` explicitly (or it will
      fall back to a generic demo grid).
    """
    # Prefer using the surface's own grid if present
    if expiries is None and hasattr(surface, "expiries"):
        expiries = np.asarray(getattr(surface, "expiries"), dtype=float).reshape(-1)
    if strikes is None and hasattr(surface, "strikes"):
        strikes = np.asarray(getattr(surface, "strikes"), dtype=float).reshape(-1)

    # Fallback demo grids (only if caller didn't provide anything)
    if expiries is None:
        expiries = np.linspace(0.1, 2.0, int(n_expiries), dtype=float)
    if strikes is None:
        # sensible default range; caller should provide in real use
        strikes = np.linspace(0.8, 1.2, int(n_strikes), dtype=float)

    expiries = np.asarray(expiries, dtype=float).reshape(-1)
    strikes = np.asarray(strikes, dtype=float).reshape(-1)

    # Build Z grid by sampling implied_vol(T,K)
    Z = np.empty((expiries.size, strikes.size), dtype=float)
    for i, t in enumerate(expiries):
        for j, k in enumerate(strikes):
            Z[i, j] = float(surface.implied_vol(float(t), float(k)))

    # Mesh for plotting
    T, K = np.meshgrid(expiries, strikes, indexing="ij")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    norm = mpl.colors.Normalize(vmin=float(np.min(Z)), vmax=float(np.max(Z)))

    surf = ax.plot_surface(
        T, K, Z,
        cmap="viridis",  # or "plasma", "inferno", "magma", etc.
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
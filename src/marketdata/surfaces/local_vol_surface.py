"""
Local Volatility Surface Implementation.

This module provides the `LocalVolSurface` class for representing and interpolating
local volatility surfaces σ(S, t) as used in Dupire's local volatility model.

Mathematical Framework
----------------------
In the local volatility model, the spot price follows:

    dS_t = (r - q) S_t dt + σ_LV(S_t, t) S_t dW_t

where σ_LV(S, t) is a deterministic function of spot and time, called the
**local volatility**. Unlike Black-Scholes (constant vol) or implied vol
(function of K, T), local vol is a function of the current spot level.

Key Properties
--------------
- σ_LV(S, t) can be uniquely determined from market implied vols (Dupire's formula)
- Local vol model is complete and arbitrage-free by construction
- Provides exact fit to vanilla option prices across all strikes and expiries
- Used extensively in FX and equity exotic pricing

Relationship to Implied Volatility
----------------------------------
Given a continuum of European call prices C(K, T), the local volatility is:

    σ_LV²(K, T) = [∂C/∂T + (r-q)K ∂C/∂K + qC] / [½K² ∂²C/∂K²]

Or equivalently in terms of implied vol σ_BS(K, T):

    σ_LV²(K, T) = 2∂σ_BS/∂T / [K²(∂²C/∂K²)/C]

Note: At the spot level S=K, local vol equals "instantaneous" vol.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional

# Extrapolation mode (consistent with vol_surface.py).
ExtrapolationMode = Literal["flat", "error"]


@dataclass(frozen=True, slots=True)
class LocalVolSurface:
    """
    2D Local Volatility surface σ(S, t) defined on a grid with bilinear interpolation.

    The local volatility is the diffusion coefficient in Dupire's model:
        dS_t = μ S_t dt + σ(S_t, t) S_t dW_t

    Grid Definition
    ---------------
    times : np.ndarray
        1D strictly-increasing array of time points (year fractions, >= 0).
        Shape: (n_times,).
    spots : np.ndarray
        1D strictly-increasing array of spot levels (must be > 0).
        Shape: (n_spots,).
    local_vols : np.ndarray
        2D array of local volatility values.
        Shape: (n_times, n_spots).
        local_vols[i, j] = σ(spots[j], times[i]).

    Interpolation
    -------------
    - Bilinear interpolation within the grid.
    - Flat extrapolation outside grid boundaries (or error if extrapolation="error").

    Parameters
    ----------
    times : np.ndarray
        Time grid (year fractions).
    spots : np.ndarray
        Spot grid (positive values).
    local_vols : np.ndarray
        Local vol values on the grid.
    extrapolation : ExtrapolationMode
        "flat" for flat extrapolation, "error" to raise on out-of-bounds queries.
    surface_id : str or None
        Optional identifier for debugging/logging.

    Examples
    --------
    >>> import numpy as np
    >>> from src.marketdata.surfaces.local_vol import LocalVolSurface
    >>> times = np.array([0.0, 0.5, 1.0])
    >>> spots = np.array([80.0, 100.0, 120.0])
    >>> local_vols = np.array([
    ...     [0.22, 0.20, 0.18],  # t=0.0
    ...     [0.21, 0.19, 0.17],  # t=0.5
    ...     [0.20, 0.18, 0.16],  # t=1.0
    ... ])
    >>> surface = LocalVolSurface(times=times, spots=spots, local_vols=local_vols)
    >>> surface.local_vol(spot=100.0, time=0.5)
    0.19
    """

    # Grid axes.
    times: np.ndarray    # Shape (n_times,), time axis.
    spots: np.ndarray    # Shape (n_spots,), spot axis.
    local_vols: np.ndarray  # Shape (n_times, n_spots), local vol values.

    # Extrapolation mode.
    extrapolation: ExtrapolationMode = "flat"

    # Optional metadata.
    surface_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate and normalize grid arrays."""
        # Convert to numpy arrays.
        time_grid = np.asarray(self.times, dtype=float).reshape(-1)
        spot_grid = np.asarray(self.spots, dtype=float).reshape(-1)
        vol_grid = np.asarray(self.local_vols, dtype=float)

        # ---- Validate time axis ----
        if time_grid.size == 0:
            raise ValueError("LocalVolSurface.times must not be empty.")
        if np.any(~np.isfinite(time_grid)):
            raise ValueError("LocalVolSurface.times must be finite.")
        if np.any(time_grid < 0.0):
            raise ValueError("LocalVolSurface.times must be >= 0.")
        if time_grid.size > 1 and np.any(np.diff(time_grid) <= 0.0):
            raise ValueError("LocalVolSurface.times must be strictly increasing.")

        # ---- Validate spot axis ----
        if spot_grid.size == 0:
            raise ValueError("LocalVolSurface.spots must not be empty.")
        if np.any(~np.isfinite(spot_grid)):
            raise ValueError("LocalVolSurface.spots must be finite.")
        if np.any(spot_grid <= 0.0):
            raise ValueError("LocalVolSurface.spots must be > 0.")
        if spot_grid.size > 1 and np.any(np.diff(spot_grid) <= 0.0):
            raise ValueError("LocalVolSurface.spots must be strictly increasing.")

        # ---- Validate local vol grid ----
        if vol_grid.ndim != 2:
            raise ValueError(
                "LocalVolSurface.local_vols must be 2D array [n_times, n_spots]."
            )
        if vol_grid.shape != (time_grid.size, spot_grid.size):
            raise ValueError(
                f"LocalVolSurface.local_vols shape must be ({time_grid.size}, {spot_grid.size}); "
                f"got {vol_grid.shape}."
            )
        if np.any(~np.isfinite(vol_grid)):
            raise ValueError("LocalVolSurface.local_vols must be finite.")
        if np.any(vol_grid <= 0.0):
            raise ValueError("LocalVolSurface.local_vols must be > 0.")

        # ---- Validate extrapolation mode ----
        if self.extrapolation not in ("flat", "error"):
            raise ValueError("LocalVolSurface.extrapolation must be 'flat' or 'error'.")

        # Store normalized arrays.
        object.__setattr__(self, "times", time_grid)
        object.__setattr__(self, "spots", spot_grid)
        object.__setattr__(self, "local_vols", vol_grid)

    def local_vol(self, spot: float, time: float) -> float:
        """
        Interpolate local volatility at given (spot, time).

        Parameters
        ----------
        spot : float
            Spot price level (must be > 0).
        time : float
            Time point (year fraction, >= 0).

        Returns
        -------
        float
            Interpolated local volatility σ(spot, time).

        Raises
        ------
        ValueError
            If inputs are invalid or out of bounds (when extrapolation="error").
        """
        s = float(spot)
        t = float(time)

        # Validate inputs.
        if not np.isfinite(s):
            raise ValueError("spot must be finite.")
        if s <= 0.0:
            raise ValueError("spot must be > 0.")
        if not np.isfinite(t):
            raise ValueError("time must be finite.")
        if t < 0.0:
            raise ValueError("time must be >= 0.")

        # Check bounds if error mode.
        if self.extrapolation == "error":
            if t < self.times[0] or t > self.times[-1]:
                raise ValueError(
                    f"time={t} outside grid [{self.times[0]}, {self.times[-1]}]."
                )
            if s < self.spots[0] or s > self.spots[-1]:
                raise ValueError(
                    f"spot={s} outside grid [{self.spots[0]}, {self.spots[-1]}]."
                )

        # Bilinear interpolation with flat extrapolation.
        return float(_bilinear_interp_local_vol(
            time_grid=self.times,
            spot_grid=self.spots,
            vol_grid=self.local_vols,
            time_query=t,
            spot_query=s,
        ))

    def __call__(self, spot: float, time: float) -> float:
        """Alias for local_vol() - allows surface(spot, time) syntax."""
        return self.local_vol(spot, time)

    @property
    def time_range(self) -> tuple[float, float]:
        """Return (min_time, max_time) covered by the grid."""
        return (float(self.times[0]), float(self.times[-1]))

    @property
    def spot_range(self) -> tuple[float, float]:
        """Return (min_spot, max_spot) covered by the grid."""
        return (float(self.spots[0]), float(self.spots[-1]))

    @property
    def shape(self) -> tuple[int, int]:
        """Return (n_times, n_spots) shape of the grid."""
        return (len(self.times), len(self.spots))


@dataclass(frozen=True, slots=True)
class FlatLocalVolSurface:
    """
    Flat (constant) local volatility surface.

    This is equivalent to the Black-Scholes model where σ(S, t) = σ = const.

    Useful for:
    - Testing and debugging
    - Baseline comparisons
    - When vol smile is negligible

    Parameters
    ----------
    sigma : float
        Constant local volatility (must be > 0).

    Examples
    --------
    >>> from src.marketdata.surfaces.local_vol import FlatLocalVolSurface
    >>> surface = FlatLocalVolSurface(sigma=0.20)
    >>> surface.local_vol(spot=100.0, time=0.5)
    0.2
    """

    sigma: float

    def __post_init__(self) -> None:
        """Validate constant volatility."""
        if not np.isfinite(self.sigma):
            raise ValueError("FlatLocalVolSurface.sigma must be finite.")
        if self.sigma <= 0.0:
            raise ValueError("FlatLocalVolSurface.sigma must be > 0.")

    def local_vol(self, spot: float, time: float) -> float:
        """Return constant local volatility (ignores spot and time)."""
        # Validate inputs for consistency.
        s = float(spot)
        t = float(time)
        if not np.isfinite(s) or s <= 0.0:
            raise ValueError("spot must be finite and > 0.")
        if not np.isfinite(t) or t < 0.0:
            raise ValueError("time must be finite and >= 0.")
        return float(self.sigma)

    def __call__(self, spot: float, time: float) -> float:
        """Alias for local_vol()."""
        return self.local_vol(spot, time)


# =============================================================================
# Internal interpolation helper
# =============================================================================

def _bilinear_interp_local_vol(
    time_grid: np.ndarray,
    spot_grid: np.ndarray,
    vol_grid: np.ndarray,
    time_query: float,
    spot_query: float,
) -> float:
    """
    Bilinear interpolation with flat extrapolation for local vol surface.

    Parameters
    ----------
    time_grid : np.ndarray
        1D increasing time array, shape (n_times,).
    spot_grid : np.ndarray
        1D increasing spot array, shape (n_spots,).
    vol_grid : np.ndarray
        2D local vol array, shape (n_times, n_spots).
    time_query : float
        Query time.
    spot_query : float
        Query spot.

    Returns
    -------
    float
        Interpolated local vol value.
    """
    # Clamp to grid boundaries (flat extrapolation).
    t = float(np.clip(time_query, time_grid[0], time_grid[-1]))
    s = float(np.clip(spot_query, spot_grid[0], spot_grid[-1]))

    n_times = len(time_grid)
    n_spots = len(spot_grid)

    # Handle edge cases (single-point grids).
    if n_times == 1 and n_spots == 1:
        return float(vol_grid[0, 0])
    if n_times == 1:
        # Interpolate along spot axis only.
        return float(np.interp(s, spot_grid, vol_grid[0, :]))
    if n_spots == 1:
        # Interpolate along time axis only.
        return float(np.interp(t, time_grid, vol_grid[:, 0]))

    # Find bracketing indices for time.
    i_t = int(np.searchsorted(time_grid, t, side="right") - 1)
    i_t = int(np.clip(i_t, 0, n_times - 2))

    # Find bracketing indices for spot.
    i_s = int(np.searchsorted(spot_grid, s, side="right") - 1)
    i_s = int(np.clip(i_s, 0, n_spots - 2))

    # Extract bracket boundaries.
    t0 = float(time_grid[i_t])
    t1 = float(time_grid[i_t + 1])
    s0 = float(spot_grid[i_s])
    s1 = float(spot_grid[i_s + 1])

    # Compute interpolation weights.
    w_t = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
    w_s = 0.0 if s1 == s0 else (s - s0) / (s1 - s0)

    # Extract corner values.
    v00 = float(vol_grid[i_t, i_s])          # (t0, s0)
    v01 = float(vol_grid[i_t, i_s + 1])      # (t0, s1)
    v10 = float(vol_grid[i_t + 1, i_s])      # (t1, s0)
    v11 = float(vol_grid[i_t + 1, i_s + 1])  # (t1, s1)

    # Bilinear interpolation.
    # First interpolate along spot axis at both time points.
    v_t0 = (1.0 - w_s) * v00 + w_s * v01
    v_t1 = (1.0 - w_s) * v10 + w_s * v11

    # Then interpolate along time axis.
    return (1.0 - w_t) * v_t0 + w_t * v_t1

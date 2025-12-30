from __future__ import annotations

import numpy as np
from typing import Literal
from dataclasses import dataclass

# define extrapolation literals
ExtrapolationMode = Literal["flat", "error"]

@dataclass(frozen=True, slots=True)
class FlatVolSurface:
    """
    Flat implied volatility surface.

    This is the minimal and most stable surface for V1 because:
      - it requires only one parameter (a single implied vol),
      - it is easy to generate synthetically and to test,
      - it supports immediate option pricing (Black–Scholes / Garman–Kohlhagen style).

    Parameters
    ----------
    implied_vol:
        Constant implied volatility (e.g., 0.12 for 12% annualized).

    Notes
    -----
    - `expiry` is expressed in year fractions from as-of.
    - `strike` is passed for interface compatibility but is not used for a flat surface.
    - This object is intentionally pure and side-effect free.
    """

    implied_vol: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.implied_vol):
            raise ValueError("FlatVolSurface.implied_vol must be finite.")
        if self.implied_vol <= 0.0:
            raise ValueError("FlatVolSurface.implied_vol must be strictly positive.")

    def vol(self, expiry: float, strike: float) -> float:
        """
        Return the implied volatility for a given expiry and strike.

        Parameters
        ----------
        expiry:
            Time to expiry in year fractions (>= 0).
        strike:
            Strike (unused for flat surface; kept for interface consistency).

        Returns
        -------
        float
            Constant implied volatility.
        """
        # Expiry validation is helpful to catch upstream bugs early.
        if not np.isfinite(expiry):
            raise ValueError("expiry must be finite.")
        if expiry < 0.0:
            raise ValueError("expiry must be >= 0.")
        # strike is not used; we do not validate it here to keep the surface generic.
        return float(self.implied_vol)


@dataclass(frozen=True, slots=True)
class GridVolSurface:
    """
    Implied volatility surface defined on a 2D grid (expiry x strike) with bilinear interpolation.

    This is the natural complexity upgrade from FlatVolSurface:
      - still provides the same interface: vol(expiry, strike) -> float
      - supports term-structure + smile/skew in a simple, stable way

    Grid definition
    --------------
    expiries:
        1D strictly-increasing array of expiry times (year fractions, >= 0).
    strikes:
        1D strictly-increasing array of strikes (units depend on instrument convention).
    implied_vols:
        2D array of shape (len(expiries), len(strikes)) containing implied vols at grid points.

    Interpolation & extrapolation
    -----------------------------
    - Inside the grid: bilinear interpolation
    - Outside the grid:
        * extrapolation="flat": clamp to nearest boundary in each dimension (recommended for stability)
        * extrapolation="error": raise ValueError if outside grid

    Notes
    -----
    - This class is intentionally minimal: no arbitrage checks or smoothing.
    - Works for FX/EQ/IR as long as `strike` is used consistently by the pricer.
    """

    expiries: np.ndarray
    strikes: np.ndarray
    implied_vols: np.ndarray
    extrapolation: ExtrapolationMode = "flat"

    def __post_init__(self) -> None:
        expiry_grid = np.asarray(self.expiries, dtype=float).reshape(-1)
        strike_grid = np.asarray(self.strikes, dtype=float).reshape(-1)
        vol_grid = np.asarray(self.implied_vols, dtype=float)

        # ---- Validate expiry axis ----
        if expiry_grid.size == 0:
            raise ValueError("GridVolSurface.expiries must not be empty.")
        if np.any(~np.isfinite(expiry_grid)):
            raise ValueError("GridVolSurface.expiries must be finite.")
        if np.any(expiry_grid < 0.0):
            raise ValueError("GridVolSurface.expiries must be >= 0.")
        if np.any(np.diff(expiry_grid) <= 0.0):
            raise ValueError("GridVolSurface.expiries must be strictly increasing.")

        # ---- Validate strike axis ----
        if strike_grid.size == 0:
            raise ValueError("GridVolSurface.strikes must not be empty.")
        if np.any(~np.isfinite(strike_grid)):
            raise ValueError("GridVolSurface.strikes must be finite.")
        if np.any(np.diff(strike_grid) <= 0.0):
            raise ValueError("GridVolSurface.strikes must be strictly increasing.")

        # ---- Validate vol grid ----
        if vol_grid.ndim != 2:
            raise ValueError("GridVolSurface.implied_vols must be a 2D array [n_expiries, n_strikes].")
        if vol_grid.shape != (expiry_grid.size, strike_grid.size):
            raise ValueError(
                "GridVolSurface.implied_vols shape must be (len(expiries), len(strikes)); "
                f"got {vol_grid.shape} vs {(expiry_grid.size, strike_grid.size)}."
            )
        if np.any(~np.isfinite(vol_grid)):
            raise ValueError("GridVolSurface.implied_vols must be finite.")
        if np.any(vol_grid <= 0.0):
            raise ValueError("GridVolSurface.implied_vols must be strictly positive.")

        if self.extrapolation not in ("flat", "error"):
            raise ValueError("GridVolSurface.extrapolation must be one of {'flat','error'}.")

        # Store canonical arrays back on frozen dataclass.
        object.__setattr__(self, "expiries", expiry_grid)
        object.__setattr__(self, "strikes", strike_grid)
        object.__setattr__(self, "implied_vols", vol_grid)

    def vol(self, expiry: float, strike: float) -> float:
        """
        Return the implied volatility for a given expiry and strike.

        Parameters
        ----------
        expiry:
            Time to expiry in year fractions (>= 0).
        strike:
            Strike (units depend on instrument convention).

        Returns
        -------
        float
            Interpolated implied volatility.
        """
        t = float(expiry)
        k = float(strike)

        if not np.isfinite(t):
            raise ValueError("expiry must be finite.")
        if t < 0.0:
            raise ValueError("expiry must be >= 0.")
        if not np.isfinite(k):
            raise ValueError("strike must be finite.")

        # Enforce extrapolation policy.
        if self.extrapolation == "error":
            if t < self.expiries[0] or t > self.expiries[-1]:
                raise ValueError(f"expiry={t} outside grid [{self.expiries[0]}, {self.expiries[-1]}].")
            if k < self.strikes[0] or k > self.strikes[-1]:
                raise ValueError(f"strike={k} outside grid [{self.strikes[0]}, {self.strikes[-1]}].")

        return float(_bilinear_interp_flat(
            x=self.expiries,
            y=self.strikes,
            z=self.implied_vols,
            xq=t,
            yq=k,
        ))


def _bilinear_interp_flat(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    xq: float,
    yq: float,
) -> float:
    """
    Bilinear interpolation with flat extrapolation on both dimensions.

    Parameters
    ----------
    x:
        1D increasing array for the first axis (e.g., expiries), shape (nx,).
    y:
        1D increasing array for the second axis (e.g., strikes), shape (ny,).
    z:
        2D grid values, shape (nx, ny).
    xq:
        Query x (expiry).
    yq:
        Query y (strike).

    Returns
    -------
    float
        Interpolated value.
    """
    # Clamp query points for flat extrapolation.
    xqc = float(np.clip(xq, x[0], x[-1]))
    yqc = float(np.clip(yq, y[0], y[-1]))

    # Degenerate axis handling for robustness.
    if x.size == 1 and y.size == 1:
        return float(z[0, 0])
    if x.size == 1:
        return float(np.interp(yqc, y, z[0, :]))
    if y.size == 1:
        return float(np.interp(xqc, x, z[:, 0]))

    # Locate surrounding x indices: i such that x[i] <= xqc <= x[i+1].
    ix = int(np.searchsorted(x, xqc, side="right") - 1)
    ix = int(np.clip(ix, 0, x.size - 2))

    # Locate surrounding y indices: j such that y[j] <= yqc <= y[j+1].
    iy = int(np.searchsorted(y, yqc, side="right") - 1)
    iy = int(np.clip(iy, 0, y.size - 2))

    x0 = float(x[ix])
    x1 = float(x[ix + 1])
    y0 = float(y[iy])
    y1 = float(y[iy + 1])

    # Avoid divide-by-zero (should not happen due to strict increasing validation).
    tx = 0.0 if x1 == x0 else (xqc - x0) / (x1 - x0)
    ty = 0.0 if y1 == y0 else (yqc - y0) / (y1 - y0)

    # Corner values
    z00 = float(z[ix, iy])
    z10 = float(z[ix + 1, iy])
    z01 = float(z[ix, iy + 1])
    z11 = float(z[ix + 1, iy + 1])

    # Bilinear interpolation
    z0 = (1.0 - tx) * z00 + tx * z10
    z1 = (1.0 - tx) * z01 + tx * z11
    return (1.0 - ty) * z0 + ty * z1
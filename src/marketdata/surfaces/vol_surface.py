from __future__ import annotations

import numpy as np
from typing import Literal, Optional
from dataclasses import dataclass

# Extrapolation modes for grid interpolation behaviour.
ExtrapolationMode = Literal["flat", "error"]

# Metadata about what the "strike axis" represents.
StrikeSpace = Literal[
    "absolute",              # K in price units (canonical for pricing)
    "spot_moneyness",        # K = m * S0  (stored values are m, not K)  [NOT recommended to store long-term]
    "forward_moneyness",     # K = m * F0(T) (stored values are m)       [NOT recommended to store in 2D strike grid]
    "log_forward_moneyness", # log(K/F0(T))                              [NOT recommended to store in 2D strike grid]
]


@dataclass(frozen=True, slots=True)
class FlatVolSurface:
    """
    Flat implied volatility surface.

    This is stable and useful for:
      - quick demos
      - sanity tests
      - baseline pricing (BSM/GK)

    Parameters
    ----------
    sigma:
        Constant implied volatility (e.g., 0.12 for 12% annualized).

    Notes
    -----
    - The canonical interface is `implied_vol(expiry, strike)`.
    - `vol(expiry, strike)` is provided as a compatibility alias.
    """
    sigma: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.sigma):
            raise ValueError("FlatVolSurface.sigma must be finite.")
        if self.sigma <= 0.0:
            raise ValueError("FlatVolSurface.sigma must be strictly positive.")

    def implied_vol(self, expiry: float, strike: float) -> float:  # noqa: ARG002 (strike unused by design)
        expiry = float(expiry)
        if not np.isfinite(expiry):
            raise ValueError("expiry must be finite.")
        if expiry < 0.0:
            raise ValueError("expiry must be >= 0.")
        return float(self.sigma)

    def vol(self, expiry: float, strike: float) -> float:
        """Compatibility alias for implied_vol()."""
        return float(self.implied_vol(expiry, strike))


@dataclass(frozen=True, slots=True)
class GridVolSurface:
    """
    2D implied volatility surface defined on a grid (expiry x strike) with bilinear interpolation.

    Intended scope (V1)
    -------------------
    - FX and Equity style surfaces where implied vol is naturally a function of:
        (expiry T, strike K)

    Not IR swaption cubes
    ---------------------
    In IR (e.g., swaptions), market vol data is typically a cube:
        expiry x tenor x strike
    which should be represented by a different object (not this 2D surface).

    Grid definition
    --------------
    expiries:
        1D strictly-increasing array of expiry times (year fractions, >= 0).
    strikes:
        1D strictly-increasing array of strikes (absolute strike grid).
    implied_vols:
        2D array of shape (len(expiries), len(strikes)).

    Metadata
    --------
    strike_space:
        What the strikes axis represents. For V1 pricing, keep this as "absolute".
        Other values are allowed for debugging/inspection, but pricing code should
        not assume they are meaningful unless you also define a conversion rule.

    extrapolation:
        - "flat": clamp to nearest edge in each dimension (recommended)
        - "error": raise if outside grid
    """

    expiries: np.ndarray
    strikes: np.ndarray
    implied_vols: np.ndarray
    extrapolation: ExtrapolationMode = "flat"

    # Optional metadata (helps prevent confusion later)
    strike_space: StrikeSpace = "absolute"
    surface_id: Optional[str] = None  # e.g. "FX.EURUSD.SMILE" (debugging only)

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

        # Metadata validation
        if self.strike_space not in ("absolute", "spot_moneyness", "forward_moneyness", "log_forward_moneyness"):
            raise ValueError(f"GridVolSurface.strike_space is invalid: {self.strike_space!r}")

        object.__setattr__(self, "expiries", expiry_grid)
        object.__setattr__(self, "strikes", strike_grid)
        object.__setattr__(self, "implied_vols", vol_grid)

    def implied_vol(self, expiry: float, strike: float) -> float:
        """
        Interpolated implied volatility at (expiry, strike).

        Parameters
        ----------
        expiry:
            Time to expiry in year fractions (>= 0).
        strike:
            Strike query value. In V1, this should be an absolute strike.

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

    def vol(self, expiry: float, strike: float) -> float:
        """Compatibility alias for implied_vol()."""
        return float(self.implied_vol(expiry, strike))


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
    xqc = float(np.clip(xq, x[0], x[-1]))
    yqc = float(np.clip(yq, y[0], y[-1]))

    if x.size == 1 and y.size == 1:
        return float(z[0, 0])
    if x.size == 1:
        return float(np.interp(yqc, y, z[0, :]))
    if y.size == 1:
        return float(np.interp(xqc, x, z[:, 0]))

    ix = int(np.searchsorted(x, xqc, side="right") - 1)
    ix = int(np.clip(ix, 0, x.size - 2))

    iy = int(np.searchsorted(y, yqc, side="right") - 1)
    iy = int(np.clip(iy, 0, y.size - 2))

    x0 = float(x[ix])
    x1 = float(x[ix + 1])
    y0 = float(y[iy])
    y1 = float(y[iy + 1])

    tx = 0.0 if x1 == x0 else (xqc - x0) / (x1 - x0)
    ty = 0.0 if y1 == y0 else (yqc - y0) / (y1 - y0)

    z00 = float(z[ix, iy])
    z10 = float(z[ix + 1, iy])
    z01 = float(z[ix, iy + 1])
    z11 = float(z[ix + 1, iy + 1])

    z0 = (1.0 - tx) * z00 + tx * z10
    z1 = (1.0 - tx) * z01 + tx * z11
    return (1.0 - ty) * z0 + ty * z1
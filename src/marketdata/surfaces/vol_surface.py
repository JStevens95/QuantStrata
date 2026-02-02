"""
Volatility Surfaces Module

This module provides volatility surface classes for pricing:

FX/Equity Surfaces:
- FlatVolSurface: Constant volatility surface
- GridVolSurface: 2D surface (expiry x strike)

Interest Rate Surfaces:
- SwaptionVolCube: 3D surface (expiry x tenor x strike)
- CapFloorVolSurface: 2D surface (expiry x strike)
- FlatSwaptionVolCube: Constant swaption vol
- FlatCapFloorVolSurface: Constant cap/floor vol

Author: QuantStrata
"""
from __future__ import annotations

import numpy as np
from typing import Literal, Optional
from dataclasses import dataclass

from src.marketdata.core.types import ExtrapolationMode, VolType, StrikeSpace


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


# =============================================================================
# Interest Rate Volatility Surfaces
# =============================================================================


@dataclass(frozen=True, slots=True)
class SwaptionVolCube:
    """
    3D Swaption Volatility Cube: expiry x tenor x strike.

    This is the standard representation for swaption volatility data, where:
    - expiries: Option expiry times (e.g., 1M, 3M, 6M, 1Y, 2Y, 5Y, 10Y)
    - tenors: Underlying swap tenors (e.g., 1Y, 2Y, 5Y, 10Y, 30Y)
    - strikes: Strike offsets relative to ATM (e.g., -200bp, -100bp, ATM, +100bp, +200bp)
               or absolute strikes

    Volatility type:
    - "normal": Bachelier vol in basis points (e.g., 50bp = 0.0050)
    - "lognormal": Black vol as percentage (e.g., 20% = 0.20)

    Parameters
    ----------
    expiries : np.ndarray
        1D array of expiry times in years, strictly increasing.
    tenors : np.ndarray
        1D array of swap tenors in years, strictly increasing.
    strikes : np.ndarray
        1D array of strikes (absolute or relative to ATM).
    vols : np.ndarray
        3D array of shape (n_expiries, n_tenors, n_strikes).
    vol_type : VolType
        "normal" for Bachelier vol, "lognormal" for Black vol.
    strike_type : str
        "absolute" or "relative_atm" (relative to ATM forward).
    atm_forwards : Optional[np.ndarray]
        2D array of ATM forward swap rates (n_expiries, n_tenors).
    extrapolation : str
        "flat" to clamp to nearest edge, "error" to raise exception.

    Notes
    -----
    Market convention post-2015 is to quote swaption vols in normal (Bachelier)
    terms due to negative rate environments.
    """

    expiries: np.ndarray
    tenors: np.ndarray
    strikes: np.ndarray
    vols: np.ndarray
    vol_type: VolType = "normal"
    strike_type: Literal["absolute", "relative_atm"] = "absolute"
    atm_forwards: Optional[np.ndarray] = None
    extrapolation: ExtrapolationMode = "flat"

    def __post_init__(self) -> None:
        """Validate inputs."""
        object.__setattr__(self, "expiries", np.asarray(self.expiries, dtype=float))
        object.__setattr__(self, "tenors", np.asarray(self.tenors, dtype=float))
        object.__setattr__(self, "strikes", np.asarray(self.strikes, dtype=float))
        object.__setattr__(self, "vols", np.asarray(self.vols, dtype=float))

        if self.expiries.ndim != 1:
            raise ValueError("expiries must be 1D array")
        if self.tenors.ndim != 1:
            raise ValueError("tenors must be 1D array")
        if self.strikes.ndim != 1:
            raise ValueError("strikes must be 1D array")
        if self.vols.ndim != 3:
            raise ValueError("vols must be 3D array (expiry x tenor x strike)")

        n_exp, n_ten, n_str = len(self.expiries), len(self.tenors), len(self.strikes)
        if self.vols.shape != (n_exp, n_ten, n_str):
            raise ValueError(
                f"vols shape {self.vols.shape} doesn't match "
                f"({n_exp}, {n_ten}, {n_str})"
            )

        if not np.all(np.diff(self.expiries) > 0):
            raise ValueError("expiries must be strictly increasing")
        if not np.all(np.diff(self.tenors) > 0):
            raise ValueError("tenors must be strictly increasing")
        if not np.all(np.diff(self.strikes) > 0):
            raise ValueError("strikes must be strictly increasing")

        if np.any(self.expiries <= 0):
            raise ValueError("expiries must be positive")
        if np.any(self.tenors <= 0):
            raise ValueError("tenors must be positive")

        if self.atm_forwards is not None:
            atm = np.asarray(self.atm_forwards, dtype=float)
            object.__setattr__(self, "atm_forwards", atm)
            if atm.shape != (n_exp, n_ten):
                raise ValueError(f"atm_forwards shape {atm.shape} doesn't match ({n_exp}, {n_ten})")

    def implied_vol(self, expiry: float, tenor: float, strike: float) -> float:
        """
        Get implied volatility for a swaption.

        Parameters
        ----------
        expiry : float
            Option expiry in years.
        tenor : float
            Underlying swap tenor in years.
        strike : float
            Strike (absolute or relative based on strike_type).

        Returns
        -------
        float
            Implied volatility (normal or lognormal based on vol_type).
        """
        expiry, tenor, strike = float(expiry), float(tenor), float(strike)
        i_exp = self._find_index(expiry, self.expiries, "expiry")
        i_ten = self._find_index(tenor, self.tenors, "tenor")
        i_str = self._find_index(strike, self.strikes, "strike")
        return self._interpolate_3d(i_exp, i_ten, i_str)

    def atm_vol(self, expiry: float, tenor: float) -> float:
        """
        Get ATM implied volatility for a swaption.

        For strike_type="relative_atm", this returns vol at strike offset = 0.
        For strike_type="absolute", returns vol at ATM forward (if available).
        """
        if self.strike_type == "relative_atm":
            return self.implied_vol(expiry, tenor, 0.0)
        else:
            if self.atm_forwards is not None:
                i_exp = self._find_index(expiry, self.expiries, "expiry")
                i_ten = self._find_index(tenor, self.tenors, "tenor")
                atm_fwd = self._interpolate_2d_atm(i_exp, i_ten)
                return self.implied_vol(expiry, tenor, atm_fwd)
            else:
                mid_strike = self.strikes[len(self.strikes) // 2]
                return self.implied_vol(expiry, tenor, mid_strike)

    def smile(self, expiry: float, tenor: float) -> tuple[np.ndarray, np.ndarray]:
        """Get the volatility smile at a given expiry and tenor."""
        vols = np.array([self.implied_vol(expiry, tenor, k) for k in self.strikes])
        return self.strikes.copy(), vols

    def _find_index(self, value: float, grid: np.ndarray, name: str) -> tuple[int, float]:
        """Find interpolation index and weight."""
        if value < grid[0]:
            if self.extrapolation == "error":
                raise ValueError(f"{name}={value} below grid min {grid[0]}")
            return (0, 0.0)
        if value > grid[-1]:
            if self.extrapolation == "error":
                raise ValueError(f"{name}={value} above grid max {grid[-1]}")
            return (len(grid) - 2, 1.0)
        idx = np.searchsorted(grid, value, side="right") - 1
        idx = min(idx, len(grid) - 2)
        weight = (value - grid[idx]) / (grid[idx + 1] - grid[idx])
        return (idx, weight)

    def _interpolate_3d(
        self,
        i_exp: tuple[int, float],
        i_ten: tuple[int, float],
        i_str: tuple[int, float],
    ) -> float:
        """Trilinear interpolation."""
        ie, we = i_exp
        it, wt = i_ten
        ik, wk = i_str

        v000 = self.vols[ie, it, ik]
        v001 = self.vols[ie, it, ik + 1]
        v010 = self.vols[ie, it + 1, ik]
        v011 = self.vols[ie, it + 1, ik + 1]
        v100 = self.vols[ie + 1, it, ik]
        v101 = self.vols[ie + 1, it, ik + 1]
        v110 = self.vols[ie + 1, it + 1, ik]
        v111 = self.vols[ie + 1, it + 1, ik + 1]

        v00 = v000 * (1 - wk) + v001 * wk
        v01 = v010 * (1 - wk) + v011 * wk
        v10 = v100 * (1 - wk) + v101 * wk
        v11 = v110 * (1 - wk) + v111 * wk

        v0 = v00 * (1 - wt) + v01 * wt
        v1 = v10 * (1 - wt) + v11 * wt

        return float(v0 * (1 - we) + v1 * we)

    def _interpolate_2d_atm(self, i_exp: tuple[int, float], i_ten: tuple[int, float]) -> float:
        """Bilinear interpolation for ATM forwards."""
        if self.atm_forwards is None:
            raise ValueError("atm_forwards not set")
        ie, we = i_exp
        it, wt = i_ten
        v00 = self.atm_forwards[ie, it]
        v01 = self.atm_forwards[ie, it + 1]
        v10 = self.atm_forwards[ie + 1, it]
        v11 = self.atm_forwards[ie + 1, it + 1]
        v0 = v00 * (1 - wt) + v01 * wt
        v1 = v10 * (1 - wt) + v11 * wt
        return float(v0 * (1 - we) + v1 * we)


@dataclass(frozen=True, slots=True)
class FlatSwaptionVolCube:
    """
    Flat swaption volatility cube (constant vol across all expiries/tenors/strikes).

    Useful for testing and baseline pricing.

    Parameters
    ----------
    vol : float
        Constant volatility.
    vol_type : VolType
        "normal" for Bachelier, "lognormal" for Black.
    """

    vol: float
    vol_type: VolType = "normal"

    def __post_init__(self) -> None:
        if not np.isfinite(self.vol):
            raise ValueError("vol must be finite")
        if self.vol < 0:
            raise ValueError("vol must be non-negative")

    def implied_vol(self, expiry: float, tenor: float, strike: float) -> float:
        """Return constant vol."""
        return float(self.vol)

    def atm_vol(self, expiry: float, tenor: float) -> float:
        """Return constant vol."""
        return float(self.vol)

    def smile(self, expiry: float, tenor: float) -> tuple[np.ndarray, np.ndarray]:
        """Return flat smile."""
        strikes = np.linspace(-0.02, 0.02, 9)
        vols = np.full_like(strikes, self.vol)
        return strikes, vols


@dataclass(frozen=True, slots=True)
class CapFloorVolSurface:
    """
    2D Cap/Floor Volatility Surface: expiry x strike.

    Caps and floors are portfolios of caplets/floorlets, each priced with
    a single "flat" volatility. This surface provides that volatility.

    Parameters
    ----------
    expiries : np.ndarray
        1D array of cap/floor expiry times in years, strictly increasing.
    strikes : np.ndarray
        1D array of strike rates, strictly increasing.
    vols : np.ndarray
        2D array of shape (n_expiries, n_strikes).
    vol_type : VolType
        "normal" for Bachelier vol, "lognormal" for Black vol.
    extrapolation : str
        "flat" to clamp to nearest edge, "error" to raise exception.

    Notes
    -----
    Market convention:
    - Pre-2015: Log-normal (Black) volatilities
    - Post-2015: Normal (Bachelier) volatilities in bp
    """

    expiries: np.ndarray
    strikes: np.ndarray
    vols: np.ndarray
    vol_type: VolType = "normal"
    extrapolation: ExtrapolationMode = "flat"

    def __post_init__(self) -> None:
        """Validate inputs."""
        object.__setattr__(self, "expiries", np.asarray(self.expiries, dtype=float))
        object.__setattr__(self, "strikes", np.asarray(self.strikes, dtype=float))
        object.__setattr__(self, "vols", np.asarray(self.vols, dtype=float))

        if self.expiries.ndim != 1:
            raise ValueError("expiries must be 1D array")
        if self.strikes.ndim != 1:
            raise ValueError("strikes must be 1D array")
        if self.vols.ndim != 2:
            raise ValueError("vols must be 2D array (expiry x strike)")

        n_exp, n_str = len(self.expiries), len(self.strikes)
        if self.vols.shape != (n_exp, n_str):
            raise ValueError(f"vols shape {self.vols.shape} doesn't match ({n_exp}, {n_str})")

        if not np.all(np.diff(self.expiries) > 0):
            raise ValueError("expiries must be strictly increasing")
        if not np.all(np.diff(self.strikes) > 0):
            raise ValueError("strikes must be strictly increasing")

        if np.any(self.expiries <= 0):
            raise ValueError("expiries must be positive")

    def implied_vol(self, expiry: float, strike: float) -> float:
        """Get implied volatility for a cap/floor."""
        expiry, strike = float(expiry), float(strike)
        i_exp = self._find_index(expiry, self.expiries, "expiry")
        i_str = self._find_index(strike, self.strikes, "strike")
        return self._interpolate_2d(i_exp, i_str)

    def vol(self, expiry: float, strike: float) -> float:
        """Alias for implied_vol."""
        return self.implied_vol(expiry, strike)

    def smile(self, expiry: float) -> tuple[np.ndarray, np.ndarray]:
        """Get the volatility smile at a given expiry."""
        vols = np.array([self.implied_vol(expiry, k) for k in self.strikes])
        return self.strikes.copy(), vols

    def _find_index(self, value: float, grid: np.ndarray, name: str) -> tuple[int, float]:
        """Find interpolation index and weight."""
        if value < grid[0]:
            if self.extrapolation == "error":
                raise ValueError(f"{name}={value} below grid min {grid[0]}")
            return (0, 0.0)
        if value > grid[-1]:
            if self.extrapolation == "error":
                raise ValueError(f"{name}={value} above grid max {grid[-1]}")
            return (len(grid) - 2, 1.0)
        idx = np.searchsorted(grid, value, side="right") - 1
        idx = min(idx, len(grid) - 2)
        weight = (value - grid[idx]) / (grid[idx + 1] - grid[idx])
        return (idx, weight)

    def _interpolate_2d(
        self,
        i_exp: tuple[int, float],
        i_str: tuple[int, float],
    ) -> float:
        """Bilinear interpolation."""
        ie, we = i_exp
        ik, wk = i_str
        v00 = self.vols[ie, ik]
        v01 = self.vols[ie, ik + 1]
        v10 = self.vols[ie + 1, ik]
        v11 = self.vols[ie + 1, ik + 1]
        v0 = v00 * (1 - wk) + v01 * wk
        v1 = v10 * (1 - wk) + v11 * wk
        return float(v0 * (1 - we) + v1 * we)


@dataclass(frozen=True, slots=True)
class FlatCapFloorVolSurface:
    """
    Flat cap/floor volatility surface (constant vol).

    Parameters
    ----------
    vol : float
        Constant volatility.
    vol_type : VolType
        "normal" for Bachelier, "lognormal" for Black.
    """

    vol: float
    vol_type: VolType = "normal"

    def __post_init__(self) -> None:
        if not np.isfinite(self.vol):
            raise ValueError("vol must be finite")
        if self.vol < 0:
            raise ValueError("vol must be non-negative")

    def implied_vol(self, expiry: float, strike: float) -> float:
        """Return constant vol."""
        return float(self.vol)

    def vol(self, expiry: float, strike: float) -> float:
        """Alias for implied_vol."""
        return float(self.vol)

    def smile(self, expiry: float) -> tuple[np.ndarray, np.ndarray]:
        """Return flat smile."""
        strikes = np.linspace(0.01, 0.10, 10)
        vols = np.full_like(strikes, self.vol)
        return strikes, vols


# =============================================================================
# Factory Functions
# =============================================================================


def create_atm_swaption_vol_cube(
    expiries: np.ndarray,
    tenors: np.ndarray,
    atm_vols: np.ndarray,
    smile_width: float = 0.01,
    smile_curvature: float = 0.0,
    vol_type: VolType = "normal",
) -> SwaptionVolCube:
    """
    Create a swaption vol cube from ATM vols with a simple smile parameterization.

    Parameters
    ----------
    expiries : np.ndarray
        Expiry times in years.
    tenors : np.ndarray
        Swap tenors in years.
    atm_vols : np.ndarray
        2D array of ATM vols (n_expiries, n_tenors).
    smile_width : float
        Strike range around ATM (e.g., 0.01 = ±100bp).
    smile_curvature : float
        Quadratic smile coefficient (0 = flat smile).
    vol_type : VolType
        "normal" or "lognormal".

    Returns
    -------
    SwaptionVolCube
        Swaption vol cube with relative strikes.
    """
    expiries = np.asarray(expiries, dtype=float)
    tenors = np.asarray(tenors, dtype=float)
    atm_vols = np.asarray(atm_vols, dtype=float)

    strikes = np.linspace(-smile_width, smile_width, 9)
    n_exp, n_ten, n_str = len(expiries), len(tenors), len(strikes)
    vols = np.zeros((n_exp, n_ten, n_str))

    for i in range(n_exp):
        for j in range(n_ten):
            atm = atm_vols[i, j]
            for k, strike_offset in enumerate(strikes):
                vols[i, j, k] = atm + smile_curvature * strike_offset**2

    return SwaptionVolCube(
        expiries=expiries,
        tenors=tenors,
        strikes=strikes,
        vols=vols,
        vol_type=vol_type,
        strike_type="relative_atm",
    )


def create_cap_vol_surface_from_term_structure(
    expiries: np.ndarray,
    atm_vols: np.ndarray,
    strikes: np.ndarray,
    skew: float = 0.0,
    vol_type: VolType = "normal",
) -> CapFloorVolSurface:
    """
    Create a cap/floor vol surface from ATM term structure with skew.

    Parameters
    ----------
    expiries : np.ndarray
        Expiry times in years.
    atm_vols : np.ndarray
        1D array of ATM vols.
    strikes : np.ndarray
        Strike grid.
    skew : float
        Linear skew coefficient (vol change per 100bp strike move).
    vol_type : VolType
        "normal" or "lognormal".

    Returns
    -------
    CapFloorVolSurface
        Cap/floor vol surface.
    """
    expiries = np.asarray(expiries, dtype=float)
    atm_vols = np.asarray(atm_vols, dtype=float)
    strikes = np.asarray(strikes, dtype=float)

    n_exp, n_str = len(expiries), len(strikes)
    vols = np.zeros((n_exp, n_str))
    atm_strike = 0.03

    for i in range(n_exp):
        for j, k in enumerate(strikes):
            strike_offset = k - atm_strike
            vols[i, j] = atm_vols[i] + skew * strike_offset

    return CapFloorVolSurface(
        expiries=expiries,
        strikes=strikes,
        vols=vols,
        vol_type=vol_type,
    )
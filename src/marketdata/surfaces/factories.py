from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from src.marketdata.interfaces import VolSurface
from src.marketdata.surfaces.vol_surface import FlatVolSurface, GridVolSurface


def _coerce_scalar(params: np.ndarray) -> float:
    """
    Convert a parameter block into a scalar float.

    Accepts
    -------
    - 0-d ndarray: array(0.12)
    - size-1 1D ndarray: array([0.12])

    Raises
    ------
    ValueError
        If the parameter block cannot be interpreted as a scalar.
    """
    arr = np.asarray(params, dtype=float)

    if arr.ndim == 0:
        return float(arr)
    if arr.ndim == 1 and arr.size == 1:
        return float(arr[0])

    raise ValueError(f"Expected scalar params; got shape={arr.shape}.")


@dataclass(frozen=True, slots=True)
class FlatVolFactory:
    """
    Factory that builds a FlatVolSurface from a scalar parameter block.

    Parameter format
    ----------------
    params: scalar
        implied volatility sigma, where sigma > 0

    Typical MarketDataset storage
    -----------------------------
    - vol_params[mid] is a Panel with axis_names:
        ("time",) or ("time","scenario")
      and data shape:
        [T] or [T,S]
    - panel.block_at(t,s) (or your dataset slicing helper) yields a scalar.
    """

    def build(self, params: np.ndarray) -> VolSurface:
        sigma = _coerce_scalar(params)
        return FlatVolSurface(implied_vol=sigma)


@dataclass(frozen=True, slots=True)
class GridVolFactory:
    """
    Build a GridVolSurface from a parameter block.

    Design choice (V1)
    ------------------
    - expiries and strikes are supplied as *factory configuration*
    - params contains only the grid vol values, flattened row-major:
        implied_vols.flatten(order="C")

    This keeps snapshot slicing simple and avoids embedding variable-length
    grids inside the Panel block itself.

    Expected params shape
    ---------------------
    - length = len(expiries) * len(strikes), flattened row-major
      OR
    - shape = (len(expiries), len(strikes))
    """
    expiries: np.ndarray
    strikes: np.ndarray
    extrapolation: str = "flat"

    def __post_init__(self) -> None:
        exp = np.asarray(self.expiries, dtype=float).reshape(-1)
        k = np.asarray(self.strikes, dtype=float).reshape(-1)
        if exp.size == 0 or k.size == 0:
            raise ValueError("GridVolFactory expiries/strikes must be non-empty.")
        object.__setattr__(self, "expiries", exp)
        object.__setattr__(self, "strikes", k)

    def build(self, params: np.ndarray) -> GridVolSurface:
        exp = self.expiries
        k = self.strikes
        n_exp = exp.size
        n_k = k.size

        arr = np.asarray(params, dtype=float)

        if arr.ndim == 2:
            if arr.shape != (n_exp, n_k):
                raise ValueError(
                    f"GridVolFactory expected params shape {(n_exp, n_k)}, got {arr.shape}."
                )
            vol_grid = arr
        else:
            flat = arr.reshape(-1)
            expected = n_exp * n_k
            if flat.size != expected:
                raise ValueError(
                    f"GridVolFactory expected {expected} params (n_exp*n_strikes), got {flat.size}."
                )
            vol_grid = flat.reshape((n_exp, n_k), order="C")

        return GridVolSurface(
            expiries=exp,
            strikes=k,
            implied_vols=vol_grid,
            extrapolation=self.extrapolation,  # "flat" recommended
        )
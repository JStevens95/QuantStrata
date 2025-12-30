from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from src.marketdata.interfaces import VolSurface
from src.marketdata.surfaces.vol_surface import FlatVolSurface


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
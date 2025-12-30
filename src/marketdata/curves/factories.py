from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Literal

from src.marketdata.interfaces import Curve
from src.marketdata.curves.discount import FlatDiscountCurve, ZeroRateDiscountCurve


def _coerce_scalar(params: np.ndarray) -> float:
    """
    Convert a parameter block to a scalar float.

    Accepts:
    - python float/int (already an ndarray via np.asarray in calling code)
    - 0-d ndarray
    - 1-d size-1 ndarray

    Raises
    ------
    ValueError if it cannot be interpreted as a scalar.
    """
    arr = np.asarray(params, dtype=float)

    if arr.ndim == 0:
        return float(arr)
    if arr.ndim == 1 and arr.size == 1:
        return float(arr[0])

    raise ValueError(f"Expected scalar params; got shape={arr.shape}.")


def _parse_tenor_rate_grid(params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse a tenor/rate grid from a parameter block.

    Supported formats
    -----------------
    - params.shape == (K, 2): columns are [tenor, zero_rate]
    - params.shape == (2, K): first row is tenors, second row is zero_rates

    Returns
    -------
    tenors, zero_rates : np.ndarray, np.ndarray
        Both 1D arrays of length K.
    """
    arr = np.asarray(params, dtype=float)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D tenor/rate grid; got ndim={arr.ndim}, shape={arr.shape}.")

    # Case 1: (K,2)
    if arr.shape[1] == 2 and arr.shape[0] >= 1:
        tenors = arr[:, 0]
        rates = arr[:, 1]
        return tenors, rates

    # Case 2: (2,K)
    if arr.shape[0] == 2 and arr.shape[1] >= 1:
        tenors = arr[0, :]
        rates = arr[1, :]
        return tenors, rates

    raise ValueError(
        "Unsupported tenor/rate grid shape. Expected (K,2) or (2,K); "
        f"got shape={arr.shape}."
    )


@dataclass(frozen=True, slots=True)
class FlatCurveFactory:
    """
    Factory that builds a FlatDiscountCurve from a scalar parameter block.

    Parameter format
    ----------------
    params: scalar
        continuously-compounded rate r, such that df(t) = exp(-r t)

    Example Panel storage
    ---------------------
    - curve_params[mid] is a Panel with:
      - axis_names = ("time",) or ("time","scenario")
      - data shape = [T] or [T,S]
    - panel.block_at(t,s) returns scalar -> build(params)
    """

    def build(self, params: np.ndarray) -> Curve:
        r = _coerce_scalar(params)
        return FlatDiscountCurve(continuously_compounded_rate=r)


@dataclass(frozen=True, slots=True)
class ZeroCurveFactory:
    """
    Factory that builds a ZeroRateCurve from a tenor/rate grid.

    Parameter format
    ----------------
    params: 2D grid
        - (K,2) with columns [tenor, zero_rate]
        - OR (2,K) with first row tenors and second row rates

    Extrapolation
    -------------
    - "flat"  : clamp to endpoint rates outside grid (recommended)
    - "linear": linear extrapolation beyond grid endpoints
    """
    extrapolation: Literal["flat", "linear"] = "flat"

    def build(self, params: np.ndarray) -> Curve:
        tenors, rates = _parse_tenor_rate_grid(params)
        return ZeroRateDiscountCurve(tenors=tenors, zero_rates=rates, extrapolation=self.extrapolation)

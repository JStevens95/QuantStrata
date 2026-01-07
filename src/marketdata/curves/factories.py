from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.marketdata.core.types import ExtrapolationMode
from src.marketdata.curves.discount import FlatDiscountCurve, ZeroRateDiscountCurve


def _parse_zero_curve_params(params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Accept either:
      - [K,2] columns [tenor, zero_rate]
      - [2,K] rows    [tenor, zero_rate]
    Return (tenors, zero_rates) as 1D float arrays.
    """
    x = np.asarray(params, dtype=float)

    if x.ndim != 2:
        raise ValueError(f"Zero curve params must be 2D; got ndim={x.ndim}.")

    if x.shape[1] == 2 and x.shape[0] >= 1:
        return x[:, 0].reshape(-1), x[:, 1].reshape(-1)

    if x.shape[0] == 2 and x.shape[1] >= 1:
        return x[0, :].reshape(-1), x[1, :].reshape(-1)

    raise ValueError(f"Zero curve params must be [K,2] or [2,K]; got shape={x.shape}.")


@dataclass(frozen=True, slots=True)
class FlatCurveFactory:
    """
    Factory for FlatDiscountCurve.

    Expected params
    ---------------
    - scalar r, or shape [1] array-like, representing a continuously-compounded rate.
    """
    def build(self, params: np.ndarray) -> FlatDiscountCurve:
        x = np.asarray(params, dtype=float)

        if x.ndim == 0:
            r = float(x)
        else:
            x = x.reshape(-1)
            if x.size != 1:
                raise ValueError(f"FlatCurveFactory expects 1 parameter (rate); got shape={x.shape}.")
            r = float(x[0])

        if not np.isfinite(r):
            raise ValueError("Flat curve rate must be finite.")

        return FlatDiscountCurve(continuously_compounded_rate=r)


@dataclass(frozen=True, slots=True)
class ZeroCurveFactory:
    """
    Factory for ZeroRateDiscountCurve.

    Expected params
    ---------------
    - [K,2] or [2,K] where the tenor/zero columns/rows are:
        [tenor, zero_rate]
    """
    extrapolation: ExtrapolationMode = "flat"

    def build(self, params: np.ndarray) -> ZeroRateDiscountCurve:
        tenors, zeros = _parse_zero_curve_params(params)

        if tenors.size == 0:
            raise ValueError("ZeroCurveFactory received empty tenor grid.")

        return ZeroRateDiscountCurve(
            tenors=tenors,
            zero_rates=zeros,
            extrapolation=self.extrapolation,
        )
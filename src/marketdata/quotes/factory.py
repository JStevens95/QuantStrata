from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.marketdata.core.interfaces import Quote


def _coerce_scalar(params: np.ndarray | float | int) -> float:
    """
    Convert a parameter block into a scalar float.

    Accepts:
      - python float/int
      - 0-d ndarray: array(1.23)
      - size-1 1D ndarray: array([1.23])

    Raises
    ------
    ValueError if it cannot be interpreted as a scalar or is non-finite.
    """
    if isinstance(params, (float, int)):
        x = float(params)
    else:
        arr = np.asarray(params, dtype=float)
        if arr.ndim == 0:
            x = float(arr)
        elif arr.ndim == 1 and arr.size == 1:
            x = float(arr[0])
        else:
            raise ValueError(f"Expected scalar params; got shape={arr.shape}.")

    if not np.isfinite(x):
        raise ValueError("Quote value must be finite.")
    return x


@dataclass(frozen=True, slots=True)
class ScalarQuoteFactory:
    """
    Build a core Quote from a scalar value.

    Use this for: spot, simple rates, dividend yields, etc.
    """
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allow_negative: bool = True

    def build(self, params: np.ndarray | float | int) -> Quote:
        x = _coerce_scalar(params)

        if not self.allow_negative and x < 0.0:
            raise ValueError(f"Quote value must be >= 0. Got {x}.")

        if self.min_value is not None and x < float(self.min_value):
            raise ValueError(f"Quote value {x} < min_value {self.min_value}.")
        if self.max_value is not None and x > float(self.max_value):
            raise ValueError(f"Quote value {x} > max_value {self.max_value}.")

        return Quote(value=float(x))


@dataclass(frozen=True, slots=True)
class PositiveQuoteFactory(ScalarQuoteFactory):
    """
    Convenience: strictly positive quotes (e.g., vol-atm as a quote, spreads that must be > 0).
    """
    allow_negative: bool = False
    min_value: float = 1e-16
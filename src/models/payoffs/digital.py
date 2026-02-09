from __future__ import annotations

import sys
import numpy as np
from dataclasses import dataclass

from src.models.payoffs.base import BasePayoff1D, _as_float_array, _validate_option_type
from src.models.payoffs.types import OptionType

# slots=True requires Python 3.10+
_DATACLASS_KW = {"frozen": True, "slots": True} if sys.version_info >= (3, 10) else {"frozen": True}


@dataclass(**_DATACLASS_KW)
class DigitalCashPayoff(BasePayoff1D):
    """
    Cash-or-nothing digital.

    Pays `cash` if in-the-money at expiry:
      call: cash * 1{S >= K}
      put : cash * 1{S <= K}
    """
    option_type: OptionType
    strike: float
    cash: float = 1.0

    def __post_init__(self) -> None:
        _validate_option_type(self.option_type)
        if float(self.strike) <= 0.0:
            raise ValueError("strike must be > 0.")
        if not np.isfinite(float(self.cash)):
            raise ValueError("cash must be finite.")

    def terminal(self, spot: np.ndarray) -> np.ndarray:
        s = _as_float_array(spot)
        k = float(self.strike)
        cash = float(self.cash)

        if self.option_type == "call":
            return cash * (s >= k).astype(np.float64, copy=False)
        return cash * (s <= k).astype(np.float64, copy=False)


@dataclass(**_DATACLASS_KW)
class DigitalAssetPayoff(BasePayoff1D):
    """
    Asset-or-nothing digital.

    Pays `asset_units` units of the underlying if in-the-money at expiry:
      call: asset_units * S * 1{S >= K}
      put : asset_units * S * 1{S <= K}

    Notes
    -----
    - This returns payoff in *underlying currency units* (i.e. multiplied by spot),
      so for FX it corresponds to domestic value if `spot` is domestic-per-foreign.
    """
    option_type: OptionType
    strike: float
    asset_units: float = 1.0

    def __post_init__(self) -> None:
        _validate_option_type(self.option_type)
        if float(self.strike) <= 0.0:
            raise ValueError("strike must be > 0.")
        if not np.isfinite(float(self.asset_units)):
            raise ValueError("asset_units must be finite.")

    def terminal(self, spot: np.ndarray) -> np.ndarray:
        s = _as_float_array(spot)
        k = float(self.strike)
        units = float(self.asset_units)

        if self.option_type == "call":
            return (units * s) * (s >= k).astype(np.float64, copy=False)
        return (units * s) * (s <= k).astype(np.float64, copy=False)


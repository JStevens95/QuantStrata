from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from src.models.payoffs.base import BasePayoff1D, _as_float_array, _validate_option_type
from src.models.payoffs.types import OptionType


@dataclass(frozen=True, slots=True)
class VanillaPayoff(BasePayoff1D):
    """
    Vanilla payoff max(S-K,0) or max(K-S,0) per unit notional.
    """
    option_type: OptionType
    strike: float

    def __post_init__(self) -> None:
        _validate_option_type(self.option_type)
        if float(self.strike) <= 0.0:
            raise ValueError("strike must be > 0.")

    def terminal(self, spot: np.ndarray | float) -> np.ndarray:
        s = _as_float_array(spot)
        k = float(self.strike)
        if self.option_type == "call":
            return np.maximum(s - k, 0.0)
        return np.maximum(k - s, 0.0)
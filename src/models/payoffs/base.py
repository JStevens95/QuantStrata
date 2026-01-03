from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from src.models.payoffs.types import OptionType


@runtime_checkable
class Payoff1D(Protocol):
    """
    1D payoff interface.

    Conventions
    -----------
    - All methods return *per unit notional* payoff.
    - Inputs may be scalar or ndarray; outputs are ndarray (or scalar float).
    """

    def terminal(self, spot: np.ndarray) -> np.ndarray:
        """Terminal payoff at expiry as a function of spot S_T."""
        ...

    def intrinsic(self, spot: np.ndarray) -> np.ndarray:
        """
        Intrinsic value (exercise value) at time t as a function of spot S_t.

        For standard vanillas: intrinsic == terminal shape (same formula).
        For more complex products, intrinsic may differ.
        """
        ...

    @property
    def is_path_dependent(self) -> bool:
        """True if payoff requires path information (barriers, Asians, etc.)."""
        ...

    def __call__(self, spot: np.ndarray) -> np.ndarray:
        """Alias for terminal()."""
        return self.terminal(spot)


@dataclass(frozen=True, slots=True)
class BasePayoff1D:
    """
    Convenience base class: default intrinsic = terminal, non path-dependent.
    """
    def intrinsic(self, spot: np.ndarray) -> np.ndarray:
        return self.terminal(spot)

    @property
    def is_path_dependent(self) -> bool:
        return False


def _as_float_array(x: np.ndarray | float) -> np.ndarray:
    """
    Convert input to float64 ndarray without unnecessary copies.
    """
    return np.asarray(x, dtype=np.float64)


def _validate_option_type(option_type: OptionType) -> None:
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")
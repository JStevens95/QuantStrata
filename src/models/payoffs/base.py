from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from src.models.payoffs.types import OptionType


# ======================================================================================
# Protocols (typing-only contracts)
# ======================================================================================
#
# These are OPTIONAL at runtime (duck-typing), but very useful for:
# - keeping pricers generic (accept any object with the right methods)
# - static type checking (mypy/pyright) across the codebase
# - future-proofing (Numba/JAX payoffs, external payoffs, etc.)
# ======================================================================================

@runtime_checkable
class TerminalPayoff1D(Protocol):
    """
    Terminal-only payoff interface (non path-dependent).

    Conventions
    -----------
    - Methods return payoff *per unit foreign notional* unless stated otherwise
      (your pricers then multiply by trade.notional and discount as needed).
    - `spot` is interpreted as S_T (domestic-per-foreign FX spot at expiry).
    - Inputs may be scalar-like or ndarray; outputs are float64 ndarray.
    """

    def terminal(self, spot: np.ndarray) -> np.ndarray:
        """Payoff at expiry as a function of terminal spot S_T."""
        ...

    def intrinsic(self, spot: np.ndarray) -> np.ndarray:
        """
        Exercise/intrinsic value at time t as a function of spot S_t.

        For European vanillas/digitals, intrinsic typically equals terminal formula.
        For American-style products (future), intrinsic may differ.
        """
        ...

    @property
    def is_path_dependent(self) -> bool:
        """False for terminal-only payoffs."""
        ...

    def __call__(self, spot: np.ndarray) -> np.ndarray:
        """Alias for terminal()."""
        ...


@runtime_checkable
class PathPayoff1D(Protocol):
    """
    Path-dependent payoff interface.

    This is the Vn extension point for barriers, Asians, lookbacks, etc.

    Conventions
    -----------
    - `paths` is a 2D array with shape (n_paths, n_steps + 1).
      Each row is a simulated path of spot values including S0 at column 0.
    - Returns a 1D array shape (n_paths,) of payoffs per unit notional.
    """

    def terminal_from_paths(self, paths: np.ndarray) -> np.ndarray:
        """Payoff at expiry computed from the full simulated path."""
        ...

    def intrinsic_from_paths(self, paths: np.ndarray) -> np.ndarray:
        """
        Path-aware intrinsic value.

        Default behaviour for many products: intrinsic_from_paths == terminal_from_paths,
        but early-exercise products may differ (future extension).
        """
        ...

    @property
    def is_path_dependent(self) -> bool:
        """True for path-dependent payoffs."""
        ...

    def __call__(self, paths: np.ndarray) -> np.ndarray:
        """Alias for terminal_from_paths()."""
        ...


# ======================================================================================
# Base classes (runtime convenience)
# ======================================================================================

@dataclass(frozen=True, slots=True)
class BasePayoff1D:
    """
    Convenience base class for terminal-only payoffs.

    - Subclasses implement: terminal(spot)
    - Default intrinsic = terminal
    - Default is_path_dependent = False
    """

    def terminal(self, spot: np.ndarray) -> np.ndarray:  # pragma: no cover
        # Subclasses MUST implement terminal()
        raise NotImplementedError("Subclasses must implement terminal(spot).")

    def intrinsic(self, spot: np.ndarray) -> np.ndarray:
        # For standard European-style payoffs intrinsic matches terminal formula.
        return self.terminal(spot)

    @property
    def is_path_dependent(self) -> bool:
        return False

    def __call__(self, spot: np.ndarray) -> np.ndarray:
        return self.terminal(spot)


@dataclass(frozen=True, slots=True)
class BasePathPayoff1D:
    """
    Convenience base class for path-dependent payoffs.

    - Subclasses implement: terminal_from_paths(paths)
    - Default intrinsic_from_paths = terminal_from_paths
    - Default is_path_dependent = True

    Notes
    -----
    We intentionally do NOT provide a terminal(spot) method here because the product
    genuinely needs path information. This avoids accidental misuse (e.g. passing S_T
    only and silently getting the wrong answer).
    """

    def terminal_from_paths(self, paths: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError("Subclasses must implement terminal_from_paths(paths).")

    def intrinsic_from_paths(self, paths: np.ndarray) -> np.ndarray:
        return self.terminal_from_paths(paths)

    @property
    def is_path_dependent(self) -> bool:
        return True

    def __call__(self, paths: np.ndarray) -> np.ndarray:
        return self.terminal_from_paths(paths)


# ======================================================================================
# Helpers (shared by payoff implementations)
# ======================================================================================

def _as_float_array(x: np.ndarray | float) -> np.ndarray:
    """
    Convert input to float64 ndarray without unnecessary copies.

    Why:
    - Keeps payoff implementations concise.
    - Ensures consistent dtype across payoffs (important for numerical stability).
    """
    return np.asarray(x, dtype=np.float64)


def _as_paths_array(paths: np.ndarray) -> np.ndarray:
    """
    Validate/normalize a paths array for path-dependent payoffs.

    Expected:
    - 2D float array shape (n_paths, n_steps + 1)
    """
    p = np.asarray(paths, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError("paths must be a 2D array with shape (n_paths, n_steps+1).")
    if p.size == 0:
        raise ValueError("paths must be non-empty.")
    return p


def _validate_option_type(option_type: OptionType) -> None:
    """
    Central validation for option_type strings.

    Keeping this here avoids duplicating validation in every payoff class.
    """
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")
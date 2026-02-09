"""
Touch Option Payoff Implementation.

This module provides the `TouchPayoff` class for computing path-dependent
payoffs for touch (binary barrier) options, including one-touch and no-touch.

Mathematical Framework
----------------------
Touch options pay a fixed amount based on barrier touch status:

**One-Touch:**
    Payoff = Q * 1_{touched}

**No-Touch:**
    Payoff = Q * 1_{NOT touched}

where:
    - Q = payout_amount (fixed)
    - touched = (min(S_t) <= H) for down, (max(S_t) >= H) for up

Key Properties
--------------
- Binary payout: either Q or 0, no intermediate values.
- Path-dependent: requires full path to determine touch status.
- Does NOT depend on terminal spot value (only on touch status).
- Discrete monitoring: checked at each simulated point.

Implementation Notes
--------------------
- Returns payoffs per unit notional.
- Uses vectorized NumPy operations for efficiency.
- Inherits from BasePathPayoff1D for MC compatibility.
"""

from __future__ import annotations

import sys
import numpy as np
from dataclasses import dataclass
from typing import Literal

from src.models.payoffs.base import BasePathPayoff1D, _as_paths_array
from src.models.payoffs.types import BarrierDirection
from src.instruments.core.types import TouchStyle

# slots=True requires Python 3.10+
_DATACLASS_KW = {"frozen": True, "slots": True} if sys.version_info >= (3, 10) else {"frozen": True}


@dataclass(**_DATACLASS_KW)
class TouchPayoff(BasePathPayoff1D):
    """
    Touch (binary barrier) payoff for one-touch and no-touch options.

    This payoff pays a fixed amount (payout_amount) based on whether the
    spot price path touches (one-touch) or avoids (no-touch) a barrier level.

    What This Returns
    -----------------
    - A 1D array of payoffs (shape (n_paths,)) in domestic currency per unit
      notional.
    - Each payoff is either payout_amount or 0.

    Touch Styles
    ------------
    - one_touch: Pays payout_amount if barrier IS touched.
    - no_touch:  Pays payout_amount if barrier is NOT touched.

    Barrier Directions
    ------------------
    - up:   Touch = max(path) >= barrier_level.
    - down: Touch = min(path) <= barrier_level.

    Parameters
    ----------
    touch_style : TouchStyle
        "one_touch" or "no_touch".
    barrier_direction : BarrierDirection
        "up" or "down".
    barrier_level : float
        The barrier level H > 0.
    payout_amount : float
        Fixed payout Q >= 0 (per unit notional).

    Examples
    --------
    >>> import numpy as np
    >>> from src.models.payoffs.touch import TouchPayoff
    >>> # One-touch up: pays 1 if path touches 110
    >>> payoff = TouchPayoff(
    ...     touch_style="one_touch",
    ...     barrier_direction="up",
    ...     barrier_level=110.0,
    ...     payout_amount=1.0,
    ... )
    >>> paths_touch = np.array([[100, 105, 115, 108]])  # max=115 >= 110
    >>> payoff.terminal_from_paths(paths_touch)
    array([1.])
    >>> paths_no_touch = np.array([[100, 105, 108, 105]])  # max=108 < 110
    >>> payoff.terminal_from_paths(paths_no_touch)
    array([0.])
    """

    # Touch specification.
    touch_style: TouchStyle          # "one_touch" | "no_touch"
    barrier_direction: BarrierDirection  # "up" | "down"
    barrier_level: float             # Barrier H.
    payout_amount: float             # Fixed payout Q.

    def __post_init__(self) -> None:
        """
        Validate payoff parameters.

        Raises
        ------
        ValueError
            If parameters fail validation checks.
        """
        # Validate touch style.
        if self.touch_style not in ("one_touch", "no_touch"):
            raise ValueError("touch_style must be 'one_touch' or 'no_touch'.")

        # Validate barrier direction.
        if self.barrier_direction not in ("up", "down"):
            raise ValueError("barrier_direction must be 'up' or 'down'.")

        # Validate barrier level.
        if float(self.barrier_level) <= 0.0:
            raise ValueError("barrier_level must be > 0.")

        # Validate payout amount.
        payout = float(self.payout_amount)
        if not np.isfinite(payout):
            raise ValueError("payout_amount must be finite.")
        if payout < 0.0:
            raise ValueError("payout_amount must be >= 0.")

    # ------------------------------------------------------------------
    # Core payoff: per-path payoff using full simulated paths.
    # ------------------------------------------------------------------

    def terminal_from_paths(self, paths: np.ndarray) -> np.ndarray:
        """
        Compute payoff per path based on barrier touch status.

        Parameters
        ----------
        paths : np.ndarray
            Spot paths, shape (n_paths, n_steps+1), including S0 at column 0.

        Returns
        -------
        np.ndarray
            Payoffs per unit notional, shape (n_paths,).
            Each value is either payout_amount or 0.
        """
        # Convert to standard 2D paths array.
        p = _as_paths_array(paths)

        # Determine barrier touch status per path.
        touched = self._barrier_touched_mask(p)

        # Get payout amount.
        payout = float(self.payout_amount)

        # Apply touch style logic.
        if self.touch_style == "one_touch":
            # One-touch: pays payout if touched.
            return np.where(touched, payout, 0.0).astype(np.float64, copy=False)

        # No-touch: pays payout if NOT touched.
        return np.where(~touched, payout, 0.0).astype(np.float64, copy=False)

    # ------------------------------------------------------------------
    # Internal helper methods.
    # ------------------------------------------------------------------

    def _barrier_touched_mask(self, paths: np.ndarray) -> np.ndarray:
        """
        Determine if each path touched the barrier.

        Parameters
        ----------
        paths : np.ndarray
            Spot paths, shape (n_paths, n_steps+1).

        Returns
        -------
        np.ndarray
            Boolean array, shape (n_paths,). True if barrier was touched.
        """
        H = float(self.barrier_level)

        if self.barrier_direction == "up":
            # Up barrier: touched if max(path) >= H.
            max_s = np.max(paths, axis=1)
            return (max_s >= H)

        # Down barrier: touched if min(path) <= H.
        min_s = np.min(paths, axis=1)
        return (min_s <= H)

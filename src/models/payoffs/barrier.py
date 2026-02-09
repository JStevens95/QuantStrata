from __future__ import annotations

import sys
import numpy as np
from dataclasses import dataclass

from src.models.payoffs.base import BasePathPayoff1D, _as_float_array, _as_paths_array, _validate_option_type
from src.models.payoffs.types import OptionType, BarrierDirection, BarrierStyle

# slots=True requires Python 3.10+
_DATACLASS_KW = {"frozen": True, "slots": True} if sys.version_info >= (3, 10) else {"frozen": True}


@dataclass(**_DATACLASS_KW)
class SingleBarrierPayoff(BasePathPayoff1D):
    """
    European single-barrier payoff (path-dependent) for FX under discrete monitoring.

    What this returns
    -----------------
    - A 1D array of payoffs (shape (n_paths,)) in *domestic currency per 1 unit foreign notional*.

    Discrete monitoring rule (V1)
    -----------------------------
    - `paths` has shape (n_paths, n_steps+1) and includes S0 at column 0.
    - Barrier is considered "hit" if ANY monitored point crosses the barrier:
        * up   barrier: hit if max(path) >= B
        * down barrier: hit if min(path) <= B

    Barrier styles
    --------------
    - knock_out:
        payoff = vanilla(S_T) if NOT hit else rebate
    - knock_in:
        payoff = vanilla(S_T) if hit else rebate

    Rebate
    ------
    - `rebate_amount` is interpreted as *domestic* amount paid at expiry per unit notional.

    Notes
    -----
    - This class intentionally does NOT implement terminal(spot) because the payoff needs the
      full path to determine hit status.
    """

    option_type: OptionType
    strike: float

    barrier_direction: BarrierDirection  # "up" | "down"
    barrier_style: BarrierStyle          # "knock_out" | "knock_in"
    barrier_level: float

    rebate_amount: float = 0.0

    def __post_init__(self) -> None:
        # Validate vanilla leg
        _validate_option_type(self.option_type)
        if float(self.strike) <= 0.0:
            raise ValueError("strike must be > 0.")

        # Validate barrier definition
        if self.barrier_direction not in ("up", "down"):
            raise ValueError("barrier_direction must be 'up' or 'down'.")
        if self.barrier_style not in ("knock_out", "knock_in"):
            raise ValueError("barrier_style must be 'knock_out' or 'knock_in'.")
        if float(self.barrier_level) <= 0.0:
            raise ValueError("barrier_level must be > 0.")

        # Validate rebate
        rebate = float(self.rebate_amount)
        if not np.isfinite(rebate):
            raise ValueError("rebate_amount must be finite.")
        if rebate < 0.0:
            raise ValueError("rebate_amount must be >= 0.")

    # ------------------------------------------------------------------
    # Core payoff: per-path payoff using full simulated paths
    # ------------------------------------------------------------------

    def terminal_from_paths(self, paths: np.ndarray) -> np.ndarray:
        """
        Compute expiry payoff per path using the full simulated spot path.

        Parameters
        ----------
        paths:
            Spot paths, shape (n_paths, n_steps+1), including S0 in column 0.

        Returns
        -------
        np.ndarray
            Payoffs per unit notional, shape (n_paths,).
        """
        p = _as_paths_array(paths)

        # Terminal spots S_T are last column
        s_t = p[:, -1]

        # Determine barrier hit per path (discrete monitoring, inclusive)
        hit = self._barrier_hit_mask(p)

        # Vanilla payoff on terminal spot
        vanilla = self._vanilla_terminal(s_t)

        # Rebate is a constant paid at expiry per unit notional
        rebate = float(self.rebate_amount)

        # Knock-out / knock-in selection
        if self.barrier_style == "knock_out":
            # Pay vanilla if NOT hit, else rebate
            return np.where(~hit, vanilla, rebate).astype(np.float64, copy=False)

        # knock_in:
        # Pay vanilla if hit, else rebate
        return np.where(hit, vanilla, rebate).astype(np.float64, copy=False)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _vanilla_terminal(self, spot_t: np.ndarray) -> np.ndarray:
        """
        Vanilla payoff max(S-K,0) or max(K-S,0) per unit notional.
        """
        s = _as_float_array(spot_t)
        k = float(self.strike)

        if self.option_type == "call":
            return np.maximum(s - k, 0.0)
        return np.maximum(k - s, 0.0)

    def _barrier_hit_mask(self, paths: np.ndarray) -> np.ndarray:
        """
        Boolean array (n_paths,) indicating whether barrier was hit on each path.
        """
        b = float(self.barrier_level)

        if self.barrier_direction == "up":
            # Hit if any monitored spot >= barrier
            max_s = np.max(paths, axis=1)
            return (max_s >= b)

        # down:
        # Hit if any monitored spot <= barrier
        min_s = np.min(paths, axis=1)
        return (min_s <= b)
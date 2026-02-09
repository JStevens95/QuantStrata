"""
Double Barrier Payoff Implementation.

This module provides the `DoubleBarrierPayoff` class for computing path-dependent
payoffs for double barrier options. The payoff depends on whether the spot price
stays within or exits a corridor defined by lower and upper barriers.

Mathematical Framework
----------------------
For a knock-out double barrier:
    Payoff = max(S_T - K, 0) * 1_{L < min(S_t) and max(S_t) < U} + R * 1_{barrier hit}

For a knock-in double barrier:
    Payoff = max(S_T - K, 0) * 1_{min(S_t) <= L or max(S_t) >= U} + R * 1_{no hit}

where:
    - L = lower barrier
    - U = upper barrier  
    - S_t = spot path
    - K = strike
    - R = rebate

Key Properties
--------------
- Path-dependent: requires full simulated path to determine barrier status.
- Discrete monitoring: barriers checked at each simulated point.
- Both barriers checked simultaneously: hit if either is breached.

Implementation Notes
--------------------
- Uses vectorized NumPy operations for efficiency.
- Returns payoffs per unit notional (scaling done by pricer).
- Inherits from BasePathPayoff1D for compatibility with MC framework.
"""

from __future__ import annotations

import sys
import numpy as np
from dataclasses import dataclass

from src.models.payoffs.base import BasePathPayoff1D, _as_float_array, _as_paths_array, _validate_option_type
from src.models.payoffs.types import OptionType, BarrierStyle

# slots=True requires Python 3.10+
_DATACLASS_KW = {"frozen": True, "slots": True} if sys.version_info >= (3, 10) else {"frozen": True}


@dataclass(**_DATACLASS_KW)
class DoubleBarrierPayoff(BasePathPayoff1D):
    """
    European double-barrier payoff (path-dependent) for FX under discrete monitoring.

    This payoff checks both upper and lower barriers simultaneously. The option
    either survives (knock-out) or activates (knock-in) based on whether the
    spot path stays within or exits the [lower_barrier, upper_barrier] corridor.

    What This Returns
    -----------------
    - A 1D array of payoffs (shape (n_paths,)) in domestic currency per unit
      foreign notional.

    Discrete Monitoring Rule
    ------------------------
    - `paths` has shape (n_paths, n_steps+1) and includes S0 at column 0.
    - Barrier is considered "hit" if ANY monitored point exits the corridor:
        * Lower barrier hit: min(path) <= lower_barrier
        * Upper barrier hit: max(path) >= upper_barrier

    Barrier Styles
    --------------
    - knock_out:
        payoff = vanilla(S_T) if stayed in corridor else rebate
    - knock_in:
        payoff = vanilla(S_T) if exited corridor else rebate

    Parameters
    ----------
    option_type : OptionType
        "call" or "put" for vanilla payoff calculation.
    strike : float
        Strike price K > 0.
    barrier_style : BarrierStyle
        "knock_out" or "knock_in".
    lower_barrier : float
        Lower barrier level L > 0, must be < upper_barrier.
    upper_barrier : float
        Upper barrier level U > 0, must be > lower_barrier.
    rebate_amount : float, optional
        Rebate paid at expiry when barrier condition fails. Default 0.0.

    Examples
    --------
    >>> import numpy as np
    >>> from src.models.payoffs.double_barrier import DoubleBarrierPayoff
    >>> payoff = DoubleBarrierPayoff(
    ...     option_type="call",
    ...     strike=100.0,
    ...     barrier_style="knock_out",
    ...     lower_barrier=90.0,
    ...     upper_barrier=110.0,
    ... )
    >>> # Path that stays in corridor [90, 110].
    >>> paths_in = np.array([[100, 102, 98, 105]])  # S_T = 105
    >>> payoff.terminal_from_paths(paths_in)
    array([5.])  # max(105 - 100, 0) = 5
    >>> # Path that hits lower barrier.
    >>> paths_out = np.array([[100, 95, 85, 105]])  # min = 85 <= 90
    >>> payoff.terminal_from_paths(paths_out)
    array([0.])  # Knocked out, rebate = 0
    """

    # Vanilla leg specification.
    option_type: OptionType  # "call" | "put"
    strike: float            # Strike price K.

    # Double barrier specification.
    barrier_style: BarrierStyle  # "knock_out" | "knock_in"
    lower_barrier: float         # Lower barrier L.
    upper_barrier: float         # Upper barrier U.

    # Rebate paid at expiry when barrier condition fails.
    rebate_amount: float = 0.0

    def __post_init__(self) -> None:
        """
        Validate payoff parameters.

        Raises
        ------
        ValueError
            If parameters fail validation checks.
        """
        # Validate option type using shared helper.
        _validate_option_type(self.option_type)

        # Validate strike.
        if float(self.strike) <= 0.0:
            raise ValueError("strike must be > 0.")

        # Validate barrier style.
        if self.barrier_style not in ("knock_out", "knock_in"):
            raise ValueError("barrier_style must be 'knock_out' or 'knock_in'.")

        # Validate barrier levels.
        lower = float(self.lower_barrier)
        upper = float(self.upper_barrier)

        if lower <= 0.0:
            raise ValueError("lower_barrier must be > 0.")
        if upper <= 0.0:
            raise ValueError("upper_barrier must be > 0.")
        if lower >= upper:
            raise ValueError(
                f"lower_barrier ({lower}) must be < upper_barrier ({upper})."
            )

        # Validate rebate.
        rebate = float(self.rebate_amount)
        if not np.isfinite(rebate):
            raise ValueError("rebate_amount must be finite.")
        if rebate < 0.0:
            raise ValueError("rebate_amount must be >= 0.")

    # ------------------------------------------------------------------
    # Core payoff: per-path payoff using full simulated paths.
    # ------------------------------------------------------------------

    def terminal_from_paths(self, paths: np.ndarray) -> np.ndarray:
        """
        Compute expiry payoff per path using the full simulated spot path.

        The payoff depends on whether the path stayed within or exited the
        [lower_barrier, upper_barrier] corridor.

        Parameters
        ----------
        paths : np.ndarray
            Spot paths, shape (n_paths, n_steps+1), including S0 at column 0.

        Returns
        -------
        np.ndarray
            Payoffs per unit notional, shape (n_paths,), in domestic currency.
        """
        # Convert to standard 2D paths array.
        p = _as_paths_array(paths)

        # Terminal spots S_T are last column.
        s_t = p[:, -1]

        # Determine if corridor was exited (either barrier hit).
        exited_corridor = self._corridor_exited_mask(p)

        # Compute vanilla payoff on terminal spot.
        vanilla = self._vanilla_terminal(s_t)

        # Get rebate amount (constant per unit notional).
        rebate = float(self.rebate_amount)

        # Apply knock-out or knock-in logic.
        if self.barrier_style == "knock_out":
            # Knock-out: pay vanilla if stayed in corridor, else rebate.
            # stayed_in_corridor = NOT exited_corridor.
            return np.where(~exited_corridor, vanilla, rebate).astype(np.float64, copy=False)

        # Knock-in: pay vanilla if exited corridor, else rebate.
        return np.where(exited_corridor, vanilla, rebate).astype(np.float64, copy=False)

    # ------------------------------------------------------------------
    # Internal helper methods.
    # ------------------------------------------------------------------

    def _vanilla_terminal(self, spot_t: np.ndarray) -> np.ndarray:
        """
        Compute vanilla payoff max(S_T - K, 0) or max(K - S_T, 0).

        Parameters
        ----------
        spot_t : np.ndarray
            Terminal spot prices, shape (n_paths,).

        Returns
        -------
        np.ndarray
            Vanilla payoffs per unit notional, shape (n_paths,).
        """
        s = _as_float_array(spot_t)
        k = float(self.strike)

        if self.option_type == "call":
            return np.maximum(s - k, 0.0)
        return np.maximum(k - s, 0.0)

    def _corridor_exited_mask(self, paths: np.ndarray) -> np.ndarray:
        """
        Determine if each path exited the corridor (hit either barrier).

        Corridor exit occurs if:
            - min(path) <= lower_barrier (hit lower), OR
            - max(path) >= upper_barrier (hit upper)

        Parameters
        ----------
        paths : np.ndarray
            Spot paths, shape (n_paths, n_steps+1).

        Returns
        -------
        np.ndarray
            Boolean array, shape (n_paths,). True if path exited corridor.
        """
        lower = float(self.lower_barrier)
        upper = float(self.upper_barrier)

        # Compute path extrema (vectorized across all paths).
        min_s = np.min(paths, axis=1)  # Minimum spot per path.
        max_s = np.max(paths, axis=1)  # Maximum spot per path.

        # Check if either barrier was hit.
        hit_lower = (min_s <= lower)  # Hit lower barrier.
        hit_upper = (max_s >= upper)  # Hit upper barrier.

        # Exited corridor if either barrier hit.
        return hit_lower | hit_upper
